##### GPUCellIterator #####


abstract type AbstractCUDACellIterator <: Ferrite.AbstractKernelCellIterator end

"""
    CUDACellIterator{DH<:Ferrite.AbstractGPUDofHandler,GRID<: Ferrite.AbstractGPUGrid,KDynamicSharedMem,FDynamicSharedMem}

Create `CUDACellIterator` object for each thread with local id `thread_id` in order to iterate over some elements in the grid
on the GPU and these elements are associated with the thread based on a stride = `blockDim().x * gridDim().x`.
The elements of the iterator are `GPUCellCache` objects.
"""
struct CUDACellIterator{DH <: Ferrite.GPUDofHandler, GRID <: Ferrite.AbstractGPUGrid, KDynamicSharedMem, FDynamicSharedMem} <: AbstractCUDACellIterator
    dh::DH # TODO: subdofhandlers are not supported yet.
    grid::GRID
    n_cells::Int32
    block_ke::KDynamicSharedMem # dynamic shared memory for the block (3rd order tensor (e,i,j))
    block_fe::FDynamicSharedMem # dynamic shared memory for the block (2nd order tensor (e,i))
    thread_id::Int32 # local thread id (maps to the index of the element in block_ke and block_fe)
end

"""
    Ferrite.CellIterator(dh::Ferrite.AbstractGPUDofHandler, n_basefuncs::Int32)

Create a `CUDACellIterator` object which is used to iterate over the cells of the grid on the GPU.
This function also initializes the dynamic shared memory (`block_ke` and `block_fe`) to store the stiffness matrix and force vector
per element for the base functions (`n_basefuncs`) being used.
Arguments:
- `dh`: The degree of freedom handler for the GPU.
- `n_basefuncs`: Number of base functions (shape functions) for each element.

Returns:
- A `CUDACellIterator` object.
"""
function Ferrite.CellIterator(dh::Ferrite.GPUDofHandler, n_basefuncs::Int32)
    ## cell iterator that uses dynamic shared memory
    grid = get_grid(dh)
    n_cells = grid |> getncells |> Int32
    bd = blockDim().x
    ke_shared = @cuDynamicSharedMem(Float32, (bd, n_basefuncs, n_basefuncs))
    fe_shared = @cuDynamicSharedMem(Float32, (bd, n_basefuncs), sizeof(Float32) * bd * n_basefuncs * n_basefuncs)
    local_thread_id = threadIdx().x
    return CUDACellIterator(dh, grid, n_cells, ke_shared, fe_shared, local_thread_id)
end


"""
    ncells(iterator::AbstractCUDACellIterator)

Return the total number of cells in the grid that the iterator is iterating over.

Arguments:
- `iterator`: The subtype of `AbstractCUDACellIterator` type.

Returns:
- The total number of cells as `Int32`.
"""
ncells(iterator::AbstractCUDACellIterator) = iterator.n_cells


function Base.iterate(iterator::AbstractCUDACellIterator)
    i = (blockIdx().x - Int32(1)) * blockDim().x + threadIdx().x # global thread id
    i <= iterator.n_cells || return nothing
    return (_makecache(iterator, i), i)
end


function Base.iterate(iterator::AbstractCUDACellIterator, state)
    stride = blockDim().x * gridDim().x
    i = state + stride # next strided element id
    i <= iterator.n_cells || return nothing
    return (_makecache(iterator, i), i)
end


## Cell iterator that uses global memory ##

struct CUDAGlobalCellIterator{DH <: Ferrite.GPUDofHandler, GRID <: Ferrite.AbstractGPUGrid, MAT, VEC} <: AbstractCUDACellIterator
    dh::DH # TODO: subdofhandlers are not supported yet.
    grid::GRID
    n_cells::Int32
    ke::MAT # reference to the global memory for the stiffness matrix
    fe::VEC  # reference to the global memory for the force vector
    thread_id::Int32 # local thread id
end

function Ferrite.CellIterator(dh_::Ferrite.LocalsGPUDofHandler, n_basefuncs::Int32)
    ## cell iterator that uses global memory
    dh = dh_ |> dofhandler
    grid = get_grid(dh)
    n_cells = grid |> getncells |> Int32
    bd = blockDim().x
    local_thread_id = threadIdx().x
    global_thread_id = (blockIdx().x - Int32(1)) * bd + local_thread_id
    ke = cellke(dh_, global_thread_id)
    fe = cellfe(dh_, global_thread_id)
    return CUDAGlobalCellIterator(dh, grid, n_cells, ke, fe, local_thread_id)
end


##### GPUCellCache #####

"""
    GPUCellCache{DOFS,NN,NODES,COORDS,KDynamicSharedMem,FDynamicSharedMem}

This structure holds the data needed for each finite element cell during GPU computations.
It includes the coordinates of the cell's nodes, the degrees of freedom (DoFs), the cell ID,
and views into dynamic shared memory for the stiffness matrix (`ke`) and force vector (`fe`).

Arguments:
- `coords`: Coordinates of the nodes of the cell.
- `dofs`: Degrees of freedom associated with the cell.
- `cellid`: ID of the current cell.
- `nodes`: Node IDs of the cell (as a static vector for performance).
- `ke`: View into shared memory for the cell's stiffness matrix.
- `fe`: View into shared memory for the cell's force vector.
"""
struct GPUCellCache{DOFS <: AbstractVector{Int32}, NN, NODES <: SVector{NN, Int32}, X, COORDS <: SVector{X}, KDynamicSharedMem, FDynamicSharedMem} <: Ferrite.AbstractKernelCellCache
    coords::COORDS
    dofs::DOFS
    cellid::Int32
    nodes::NODES
    ke::KDynamicSharedMem # view of the dynamic shared memory for the cell (i.e. element local stiffness matrix).
    fe::FDynamicSharedMem # view of the dynamic shared memory for the cell (i.e. element local force vector).
end


function _makecache(iterator::CUDACellIterator, e::Int32)
    ke_fun = () -> (@view iterator.block_ke[iterator.thread_id, :, :])
    fe_fun = () -> (@view iterator.block_fe[iterator.thread_id, :])
    return _makecache(iterator, e, ke_fun, fe_fun)
end

function _makecache(iterator::CUDAGlobalCellIterator, e::Int32)
    ke_fun = () -> iterator.ke
    fe_fun = () -> iterator.fe
    return _makecache(iterator, e, ke_fun, fe_fun)
end

function _makecache(iterator::AbstractCUDACellIterator, e::Int32, ke_func::Function, fe_func::Function)
    dh = iterator.dh
    grid = iterator.grid
    cellid = e
    cell = Ferrite.getcells(grid, e)

    # Extract the node IDs of the cell.
    nodes = SVector(convert.(Int32, Ferrite.get_node_ids(cell))...)

    # Extract the degrees of freedom for the cell.
    dofs = Ferrite.celldofs(dh, e)

    # Get the coordinates of the nodes of the cell.
    CT = Ferrite.get_coordinate_type(grid)
    N = Ferrite.nnodes(cell)
    x = MVector{N, CT}(undef)
    for i in eachindex(x)
        x[i] = Ferrite.get_node_coordinate(grid, nodes[i])
    end
    coords = SVector(x...)

    # Return the GPUCellCache containing the cell's data.
    return GPUCellCache(coords, dofs, cellid, nodes, ke_func(), fe_func())
end

"""
    getnodes(cc::GPUCellCache)

Return the node IDs associated with the current cell in the cache.

Arguments:
- `cc`: The `GPUCellCache` object.

Returns:
- The node IDs of the current cell.
"""
Ferrite.getnodes(cc::GPUCellCache) = cc.nodes

"""
    getcoordinates(cc::GPUCellCache)

Return the coordinates of the current cell's nodes.

Arguments:
- `cc`: The `GPUCellCache` object.

Returns:
- The coordinates of the nodes of the current cell.
"""
Ferrite.getcoordinates(cc::GPUCellCache) = cc.coords

"""
    celldofs(cc::GPUCellCache)

Return the degrees of freedom (DoFs) for the current cell from the cache.

Arguments:
- `cc`: The `GPUCellCache` object.

Returns:
- The degrees of freedom (DoFs) associated with the current cell.
"""
Ferrite.celldofs(cc::GPUCellCache) = cc.dofs

"""
    cellid(cc::GPUCellCache)

Return the ID of the current cell stored in the cache.

Arguments:
- `cc`: The `GPUCellCache` object.

Returns:
- The ID of the current cell.
"""
Ferrite.cellid(cc::GPUCellCache) = cc.cellid

"""
    Ferrite.cellke(cc::GPUCellCache)

Access the stiffness matrix (`ke`) of the current cell from shared memory and reset it to zero.

Arguments:
- `cc`: The `GPUCellCache` object.

Returns:
- The stiffness matrix filled with zeros.
"""
@inline function Ferrite.cellke(cc::GPUCellCache)
    ke = cc.ke
    return fill!(ke, 0.0f0)
end

"""
    Ferrite.cellfe(cc::GPUCellCache)

Access the force vector (`fe`) of the current cell from shared memory and reset it to zero.

Arguments:
- `cc`: The `GPUCellCache` object.

Returns:
- The force vector filled with zeros.
"""
@inline function Ferrite.cellfe(cc::GPUCellCache)
    fe = cc.fe
    return fill!(fe, 0.0f0)
end
