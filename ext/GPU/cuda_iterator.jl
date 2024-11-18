##### GPUCellIterator #####

"""
    AbstractCUDACellIterator <: Ferrite.AbstractKernelCellIterator

Abstract type representing CUDA cell iterators for finite element computations
on the GPU. It provides the base for implementing multiple cell iteration strategies (e.g. with shared memory, with global memory).
"""
abstract type AbstractCUDACellIterator <: Ferrite.AbstractKernelCellIterator end

"""
    ncells(iterator::AbstractCUDACellIterator)

Get the total number of cells the iterator will process.

# Arguments
- `iterator`: A subtype of `AbstractCUDACellIterator`.

# Returns
The number of cells as an `Int32`.
"""
ncells(iterator::AbstractCUDACellIterator) = iterator.n_cells ## any subtype has to have `n_cells` field


"""
    CUDACellIterator{DH, GRID, KDynamicSharedMem, FDynamicSharedMem}

A CUDA-specific cell iterator used for iterating over elements of a finite element
grid on the GPU. This iterator is designed to work with shared memory for local stiffness matrices and force vectors.

# Type Parameters
- `DH<:Ferrite.AbstractGPUDofHandler`: Degree-of-freedom handler for GPU data.
- `GRID<:Ferrite.AbstractGPUGrid`: GPU-based grid structure.
- `KDynamicSharedMem`: Dynamic shared memory type for stiffness matrices.
- `FDynamicSharedMem`: Dynamic shared memory type for force vectors.

# Fields
- `dh`: The degree-of-freedom handler.
- `grid`: The GPU grid being processed.
- `n_cells`: Total number of cells in the grid.
- `block_ke`: Block local stifness matrices (i.e. 3rd order tensor (e,i,j)).
- `block_fe`: Block local force vectors (i.e. 2nd order tensor (e,i)).
- `thread_id`: Local thread ID in the CUDA block.
"""
struct CUDACellIterator{DH <: Ferrite.GPUDofHandler, GRID <: Ferrite.AbstractGPUGrid, KDynamicSharedMem, FDynamicSharedMem} <: AbstractCUDACellIterator
    dh::DH
    grid::GRID
    n_cells::Int32
    block_ke::KDynamicSharedMem
    block_fe::FDynamicSharedMem
    thread_id::Int32
end

"""
    Ferrite.CellIterator(dh::Ferrite.GPUDofHandler, n_basefuncs::Int32)

Create a `CUDACellIterator` for iterating over grid cells on the GPU.

# Arguments
- `dh`: Degree-of-freedom handler for the GPU.
- `n_basefuncs`: Number of shape functions per cell.

# Returns
A `CUDACellIterator` configured with dynamic shared memory for stiffness matrices
and force vectors.
"""
function Ferrite.CellIterator(dh::Ferrite.GPUDofHandler, n_basefuncs::Int32)
    grid = get_grid(dh)
    n_cells = grid |> getncells |> Int32
    bd = blockDim().x
    ke_shared = @cuDynamicSharedMem(Float32, (bd, n_basefuncs, n_basefuncs))
    fe_shared = @cuDynamicSharedMem(Float32, (bd, n_basefuncs), sizeof(Float32) * bd * n_basefuncs * n_basefuncs)
    local_thread_id = threadIdx().x
    return CUDACellIterator(dh, grid, n_cells, ke_shared, fe_shared, local_thread_id)
end


function Base.iterate(iterator::AbstractCUDACellIterator)
    i = (blockIdx().x - Int32(1)) * blockDim().x + threadIdx().x
    i <= iterator.n_cells || return nothing
    return (_makecache(iterator, i), i)
end


function Base.iterate(iterator::AbstractCUDACellIterator, state)
    stride = blockDim().x * gridDim().x
    i = state + stride
    i <= iterator.n_cells || return nothing
    return (_makecache(iterator, i), i)
end

"""
    CUDAGlobalCellIterator{DH, GRID, MAT, VEC}

A CUDA-specific cell iterator that uses global memory instead of shared memory.

# Type Parameters
- `DH<:Ferrite.GPUDofHandler`: Degree-of-freedom handler.
- `GRID<:Ferrite.AbstractGPUGrid`: GPU-based grid structure.
- `MAT`: Type of the global memory for stiffness matrices.
- `VEC`: Type of the global memory for force vectors.

# Fields
- `dh`: The degree-of-freedom handler.
- `grid`: The GPU grid being processed.
- `n_cells`: Total number of cells in the grid.
- `ke`: Reference to global memory for cell stiffness matrix.
- `fe`: Reference to global memory for cell force vector.
- `thread_id`: Local thread ID in the CUDA block.
"""
struct CUDAGlobalCellIterator{DH <: Ferrite.GPUDofHandler, GRID <: Ferrite.AbstractGPUGrid, MAT, VEC} <: AbstractCUDACellIterator
    dh::DH
    grid::GRID
    n_cells::Int32
    ke::MAT
    fe::VEC
    thread_id::Int32
end

"""
    Ferrite.CellIterator(dh_::Ferrite.LocalsGPUDofHandler, ::Int32)

Create a `CUDAGlobalCellIterator` for iterating over grid cells using global memory.

# Arguments
- `dh_`: A `LocalsGPUDofHandler` instance containing GPU-local degrees of freedom.
- `::Int32`: Unused placeholder argument.

# Returns
A `CUDAGlobalCellIterator` configured for processing grid cells using global memory.
"""
function Ferrite.CellIterator(dh_::Ferrite.LocalsGPUDofHandler, ::Int32)
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

"""
    GPUCellCache{DOFS, NN, NODES, COORDS, KDynamicSharedMem, FDynamicSharedMem}

Structure to store data for a single finite element cell during GPU computations.

# Fields
- `coords`: Node coordinates of the cell.
- `dofs`: Degrees of freedom associated with the cell.
- `cellid`: ID of the cell.
- `nodes`: Node IDs as a static vector.
- `ke`: View of shared memory for the stiffness matrix.
- `fe`: View of shared memory for the force vector.

# Returns
A `GPUCellCache` object holding cell-specific data for computations.
"""
struct GPUCellCache{DOFS <: AbstractVector{Int32}, NN, NODES <: SVector{NN, Int32}, X, COORDS <: SVector{X}, KDynamicSharedMem, FDynamicSharedMem} <: Ferrite.AbstractKernelCellCache
    coords::COORDS
    dofs::DOFS
    cellid::Int32
    nodes::NODES
    ke::KDynamicSharedMem
    fe::FDynamicSharedMem
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

# Arguments
- `cc`: The `GPUCellCache` object.

# Returns
The node IDs of the current cell.
"""
Ferrite.getnodes(cc::GPUCellCache) = cc.nodes

"""
    getcoordinates(cc::GPUCellCache)

Return the coordinates of the current cell's nodes.

# Arguments
- `cc`: The `GPUCellCache` object.

# Returns
The coordinates of the nodes of the current cell.
"""
Ferrite.getcoordinates(cc::GPUCellCache) = cc.coords

"""
    celldofs(cc::GPUCellCache)

Return the degrees of freedom (DoFs) for the current cell from the cache.

# Arguments
- `cc`: The `GPUCellCache` object.

# Returns
The degrees of freedom (DoFs) associated with the current cell.
"""
Ferrite.celldofs(cc::GPUCellCache) = cc.dofs

"""
    cellid(cc::GPUCellCache)

Return the ID of the current cell stored in the cache.

# Arguments
- `cc`: The `GPUCellCache` object.

# Returns
The ID of the current cell.
"""
Ferrite.cellid(cc::GPUCellCache) = cc.cellid

"""
    Ferrite.cellke(cc::GPUCellCache)

Access the stiffness matrix (`ke`) of the current cell from shared memory and reset it to zero.

# Arguments
- `cc`: The `GPUCellCache` object.

# Returns
The stiffness matrix filled with zeros.
"""
@inline function Ferrite.cellke(cc::GPUCellCache)
    ke = cc.ke
    return fill!(ke, 0.0f0)
end

"""
    Ferrite.cellfe(cc::GPUCellCache)

Access the force vector (`fe`) of the current cell from shared memory and reset it to zero.

# Arguments
- `cc`: The `GPUCellCache` object.

# Returns
The force vector filled with zeros.
"""
@inline function Ferrite.cellfe(cc::GPUCellCache)
    fe = cc.fe
    return fill!(fe, 0.0f0)
end
