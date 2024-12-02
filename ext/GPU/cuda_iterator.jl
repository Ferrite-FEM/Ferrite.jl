##### GPUCellIterator #####

"""
    AbstractCUDACellIterator <: Ferrite.AbstractKernelCellIterator

Abstract type representing CUDA cell iterators for finite element computations
on the GPU. It provides the base for implementing multiple cell iteration strategies (e.g. with shared memory, with global memory).
"""
abstract type AbstractCUDACellIterator <: Ferrite.AbstractKernelCellIterator end


ncells(iterator::AbstractCUDACellIterator) = iterator.n_cells ## any subtype has to have `n_cells` field


struct CUDACellIterator{DH <: Ferrite.GPUDofHandler, GRID <: Ferrite.AbstractGPUGrid, Ti <: Integer, MatrixType, VectorType} <: AbstractCUDACellIterator
    dh::DH
    grid::GRID
    n_cells::Ti
    cell_ke::MatrixType
    cell_fe::VectorType
    #alloc_type::AllocType
end

struct CudaOutOfBoundCellIterator <: AbstractCUDACellIterator end  # used to handle the case for out of bound threads


function Ferrite.CellIterator(dh::Ferrite.GPUDofHandler, buffer_alloc::GlobalMemAlloc)
    grid = get_grid(dh)
    n_cells = grid |> getncells |> Int32
    bd = blockDim().x
    local_thread_id = threadIdx().x
    global_thread_id = (blockIdx().x - Int32(1)) * bd + local_thread_id
    global_thread_id <= n_cells || return CudaOutOfBoundCellIterator()
    cell_ke = cellke(buffer_alloc, global_thread_id)
    cell_fe = cellfe(buffer_alloc, global_thread_id)
    return CUDACellIterator(dh, grid, n_cells, cell_ke, cell_fe)
end


function Ferrite.CellIterator(dh::Ferrite.GPUDofHandler, buffer_alloc::SharedMemAlloc)
    grid = get_grid(dh)
    n_cells = grid |> getncells |> Int32
    block_ke = buffer_alloc.Ke()
    block_fe = buffer_alloc.fe()
    local_thread_id = threadIdx().x
    cell_ke = block_ke[local_thread_id, :, :]
    cell_fe = block_fe[local_thread_id, :]
    #return CUDACellIterator(dh, grid, n_cells, cell_ke, cell_fe, SharedMemAlloc)
    return CUDACellIterator(dh, grid, n_cells, cell_ke, cell_fe)
end


function Base.iterate(iterator::CUDACellIterator)
    i = (blockIdx().x - Int32(1)) * blockDim().x + threadIdx().x
    i <= ncells(iterator)  || return nothing
    return (_makecache(iterator, i), i)
end


function Base.iterate(iterator::CUDACellIterator, state)
    stride = blockDim().x * gridDim().x
    i = state + stride
    i <= ncells(iterator) || return nothing
    return (_makecache(iterator, i), i)
end


Base.iterate(::CudaOutOfBoundCellIterator) = nothing

Base.iterate(::CudaOutOfBoundCellIterator, state) = nothing # I believe this is not necessary


struct GPUCellCache{Ti <: Integer, DOFS <: AbstractVector{Ti}, NN, NODES <: SVector{NN, Ti}, X, COORDS <: SVector{X}, MatrixType, VectorType} <: Ferrite.AbstractKernelCellCache
    coords::COORDS
    dofs::DOFS
    cellid::Ti
    nodes::NODES
    ke::MatrixType
    fe::VectorType
end


function _makecache(iterator::AbstractCUDACellIterator, e::Integer)
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
    return GPUCellCache(coords, dofs, cellid, nodes, iterator.cell_ke, iterator.cell_fe)
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
    return CUDA.fill!(ke, 0.0f0)
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
    return CUDA.fill!(fe, 0.0f0)
end
