# abstract types and interfaces
abstract type AbstractIterator end
abstract type AbstractCellCache end

abstract type AbstractGPUCellCache <: AbstractCellCache end
abstract type AbstractGPUCellIterator <: AbstractIterator end


function _makecache(iterator::AbstractGPUCellIterator, i::Int32)
    throw(ArgumentError("makecache should be implemented in the derived type"))
end

# concrete types
##### GPUCellIterator #####
struct GPUCellIterator{DH<:AbstractGPUDofHandler,GRID<: AbstractGPUGrid,KDynamicSharedMem,FDynamicSharedMem} <: AbstractGPUCellIterator
    dh::DH # TODO: subdofhandlers are not supported yet.
    grid::GRID
    n_cells::Int32
    block_ke:: KDynamicSharedMem # dynamic shared memory for the block (3rd order tensor (e,i,j))
    block_fe:: FDynamicSharedMem # dynamic shared memory for the block (2nd order tensor (e,i))
    thread_id::Int32 # local thread id (maps to the index of the element in block_ke and block_fe)
end


function CellIterator(dh::AbstractGPUDofHandler,n_basefuncs::Int32)
    grid = get_grid(dh)
    n_cells = grid |> getncells |> Int32
    bd = blockDim().x
    ke_shared = @cuDynamicSharedMem(Float32,(bd,n_basefuncs,n_basefuncs))
    fe_shared = @cuDynamicSharedMem(Float32,(bd,n_basefuncs),sizeof(Float32)*bd*n_basefuncs*n_basefuncs)
    local_thread_id = threadIdx().x
    GPUCellIterator(dh, grid,n_cells,ke_shared,fe_shared,local_thread_id)
end

ncells(iterator::GPUCellIterator) = iterator.n_cells

function Base.iterate(iterator::GPUCellIterator)
    i = (blockIdx().x - Int32(1)) * blockDim().x + threadIdx().x
    i <= iterator.n_cells || return nothing
    return (_makecache(iterator, i), i)
end

function Base.iterate(iterator::GPUCellIterator, state)
    stride = blockDim().x * gridDim().x
    i = state + stride
    i <= iterator.n_cells || return nothing
    return (_makecache(iterator, i), i)
end


##### GPUCellCache #####
struct GPUCellCache{ DOFS <: AbstractVector{Int32},NN,NODES <: SVector{NN,Int32},X, COORDS<: SVector{X},KDynamicSharedMem,FDynamicSharedMem} <: AbstractGPUCellCache
    # these are the basic fields that are required for the cache (at least from my point of view).
    # we don't want to make this a heavy object, because there will be stanbdalone instances of this object on the GPU.
    coords::COORDS
    dofs::DOFS
    cellid::Int32
    nodes::NODES
    ke::KDynamicSharedMem # view of the dynamic shared memory for the cell (i.e. element local stiffness matrix).
    fe::FDynamicSharedMem # view of the dynamic shared memory for the cell (i.e. element local force vector).
end


function _makecache(iterator::GPUCellIterator, e::Int32)
    # Note: here required fields are all extracted in one single functions,
    # although there are seperate functions to extract each field, because
    # On GPU, we want to minimize the number of memomry accesses.
    dh = iterator.dh
    grid = iterator.grid
    cellid = e
    cell = getcells(grid,e);
    nodes = SVector(convert.(Int32,Ferrite.get_node_ids(cell))...)
    dofs = celldofs(dh, e)


    # get the coordinates of the nodes of the cell.
    CT = get_coordinate_type(grid)
    N = nnodes(cell)
    x = MVector{N, CT}(undef) # local array to store the coordinates of the nodes of the cell.
    for i in eachindex(x)
        x[i] = get_node_coordinate(grid, nodes[i])
    end
    coords = SVector(x...)
    return GPUCellCache(coords, dofs, cellid, nodes, (@view iterator.block_ke[iterator.thread_id,:,:]), (@view iterator.block_fe[iterator.thread_id,:,:]))
end


getnodes(cc::GPUCellCache) = cc.nodes
getcoordinates(cc::GPUCellCache) = cc.coords
celldofs(cc::GPUCellCache) = cc.dofs
cellid(cc::GPUCellCache) = cc.cellid
@inline function cellke(cc::GPUCellCache)
    ke =  cc.ke
    fill!(ke, 0.0f0)
end

@inline function cellfe(cc::GPUCellCache)
    fe =  cc.fe
    fill!(fe, 0.0f0)
end
