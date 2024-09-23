# abstract types and interfaces
abstract type AbstractIterator end
abstract type AbstractCellCache end

abstract type AbstractGPUCellCache <: AbstractCellCache end
abstract type AbstractGPUCellIterator <: AbstractIterator end


function cellCache(iterator::AbstractGPUCellIterator, i::Int32)
    throw(ArgumentError("makecache should be implemented in the derived type"))
end

# concrete types
##### GPU #####
struct GPUCellCache{ DOFS <: AbstractVector{Int32},NN,NODES <: SVector{NN,Int32},X, COORDS<: SVector{X}} <: AbstractGPUCellCache
    # these are the basic fields that are required for the cache (at least from my point of view).
    # we don't want to make this a heavy object, because there will be stanbdalone instances of this object on the GPU.
    coords::COORDS
    dofs::DOFS
    cellid::Int32
    nodes::NODES
end


struct GPUCellIterator{DH<:AbstractGPUDofHandler,GRID<: AbstractGPUGrid} <: AbstractGPUCellIterator
    dh::DH # TODO: subdofhandlers are not supported yet.
    grid::GRID
    n_cells::Int32 # not sure if this is needed.
end


function CellIterator(dh::AbstractGPUDofHandler)
    grid = get_grid(dh)
    n_cells = grid |> getncells |> Int32
    GPUCellIterator(dh, grid,n_cells)
end

ncells(iterator::GPUCellIterator) = iterator.n_cells


function cellcache(iterator::GPUCellIterator, i::Int32)
    # Note: here required fields are all extracted in one single functions,
    # although there are seperate functions to extract each field, because
    # On GPU, we want to minimize the number of memomry accesses.
    dh = iterator.dh
    grid = iterator.grid
    cellid = i
    cell = getcells(grid,i);
    nodes = SVector(convert.(Int32,Ferrite.get_node_ids(cell))...)
    dofs = celldofs(dh, i)  # cannot be a SVectors, because the size is not known at compile time.


    # get the coordinates of the nodes of the cell.
    CT = get_coordinate_type(grid)
    N = nnodes(cell)
    x = MVector{N, CT}(undef) # local array to store the coordinates of the nodes of the cell.
    for i in eachindex(x)
        x[i] = get_node_coordinate(grid, nodes[i])
    end
    coords = SVector(x...)
    return GPUCellCache(coords, dofs, cellid, nodes)
end

# Accessor functions (TODO: Deprecate? We are so inconsistent with `getxx` vs `xx`...)
getnodes(cc::GPUCellCache) = cc.nodes
getcoordinates(cc::GPUCellCache) = cc.coords
celldofs(cc::GPUCellCache) = cc.dofs
cellid(cc::GPUCellCache) = cc.cellid
