# This files defines the abstract types and interfaces for GPU iterators.
# The concrete implementations are defined in the extension.

# abstract types and interfaces
abstract type AbstractIterator end
abstract type AbstractCellCache end

abstract type AbstractKernelCellCache <: AbstractCellCache end
abstract type AbstractKernelCellIterator <: AbstractIterator end

@inline function cellke(::AbstractKernelCellCache)
    throw(ArgumentError("cellke should be implemented in the derived type"))
end

@inline function cellfe(::AbstractKernelCellCache)
    throw(ArgumentError("cellfe should be implemented in the derived type"))
end


## Concrete Implementation for CPU Multithreading ##


##### CPUKernelCellCache #####
mutable struct CPUKernelCellCache{G <: AbstractGrid, X, Tv <: Real} <: AbstractKernelCellCache
    const flags::UpdateFlags
    const grid::G
    const dh::AbstractDofHandler
    const coords::Vector{X}
    const dofs::Vector{Int}
    cellid::Int
    const nodes::Vector{Int}
    const ke::Matrix{Tv}
    const fe::Vector{Tv}
end


function CellCache(dh::DofHandler{dim}, n_basefuncs::Int, flags::UpdateFlags = UpdateFlags()) where {dim}
    ke = zeros(Float64, n_basefuncs, n_basefuncs)
    fe = zeros(Float64, n_basefuncs)
    grid = dh |> get_grid
    N = nnodes_per_cell(grid, 1) # nodes and coords will be resized in `reinit!`
    nodes = zeros(Int, N)
    coords = zeros(Vec{dim, Float64}, N)
    return CPUKernelCellCache(flags, grid, dh, coords, Int[], -1, nodes, ke, fe)
end

# function _makecache(iterator::CPUKernelCellIterator, e::Ti) where {Ti <: Integer}
#     dh = iterator.dh |> dofhandler
#     grid = iterator.grid
#     cellid = e
#     cell = getcells(grid, e)

#     # Extract the node IDs of the cell.
#     nodes = SVector(get_node_ids(cell)...)

#     # Extract the degrees of freedom for the cell.
#     dofs = celldofs(dh, e)
#     # Get the coordinates of the nodes of the cell.
#     CT = get_coordinate_type(grid)
#     N = nnodes(cell)
#     x = MVector{N, CT}(undef)
#     for i in eachindex(x)
#         x[i] = get_node_coordinate(grid, nodes[i])
#     end
#     coords = SVector(x...)

#     # Return the GPUCellCache containing the cell's data.
#     return CPUKernelCellCache(coords, dofs, cellid, nodes, iterator.ke, iterator.fe)
# end

function _reinit!(cc::CPUKernelCellCache, i::Int)
    cc.cellid = i
    fill!(cc.ke, zero(eltype(cc.ke)))
    fill!(cc.fe, zero(eltype(cc.fe)))
    if cc.flags.nodes
        resize!(cc.nodes, nnodes_per_cell(cc.grid, i))
        cellnodes!(cc.nodes, cc.grid, i)
    end
    if cc.flags.coords
        resize!(cc.coords, nnodes_per_cell(cc.grid, i))
        getcoordinates!(cc.coords, cc.grid, i)
    end
    if cc.flags.dofs
        resize!(cc.dofs, ndofs_per_cell(cc.dh, i))
        celldofs!(cc.dofs, cc.dh, i)
    end
    return cc
end

## Accessors ##
getnodes(cc::CPUKernelCellCache) = cc.nodes


getcoordinates(cc::CPUKernelCellCache) = cc.coords


celldofs(cc::CPUKernelCellCache) = cc.dofs


cellid(cc::CPUKernelCellCache) = cc.cellid


cellke(cc::CPUKernelCellCache) = cc.ke

cellfe(cc::CPUKernelCellCache) = cc.fe


##### CPUKernelCellIterator #####
struct CPUKernelCellIterator{CC <: CPUKernelCellCache, DH <: ColoringDofHandler} <: AbstractKernelCellIterator
    cache::CC
    dh::DH
    n_cells::Int
    thread_id::Int # thread id that the iterator is working on
end


function CellIterator(dh::ColoringDofHandler, n_basefuncs::Int)
    grid = dh |> dofhandler |> get_grid
    n_cells = grid |> getncells
    cache = CellCache(dh |> dofhandler, n_basefuncs)
    local_thread_id = Threads.threadid()
    return CPUKernelCellIterator(cache, dh, n_cells, local_thread_id)
end


ncells(iterator::CPUKernelCellIterator) = iterator.n_cells
_cache(iterator::CPUKernelCellIterator) = iterator.cache


function Base.iterate(iterator::CPUKernelCellIterator)
    i = iterator.thread_id
    curr_color = iterator.dh |> current_color # current color that's being processed
    eles_color = eles_in_color(iterator.dh, curr_color) # elements in the current color
    ncells = length(eles_color)
    i <= ncells || return nothing
    cache = _cache(iterator)
    _reinit!(cache, eles_color[i])
    return (cache, i)
end


function Base.iterate(iterator::CPUKernelCellIterator, state)
    stride = Threads.nthreads()
    i = state + stride # next strided element id
    curr_color = iterator.dh |> current_color # current color that's being processed
    eles_color = eles_in_color(iterator.dh, curr_color) # elements in the current color
    ncells = length(eles_color)
    i <= ncells || return nothing
    cache = _cache(iterator)
    _reinit!(cache, eles_color[i])
    return (cache, i)
end
