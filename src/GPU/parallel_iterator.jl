# This files defines the abstract types and interfaces for GPU iterators.
# The concrete implementations are defined in the extension.

# abstract types and interfaces
abstract type AbstractIterator end
abstract type AbstractCellCache end

abstract type AbstractKernelCellCache <: AbstractCellCache end
abstract type AbstractKernelCellIterator <: AbstractIterator end


function _makecache(iterator::AbstractKernelCellIterator, i::Ti) where {Ti <: Integer}
    throw(ArgumentError("makecache should be implemented in the derived type"))
end

@inline function cellke(::AbstractKernelCellCache)
    throw(ArgumentError("cellke should be implemented in the derived type"))
end

@inline function cellfe(::AbstractKernelCellCache)
    throw(ArgumentError("cellfe should be implemented in the derived type"))
end


## Concrete Implementation for CPU Multithreading ##

##### CPUKernelCellIterator #####
struct CPUKernelCellIterator{DH <: ColoringDofHandler, GRID <: AbstractGrid, Tv} <: AbstractKernelCellIterator
    dh::DH
    grid::GRID
    n_cells::Int
    ke::Matrix{Tv} # 2d local stiffness matrix that is shared among the same thread
    fe::Vector{Tv} # 1d local force vector that is shared among the same thread
    thread_id::Int # thread id that the iterator is working on
end


function CellIterator(dh::ColoringDofHandler, n_basefuncs::Ti) where {Ti <: Integer}
    grid = dh |> dofhandler |> get_grid
    n_cells = grid |> getncells
    ## TODO: Float64 needs to be dependant of the eltype of the matrix
    ke = zeros(Float64, n_basefuncs, n_basefuncs)
    fe = zeros(Float64, n_basefuncs)
    local_thread_id = Threads.threadid()
    return CPUKernelCellIterator(dh, grid, n_cells, ke, fe, local_thread_id)
end


ncells(iterator::CPUKernelCellIterator) = iterator.n_cells


function Base.iterate(iterator::CPUKernelCellIterator)
    i = iterator.thread_id
    curr_color = iterator.dh |> current_color # current color that's being processed
    eles_color = eles_in_color(iterator.dh, curr_color) # elements in the current color
    ncells = length(eles_color)
    i <= ncells || return nothing
    return (_makecache(iterator, eles_color[i]), i)
end


function Base.iterate(iterator::CPUKernelCellIterator, state)
    stride = Threads.nthreads()
    i = state + stride # next strided element id
    curr_color = iterator.dh |> current_color # current color that's being processed
    eles_color = eles_in_color(iterator.dh, curr_color) # elements in the current color
    ncells = length(eles_color)
    i <= ncells || return nothing
    return (_makecache(iterator, eles_color[i]), i)
end


##### CPUKernelCellCache #####
## Future IDEA: we can make this cache mutable, since we are using it on CPU.
struct CPUKernelCellCache{Ti <: Integer, DOFS <: AbstractVector{Ti}, NN, NODES <: SVector{NN, Ti}, X, COORDS <: SVector{X}, Tv <: Real} <: AbstractKernelCellCache
    coords::COORDS
    dofs::DOFS
    cellid::Ti
    nodes::NODES
    ke::Matrix{Tv}
    fe::Vector{Tv}
end


function _makecache(iterator::CPUKernelCellIterator, e::Ti) where {Ti <: Integer}
    dh = iterator.dh |> dofhandler
    grid = iterator.grid
    cellid = e
    cell = getcells(grid, e)

    # Extract the node IDs of the cell.
    nodes = SVector(get_node_ids(cell)...)

    # Extract the degrees of freedom for the cell.
    dofs = celldofs(dh, e)
    # Get the coordinates of the nodes of the cell.
    CT = get_coordinate_type(grid)
    N = nnodes(cell)
    x = MVector{N, CT}(undef)
    for i in eachindex(x)
        x[i] = get_node_coordinate(grid, nodes[i])
    end
    coords = SVector(x...)

    # Return the GPUCellCache containing the cell's data.
    return CPUKernelCellCache(coords, dofs, cellid, nodes, iterator.ke, iterator.fe)
end


getnodes(cc::CPUKernelCellCache) = cc.nodes


getcoordinates(cc::CPUKernelCellCache) = cc.coords


celldofs(cc::CPUKernelCellCache) = cc.dofs


cellid(cc::CPUKernelCellCache) = cc.cellid


@inline function cellke(cc::CPUKernelCellCache)
    ke = cc.ke
    return fill!(ke, zero(eltype(ke)))
end

@inline function cellfe(cc::CPUKernelCellCache)
    fe = cc.fe
    return fill!(fe, zero(eltype(fe)))
end
