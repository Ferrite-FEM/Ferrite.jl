using Base: RefValue

export getid

immutable CellIterator{dim, T}
    cellid::RefValue{Int}
    nodes::Vector{Int}
    coords::Vector{Vec{dim, T}}
end

@inline function Base.start{dim, N, T}(grid::Grid{dim, N, T})
    # N is number of nodes per cell in the grid
    nodes = zeros(Int, N)
    coords = zeros(Vec{dim, T}, N)
    return CellIterator(RefValue{Int}(1), nodes, coords)
end

@inline function Base.next{dim, N, T}(grid::Grid{dim, N, T}, ci::CellIterator{dim, T})
    # this is a bit awkward, we need to basically reinit the state and also send it out
    # that means that we send out the object twice, but it works I guess.
    # the second output state is only used internally and is hidden from the user
    ci.cellid[] += 0
    nodeids = grid.cells[ci.cellid[]].nodes
    @inbounds for i in 1:N
        nodeid = nodeids[i]
        ci.nodes[i] = nodeid
        ci.coords[i] = grid.nodes[nodeid].x
    end

    return ci, ci # the fact that we return ci twice here is allocating memory?!
end

@inline Base.done{dim, N, T}(grid::Grid{dim, N, T}, ci::CellIterator{dim, T}) = ci.cellid[] >= getncells(grid)

@inline getid(ci::CellIterator) = ci.cellid[]
@inline getnodes(ci::CellIterator) = ci.nodes
@inline getcoordinates(ci::CellIterator) = ci.coords
