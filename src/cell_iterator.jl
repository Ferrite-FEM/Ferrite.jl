
export CellIterator

immutable CellIterator{dim, T}
    cellid::Ref{Int}
    nodes::Vector{Int}
    coords::Vector{Vec{dim, T}}
end

function Base.start{dim, N, T}(grid::Grid{dim, N, T})
    # N is number of nodes per cell in the grid
    nodes = zeros(Int, N)
    coords = zeros(Vec{dim, T}, N)
    return CellIterator(Ref(0), nodes, coords)
end

function Base.next{dim, N, T}(grid::Grid{dim, N, T}, ci::CellIterator{dim, T})
    # this is a bit awkward, we need to basically reinit the state and also send it out
    # that means that we send out the object twice, but it works I guess.
    # the second output state is only used internally and is hidden from the user
    ci.cellid[] += 1
    id = ci.cellid[]

    @inbounds for i in 1:N
        ci.nodes[i] = grid.cells[id].nodes[i]
        ci.coords[i] = grid.nodes[grid.cells[id].nodes[i]].x # this could also use getcoordinates! but this is the same
    end

    return ci, ci
end

Base.done{dim, N, T}(grid::Grid{dim, N, T}, ci::CellIterator{dim, T}) = ci.cellid[] >= getncells(grid)
