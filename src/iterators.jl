# this file defines iterators used for looping over a grid

export CellIterator

"""
A `CellIterator` contains information about the cell which can be queried from the object.

**Example:**

```julia
for cell in CellIterator(grid)
    coords = getcoordinates(cell) # get the coordinates
    nodes = getnodes(cell)        # get the node numbers

    reinit!(cv, cell)             # reinit! the FE-base with a CellIterator
end
```
"""
immutable CellIterator{dim, N, T}
    grid::Grid{dim, N, T}
    nodes::Vector{Int}
    coords::Vector{Vec{dim, T}}
end

function CellIterator{dim, N, T}(grid::Grid{dim, N, T})
    nodes = zeros(Int, N)
    coords = zeros(Vec{dim, T}, N)
    return CellIterator(grid, nodes, coords)
end

@inline Base.start(::CellIterator) = 1

@inline function Base.next{dim, N, T}(ci::CellIterator{dim, N, T}, i)
    nodeids = ci.grid.cells[i].nodes
    @inbounds for j in 1:N
        nodeid = nodeids[j]
        ci.nodes[j] = nodeid
        ci.coords[j] = ci.grid.nodes[nodeid].x
    end
    return (ci, i+1)
end

@inline Base.done(ci::CellIterator, i) = i > getncells(ci.grid)

# utility
@inline getnodes(ci::CellIterator) = ci.nodes
@inline getcoordinates(ci::CellIterator) = ci.coords
@inline reinit!(ci::CellIterator, i::Int) = next(ci, i) # for manual updating

@inline reinit!{dim, N, T}(cv::CellValues{dim, T}, ci::CellIterator{dim, N, T}) = reinit!(cv, ci.coords)
