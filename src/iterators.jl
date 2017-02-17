# this file defines iterators used for looping over a grid

export CellIterator

"""
```julia
CellIterator(grid::Grid)
```

A `CellIterator` is used to conveniently loop over all the cells in a grid.

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

Base.start(::CellIterator) = 1
Base.next{dim, N, T}(ci::CellIterator{dim, N, T}, i) = (reinit!(ci, i), i)
Base.done(ci::CellIterator, i) = i > getncells(ci.grid)

# utility
@inline getnodes(ci::CellIterator) = ci.nodes
@inline getcoordinates(ci::CellIterator) = ci.coords

function reinit!{dim, N, T}(ci::CellIterator{dim, N, T}, i::Int)
    nodeids = ci.grid.cells[i].nodes
    @inbounds for j in 1:N
        nodeid = nodeids[j]
        ci.nodes[j] = nodeid
        ci.coords[j] = ci.grid.nodes[nodeid].x
    end
    return ci
end

@inline reinit!{dim, N, T}(cv::CellValues{dim, T}, ci::CellIterator{dim, N, T}) = reinit!(cv, ci.coords)
