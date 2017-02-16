# this file defines iterators used for looping over a grid

export CellIterator

"""
A `CellIterator` contains information about the cell which can be queried from the object.

**Example:**

```julia
ci = CellIterator(grid)

for i in 1:getncells(grid)
    reinit!(grid, ci, i)          # update the CellIterator for cell i
    coords = getcoordinates(cell) # get the coordinates
    nodes = getnodes(cell)        # get the node numbers

    reinit!(cv, ci)               # reinit the FE-base with a CellIterator
end
```
"""
immutable CellIterator{dim, T}
    nodes::Vector{Int}
    coords::Vector{Vec{dim, T}}
end

@inline function CellIterator{dim, N, T}(grid::Grid{dim, N, T})
    nodes = zeros(Int, N)
    coords = zeros(Vec{dim, T}, N)
    return CellIterator(nodes, coords)
end

@inline function reinit!{dim, N, T}(ci::CellIterator{dim, T}, grid::Grid{dim, N, T}, i::Int)
    nodeids = grid.cells[i].nodes
    @inbounds for i in 1:N
        nodeid = nodeids[i]
        ci.nodes[i] = nodeid
        ci.coords[i] = grid.nodes[nodeid].x
    end
end

# utility
@inline getnodes(ci::CellIterator) = ci.nodes
@inline getcoordinates(ci::CellIterator) = ci.coords

@inline reinit!{dim, T}(cv::CellValues{dim, T}, ci::CellIterator{dim, T}) = reinit!(cv, ci.coords)
