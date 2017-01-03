using Base: RefValue

export getid

"""
A `CellIterator` is returned when looping over the cells in a grid. The `CellIterator`
contains information about the cell which can be queried from the object.

**Example:**

```julia
for cell in grid                 # cell is now a CellIterator

    id = getid(cell)             # get the cell number
    coord = getcoordinates(cell) # get the coordinates
    nodes = getnodes(cell)       # get the node numbers

end
```
The `CellIterator` can also be used directly to [`reinit!`](@ref) the [`CellValues`](@ref):

```julia
for cell in grid

    reinit!(cv, cell)

    # do stuff

end
```
"""
immutable CellIterator{dim, T}
    cellid::RefValue{Int}
    nodes::Vector{Int}
    coords::Vector{Vec{dim, T}}
end

@inline function Base.start{dim, N, T}(grid::Grid{dim, N, T})
    nodes = zeros(Int, N)
    coords = zeros(Vec{dim, T}, N)
    return CellIterator(RefValue{Int}(0), nodes, coords)
end

@inline function Base.next{dim, N, T}(grid::Grid{dim, N, T}, ci::CellIterator{dim, T})
    ci.cellid[] += 1
    nodeids = grid.cells[ci.cellid[]].nodes
    @inbounds for i in 1:N
        nodeid = nodeids[i]
        ci.nodes[i] = nodeid
        ci.coords[i] = grid.nodes[nodeid].x
    end

    return ci, ci
end

@inline Base.done{dim, N, T}(grid::Grid{dim, N, T}, ci::CellIterator{dim, T}) = ci.cellid[] >= getncells(grid)

@inline getid(ci::CellIterator) = ci.cellid[]
@inline getnodes(ci::CellIterator) = ci.nodes
@inline getcoordinates(ci::CellIterator) = ci.coords

@inline reinit!{dim, T}(cv::CellValues{dim, T}, ci::CellIterator{dim, T}) = reinit!(cv, ci.coords)
