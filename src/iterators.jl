# this file defines iterators used for looping over a grid

export CellIterator, FaceIterator

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
immutable CellIterator{dim, N, T, M}
    grid::Grid{dim, N, T, M}
    nodes::Vector{Int}
    coords::Vector{Vec{dim, T}}
    current_cellid::ScalarWrapper{Int}
    dh::DofHandler{dim, N, T, M}
    celldofs::Vector{Int}

    function (::Type{CellIterator{dim, N, T, M}}){dim, N, T, M}(dh::DofHandler{dim, N, T, M})
        nodes = zeros(Int, N)
        coords = zeros(Vec{dim, T}, N)
        cell = ScalarWrapper(0)
        celldofs = zeros(Int, ndofs_per_cell(dh))
        return new{dim, N, T, M}(dh.grid, nodes, coords, ScalarWrapper(0), dh, celldofs)
    end

    function (::Type{CellIterator{dim, N, T, M}}){dim, N, T, M}(grid::Grid{dim, N, T, M})
        nodes = zeros(Int, N)
        coords = zeros(Vec{dim, T}, N)
        cell = ScalarWrapper(0)
        return new{dim, N, T, M}(grid, nodes, coords, cell)
    end
end

CellIterator{dim, N, T, M}(grid::Grid{dim, N, T, M}) = CellIterator{dim,N,T,M}(grid)
CellIterator{dim, N, T, M}(dh::DofHandler{dim, N, T, M}) = CellIterator{dim,N,T,M}(dh)

# iterator interface
Base.start(::CellIterator)     = 1
Base.next(ci::CellIterator, i) = (reinit!(ci, i), i+1)
Base.done(ci::CellIterator, i) = i > getncells(ci.grid)
Base.length(ci::CellIterator)  = getncells(ci.grid)

Base.iteratorsize{T <: CellIterator}(::Type{T})   = Base.HasLength()   # this is default in Base
Base.iteratoreltype{T <: CellIterator}(::Type{T}) = Base.HasEltype() # this is default in Base
Base.eltype{T <: CellIterator}(::Type{T})         = T

# utility
@inline getnodes(ci::CellIterator) = ci.nodes
@inline getcoordinates(ci::CellIterator) = ci.coords
@inline nfaces(ci::CellIterator) = nfaces(eltype(ci.grid.cells))
@inline onboundary(ci::CellIterator, face::Int) = ci.grid.boundary_matrix[face, ci.current_cellid[]]
@inline cellid(ci::CellIterator) = ci.current_cellid[]
@inline celldofs!(v::Vector, ci::CellIterator) = celldofs!(v, ci.dh, ci.current_cellid[])
@inline celldofs(ci::CellIterator) = ci.celldofs

function reinit!{dim, N}(ci::CellIterator{dim, N}, i::Int)
    nodeids = ci.grid.cells[i].nodes
    ci.current_cellid[] = i
    @inbounds for j in 1:N
        nodeid = nodeids[j]
        ci.nodes[j] = nodeid
        ci.coords[j] = ci.grid.nodes[nodeid].x
    end
    if isdefined(ci, :dh) # update celldofs
        celldofs!(ci.celldofs, ci)
    end
    return ci
end

@inline reinit!{dim, N, T}(cv::CellValues{dim, T}, ci::CellIterator{dim, N, T}) = reinit!(cv, ci.coords)
@inline reinit!{dim, N, T}(fv::FaceValues{dim, T}, ci::CellIterator{dim, N, T}, face::Int) = reinit!(fv, ci.coords, face)
