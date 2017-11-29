# this file defines iterators used for looping over a grid
struct UpdateFlags
    nodes::Bool
    coords::Bool
    celldofs::Bool
end

UpdateFlags(; nodes::Bool=true, coords::Bool=true, celldofs::Bool=true) =
    UpdateFlags(nodes, coords, celldofs)

"""
    CellIterator(grid::Grid)

Return a `CellIterator` to conveniently loop over all the cells in a grid.

# Examples
```julia
for cell in CellIterator(grid)
    coords = getcoordinates(cell) # get the coordinates
    nodes = getnodes(cell)        # get the node numbers

    reinit!(cv, cell)             # reinit! the FE-base with a CellIterator
end
```
"""
struct CellIterator{dim,N,T,M}
    flags::UpdateFlags
    grid::Grid{dim,N,T,M}
    current_cellid::ScalarWrapper{Int}
    nodes::Vector{Int}
    coords::Vector{Vec{dim,T}}
    dh::DofHandler{dim,N,T,M}
    celldofs::Vector{Int}

    function CellIterator{dim,N,T,M}(dh::DofHandler{dim,N,T,M}, flags::UpdateFlags) where {dim,N,T,M}
        cell = ScalarWrapper(0)
        nodes = zeros(Int, N)
        coords = zeros(Vec{dim,T}, N)
        n = ndofs_per_cell(dh)
        celldofs = zeros(Int, n)
        return new{dim,N,T,M}(flags, dh.grid, cell, nodes, coords, dh, celldofs)
    end

    function CellIterator{dim,N,T,M}(grid::Grid{dim,N,T,M}, flags::UpdateFlags) where {dim,N,T,M}
        cell = ScalarWrapper(0)
        nodes = zeros(Int, N)
        coords = zeros(Vec{dim,T}, N)
        return new{dim,N,T,M}(flags, grid, cell, nodes, coords)
    end
end

CellIterator(grid::Grid{dim,N,T,M},     flags::UpdateFlags=UpdateFlags()) where {dim,N,T,M} =
    CellIterator{dim,N,T,M}(grid, flags)
CellIterator(dh::DofHandler{dim,N,T,M}, flags::UpdateFlags=UpdateFlags()) where {dim,N,T,M} =
    CellIterator{dim,N,T,M}(dh, flags)

# iterator interface
Base.start(::CellIterator)     = 1
Base.next(ci::CellIterator, i) = (reinit!(ci, i), i+1)
Base.done(ci::CellIterator, i) = i > getncells(ci.grid)
Base.length(ci::CellIterator)  = getncells(ci.grid)

Base.iteratorsize(::Type{T})   where {T<:CellIterator} = Base.HasLength() # this is default in Base
Base.iteratoreltype(::Type{T}) where {T<:CellIterator} = Base.HasEltype() # this is default in Base
Base.eltype(::Type{T})         where {T<:CellIterator} = T

# utility
@inline getnodes(ci::CellIterator) = ci.nodes
@inline getcoordinates(ci::CellIterator) = ci.coords
@inline nfaces(ci::CellIterator) = nfaces(eltype(ci.grid.cells))
@inline onboundary(ci::CellIterator, face::Int) = ci.grid.boundary_matrix[face, ci.current_cellid[]]
@inline cellid(ci::CellIterator) = ci.current_cellid[]
@inline celldofs!(v::Vector, ci::CellIterator) = celldofs!(v, ci.dh, ci.current_cellid[])
@inline celldofs(ci::CellIterator) = ci.celldofs

function reinit!(ci::CellIterator{dim,N}, i::Int) where {dim,N}
    nodeids = ci.grid.cells[i].nodes
    ci.current_cellid[] = i
    @inbounds for j in 1:N
        nodeid = nodeids[j]
        ci.flags.nodes  && (ci.nodes[j] = nodeid)
        ci.flags.coords && (ci.coords[j] = ci.grid.nodes[nodeid].x)
    end
    if isdefined(ci, :dh) && ci.flags.celldofs # update celldofs
        celldofs!(ci.celldofs, ci)
    end
    return ci
end

@inline reinit!(cv::CellValues{dim,T}, ci::CellIterator{dim,N,T}) where {dim,N,T} = reinit!(cv, ci.coords)
@inline reinit!(fv::FaceValues{dim,T}, ci::CellIterator{dim,N,T}, face::Int) where {dim,N,T} = reinit!(fv, ci.coords, face)
