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
    CellIterator(grid::DofHandler)

Return a `CellIterator` to conveniently loop over all the cells in a grid.

# Examples
```julia
for cell in CellIterator(grid)
    coords = getcoordinates(cell) # get the coordinates
    dofs = celldofs(cell)         # get the dofs for this cell
    reinit!(cv, cell)             # reinit! the FE-base with a CellIterator
end
```
"""
struct CellIterator{dim,C,T}
    flags::UpdateFlags
    grid::Grid{dim,C,T}
    current_cellid::ScalarWrapper{Int}
    nodes::Vector{Int}
    coords::Vector{Vec{dim,T}}
    cellset::Vector{Int}
    dh::Union{DofHandler{dim,C,T}, MixedDofHandler{dim,C,T}} #Future: remove DofHandler and rename MixedDofHandler->DofHandler
    celldofs::Vector{Int}

    function CellIterator{dim,C,T}(dh::Union{DofHandler{dim,C,T}, MixedDofHandler{dim,C,T}}, cellset::AbstractVector{Int}, flags::UpdateFlags) where {dim,C,T}
        isconcretetype(C) || _check_same_celltype(grid, cellset)
        N = nnodes_per_cell(dh.grid, first(cellset))
        cell = ScalarWrapper(0)
        nodes = zeros(Int, N)
        coords = zeros(Vec{dim,T}, N)
        n = ndofs_per_cell(dh)
        celldofs = zeros(Int, n)
        return new{dim,C,T}(flags, dh.grid, cell, nodes, coords, cellset, dh, celldofs)
    end

    function CellIterator{dim,C,T}(grid::Grid{dim,C,T}, cellset::AbstractVector{Int}, flags::UpdateFlags) where {dim,C,T}
        isconcretetype(C) || _check_same_celltype(grid, cellset)
        N = nnodes_per_cell(grid, first(cellset))
        cell = ScalarWrapper(0)
        nodes = zeros(Int, N)
        coords = zeros(Vec{dim,T}, N)
        return new{dim,C,T}(flags, grid, cell, nodes, coords)
    end
end

CellIterator(grid::Grid{dim,C,T}, cellset::AbstractVector{Int}=1:getncells(grid), flags::UpdateFlags=UpdateFlags()) where {dim,C,T} =
    CellIterator{dim,C,T}(grid, cellset, flags)
CellIterator(dh::Union{DofHandler{dim,C,T}, MixedDofHandler{dim,C,T}}, cellset::AbstractVector{Int}=1:getncells(dh.grid), flags::UpdateFlags=UpdateFlags()) where {dim,C,T} =
    CellIterator{dim,C,T}(dh, cellset, flags)

# iterator interface
function Base.iterate(ci::CellIterator, state = 1)
    if state > getncells(ci.grid)
        return nothing
    else
        return (reinit!(ci, state), state+1)
    end
end
Base.length(ci::CellIterator)  = getncells(ci.grid)

Base.IteratorSize(::Type{T})   where {T<:CellIterator} = Base.HasLength() # this is default in Base
Base.IteratorEltype(::Type{T}) where {T<:CellIterator} = Base.HasEltype() # this is default in Base
Base.eltype(::Type{T})         where {T<:CellIterator} = T

# utility
@inline getnodes(ci::CellIterator) = ci.nodes
@inline getcoordinates(ci::CellIterator) = ci.coords
@inline nfaces(ci::CellIterator) = nfaces(eltype(ci.grid.cells))
@inline onboundary(ci::CellIterator, face::Int) = ci.grid.boundary_matrix[face, ci.current_cellid[]]
@inline cellid(ci::CellIterator) = ci.current_cellid[]
@inline celldofs(ci::CellIterator) = ci.celldofs

function reinit!(ci::CellIterator{dim,C}, i::Int) where {dim,C}
    ci.current_cellid[] = ci.cellset[i]
    
    ci.flags.nodes  && cellnodes!(ci.nodes, ci.dh, ci.current_cellid[])
    ci.flags.coords && cellcoords!(ci.coords, ci.dh, ci.current_cellid[])
    
    if isdefined(ci, :dh) && ci.flags.celldofs # update celldofs
        celldofs!(ci.celldofs, ci.dh, ci.current_cellid[])
    end
    return ci
end

function check_compatible_geointerpolation(cv::Union{CellValues, FaceValues}, ci::CellIterator)
    if length(getnodes(ci)) != getngeobasefunctions(cv)
        msg = """The given CellValues and CellIterator are incompatiblet.
        Likely an appropriate geometry interpolate must be passed when constructing the CellValues.
        See also issue #265: https://github.com/KristofferC/JuAFEM.jl/issues/265"""
        throw(ArgumentError(msg))
    end
end

@inline function reinit!(cv::CellValues{dim,T}, ci::CellIterator{dim,N,T}) where {dim,N,T}
    check_compatible_geointerpolation(cv, ci)
    reinit!(cv, ci.coords)
end

@inline function reinit!(fv::FaceValues{dim,T}, ci::CellIterator{dim,N,T}, face::Int) where {dim,N,T}
    check_compatible_geointerpolation(fv, ci)
    reinit!(fv, ci.coords, face)
end

function _check_same_celltype(grid::AbstractGrid, cellset::AbstractVector{Int})
    celltype = typeof(grid.cells[first(cellset)])
    for cellid in cellset
        if celltype != typeof(grid.cells[cellid])
            error("You are trying to use a CellIterator to loop over a cellset with different celltypes.")
        end
    end
end