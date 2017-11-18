# this file defines iterators used for looping over a grid
struct UpdateFlags
    nodes::Bool
    coords::Bool
    celldofs::Bool
end

UpdateFlags(; nodes::Bool=true, coords::Bool=true, celldofs::Bool=true) =
    UpdateFlags(nodes, coords, celldofs)

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
struct CellIterator{dim, T}
    flags::UpdateFlags
    grid::Grid{dim, T}
    current_cellid::ScalarWrapper{Int}
    cellgroupid::Int
    nodes::Vector{Int}
    coords::Vector{Vec{dim, T}}
    dh::DofHandler{dim, T}
    celldofs::Vector{Int}

    function CellIterator{dim, T}(dh::DofHandler{dim, T}, cellgroupid::Int, flags::UpdateFlags) where {dim, T}
        cell = ScalarWrapper(0)
        N = ndofs_per_cell(dh, cellgroupid)
        nodes = zeros(Int, N)
        coords = zeros(Vec{dim, T}, N)
        n = ndofs_per_cell(dh, cellgroupid)
        celldofs = zeros(Int, n)
        return new{dim,T}(flags, dh.grid, cell, cellgroupid, nodes, coords, dh, celldofs)
    end

    function CellIterator{dim, T}(grid::Grid{dim, T}, cellgroupid::Int, flags::UpdateFlags) where {dim, T}
        cell = ScalarWrapper(0)
        N = ndofs_per_cell(dh, cellgroupid)
        nodes = zeros(Int, N)
        coords = zeros(Vec{dim, T}, N)
        return new{dim, T}(flags, grid, cell, cellgroupid, nodes, coords)
    end
end

CellIterator(grid::Grid{dim, T},     cellgroupid::Int,     flags::UpdateFlags=UpdateFlags()) where {dim, T} =
    CellIterator{dim,T}(grid, cellgroupid, flags)
CellIterator(dh::DofHandler{dim, T}, cellgroupid::Int,     flags::UpdateFlags=UpdateFlags()) where {dim, T} =
    CellIterator{dim,T}(dh, cellgroupid, flags)

# iterator interface
Base.start(::CellIterator)     = 1
Base.next(ci::CellIterator, i) = (reinit!(ci, i), i+1)
Base.done(ci::CellIterator, i) = i > getncells(ci.grid, ci.cellgroupid)
Base.length(ci::CellIterator)  = getncells(ci.grid, ci.cellgroupid)

Base.iteratorsize(::Type{T})   where {T <: CellIterator} = Base.HasLength() # this is default in Base
Base.iteratoreltype(::Type{T}) where {T <: CellIterator} = Base.HasEltype() # this is default in Base
Base.eltype(::Type{T})         where {T <: CellIterator} = T

# utility
@inline getnodes(ci::CellIterator) = ci.nodes
@inline getcoordinates(ci::CellIterator) = ci.coords
@inline nfaces(ci::CellIterator) = nfaces(eltype(ci.grid.cells))
@inline onboundary(ci::CellIterator, face::Int) = ci.grid.boundary_matrix[face, ci.current_cellid[]]
@inline cellid(ci::CellIterator) = ci.current_cellid[]
@inline celldofs!(v::Vector, ci::CellIterator) = celldofs!(v, ci.dh, ci.cellgroupid, ci.current_cellid[])
@inline celldofs(ci::CellIterator) = ci.celldofs

function reinit!(ci::CellIterator{dim}, i::Int) where {dim}
    nodeids = ci.grid.cellgroups[ci.cellgroupid][i].nodes
    ci.current_cellid[] = i
    N = ndofs_per_cell(ci.dh, ci.cellgroupid)
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

@inline reinit!(cv::CellValues{dim, T}, ci::CellIterator{dim, T}) where {dim, T} = reinit!(cv, ci.coords)
@inline reinit!(fv::FaceValues{dim, T}, ci::CellIterator{dim, T}, face::Int) where {dim, T} = reinit!(fv, ci.coords, face)


#=
"""
```julia
FaceIterator(grid::Grid)
```

A `FaceIterator` is used to conveniently loop over all the faces in a grid.

**Example:**

```julia
for face in FaceIterator(grid)
    coords = getcoordinates(face) # get the coordinates
    nodes = getnodes(face)        # get the node numbers

    reinit!(fv, face)             # reinit! the FE-base with a FaceIterator
end
```
"""


struct FaceIterator{dim, N, T}
    grid::Grid{dim, N, T}
    nodes_face::Vector{Int}
    coords_cell::Vector{Vec{dim, T}}
    coords_face::Vector{Vec{dim, T}}
    current_face::ScalarWrapper{Int}
end

function FaceIterator{dim, N, T}(grid::Grid{dim, N, T})
    coords_cell = zeros(Vec{dim, T}, N)
    n_vertices_face = length(getfacelist(grid)[1])
    nodes_face = zeros(Int, n_vertices_face)
    coords_face = zeros(Vec{dim, T}, n_vertices_face)
    return FaceIterator(grid, nodes_face, coords_cell, coords_face, ScalarWrapper(1))
end

# iterator interface
# cell, face
Base.start(::FaceIterator) = (1, 1)
function Base.next{dim, N, T}(fi::FaceIterator{dim, N, T}, state)
    cell, face = state
    fi2 = reinit!(fi, cell, face)
    fi.current_face[] = face
    face += 1
    if face > n_faces_per_cell(fi.grid)
        face = 1
        cell += 1
    end
    (fi2, (cell, face))
end

Base.done(fi::FaceIterator, state) = state[1] > getncells(fi.grid)
Base.length(fi::FaceIterator) = getncells(fi.grid) * n_faces_per_cell(grid)

Base.iteratorsize{dim, N, T}(::Type{FaceIterator{dim, N, T}}) = Base.HasLength()   # this is default in Base
Base.iteratoreltype{dim, N, T}(::Type{FaceIterator{dim, N, T}}) = Base.HasEltype() # this is default in Base
Base.eltype{dim, N, T}(::Type{FaceIterator{dim, N, T}}) = FaceIterator{dim, N, T}

# utility
@inline getnodes(fi::FaceIterator) = fi.nodes_face
@inline getcoordinates(fi::FaceIterator) = fi.coords_face

function reinit!{dim, N, T}(fi::FaceIterator{dim, N, T}, cell::Int, face::Int)
    nodeids = fi.grid.cells[cell].nodes
    @inbounds for j = 1:N
        nodeid = nodeids[j]
        fi.coords_cell[j] = fi.grid.nodes[nodeid].x
    end

    @inbounds for (j, v) in enumerate(getfacelist(fi.grid)[fi.current_face[]])
        nodeid = nodeids[v]
        fi.coords_face[j] = fi.grid.nodes[nodeid].x
        fi.nodes_face[j] =  nodeid
    end
    return fi
end
=#

