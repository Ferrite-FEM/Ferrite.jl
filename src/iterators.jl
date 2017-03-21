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

# iterator interface
Base.start(::CellIterator) = 1
Base.next{dim, N, T}(ci::CellIterator{dim, N, T}, i) = (reinit!(ci, i), i+1)
Base.done(ci::CellIterator, i) = i > getncells(ci.grid)
Base.length(ci::CellIterator) = getncells(ci.grid)

Base.iteratorsize{dim, N, T}(::Type{CellIterator{dim, N, T}}) = Base.HasLength()   # this is default in Base
Base.iteratoreltype{dim, N, T}(::Type{CellIterator{dim, N, T}}) = Base.HasEltype() # this is default in Base
Base.eltype{dim, N, T}(::Type{CellIterator{dim, N, T}}) = CellIterator{dim, N, T}

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

immutable FaceIterator{dim, N, T}
    grid::Grid{dim, N, T}
    nodes_face::Vector{Int}
    coords_cell::Vector{Vec{dim, T}}
    coords_face::Vector{Vec{dim, T}}
    current_face::Ref{Int}
end

function FaceIterator{dim, N, T}(grid::Grid{dim, N, T})
    coords_cell = zeros(Vec{dim, T}, N)
    n_vertices_face = length(getfacelist(grid)[1])
    nodes_face = zeros(Int, n_vertices_face)
    coords_face = zeros(Vec{dim, T}, n_vertices_face)
    return FaceIterator(grid, nodes_face, coords_cell, coords_face, Ref(1))
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

@inline reinit!{dim, N, T}(fv::FaceValues{dim, T}, fi::FaceIterator{dim, N, T}) = reinit!(fv, fi.coords_cell, fi.current_face[])
