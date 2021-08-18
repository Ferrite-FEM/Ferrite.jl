#########################
# Main types for meshes #
#########################
"""
A `Node` is a point in space.
"""
struct Node{dim,T}
    x::Vec{dim,T}
end
Node(x::NTuple{dim,T}) where {dim,T} = Node(Vec{dim,T}(x))
getcoordinates(n::Node) = n.x

"""
A `Cell` is a sub-domain defined by a collection of `Node`s as it's vertices.
"""
abstract type AbstractCell{dim,N,M} end
struct Cell{dim,N,M} <: AbstractCell{dim,N,M}
    nodes::NTuple{N,Int}
end
nfaces(c::C) where {C<:AbstractCell} = nfaces(typeof(c))
nfaces(::Type{<:AbstractCell{dim,N,M}}) where {dim,N,M} = M
nnodes(c::C) where {C<:AbstractCell} = nnodes(typeof(c))
nnodes(::Type{<:AbstractCell{dim,N,M}}) where {dim,N,M} = N

# Typealias for commonly used cells
const Line  = Cell{1,2,2}
const Line2D = Cell{2,2,1}
const Line3D = Cell{3,2,0}
const QuadraticLine = Cell{1,3,2}

const Triangle = Cell{2,3,3}
const QuadraticTriangle = Cell{2,6,3}

const Quadrilateral = Cell{2,4,4}
const Quadrilateral3D = Cell{3,4,1}
const QuadraticQuadrilateral = Cell{2,9,4}

const Tetrahedron = Cell{3,4,4}
const QuadraticTetrahedron = Cell{3,10,4}

const Hexahedron = Cell{3,8,6}

"""
A `CellIndex` wraps an Int and corresponds to a cell with that number in the mesh
"""
struct CellIndex
    idx::Int
end

"""
A `FaceIndex` wraps an (Int, Int) and defines a face by pointing to a (cell, face).
"""
struct FaceIndex <: BoundaryIndex
    idx::Tuple{Int,Int} # cell and side
end

"""
A `EdgeIndex` wraps an (Int, Int) and defines a face by pointing to a (cell, edge).
"""
struct EdgeIndex <: BoundaryIndex
    idx::Tuple{Int,Int} # cell and side
end

"""
A `VertexIndex` wraps an (Int, Int) and defines a face by pointing to a (cell, vert).
"""
struct VertexIndex <: BoundaryIndex
    idx::Tuple{Int,Int} # cell and side
end

struct Neighbor{T<:BoundaryIndex}
    neighbor_info::Vector{T}
end

Neighbor(info::T) where T <: BoundaryIndex = Neighbor([info])
Base.zero(::Type{Neighbor}) = Neighbor(BoundaryIndex[])
Base.length(n::Neighbor) = length(n.neighbor_info)
Base.getindex(n::Neighbor,i) = getindex(n.neighbor_info,i)
Base.firstindex(n::Neighbor) = 1
Base.lastindex(n::Neighbor) = length(n.neighbor_info)
Base.:(==)(n1::Neighbor, n2::Neighbor) = n1.neighbor_info == n2.neighbor_info

function Base.:+(n1::Neighbor, n2::Neighbor)
    neighbor_info = [n1.neighbor_info; n2.neighbor_info]
    return Neighbor(neighbor_info)
end

function Base.show(io::IO, ::MIME"text/plain", n::Neighbor)
    if length(n) == 0
        println(io, "No Neighbor")
    elseif length(n) == 1
        println(io, "$(n.neighbor_info[1])")
    else
        println(io, "$(n.neighbor_info...)")
    end
end

face_npoints(::Cell{2,N,M}) where {N,M} = 2
face_npoints(::Cell{3,4,1}) = 4 #not sure how to handle embedded cells e.g. Quadrilateral3D
edge_npoints(::Cell{3,4,1}) = 2 #not sure how to handle embedded cells e.g. Quadrilateral3D
face_npoints(::Cell{3,N,6}) where N = 4
face_npoints(::Cell{3,N,4}) where N = 3
edge_npoints(::Cell{3,N,M}) where {N,M} = 2

getdim(::Cell{dim,N,M}) where {dim,N,M} = dim

struct Topology
    face_neighbor::SparseMatrixCSC{Neighbor,Int}
    corner_neighbor::SparseMatrixCSC{Neighbor,Int}
    edge_neighbor::SparseMatrixCSC{Neighbor,Int}
end

function Topology()
    face_neighbor = spzeros(Neighbor,0,0)
    corner_neighbor = spzeros(Neighbor,0,0)
    edge_neighbor = spzeros(Neighbor,0,0)
    return Topology(face_neighbor,corner_neighbor,edge_neighbor)
end

function Topology(cells::Vector{C}) where C <: AbstractCell
    I_face = Int[]; J_face = Int[]; V_face = Neighbor[]
    I_edge = Int[]; J_edge = Int[]; V_edge = Neighbor[]
    I_corner = Int[]; J_corner = Int[]; V_corner = Neighbor[]
    for (cellid,cell) in enumerate(cells)
        neighbors = findall.(x->x ∈ cell.nodes,getproperty.(cells,:nodes)) 
        for (neighborid,neighbor) in enumerate(neighbors)
            neighbor_cell = cells[neighborid]
            if length(neighbor) == 0
                #not a neighbor
                continue
            elseif length(neighbor) == 1
                #corner neighbor
                _corner_neighbor!(V_corner,I_corner,J_corner,cellid,cell,neighbor,neighborid,neighbor_cell)
            elseif getdim(cell) > 2 && length(neighbor) == edge_npoints(cell) 
                #edge neighbor
                _edge_neighbor!(V_edge,I_edge,J_edge,cellid,cell,neighbor,neighborid,neighbor_cell)
            elseif getdim(cell) > 1 && length(neighbor) == face_npoints(cell)
                #face neighbor
                _face_neighbor!(V_face,I_face,J_face,cellid,cell,neighbor,neighborid,neighbor_cell)
            else
                continue
            end 
        end
    end
    face_neighbor = sparse(I_face,J_face,V_face)
    corner_neighbor = sparse(I_corner,J_corner,V_corner) 
    edge_neighbor = sparse(I_edge,J_edge,V_edge)
    return Topology(face_neighbor,corner_neighbor,edge_neighbor) 
end

function _corner_neighbor!(V_corner, I_corner, J_corner, cellid, cell, neighbor, neighborid, neighbor_cell)
    corner_neighbor = VertexIndex((neighborid, neighbor[1]))
    cell_corner_id = findfirst(x->x==neighbor_cell.nodes[neighbor[1]], cell.nodes)
    push!(V_corner,Neighbor(corner_neighbor))
    push!(I_corner,cellid)
    push!(J_corner,cell_corner_id)
end

function _edge_neighbor!(V_edge, I_edge, J_edge, cellid, cell, neighbor, neighborid, neighbor_cell)
    neighbor_edge = neighbor_cell.nodes[neighbor]
    if getdim(neighbor_cell) < 3
        neighbor_edge_id = findfirst(x->issubset(x,neighbor_edge), faces(neighbor_cell))
        edge_neighbor = FaceIndex((neighborid, neighbor_edge_id))
    else
        neighbor_edge_id = findfirst(x->issubset(x,neighbor_edge), edges(neighbor_cell))
        edge_neighbor = EdgeIndex((neighborid, neighbor_edge_id))
    end
    cell_edge_id = findfirst(x->issubset(x,neighbor_edge),edges(cell))
    push!(V_edge, Neighbor(edge_neighbor))
    push!(I_edge, cellid)
    push!(J_edge, cell_edge_id)
end

function _face_neighbor!(V_face, I_face, J_face, cellid, cell, neighbor, neighborid, neighbor_cell)
    neighbor_face = neighbor_cell.nodes[neighbor]
    if getdim(neighbor_cell) == getdim(cell)
        neighbor_face_id = findfirst(x->issubset(x,neighbor_face), faces(neighbor_cell))
        face_neighbor = FaceIndex((neighborid, neighbor_face_id))
    else
        neighbor_face_id = findfirst(x->issubset(x,neighbor_face), edges(neighbor_cell))
        face_neighbor = EdgeIndex((neighborid, neighbor_face_id))
    end
    cell_face_id = findfirst(x->issubset(x,neighbor_face),faces(cell))
    push!(V_face, Neighbor(face_neighbor))
    push!(I_face, cellid)
    push!(J_face, cell_face_id)
end

abstract type AbstractGrid{dim} end

"""
A `Grid` is a collection of `Cells` and `Node`s which covers the computational domain, together with Sets of cells, nodes and faces.
"""
mutable struct Grid{dim,C<:AbstractCell,T<:Real} <: AbstractGrid{dim}
    cells::Vector{C}
    nodes::Vector{Node{dim,T}}
    # Sets
    cellsets::Dict{String,Set{Int}}
    nodesets::Dict{String,Set{Int}}
    facesets::Dict{String,Set{FaceIndex}} 
    edgesets::Dict{String,Set{EdgeIndex}} 
    vertexsets::Dict{String,Set{VertexIndex}} 
    # Boundary matrix (faces per cell × cell)
    boundary_matrix::SparseMatrixCSC{Bool,Int}
    #topology
    topology::Topology
end

function Grid(cells::Vector{C},
              nodes::Vector{Node{dim,T}};
              cellsets::Dict{String,Set{Int}}=Dict{String,Set{Int}}(),
              nodesets::Dict{String,Set{Int}}=Dict{String,Set{Int}}(),
              facesets::Dict{String,Set{FaceIndex}}=Dict{String,Set{FaceIndex}}(),
              edgesets::Dict{String,Set{EdgeIndex}}=Dict{String,Set{EdgeIndex}}(),
              vertexsets::Dict{String,Set{VertexIndex}}=Dict{String,Set{VertexIndex}}(),
              boundary_matrix::SparseMatrixCSC{Bool,Int}=spzeros(Bool, 0, 0),
              topology::Topology=Topology(cells)) where {dim,C,T}
    return Grid(cells, nodes, cellsets, nodesets, facesets, edgesets, vertexsets, boundary_matrix, topology)
end

##########################
# Grid utility functions #
##########################
@inline getdim(::AbstractGrid{dim}) where {dim} = dim
@inline getcells(grid::AbstractGrid) = grid.cells
@inline getcells(grid::AbstractGrid, v::Union{Int, Vector{Int}}) = grid.cells[v]
@inline getcells(grid::AbstractGrid, set::String) = grid.cells[collect(grid.cellsets[set])]
@inline getncells(grid::AbstractGrid) = length(grid.cells)
@inline getcelltype(grid::AbstractGrid) = eltype(grid.cells)
@inline getcelltype(grid::AbstractGrid, i::Int) = typeof(grid.cells[i])

@inline getnodes(grid::AbstractGrid) = grid.nodes
@inline getnodes(grid::AbstractGrid, v::Union{Int, Vector{Int}}) = grid.nodes[v]
@inline getnodes(grid::AbstractGrid, set::String) = grid.nodes[collect(grid.nodesets[set])]
@inline getnnodes(grid::AbstractGrid) = length(grid.nodes)
@inline nnodes_per_cell(grid::AbstractGrid, i::Int=1) = nnodes(grid.cells[i])

@inline getcellset(grid::AbstractGrid, set::String) = grid.cellsets[set]
@inline getcellsets(grid::AbstractGrid) = grid.cellsets

@inline getnodeset(grid::AbstractGrid, set::String) = grid.nodesets[set]
@inline getnodesets(grid::AbstractGrid) = grid.nodesets

@inline getfaceset(grid::AbstractGrid, set::String) = grid.facesets[set]
@inline getfacesets(grid::AbstractGrid) = grid.facesets

@inline getedgeset(grid::AbstractGrid, set::String) = grid.edgesets[set]
@inline getedgesets(grid::AbstractGrid) = grid.edgesets

@inline getvertexset(grid::AbstractGrid, set::String) = grid.vertexsets[set]
@inline getvertexsets(grid::AbstractGrid) = grid.vertexsets

n_faces_per_cell(grid::Grid) = nfaces(eltype(grid.cells))


@inline function compute_vertex_values(nodes::Vector{Node{dim,T}}, f::Function) where{dim,T}
    map(n -> f(getcoordinates(n)), nodes)
end

@inline function compute_vertex_values(grid::AbstractGrid, f::Function)
    compute_vertex_values(getnodes(grid), f::Function)
end

@inline function compute_vertex_values(grid::AbstractGrid, v::Vector{Int}, f::Function)
    compute_vertex_values(getnodes(grid, v), f::Function)
end

@inline function compute_vertex_values(grid::AbstractGrid, set::String, f::Function)
    compute_vertex_values(getnodes(grid, set), f::Function)
end

# Transformations

function transform!(g::AbstractGrid, f::Function)
    c = similar(g.nodes)
    for i in 1:length(c)
        c[i] = Node(f(g.nodes[i].x))
    end
    copyto!(g.nodes, c)
    g
end

# Sets

_check_setname(dict, name) = haskey(dict, name) && throw(ArgumentError("there already exists a set with the name: $name"))
_warn_emptyset(set) = length(set) == 0 && @warn("no entities added to set")

function addcellset!(grid::AbstractGrid, name::String, cellid::Union{Set{Int},Vector{Int}})
    _check_setname(grid.cellsets,  name)
    cells = Set(cellid)
    _warn_emptyset(cells)
    grid.cellsets[name] = cells
    grid
end

function addcellset!(grid::AbstractGrid, name::String, f::Function; all::Bool=true)
    _check_setname(grid.cellsets,  name)
    cells = Set{Int}()
    for (i, cell) in enumerate(getcells(grid))
        pass = all
        for node_idx in cell.nodes
            node = grid.nodes[node_idx]
            v = f(node.x)
            all ? (!v && (pass = false; break)) : (v && (pass = true; break))
        end
        pass && push!(cells, i)
    end
    _warn_emptyset(cells)
    grid.cellsets[name] = cells
    grid
end

addfaceset!(grid::Grid, name::String, set::Union{Set{FaceIndex},Vector{FaceIndex}}) = 
    _addset!(grid, name, set, grid.facesets)
addedgeset!(grid::Grid, name::String, set::Union{Set{EdgeIndex},Vector{EdgeIndex}}) = 
    _addset!(grid, name, set, grid.edgesets)
addvertexset!(grid::Grid, name::String, set::Union{Set{VertexIndex},Vector{VertexIndex}}) = 
    _addset!(grid, name, set, grid.vertexsets)
function _addset!(grid::AbstractGrid, name::String, _set, dict::Dict)
    _check_setname(dict, name)
    set = Set(_set)
    _warn_emptyset(set)
    dict[name] = set
    grid
end

addfaceset!(grid::AbstractGrid, name::String, f::Function; all::Bool=true) = 
    _addset!(grid, name, f, Ferrite.faces, grid.facesets, FaceIndex; all=all)
addedgeset!(grid::AbstractGrid, name::String, f::Function; all::Bool=true) = 
    _addset!(grid, name, f, Ferrite.edges, grid.edgesets, EdgeIndex; all=all)
addvertexset!(grid::AbstractGrid, name::String, f::Function; all::Bool=true) = 
    _addset!(grid, name, f, Ferrite.vertices, grid.vertexsets, VertexIndex; all=all)
function _addset!(grid::AbstractGrid, name::String, f::Function, _ftype::Function, dict::Dict, _indextype::Type; all::Bool=true)
    _check_setname(dict, name)
    _set = Set{_indextype}()
    for (cell_idx, cell) in enumerate(getcells(grid))
        for (face_idx, face) in enumerate(_ftype(cell))
            pass = all
            for node_idx in face
                v = f(grid.nodes[node_idx].x)
                all ? (!v && (pass = false; break)) : (v && (pass = true; break))
            end
            pass && push!(_set, _indextype(cell_idx, face_idx))
        end
    end
    _warn_emptyset(_set)
    dict[name] = _set
    grid
end

function addnodeset!(grid::AbstractGrid, name::String, nodeid::Union{Vector{Int},Set{Int}})
    _check_setname(grid.nodesets, name)
    grid.nodesets[name] = Set(nodeid)
    _warn_emptyset(grid.nodesets[name])
    grid
end

function addnodeset!(grid::AbstractGrid, name::String, f::Function)
    _check_setname(grid.nodesets, name)
    nodes = Set{Int}()
    for (i, n) in enumerate(getnodes(grid))
        f(n.x) && push!(nodes, i)
    end
    grid.nodesets[name] = nodes
    _warn_emptyset(grid.nodesets[name])
    grid
end

"""
    getcoordinates!(x::Vector, grid::Grid, cell::Int)
Update the coordinate vector `x` for cell number `cell`.
"""
@inline function getcoordinates!(x::Vector{Vec{dim,T}}, grid::AbstractGrid, cell::Int) where {dim,T}
    #@assert length(x) == N
    @inbounds for i in 1:length(x)
        x[i] = grid.nodes[grid.cells[cell].nodes[i]].x
    end
end
@inline getcoordinates!(x::Vector{Vec{dim,T}}, grid::AbstractGrid, cell::CellIndex) where {dim, T} = getcoordinates!(x, grid, cell.idx)
@inline getcoordinates!(x::Vector{Vec{dim,T}}, grid::AbstractGrid, face::FaceIndex) where {dim, T} = getcoordinates!(x, grid, face.idx[1])

"""
    getcoordinates(grid::Grid, cell)
Return a vector with the coordinates of the vertices of cell number `cell`.
"""
@inline function getcoordinates(grid::AbstractGrid, cell::Int)
    # TODO pretty ugly, worth it?
    dim = typeof(grid.cells[cell]).parameters[1]
    T = typeof(grid).parameters[3]
    nodeidx = grid.cells[cell].nodes
    return [grid.nodes[i].x for i in nodeidx]::Vector{Vec{dim,T}}
end
@inline getcoordinates(grid::AbstractGrid, cell::CellIndex) = getcoordinates(grid, cell.idx)
@inline getcoordinates(grid::AbstractGrid, face::FaceIndex) = getcoordinates(grid, face.idx[1])

# Iterate over cell vector
function Base.iterate(c::Vector{Cell{dim,N}}, state = 1) where {dim, N}
    if state > length(c)
        return nothing
    else
        return (CellIndex(state), state + 1)
    end
end

function Base.show(io::IO, ::MIME"text/plain", grid::Grid)
    print(io, "$(typeof(grid)) with $(getncells(grid)) ")
    typestrs = sort!(collect(Set(celltypes[typeof(x)] for x in grid.cells)))
    str = join(io, typestrs, '/')
    print(io, " cells and $(getnnodes(grid)) nodes")
end

const celltypes = Dict{DataType, String}(Cell{1,2,2}  => "Line",
                                         Cell{2,2,2}  => "2D-Line",
                                         Cell{3,2,0}  => "3D-Line",
                                         Cell{1,3,2}  => "QuadraticLine",
                                         Cell{2,3,3}  => "Triangle",
                                         Cell{2,6,3}  => "QuadraticTriangle",
                                         Cell{2,4,4}  => "Quadrilateral",
                                         Cell{3,4,1}  => "3D-Quadrilateral",
                                         Cell{2,9,4}  => "QuadraticQuadrilateral",
                                         Cell{3,4,4}  => "Tetrahedron",
                                         Cell{3,10,4} => "QuadraticTetrahedron",
                                         Cell{3,8,6}  => "Hexahedron",
                                         Cell{3,20,6} => "Cell{3,20,6}")

# Functions to uniquely identify vertices, edges and faces, used when distributing
# dofs over a mesh. For this we can ignore the nodes on edged, faces and inside cells,
# we only need to use the nodes that are vertices.
# 1D: vertices
faces(c::Union{Line,QuadraticLine}) = (c.nodes[1], c.nodes[2])
vertices(c::Union{Line,Line2D,Line3D,QuadraticLine}) = (c.nodes[1], c.nodes[2])
# 2D: vertices, faces
faces(c::Line2D) = ((c.nodes[1],c.nodes[2]),)
vertices(c::Union{Triangle,QuadraticTriangle}) = (c.nodes[1], c.nodes[2], c.nodes[3])
faces(c::Union{Triangle,QuadraticTriangle}) = ((c.nodes[1],c.nodes[2]), (c.nodes[2],c.nodes[3]), (c.nodes[3],c.nodes[1]))
vertices(c::Union{Quadrilateral,Quadrilateral3D,QuadraticQuadrilateral}) = (c.nodes[1], c.nodes[2], c.nodes[3], c.nodes[4])
faces(c::Union{Quadrilateral,QuadraticQuadrilateral}) = ((c.nodes[1],c.nodes[2]), (c.nodes[2],c.nodes[3]), (c.nodes[3],c.nodes[4]), (c.nodes[4],c.nodes[1]))
# 3D: vertices, edges, faces
edges(c::Line3D) = ((c.nodes[1],c.nodes[2]),)
vertices(c::Union{Tetrahedron,QuadraticTetrahedron}) = (c.nodes[1], c.nodes[2], c.nodes[3], c.nodes[4])
edges(c::Union{Tetrahedron,QuadraticTetrahedron}) = ((c.nodes[1],c.nodes[2]), (c.nodes[2],c.nodes[3]), (c.nodes[3],c.nodes[1]), (c.nodes[1],c.nodes[4]), (c.nodes[2],c.nodes[4]), (c.nodes[3],c.nodes[4]))
faces(c::Union{Tetrahedron,QuadraticTetrahedron}) = ((c.nodes[1],c.nodes[2],c.nodes[3]), (c.nodes[1],c.nodes[2],c.nodes[4]), (c.nodes[2],c.nodes[3],c.nodes[4]), (c.nodes[1],c.nodes[4],c.nodes[3]))
vertices(c::Union{Hexahedron,Cell{3,20,6}}) = (c.nodes[1], c.nodes[2], c.nodes[3], c.nodes[4], c.nodes[5], c.nodes[6], c.nodes[7], c.nodes[8])
edges(c::Union{Hexahedron,Cell{3,20,6}}) = ((c.nodes[1],c.nodes[2]), (c.nodes[2],c.nodes[3]), (c.nodes[3],c.nodes[4]), (c.nodes[4],c.nodes[1]), (c.nodes[5],c.nodes[6]), (c.nodes[6],c.nodes[7]), (c.nodes[7],c.nodes[8]), (c.nodes[8],c.nodes[5]), (c.nodes[1],c.nodes[5]), (c.nodes[2],c.nodes[6]), (c.nodes[3],c.nodes[7]), (c.nodes[4],c.nodes[8]))
faces(c::Union{Hexahedron,Cell{3,20,6}}) = ((c.nodes[1],c.nodes[4],c.nodes[3],c.nodes[2]), (c.nodes[1],c.nodes[2],c.nodes[6],c.nodes[5]), (c.nodes[2],c.nodes[3],c.nodes[7],c.nodes[6]), (c.nodes[3],c.nodes[4],c.nodes[8],c.nodes[7]), (c.nodes[1],c.nodes[5],c.nodes[8],c.nodes[4]), (c.nodes[5],c.nodes[6],c.nodes[7],c.nodes[8]))
edges(c::Union{Quadrilateral3D}) = ((c.nodes[1],c.nodes[2]), (c.nodes[2],c.nodes[3]), (c.nodes[3],c.nodes[4]), (c.nodes[4],c.nodes[1]))
faces(c::Union{Quadrilateral3D}) = ((c.nodes[1],c.nodes[2],c.nodes[3],c.nodes[4]),)

# random stuff
default_interpolation(::Union{Type{Line},Type{Line2D},Type{Line3D}}) = Lagrange{1,RefCube,1}()
default_interpolation(::Type{QuadraticLine}) = Lagrange{1,RefCube,2}()
default_interpolation(::Type{Triangle}) = Lagrange{2,RefTetrahedron,1}()
default_interpolation(::Type{QuadraticTriangle}) = Lagrange{2,RefTetrahedron,2}()
default_interpolation(::Union{Type{Quadrilateral},Type{Quadrilateral3D}}) = Lagrange{2,RefCube,1}()
default_interpolation(::Type{QuadraticQuadrilateral}) = Lagrange{2,RefCube,2}()
default_interpolation(::Type{Tetrahedron}) = Lagrange{3,RefTetrahedron,1}()
default_interpolation(::Type{QuadraticTetrahedron}) = Lagrange{3,RefTetrahedron,2}()
default_interpolation(::Type{Hexahedron}) = Lagrange{3,RefCube,1}()
default_interpolation(::Type{Cell{3,20,6}}) = Serendipity{3,RefCube,2}()

boundaryfunction(::Type{FaceIndex}) = Ferrite.faces
boundaryfunction(::Type{EdgeIndex}) = Ferrite.edges
boundaryfunction(::Type{VertexIndex}) = Ferrite.vertices

for INDEX in (:VertexIndex, :EdgeIndex, :FaceIndex)
    @eval begin  
        #Constructor
        ($INDEX)(a::Int, b::Int) = ($INDEX)((a,b))

        Base.getindex(I::($INDEX), i::Int) = I.idx[i]
        
        #To be able to do a,b = faceidx
        Base.iterate(I::($INDEX), state::Int=1) = (state==3) ?  nothing : (I[state], state+1)

        #For (cellid, faceidx) in faceset
        Base.in(v::Tuple{Int, Int}, s::Set{$INDEX}) = in($INDEX(v), s)
    end
end
