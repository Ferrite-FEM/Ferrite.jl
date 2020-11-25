#########################
# Main types for meshes #
#########################
"""
    Node{dim, T}

A `Node` is a point in space.

# Fields
- `x::Vec{dim,T}`: stores the coordinates
"""
struct Node{dim,T}
    x::Vec{dim,T}
end
Node(x::NTuple{dim,T}) where {dim,T} = Node(Vec{dim,T}(x))
getcoordinates(n::Node) = n.x


abstract type AbstractCell{dim,N,M} end
"""
    Cell{dim,N,M} <: AbstractCell{dim,N,M}
A `Cell` is a sub-domain defined by a collection of `Node`s as it's vertices.
However, a `cell` is not defined by the nodes but rather by the node ids

# Fields
- `nodes::Ntuple{N,Int}`: N-tuple that stores the node ids
"""
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
const QuadraticHexahedron = Cell{3,20,6} # Function interpolation for this doesn't exist in JuAFEM yet

"""
A `CellIndex` wraps an Int and corresponds to a cell with that number in the mesh
"""
struct CellIndex
    idx::Int
end

"""
A `FaceIndex` wraps an (Int, Int) and defines a face by pointing to a (cell, face).
"""
struct FaceIndex
    idx::Tuple{Int,Int} # cell and side
end

abstract type AbstractGrid end

"""
    Grid{dim, C<:AbstractCell, T<:Real} <: AbstractGrid}

A `Grid` is a collection of `Cells` and `Node`s which covers the computational domain, together with Sets of cells, nodes and faces.

# Fields
- `cells::Vector{C}`: stores all cells of the grid
- `nodes::Vector{Node{dim,T}}`: stores the `dim` dimensional nodes of the grid
- `cellsets::Dict{String,Set{Int}}`: maps a `String` key to a `Set` of cell ids
- `nodesets::Dict{String,Set{Int}}`: maps a `String` key to a `Set` of node ids
- `facesets::Dict{String,Set{Tuple{Int,Int}}}`: maps a `String` to a `Set` of `Tuple{Int,Int} (global_cell_id, local_face_id)`
- `boundary_matrix::SparseMatrixCSC{Bool,Int}`: optional, only needed by `onboundary` to check if a cell is on the boundary, see, e.g. Helmholtz example
"""
mutable struct Grid{dim,C<:AbstractCell,T<:Real} <: AbstractGrid
    cells::Vector{C}
    nodes::Vector{Node{dim,T}}
    # Sets
    cellsets::Dict{String,Set{Int}}
    nodesets::Dict{String,Set{Int}}
    facesets::Dict{String,Set{Tuple{Int,Int}}} # TODO: This could be Set{FaceIndex} which could result in nicer use later
    # Boundary matrix (faces per cell × cell)
    boundary_matrix::SparseMatrixCSC{Bool,Int}
end

function Grid(cells::Vector{C},
              nodes::Vector{Node{dim,T}};
              cellsets::Dict{String,Set{Int}}=Dict{String,Set{Int}}(),
              nodesets::Dict{String,Set{Int}}=Dict{String,Set{Int}}(),
              facesets::Dict{String,Set{Tuple{Int,Int}}}=Dict{String,Set{Tuple{Int,Int}}}(),
              boundary_matrix::SparseMatrixCSC{Bool,Int}=spzeros(Bool, 0, 0)) where {dim,C,T}
    return Grid(cells, nodes, cellsets, nodesets, facesets, boundary_matrix)
end

##########################
# Grid utility functions #
##########################
@inline getdim(grid::Grid{dim}) where {dim} = dim
"""
    getcells(grid::AbstractGrid) 
    getcells(grid::AbstractGrid, v::Union{Int,Vector{Int}} 
    getcells(grid::AbstractGrid, set::String)

Returns either all `cells::Vector{C<:AbstractCell}` of a `grid` or a subset based on an `Int`, `Vector{Int}` or `String`.
Whereas the last option tries to call a `cellset` of the `grid`.
"""
@inline getcells(grid::AbstractGrid) = grid.cells
@inline getcells(grid::AbstractGrid, v::Union{Int, Vector{Int}}) = grid.cells[v]
@inline getcells(grid::AbstractGrid, set::String) = grid.cells[collect(grid.cellsets[set])]
"Returns and `Int` corresponding to how many cells are in the `grid`."
@inline getncells(grid::AbstractGrid) = length(grid.cells)
"Returns the celltype of the `grid`."
@inline getcelltype(grid::AbstractGrid) = eltype(grid.cells)

"""
    getnodes(grid::AbstractGrid) 
    getnodes(grid::AbstractGrid, v::Union{Int,Vector{Int}}
    getnodes(grid::AbstractGrid, set::String)

Returns either all `nodes::Vector{Node{dim,T}}` of a `grid` or a subset based on an `Int`, `Vector{Int}` or `String`.
The last option tries to call a `nodeset` of the `grid`.
"""
@inline getnodes(grid::AbstractGrid) = grid.nodes
@inline getnodes(grid::AbstractGrid, v::Union{Int, Vector{Int}}) = grid.nodes[v]
@inline getnodes(grid::AbstractGrid, set::String) = grid.nodes[collect(grid.nodesets[set])]
"returns an `Int` corresponding to how many nodes are in the `grid`"
@inline getnnodes(grid::AbstractGrid) = length(grid.nodes)
"returns an `Int` of how many nodes are in one `cell`"
@inline nnodes_per_cell(grid::AbstractGrid, i::Int=1) = nnodes(grid.cells[i])

"Accesses the cellset which is mapped to the key `set::String`"
@inline getcellset(grid::AbstractGrid, set::String) = grid.cellsets[set]
"Returns all cellsets of the `grid`"
@inline getcellsets(grid::AbstractGrid) = grid.cellsets

"Accesses the nodeset which is mapped to the key `set::String`"
@inline getnodeset(grid::AbstractGrid, set::String) = grid.nodesets[set]
"Returns all nodesets of the `grid`"
@inline getnodesets(grid::AbstractGrid) = grid.nodesets

"Accesses the faceset which is mapped to the key `set::String`"
@inline getfaceset(grid::AbstractGrid, set::String) = grid.facesets[set]
"Returns all facesets of the `grid`"
@inline getfacesets(grid::AbstractGrid) = grid.facesets

n_faces_per_cell(grid::Grid) = nfaces(eltype(grid.cells))

"""
    function compute_vertex_values(grid::AbstractGrid, f::Function)
    function compute_vertex_values(grid::AbstractGrid, v::Vector{Int}, f::Function)    
    function compute_vertex_values(grid::AbstractGrid, set::String, f::Function)

Given a `grid` and some function `f`, `compute_vertex_values` computes all nodal values,
 i.e. values at the nodes,  of the function `f`. 
The function implements two dispatches, where only a subset of the grid's node is used.

```julia
    compute_vertex_values(grid, x -> sin(x[1]) + cos([2]))
    compute_vertex_values(grid, [9, 6, 3], x -> sin(x[1]) + cos([2])) #compute function values at nodes with id 9,6,3
    compute_vertex_values(grid, "right", x -> sin(x[1]) + cos([2])) #compute function values at nodes belonging to nodeset right
```

"""
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
"""
    transform!(grid::Abstractgrid, f::Function)

Transform all nodes of the `grid` based on some transformation function `f`.
"""
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

"""
    addcellset!(grid::AbstractGrid, name::String, cellid::Union{Set{Int}, Vector{Int}})
    addcellset!(grid::AbstractGrid, name::String, f::function; all::Bool=true)

Adds a `cellset::Dict{String,Set{Int}}` to the `grid` with key `name`.
Cellsets can be used to specify a boundary for `Dirichlet`, which is needed by the `ConstraintHandler`

```julia
addcellset!(grid, "left", Set((1,3))) #add cells with id 1 and 3 to cellset left
addcellset!(grid, "right", x -> norm(x[1]) < 2.0 ) #add cell to cellset right, if x[1] of each cell's node is smaller than 2.0
```
"""
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

"""
    addfaceset!(grid::AbstractGrid, name::String, faceid::Set{Tuple{Int,Int}})
    addfaceset!(grid::AbstractGrid, name::String, f::Function; all::Bool=true) 

Adds a `faceset::Dict{String, Set{Tuple{Int,Int}}` to the `grid` with key `name`.
A `faceset` maps a `String` key to a `Set` of tuples corresponding to `(global_cell_id, local_face_id)`.
Facesets are used to initialize `Dirichlet` structs, that are needed to specify the boundary for the `ConstraintHandler`.

```julia
addfaceset!(gird, "right", Set(((2,2),(4,2))) #see grid manual example for reference
addfaceset!(grid, "clamped", x -> norm(x[1]) ≈ 0.0) #see incompressible elasticity example for reference
```
"""
function addfaceset!(grid::AbstractGrid, name::String, faceid::Set{Tuple{Int,Int}})
    _check_setname(grid.facesets, name)
    faceset = Set(faceid)
    _warn_emptyset(faceset)
    grid.facesets[name] = faceset
    grid
end

function addfaceset!(grid::AbstractGrid, name::String, f::Function; all::Bool=true)
    _check_setname(grid.facesets, name)
    faceset = Set{Tuple{Int,Int}}()
    for (cell_idx, cell) in enumerate(getcells(grid))
        for (face_idx, face) in enumerate(faces(cell))
            pass = all
            for node_idx in face
                v = f(grid.nodes[node_idx].x)
                all ? (!v && (pass = false; break)) : (v && (pass = true; break))
            end
            pass && push!(faceset, (cell_idx, face_idx))
        end
    end
    _warn_emptyset(faceset)
    grid.facesets[name] = faceset
    grid
end

"""
    addnodeset!(grid::AbstractGrid, name::String, nodeid::Union{Vector{Int},Set{Int}})
    addnodeset!(grid::AbstractGrid, name::String, f::Function)    

Adds a `nodeset::Dict{String, Set{Int}}` to the `grid` with key `name`. Has the same interface as `addcellset`. 
However, instead of mapping a cell id to the `String` key, a set of node ids is returned.
"""
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
    getcoordinates!(x::Vector{Vec{dim,T}}, grid::AbstractGrid, cell::Int)
Fills the vector `x` with the coordinates of a cell, defined by its cell id.
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
    getcoordinates(grid::AbstractGrid, cell)
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
                                         Cell{3,20,6} => "QuadraticHexahedron")

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
vertices(c::Union{Hexahedron,QuadraticHexahedron}) = (c.nodes[1], c.nodes[2], c.nodes[3], c.nodes[4], c.nodes[5], c.nodes[6], c.nodes[7], c.nodes[8])
edges(c::Union{Hexahedron,QuadraticHexahedron}) = ((c.nodes[1],c.nodes[2]), (c.nodes[2],c.nodes[3]), (c.nodes[3],c.nodes[4]), (c.nodes[4],c.nodes[1]), (c.nodes[1],c.nodes[5]), (c.nodes[2],c.nodes[6]), (c.nodes[3],c.nodes[7]), (c.nodes[4],c.nodes[8]), (c.nodes[5],c.nodes[6]), (c.nodes[6],c.nodes[7]), (c.nodes[7],c.nodes[8]), (c.nodes[8],c.nodes[5]))
faces(c::Union{Hexahedron,QuadraticHexahedron}) = ((c.nodes[1],c.nodes[4],c.nodes[3],c.nodes[2]), (c.nodes[1],c.nodes[2],c.nodes[6],c.nodes[5]), (c.nodes[2],c.nodes[3],c.nodes[7],c.nodes[6]), (c.nodes[3],c.nodes[4],c.nodes[8],c.nodes[7]), (c.nodes[1],c.nodes[5],c.nodes[8],c.nodes[4]), (c.nodes[5],c.nodes[6],c.nodes[7],c.nodes[8]))
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
default_interpolation(::Type{QuadraticHexahedron}) = Lagrange{3,RefCube,2}()
