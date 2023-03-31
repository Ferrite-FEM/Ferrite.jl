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
get_node_coordinate(n::Node) = n.x

"""
    Ferrite.get_coordinate_eltype(::Node)

Get the data type of the components of the nodes coordinate.
"""
get_coordinate_eltype(::Node{dim,T}) where {dim,T} = T

"""
    Ferrite.get_coordinate_type(::Node)

Get the data type of the node's coordinate.
"""
get_coordinate_type(::Node{dim,T}) where {dim,T} = Vec{dim,T}


"""
    Cell{dim,N,M} <: AbstractCell{dim,N,M}

A `Cell` is a subdomain defined by a collection of `Node`s.
The parameter `dim` refers here to the geometrical/ambient dimension, i.e. the dimension of the `nodes` in the grid and **not** the topological dimension of the cell (i.e. the dimension of the reference element obtained by default_interpolation).
A `Cell` has `N` nodes and `M` faces.
Note that a `Cell` is not defined geometrically by node coordinates, but rather topologically by node indices into the node vector of some grid.

# Fields
- `nodes::Ntuple{N,Int}`: N-tuple that stores the node ids. The ordering defines a cell's and its subentities' orientations.
"""
struct Cell{dim,N,M} <: AbstractCell{dim,N,M}
    nodes::NTuple{N,Int}
end

# Typealias for commonly used cells
const implemented_celltypes = (
    (const Line  = Cell{1,2,2}),
    (const Line2D = Cell{2,2,1}),
    (const Line3D = Cell{3,2,0}),
    (const QuadraticLine = Cell{1,3,2}),
    
    (const Triangle = Cell{2,3,3}),
    (const QuadraticTriangle = Cell{2,6,3}),
    
    (const Quadrilateral = Cell{2,4,4}),
    (const Quadrilateral3D = Cell{3,4,1}),
    (const QuadraticQuadrilateral = Cell{2,9,4}),
    
    (const Tetrahedron = Cell{3,4,4}),
    (const QuadraticTetrahedron = Cell{3,10,4}),
    
    (const Hexahedron = Cell{3,8,6}),
    (Cell{2,20,6}),

    (const Wedge = Cell{3,6,5})
)

"""
    face_npoints(::AbstractCell{dim,N,M)
Specifies for each subtype of AbstractCell how many nodes form a face
"""
face_npoints(::Cell{2,N,M}) where {N,M} = 2
face_npoints(::Cell{3,4,1}) = 4 #not sure how to handle embedded cells e.g. Quadrilateral3D
"""
    edge_npoints(::AbstractCell{dim,N,M)
Specifies for each subtype of AbstractCell how many nodes form an edge
"""
edge_npoints(::Cell{3,4,1}) = 2 #not sure how to handle embedded cells e.g. Quadrilateral3D
face_npoints(::Cell{3,N,6}) where N = 4
face_npoints(::Cell{3,N,4}) where N = 3
edge_npoints(::Cell{3,N,M}) where {N,M} = 2

getdim(::Cell{dim}) where dim = dim

getnodeidxs(cell::Cell) = cell.nodes

"""
    Grid{dim, C<:AbstractCell, T<:Real} <: AbstractGrid}

A `Grid` is a collection of `Cells` and `Node`s which covers the computational domain, together with Sets of cells, nodes and faces.
There are multiple helper structures to apply boundary conditions or define subdomains. They are gathered in the `cellsets`, `nodesets`,
`facesets`, `edgesets` and `vertexsets`. 

# Fields
- `cells::Vector{C}`: stores all cells of the grid
- `nodes::Vector{Node{dim,T}}`: stores the `dim` dimensional nodes of the grid
- `cellsets::Dict{String,Set{Int}}`: maps a `String` key to a `Set` of cell ids
- `nodesets::Dict{String,Set{Int}}`: maps a `String` key to a `Set` of global node ids
- `facesets::Dict{String,Set{FaceIndex}}`: maps a `String` to a `Set` of `Set{FaceIndex} (global_cell_id, local_face_id)`
- `edgesets::Dict{String,Set{EdgeIndex}}`: maps a `String` to a `Set` of `Set{EdgeIndex} (global_cell_id, local_edge_id` 
- `vertexsets::Dict{String,Set{VertexIndex}}`: maps a `String` key to a `Set` of local vertex ids
- `boundary_matrix::SparseMatrixCSC{Bool,Int}`: optional, only needed by `onboundary` to check if a cell is on the boundary, see, e.g. Helmholtz example
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
end

function Grid(cells::Vector{C},
              nodes::Vector{Node{dim,T}};
              cellsets::Dict{String,Set{Int}}=Dict{String,Set{Int}}(),
              nodesets::Dict{String,Set{Int}}=Dict{String,Set{Int}}(),
              facesets::Dict{String,Set{FaceIndex}}=Dict{String,Set{FaceIndex}}(),
              edgesets::Dict{String,Set{EdgeIndex}}=Dict{String,Set{EdgeIndex}}(),
              vertexsets::Dict{String,Set{VertexIndex}}=Dict{String,Set{VertexIndex}}(),
              boundary_matrix::SparseMatrixCSC{Bool,Int}=spzeros(Bool, 0, 0)) where {dim,C,T}
    return Grid(cells, nodes, cellsets, nodesets, facesets, edgesets, vertexsets, boundary_matrix)
end

##########################
# Grid utility functions #
##########################
"""
    toglobal(grid::AbstractGrid, vertexidx::VertexIndex) -> Int
    toglobal(grid::AbstractGrid, vertexidx::Vector{VertexIndex}) -> Vector{Int}
This function takes the local vertex representation (a `VertexIndex`) and looks up the unique global id (an `Int`).
"""
toglobal(grid::AbstractGrid,vertexidx::VertexIndex) = vertices(getcells(grid,vertexidx[1]))[vertexidx[2]]
toglobal(grid::AbstractGrid,vertexidx::Vector{VertexIndex}) = unique(toglobal.((grid,),vertexidx))

@inline getcells(grid::Grid) = grid.cells

@inline getnodes(grid::AbstractGrid) = grid.nodes


"""
    getcellset(grid::Grid, setname::String)

Returns all cells as cellid in a `Set` of a given `setname`.
"""
@inline getcellset(grid::Grid, setname::String) = grid.cellsets[setname]
"""
    getcellsets(grid::Grid)

Returns all cellsets of the `grid`.
"""
@inline getcellsets(grid::Grid) = grid.cellsets

"""
    getnodeset(grid::Grid, setname::String)

Returns all nodes as nodeid in a `Set` of a given `setname`.
"""
@inline getnodeset(grid::Grid, setname::String) = grid.nodesets[setname]
"""
    getnodesets(grid::Grid)

Returns all nodesets of the `grid`.
"""
@inline getnodesets(grid::Grid) = grid.nodesets

"""
    getfaceset(grid::Grid, setname::String)

Returns all faces as `FaceIndex` in a `Set` of a given `setname`.
"""
@inline getfaceset(grid::Grid, setname::String) = grid.facesets[setname]
"""
    getfacesets(grid::Grid)

Returns all facesets of the `grid`.
"""
@inline getfacesets(grid::Grid) = grid.facesets

"""
    getedgeset(grid::Grid, setname::String)

Returns all edges as `EdgeIndex` in a `Set` of a given `setname`.
"""
@inline getedgeset(grid::Grid, setname::String) = grid.edgesets[setname]
"""
    getedgesets(grid::Grid)

Returns all edge sets of the grid.
"""
@inline getedgesets(grid::Grid) = grid.edgesets

"""
    getedgeset(grid::Grid, setname::String)

Returns all vertices as `VertexIndex` in a `Set` of a given `setname`.
"""
@inline getvertexset(grid::Grid, setname::String) = grid.vertexsets[setname]
"""
    getvertexsets(grid::Grid)

Returns all vertex sets of the grid.
"""
@inline getvertexsets(grid::Grid) = grid.vertexsets

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
    map(n -> f(get_node_coordinate(n)), nodes)
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
    transform!(grid::Grid, f::Function)

Transform all nodes of the `grid` based on some transformation function `f`.
"""
function transform!(g::Grid, f::Function)
    c = similar(g.nodes)
    for i in 1:length(c)
        c[i] = Node(f(g.nodes[i].x))
    end
    copyto!(g.nodes, c)
    g
end

# Sets

_check_setname(dict, name) = haskey(dict, name) && throw(ArgumentError("there already exists a set with the name: $name"))
_warn_emptyset(set, name) = length(set) == 0 && @warn("no entities added to the set with name: $name")

"""
    addcellset!(grid::AbstractGrid, name::String, cellid::Union{Set{Int}, Vector{Int}})
    addcellset!(grid::AbstractGrid, name::String, f::function; all::Bool=true)

Adds a cellset to the grid with key `name`.
Cellsets are typically used to define subdomains of the problem, e.g. two materials in the computational domain.
The `MixedDofHandler` can construct different fields which live not on the whole domain, but rather on a cellset.
`all=true` implies that `f(x)` must return `true` for all nodal coordinates `x` in the cell if the cell
should be added to the set, otherwise it suffices that `f(x)` returns `true` for one node. 

```julia
addcellset!(grid, "left", Set((1,3))) #add cells with id 1 and 3 to cellset left
addcellset!(grid, "right", x -> norm(x[1]) < 2.0 ) #add cell to cellset right, if x[1] of each cell's node is smaller than 2.0
```
"""
function addcellset!(grid::Grid, name::String, cellid::Union{Set{Int},Vector{Int}})
    _check_setname(grid.cellsets,  name)
    cells = Set(cellid)
    _warn_emptyset(cells, name)
    grid.cellsets[name] = cells
    grid
end

function addcellset!(grid::Grid, name::String, f::Function; all::Bool=true)
    _check_setname(grid.cellsets, name)
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
    _warn_emptyset(cells, name)
    grid.cellsets[name] = cells
    grid
end

"""
    addfaceset!(grid::Grid, name::String, faceid::Union{Set{FaceIndex},Vector{FaceIndex}})
    addfaceset!(grid::Grid, name::String, f::Function; all::Bool=true) 

Adds a faceset to the grid with key `name`.
A faceset maps a `String` key to a `Set` of tuples corresponding to `(global_cell_id, local_face_id)`.
Facesets are used to initialize `Dirichlet` structs, that are needed to specify the boundary for the `ConstraintHandler`.
`all=true` implies that `f(x)` must return `true` for all nodal coordinates `x` on the face if the face
should be added to the set, otherwise it suffices that `f(x)` returns `true` for one node. 

```julia
addfaceset!(grid, "right", Set(((2,2),(4,2))) #see grid manual example for reference
addfaceset!(grid, "clamped", x -> norm(x[1]) ≈ 0.0) #see incompressible elasticity example for reference
```
"""
addfaceset!(grid::Grid, name::String, set::Union{Set{FaceIndex},Vector{FaceIndex}}) = 
    _addset!(grid, name, set, grid.facesets)
addedgeset!(grid::Grid, name::String, set::Union{Set{EdgeIndex},Vector{EdgeIndex}}) = 
    _addset!(grid, name, set, grid.edgesets)
addvertexset!(grid::Grid, name::String, set::Union{Set{VertexIndex},Vector{VertexIndex}}) = 
    _addset!(grid, name, set, grid.vertexsets)
function _addset!(grid::Grid, name::String, _set, dict::Dict)
    _check_setname(dict, name)
    set = Set(_set)
    _warn_emptyset(set, name)
    dict[name] = set
    grid
end

addfaceset!(grid::Grid, name::String, f::Function; all::Bool=true) = 
    _addset!(grid, name, f, Ferrite.faces, grid.facesets, FaceIndex; all=all)
addedgeset!(grid::Grid, name::String, f::Function; all::Bool=true) = 
    _addset!(grid, name, f, Ferrite.edges, grid.edgesets, EdgeIndex; all=all)
addvertexset!(grid::Grid, name::String, f::Function; all::Bool=true) = 
    _addset!(grid, name, f, Ferrite.vertices, grid.vertexsets, VertexIndex; all=all)
function _addset!(grid::Grid, name::String, f::Function, _ftype::Function, dict::Dict, _indextype::Type; all::Bool=true)
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
    _warn_emptyset(_set, name)
    dict[name] = _set
    grid
end

"""
    addnodeset!(grid::AbstractGrid, name::String, nodeid::Union{Vector{Int},Set{Int}})
    addnodeset!(grid::AbstractGrid, name::String, f::Function)    

Adds a `nodeset::Dict{String, Set{Int}}` to the `grid` with key `name`. Has the same interface as `addcellset`. 
However, instead of mapping a cell id to the `String` key, a set of node ids is returned.
"""
function addnodeset!(grid::Grid, name::String, nodeid::Union{Vector{Int},Set{Int}})
    _check_setname(grid.nodesets, name)
    grid.nodesets[name] = Set(nodeid)
    _warn_emptyset(grid.nodesets[name], name)
    grid
end

function addnodeset!(grid::Grid, name::String, f::Function)
    _check_setname(grid.nodesets, name)
    nodes = Set{Int}()
    for (i, n) in enumerate(getnodes(grid))
        f(n.x) && push!(nodes, i)
    end
    grid.nodesets[name] = nodes
    _warn_emptyset(grid.nodesets[name], name)
    grid
end

function Base.show(io::IO, ::MIME"text/plain", grid::Grid)
    print(io, "$(typeof(grid)) with $(getncells(grid)) ")
    if isconcretetype(eltype(grid.cells))
        typestrs = [repr(eltype(grid.cells))]
    else
        typestrs = sort!(repr.(Set(typeof(x) for x in grid.cells)))
    end
    join(io, typestrs, '/')
    print(io, " cells and $(getnnodes(grid)) nodes")
end

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
faces(c::Union{Tetrahedron,QuadraticTetrahedron}) = ((c.nodes[1],c.nodes[3],c.nodes[2]), (c.nodes[1],c.nodes[2],c.nodes[4]), (c.nodes[2],c.nodes[3],c.nodes[4]), (c.nodes[1],c.nodes[4],c.nodes[3]))
vertices(c::Union{Hexahedron,Cell{3,20,6}}) = (c.nodes[1], c.nodes[2], c.nodes[3], c.nodes[4], c.nodes[5], c.nodes[6], c.nodes[7], c.nodes[8])
edges(c::Union{Hexahedron,Cell{3,20,6}}) = ((c.nodes[1],c.nodes[2]), (c.nodes[2],c.nodes[3]), (c.nodes[3],c.nodes[4]), (c.nodes[4],c.nodes[1]), (c.nodes[5],c.nodes[6]), (c.nodes[6],c.nodes[7]), (c.nodes[7],c.nodes[8]), (c.nodes[8],c.nodes[5]), (c.nodes[1],c.nodes[5]), (c.nodes[2],c.nodes[6]), (c.nodes[3],c.nodes[7]), (c.nodes[4],c.nodes[8]))
faces(c::Union{Hexahedron,Cell{3,20,6}}) = ((c.nodes[1],c.nodes[4],c.nodes[3],c.nodes[2]), (c.nodes[1],c.nodes[2],c.nodes[6],c.nodes[5]), (c.nodes[2],c.nodes[3],c.nodes[7],c.nodes[6]), (c.nodes[3],c.nodes[4],c.nodes[8],c.nodes[7]), (c.nodes[1],c.nodes[5],c.nodes[8],c.nodes[4]), (c.nodes[5],c.nodes[6],c.nodes[7],c.nodes[8]))
edges(c::Union{Quadrilateral3D}) = ((c.nodes[1],c.nodes[2]), (c.nodes[2],c.nodes[3]), (c.nodes[3],c.nodes[4]), (c.nodes[4],c.nodes[1]))
faces(c::Union{Quadrilateral3D}) = ((c.nodes[1],c.nodes[2],c.nodes[3],c.nodes[4]),)

vertices(c::Wedge) = (c.nodes[1], c.nodes[2], c.nodes[3], c.nodes[4], c.nodes[5], c.nodes[6])
edges(c::Wedge) = ((c.nodes[2],c.nodes[1]), (c.nodes[1],c.nodes[3]), (c.nodes[1],c.nodes[4]), (c.nodes[3],c.nodes[2]), (c.nodes[2],c.nodes[5]), (c.nodes[3],c.nodes[6]), (c.nodes[4],c.nodes[5]), (c.nodes[4],c.nodes[6]), (c.nodes[6],c.nodes[5]))
faces(c::Wedge) = ((c.nodes[1],c.nodes[3],c.nodes[2]), (c.nodes[1],c.nodes[2],c.nodes[5],c.nodes[4]), (c.nodes[3],c.nodes[1],c.nodes[4],c.nodes[6]), (c.nodes[2],c.nodes[3],c.nodes[6],c.nodes[5]), (c.nodes[4],c.nodes[5],c.nodes[6]))

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
default_interpolation(::Type{Wedge}) = Lagrange{3,RefPrism,1}()

"""
    boundaryfunction(::Type{<:BoundaryIndex})

Helper function to dispatch on the correct entity from a given boundary index.
"""
boundaryfunction(::Type{<:BoundaryIndex})

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
