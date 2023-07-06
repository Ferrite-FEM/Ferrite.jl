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

"""
    get_node_coordinate(::Node)
    
Get the value of the node coordinate.
"""
get_node_coordinate(n::Node) = n.x

"""
    get_coordinate_type(::Node)

Get the data type of the the node coordinate.
"""
get_coordinate_type(::Node{dim,T}) where {dim,T}  = Vec{dim,T}

"""
    get_coordinate_eltype(::Node)

Get the data type of the components of the nodes coordinate.
"""
get_coordinate_eltype(::Node{dim,T}) where {dim,T} = T

##########################
# AbstractCell interface #
##########################

abstract type AbstractCell{refshape <: AbstractRefShape} end

getrefshape(::AbstractCell{refshape}) where refshape = refshape

nvertices(c::AbstractCell) = length(vertices(c))
nedges(   c::AbstractCell) = length(edges(c))
nfaces(   c::AbstractCell) = length(faces(c))
nnodes(   c::AbstractCell) = length(get_node_ids(c))

"""
    Ferrite.vertices(::AbstractCell)

Returns a tuple with the node indices (of the nodes in a grid) for each vertex in a given cell.
This function induces the [`VertexIndex`](@ref), where the second index 
corresponds to the local index into this tuple.
"""
vertices(::AbstractCell)

"""
    Ferrite.edges(::AbstractCell)

Returns a tuple of 2-tuples containing the ordered node indices (of the nodes in a grid) corresponding to
the vertices that define an *oriented edge*. This function induces the 
[`EdgeIndex`](@ref), where the second index corresponds to the local index into this tuple.

Note that the vertices are sufficient to define an edge uniquely.
"""
edges(::AbstractCell)

"""
    Ferrite.faces(::AbstractCell)

Returns a tuple of n-tuples containing the ordered node indices (of the nodes in a grid) corresponding to
the vertices that define an *oriented face*. This function induces the 
[`FaceIndex`](@ref), where the second index corresponds to the local index into this tuple.

Note that the vertices are sufficient to define a face uniquely.
"""
faces(::AbstractCell)

"""
    Ferrite.default_interpolation(::AbstractCell)::Interpolation

Returns the interpolation which defines the geometry of a given cell.
"""
default_interpolation(::AbstractCell)

"""
    Ferrite.get_node_ids(c::AbstractCell)

Return the node id's for cell `c` in the order determined by the cell's reference cell.

Default implementation: `c.nodes`.
"""
get_node_ids(c::AbstractCell) = c.nodes

# Default implementations of vertices/edges/faces that work as long as get_node_ids is
# correctly implemented for the cell.

# RefLine (refdim = 1): vertices for vertexdofs, faces for BC
function vertices(c::AbstractCell{RefLine})
    ns = get_node_ids(c)
    return (ns[1], ns[2]) # v1, v2
end
function faces(c::AbstractCell{RefLine})
    ns = get_node_ids(c)
    return ((ns[1],), (ns[2],)) # f1, f2
end

# RefTriangle (refdim = 2): vertices for vertexdofs, faces for facedofs (edgedofs) and BC
function vertices(c::AbstractCell{RefTriangle})
    ns = get_node_ids(c)
    return (ns[1], ns[2], ns[3]) # v1, v2, v3
end
function faces(c::AbstractCell{RefTriangle})
    ns = get_node_ids(c)
    return (
        (ns[1], ns[2]), (ns[2], ns[3]), (ns[3], ns[1]), # f1, f2, f3
    )
end

# RefQuadrilateral (refdim = 2): vertices for vertexdofs, faces for facedofs (edgedofs) and BC
function vertices(c::AbstractCell{RefQuadrilateral})
    ns = get_node_ids(c)
    return (ns[1], ns[2], ns[3], ns[4]) # v1, v2, v3, v4
end
function faces(c::AbstractCell{RefQuadrilateral})
    ns = get_node_ids(c)
    return (
        (ns[1], ns[2]), (ns[2], ns[3]), (ns[3], ns[4]), (ns[4], ns[1]), # f1, f2, f3, f4
    )
end

# RefTetrahedron (refdim = 3): vertices for vertexdofs, edges for edgedofs, faces for facedofs and BC
function vertices(c::AbstractCell{RefTetrahedron})
    ns = get_node_ids(c)
    return (ns[1], ns[2], ns[3], ns[4]) # v1, v2, v3, v4
end
function edges(c::AbstractCell{RefTetrahedron})
    ns = get_node_ids(c)
    return (
        (ns[1], ns[2]), (ns[2], ns[3]), (ns[3], ns[1]), # e1, e2, e3
        (ns[1], ns[4]), (ns[2], ns[4]), (ns[3], ns[4]), # e4, e5, e6
    )
end
function faces(c::AbstractCell{RefTetrahedron})
    ns = get_node_ids(c)
    return (
        (ns[1], ns[3], ns[2]), (ns[1], ns[2], ns[4]), # f1, f2
        (ns[2], ns[3], ns[4]), (ns[1], ns[4], ns[3]), # f3, f4
    )
end

# RefHexahedron (refdim = 3): vertices for vertexdofs, edges for edgedofs, faces for facedofs and BC
function vertices(c::AbstractCell{RefHexahedron})
    ns = get_node_ids(c)
    return (
        ns[1], ns[2], ns[3], ns[4], ns[5], ns[6], ns[7], ns[8], # v1, ..., v8
    )
end
function edges(c::AbstractCell{RefHexahedron})
    ns = get_node_ids(c)
    return (
        (ns[1], ns[2]), (ns[2], ns[3]), (ns[3], ns[4]), (ns[4], ns[1]), # e1, e2, e3, e4
        (ns[5], ns[6]), (ns[6], ns[7]), (ns[7], ns[8]), (ns[8], ns[5]), # e5, e6, e7, e8
        (ns[1], ns[5]), (ns[2], ns[6]), (ns[3], ns[7]), (ns[4], ns[8]), # e9, e10, e11, e12
    )
end
function faces(c::AbstractCell{RefHexahedron})
    ns = get_node_ids(c)
    return (
        (ns[1], ns[4], ns[3], ns[2]), (ns[1], ns[2], ns[6], ns[5]), # f1, f2
        (ns[2], ns[3], ns[7], ns[6]), (ns[3], ns[4], ns[8], ns[7]), # f3, f4
        (ns[1], ns[5], ns[8], ns[4]), (ns[5], ns[6], ns[7], ns[8]), # f5, f6
    )
end

# RefPrism (refdim = 3): vertices for vertexdofs, edges for edgedofs, faces for facedofs and BC
function vertices(c::AbstractCell{RefPrism})
    ns = get_node_ids(c)
    return (ns[1], ns[2], ns[3], ns[4], ns[5], ns[6]) # v1, ..., v6
end
function edges(c::AbstractCell{RefPrism})
    ns = get_node_ids(c)
    return (
        (ns[2], ns[1]), (ns[1], ns[3]), (ns[1], ns[4]), (ns[3], ns[2]), # e1, e2, e3, e4
        (ns[2], ns[5]), (ns[3], ns[6]), (ns[4], ns[5]), (ns[4], ns[6]), # e5, e6, e7, e8
        (ns[6], ns[5]),                                                 # e9
    )
end
function faces(c::AbstractCell{RefPrism})
    ns = get_node_ids(c)
    return (
        (ns[1], ns[3], ns[2]),        (ns[1], ns[2], ns[5], ns[4]), # f1, f2
        (ns[3], ns[1], ns[4], ns[6]), (ns[2], ns[3], ns[6], ns[5]), # f3, f4
        (ns[4], ns[5], ns[6]),                                      # f5
    )
end


######################################################
# Concrete implementations of AbstractCell interface #
######################################################

# Lagrange interpolation based cells
struct Line                   <: AbstractCell{RefLine}          nodes::NTuple{ 2, Int} end
struct QuadraticLine          <: AbstractCell{RefLine}          nodes::NTuple{ 3, Int} end
struct Triangle               <: AbstractCell{RefTriangle}      nodes::NTuple{ 3, Int} end
struct QuadraticTriangle      <: AbstractCell{RefTriangle}      nodes::NTuple{ 6, Int} end
struct Quadrilateral          <: AbstractCell{RefQuadrilateral} nodes::NTuple{ 4, Int} end
struct QuadraticQuadrilateral <: AbstractCell{RefQuadrilateral} nodes::NTuple{ 9, Int} end
struct Tetrahedron            <: AbstractCell{RefTetrahedron}   nodes::NTuple{ 4, Int} end
struct QuadraticTetrahedron   <: AbstractCell{RefTetrahedron}   nodes::NTuple{10, Int} end
struct Hexahedron             <: AbstractCell{RefHexahedron}    nodes::NTuple{ 8, Int} end
struct QuadraticHexahedron    <: AbstractCell{RefHexahedron}    nodes::NTuple{27, Int} end
struct Wedge                  <: AbstractCell{RefPrism}         nodes::NTuple{ 6, Int} end

default_interpolation(::Type{Line})                   = Lagrange{RefLine,          1}()
default_interpolation(::Type{QuadraticLine})          = Lagrange{RefLine,          2}()
default_interpolation(::Type{Triangle})               = Lagrange{RefTriangle,      1}()
default_interpolation(::Type{QuadraticTriangle})      = Lagrange{RefTriangle,      2}()
default_interpolation(::Type{Quadrilateral})          = Lagrange{RefQuadrilateral, 1}()
default_interpolation(::Type{QuadraticQuadrilateral}) = Lagrange{RefQuadrilateral, 2}()
default_interpolation(::Type{Tetrahedron})            = Lagrange{RefTetrahedron,   1}()
default_interpolation(::Type{QuadraticTetrahedron})   = Lagrange{RefTetrahedron,   2}()
default_interpolation(::Type{Hexahedron})             = Lagrange{RefHexahedron,    1}()
default_interpolation(::Type{QuadraticHexahedron})    = Lagrange{RefHexahedron,    2}()
default_interpolation(::Type{Wedge})                  = Lagrange{RefPrism,         1}()

# TODO: Remove this, used for Quadrilateral3D
edges(c::Quadrilateral#=3D=#) = faces(c)

# Serendipity interpolation based cells
struct SerendipityQuadraticQuadrilateral <: AbstractCell{RefQuadrilateral} nodes::NTuple{ 8, Int} end
struct SerendipityQuadraticHexahedron    <: AbstractCell{RefHexahedron}    nodes::NTuple{20, Int} end

default_interpolation(::Type{SerendipityQuadraticQuadrilateral}) = Serendipity{RefQuadrilateral, 2}()
default_interpolation(::Type{SerendipityQuadraticHexahedron})    = Serendipity{RefHexahedron,    2}()

"""
    nvertices_on_face(cell::AbstractCell, local_face_index::Int)
Specifies for each subtype of AbstractCell how many nodes form a face.
"""
nvertices_on_face(cell::AbstractCell, local_face_index::Int) = length(faces(cell)[local_face_index])
"""
    nvertices_on_edge(::AbstractCell, local_edge_index::Int)
Specifies for each subtype of AbstractCell how many nodes form an edge.
"""
nvertices_on_edge(cell::AbstractCell, local_edge_index::Int) = length(edges(cell)[local_edge_index])

getdim(::Union{AbstractCell{refshape},Type{<:AbstractCell{refshape}}}) where {refdim, refshape <: AbstractRefShape{refdim}} = refdim


######################
### Mesh interface ###
######################
abstract type AbstractGrid{dim} end

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
    get_coordinate_type(::AbstractGrid)

Get the datatype for a single point in the grid.
"""
get_coordinate_type(grid::Grid{dim,C,T}) where {dim,C,T} = Vec{dim,T} # Node is baked into the mesh type.

"""
    toglobal(grid::AbstractGrid, vertexidx::VertexIndex) -> Int
    toglobal(grid::AbstractGrid, vertexidx::Vector{VertexIndex}) -> Vector{Int}
This function takes the local vertex representation (a `VertexIndex`) and looks up the unique global id (an `Int`).
"""
toglobal(grid::AbstractGrid,vertexidx::VertexIndex) = vertices(getcells(grid,vertexidx[1]))[vertexidx[2]]
toglobal(grid::AbstractGrid,vertexidx::Vector{VertexIndex}) = unique(toglobal.((grid,),vertexidx))

@inline getdim(::AbstractGrid{dim}) where {dim} = dim
"""
    getcells(grid::AbstractGrid)
    getcells(grid::AbstractGrid, v::Union{Int,Vector{Int}}
    getcells(grid::AbstractGrid, setname::String)

Returns either all `cells::Collection{C<:AbstractCell}` of a `<:AbstractGrid` or a subset based on an `Int`, `Vector{Int}` or `String`.
Whereas the last option tries to call a `cellset` of the `grid`. `Collection` can be any indexable type, for `Grid` it is `Vector{C<:AbstractCell}`.
"""
@inline getcells(grid::AbstractGrid) = grid.cells
@inline getcells(grid::AbstractGrid, v::Union{Int, Vector{Int}}) = grid.cells[v]
@inline getcells(grid::AbstractGrid, setname::String) = grid.cells[collect(getcellset(grid,setname))]
"Returns the number of cells in the `<:AbstractGrid`."
@inline getncells(grid::AbstractGrid) = length(grid.cells)
"Returns the celltype of the `<:AbstractGrid`."
@inline getcelltype(grid::AbstractGrid) = eltype(grid.cells)
@inline getcelltype(grid::AbstractGrid, i::Int) = typeof(grid.cells[i])

"""
    getnodes(grid::AbstractGrid)
    getnodes(grid::AbstractGrid, v::Union{Int,Vector{Int}}
    getnodes(grid::AbstractGrid, setname::String)

Returns either all `nodes::Collection{N}` of a `<:AbstractGrid` or a subset based on an `Int`, `Vector{Int}` or `String`.
The last option tries to call a `nodeset` of the `<:AbstractGrid`. `Collection{N}` refers to some indexable collection where each element corresponds
to a Node.
"""
@inline getnodes(grid::AbstractGrid) = grid.nodes
@inline getnodes(grid::AbstractGrid, v::Union{Int, Vector{Int}}) = grid.nodes[v]
@inline getnodes(grid::AbstractGrid, setname::String) = grid.nodes[collect(getnodeset(grid,setname))]
"Returns the number of nodes in the grid."
@inline getnnodes(grid::AbstractGrid) = length(grid.nodes)
"Returns the number of nodes of the `i`-th cell."
@inline nnodes_per_cell(grid::AbstractGrid, i::Int=1) = nnodes(grid.cells[i])

get_node_coordinate(grid, nodeid) = get_node_coordinate(getnodes(grid, nodeid))
"Return the number type of the nodal coordinates."
@inline get_coordinate_eltype(grid::AbstractGrid) = get_coordinate_eltype(first(getnodes(grid)))

"""
    getcellset(grid::AbstractGrid, setname::String)

Returns all cells as cellid in a `Set` of a given `setname`.
"""
@inline getcellset(grid::AbstractGrid, setname::String) = grid.cellsets[setname]
"""
    getcellsets(grid::AbstractGrid)

Returns all cellsets of the `grid`.
"""
@inline getcellsets(grid::AbstractGrid) = grid.cellsets

"""
    getnodeset(grid::AbstractGrid, setname::String)

Returns all nodes as nodeid in a `Set` of a given `setname`.
"""
@inline getnodeset(grid::AbstractGrid, setname::String) = grid.nodesets[setname]
"""
    getnodesets(grid::AbstractGrid)

Returns all nodesets of the `grid`.
"""
@inline getnodesets(grid::AbstractGrid) = grid.nodesets

"""
    getfaceset(grid::AbstractGrid, setname::String)

Returns all faces as `FaceIndex` in a `Set` of a given `setname`.
"""
@inline getfaceset(grid::AbstractGrid, setname::String) = grid.facesets[setname]
"""
    getfacesets(grid::AbstractGrid)

Returns all facesets of the `grid`.
"""
@inline getfacesets(grid::AbstractGrid) = grid.facesets

"""
    getedgeset(grid::AbstractGrid, setname::String)

Returns all edges as `EdgeIndex` in a `Set` of a given `setname`.
"""
@inline getedgeset(grid::AbstractGrid, setname::String) = grid.edgesets[setname]
"""
    getedgesets(grid::AbstractGrid)

Returns all edge sets of the grid.
"""
@inline getedgesets(grid::AbstractGrid) = grid.edgesets

"""
    getedgeset(grid::AbstractGrid, setname::String)

Returns all vertices as `VertexIndex` in a `Set` of a given `setname`.
"""
@inline getvertexset(grid::AbstractGrid, setname::String) = grid.vertexsets[setname]
"""
    getvertexsets(grid::AbstractGrid)

Returns all vertex sets of the grid.
"""
@inline getvertexsets(grid::AbstractGrid) = grid.vertexsets

n_faces_per_cell(grid::Grid) = nfaces(getcelltype(grid))

# Transformations
"""
    transform!(grid::Abstractgrid, f::Function)

Transform all nodes of the `grid` based on some transformation function `f`.
"""
function transform!(g::Grid, f::Function)
    map!(n -> Node(f(get_node_coordinate(n))), g.nodes, g.nodes)
    return g
end

# Sets

_check_setname(dict, name) = haskey(dict, name) && throw(ArgumentError("there already exists a set with the name: $name"))
_warn_emptyset(set, name) = length(set) == 0 && @warn("no entities added to the set with name: $name")

"""
    addcellset!(grid::AbstractGrid, name::String, cellid::Union{Set{Int}, Vector{Int}})
    addcellset!(grid::AbstractGrid, name::String, f::function; all::Bool=true)

Adds a cellset to the grid with key `name`.
Cellsets are typically used to define subdomains of the problem, e.g. two materials in the computational domain.
The `DofHandler` can construct different fields which live not on the whole domain, but rather on a cellset.
`all=true` implies that `f(x)` must return `true` for all nodal coordinates `x` in the cell if the cell
should be added to the set, otherwise it suffices that `f(x)` returns `true` for one node. 

```julia
addcellset!(grid, "left", Set((1,3))) #add cells with id 1 and 3 to cellset left
addcellset!(grid, "right", x -> norm(x[1]) < 2.0 ) #add cell to cellset right, if x[1] of each cell's node is smaller than 2.0
```
"""
function addcellset!(grid::AbstractGrid, name::String, cellid::Union{Set{Int},Vector{Int}})
    _check_setname(grid.cellsets,  name)
    cells = Set(cellid)
    _warn_emptyset(cells, name)
    grid.cellsets[name] = cells
    grid
end

function addcellset!(grid::AbstractGrid, name::String, f::Function; all::Bool=true)
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
    addfaceset!(grid::AbstractGrid, name::String, faceid::Union{Set{FaceIndex},Vector{FaceIndex}})
    addfaceset!(grid::AbstractGrid, name::String, f::Function; all::Bool=true) 

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
addfaceset!(grid::AbstractGrid, name::String, set::Union{Set{FaceIndex},Vector{FaceIndex}}) = 
    _addset!(grid, name, set, grid.facesets)
addedgeset!(grid::AbstractGrid, name::String, set::Union{Set{EdgeIndex},Vector{EdgeIndex}}) = 
    _addset!(grid, name, set, grid.edgesets)
addvertexset!(grid::AbstractGrid, name::String, set::Union{Set{VertexIndex},Vector{VertexIndex}}) = 
    _addset!(grid, name, set, grid.vertexsets)
function _addset!(grid::AbstractGrid, name::String, _set, dict::Dict)
    _check_setname(dict, name)
    set = Set(_set)
    _warn_emptyset(set, name)
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
    _warn_emptyset(_set, name)
    dict[name] = _set
    grid
end


"""
    getfaceedges(grid::AbstractGrid, face::FaceIndex)
    getfaceedges(cell::AbstractCell, face::FaceIndex)

Returns the edges represented as `Set{EdgeIndex}` in a given face represented as
`FaceIndex`.

```julia-repl
julia> using Ferrite; using Ferrite: getfaceedges

julia> grid = generate_grid(Tetrahedron, (2,1,1));

julia> getfaceedges(grid, FaceIndex(4,2))
Set{EdgeIndex} with 3 elements:
  EdgeIndex((4, 4))
  EdgeIndex((4, 5))
  EdgeIndex((4, 1))
```
"""
function getfaceedges end

"""
    getfacevertices(grid::AbstractGrid, face::FaceIndex)
    getfacevertices(cell::AbstractCell, face::FaceIndex)

Returns the vertices represented as `Set{VertexIndex}` in a given face represented as
`FaceIndex`.

```julia-repl
julia> using Ferrite; using Ferrite: getfacevertices

julia> grid = generate_grid(Tetrahedron, (2,1,1));

julia> getfacevertices(grid, FaceIndex(4,2))
Set{VertexIndex} with 3 elements:
  VertexIndex((4, 2))
  VertexIndex((4, 4))
  VertexIndex((4, 1))
```
"""
function getfacevertices end

"""
    getedgevertices(grid::AbstractGrid, edge::EdgeIndex)
    getedgevertices(cell::AbstractCell, edge::EdgeIndex)

Returns the vertices represented as `Set{VertexIndex}` in a given edge represented as
`EdgeIndex`.

```julia-repl
julia> using Ferrite; using Ferrite: getedgevertices

julia> grid = generate_grid(Tetrahedron, (2,1,1));

julia> getedgevertices(grid, EdgeIndex(4,2))
Set{EdgeIndex} with 2 elements:
  VertexIndex((4, 2))
  VertexIndex((4, 3))
```
"""
function getedgevertices end

for (func,             entity_f, subentity_f, entity_t,   subentity_t) in (
    (:getfaceedges,    :faces,   :edges,      :FaceIndex, :EdgeIndex),
    (:getfacevertices, :faces,   :vertices,   :FaceIndex, :VertexIndex),
    (:getedgevertices, :edges,   :vertices,   :EdgeIndex, :VertexIndex),
)
    @eval begin
        function $(func)(grid::AbstractGrid, entity_idx::$(entity_t))
            cell = getcells(grid)[entity_idx[1]]
            return $(func)(cell, entity_idx)
        end
        function $(func)(cell::AbstractCell, entity_idx::$(entity_t))
            _set = Set{$(subentity_t)}()
            subentities = $(subentity_f)(cell)
            entity = $(entity_f)(cell)[entity_idx[2]]
            for (subentity_idx, subentity) in pairs(subentities)
                if all(x -> x in entity, subentity)
                    push!(_set, $(subentity_t)((entity_idx[1], subentity_idx)))
                end
            end
            return _set
        end
    end
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
    _warn_emptyset(grid.nodesets[name], name)
    grid
end

function addnodeset!(grid::AbstractGrid, name::String, f::Function)
    _check_setname(grid.nodesets, name)
    nodes = Set{Int}()
    for (i, n) in enumerate(getnodes(grid))
        f(n.x) && push!(nodes, i)
    end
    grid.nodesets[name] = nodes
    _warn_emptyset(grid.nodesets[name], name)
    grid
end

"""
    get_cell_coordinates!(x::Vector{Vec{dim,T}}, grid::AbstractGrid, cell::Int)
    get_cell_coordinates!(x::Vector{Vec{dim,T}}, grid::AbstractGrid, cell::AbstractCell)

Fills the vector `x` with the coordinates of a cell defined by either its cellid or the cell object itself.
"""
@inline function get_cell_coordinates!(x::Vector{Vec{dim,T}}, grid::Ferrite.AbstractGrid, cellid::Int) where {dim,T} 
    cell = getcells(grid, cellid)
    get_cell_coordinates!(x, grid, cell)
end

@inline function get_cell_coordinates!(x::Vector{Vec{dim,T}}, grid::Ferrite.AbstractGrid, cell::Ferrite.AbstractCell) where {dim,T}
    @inbounds for i in 1:length(x)
        x[i] = get_node_coordinate(grid, cell.nodes[i])
    end
    return x
end

@inline get_cell_coordinates!(x::Vector{Vec{dim,T}}, grid::AbstractGrid, cell::CellIndex) where {dim, T} = get_cell_coordinates!(x, grid, cell.idx)
@inline get_cell_coordinates!(x::Vector{Vec{dim,T}}, grid::AbstractGrid, face::FaceIndex) where {dim, T} = get_cell_coordinates!(x, grid, face.idx[1])


"""
    get_cell_coordinates(grid::AbstractGrid, cell)
Return a vector with the coordinates of the vertices of cell number `cell`.
"""
@inline function get_cell_coordinates(grid::AbstractGrid, cell::Int)
    dim = getdim(grid)
    T = get_coordinate_eltype(grid)
    _cell = getcells(grid, cell)
    N = nnodes(_cell)
    x = Vector{Vec{dim, T}}(undef, N)
    get_cell_coordinates!(x, grid, _cell)
end
@inline get_cell_coordinates(grid::AbstractGrid, cell::CellIndex) = get_cell_coordinates(grid, cell.idx)
@inline get_cell_coordinates(grid::AbstractGrid, face::FaceIndex) = get_cell_coordinates(grid, face.idx[1])

function cellnodes!(global_nodes::Vector{Int}, grid::AbstractGrid, i::Int)
    cell = getcells(grid, i)
    _cellnodes!(global_nodes, cell)
end
function _cellnodes!(global_nodes::Vector{Int}, cell::AbstractCell)
    @assert length(global_nodes) == nnodes(cell)
    @inbounds for i in 1:length(global_nodes)
        global_nodes[i] = cell.nodes[i]
    end
    return global_nodes
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

#################################
#### Orientation of Entities ####
#################################
# @TODO merge this code with into the logic in `ConstraintHandler`.

"""
    PathOrientationInfo

Orientation information for 1D entities.

The orientation for 1D entities is defined by the indices of the grid nodes
associated to the vertices. To give an example, the oriented path
```
1 ---> 2
```
is called *regular*, indicated by `regular=true`, while the oriented path
```
2 ---> 1
```
is called *inverted*, indicated by `regular=false`.
"""
struct PathOrientationInfo
    regular::Bool # Indicator whether the orientation is regular or inverted.
end

"""
    SurfaceOrientationInfo

Orientation information for 2D entities. Such an entity can be 
possibly flipped (i.e. the defining vertex order is reverse to the 
spanning vertex order) and the vertices can be rotated against each other.
Take for example the faces
```
1---2 2---3
| A | | B |
4---3 1---4
```
which are rotated against each other by 90° (shift index is 1) or the faces
```
1---2 2---1
| A | | B |
4---3 3---4
```
which are flipped against each other. Any combination of these can happen. 
The combination to map this local face to the defining face is encoded with
this data structure via ``rotate \\circ flip`` where the rotation is indiced by
the shift index.
    !!!NOTE TODO implement me.
"""
struct SurfaceOrientationInfo
    #flipped::Bool
    #shift_index::Int
end
