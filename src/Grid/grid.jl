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

# Defined in src/Ferrite.jl
# abstract type AbstractCell{refshape <: AbstractRefShape} end

getrefshape(::AbstractCell{refshape}) where refshape = refshape

nvertices(c::AbstractCell) = length(vertices(c))
nedges(   c::AbstractCell) = length(edges(c))
nfaces(   c::AbstractCell) = length(faces(c))
nvertices(::Type{T}) where {T <: AbstractRefShape} = length(reference_vertices(T))
nedges(   ::Type{T}) where {T <: AbstractRefShape} = length(reference_edges(T))
nfaces(   ::Type{T}) where {T <: AbstractRefShape} = length(reference_faces(T))
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
    reference_faces(::AbstractRefShape)

Returns a tuple of n-tuples containing the ordered local node indices corresponding to
the vertices that define an *oriented face*.

An *oriented face* is a face with the first node having the local index and the other
nodes spanning such that the normal to the face is pointing outwards.

Note that the vertices are sufficient to define a face uniquely.
"""
reference_faces(::AbstractRefShape)

"""
    Ferrite.faces(::AbstractCell)

Returns a tuple of n-tuples containing the ordered node indices (of the nodes in a grid) corresponding to
the vertices that define an *oriented face*. This function induces the 
[`FaceIndex`](@ref), where the second index corresponds to the local index into this tuple.

An *oriented face* is a face with the first node having the local index and the other
nodes spanning such that the normal to the face is pointing outwards.

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
function edges(c::AbstractCell{RefLine})
    ns = get_node_ids(c)
    return ((ns[1],ns[2]),) # e1
end
function faces(c::AbstractCell{RefLine})
    return ()
end
function reference_vertices(::Type{RefLine})
    return (1, 2) 
end

# RefTriangle (refdim = 2): vertices for vertexdofs, faces for facedofs (edgedofs) and BC
function vertices(c::AbstractCell{RefTriangle})
    ns = get_node_ids(c)
    return (ns[1], ns[2], ns[3]) # v1, v2, v3
end
function edges(c::AbstractCell{RefTriangle})
    ns = get_node_ids(c)
    return (
        (ns[1], ns[2]), (ns[2], ns[3]), (ns[3], ns[1]), # e1, e2, e3
    )
end
function faces(c::AbstractCell{RefTriangle})
    ns = get_node_ids(c)
    return (
        (ns[1], ns[2], ns[3]), # f1
    )
end
function reference_edges(::Type{RefTriangle})
    return (
        (1, 2), (2, 3), (3, 1), # e1, e2, e3
    )
end

# RefQuadrilateral (refdim = 2): vertices for vertexdofs, faces for facedofs (edgedofs) and BC
function vertices(c::AbstractCell{RefQuadrilateral})
    ns = get_node_ids(c)
    return (ns[1], ns[2], ns[3], ns[4]) # v1, v2, v3, v4
end
function edges(c::AbstractCell{RefQuadrilateral})
    ns = get_node_ids(c)
    return (
        (ns[1], ns[2]), (ns[2], ns[3]), (ns[3], ns[4]), (ns[4], ns[1]), # e1, e2, e3, e4
    )
end
function faces(c::AbstractCell{RefQuadrilateral})
    ns = get_node_ids(c)
    return (
        (ns[1], ns[2], ns[3], ns[4]), # f1
    )
end
function reference_edges(::Type{RefQuadrilateral})
    return (
        (1, 2), (2, 3), (3, 4), (4, 1), # e1, e2, e3, e4
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
function reference_faces(::Type{RefTetrahedron})
    return (
        (1, 3, 2), (1, 2, 4), # f1, f2
        (2, 3, 4), (1, 4, 3), # f3, f4
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
function reference_faces(::Type{RefHexahedron})
    return (
        (1, 4, 3, 2), (1, 2, 6, 5), # f1, f2
        (2, 3, 7, 6), (3, 4, 8, 7), # f3, f4
        (1, 5, 8, 4), (5, 6, 7, 8), # f5, f6
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
function reference_faces(::Type{RefPrism})
    return (
        (1, 3, 2),    (1, 2, 5, 4), # f1, f2
        (3, 1, 4, 6), (2, 3, 6, 5), # f3, f4
        (4, 5, 6),                  # f5
    )
end

# RefPyramid (refdim = 3): vertices for vertexdofs, edges for edgedofs, faces for facedofs and BC
function vertices(c::AbstractCell{RefPyramid})
    ns = get_node_ids(c)
    return (ns[1], ns[2], ns[3], ns[4], ns[5],) # v1, ..., v5
end
function edges(c::AbstractCell{RefPyramid})
    ns = get_node_ids(c)
    return (
        (ns[1], ns[2]), (ns[1], ns[3]), (ns[1], ns[5]), (ns[2], ns[4]), 
        (ns[2], ns[5]), (ns[4], ns[3]), (ns[3], ns[5]), (ns[4], ns[5]), 
    )
end
function faces(c::AbstractCell{RefPyramid})
    ns = get_node_ids(c)
    return (
        (ns[1], ns[3], ns[4], ns[2]), (ns[1], ns[2], ns[5]), 
        (ns[1], ns[5], ns[3]), (ns[2], ns[4], ns[5]), 
        (ns[3], ns[5], ns[4]),                                      
    )
end
function reference_faces(::Type{RefPyramid})
    return (
        (1, 3, 4, 2), (1, 2, 5), # f1, f2
        (1, 5, 3), (2, 4, 5),    # f3, f4
        (3, 5, 4),               # f5
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
struct Pyramid                <: AbstractCell{RefPyramid}       nodes::NTuple{ 5, Int} end

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
default_interpolation(::Type{Pyramid})                = Lagrange{RefPyramid,       1}()

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
- `facetsets::Dict{String,Set{FacetIndex}}`: maps a `String` to a `Set` of `Set{FacetIndex} (global_cell_id, local_facet_id)`
- `vertexsets::Dict{String,Set{VertexIndex}}`: maps a `String` key to a `Set` of local vertex ids
- `boundary_matrix::SparseMatrixCSC{Bool,Int}`: optional, only needed by `onboundary` to check if a cell is on the boundary, see, e.g. Helmholtz example
"""
mutable struct Grid{dim,C<:AbstractCell,T<:Real} <: AbstractGrid{dim}
    cells::Vector{C}
    nodes::Vector{Node{dim,T}}
    # Sets
    cellsets::Dict{String,Set{Int}}
    nodesets::Dict{String,Set{Int}}
    facetsets::Dict{String,Set{FacetIndex}}
    vertexsets::Dict{String,Set{VertexIndex}}
    # Boundary matrix (faces per cell × cell)
    boundary_matrix::SparseMatrixCSC{Bool,Int} # TODO: Deprecate!
end

function Grid(cells::Vector{C},
              nodes::Vector{Node{dim,T}};
              cellsets::Dict{String,Set{Int}}=Dict{String,Set{Int}}(),
              nodesets::Dict{String,Set{Int}}=Dict{String,Set{Int}}(),
              facetsets::Dict{String,Set{FacetIndex}}=Dict{String,Set{FacetIndex}}(),
              facesets = nothing,
              vertexsets::Dict{String,Set{VertexIndex}}=Dict{String,Set{VertexIndex}}(),
              boundary_matrix::SparseMatrixCSC{Bool,Int}=spzeros(Bool, 0, 0)) where {dim,C,T}
    if facesets !== nothing 
        if isempty(facetsets)
            @warn "facesets in Grid is deprecated, use facetsets instead" maxlog=1
            for (key, set) in facesets
                facetsets[key] = Set(FacetIndex(cellnr, facenr) for (cellnr, facenr) in set)
            end
        else
            error("facesets are deprecated, use only facetsets")
        end
    end
    return Grid(cells, nodes, cellsets, nodesets, facetsets, vertexsets, boundary_matrix)
end

##########################
# Grid utility functions #
##########################
"""
    get_coordinate_type(::AbstractGrid)

Get the datatype for a single point in the grid.
"""
get_coordinate_type(::Grid{dim,C,T}) where {dim,C,T} = Vec{dim,T} # Node is baked into the mesh type.

"""
    toglobal(grid::AbstractGrid, vertexidx::VertexIndex) -> Int
    toglobal(grid::AbstractGrid, vertexidx::Vector{VertexIndex}) -> Vector{Int}
This function takes the local vertex representation (a `VertexIndex`) and looks up the unique global id (an `Int`).
"""
toglobal(grid::AbstractGrid,vertexidx::VertexIndex) = vertices(getcells(grid,vertexidx[1]))[vertexidx[2]]
toglobal(grid::AbstractGrid,vertexidx::Vector{VertexIndex}) = unique(toglobal.((grid,),vertexidx))

getsdim(::AbstractGrid{sdim}) where sdim = sdim
@inline getdim(g::AbstractGrid) = getsdim(g) # TODO: Deprecate

"""
    get_reference_dimensionality(grid::AbstractGrid) -> Union{Int, Symbol}

Get information about the reference dimensions of the cells in the grid. 
If all cells have the same reference dimension, `rdim::Int` is returned. 
Otherwise, the `Symbol` `:mixed` is returned indicating a mixed-rdimensionality grid.
"""
get_reference_dimensionality(g::AbstractGrid) = _get_reference_dimensionality(getcells(g))
_get_reference_dimensionality(::AbstractVector{C}) where C <: AbstractCell{<:AbstractRefShape{rdim}} where rdim = rdim # Fast path for single rdim inferable from eltype 
function _get_reference_dimensionality(cells::AbstractVector{<:AbstractCell})
    # Could make fast-path for eltype being union of cells with different rdims, but @KristofferC recommends against that,
    # https://discourse.julialang.org/t/iterating-through-types-of-a-union-in-a-type-stable-manner/58285/3
    # Note, this function is inherently type-instable.
    rdims = Set{Int}()
    for cell in cells
        push!(rdims, getdim(cell))
    end
    length(rdims) == 1 && return first(rdims)
    return :mixed
end

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
function nnodes_per_cell(grid::AbstractGrid) 
    if !isconcretetype(getcelltype(grid))
        error("There are different celltypes in the `grid`. Use `nnodes_per_cell(grid, cellid::Int)` instead")
    end
    return nnodes(first(grid.cells))
end
@inline nnodes_per_cell(grid::AbstractGrid, i::Int) = nnodes(grid.cells[i])

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
    getfacetset(grid::AbstractGrid, setname::String)

Returns all facets as `FacetIndex` in a `Set` of a given `setname`.
"""
@inline getfacetset(grid::AbstractGrid, setname::String) = grid.facetsets[setname]
"""
    getfacetsets(grid::AbstractGrid)

Returns all facet sets of the `grid`.
"""
@inline getfacetsets(grid::AbstractGrid) = grid.facetsets


"""
    getvertexset(grid::AbstractGrid, setname::String)

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
    transform_coordinates!(grid::Abstractgrid, f::Function)

Transform all nodes of the `grid` based on some transformation function `f`.
"""
function transform_coordinates!(g::Grid, f::Function)
    replace!(n -> Node(f(get_node_coordinate(n))), g.nodes)
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
    _addset!(grid, name, cellid, grid.cellsets)
end

function addcellset!(grid::AbstractGrid, name::String, f::Function; all::Bool=true)
    _addset!(grid, name, create_cellset(grid, f; all), grid.cellsets)
end

"""
    addfacetset!(grid::AbstractGrid, name::String, faceid::Union{Set{FacetIndex},Vector{FacetIndex}})
    addfacetset!(grid::AbstractGrid, name::String, f::Function; all::Bool=true) 

Adds a facetset to the grid with key `name`.
A facetset maps a `String` key to a `Set` of tuples corresponding to `(global_cell_id, local_facet_id)`.
Facetsets can be used to initialize `Dirichlet` boundary conditions for the `ConstraintHandler`.
`all=true` implies that `f(x)` must return `true` for all nodal coordinates `x` on the facet if the facet
should be added to the set, otherwise it suffices that `f(x)` returns `true` for one node. 

```julia
addfacetset!(grid, "right", Set((FacetIndex(2,2), FacetIndex(4,2)))) #see grid manual example for reference
addfacetset!(grid, "clamped", x -> norm(x[1]) ≈ 0.0) #see incompressible elasticity example for reference
```
"""
addfacetset!(grid::AbstractGrid, name::String, set::Union{Set{FacetIndex},Vector{FacetIndex}}) = 
    _addset!(grid, name, set, grid.facetsets)

addfacetset!(grid::AbstractGrid, name::String, f::Function; all::Bool=true) = 
    _addset!(grid, name, create_facetset(grid, f; all=all), grid.facetsets)

"""
    addvertexset!(grid::AbstractGrid, name::String, faceid::Union{Set{FaceIndex},Vector{FaceIndex}})
    addvertexset!(grid::AbstractGrid, name::String, f::Function) 

Adds a vertexset to the grid with key `name`.
A vertexset maps a `String` key to a `Set` of tuples corresponding to `(global_cell_id, local_vertex_id)`.
Vertexsets can be used to initialize `Dirichlet` boundary conditions for the `ConstraintHandler`.

```julia
addvertexset!(grid, "right", Set((VertexIndex(2,2), VertexIndex(4,2))))
addvertexset!(grid, "clamped", x -> norm(x[1]) ≈ 0.0)
```
"""
addvertexset!(grid::AbstractGrid, name::String, set::Union{Set{VertexIndex},Vector{VertexIndex}}) = 
    _addset!(grid, name, set, grid.vertexsets)

addvertexset!(grid::AbstractGrid, name::String, f::Function) = 
    _addset!(grid, name, create_vertexset(grid, f; all=true), grid.vertexsets)

function _addset!(grid::AbstractGrid, name::String, _set, dict::Dict)
    _check_setname(dict, name)
    set = Set(_set)
    _warn_emptyset(set, name)
    dict[name] = set
    grid
end

function _create_set(f::Function, grid::AbstractGrid, ::Type{BI}; all=true) where {BI <: BoundaryIndex}
    set = Set{BI}()
    for (cell_idx, cell) in enumerate(getcells(grid))
        for (entity_idx, entity) in enumerate(boundaryfunction(BI)(cell))
            pass = all
            for node_idx in entity
                v = f(get_node_coordinate(grid, node_idx))
                all ? (!v && (pass = false; break)) : (v && (pass = true; break))
            end
            pass && push!(set, BI(cell_idx, entity_idx))
        end
    end
    return set
end

function push_entity_instances!(set::Set{BI}, grid::AbstractGrid, top, entity::BI) where {BI <: BoundaryIndex}
    push!(set, entity) # Add the given entity
    cell = getcells(grid, entity[1])
    verts = boundaryfunction(BI)(cell)[entity[2]]
    for cell_idx in top.vertex_to_cell[verts[1]]# Since all vertices should be shared, the first one can be used here
        cell_entities = boundaryfunction(BI)(getcells(grid, cell_idx))
        for (entity_idx, cell_entity) in pairs(cell_entities)
            if all(x -> x in verts, cell_entity)
                push!(set, BI(cell_idx, entity_idx))
            end
        end
    end
    return set
end

function _create_boundaryset(f::Function, grid::AbstractGrid, top #=::ExclusiveTopology=#, ::Type{BI}; all = true) where {BI <: BoundaryIndex}
    # Function barrier as get_facet_facet_neighborhood is not always type stable
    function _makeset(ff_nh)
        set = Set{BI}()
        for (ff_nh_idx, neighborhood) in pairs(ff_nh)
            # ff_nh_idx::CartesianIndex into Matrix{<:EntityNeighborhood}
            isempty(neighborhood) || continue # Skip any facets with neighbors (not on boundary)
            cell_idx  = ff_nh_idx[1]
            facet_nr = ff_nh_idx[2]
            cell = getcells(grid, cell_idx)
            facet_nodes = facets(cell)[facet_nr]
            for (subentity_idx, subentity_nodes) in pairs(boundaryfunction(BI)(cell))
                if Base.all(n -> n in facet_nodes, subentity_nodes)
                    pass = all
                    for node_idx in subentity_nodes
                        v = f(get_node_coordinate(grid, node_idx))
                        all ? (!v && (pass = false; break)) : (v && (pass = true; break))
                    end
                    pass && push_entity_instances!(set, grid, top, BI(cell_idx, subentity_idx))
                end
            end
        end
        return set
    end
    return _makeset(get_facet_facet_neighborhood(top, grid))::Set{BI}
end

function create_cellset(grid::AbstractGrid, f::Function; all::Bool=true)
    cells = Set{Int}()
    for (i, cell) in enumerate(getcells(grid))
        pass = all
        for node_idx in get_node_ids(cell)
            v = f(get_node_coordinate(grid, node_idx))
            all ? (!v && (pass = false; break)) : (v && (pass = true; break))
        end
        pass && push!(cells, i)
    end
    return cells 
end
create_vertexset(grid::AbstractGrid, f::Function; kwargs...) = _create_set(f, grid, VertexIndex; kwargs...)
create_edgeset(  grid::AbstractGrid, f::Function; kwargs...) = _create_set(f, grid, EdgeIndex;   kwargs...)
create_faceset(  grid::AbstractGrid, f::Function; kwargs...) = _create_set(f, grid, FaceIndex;   kwargs...)
create_facetset( grid::AbstractGrid, f::Function; kwargs...) = _create_set(f, grid, FacetIndex;  kwargs...)

create_boundaryvertexset(grid::AbstractGrid, top, f::Function; kwargs...) = _create_boundaryset(f, grid, top, VertexIndex; kwargs...)
create_boundaryedgeset(  grid::AbstractGrid, top, f::Function; kwargs...) = _create_boundaryset(f, grid, top, EdgeIndex; kwargs...)
create_boundaryfaceset(  grid::AbstractGrid, top, f::Function; kwargs...) = _create_boundaryset(f, grid, top, FaceIndex; kwargs...)
create_boundaryfacetset( grid::AbstractGrid, top, f::Function; kwargs...) = _create_boundaryset(f, grid, top, FacetIndex; kwargs...)

"""
addboundaryvertexset!(grid::AbstractGrid, topology::ExclusiveTopology, name::String, f::Function; all::Bool=true)

Adds a boundary vertexset to the grid with key `name`.
A vertexset maps a `String` key to a `Set` of tuples corresponding to `(global_cell_id,
local_vertex_id)`. `all=true` implies that `f(x)` must return `true` for all nodal
coordinates `x` on the face if the face should be added to the set, otherwise it suffices
that `f(x)` returns `true` for one node.
"""
function addboundaryvertexset!(grid::AbstractGrid, top, name::String, f::Function; kwargs...)
    set = create_boundaryvertexset(grid, top, f; kwargs...)
    return _addset!(grid, name, set, grid.vertexsets)
end

"""
addboundaryfacetset!(grid::AbstractGrid, topology::ExclusiveTopology, name::String, f::Function; all::Bool=true)

Adds a boundary facetset to the grid with key `name`.
A facetset maps a `String` key to a `Set` of tuples corresponding to `(global_cell_id,
local_facet_id)`. Facetsets are used to initialize `Dirichlet` structs, that are needed to
specify the boundary for the `ConstraintHandler`. `all=true` implies that `f(x)` must return
`true` for all nodal coordinates `x` on the facet if the facet should be added to the set,
otherwise it suffices that `f(x)` returns `true` for one node.
"""
function addboundaryfacetset!(grid::AbstractGrid, top, name::String, f::Function; kwargs...)
    set = create_boundaryfacetset(grid, top, f; kwargs...)
    return _addset!(grid, name, set, grid.facetsets)
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
    getcoordinates(grid::AbstractGrid, idx::Union{Int,CellIndex})
    getcoordinates(cache::CellCache)
    
Get a vector with the coordinates of the cell corresponding to `idx` or `cache`
"""
@inline function getcoordinates(grid::AbstractGrid, idx::Int)
    CT = get_coordinate_type(grid)
    cell = getcells(grid, idx)
    N = nnodes(cell)
    x = Vector{CT}(undef, N)
    getcoordinates!(x, grid, cell)
end
@inline getcoordinates(grid::AbstractGrid, cell::CellIndex) = getcoordinates(grid, cell.idx)

"""
    getcoordinates!(x::Vector{<:Vec}, grid::AbstractGrid, idx::Union{Int,CellIndex})
    getcoordinates!(x::Vector{<:Vec}, grid::AbstractGrid, cell::AbstractCell)
    
Mutate `x` to the coordinates of the cell corresponding to `idx` or `cell`.
"""
@inline function getcoordinates!(x::Vector{Vec{dim,T}}, grid::Ferrite.AbstractGrid, cell::Ferrite.AbstractCell) where {dim,T}
    node_ids = get_node_ids(cell)
    @inbounds for i in 1:length(x)
        x[i] = get_node_coordinate(grid, node_ids[i])
    end
    return x
end
@inline function getcoordinates!(x::Vector{Vec{dim,T}}, grid::Ferrite.AbstractGrid, cellid::Int) where {dim,T} 
    cell = getcells(grid, cellid)
    getcoordinates!(x, grid, cell)
end
@inline getcoordinates!(x::Vector{Vec{dim,T}}, grid::AbstractGrid, cell::CellIndex) where {dim, T} = getcoordinates!(x, grid, cell.idx)

"""
    get_node_coordinate(grid::AbstractGrid, n::Int)
    
Return the coordinate of the `n`th node in `grid`
"""
get_node_coordinate(grid, n) = get_node_coordinate(getnodes(grid, n))

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

boundaryfunction(::Type{FaceIndex}) = faces
boundaryfunction(::Type{EdgeIndex}) = edges
boundaryfunction(::Type{VertexIndex}) = vertices
boundaryfunction(::Type{FacetIndex}) = facets

for INDEX in (:VertexIndex, :EdgeIndex, :FaceIndex, :FacetIndex)
    @eval begin  
        #Constructor
        ($INDEX)(a::Int, b::Int) = ($INDEX)((a,b))

        Base.getindex(I::($INDEX), i::Int) = I.idx[i]
        
        #To be able to do a,b = faceidx
        Base.iterate(I::($INDEX), state::Int=1) = (state==3) ?  nothing : (I[state], state+1)

        # Necessary to check if, e.g. `(cellid, faceidx) in faceset`
        Base.isequal(x::$INDEX, y::$INDEX) = x.idx == y.idx
        Base.isequal(x::Tuple{Int, Int}, y::$INDEX) = x[1] == y.idx[1] && x[2] == y.idx[2]
        Base.isequal(y::$INDEX, x::Tuple{Int, Int}) = x[1] == y.idx[1] && x[2] == y.idx[2]
        Base.hash(x::$INDEX, h::UInt) = hash(x.idx, h)
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


@doc raw"""
    InterfaceOrientationInfo

Orientation information for 1D and 2D entities.
The orientation is defined by the indices of the grid nodes
associated to the vertices. To give an example, the oriented path
```
1 ---> 2
```
is called *regular*, indicated by `flipped=false`, while the oriented path
```
2 ---> 1
```
is called *inverted*, indicated by `flipped=true`.

2D entities can be flipped (i.e. the defining vertex order is reverse to the 
spanning vertex order) and the vertices can be rotated against each other.

The reference entity is a one with it's first node is the lowest index vertex
and its vertices span counter-clock-wise.
Take for example the faces
```
1           2
| \         | \
|  \        |  \
| A \       | B \
|    \      |    \
2-----3     3-----1
```
which are rotated against each other by 240° after tranfroming to an
equilateral triangle (shift index is 2). Or the faces
```
3           2
| \         | \
|  \        |  \
| A \       | B \
|    \      |    \
2-----1     3-----1
```
which are flipped against each other.
"""
struct OrientationInfo
    flipped::Bool
    shift_index::Int
end

function OrientationInfo(path::NTuple{2, Int})
    flipped = first(path) < last(path)
    return OrientationInfo(flipped, 0)
end

function OrientationInfo(surface::NTuple{N, Int}) where N
    min_idx = argmin(surface)
    shift_index = min_idx - 1
    if min_idx == 1
        flipped = surface[2] < surface[end]
    elseif min_idx == length(surface)
        flipped = surface[1] < surface[end-1]
    else
        flipped = surface[min_idx + 1] < surface[min_idx - 1]
    end
    return OrientationInfo(flipped, shift_index)
end

# Dispatches for facets

@inline facets(c::AbstractCell{<:AbstractRefShape{1}}) = map(i -> (i,), vertices(c)) # facet always tuple of tuple
@inline facets(c::AbstractCell{<:AbstractRefShape{2}}) = edges(c)
@inline facets(c::AbstractCell{<:AbstractRefShape{3}}) = faces(c)

@inline reference_facets(refshape::Type{<:AbstractRefShape{1}}) = map(i -> (i,), reference_vertices(refshape)) 
@inline reference_facets(refshape::Type{<:AbstractRefShape{2}}) = reference_edges(refshape)
@inline reference_facets(refshape::Type{<:AbstractRefShape{3}}) = reference_faces(refshape)
nfacets(::Type{T}) where {T <: AbstractRefShape} = length(reference_facets(T))
nfacets(c::AbstractCell) = length(facets(c))
# Deprecation (TODO: Move to deprecated.jl)
function getfaceset(grid::AbstractGrid, name::String)
    @warn "getfaceset is deprecated, use getfacetset instead" maxlog=1
    getfacetset(grid, name)
end