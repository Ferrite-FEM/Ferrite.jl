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
nfacets(  c::AbstractCell) = length(facets(c))
nnodes(   c::AbstractCell) = length(get_node_ids(c))

nvertices(::Type{T}) where {T <: AbstractRefShape} = length(reference_vertices(T))
nedges(   ::Type{T}) where {T <: AbstractRefShape} = length(reference_edges(T))
nfaces(   ::Type{T}) where {T <: AbstractRefShape} = length(reference_faces(T))
nfacets(  ::Type{T}) where {T <: AbstractRefShape} = length(reference_facets(T))

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
    Ferrite.facets(::AbstractCell)

Returns a tuple of n-tuples containing the ordered node indices (of the nodes in a grid) corresponding to
the vertices that define an oriented facet. This function induces the
[`FacetIndex`](@ref), where the second index corresponds to the local index into this tuple.

See also [`vertices`](@ref), [`edges`](@ref), and [`faces`](@ref)
"""
facets(::AbstractCell)

@inline facets(c::AbstractCell{<:AbstractRefShape{1}}) = map(i -> (i,), vertices(c)) # facet always tuple of tuple
@inline facets(c::AbstractCell{<:AbstractRefShape{2}}) = edges(c)
@inline facets(c::AbstractCell{<:AbstractRefShape{3}}) = faces(c)

"""
    Ferrite.reference_facets(::Type{<:AbstractRefShape})

Returns a tuple of n-tuples containing the ordered local node indices corresponding to
the vertices that define an oriented facet.

See also [`reference_vertices`](@ref), [`reference_edges`](@ref), and [`reference_faces`](@ref)
"""
reference_facets(::Type{<:AbstractRefShape})

@inline reference_facets(refshape::Type{<:AbstractRefShape{1}}) = map(i -> (i,), reference_vertices(refshape))
@inline reference_facets(refshape::Type{<:AbstractRefShape{2}}) = reference_edges(refshape)
@inline reference_facets(refshape::Type{<:AbstractRefShape{3}}) = reference_faces(refshape)

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

# Default implementations of <entity> = vertices/edges/faces that work as long as get_node_ids
# and `reference_<entity>` are correctly implemented for the cell / reference shape.

function vertices(c::AbstractCell{RefShape}) where RefShape
    ns = get_node_ids(c)
    return map(i -> ns[i], reference_vertices(RefShape))
end

function edges(c::AbstractCell{RefShape}) where RefShape
    ns = get_node_ids(c)
    return map(reference_edges(RefShape)) do re
        map(i -> ns[i], re)
    end
end

function faces(c::AbstractCell{RefShape}) where RefShape
    ns = get_node_ids(c)
    return map(reference_faces(RefShape)) do rf
        map(i -> ns[i], rf)
    end
end

# RefLine (refdim = 1)
reference_vertices(::Type{RefLine}) = (1, 2)
reference_edges(::Type{RefLine}) = ((1, 2),) # e1
reference_faces(::Type{RefLine}) = () # -

# RefTriangle (refdim = 2)
reference_vertices(::Type{RefTriangle}) = (1, 2, 3)
reference_edges(::Type{RefTriangle}) = ((1, 2), (2, 3), (3, 1)) # e1 ... e3
reference_faces(::Type{RefTriangle}) = ((1, 2, 3),) # f1

# RefQuadrilateral (refdim = 2)
reference_vertices(::Type{RefQuadrilateral}) = (1, 2, 3, 4)
reference_edges(::Type{RefQuadrilateral}) = ((1, 2), (2, 3), (3, 4), (4, 1)) # e1 ... e4
reference_faces(::Type{RefQuadrilateral}) = ((1, 2, 3, 4),) # f1

# RefTetrahedron (refdim = 3)
reference_vertices(::Type{RefTetrahedron}) = (1, 2, 3, 4)
reference_edges(::Type{RefTetrahedron}) = ((1, 2), (2, 3), (3, 1), (1, 4), (2, 4), (3, 4)) # e1 ... e6
reference_faces(::Type{RefTetrahedron}) = ((1, 3, 2), (1, 2, 4), (2, 3, 4), (1, 4, 3)) # f1 ... f4

# RefHexahedron (refdim = 3)
reference_vertices(::Type{RefHexahedron}) = (1, 2, 3, 4, 5, 6, 7, 8)
function reference_edges(::Type{RefHexahedron})
    return ((1, 2), (2, 3), (3, 4), (4, 1), (5, 6), (6, 7), # e1 ... e6
            (7, 8), (8, 5), (1, 5), (2, 6), (3, 7), (4, 8)) # e7 ... e12
end
function reference_faces(::Type{RefHexahedron})
    return ((1, 4, 3, 2), (1, 2, 6, 5), (2, 3, 7, 6), # f1, f2, f3
            (3, 4, 8, 7), (1, 5, 8, 4), (5, 6, 7, 8)) # f4, f5, f6
end

# RefPrism (refdim = 3)
reference_vertices(::Type{RefPrism}) = (1, 2, 3, 4, 5, 6)
function reference_edges(::Type{RefPrism})
    return ((2, 1), (1, 3), (1, 4), (3, 2), (2, 5), # e1, e2, e3, e4, e5
            (3, 6), (4, 5), (4, 6), (6, 5))         # e6, e7, e8, e9
end
function reference_faces(::Type{RefPrism})
    return ((1, 3, 2), (1, 2, 5, 4), (3, 1, 4, 6), # f1, f2, f3
            (2, 3, 6, 5), (4, 5, 6))               # f4, f5
end

# RefPyramid (refdim = 3)
reference_vertices(::Type{RefPyramid}) = (1, 2, 3, 4, 5)
function reference_edges(::Type{RefPyramid})
    return ((1, 2), (1, 3), (1, 5), (2, 4), # e1 ... e4
            (2, 5), (4, 3), (3, 5), (4, 5)) # e5 ... e8
end
function reference_faces(::Type{RefPyramid})
    return ((1, 3, 4, 2), (1, 2, 5), (1, 5, 3), # f1, f2, f3
            (2, 4, 5), (3, 5, 4))               # f4, f5
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

A `Grid` is a collection of `Ferrite.AbstractCell`s and `Ferrite.Node`s which covers the computational domain.
Helper structures for applying boundary conditions or define subdomains are gathered in `cellsets`, `nodesets`,
`facetsets`, and `vertexsets`.

# Fields
- `cells::Vector{C}`: stores all cells of the grid
- `nodes::Vector{Node{dim,T}}`: stores the `dim` dimensional nodes of the grid
- `cellsets::Dict{String, OrderedSet{Int}}`: maps a `String` key to an `OrderedSet` of cell ids
- `nodesets::Dict{String, OrderedSet{Int}}`: maps a `String` key to an `OrderedSet` of global node ids
- `facetsets::Dict{String, OrderedSet{FacetIndex}}`: maps a `String` to an `OrderedSet` of `FacetIndex`
- `vertexsets::Dict{String, OrderedSet{VertexIndex}}`: maps a `String` key to an `OrderedSet` of `VertexIndex`
"""
mutable struct Grid{dim,C<:AbstractCell,T<:Real} <: AbstractGrid{dim}
    cells::Vector{C}
    nodes::Vector{Node{dim,T}}
    # Sets
    cellsets::Dict{String,OrderedSet{Int}}
    nodesets::Dict{String,OrderedSet{Int}}
    facetsets::Dict{String,OrderedSet{FacetIndex}}
    vertexsets::Dict{String,OrderedSet{VertexIndex}}
end

function Grid(cells::Vector{C},
              nodes::Vector{Node{dim,T}};
              cellsets::Dict{String, <:AbstractVecOrSet{Int}}=Dict{String,OrderedSet{Int}}(),
              nodesets::Dict{String, <:AbstractVecOrSet{Int}}=Dict{String,OrderedSet{Int}}(),
              facetsets::Dict{String, <:AbstractVecOrSet{FacetIndex}}=Dict{String,OrderedSet{FacetIndex}}(),
              facesets=nothing, # deprecated
              vertexsets::Dict{String, <:AbstractVecOrSet{VertexIndex}}=Dict{String,OrderedSet{VertexIndex}}(),
              boundary_matrix = nothing) where {dim,C,T}
    if facesets !== nothing
        if isempty(facetsets)
            @warn "facesets in Grid is deprecated, use facetsets instead" maxlog=1
            for (key, set) in facesets
                facetsets[key] = OrderedSet(FacetIndex(cellnr, facenr) for (cellnr, facenr) in set)
            end
        else
            error("facesets are deprecated, use only facetsets")
        end
    end
    if boundary_matrix !== nothing
        error("`boundary_matrix` is not part of the Grid anymore and thus not a supported keyword argument.")
    end
    return Grid(
        cells,
        nodes,
        convert_to_orderedsets(cellsets),
        convert_to_orderedsets(nodesets),
        convert_to_orderedsets(facetsets),
        convert_to_orderedsets(vertexsets),
    )
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
    get_reference_dimension(grid::AbstractGrid) -> Union{Int, Symbol}

Get information about the reference dimensions of the cells in the grid.
If all cells have the same reference dimension, `rdim::Int` is returned.
For grids with mixed reference dimensions, `:mixed` is returned.
Used internally to dispatch facet-calls to the correct entity when `rdim isa Int`.
"""
get_reference_dimension(g::AbstractGrid) = _get_reference_dimension(getcells(g))
_get_reference_dimension(::AbstractVector{C}) where C <: AbstractCell{<:AbstractRefShape{rdim}} where rdim = rdim # Fast path for single rdim inferable from eltype
function _get_reference_dimension(cells::AbstractVector{<:AbstractCell})
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
@inline getnodes(grid::AbstractGrid, v::Union{Int,Vector{Int}}) = grid.nodes[v]
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

Returns all cells as cellid in the set with name `setname`.
"""
@inline getcellset(grid::AbstractGrid, setname::String) = grid.cellsets[setname]
"""
    getcellsets(grid::AbstractGrid)

Returns all cellsets of the `grid`.
"""
@inline getcellsets(grid::AbstractGrid) = grid.cellsets

"""
    getnodeset(grid::AbstractGrid, setname::String)

Returns all nodes as nodeid in the set with name `setname`.
"""
@inline getnodeset(grid::AbstractGrid, setname::String) = grid.nodesets[setname]
"""
    getnodesets(grid::AbstractGrid)

Returns all nodesets of the `grid`.
"""
@inline getnodesets(grid::AbstractGrid) = grid.nodesets

"""
    getfacetset(grid::AbstractGrid, setname::String)

Returns all faces as `FacetIndex` in the set with name `setname`.
"""
@inline getfacetset(grid::AbstractGrid, setname::String) = grid.facetsets[setname]
"""
    getfacetsets(grid::AbstractGrid)

Returns all facet sets of the `grid`.
"""
@inline getfacetsets(grid::AbstractGrid) = grid.facetsets


"""
    getvertexset(grid::AbstractGrid, setname::String)

Returns all vertices as `VertexIndex` in the set with name `setname`.
"""
@inline getvertexset(grid::AbstractGrid, setname::String) = grid.vertexsets[setname]
"""
    getvertexsets(grid::AbstractGrid)

Returns all vertex sets of the grid.
"""
@inline getvertexsets(grid::AbstractGrid) = grid.vertexsets

# Transformations
"""
    transform_coordinates!(grid::Abstractgrid, f::Function)

Transform the coordinates of all nodes of the `grid` based on some transformation function `f(x)`.
"""
function transform_coordinates!(g::AbstractGrid, f::Function)
    replace!(n -> Node(f(get_node_coordinate(n))), getnodes(g))
    return g
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
@inline function getcoordinates!(x::Vector{Vec{dim,T}}, grid::AbstractGrid, cell::AbstractCell) where {dim,T}
    node_ids = get_node_ids(cell)
    @inbounds for i in 1:length(x)
        x[i] = get_node_coordinate(grid, node_ids[i])
    end
    return x
end
@inline function getcoordinates!(x::Vector{Vec{dim,T}}, grid::AbstractGrid, cellid::Int) where {dim,T}
    cell = getcells(grid, cellid)
    getcoordinates!(x, grid, cell)
end
@inline getcoordinates!(x::Vector{Vec{dim,T}}, grid::AbstractGrid, cell::CellIndex) where {dim,T} = getcoordinates!(x, grid, cell.idx)

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
        typestrs = sort!(repr.(OrderedSet(typeof(x) for x in grid.cells)))
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
