#########################################################
#           Geometrical Interface for Meshes            #
#########################################################

abstract type AbstractMesh{sdim} <: AbstractGrid{sdim} end

"""
    Mesh{dim, C<:AbstractElement, T<:Real} <: AbstractMesh}

A `Mesh` is a collection of `Elements` and `Nodes` which covers the computational domain.

# Fields
- `elements::Vector{C}`: stores all elements of the mesh
- `nodes::Vector{Node{dim,T}}`: stores the `dim` dimensional nodes of the mesh
- `elementsets::Dict{String,Set{ElementIndex}}`
- `nodesets::Dict{String,Set{Int}}`
- `facesets::Dict{String,Set{FaceIndex}}`
"""
# TODO Introduce "node type"
# TODO Sets via element dimensionality
mutable struct Mesh{sdim,ElementType<:AbstractElement,T<:Real} <: AbstractMesh{sdim}
    elements::Vector{ElementType}
    nodes::Vector{Node{sdim,T}}

    # Sets
    elementsets::Dict{String,Set{ElementIndex}}
    nodesets::Dict{String,Set{Int}}
    facesets::Dict{String,Set{FaceIndex}}
    edgesets::Dict{String,Set{EdgeIndex}}
    vertexsets::Dict{String,Set{VertexIndex}}
end

function Mesh(elements::Vector{C},
              nodes::Vector{Node{sdim,T}};
              elementsets::Dict{String,Set{ElementIndex}}=Dict{String,Set{ElementIndex}}(),
              nodesets::Dict{String,Set{Int}}=Dict{String,Set{Int}}(),
              facesets::Dict{String,Set{FaceIndex}}=Dict{String,Set{FaceIndex}}(),
              edgesets::Dict{String,Set{EdgeIndex}}=Dict{String,Set{EdgeIndex}}(),
              vertexsets::Dict{String,Set{VertexIndex}}=Dict{String,Set{VertexIndex}}()
              ) where {sdim,C,T}
    return Mesh(elements, nodes, elementsets, nodesets, facesets, edgesets, vertexsets)
end

toglobal(mesh::Mesh, vertexidx::VertexIndex) = vertices(getelements(mesh,vertexidx[1]))[vertexidx[2]]
toglobal(mesh::Mesh, vertexidx::Vector{VertexIndex}) = unique(toglobal.((mesh,),vertexidx))

# Compat helper
@inline getcelltype(mesh::AbstractMesh) = getelementtype(mesh)
@inline getcelltype(mesh::AbstractMesh, i::Int) = getelementtype(mesh, i)
@inline getcells(mesh::AbstractMesh) = getelements(mesh)
@inline getcells(mesh::AbstractMesh, v::Union{Int, Vector{Int}}) = getelements(mesh, v)
@inline getncells(mesh::AbstractMesh) = getnelements(mesh)
@inline getelementtypes(mesh::Mesh{sdim, ElementType, T}) where {sdim, ElementType, T} = ElementType
@inline nnodes_per_cell(dh::AbstractMesh, i::Int=1) = nnodes_per_element(dh, i)

"""
Get the spatial dimension of the mesh.
"""
@inline getdim(::AbstractMesh{sdim}) where {sdim} = sdim
"""
    getelements(mesh::AbstractMesh)
    getelements(mesh::AbstractMesh, v::Union{Int,Vector{Int}}
    getelements(mesh::AbstractMesh, setname::String)

Returns either all `elements::Collection{C<:AbstractElement}` of a `<:AbstractMesh` or a subset based on an `Int`, `Vector{Int}` or `String`.
Whereas the last option tries to call a `elementset` of the `mesh`. `Collection` can be any indexable type, for `Mesh` it is `Vector{C<:AbstractElement}`.
"""
@inline getelements(mesh::AbstractMesh) = mesh.elements
@inline getelements(mesh::AbstractMesh, v::Union{Int, Vector{Int}}) = mesh.elements[v]
@inline getelements(mesh::AbstractMesh, setname::String) = mesh.elements[collect(getelementset(mesh,setname))]
"Returns the number of elements in the `<:AbstractMesh`."
@inline getnelements(mesh::AbstractMesh) = length(mesh.elements)
"Returns the elementtype of the `<:AbstractMesh`."
@inline getelementtype(mesh::AbstractMesh) = eltype(mesh.elements)
@inline getelementtype(mesh::AbstractMesh, i::Int) = typeof(mesh.elements[i])

"""
    getnodes(mesh::AbstractMesh)
    getnodes(mesh::AbstractMesh, v::Union{Int,Vector{Int}}
    getnodes(mesh::AbstractMesh, setname::String)

Returns either all `nodes::Collection{N}` of a `<:AbstractMesh` or a subset based on an `Int`, `Vector{Int}` or `String`.
The last option tries to call a `nodeset` of the `<:AbstractMesh`. `Collection{N}` refers to some indexable collection where each element corresponds
to a Node.
"""
@inline getnodes(mesh::AbstractMesh) = mesh.nodes
@inline getnodes(mesh::AbstractMesh, v::Union{Int, Vector{Int}}) = mesh.nodes[v]
@inline getnode(mesh::AbstractMesh, v::Int) = mesh.nodes[v]
@inline getnodes(mesh::AbstractMesh, setname::String) = mesh.nodes[collect(getnodeset(mesh,setname))]
"Returns the number of nodes in the mesh."
@inline getnnodes(mesh::AbstractMesh) = length(mesh.nodes)
"Returns the number of nodes of the `i`-th element."
@inline nnodes_per_element(mesh::AbstractMesh, i::Int) = nnodes(mesh.elements[i])

"""
    getelementset(mesh::AbstractMesh, setname::String)

Returns all elements as elementid in a `Set` of a given `setname`.
"""
@inline getelementset(mesh::AbstractMesh, setname::String) = mesh.elementsets[setname]
"""
    getelementsets(mesh::AbstractMesh)

Returns all elementsets of the `mesh`.
"""
@inline getelementsets(mesh::AbstractMesh) = mesh.elementsets

"""
    getnodeset(mesh::AbstractMesh, setname::String)

Returns all nodes as nodeid in a `Set` of a given `setname`.
"""
@inline getnodeset(mesh::AbstractMesh, setname::String) = mesh.nodesets[setname]
"""
    getnodesets(mesh::AbstractMesh)

Returns all nodesets of the `mesh`.
"""
@inline getnodesets(mesh::AbstractMesh) = mesh.nodesets

"""
    getfaceset(mesh::AbstractMesh, setname::String)

Returns all faces as `FaceIndex` in a `Set` of a given `setname`.
"""
@inline getfaceset(mesh::AbstractMesh, setname::String) = mesh.facesets[setname]
"""
    getfacesets(mesh::AbstractMesh)

Returns all facesets of the `mesh`.
"""
@inline getfacesets(mesh::AbstractMesh) = mesh.facesets

"""
    addelementset!(mesh::AbstractMesh, name::String, elementid::Union{Set{Int}, Vector{Int}})
    addelementset!(mesh::AbstractMesh, name::String, f::function; all::Bool=true)

Adds a elementset to the mesh with key `name`.
Elementsets are typically used to define subdomains of the problem, e.g. two materials in the computational domain.
The `MixedDofHandler` can construct different fields which live not on the whole domain, but rather on a elementset.

```julia
addelementset!(mesh, "left", Set((1,3))) #add elements with id 1 and 3 to elementset left
addelementset!(mesh, "right", x -> norm(x[1]) < 2.0 ) #add element to elementset right, if x[1] of each element's node is smaller than 2.0
```
"""
function addelementset!(mesh::AbstractMesh, name::String, elementid::Union{Set{Int},Vector{Int}})
    _check_setname(mesh.elementsets,  name)
    elements = Set(elementid)
    _warn_emptyset(elements, name)
    mesh.elementsets[name] = elements
    mesh
end

function addelementset!(mesh::AbstractMesh, name::String, f::Function; all::Bool=true)
    _check_setname(mesh.elementsets, name)
    elements = Set{Int}()
    for (i, element) in enumerate(getelements(mesh))
        pass = all
        for node_idx in element.nodes
            node = mesh.nodes[node_idx]
            v = f(node.x)
            all ? (!v && (pass = false; break)) : (v && (pass = true; break))
        end
        pass && push!(elements, i)
    end
    _warn_emptyset(elements, name)
    mesh.elementsets[name] = elements
    mesh
end

"""
    addfaceset!(mesh::AbstractMesh, name::String, faceid::Union{Set{FaceIndex},Vector{FaceIndex}})
    addfaceset!(mesh::AbstractMesh, name::String, f::Function; all::Bool=true)

Adds a faceset to the mesh with key `name`.
A faceset maps a `String` key to a `Set` of tuples corresponding to `(global_element_id, local_face_id)`.
Facesets are used to initialize `Dirichlet` structs, that are needed to specify the boundary for the `ConstraintHandler`.

```julia
addfaceset!(gird, "right", Set(((2,2),(4,2))) #see mesh manual example for reference
addfaceset!(mesh, "clamped", x -> norm(x[1]) ≈ 0.0) #see incompressible elasticity example for reference
```
"""
addfaceset!(mesh::Mesh, name::String, set::Union{Set{FaceIndex},Vector{FaceIndex}}) =
    _addset!(mesh, name, set, mesh.facesets)
function _addset!(mesh::AbstractMesh, name::String, _set, dict::Dict)
    _check_setname(dict, name)
    set = Set(_set)
    _warn_emptyset(set, name)
    dict[name] = set
    mesh
end

addfaceset!(mesh::AbstractMesh, name::String, f::Function; all::Bool=true) =
    _addset!(mesh, name, f, Ferrite.faces, mesh.facesets, FaceIndex; all=all)
function _addset!(mesh::AbstractMesh, name::String, f::Function, _ftype::Function, dict::Dict, _indextype::Type; all::Bool=true)
    _check_setname(dict, name)
    _set = Set{_indextype}()
    for (element_idx, element) in enumerate(getelements(mesh))
        for (face_idx, face) in enumerate(_ftype(element))
            pass = all
            for node_idx in face
                v = f(mesh.nodes[node_idx].x)
                all ? (!v && (pass = false; break)) : (v && (pass = true; break))
            end
            pass && push!(_set, _indextype(element_idx, face_idx))
        end
    end
    _warn_emptyset(_set, name)
    dict[name] = _set
    mesh
end

"""
    addnodeset!(mesh::AbstractMesh, name::String, nodeid::Union{Vector{Int},Set{Int}})
    addnodeset!(mesh::AbstractMesh, name::String, f::Function)

Adds a `nodeset::Dict{String, Set{Int}}` to the `mesh` with key `name`. Has the same interface as `addelementset`.
However, instead of mapping a element id to the `String` key, a set of node ids is returned.
"""
function addnodeset!(mesh::AbstractMesh, name::String, nodeid::Union{Vector{Int},Set{Int}})
    _check_setname(mesh.nodesets, name)
    mesh.nodesets[name] = Set(nodeid)
    _warn_emptyset(mesh.nodesets[name], name)
    mesh
end

function addnodeset!(mesh::AbstractMesh, name::String, f::Function)
    _check_setname(mesh.nodesets, name)
    nodes = Set{Int}()
    for (i, n) in enumerate(getnodes(mesh))
        f(n.x) && push!(nodes, i)
    end
    mesh.nodesets[name] = nodes
    _warn_emptyset(mesh.nodesets[name], name)
    mesh
end

"""
    getcoordinates!(x::Vector{Vec{dim,T}}, mesh::AbstractMesh, element::Int)
Fills the vector `x` with the coordinates of a element, defined by its element id.
"""
@inline function getcoordinates!(x::Vector{Vec{dim,T}}, mesh::AbstractMesh, element_idx::Int) where {dim,T}
    element = getelements(mesh, element_idx)
    @inbounds for i in 1:length(x)
        node_idx = element.nodes[i]
        x[i] = getcoordinates(mesh.nodes[node_idx])
    end
end
@inline getcoordinates!(x::Vector{Vec{dim,T}}, mesh::AbstractMesh, element::ElementIndex) where {dim, T} = getcoordinates!(x, mesh, element.idx)
@inline getcoordinates!(x::Vector{Vec{dim,T}}, mesh::AbstractMesh, face::FaceIndex) where {dim, T} = getcoordinates!(x, mesh, face.idx[1])

"""
    getcoordinates(mesh::AbstractMesh, element)
Return a vector with the coordinates of the vertices of element number `element`.
"""
@inline function getcoordinates(mesh::AbstractMesh, element::Int)
    # TODO pretty ugly, worth it?
    dim = typeof(mesh.elements[element]).parameters[1]
    T = typeof(mesh).parameters[3]
    nodeidx = mesh.elements[element].nodes
    return [mesh.nodes[i].x for i in nodeidx]::Vector{Vec{dim,T}}
end
@inline getcoordinates(mesh::AbstractMesh, element::ElementIndex) = getcoordinates(mesh, element.idx)
@inline getcoordinates(mesh::AbstractMesh, face::FaceIndex) = getcoordinates(mesh, face.idx[1])

function Base.show(io::IO, ::MIME"text/plain", mesh::Mesh)
    print(io, "$(typeof(mesh)) with $(getnelements(mesh)) ")
    typestrs = sort!(collect(Set(elementtypes[typeof(x)] for x in mesh.elements)))
    str = join(io, typestrs, '/')
    print(io, " elements and $(getnnodes(mesh)) nodes")
end
