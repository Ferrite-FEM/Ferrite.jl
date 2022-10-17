#######################################################
#        Geometrical Interface for Elements           #
#######################################################

"""
    AbstractElement{Dim, RefGeo}

Element supertype to describe geometries which are transformable 
to some `Dim`-dimensional reference geometry of type `RefGeo`.
"""
abstract type AbstractElement{Dim, RefGeo} end

"""
    Element{Dim, RefGeo, N} <: AbstractElement{Dim, RefGeo}

Standart data structure to describe nodal finite elements.
* `Dim` refers to the dimension of the reference geometry
* `RefGeo` is the type of reference geometry
* `N` refers to the number of nodes
"""
struct Element{Dim, RefGeo, N} <: AbstractElement{Dim, RefGeo}
    nodes::NTuple{N, Int}

    function Element(ip_geo::InterpolationType) where {InterpolationType <: Interpolation}
        return Element(typeof(ip_geo))
    end

    function Element(ip_geo::Type{<:Interpolation{Dim, RefGeo}}) where {Dim, RefGeo}
        return new{Dim, RefGeo, ndofs(ip_geo)}()
    end

    function Element{Dim, RefGeo, N}() where {Dim, RefGeo, N}
        return new{Dim, RefGeo, N}()
    end

    function Element{Dim, RefGeo, N}(nodes::NTuple{N, Int}) where {Dim, RefGeo, N}
        return new{Dim, RefGeo, N}(nodes)
    end
end

nfaces(c::C) where {C<:AbstractElement} = nfaces(typeof(c))
nfaces(::Type{<:AbstractElement{dim, RefCube}}) where {dim} = 2*dim
nfaces(::Type{<:AbstractElement{dim, RefSimplex}}) where {dim} = (dim+1)

nnodes(c::C) where {C<:AbstractElement} = nnodes(typeof(c))
nnodes(::Type{Element{Dim, RefGeo, N}}) where {Dim, RefGeo, N} = N

getdim(::Element{Dim, RefGeo, N}) where {Dim, RefGeo, N} = Dim
getdim(::Type{Element{Dim, RefGeo, N}}) where {Dim, RefGeo, N} = Dim

"""
A `ElementIndex` wraps an Int and corresponds to a element with that number in the mesh
"""
struct ElementIndex
    idx::Int
end

# Typealias for commonly used elements
# TODO erge with elementtypes...
const PointElement                  = Element{0,RefCube,1}
const LineElement                   = Element{1,RefCube,2}
const QuadraticLineElement          = Element{1,RefCube,3}
const QuadrilateralElement          = Element{2,RefCube,4}
const QuadraticQuadrilateralElement = Element{2,RefCube,9}
const HexahedronElement             = Element{3,RefCube,8}
const TesserractElement             = Element{4,RefCube,16}
const QuadraticHexahedronElement    = Element{3,RefCube,27}
const TriangleElement               = Element{2,RefSimplex,3}
const QuadraticTriangleElement      = Element{2,RefSimplex,6}
const TetrahedronElement            = Element{3,RefSimplex,4}
const QuadraticTetrahedronElement   = Element{3,RefSimplex,10}

const elementtypes = Dict{DataType, String}(Element{1,RefCube,2}     => "Line",
                                            Element{1,RefCube,3}     => "QuadraticLine",
                                            Element{2,RefCube,4}     => "Quadrilateral",
                                            Element{2,RefCube,9}     => "QuadraticQuadrilateral",
                                            Element{3,RefCube,8}     => "Hexahedron",
                                            Element{4,RefCube,16}    => "Tesserract",
                                            Element{3,RefCube,27}    => "QuadraticHexahedron",
                                            Element{3,RefCube,20}    => "SerendipityHexahedron",
                                            Element{2,RefSimplex,3}  => "Triangle",
                                            Element{2,RefSimplex,6}  => "QuadraticTriangle",
                                            Element{3,RefSimplex,4}  => "Tetrahedron",
                                            Element{3,RefSimplex,10} => "QuadraticTetrahedron");

# Helper to always get the correct geometric interpolation
get_geo_interpolation(::Type{LineElement})                    = Lagrange{1, RefCube, 1}()
get_geo_interpolation(::Type{QuadraticLineElement})           = Lagrange{1, RefCube, 2}()
get_geo_interpolation(::Type{QuadrilateralElement})           = Lagrange{2, RefCube, 1}()
get_geo_interpolation(::Type{QuadraticQuadrilateralElement})  = Lagrange{2, RefCube, 2}()
get_geo_interpolation(::Type{HexahedronElement})              = Lagrange{3, RefCube, 1}()
get_geo_interpolation(::Type{QuadraticHexahedronElement})     = Lagrange{3, RefCube, 2}()
get_geo_interpolation(::Type{TesserractElement})              = Lagrange{1, RefCube, 1}()
get_geo_interpolation(::Type{TriangleElement})                = Lagrange{2, RefSimplex, 1}()
get_geo_interpolation(::Type{QuadraticTriangleElement})       = Lagrange{2, RefSimplex, 2}()
get_geo_interpolation(::Type{TetrahedronElement})             = Lagrange{3, RefSimplex, 1}()
get_geo_interpolation(::Type{QuadraticTetrahedronElement})    = Lagrange{3, RefSimplex, 2}()
# Redirection helper
get_geo_interpolation(e::Element{Dim, RefGeo, N}) where {Dim, RefGeo, N} = get_geo_interpolation(typeof(e)) 

# Compat code against old API.
default_interpolation(e::Element{Dim, RefGeo, N}) where {Dim, RefGeo, N} = get_geo_interpolation(e)
default_interpolation(et::Type{Element{Dim, RefGeo, N}}) where {Dim, RefGeo, N} = get_geo_interpolation(et)

# ---------------------------------- Topological Entity Interface --------------------------------
# We start with the dimension-agnostic interface. This table stores the mapping from dim to codim function
const codim_table = [
    [x->(1,),vertices],
    [x->(1,),faces,vertices],
    [x->(1,),faces,edges,vertices],
];
"""
Return the nodes associated to the codimensional portion of an element.
"""
entities_with_codim(element::Element{Dim,RefGeo,N}, codim::Int) where{Dim,RefGeo, N} = codim_table[Dim][codim+1](element)
num_entities_with_codim(element::Element{Dim,RefGeo,N}, codim::Int) where{Dim,RefGeo, N} = length(entities_with_codim(element, codim))

# Functions to uniquely identify vertices, edges and faces, used when distributing
# dofs over a mesh. For this we can ignore the nodes on edged, faces and inside elements,
# we only need to use the nodes that are vertices.
# 1D: vertices
vertices(c::Union{LineElement,QuadraticLineElement}) = (c.nodes[1], c.nodes[2])

# 2D: vertices, faces
vertices(c::Union{TriangleElement,QuadraticTriangleElement}) = (c.nodes[1], c.nodes[2], c.nodes[3])
faces(c::Union{TriangleElement,QuadraticTriangleElement}) = ((c.nodes[1],c.nodes[2]), (c.nodes[2],c.nodes[3]), (c.nodes[3],c.nodes[1]))

vertices(c::Union{QuadrilateralElement,QuadraticQuadrilateralElement}) = (c.nodes[1], c.nodes[2], c.nodes[3], c.nodes[4])
faces(c::Union{QuadrilateralElement,QuadraticQuadrilateralElement}) = ((c.nodes[1],c.nodes[2]), (c.nodes[2],c.nodes[3]), (c.nodes[3],c.nodes[4]), (c.nodes[4],c.nodes[1]))

# 3D: vertices, edges, faces
vertices(c::Union{TetrahedronElement,QuadraticTetrahedronElement}) = (c.nodes[1], c.nodes[2], c.nodes[3], c.nodes[4])
edges(c::Union{TetrahedronElement,QuadraticTetrahedronElement}) = ((c.nodes[1],c.nodes[2]), (c.nodes[2],c.nodes[3]), (c.nodes[3],c.nodes[1]), (c.nodes[1],c.nodes[4]), (c.nodes[2],c.nodes[4]), (c.nodes[3],c.nodes[4]))
faces(c::Union{TetrahedronElement,QuadraticTetrahedronElement}) = ((c.nodes[1],c.nodes[3],c.nodes[2]), (c.nodes[1],c.nodes[2],c.nodes[4]), (c.nodes[2],c.nodes[3],c.nodes[4]), (c.nodes[1],c.nodes[4],c.nodes[3]))

vertices(c::Union{HexahedronElement,Element{3,RefCube,20}}) = (c.nodes[1], c.nodes[2], c.nodes[3], c.nodes[4], c.nodes[5], c.nodes[6], c.nodes[7], c.nodes[8])
edges(c::Union{HexahedronElement,Element{3,RefCube,20}}) = ((c.nodes[1],c.nodes[2]), (c.nodes[2],c.nodes[3]), (c.nodes[3],c.nodes[4]), (c.nodes[4],c.nodes[1]), (c.nodes[5],c.nodes[6]), (c.nodes[6],c.nodes[7]), (c.nodes[7],c.nodes[8]), (c.nodes[8],c.nodes[5]), (c.nodes[1],c.nodes[5]), (c.nodes[2],c.nodes[6]), (c.nodes[3],c.nodes[7]), (c.nodes[4],c.nodes[8]))
faces(c::Union{HexahedronElement,Element{3,RefCube,20}}) = ((c.nodes[1],c.nodes[4],c.nodes[3],c.nodes[2]), (c.nodes[1],c.nodes[2],c.nodes[6],c.nodes[5]), (c.nodes[2],c.nodes[3],c.nodes[7],c.nodes[6]), (c.nodes[3],c.nodes[4],c.nodes[8],c.nodes[7]), (c.nodes[1],c.nodes[5],c.nodes[8],c.nodes[4]), (c.nodes[5],c.nodes[6],c.nodes[7],c.nodes[8]))

# Cubes
getbasegeometry(::Type{Element{1, RefCube, N}}) where {N} = LineElement 
getbasegeometry(::Type{Element{2, RefCube, N}}) where {N} = QuadrilateralElement
getbasegeometry(::Type{Element{3, RefCube, N}}) where {N} = HexahedronElement
getbasegeometry(::Type{Element{4, RefCube, N}}) where {N} = TesserractElement
# Simplices
getbasegeometry(::Type{Element{1, RefSimplex, N}}) where {N} = LineElement
getbasegeometry(::Type{Element{2, RefSimplex, N}}) where {N} = TriangleElement
getbasegeometry(::Type{Element{3, RefSimplex, N}}) where {N} = TetrahedronElement
# Redirection helper
getbasegeometry(e::Element{Dim, RefGeo, N}) where {Dim, RefGeo, N} = getbasegeometry(typeof(e))
