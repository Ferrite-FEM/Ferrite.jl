"""
    facet_to_element_transformation(point::Vec, ::Type{<:AbstractRefShape}, facet::Int)

Transform quadrature point from the facet's reference coordinates to coordinates on the
cell's facet, increasing the number of dimensions by one.
"""
facet_to_element_transformation

"""
    element_to_facet_transformation(point::AbstractVector, ::Type{<:AbstractRefShape}, facet::Int)

Transform quadrature point from the cell's coordinates to the facet's reference coordinates, decreasing the number of dimensions by one.
This is the inverse of `facet_to_element_transformation`.
"""
element_to_facet_transformation

"""
    weighted_normal(J::AbstractTensor, fv::FacetValues, face::Int)
    weighted_normal(J::AbstractTensor, ::Type{<:AbstractRefShape}, face::Int)

Compute the vector normal to the facet weighted by the area ratio between the facet and the
reference facet. This is computed by taking the cross product of the Jacobian components that
align to the facet's local axis.
"""
function weighted_normal end

"""
    create_facet_quad_rule(::Type{RefShape}, w::Vector{T}, p::Vector{Vec{N, T}})
    create_facet_quad_rule(
        ::Type{RefShape},
        quad_faces::Vector{Int}, w_quad::Vector{T}, p_quad::Vector{Vec{N, T}},
        tri_faces::Vector{Int}, w_tri::Vector{T}, p_tri::Vector{Vec{N, T}}
    )

Create a ["FacetQuadratureRule"](@ref) for the given cell type, weights and points. If the
cell has facets of different shapes (i.e. quadrilaterals and triangles) then each shape's
facets indices, weights and points are passed separately.
"""
function create_facet_quad_rule(::Type{RefShape}, w::Vector{T}, p::Vector{Vec{N, T}}) where {N, T, RefShape <: AbstractRefShape}
    facet_quad_rule = QuadratureRule{RefShape, T, getdim(AbstractCell{RefShape})}[]
    for facet in 1:nfacets(RefShape)
        new_points = [facet_to_element_transformation(p[i], RefShape, facet) for i in 1:length(w)]
        push!(facet_quad_rule, QuadratureRule{RefShape, T}(w, new_points))
    end
    return FacetQuadratureRule(facet_quad_rule)
end

# For cells with mixed faces
function create_facet_quad_rule(
    ::Type{RefShape},
    quad_facets::Vector{Int}, w_quad::Vector{T}, p_quad::Vector{Vec{N, T}},
    tri_facets::Vector{Int}, w_tri::Vector{T}, p_tri::Vector{Vec{N, T}}
) where {N, T, RefShape <: Union{RefPrism, RefPyramid}}
    facet_quad_rule = Vector{QuadratureRule{RefShape, T, getdim(AbstractCell{RefShape})}}(undef, nfacets(RefShape))
    for facet in quad_facets
        new_points = [facet_to_element_transformation(p_quad[i], RefShape, facet) for i in 1:length(w_quad)]
        facet_quad_rule[facet] = QuadratureRule{RefShape, T}(w_quad, new_points)
    end
    for facet in tri_facets
        new_points = [facet_to_element_transformation(p_tri[i], RefShape, facet) for i in 1:length(w_tri)]
        facet_quad_rule[facet] = QuadratureRule{RefShape, T}(w_tri, new_points)
    end
    return FacetQuadratureRule(facet_quad_rule)
end

##################
# All 1D RefLine #
##################

# Mapping from to 0D node to 1D line vertex.
function facet_to_element_transformation(::Union{Vec{0, T},Vec{1, T}}, ::Type{RefLine}, face::Int) where {T}
    face == 1 && return Vec{1, T}(( -one(T),))
    face == 2 && return Vec{1, T}((  one(T),))
    throw(ArgumentError("unknown facet number"))
end

# Mapping from 1D line to point.
function element_to_facet_transformation(point::Vec{1, T}, ::Type{RefLine}, face::Int) where T
    x = point[]
    face == 1 && return Vec(-x)
    face == 2 && return Vec( x)
    throw(ArgumentError("unknown facet number"))
end

function weighted_normal(::Tensor{2,1,T}, ::Type{RefLine}, face::Int) where {T}
    face == 1 && return Vec{1,T}((-one(T),))
    face == 2 && return Vec{1,T}(( one(T),))
    throw(ArgumentError("unknown facet number"))
end

###########################
# All 2D RefQuadrilateral #
###########################

# Mapping from 1D line to 2D face of a quadrilateral.
function facet_to_element_transformation(point::Vec{1, T}, ::Type{RefQuadrilateral}, face::Int) where T
    x = point[1]
    face == 1 && return Vec{2, T}(( x,          -one(T)))
    face == 2 && return Vec{2, T}(( one(T),     x))
    face == 3 && return Vec{2, T}(( -x,         one(T)))
    face == 4 && return Vec{2, T}(( -one(T),    -x))
    throw(ArgumentError("unknown facet number"))
end

# Mapping from 2D face of a quadrilateral to 1D line.
function element_to_facet_transformation(point::Vec{2, T}, ::Type{RefQuadrilateral}, face::Int) where T
    x, y = point
    face == 1 && return Vec( x)
    face == 2 && return Vec( y)
    face == 3 && return Vec( -x)
    face == 4 && return Vec( -y)
    throw(ArgumentError("unknown facet number"))
end

function weighted_normal(J::Tensor{2,2}, ::Type{RefQuadrilateral}, face::Int)
    @inbounds begin
        face == 1 && return Vec{2}(( J[2,1], -J[1,1]))
        face == 2 && return Vec{2}(( J[2,2], -J[1,2]))
        face == 3 && return Vec{2}((-J[2,1],  J[1,1]))
        face == 4 && return Vec{2}((-J[2,2],  J[1,2]))
    end
    throw(ArgumentError("unknown facet number"))
end

######################
# All RefTriangle 2D #
######################

# Mapping from 1D line to 2D face of a triangle.
function facet_to_element_transformation(point::Vec{1, T},  ::Type{RefTriangle}, face::Int) where T
    x = (point[1] + one(T)) / 2
    face == 1 && return Vec{2, T}(( one(T) - x,     x ))
    face == 2 && return Vec{2, T}(( zero(T),        one(T) -x))
    face == 3 && return Vec{2, T}(( x,              zero(T)))
    throw(ArgumentError("unknown facet number"))
end

# Mapping from 2D face of a triangle to 1D line.
function element_to_facet_transformation(point::Vec{2, T}, ::Type{RefTriangle}, face::Int) where T
    x, y = point
    face == 1 && return Vec( one(T) - x * 2)
    face == 2 && return Vec( one(T) - y * 2 )
    face == 3 && return Vec( x * 2 - one(T))
    throw(ArgumentError("unknown facet number"))
end

function weighted_normal(J::Tensor{2,2}, ::Type{RefTriangle}, face::Int)
    @inbounds begin
        face == 1 && return Vec{2}((-(J[2,1] - J[2,2]), J[1,1] - J[1,2]))
        face == 2 && return Vec{2}((-J[2,2], J[1,2]))
        face == 3 && return Vec{2}((J[2,1], -J[1,1]))
    end
    throw(ArgumentError("unknown facet number"))
end

########################
# All RefHexahedron 3D #
########################

# Mapping from 2D quadrilateral to 3D face of a hexahedron.
function facet_to_element_transformation(point::Vec{2, T}, ::Type{RefHexahedron}, face::Int) where T
    x, y = point
    face == 1 && return Vec{3, T}(( y,      x,          -one(T)))
    face == 2 && return Vec{3, T}(( x,      -one(T),    y))
    face == 3 && return Vec{3, T}(( one(T), x,          y))
    face == 4 && return Vec{3, T}(( -x,     one(T),     y))
    face == 5 && return Vec{3, T}((-one(T), y,          x))
    face == 6 && return Vec{3, T}(( x,      y,          one(T)))
    throw(ArgumentError("unknown facet number"))
end

# Mapping from 3D face of a hexahedron to 2D quadrilateral.
function element_to_facet_transformation(point::Vec{3, T}, ::Type{RefHexahedron}, face::Int) where T
    x, y, z = point
    face == 1 && return Vec( y, x)
    face == 2 && return Vec( x, z)
    face == 3 && return Vec( y,  z)
    face == 4 && return Vec( -x,  z)
    face == 5 && return Vec( z,  y)
    face == 6 && return Vec( x,  y)
    throw(ArgumentError("unknown facet number"))
end

function weighted_normal(J::Tensor{2,3}, ::Type{RefHexahedron}, face::Int)
    @inbounds begin
        face == 1 && return J[:,2] × J[:,1]
        face == 2 && return J[:,1] × J[:,3]
        face == 3 && return J[:,2] × J[:,3]
        face == 4 && return J[:,3] × J[:,1]
        face == 5 && return J[:,3] × J[:,2]
        face == 6 && return J[:,1] × J[:,2]
    end
    throw(ArgumentError("unknown facet number"))
end

#########################
# All RefTetrahedron 3D #
#########################

# Mapping from 2D triangle to 3D face of a tetrahedon.
function facet_to_element_transformation(point::Vec{2, T}, ::Type{RefTetrahedron}, face::Int) where T
    x, y = point
    face == 1 && return Vec{3, T}( (one(T)-x-y,     y,              zero(T)))
    face == 2 && return Vec{3, T}( (y,              zero(T),        one(T)-x-y))
    face == 3 && return Vec{3, T}( (x,              y,              one(T)-x-y))
    face == 4 && return Vec{3, T}( (zero(T),        one(T)-x-y,     y))
    throw(ArgumentError("unknown facet number"))
end

# Mapping from 3D face of a tetrahedon to 2D triangle.
function element_to_facet_transformation(point::Vec{3, T}, ::Type{RefTetrahedron}, face::Int) where T
    x, y, z = point
    face == 1 && return Vec( one(T)-x-y,  y)
    face == 2 && return Vec( one(T)-z-x,  x)
    face == 3 && return Vec( x,  y)
    face == 4 && return Vec( one(T)-y-z,  z)
    throw(ArgumentError("unknown facet number"))
end

function weighted_normal(J::Tensor{2,3}, ::Type{RefTetrahedron}, face::Int)
    @inbounds begin
        face == 1 && return J[:,2] × J[:,1]
        face == 2 && return J[:,1] × J[:,3]
        face == 3 && return (J[:,1]-J[:,3]) × (J[:,2]-J[:,3])
        face == 4 && return J[:,3] × J[:,2]
    end
    throw(ArgumentError("unknown facet number"))
end

###################
# All RefPrism 3D #
###################

# Mapping from 2D quadrilateral/triangle to 3D face of a wedge.
function facet_to_element_transformation(point::Vec{2, T}, ::Type{RefPrism}, face::Int) where T
    # Note that for quadrilaterals the domain is [-1, 1]² but for triangles it is [0, 1]²
    x, y = point
    face == 1 && return Vec{3, T}(( one(T)-x-y,             y,                      zero(T)))
    face == 2 && return Vec{3, T}(( (one(T)+x)/2,           zero(T),                (one(T)+y)/2))
    face == 3 && return Vec{3, T}(( zero(T),                one(T)-(one(T)+x)/2,    (one(T)+y)/2))
    face == 4 && return Vec{3, T}(( one(T)-(one(T)+x)/2,   (one(T)+x)/2,            (one(T)+y)/2))
    face == 5 && return Vec{3, T}(( y,                      one(T)-x-y,             one(T)))
    throw(ArgumentError("unknown facet number"))
end

# Mapping from 3D face of a wedge to 2D triangle or 2D quadrilateral.
function element_to_facet_transformation(point::Vec{3, T}, ::Type{RefPrism}, face::Int) where T
    x, y, z = point
    face == 1 && return Vec( one(T)-x-y,  y)
    face == 2 && return Vec( 2*x - one(T), 2*z - one(T) )
    face == 3 && return Vec( 2*(one(T) - y) - one(T), 2*z - one(T) )
    face == 4 && return Vec( 2*y - one(T), 2*z - one(T) )
    face == 5 && return Vec( one(T) - x - y,  x)
    throw(ArgumentError("unknown facet number"))
end

function weighted_normal(J::Tensor{2,3}, ::Type{RefPrism}, face::Int)
    @inbounds begin
        face == 1 && return J[:,2] × J[:,1]
        face == 2 && return J[:,1] × J[:,3]
        face == 3 && return J[:,3] × J[:,2]
        face == 4 && return (J[:,2]-J[:,1]) × J[:,3]
        face == 5 && return J[:,1] × J[:,2]
    end
    throw(ArgumentError("unknown facet number"))
end

#####################
# All RefPyramid 3D #
#####################

# Mapping from 2D face to 3D face of a pyramid.
function facet_to_element_transformation(point::Vec{2, T}, ::Type{RefPyramid}, face::Int) where T
    x, y = point
    face == 1 && return Vec{3, T}(( (y+one(T))/2,   (x+one(T))/2,       zero(T)))
    face == 2 && return Vec{3, T}(( y,              zero(T),            one(T)-x-y))
    face == 3 && return Vec{3, T}(( zero(T),        one(T)-x-y,         y))
    face == 4 && return Vec{3, T}(( x+y,            y,                  one(T)-x-y))
    face == 5 && return Vec{3, T}(( one(T)-x-y,     one(T)-y,           y))
    throw(ArgumentError("unknown facet number"))
end

# Mapping from 3D face of a pyramid to 2D triangle or 2D quadrilateral.
function element_to_facet_transformation(point::Vec{3, T}, ::Type{RefPyramid}, face::Int) where T
    x, y, z = point
    face == 1 && return Vec( 2*y - one(T),  2*x - one(T))
    face == 2 && return Vec( one(T) - z - x, x)
    face == 3 && return Vec( one(T) - y - z, z)
    face == 4 && return Vec( x - y, y)
    face == 5 && return Vec( one(T) - x - z,  z)
    throw(ArgumentError("unknown facet number"))
end

function weighted_normal(J::Tensor{2,3}, ::Type{RefPyramid}, face::Int)
    @inbounds begin
        face == 1 && return J[:,2] × J[:,1]
        face == 2 && return J[:,1] × J[:,3]
        face == 3 && return J[:,3] × J[:,2]
        face == 4 && return J[:,2] × (J[:,3]-J[:,1])
        face == 5 && return (J[:,3]-J[:,2]) × J[:,1]
    end
    throw(ArgumentError("unknown facet number"))
end
