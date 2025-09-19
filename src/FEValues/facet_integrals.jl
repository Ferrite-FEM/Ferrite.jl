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
    weighted_normal(J::AbstractTensor, fv::FacetValues, facet::Int)
    weighted_normal(J::AbstractTensor, ::Type{<:AbstractRefShape}, facet::Int)

Compute the vector normal to the facet weighted by the area ratio between the facet and the
reference facet. This is computed by taking the cross product of the Jacobian components that
align to the facet's local axis.
"""
function weighted_normal end

"""
    create_facet_quad_rule(::Type{RefShape}, w::AbstractVectorä{T}, p::AbstractVectorä{Vec{N, T}})
    create_facet_quad_rule(
        ::Type{RefShape},
        quad_faces::AbstractVectorä{Int}, w_quad::AbstractVector{T}, p_quad::AbstractVector{Vec{N, T}},
        tri_faces::AbstractVector{Int}, w_tri::AbstractVector{T}, p_tri::AbstractVector{Vec{N, T}}
    )

Create a ["FacetQuadratureRule"](@ref) for the given cell type, weights and points. If the
cell has facets of different shapes (i.e. quadrilaterals and triangles) then each shape's
facets indices, weights and points are passed separately.
"""
function create_facet_quad_rule(::Type{RefShape}, w::AbstractVector{T}, p::AbstractVector{Vec{N, T}}) where {N, T, RefShape <: AbstractRefShape}
    facet_quad_rule = QuadratureRule{RefShape, Vector{T}, Vector{Vec{N + 1, T}}}[]
    for facet in 1:nfacets(RefShape)
        new_points = [facet_to_element_transformation(p[i], RefShape, facet) for i in 1:length(w)]
        push!(facet_quad_rule, QuadratureRule{RefShape}(copy(w), new_points))
    end
    return FacetQuadratureRule(facet_quad_rule)
end

# For cells with mixed faces
function create_facet_quad_rule(
        ::Type{RefShape},
        quad_facets::AbstractVector{Int}, w_quad::AbstractVector{T}, p_quad::AbstractVector{Vec{N, T}},
        tri_facets::AbstractVector{Int}, w_tri::AbstractVector{T}, p_tri::AbstractVector{Vec{N, T}}
    ) where {N, T, RefShape <: Union{RefPrism, RefPyramid}}
    facet_quad_rule = Vector{QuadratureRule{RefShape, Vector{T}, Vector{Vec{N + 1, T}}}}(undef, nfacets(RefShape))
    for facet in quad_facets
        new_points = [facet_to_element_transformation(p_quad[i], RefShape, facet) for i in 1:length(w_quad)]
        facet_quad_rule[facet] = QuadratureRule{RefShape}(copy(w_quad), new_points)
    end
    for facet in tri_facets
        new_points = [facet_to_element_transformation(p_tri[i], RefShape, facet) for i in 1:length(w_tri)]
        facet_quad_rule[facet] = QuadratureRule{RefShape}(copy(w_tri), new_points)
    end
    return FacetQuadratureRule(facet_quad_rule)
end

##################
# All 1D RefLine #
##################

# Mapping from to 0D node to 1D line vertex.
function facet_to_element_transformation(::Union{Vec{0, T}, Vec{1, T}}, ::Type{RefLine}, facet::Int) where {T}
    facet == 1 && return Vec{1, T}((-one(T),))
    facet == 2 && return Vec{1, T}((one(T),))
    throw(ArgumentError("unknown facet number"))
end

# Mapping from 1D line to point.
function element_to_facet_transformation(point::Vec{1, T}, ::Type{RefLine}, facet::Int) where {T}
    x = point[]
    facet == 1 && return Vec(-x)
    facet == 2 && return Vec(x)
    throw(ArgumentError("unknown facet number"))
end

function weighted_normal(::Tensor{2, 1, T}, ::Type{RefLine}, facet::Int) where {T}
    facet == 1 && return Vec{1, T}((-one(T),))
    facet == 2 && return Vec{1, T}((one(T),))
    throw(ArgumentError("unknown facet number"))
end

# Embedded lines in 2D space
function weighted_normal(J::SMatrix{2, 1, T}, ::Type{RefLine}, facet::Int) where {T}
    facet == 1 && return Vec{2, T}((J[2], -J[1]))
    facet == 2 && return Vec{2, T}((-J[2], J[1]))
    throw(ArgumentError("unknown facet number"))
end

###########################
# All 2D RefQuadrilateral #
###########################

# Mapping from 1D line to 2D face of a quadrilateral.
function facet_to_element_transformation(point::Vec{1, T}, ::Type{RefQuadrilateral}, facet::Int) where {T}
    x = point[1]
    facet == 1 && return Vec{2, T}((x, -one(T)))
    facet == 2 && return Vec{2, T}((one(T), x))
    facet == 3 && return Vec{2, T}((-x, one(T)))
    facet == 4 && return Vec{2, T}((-one(T), -x))
    throw(ArgumentError("unknown facet number"))
end

# Mapping from 2D face of a quadrilateral to 1D line.
function element_to_facet_transformation(point::Vec{2, T}, ::Type{RefQuadrilateral}, facet::Int) where {T}
    x, y = point
    facet == 1 && return Vec(x)
    facet == 2 && return Vec(y)
    facet == 3 && return Vec(-x)
    facet == 4 && return Vec(-y)
    throw(ArgumentError("unknown facet number"))
end

function weighted_normal(J::Tensor{2, 2}, ::Type{RefQuadrilateral}, facet::Int)
    @inbounds begin
        facet == 1 && return Vec{2}((J[2, 1], -J[1, 1]))
        facet == 2 && return Vec{2}((J[2, 2], -J[1, 2]))
        facet == 3 && return Vec{2}((-J[2, 1], J[1, 1]))
        facet == 4 && return Vec{2}((-J[2, 2], J[1, 2]))
    end
    throw(ArgumentError("unknown facet number"))
end

######################
# All RefTriangle 2D #
######################

# Mapping from 1D line to 2D face of a triangle.
function facet_to_element_transformation(point::Vec{1, T}, ::Type{RefTriangle}, facet::Int) where {T}
    x = (point[1] + one(T)) / 2
    facet == 1 && return Vec{2, T}((one(T) - x, x))
    facet == 2 && return Vec{2, T}((zero(T), one(T) - x))
    facet == 3 && return Vec{2, T}((x, zero(T)))
    throw(ArgumentError("unknown facet number"))
end

# Mapping from 2D face of a triangle to 1D line.
function element_to_facet_transformation(point::Vec{2, T}, ::Type{RefTriangle}, facet::Int) where {T}
    x, y = point
    facet == 1 && return Vec(one(T) - x * 2)
    facet == 2 && return Vec(one(T) - y * 2)
    facet == 3 && return Vec(x * 2 - one(T))
    throw(ArgumentError("unknown facet number"))
end

function weighted_normal(J::Tensor{2, 2}, ::Type{RefTriangle}, facet::Int)
    @inbounds begin
        facet == 1 && return Vec{2}((-(J[2, 1] - J[2, 2]), J[1, 1] - J[1, 2]))
        facet == 2 && return Vec{2}((-J[2, 2], J[1, 2]))
        facet == 3 && return Vec{2}((J[2, 1], -J[1, 1]))
    end
    throw(ArgumentError("unknown facet number"))
end

########################
# All RefHexahedron 3D #
########################

# Mapping from 2D quadrilateral to 3D face of a hexahedron.
function facet_to_element_transformation(point::Vec{2, T}, ::Type{RefHexahedron}, facet::Int) where {T}
    x, y = point
    facet == 1 && return Vec{3, T}((y, x, -one(T)))
    facet == 2 && return Vec{3, T}((x, -one(T), y))
    facet == 3 && return Vec{3, T}((one(T), x, y))
    facet == 4 && return Vec{3, T}((-x, one(T), y))
    facet == 5 && return Vec{3, T}((-one(T), y, x))
    facet == 6 && return Vec{3, T}((x, y, one(T)))
    throw(ArgumentError("unknown facet number"))
end

# Mapping from 3D face of a hexahedron to 2D quadrilateral.
function element_to_facet_transformation(point::Vec{3, T}, ::Type{RefHexahedron}, facet::Int) where {T}
    x, y, z = point
    facet == 1 && return Vec(y, x)
    facet == 2 && return Vec(x, z)
    facet == 3 && return Vec(y, z)
    facet == 4 && return Vec(-x, z)
    facet == 5 && return Vec(z, y)
    facet == 6 && return Vec(x, y)
    throw(ArgumentError("unknown facet number"))
end

function weighted_normal(J::Tensor{2, 3}, ::Type{RefHexahedron}, facet::Int)
    @inbounds begin
        facet == 1 && return J[:, 2] × J[:, 1]
        facet == 2 && return J[:, 1] × J[:, 3]
        facet == 3 && return J[:, 2] × J[:, 3]
        facet == 4 && return J[:, 3] × J[:, 1]
        facet == 5 && return J[:, 3] × J[:, 2]
        facet == 6 && return J[:, 1] × J[:, 2]
    end
    throw(ArgumentError("unknown facet number"))
end

#########################
# All RefTetrahedron 3D #
#########################

# Mapping from 2D triangle to 3D face of a tetrahedon.
function facet_to_element_transformation(point::Vec{2, T}, ::Type{RefTetrahedron}, facet::Int) where {T}
    x, y = point
    facet == 1 && return Vec{3, T}((one(T) - x - y, y, zero(T)))
    facet == 2 && return Vec{3, T}((y, zero(T), one(T) - x - y))
    facet == 3 && return Vec{3, T}((x, y, one(T) - x - y))
    facet == 4 && return Vec{3, T}((zero(T), one(T) - x - y, y))
    throw(ArgumentError("unknown facet number"))
end

# Mapping from 3D face of a tetrahedon to 2D triangle.
function element_to_facet_transformation(point::Vec{3, T}, ::Type{RefTetrahedron}, facet::Int) where {T}
    x, y, z = point
    facet == 1 && return Vec(one(T) - x - y, y)
    facet == 2 && return Vec(one(T) - z - x, x)
    facet == 3 && return Vec(x, y)
    facet == 4 && return Vec(one(T) - y - z, z)
    throw(ArgumentError("unknown facet number"))
end

function weighted_normal(J::Tensor{2, 3}, ::Type{RefTetrahedron}, facet::Int)
    @inbounds begin
        facet == 1 && return J[:, 2] × J[:, 1]
        facet == 2 && return J[:, 1] × J[:, 3]
        facet == 3 && return (J[:, 1] - J[:, 3]) × (J[:, 2] - J[:, 3])
        facet == 4 && return J[:, 3] × J[:, 2]
    end
    throw(ArgumentError("unknown facet number"))
end

###################
# All RefPrism 3D #
###################

# Mapping from 2D quadrilateral/triangle to 3D face of a wedge.
function facet_to_element_transformation(point::Vec{2, T}, ::Type{RefPrism}, facet::Int) where {T}
    # Note that for quadrilaterals the domain is [-1, 1]² but for triangles it is [0, 1]²
    x, y = point
    facet == 1 && return Vec{3, T}((one(T) - x - y, y, zero(T)))
    facet == 2 && return Vec{3, T}(((one(T) + x) / 2, zero(T), (one(T) + y) / 2))
    facet == 3 && return Vec{3, T}((zero(T), one(T) - (one(T) + x) / 2, (one(T) + y) / 2))
    facet == 4 && return Vec{3, T}((one(T) - (one(T) + x) / 2, (one(T) + x) / 2, (one(T) + y) / 2))
    facet == 5 && return Vec{3, T}((y, one(T) - x - y, one(T)))
    throw(ArgumentError("unknown facet number"))
end

# Mapping from 3D face of a wedge to 2D triangle or 2D quadrilateral.
function element_to_facet_transformation(point::Vec{3, T}, ::Type{RefPrism}, facet::Int) where {T}
    x, y, z = point
    facet == 1 && return Vec(one(T) - x - y, y)
    facet == 2 && return Vec(2 * x - one(T), 2 * z - one(T))
    facet == 3 && return Vec(2 * (one(T) - y) - one(T), 2 * z - one(T))
    facet == 4 && return Vec(2 * y - one(T), 2 * z - one(T))
    facet == 5 && return Vec(one(T) - x - y, x)
    throw(ArgumentError("unknown facet number"))
end

function weighted_normal(J::Tensor{2, 3}, ::Type{RefPrism}, facet::Int)
    @inbounds begin
        facet == 1 && return J[:, 2] × J[:, 1]
        facet == 2 && return J[:, 1] × J[:, 3]
        facet == 3 && return J[:, 3] × J[:, 2]
        facet == 4 && return (J[:, 2] - J[:, 1]) × J[:, 3]
        facet == 5 && return J[:, 1] × J[:, 2]
    end
    throw(ArgumentError("unknown facet number"))
end

#####################
# All RefPyramid 3D #
#####################

# Mapping from 2D face to 3D face of a pyramid.
function facet_to_element_transformation(point::Vec{2, T}, ::Type{RefPyramid}, facet::Int) where {T}
    x, y = point
    facet == 1 && return Vec{3, T}(((y + one(T)) / 2, (x + one(T)) / 2, zero(T)))
    facet == 2 && return Vec{3, T}((y, zero(T), one(T) - x - y))
    facet == 3 && return Vec{3, T}((zero(T), one(T) - x - y, y))
    facet == 4 && return Vec{3, T}((x + y, y, one(T) - x - y))
    facet == 5 && return Vec{3, T}((one(T) - x - y, one(T) - y, y))
    throw(ArgumentError("unknown facet number"))
end

# Mapping from 3D face of a pyramid to 2D triangle or 2D quadrilateral.
function element_to_facet_transformation(point::Vec{3, T}, ::Type{RefPyramid}, facet::Int) where {T}
    x, y, z = point
    facet == 1 && return Vec(2 * y - one(T), 2 * x - one(T))
    facet == 2 && return Vec(one(T) - z - x, x)
    facet == 3 && return Vec(one(T) - y - z, z)
    facet == 4 && return Vec(x - y, y)
    facet == 5 && return Vec(one(T) - x - z, z)
    throw(ArgumentError("unknown facet number"))
end

function weighted_normal(J::Tensor{2, 3}, ::Type{RefPyramid}, facet::Int)
    @inbounds begin
        facet == 1 && return J[:, 2] × J[:, 1]
        facet == 2 && return J[:, 1] × J[:, 3]
        facet == 3 && return J[:, 3] × J[:, 2]
        facet == 4 && return J[:, 2] × (J[:, 3] - J[:, 1])
        facet == 5 && return (J[:, 3] - J[:, 2]) × J[:, 1]
    end
    throw(ArgumentError("unknown facet number"))
end
