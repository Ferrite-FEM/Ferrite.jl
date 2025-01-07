# Generalization of `face[t]_integrals.jl`
# Here we use consistent naming, so quick backwards compat for testing,
facet_to_element_transformation(ξ, ::Type{RS}, nr) where {RS} = facet_to_cell_transformation(ξ, RS, nr)
element_to_facet_transformation(ξ, ::Type{RS}, nr) where {RS} = cell_to_facet_transformation(ξ, RS, nr)

"""
    edge_to_cell_transformation(ξ::Vec{1}, cell_refshape::Type{<:AbstractRefShape{rdim}}, edgenr::Int) where rdim

Transform local point, `ξ`, from the edge's reference coordinates to the cell's
reference coordinates. The returned `Vec{rdim}` is located on the cell's edge `edgenr`.
"""
edge_to_cell_transformation

"""
    cell_to_edge_transformation(ξ::Vec{rdim}, ::Type{<:AbstractRefShape{rdim}}, edgenr::Int) where rdim

This is the inverse of `edge_to_cell_transformation`, returning a `Vec{1}`.
`ξ` must be located on edge `edgenr` for correct transformation.
"""
cell_to_edge_transformation

"""
    face_to_cell_transformation(ξ::Vec{2}, ::Type{<:AbstractRefShape{rdim}}, facenr::Int) where rdim

Transform local point, `ξ`, from the face's reference coordinates to the cell's
reference coordinates. The returned `Vec{rdim}` is located on the cell's face `facenr`.
"""
face_to_cell_transformation

"""
    cell_to_face_transformation(ξ::Vec{rdim}, ::Type{<:AbstractRefShape{rdim}}, facenr::Int) where rdim

This is the inverse of `face_to_cell_transformation`, returning a `Vec{2}`.
`ξ` must be located on face `facenr` for correct transformation.
"""
cell_to_face_transformation

"""
    weighted_normal(J::AbstractTensor, fv::FacetValues, face::Int)
    weighted_normal(J::AbstractTensor, ::Type{<:AbstractRefShape}, face::Int)

Compute the vector normal to the facet weighted by the area ratio between the facet and the
reference facet. This is computed by taking the cross product of the Jacobian components that
align to the facet's local axis.
"""
function weighted_normal end

@doc raw"""
    weighted_tangent(J, RefShape, edgenr::Int)

To integrate a function, `f(x)=f̂(ξ(x))`, along the edge, ``E``,
using a quadrature rule described in the reference coordinates, `ξ`,
we transform the integral to an integral over the reference edge, ``\hat{E}`` as,
```math
\int_{E} \boldsymbol{f} ⋅ \mathrm{d}\boldsymbol{E} = \int_{\hat{E}} \boldsymbol{f} ⋅ \boldsymbol{J} ⋅ \mathrm{d}\hat{\boldsymbol{E}},
```
where ``\boldsymbol{J}  = \frac{\partial x}{\partial \xi}`` is the jacobian of the mapping.
When integrating over one edge on the reference element, the direction, ``d\hat{\boldsymbol{E}}``,
is known and constant. W can thus express this integral by using the weighted tangent,
``\boldsymbol{W}_t``, and the scalar differential, ``\mathrm{d}s``, such that
```math
\boldsymbol{W}_t \mathrm{d}s = \boldsymbol{J} ⋅ \mathrm{d}\hat{\boldsymbol{E}}
```
"""
function weighted_tangent end

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

function create_edge_quad_rule(::Type{RefShape}, w::AbstractVector{T}, p::AbstractVector{<:Vec{1, T}}) where {T, RefShape <: AbstractRefShape}
    edge_quad_rule = ntuple(nedges(RefShape)) do edgenr
        new_points = map(ξ -> edge_to_cell_transformation(ξ, RefShape, edgenr), p)
        QuadratureRule{RefShape}(copy(w), new_points)
    end
    return EdgeQuadratureRule(SVector(edge_quad_rule))
end

# facet definitions for differenent reference dimensions
# rdim = 2, facet == edge
function facet_to_cell_transformation(ξ::Vec{1}, ::Type{RS}, facetnr::Int) where {RS <: AbstractRefShape{2}}
    return edge_to_cell_transformation(ξ, RS, facetnr)
end
function cell_to_facet_transformation(ξ::Vec{2}, ::Type{RS}, facetnr::Int) where {RS <: AbstractRefShape{2}}
    return cell_to_edge_transformation(ξ, RS, facetnr)
end
# rdim = 3, facet == face
function facet_to_cell_transformation(ξ::Vec{2}, ::Type{RS}, facetnr::Int) where {RS <: AbstractRefShape{3}}
    return face_to_cell_transformation(ξ, RS, facetnr)
end
function cell_to_facet_transformation(ξ::Vec{3}, ::Type{RS}, facetnr::Int) where {RS <: AbstractRefShape{3}}
    return cell_to_face_transformation(ξ, RS, facetnr)
end

##################
# All 1D RefLine #
##################
# Special cases since we don't have cell_to_vertex and vertex_to_cell
# Mapping from to 0D node to 1D line vertex.
function facet_to_element_transformation(::Union{Vec{0, T}, Vec{1, T}}, ::Type{RefLine}, facetnr::Int) where {T}
    facetnr == 1 && return Vec{1, T}((-one(T),))
    facetnr == 2 && return Vec{1, T}((one(T),))
    throw(ArgumentError("unknown facet number"))
end

# Mapping from 1D line to point.
function element_to_facet_transformation(ξ::Vec{1, T}, ::Type{RefLine}, facetnr::Int) where {T}
    x = ξ[1]
    facetnr == 1 && return Vec(-x)
    facetnr == 2 && return Vec(x)
    throw(ArgumentError("unknown facet number"))
end

function weighted_normal(::Tensor{2, 1, T}, ::Type{RefLine}, facetnr::Int) where {T}
    facetnr == 1 && return Vec{1, T}((-one(T),))
    facetnr == 2 && return Vec{1, T}((one(T),))
    throw(ArgumentError("unknown facet number"))
end

function weighted_tangent(J::Tensor{2, 1}, ::Type{RefLine}, edgenr::Int)
    @inbounds begin
        edgenr == 1 && return Vec((J[1, 1],))
    end
    throw(ArgumentError("RefLine requires edgenr == 1"))
end

###########################
# All 2D RefQuadrilateral #
###########################

# Mapping from 1D line to 2D face of a quadrilateral.
function edge_to_cell_transformation(ξ::Vec{1, T}, ::Type{RefQuadrilateral}, edgenr::Int) where {T}
    x = ξ[1]
    edgenr == 1 && return Vec{2, T}((x, -one(T)))
    edgenr == 2 && return Vec{2, T}((one(T), x))
    edgenr == 3 && return Vec{2, T}((-x, one(T)))
    edgenr == 4 && return Vec{2, T}((-one(T), -x))
    throw(ArgumentError("unknown edge number"))
end

# Mapping from 2D face of a quadrilateral to 1D line.
function cell_to_edge_transformation(ξ::Vec{2, T}, ::Type{RefQuadrilateral}, edgenr::Int) where {T}
    x, y = ξ
    edgenr == 1 && return Vec(x)
    edgenr == 2 && return Vec(y)
    edgenr == 3 && return Vec(-x)
    edgenr == 4 && return Vec(-y)
    throw(ArgumentError("unknown edge number"))
end

function weighted_normal(J::Tensor{2, 2}, ::Type{RefQuadrilateral}, edgenr::Int)
    @inbounds begin
        edgenr == 1 && return Vec{2}((J[2, 1], -J[1, 1]))
        edgenr == 2 && return Vec{2}((J[2, 2], -J[1, 2]))
        edgenr == 3 && return Vec{2}((-J[2, 1], J[1, 1]))
        edgenr == 4 && return Vec{2}((-J[2, 2], J[1, 2]))
    end
    throw(ArgumentError("unknown facet number"))
end

function weighted_tangent(J::Tensor{2, 2}, ::Type{RefQuadrilateral}, edgenr::Int)
    @inbounds begin
        edgenr == 1 && return J[:, 1] # dÊ = [ 1,  0] ds
        edgenr == 2 && return J[:, 2] # dÊ = [ 0,  1] ds
        edgenr == 3 && return -J[:, 1] # dÊ = [-1,  0] ds
        edgenr == 4 && return -J[:, 2] # dÊ = [ 0, -1] ds
    end
    throw(ArgumentError("RefQuadrilateral requires edgenr ∈ 1:4"))
end

######################
# All RefTriangle 2D #
######################

# Mapping from 1D line to 2D face of a triangle.
function edge_to_cell_transformation(ξ::Vec{1, T}, ::Type{RefTriangle}, edgenr::Int) where {T}
    x = (ξ[1] + one(T)) / 2
    edgenr == 1 && return Vec{2, T}((one(T) - x, x))
    edgenr == 2 && return Vec{2, T}((zero(T), one(T) - x))
    edgenr == 3 && return Vec{2, T}((x, zero(T)))
    throw(ArgumentError("unknown edgenr number"))
end

# Mapping from 2D face of a triangle to 1D line.
function cell_to_edge_transformation(ξ::Vec{2, T}, ::Type{RefTriangle}, facet::Int) where {T}
    x, y = ξ
    edgenr == 1 && return Vec(one(T) - x * 2)
    edgenr == 2 && return Vec(one(T) - y * 2)
    edgenr == 3 && return Vec(x * 2 - one(T))
    throw(ArgumentError("unknown edgenr number"))
end

function weighted_normal(J::Tensor{2, 2}, ::Type{RefTriangle}, facet::Int)
    @inbounds begin
        facet == 1 && return Vec{2}((-(J[2, 1] - J[2, 2]), J[1, 1] - J[1, 2]))
        facet == 2 && return Vec{2}((-J[2, 2], J[1, 2]))
        facet == 3 && return Vec{2}((J[2, 1], -J[1, 1]))
    end
    throw(ArgumentError("unknown facet number"))
end

function weighted_tangent(J::Tensor{2, 2}, ::Type{RefTriangle}, edgenr::Int)
    @inbounds begin
        edgenr == 1 && return (J[:, 2] - J[:, 1]) # dÊ = [-1, 1] ds
        edgenr == 2 && return -J[:, 2] # dÊ = [0, -1] ds
        edgenr == 3 && return J[:, 1] # dÊ = [1,  0] ds
    end
    throw(ArgumentError("RefTriangle requires edgenr ∈ 1:3"))
end

########################
# All RefHexahedron 3D #
########################

# TODO: Mapping from 1D line to 3D edge of a hexahedron
# function edge_to_cell_transformation(ξ::Vec{1, T}, ::Type{RefHexahedron}, edgenr::Int) where {T}
# TODO: Mapping from 3D edge of a hexahedron to 1D line
# function cell_to_edge_transformation(ξ::Vec{3, T}, ::Type{RefHexahedron}, edgenr::Int) where {T}

# Mapping from 2D quadrilateral to 3D face of a hexahedron.
function face_to_cell_transformation(ξ::Vec{2, T}, ::Type{RefHexahedron}, facenr::Int) where {T}
    x, y = ξ
    facenr == 1 && return Vec{3, T}((y, x, -one(T)))
    facenr == 2 && return Vec{3, T}((x, -one(T), y))
    facenr == 3 && return Vec{3, T}((one(T), x, y))
    facenr == 4 && return Vec{3, T}((-x, one(T), y))
    facenr == 5 && return Vec{3, T}((-one(T), y, x))
    facenr == 6 && return Vec{3, T}((x, y, one(T)))
    throw(ArgumentError("unknown face number"))
end

# Mapping from 3D face of a hexahedron to 2D quadrilateral.
function cell_to_face_transformation(ξ::Vec{3, T}, ::Type{RefHexahedron}, facenr::Int) where {T}
    x, y, z = ξ
    facenr == 1 && return Vec(y, x)
    facenr == 2 && return Vec(x, z)
    facenr == 3 && return Vec(y, z)
    facenr == 4 && return Vec(-x, z)
    facenr == 5 && return Vec(z, y)
    facenr == 6 && return Vec(x, y)
    throw(ArgumentError("unknown face number"))
end

function weighted_normal(J::Tensor{2, 3}, ::Type{RefHexahedron}, face::Int)
    @inbounds begin
        face == 1 && return J[:, 2] × J[:, 1]
        face == 2 && return J[:, 1] × J[:, 3]
        face == 3 && return J[:, 2] × J[:, 3]
        face == 4 && return J[:, 3] × J[:, 1]
        face == 5 && return J[:, 3] × J[:, 2]
        face == 6 && return J[:, 1] × J[:, 2]
    end
    throw(ArgumentError("unknown facet number"))
end

#########################
# All RefTetrahedron 3D #
#########################

# TODO: Mapping from 1D line to 3D edge of a tetrahedron
# function edge_to_cell_transformation(ξ::Vec{1, T}, ::Type{RefTetrahedron}, edgenr::Int) where {T}
# TODO: Mapping from 3D edge of a tetrahedron to 1D line
# function cell_to_edge_transformation(ξ::Vec{3, T}, ::Type{RefTetrahedron}, edgenr::Int) where {T}

# Mapping from 2D triangle to 3D face of a tetrahedon.
function face_to_cell_transformation(ξ::Vec{2, T}, ::Type{RefTetrahedron}, facenr::Int) where {T}
    x, y = ξ
    facenr == 1 && return Vec{3, T}((one(T) - x - y, y, zero(T)))
    facenr == 2 && return Vec{3, T}((y, zero(T), one(T) - x - y))
    facenr == 3 && return Vec{3, T}((x, y, one(T) - x - y))
    facenr == 4 && return Vec{3, T}((zero(T), one(T) - x - y, y))
    throw(ArgumentError("unknown facet number"))
end

# Mapping from 3D face of a tetrahedon to 2D triangle.
function cell_to_face_transformation(ξ::Vec{3, T}, ::Type{RefTetrahedron}, facenr::Int) where {T}
    x, y, z = ξ
    facenr == 1 && return Vec(one(T) - x - y, y)
    facenr == 2 && return Vec(one(T) - z - x, x)
    facenr == 3 && return Vec(x, y)
    facenr == 4 && return Vec(one(T) - y - z, z)
    throw(ArgumentError("unknown face number"))
end

function weighted_normal(J::Tensor{2, 3}, ::Type{RefTetrahedron}, face::Int)
    @inbounds begin
        face == 1 && return J[:, 2] × J[:, 1]
        face == 2 && return J[:, 1] × J[:, 3]
        face == 3 && return (J[:, 1] - J[:, 3]) × (J[:, 2] - J[:, 3])
        face == 4 && return J[:, 3] × J[:, 2]
    end
    throw(ArgumentError("unknown facet number"))
end

###################
# All RefPrism 3D #
###################

# TODO: Mapping from 1D line to 3D edge of a prism
# function edge_to_cell_transformation(ξ::Vec{1, T}, ::Type{RefPrism}, edgenr::Int) where {T}
# TODO: Mapping from 3D edge of a prism to 1D line
# function cell_to_edge_transformation(ξ::Vec{3, T}, ::Type{RefPrism}, edgenr::Int) where {T}

# Mapping from 2D quadrilateral/triangle to 3D face of a wedge.
function face_to_cell_transformation(ξ::Vec{2, T}, ::Type{RefPrism}, face::Int) where {T}
    # Note that for quadrilaterals the domain is [-1, 1]² but for triangles it is [0, 1]²
    x, y = ξ
    face == 1 && return Vec{3, T}((one(T) - x - y, y, zero(T)))
    face == 2 && return Vec{3, T}(((one(T) + x) / 2, zero(T), (one(T) + y) / 2))
    face == 3 && return Vec{3, T}((zero(T), one(T) - (one(T) + x) / 2, (one(T) + y) / 2))
    face == 4 && return Vec{3, T}((one(T) - (one(T) + x) / 2, (one(T) + x) / 2, (one(T) + y) / 2))
    face == 5 && return Vec{3, T}((y, one(T) - x - y, one(T)))
    throw(ArgumentError("unknown facet number"))
end

# Mapping from 3D face of a wedge to 2D triangle or 2D quadrilateral.
function cell_to_face_transformation(ξ::Vec{3, T}, ::Type{RefPrism}, face::Int) where {T}
    x, y, z = ξ
    face == 1 && return Vec(one(T) - x - y, y)
    face == 2 && return Vec(2 * x - one(T), 2 * z - one(T))
    face == 3 && return Vec(2 * (one(T) - y) - one(T), 2 * z - one(T))
    face == 4 && return Vec(2 * y - one(T), 2 * z - one(T))
    face == 5 && return Vec(one(T) - x - y, x)
    throw(ArgumentError("unknown facet number"))
end

function weighted_normal(J::Tensor{2, 3}, ::Type{RefPrism}, face::Int)
    @inbounds begin
        face == 1 && return J[:, 2] × J[:, 1]
        face == 2 && return J[:, 1] × J[:, 3]
        face == 3 && return J[:, 3] × J[:, 2]
        face == 4 && return (J[:, 2] - J[:, 1]) × J[:, 3]
        face == 5 && return J[:, 1] × J[:, 2]
    end
    throw(ArgumentError("unknown facet number"))
end

#####################
# All RefPyramid 3D #
#####################

# TODO: Mapping from 1D line to 3D edge of a pyramid
# function edge_to_cell_transformation(ξ::Vec{1, T}, ::Type{RefPyramid}, edgenr::Int) where {T}
# TODO: Mapping from 3D edge of a pyramid to 1D line
# function cell_to_edge_transformation(ξ::Vec{3, T}, ::Type{RefPyramid}, edgenr::Int) where {T}

# Mapping from 2D face to 3D face of a pyramid.
function face_to_cell_transformation(ξ::Vec{2, T}, ::Type{RefPyramid}, face::Int) where {T}
    x, y = ξ
    face == 1 && return Vec{3, T}(((y + one(T)) / 2, (x + one(T)) / 2, zero(T)))
    face == 2 && return Vec{3, T}((y, zero(T), one(T) - x - y))
    face == 3 && return Vec{3, T}((zero(T), one(T) - x - y, y))
    face == 4 && return Vec{3, T}((x + y, y, one(T) - x - y))
    face == 5 && return Vec{3, T}((one(T) - x - y, one(T) - y, y))
    throw(ArgumentError("unknown facet number"))
end

# Mapping from 3D face of a pyramid to 2D triangle or 2D quadrilateral.
function cell_to_face_transformation(ξ::Vec{3, T}, ::Type{RefPyramid}, face::Int) where {T}
    x, y, z = ξ
    face == 1 && return Vec(2 * y - one(T), 2 * x - one(T))
    face == 2 && return Vec(one(T) - z - x, x)
    face == 3 && return Vec(one(T) - y - z, z)
    face == 4 && return Vec(x - y, y)
    face == 5 && return Vec(one(T) - x - z, z)
    throw(ArgumentError("unknown facet number"))
end

function weighted_normal(J::Tensor{2, 3}, ::Type{RefPyramid}, face::Int)
    @inbounds begin
        face == 1 && return J[:, 2] × J[:, 1]
        face == 2 && return J[:, 1] × J[:, 3]
        face == 3 && return J[:, 3] × J[:, 2]
        face == 4 && return J[:, 2] × (J[:, 3] - J[:, 1])
        face == 5 && return (J[:, 3] - J[:, 2]) × J[:, 1]
    end
    throw(ArgumentError("unknown facet number"))
end
