"""
    RannacherTurek{refshape, order} <: ScalarInterpolation

Classical non-conforming Rannacher-Turek element.

This element is basically the idea from Crouzeix and Raviart applied to hypercubes.
The following interpolations are implemented:

* `RannacherTurek{RefQuadrilateral, 1}`
* `RannacherTurek{RefHexahedron, 1}`

For details see the original paper [RanTur:1992:snq](@cite).
"""
struct RannacherTurek{shape, order} <: ScalarInterpolation{shape, order} end

# CR-type elements are characterized by not having vertex dofs
vertexdof_indices(ip::RannacherTurek) = ntuple(i -> (), nvertices(ip))

adjust_dofs_during_distribution(::RannacherTurek) = true
adjust_dofs_during_distribution(::RannacherTurek{<:Any, 1}) = false

#################################
# Rannacher-Turek dim 2 order 1 #
#################################
getnbasefunctions(::RannacherTurek{RefQuadrilateral, 1}) = 4

edgedof_indices(::RannacherTurek{RefQuadrilateral, 1}) = ((1,), (2,), (3,), (4,))
edgedof_interior_indices(::RannacherTurek{RefQuadrilateral, 1}) = ((1,), (2,), (3,), (4,))
facedof_indices(ip::RannacherTurek{RefQuadrilateral, 1}) = (ntuple(i -> i, getnbasefunctions(ip)),)

function reference_coordinates(::RannacherTurek{RefQuadrilateral, 1})
    return [
        Vec{2, Float64}((0.0, -1.0)),
        Vec{2, Float64}((1.0, 0.0)),
        Vec{2, Float64}((0.0, 1.0)),
        Vec{2, Float64}((-1.0, 0.0)),
    ]
end

function reference_shape_value(ip::RannacherTurek{RefQuadrilateral, 1}, ξ::Vec{2, T}, i::Int) where {T}
    (x, y) = ξ

    i == 1 && return -(x + 1)^2 / 4 + (y + 1)^2 / 4 + (x + 1) / 2 - (y + 1) + T(3) / 4
    i == 2 && return (x + 1)^2 / 4 - (y + 1)^2 / 4 + (y + 1) / 2 - T(1) / 4
    i == 3 && return -(x + 1)^2 / 4 + (y + 1)^2 / 4 + (x + 1) / 2 - T(1) / 4
    i == 4 && return (x + 1)^2 / 4 - (y + 1)^2 / 4 - (x + 1) + (y + 1) / 2 + T(3) / 4
    throw(ArgumentError("no shape function $i for interpolation $ip"))
end

#################################
# Rannacher-Turek dim 3 order 1 #
#################################
getnbasefunctions(::RannacherTurek{RefHexahedron, 1}) = 6

edgedof_indices(ip::RannacherTurek{RefHexahedron, 1}) = ntuple(i -> (), nedges(ip))
edgedof_interior_indices(ip::RannacherTurek{RefHexahedron, 1}) = ntuple(i -> (), nedges(ip))
facedof_indices(::RannacherTurek{RefHexahedron, 1}) = ((1,), (2,), (3,), (4,), (5,), (6,))
facedof_interior_indices(::RannacherTurek{RefHexahedron, 1}) = ((1,), (2,), (3,), (4,), (5,), (6,))

function reference_coordinates(::RannacherTurek{RefHexahedron, 1})
    return [
        Vec{3, Float64}((0.0, 0.0, -1.0)),
        Vec{3, Float64}((0.0, -1.0, 0.0)),
        Vec{3, Float64}((1.0, 0.0, 0.0)),
        Vec{3, Float64}((0.0, 1.0, 0.0)),
        Vec{3, Float64}((-1.0, 0.0, 0.0)),
        Vec{3, Float64}((0.0, 0.0, 1.0)),
    ]
end

function reference_shape_value(ip::RannacherTurek{RefHexahedron, 1}, ξ::Vec{3, T}, i::Int) where {T}
    (x, y, z) = ξ

    i == 1 && return -2((x + 1))^2 / 12 + 1(x + 1) / 3 - 2((y + 1))^2 / 12 + 1(y + 1) / 3 + 4((z + 1))^2 / 12 - 7(z + 1) / 6 + T(2) / 3
    i == 2 && return -2((x + 1))^2 / 12 + 1(x + 1) / 3 + 4((y + 1))^2 / 12 - 7(y + 1) / 6 - 2((z + 1))^2 / 12 + 1(z + 1) / 3 + T(2) / 3
    i == 3 && return 4((x + 1))^2 / 12 - 1(x + 1) / 6 - 2((y + 1))^2 / 12 + 1(y + 1) / 3 - 2((z + 1))^2 / 12 + 1(z + 1) / 3 - T(1) / 3
    i == 4 && return -2((x + 1))^2 / 12 + 1(x + 1) / 3 + 4((y + 1))^2 / 12 - 1(y + 1) / 6 - 2((z + 1))^2 / 12 + 1(z + 1) / 3 - T(1) / 3
    i == 5 && return 4((x + 1))^2 / 12 - 7(x + 1) / 6 - 2((y + 1))^2 / 12 + 1(y + 1) / 3 - 2((z + 1))^2 / 12 + 1(z + 1) / 3 + T(2) / 3
    i == 6 && return -2((x + 1))^2 / 12 + 1(x + 1) / 3 - 2((y + 1))^2 / 12 + 1(y + 1) / 3 + 4((z + 1))^2 / 12 - 1(z + 1) / 6 - T(1) / 3

    throw(ArgumentError("no shape function $i for interpolation $ip"))
end
