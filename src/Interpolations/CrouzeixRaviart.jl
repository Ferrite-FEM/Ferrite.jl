"""
    CrouzeixRaviart{refshape, order} <: ScalarInterpolation

Classical non-conforming Crouzeix–Raviart element.
The following interpolations are implemented:

* `CrouzeixRaviart{RefTriangle, 1}`
* `CrouzeixRaviart{RefTetrahedron, 1}`

For details we refer to the original paper [CroRav:1973:cnf](@cite).
"""
struct CrouzeixRaviart{shape, order} <: ScalarInterpolation{shape, order}
    CrouzeixRaviart{RefTriangle, 1}() = new{RefTriangle, 1}()
    CrouzeixRaviart{RefTetrahedron, 1}() = new{RefTetrahedron, 1}()
end

# CR elements are characterized by not having vertex dofs
vertexdof_indices(ip::CrouzeixRaviart) = ntuple(i -> (), nvertices(ip))

#################################################
# Non-conforming Crouzeix-Raviart dim 2 order 1 #
#################################################
getnbasefunctions(::CrouzeixRaviart{RefTriangle, 1}) = 3

adjust_dofs_during_distribution(::CrouzeixRaviart) = true
adjust_dofs_during_distribution(::CrouzeixRaviart{<:Any, 1}) = false

edgedof_indices(::CrouzeixRaviart{RefTriangle, 1}) = ((1,), (2,), (3,))
edgedof_interior_indices(::CrouzeixRaviart{RefTriangle, 1}) = ((1,), (2,), (3,))
facedof_indices(ip::CrouzeixRaviart{RefTriangle, 1}) = (ntuple(i -> i, getnbasefunctions(ip)),)

function reference_coordinates(::CrouzeixRaviart{RefTriangle, 1})
    return [
        Vec{2, Float64}((0.5, 0.5)),
        Vec{2, Float64}((0.0, 0.5)),
        Vec{2, Float64}((0.5, 0.0)),
    ]
end

function reference_shape_value(ip::CrouzeixRaviart{RefTriangle, 1}, ξ::Vec{2}, i::Int)
    ξ_x = ξ[1]
    ξ_y = ξ[2]
    i == 1 && return 2 * ξ_x + 2 * ξ_y - 1
    i == 2 && return 1 - 2 * ξ_x
    i == 3 && return 1 - 2 * ξ_y
    throw(ArgumentError("no shape function $i for interpolation $ip"))
end

#################################################
# Non-conforming Crouzeix-Raviart dim 3 order 1 #
#################################################
getnbasefunctions(::CrouzeixRaviart{RefTetrahedron, 1}) = 4

facedof_indices(::CrouzeixRaviart{RefTetrahedron, 1}) = ((1,), (2,), (3,), (4,))
facedof_interior_indices(::CrouzeixRaviart{RefTetrahedron, 1}) = ((1,), (2,), (3,), (4,))

function reference_coordinates(::CrouzeixRaviart{RefTetrahedron, 1})
    return [
        Vec{3, Float64}((1 / 3, 1 / 3, 0.0)),
        Vec{3, Float64}((1 / 3, 0.0, 1 / 3)),
        Vec{3, Float64}((1 / 3, 1 / 3, 1 / 3)),
        Vec{3, Float64}((0.0, 1 / 3, 1 / 3)),
    ]
end

function reference_shape_value(ip::CrouzeixRaviart{RefTetrahedron, 1}, ξ::Vec{3}, i::Int)
    (x, y, z) = ξ
    i == 1 && return 1 - 3z
    i == 2 && return 1 - 3y
    i == 3 && return 3x + 3y + 3z - 2
    i == 4 && return 1 - 3x
    return throw(ArgumentError("no shape function $i for interpolation $ip"))
end
