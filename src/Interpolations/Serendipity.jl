"""
    Serendipity{refshape, order} <: ScalarInterpolation

Serendipity element on hypercubes. 
The following interpolations are implemented:

* `Serendipity{RefQuadrilateral,2}`
* `Serendipity{RefHexahedron,2}`
"""
struct Serendipity{shape, order} <: ScalarInterpolation{shape, order}
    function Serendipity{shape, order}() where {shape <: AbstractRefShape, order}
        return new{shape, order}()
    end
end

# Note that the edgedofs for high order serendipity elements are defined in terms of integral moments,
# so no permutation exists in general. See e.g. Scroggs et al. [2022] for an example.
# adjust_dofs_during_distribution(::Serendipity{refshape, order}) where {refshape, order} = false
adjust_dofs_during_distribution(::Serendipity{<:Any, 2}) = false
adjust_dofs_during_distribution(::Serendipity{<:Any, 1}) = false

# Vertices for all Serendipity interpolations are the same
vertexdof_indices(::Serendipity{RefQuadrilateral}) = ((1,), (2,), (3,), (4,))
vertexdof_indices(::Serendipity{RefHexahedron}) = ((1,), (2,), (3,), (4,), (5,), (6,), (7,), (8,))

########################################
# Serendipity RefQuadrilateral order 2 #
########################################
getnbasefunctions(::Serendipity{RefQuadrilateral, 2}) = 8
getlowerorder(::Serendipity{RefQuadrilateral, 2}) = Lagrange{RefQuadrilateral, 1}()

edgedof_indices(::Serendipity{RefQuadrilateral, 2}) = ((1, 2, 5), (2, 3, 6), (3, 4, 7), (4, 1, 8))
edgedof_interior_indices(::Serendipity{RefQuadrilateral, 2}) = ((5,), (6,), (7,), (8,))
facedof_indices(ip::Serendipity{RefQuadrilateral, 2}) = (ntuple(i -> i, getnbasefunctions(ip)),)

function reference_coordinates(::Serendipity{RefQuadrilateral, 2})
    return [
        Vec{2, Float64}((-1.0, -1.0)),
        Vec{2, Float64}((1.0, -1.0)),
        Vec{2, Float64}((1.0, 1.0)),
        Vec{2, Float64}((-1.0, 1.0)),
        Vec{2, Float64}((0.0, -1.0)),
        Vec{2, Float64}((1.0, 0.0)),
        Vec{2, Float64}((0.0, 1.0)),
        Vec{2, Float64}((-1.0, 0.0)),
    ]
end

function reference_shape_value(ip::Serendipity{RefQuadrilateral, 2}, ξ::Vec{2}, i::Int)
    ξ_x = ξ[1]
    ξ_y = ξ[2]
    i == 1 && return (1 - ξ_x) * (1 - ξ_y) * (-ξ_x - ξ_y - 1) / 4
    i == 2 && return (1 + ξ_x) * (1 - ξ_y) * (ξ_x - ξ_y - 1) / 4
    i == 3 && return (1 + ξ_x) * (1 + ξ_y) * (ξ_x + ξ_y - 1) / 4
    i == 4 && return (1 - ξ_x) * (1 + ξ_y) * (-ξ_x + ξ_y - 1) / 4
    i == 5 && return (1 - ξ_x * ξ_x) * (1 - ξ_y) / 2
    i == 6 && return (1 + ξ_x) * (1 - ξ_y * ξ_y) / 2
    i == 7 && return (1 - ξ_x * ξ_x) * (1 + ξ_y) / 2
    i == 8 && return (1 - ξ_x) * (1 - ξ_y * ξ_y) / 2
    throw(ArgumentError("no shape function $i for interpolation $ip"))
end

#####################################
# Serendipity RefHexahedron order 2 #
#####################################
# Note that second order serendipity hex has no interior face indices.
getnbasefunctions(::Serendipity{RefHexahedron, 2}) = 20
getlowerorder(::Serendipity{RefHexahedron, 2}) = Lagrange{RefHexahedron, 1}()

facedof_indices(::Serendipity{RefHexahedron, 2}) = (
    (1, 4, 3, 2, 12, 11, 10, 9),
    (1, 2, 6, 5, 9, 18, 13, 17),
    (2, 3, 7, 6, 10, 19, 14, 18),
    (3, 4, 8, 7, 11, 20, 15, 19),
    (1, 5, 8, 4, 17, 16, 20, 12),
    (5, 6, 7, 8, 13, 14, 15, 16),
)
edgedof_indices(::Serendipity{RefHexahedron, 2}) = (
    (1, 2, 9),
    (2, 3, 10),
    (3, 4, 11),
    (4, 1, 12),
    (5, 6, 13),
    (6, 7, 14),
    (7, 8, 15),
    (8, 5, 16),
    (1, 5, 17),
    (2, 6, 18),
    (3, 7, 19),
    (4, 8, 20),
)

edgedof_interior_indices(::Serendipity{RefHexahedron, 2}) = (
    (9,), (10,), (11,), (12,), (13,), (14,), (15,), (16,), (17), (18,), (19,), (20,),
)

function reference_coordinates(::Serendipity{RefHexahedron, 2})
    return [
        Vec{3, Float64}((-1.0, -1.0, -1.0)),
        Vec{3, Float64}((1.0, -1.0, -1.0)),
        Vec{3, Float64}((1.0, 1.0, -1.0)),
        Vec{3, Float64}((-1.0, 1.0, -1.0)),
        Vec{3, Float64}((-1.0, -1.0, 1.0)),
        Vec{3, Float64}((1.0, -1.0, 1.0)),
        Vec{3, Float64}((1.0, 1.0, 1.0)),
        Vec{3, Float64}((-1.0, 1.0, 1.0)),
        Vec{3, Float64}((0.0, -1.0, -1.0)),
        Vec{3, Float64}((1.0, 0.0, -1.0)),
        Vec{3, Float64}((0.0, 1.0, -1.0)),
        Vec{3, Float64}((-1.0, 0.0, -1.0)),
        Vec{3, Float64}((0.0, -1.0, 1.0)),
        Vec{3, Float64}((1.0, 0.0, 1.0)),
        Vec{3, Float64}((0.0, 1.0, 1.0)),
        Vec{3, Float64}((-1.0, 0.0, 1.0)),
        Vec{3, Float64}((-1.0, -1.0, 0.0)),
        Vec{3, Float64}((1.0, -1.0, 0.0)),
        Vec{3, Float64}((1.0, 1.0, 0.0)),
        Vec{3, Float64}((-1.0, 1.0, 0.0)),
    ]
end

# Inlined to resolve the recursion properly
@inline function reference_shape_value(ip::Serendipity{RefHexahedron, 2}, ξ::Vec{3}, i::Int)
    ξ_x = ξ[1]
    ξ_y = ξ[2]
    ξ_z = ξ[3]
    i == 1 && return (1 - ξ_x) * (1 - ξ_y) * (1 - ξ_z) / 8 - (reference_shape_value(ip, ξ, 12) + reference_shape_value(ip, ξ, 9) + reference_shape_value(ip, ξ, 17)) / 2
    i == 2 && return (1 + ξ_x) * (1 - ξ_y) * (1 - ξ_z) / 8 - (reference_shape_value(ip, ξ, 9) + reference_shape_value(ip, ξ, 10) + reference_shape_value(ip, ξ, 18)) / 2
    i == 3 && return (1 + ξ_x) * (1 + ξ_y) * (1 - ξ_z) / 8 - (reference_shape_value(ip, ξ, 10) + reference_shape_value(ip, ξ, 11) + reference_shape_value(ip, ξ, 19)) / 2
    i == 4 && return (1 - ξ_x) * (1 + ξ_y) * (1 - ξ_z) / 8 - (reference_shape_value(ip, ξ, 11) + reference_shape_value(ip, ξ, 12) + reference_shape_value(ip, ξ, 20)) / 2
    i == 5 && return (1 - ξ_x) * (1 - ξ_y) * (1 + ξ_z) / 8 - (reference_shape_value(ip, ξ, 16) + reference_shape_value(ip, ξ, 13) + reference_shape_value(ip, ξ, 17)) / 2
    i == 6 && return (1 + ξ_x) * (1 - ξ_y) * (1 + ξ_z) / 8 - (reference_shape_value(ip, ξ, 13) + reference_shape_value(ip, ξ, 14) + reference_shape_value(ip, ξ, 18)) / 2
    i == 7 && return (1 + ξ_x) * (1 + ξ_y) * (1 + ξ_z) / 8 - (reference_shape_value(ip, ξ, 14) + reference_shape_value(ip, ξ, 15) + reference_shape_value(ip, ξ, 19)) / 2
    i == 8 && return (1 - ξ_x) * (1 + ξ_y) * (1 + ξ_z) / 8 - (reference_shape_value(ip, ξ, 15) + reference_shape_value(ip, ξ, 16) + reference_shape_value(ip, ξ, 20)) / 2
    i == 9 && return (1 - ξ_x^2) * (1 - ξ_y) * (1 - ξ_z) / 4
    i == 10 && return (1 + ξ_x) * (1 - ξ_y^2) * (1 - ξ_z) / 4
    i == 11 && return (1 - ξ_x^2) * (1 + ξ_y) * (1 - ξ_z) / 4
    i == 12 && return (1 - ξ_x) * (1 - ξ_y^2) * (1 - ξ_z) / 4
    i == 13 && return (1 - ξ_x^2) * (1 - ξ_y) * (1 + ξ_z) / 4
    i == 14 && return (1 + ξ_x) * (1 - ξ_y^2) * (1 + ξ_z) / 4
    i == 15 && return (1 - ξ_x^2) * (1 + ξ_y) * (1 + ξ_z) / 4
    i == 16 && return (1 - ξ_x) * (1 - ξ_y^2) * (1 + ξ_z) / 4
    i == 17 && return (1 - ξ_x) * (1 - ξ_y) * (1 - ξ_z^2) / 4
    i == 18 && return (1 + ξ_x) * (1 - ξ_y) * (1 - ξ_z^2) / 4
    i == 19 && return (1 + ξ_x) * (1 + ξ_y) * (1 - ξ_z^2) / 4
    i == 20 && return (1 - ξ_x) * (1 + ξ_y) * (1 - ξ_z^2) / 4
    throw(ArgumentError("no shape function $i for interpolation $ip"))
end
