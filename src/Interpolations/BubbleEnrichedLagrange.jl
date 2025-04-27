"""
    BubbleEnrichedLagrange{refshape, order} <: ScalarInterpolation

Lagrange element with bubble stabilization.
The following interpolations are implemented:

* `BubbleEnrichedLagrange{RefTriangle,1}`
"""
struct BubbleEnrichedLagrange{shape, order} <: ScalarInterpolation{shape, order}
    function BubbleEnrichedLagrange{shape, order}() where {shape <: AbstractRefShape, order}
        return new{shape, order}()
    end
end

#######################################
# Lagrange-Bubble RefTriangle order 1 #
#######################################
# Taken from https://defelement.org/elements/examples/triangle-bubble-enriched-lagrange-1.html
getnbasefunctions(::BubbleEnrichedLagrange{RefTriangle, 1}) = 4
adjust_dofs_during_distribution(::BubbleEnrichedLagrange{RefTriangle, 1}) = false

vertexdof_indices(::BubbleEnrichedLagrange{RefTriangle, 1}) = ((1,), (2,), (3,))
edgedof_indices(::BubbleEnrichedLagrange{RefTriangle, 1}) = ((1, 2), (2, 3), (3, 1))
facedof_indices(ip::BubbleEnrichedLagrange{RefTriangle, 1}) = (ntuple(i -> i, getnbasefunctions(ip)),)
facedof_interior_indices(::BubbleEnrichedLagrange{RefTriangle, 1}) = ((4,),)

function reference_coordinates(::BubbleEnrichedLagrange{RefTriangle, 1})
    return [
        Vec{2, Float64}((1.0, 0.0)),
        Vec{2, Float64}((0.0, 1.0)),
        Vec{2, Float64}((0.0, 0.0)),
        Vec{2, Float64}((1 / 3, 1 / 3)),
    ]
end

function reference_shape_value(ip::BubbleEnrichedLagrange{RefTriangle, 1}, ξ::Vec{2}, i::Int)
    ξ_x = ξ[1]
    ξ_y = ξ[2]
    i == 1 && return ξ_x * (9ξ_y^2 + 9ξ_x * ξ_y - 9ξ_y + 1)
    i == 2 && return ξ_y * (9ξ_x^2 + 9ξ_x * ξ_y - 9ξ_x + 1)
    i == 3 && return 9ξ_x^2 * ξ_y + 9ξ_x * ξ_y^2 - 9ξ_x * ξ_y - ξ_x - ξ_y + 1
    i == 4 && return 27ξ_x * ξ_y * (1 - ξ_x - ξ_y)
    throw(ArgumentError("no shape function $i for interpolation $ip"))
end
