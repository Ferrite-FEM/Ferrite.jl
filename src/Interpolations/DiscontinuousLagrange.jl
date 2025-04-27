# TODO generalize to arbitrary basis positionings.
"""
    DiscontinuousLagrange{refshape, order} <: ScalarInterpolation

Piecewise discontinuous Lagrange basis via Gauss-Lobatto points.
The following interpolations are implemented:

* DiscontinuousLagrange{RefLine, 0}
* DiscontinuousLagrange{RefTriangle, 0}
* DiscontinuousLagrange{RefQuadrilateral, 0}
* DiscontinuousLagrange{RefTetrahedron, 0}
* DiscontinuousLagrange{RefHexahedron, 0}
"""
struct DiscontinuousLagrange{shape, order} <: ScalarInterpolation{shape, order}
    function DiscontinuousLagrange{shape, order}() where {shape <: AbstractRefShape, order}
        return new{shape, order}()
    end
end

adjust_dofs_during_distribution(::DiscontinuousLagrange) = false

getlowerorder(::DiscontinuousLagrange{shape, order}) where {shape, order} = DiscontinuousLagrange{shape, order - 1}()

getnbasefunctions(::DiscontinuousLagrange{shape, order}) where {shape, order} = getnbasefunctions(Lagrange{shape, order}())
getnbasefunctions(::DiscontinuousLagrange{shape, 0}) where {shape} = 1

# This just moves all dofs into the interior of the element.
volumedof_interior_indices(ip::DiscontinuousLagrange) = ntuple(i -> i, getnbasefunctions(ip))

# Mirror the Lagrange element for now to avoid repeating.
dirichlet_facedof_indices(ip::DiscontinuousLagrange{shape, order}) where {shape, order} = dirichlet_facedof_indices(Lagrange{shape, order}())
dirichlet_edgedof_indices(ip::DiscontinuousLagrange{shape, order}) where {shape, order} = dirichlet_edgedof_indices(Lagrange{shape, order}())
dirichlet_vertexdof_indices(ip::DiscontinuousLagrange{shape, order}) where {shape, order} = dirichlet_vertexdof_indices(Lagrange{shape, order}())

# Mirror the Lagrange element for now.
function reference_coordinates(ip::DiscontinuousLagrange{shape, order}) where {shape, order}
    return reference_coordinates(Lagrange{shape, order}())
end
function reference_shape_value(::DiscontinuousLagrange{shape, order}, ξ::Vec{dim}, i::Int) where {dim, shape <: AbstractRefShape{dim}, order}
    return reference_shape_value(Lagrange{shape, order}(), ξ, i)
end

# Excepting the L0 element.
function reference_coordinates(ip::DiscontinuousLagrange{RefHypercube{dim}, 0}) where {dim}
    return [Vec{dim, Float64}(ntuple(x -> 0.0, dim))]
end

function reference_coordinates(ip::DiscontinuousLagrange{RefTriangle, 0})
    return [Vec{2, Float64}((1 / 3, 1 / 3))]
end

function reference_coordinates(ip::DiscontinuousLagrange{RefTetrahedron, 0})
    return [Vec{3, Float64}((1 / 4, 1 / 4, 1 / 4))]
end

function reference_shape_value(ip::DiscontinuousLagrange{shape, 0}, ::Vec{dim, T}, i::Int) where {dim, shape <: AbstractRefShape{dim}, T}
    i == 1 && return one(T)
    throw(ArgumentError("no shape function $i for interpolation $ip"))
end

is_discontinuous(::Type{<:DiscontinuousLagrange}) = true
