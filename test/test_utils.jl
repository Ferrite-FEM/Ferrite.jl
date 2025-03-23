# Some utility functions for testing Ferrite

using Ferrite: reference_shape_value

#####################################
# Volume for the reference elements #
#####################################
reference_volume(::Type{Ferrite.RefHypercube{dim}}) where {dim} = 2^dim
reference_volume(::Type{Ferrite.RefSimplex{dim}}) where {dim} = 1 / factorial(dim)
reference_volume(::Type{RefPrism}) = 1 / 2
reference_volume(::Type{RefPyramid}) = 1 / 3
# For facets
reference_facet_area(::Type{Ferrite.RefHypercube{dim}}, ::Int) where {dim} = 2^(dim - 1)
reference_facet_area(::Type{RefTriangle}, facet::Int) = facet == 1 ? sqrt(2) : 1.0
reference_facet_area(::Type{RefTetrahedron}, facet::Int) = facet == 3 ? sqrt(2 * 1.5) / 2.0 : 0.5
function reference_facet_area(::Type{RefPrism}, facet::Int)
    facet == 4 && return √2
    facet ∈ [1, 5] && return 0.5
    facet ∈ [2, 3] && return 1.0
    error("Invalid facet index")
end
function reference_facet_area(::Type{RefPyramid}, facet::Int)
    facet == 1 && return 1.0
    facet ∈ [2, 3] && return 0.5
    facet ∈ [4, 5] && return sqrt(2) / 2
    error("Invalid facet index")
end

######################################################
# Coordinates and normals for the reference shapes #
######################################################

function reference_normals(::Interpolation{RefShape}) where {RefShape}
    @warn "Using reference normals of Interpolation, use of RefShape directly instead"
    return reference_normals(RefShape)
end

function reference_normals(::Type{RefLine})
    return [
        Vec{1, Float64}((-1.0,)),
        Vec{1, Float64}((1.0,)),
    ]
end
function reference_normals(::Type{RefQuadrilateral})
    return [
        Vec{2, Float64}((0.0, -1.0)),
        Vec{2, Float64}((1.0, 0.0)),
        Vec{2, Float64}((0.0, 1.0)),
        Vec{2, Float64}((-1.0, 0.0)),
    ]
end

function reference_normals(::Type{RefTriangle})
    return [
        Vec{2, Float64}((1 / √2, 1 / √2)),
        Vec{2, Float64}((-1.0, 0.0)),
        Vec{2, Float64}((0.0, -1.0)),
    ]
end

function reference_normals(::Type{RefTetrahedron})
    return [
        Vec{3, Float64}((0.0, 0.0, -1.0)),
        Vec{3, Float64}((0.0, -1.0, 0.0)),
        Vec{3, Float64}((1 / √3, 1 / √3, 1 / √3)),
        Vec{3, Float64}((-1.0, 0.0, 0.0)),
    ]
end

function reference_normals(::Type{RefHexahedron})
    return [
        Vec{3, Float64}((0.0, 0.0, -1.0)),
        Vec{3, Float64}((0.0, -1.0, 0.0)),
        Vec{3, Float64}((1.0, 0.0, 0.0)),
        Vec{3, Float64}((0.0, 1.0, 0.0)),
        Vec{3, Float64}((-1.0, 0.0, 0.0)),
        Vec{3, Float64}((0.0, 0.0, 1.0)),
    ]
end

function reference_normals(::Type{RefPrism})
    return [
        Vec{3, Float64}((0.0, 0.0, -1.0)),
        Vec{3, Float64}((0.0, -1.0, 0.0)),
        Vec{3, Float64}((-1.0, 0.0, 0.0)),
        Vec{3, Float64}((1 / √2, 1 / √2, 0.0)),
        Vec{3, Float64}((0.0, 0.0, 1.0)),
    ]
end

function reference_normals(::Type{RefPyramid})
    return [
        Vec{3, Float64}((0.0, 0.0, -1.0)),
        Vec{3, Float64}((0.0, -1.0, 0.0)),
        Vec{3, Float64}((-1.0, 0.0, 0.0)),
        Vec{3, Float64}((1 / √2, 0.0, 1 / √2)),
        Vec{3, Float64}((0.0, 1 / √2, 1 / √2)),
    ]
end

##################################
# Valid coordinates by expanding #
# and rotating reference shape   #
##################################
function rotmat(dim, θ = π / 6)
    if dim == 1
        R = Tensor{2, 1}((cos(θ),))
        return R
    elseif dim == 2
        R = Tensor{2, 2}((cos(θ), sin(θ), -sin(θ), cos(θ)))
        return R
    else
        u = Vec{3}((1.0, 2.0, 3.0)); u /= norm(u)
        ux = Tensor{2, 3}((0.0, u[3], -u[2], -u[3], 0.0, u[1], u[2], -u[1], 0.0))
        R = cos(θ) * one(Tensor{2, 3}) + sin(θ) * ux + (1 - cos(θ)) * u ⊗ u
        return R
    end
end

function valid_coordinates_and_normals(fs::Interpolation{shape, order}) where {dim, shape <: Ferrite.AbstractRefShape{dim}, order}
    x = Ferrite.reference_coordinates(fs)
    n = reference_normals(shape)
    R = rotmat(dim)
    return [2.0 * (R ⋅ x[i]) for i in 1:length(x)], [(R ⋅ n[i]) / norm((R ⋅ n[i])) for i in 1:length(n)]
end

#######################################
# Volume of cells (with planar edges) #
#######################################

calculate_volume(ip::VectorizedInterpolation, x) = calculate_volume(ip.ip, x)

function calculate_volume(::Lagrange{RefLine, 1}, x::Vector{Vec{dim, T}}) where {T, dim}
    vol = norm(x[2] - x[1])
    return vol
end

function calculate_volume(::Lagrange{RefLine, 2}, x::Vector{Vec{dim, T}}) where {T, dim}
    vol = norm(x[3] - x[1]) + norm(x[2] - x[3])
    return vol
end

function calculate_volume(::Lagrange{RefQuadrilateral, 1}, x::Vector{Vec{dim, T}}) where {T, dim}
    vol = norm((x[4] - x[1]) × (x[2] - x[1])) * 0.5 +
        norm((x[4] - x[3]) × (x[2] - x[3])) * 0.5
    return vol
end

function calculate_volume(::Lagrange{RefQuadrilateral, 2}, x::Vector{Vec{dim, T}}) where {T, dim}
    vol = norm((x[8] - x[1]) × (x[5] - x[1])) * 0.5 +
        norm((x[8] - x[9]) × (x[5] - x[9])) * 0.5 +
        norm((x[5] - x[2]) × (x[6] - x[2])) * 0.5 +
        norm((x[5] - x[9]) × (x[6] - x[9])) * 0.5 +
        norm((x[6] - x[3]) × (x[7] - x[3])) * 0.5 +
        norm((x[6] - x[9]) × (x[7] - x[9])) * 0.5 +
        norm((x[7] - x[4]) × (x[8] - x[4])) * 0.5 +
        norm((x[7] - x[9]) × (x[8] - x[9])) * 0.5
    return vol
end

function calculate_volume(::Lagrange{RefTriangle, 1}, x::Vector{Vec{dim, T}}) where {T, dim}
    vol = norm((x[1] - x[3]) × (x[2] - x[3])) * 0.5
    return vol
end

function calculate_volume(::Lagrange{RefTriangle, 2}, x::Vector{Vec{dim, T}}) where {T, dim}
    vol = norm((x[6] - x[3]) × (x[5] - x[3])) * 0.5 +
        norm((x[6] - x[4]) × (x[5] - x[4])) * 0.5 +
        norm((x[1] - x[6]) × (x[4] - x[6])) * 0.5 +
        norm((x[4] - x[5]) × (x[2] - x[5])) * 0.5
    return vol
end

# TODO: Only correct for linear sides
function calculate_volume(::Lagrange{RefTriangle, O}, x::Vector{Vec{dim, T}}) where {T, dim, O}
    vol = norm((x[1] - x[3]) × (x[2] - x[3])) * 0.5
    return vol
end

function calculate_volume(::Lagrange{RefTetrahedron, order}, x::Vector{Vec{3, T}}) where {T, order}
    vol = norm((x[2] - x[1]) ⋅ ((x[3] - x[1]) × (x[4] - x[1]))) / 6.0
    return vol
end

function calculate_volume(::Lagrange{RefHexahedron, 1}, x::Vector{Vec{3, T}}) where {T}
    vol = norm((x[1] - x[5]) ⋅ ((x[2] - x[5]) × (x[4] - x[5]))) / 6.0 +
        norm((x[2] - x[7]) ⋅ ((x[3] - x[7]) × (x[4] - x[7]))) / 6.0 +
        norm((x[2] - x[7]) ⋅ ((x[4] - x[7]) × (x[5] - x[7]))) / 6.0 +
        norm((x[2] - x[7]) ⋅ ((x[5] - x[7]) × (x[6] - x[7]))) / 6.0 +
        norm((x[4] - x[8]) ⋅ ((x[5] - x[8]) × (x[7] - x[8]))) / 6.0
    return vol
end

function calculate_volume(::Lagrange{RefPrism, order}, x::Vector{Vec{3, T}}) where {T, order}
    vol = norm((x[4] - x[1]) ⋅ ((x[2] - x[1]) × (x[3] - x[1]))) / 2.0
    return vol
end

function calculate_volume(::Lagrange{RefPyramid, order}, x::Vector{Vec{3, T}}) where {T, order}
    vol = norm((x[5] - x[1]) ⋅ ((x[2] - x[1]) × (x[3] - x[1]))) / 3.0
    return vol
end

function calculate_volume(::Serendipity{RefQuadrilateral, 2}, x::Vector{Vec{2, T}}) where {T}
    vol = norm((x[5] - x[1]) × (x[8] - x[1])) * 0.5 +
        norm((x[6] - x[2]) × (x[5] - x[2])) * 0.5 +
        norm((x[7] - x[3]) × (x[6] - x[3])) * 0.5 +
        norm((x[8] - x[4]) × (x[7] - x[4])) * 0.5 +
        norm((x[6] - x[5]) × (x[8] - x[5])) * 0.5 +
        norm((x[6] - x[7]) × (x[8] - x[7])) * 0.5
    return vol
end

function calculate_facet_area(ip::Union{Lagrange{RefLine}, DiscontinuousLagrange{RefLine}}, x::Vector{<:Vec}, faceindex::Int)
    return one(eltype(eltype(x)))
end
function calculate_facet_area(ip::Union{Lagrange{RefQuadrilateral, order}, DiscontinuousLagrange{RefQuadrilateral, order}}, x::Vector{<:Vec}, faceindex::Int) where {order}
    return calculate_volume(Lagrange{RefLine, order}(), x)
end
function calculate_facet_area(ip::Union{Lagrange{RefTriangle, order}, DiscontinuousLagrange{RefTriangle, order}}, x::Vector{<:Vec}, faceindex::Int) where {order}
    return calculate_volume(Lagrange{RefLine, order}(), x)
end
function calculate_facet_area(ip::Union{Lagrange{RefHexahedron, order}, DiscontinuousLagrange{RefHexahedron, order}}, x::Vector{<:Vec}, faceindex::Int) where {order}
    return calculate_volume(Lagrange{RefQuadrilateral, order}(), x)
end
function calculate_facet_area(ip::Serendipity{RefQuadrilateral, order}, x::Vector{<:Vec}, faceindex::Int) where {order}
    return calculate_volume(Lagrange{RefLine, order}(), x)
end
function calculate_facet_area(p::Union{Lagrange{RefTetrahedron, order}, DiscontinuousLagrange{RefTetrahedron, order}}, x::Vector{<:Vec}, faceindex::Int) where {order}
    return calculate_volume(Lagrange{RefTriangle, order}(), x)
end
function calculate_facet_area(p::Union{Lagrange{RefPrism, order}, DiscontinuousLagrange{RefPrism, order}}, x::Vector{<:Vec}, faceindex::Int) where {order}
    faceindex ∈ [1, 5] && return calculate_volume(Lagrange{RefTriangle, order}(), x)
    return calculate_volume(Lagrange{RefQuadrilateral, order}(), x)
end
function calculate_facet_area(p::Union{Lagrange{RefPyramid, order}, DiscontinuousLagrange{RefPyramid, order}}, x::Vector{<:Vec}, faceindex::Int) where {order}
    faceindex != 1 && return calculate_volume(Lagrange{RefTriangle, order}(), x)
    return calculate_volume(Lagrange{RefQuadrilateral, order}(), x)
end

coords_on_faces(x, ::Lagrange{RefLine, 1}) = ([x[1]], [x[2]])
coords_on_faces(x, ::Lagrange{RefLine, 2}) = ([x[1]], [x[2]])
coords_on_faces(x, ::Lagrange{RefQuadrilateral, 1}) =
    ([x[1], x[2]], [x[2], x[3]], [x[3], x[4]], [x[1], x[4]])
coords_on_faces(x, ::Lagrange{RefQuadrilateral, 2}) =
    ([x[1], x[2], x[5]], [x[2], x[3], x[6]], [x[3], x[4], x[7]], [x[1], x[4], x[8]])
coords_on_faces(x, ::Lagrange{RefTriangle, 1}) =
    ([x[1], x[2]], [x[2], x[3]], [x[1], x[3]])
coords_on_faces(x, ::Lagrange{RefTriangle, 2}) =
    ([x[1], x[2], x[4]], [x[2], x[3], x[5]], [x[1], x[3], x[6]])
coords_on_faces(x, ::Lagrange{RefTetrahedron, 1}) =
    ([x[1], x[2], x[3]], [x[1], x[2], x[4]], [x[2], x[3], x[4]], [x[1], x[3], x[4]])
coords_on_faces(x, ::Lagrange{RefTetrahedron, 2}) =
    ([x[1], x[2], x[3], x[5], x[6], x[7]], [x[1], x[2], x[4], x[5], x[8], x[9]], [x[2], x[3], x[4], x[6], x[9], x[10]], [x[1], x[3], x[4], x[7], x[8], x[10]])
coords_on_faces(x, ::Lagrange{RefHexahedron, 1}) =
    ([x[1], x[2], x[3], x[4]], [x[1], x[2], x[5], x[6]], [x[2], x[3], x[6], x[7]], [x[3], x[4], x[7], x[8]], [x[1], x[4], x[5], x[8]], [x[5], x[6], x[7], x[8]])
coords_on_faces(x, ::Serendipity{RefHexahedron, 2}) =
    ([x[1], x[2], x[5]], [x[2], x[3], x[6]], [x[3], x[4], x[7]], [x[1], x[4], x[8]])

check_equal_or_nan(a::Any, b::Any) = a == b || (isnan(a) && isnan(b))
check_equal_or_nan(a::Union{Tensor, Array}, b::Union{Tensor, Array}) = all(check_equal_or_nan.(a, b))

######################################################
# Helpers for testing facet_to_element_transformation #
######################################################
getfacetrefshape(::Type{<:Ferrite.AbstractRefShape{2}}, ::Int) = RefLine
getfacetrefshape(::Type{<:RefHexahedron}, ::Int) = RefQuadrilateral
getfacetrefshape(::Type{RefTetrahedron}, ::Int) = RefTriangle
getfacetrefshape(::Type{RefPyramid}, facet::Int) = facet == 1 ? RefQuadrilateral : RefTriangle
getfacetrefshape(::Type{RefPrism}, facet::Int) = facet ∈ (1, 5) ? RefTriangle : RefQuadrilateral

function perturb_standard_grid!(grid::Ferrite.AbstractGrid{dim}, strength) where {dim}
    function perturb(x::Vec{dim}) where {dim}
        for d in 1:dim
            if x[d] ≈ 1.0 || x[d] ≈ -1.0
                return x
            end
        end
        return x + Vec{dim}(0.5 * strength .* (2 .* rand(Vec{dim}) .- 1.0))
    end
    return transform_coordinates!(grid, perturb)
end

######################################################
# Dummy RefShape to test get_transformation_matrix   #
######################################################
module DummyRefShapes
    import Ferrite
    struct RefDodecahedron <: Ferrite.AbstractRefShape{3} end
    function Ferrite.reference_faces(::Type{RefDodecahedron})
        return (
            (1, 5, 4, 3, 2),
        )
    end
end

# Hypercube is simply ⨂ᵈⁱᵐ Line :)
sample_random_point(::Type{Ferrite.RefHypercube{ref_dim}}) where {ref_dim} = Vec{ref_dim}(ntuple(_ -> 2.0 * rand() - 1.0, ref_dim))

# Dirichlet type sampling
#
# The idea behind this sampling is that the d-Simplex (i.e. a generalized triangle in d dimensions)
# is just a surface in d+1 dimensions, that can be characterized by two constraints:
# 1. All coordinates are between 0 and 1
# 2. The sum of all coordinates is exactly 1
# This way we can just sample from the d+1 dimensional hypercube, transform the hypercube
# logarithmically to get a "normal distribution" over our simplex and enforce that the coordinates
# sum to 1. By dropping the last coordinate in this sample we finally obtain d numbers which lies in
# the d-simplex.
#
# A nice geometric sketch of this process is given in this stackexchange post: https://stats.stackexchange.com/a/296779
function sample_random_point(::Type{Ferrite.RefSimplex{ref_dim}}) where {ref_dim} # Note that "ref_dim = d" in the text above
    ξₜ = ntuple(_ -> -log(rand()), ref_dim + 1)
    return Vec{ref_dim}(ntuple(i -> ξₜ[i], ref_dim) ./ sum(ξₜ))
end

# Wedge = Triangle ⊗ Line
function sample_random_point(::Type{RefPrism})
    (ξ₁, ξ₂) = sample_random_point(RefTriangle)
    ξ₃ = rand(Float64)
    return Vec{3}((ξ₁, ξ₂, ξ₃))
end

# TODO what to do here? The samplig is not uniform...
function sample_random_point(::Type{RefPyramid})
    ξ₃ = (1 - 1.0e-3) * rand(Float64) # Derivative is discontinuous at the top
    # If we fix a z coordinate we get a Quad with extends (1-ξ₃)
    (ξ₁, ξ₂) = (1.0 - ξ₃) .* Vec{2}(ntuple(_ -> rand(), 2))
    return Vec{3}((ξ₁, ξ₂, ξ₃))
end

############################################################
# Inverse parametric mapping ξ = ϕ(x) for testing hessians #
############################################################
function function_value_from_physical_coord(interpolation::Interpolation, cell_coordinates, X::Vec{dim, T}, ue) where {dim, T}
    n_basefuncs = getnbasefunctions(interpolation)
    scalar_ip = interpolation isa Ferrite.ScalarInterpolation ? interpolation : interpolation.ip
    @assert length(ue) == n_basefuncs
    _, ξ = Ferrite.find_local_coordinate(scalar_ip, cell_coordinates, X, Ferrite.NewtonLineSearchPointFinder(residual_tolerance = 1.0e-16))
    u = zero(reference_shape_value(interpolation, ξ, 1))
    for j in 1:n_basefuncs
        N = reference_shape_value(interpolation, ξ, j)
        u += N * ue[j]
    end
    return u
end
