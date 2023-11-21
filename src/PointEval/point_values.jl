"""
    PointValues(cv::CellValues)
    PointValues(ip_f::Interpolation, ip_g::Interpolation=ip_f)

Similar to `CellValues` but with a single updateable
"quadrature point". `PointValues` are used for evaluation of functions/gradients in
arbitrary points of the domain together with a [`PointEvalHandler`](@ref).

`PointValues` can be created from `CellValues`, or from the interpolations directly.

`PointValues` are reinitialized like other `CellValues`, but since the local reference
coordinate of the "quadrature point" changes this needs to be passed to [`reinit!`](@ref),
in addition to the element coordinates: `reinit!(pv, coords, local_coord)`. Alternatively,
it can be reinitialized with a [`PointLocation`](@ref) when iterating a `PointEvalHandler`
with a [`PointIterator`](@ref).

For function/gradient evaluation, `PointValues` are used in the same way as
`CellValues`, i.e. by using [`function_value`](@ref), [`function_gradient`](@ref), etc,
with the exception that there is no need to specify the quadrature point index (since
`PointValues` only have 1, this is the default).
"""
struct PointValues{CV} <: AbstractValues
    cv::CV
    PointValues{CV}(cv::CV) where {CV} = new{CV}(cv)
end

function PointValues(cv::CellValues)
    T = typeof(getdetJdV(cv, 1))
    ip_fun = get_function_interpolation(cv)
    ip_geo = get_geometric_interpolation(cv)
    FunDiffOrder = get_function_difforder(cv)
    return PointValues(T, ip_fun, ip_geo; difforder=Val(FunDiffOrder))
end
function PointValues(ip::Interpolation, ipg::Interpolation = default_geometric_interpolation(ip); kwargs...)
    return PointValues(Float64, ip, ipg; kwargs...)
end
function PointValues(::Type{T}, ip::IP, ipg::GIP = default_geometric_interpolation(ip); kwargs...) where {
    T, dim, shape <: AbstractRefShape{dim},
    IP  <: Interpolation{shape},
    GIP <: Interpolation{shape}
}
    qr = QuadratureRule{shape, T}([one(T)], [zero(Vec{dim, T})])
    cv = CellValues(T, qr, ip, ipg; save_detJ=Val(false), kwargs...)
    return PointValues{typeof(cv)}(cv)
end

# Functions used by function_(value|gradient)
getnbasefunctions(pv::PointValues) = getnbasefunctions(pv.cv)
shape_value_type(pv::PointValues) = shape_value_type(pv.cv)
@propagate_inbounds shape_value(pv::PointValues, qp::Int, i::Int) = shape_value(pv.cv, qp, i)
shape_gradient_type(pv::PointValues) = shape_gradient_type(pv.cv)
@propagate_inbounds shape_gradient(pv::PointValues, qp::Int, i::Int) = shape_gradient(pv.cv, qp, i)
getnquadpoints(pv::PointValues) = 1

# PointValues can default to quadrature point 1
function_value(pv::PointValues, u::AbstractVector, args...) =
    function_value(pv, 1, u, args...)
function_gradient(pv::PointValues, u::AbstractVector, args...) =
    function_gradient(pv, 1, u, args...)
function_symmetric_gradient(pv::PointValues, u::AbstractVector, args...) =
    function_symmetric_gradient(pv, 1, u, args...)

# reinit! on PointValues must first update N and dNdξ for the new "quadrature point"
# and then call the regular reinit! for the wrapped CellValues to update dNdx
function reinit!(pv::PointValues, x::AbstractVector{<:Vec{D}}, ξ::Vec{D}) where {D}
    # Update the quadrature point location
    qr_points = getpoints(pv.cv.qr)
    qr_points[1] = ξ
    # Precompute all values again to reflect the updated ξ coordinate
    precompute_values!(pv.cv)
    # Regular reinit
    reinit!(pv.cv, x)
    return nothing
end

function Base.show(io::IO, d::MIME"text/plain", cv::PointValues)
    println(io, "PointValues containing a")
    show(io, d, cv.cv)
end
