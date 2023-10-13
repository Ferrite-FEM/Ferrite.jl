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
    return PointValues(T, ip_fun, ip_geo)
end
function PointValues(ip::Interpolation, ipg::Interpolation = default_geometric_interpolation(ip))
    return PointValues(Float64, ip, ipg)
end
function PointValues(::Type{T}, ip::IP, ipg::GIP = default_geometric_interpolation(ip)) where {
    T, dim, shape <: AbstractRefShape{dim},
    IP  <: Interpolation{shape},
    GIP <: Interpolation{shape}
}
    qr = QuadratureRule{shape, T}([one(T)], [zero(Vec{dim, T})])
    cv = CellValues(T, qr, ip, ipg)
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
    precompute_values!(pv.cv.fun_values, pv.cv.qr) # See Issue #763, should also update dMdξ!, but separate issue
    # Regular reinit
    reinit!(pv.cv, x)
    return nothing
end

# Optimized version of PointValues which avoids i) recomputation of dNdξ and
# ii) recomputation of dNdx. Only allows function evaluation (no gradients) which is
# what is used in evaluate_at_points.
struct PointValuesInternal{IP, N_t} <: AbstractValues
    N::Vector{N_t}
    ip::IP
end

function PointValuesInternal(ξ::Vec{dim, T}, ip::IP) where {dim, T, shape <: AbstractRefShape{dim}, IP <: Interpolation{shape}}
    n_func_basefuncs = getnbasefunctions(ip)
    N = [shape_value(ip, ξ, i) for i in 1:n_func_basefuncs]
    return PointValuesInternal{IP, eltype(N)}(N, ip)
end

getnquadpoints(pv::PointValuesInternal) = 1
shape_value_type(::PointValuesInternal{<:Any, N_t}) where {N_t} = N_t
shape_value(pv::PointValuesInternal, qp::Int, i::Int) = (@assert qp == 1; pv.N[i])

# allow on-the-fly updating
function reinit!(pv::PointValuesInternal{IP}, coord::Vec{dim}) where {dim, shape <: AbstractRefShape{dim}, IP <: Interpolation{shape}}
    n_func_basefuncs = getnbasefunctions(pv.ip)
    for i in 1:n_func_basefuncs
        pv.N[i] = shape_value(pv.ip, coord, i)
    end
    return nothing
end

function Base.show(io::IO, d::MIME"text/plain", cv::PointValues)
    println(io, "PointValues containing a")
    show(io, d, cv.cv)
end
