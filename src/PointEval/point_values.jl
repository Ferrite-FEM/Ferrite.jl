# Optimized version of PointScalarValues which avoids i) recomputation of dNdξ and
# ii) recomputation of dNdx. Only allows function evaluation (no gradients) which is
# what is used in get_point_values.
struct PointScalarValuesInternal{dim,T<:Real,refshape<:AbstractRefShape} <: CellValues{dim,T,refshape}
    N::Vector{T}
end

function Base.show(io::IO, ::MIME"text/plain", pv::PointScalarValuesInternal)
    print(io, "$(typeof(pv)) with $(getnbasefunctions(pv)) shape functions.")
end

function PointScalarValuesInternal(quad_rule::QuadratureRule, func_interpol::Interpolation)
    PointScalarValuesInternal(Float64, quad_rule, func_interpol)
end

function PointScalarValuesInternal(::Type{T}, quad_rule::QuadratureRule{dim,shape}, func_interpol::Interpolation) where {dim,T,shape<:AbstractRefShape}

    length(getweights(quad_rule)) == 1 || error("PointScalarValuesInternal supports only a single point.")

    # Function interpolation
    n_func_basefuncs = getnbasefunctions(func_interpol)
    N    = fill(zero(T)          * T(NaN), n_func_basefuncs)

    ξ = quad_rule.points[1]
    for i in 1:n_func_basefuncs
        N[i] = value(func_interpol, i, ξ)
    end

    PointScalarValuesInternal{dim,T,shape}(N)
end

# PointScalarValuesInternal only have one quadrature point anyways
function PointScalarValuesInternal(coord::Vec{dim,T}, ip::Interpolation{dim, refshape}) where {dim,refshape,T}
    qr = QuadratureRule{dim,refshape,T}([one(T)], [coord])
    return PointScalarValuesInternal(qr, ip)
end

getnquadpoints(pv::PointScalarValuesInternal) = 1

# allow to use function_value with any
_valuetype(::PointScalarValuesInternal{dim}, ::Vector{T}) where {dim, T<:AbstractTensor} = T

# allow on-the-fly updating
function reinit!(pv::PointScalarValuesInternal{dim,T,refshape}, coord::Vec{dim,T}, func_interpol::Interpolation{dim,refshape}) where {dim,T,refshape}
    n_func_basefuncs = getnbasefunctions(func_interpol)
    for i in 1:n_func_basefuncs
        pv.N[i] = value(func_interpol, i, coord)
    end
    return pv
end

struct PointScalarValues{D,T,R,CV,IP} <: CellValues{D,T,R}
    cv::CV
    ip::IP
end
function PointScalarValues(cv::CV, ip::IP) where {D,T,R,CV<:CellValues{D,T,R},IP<:Interpolation{D,R}}
    return PointScalarValues{D,T,R,CV,IP}(cv, ip)
end
PointScalarValues(ip::Interpolation, ipg::Interpolation=ip) = PointScalarValues(Float64, ip, ipg)
PointScalarValues(cv::CellValues{D,T}) where {D,T} = PointScalarValues(T, cv.func_interp, cv.geo_interp)
function PointScalarValues(::Type{T}, ip::Interpolation{D,R}, ipg::Interpolation = ip) where {T,D,R}
    qr = QuadratureRule{D,R,T}([one(T)], [zero(Vec{D,T})])
    cv = CellScalarValues(qr, ip, ipg)
    return PointScalarValues(cv, ip)
end

struct PointVectorValues{D,T,R,CV,IP} <: CellValues{D,T,R}
    cv::CV
    ip::IP
end
function PointVectorValues(cv::CV, ip::IP) where {D,T,R,CV<:CellValues{D,T,R},IP<:Interpolation{D,R}}
    return PointVectorValues{D,T,R,CV,IP}(cv, ip)
end
PointVectorValues(ip::Interpolation, ipg::Interpolation=ip) = PointVectorValues(Float64, ip, ipg)
PointVectorValues(cv::CellValues{D,T}) where {D,T} = PointVectorValues(T, cv.func_interp, cv.geo_interp)
function PointVectorValues(::Type{T}, ip::Interpolation{D,R}, ipg = ip) where {T,D,R}
    qr = QuadratureRule{D,R,T}([one(T)], [zero(Vec{D,T})])
    cv = CellVectorValues(qr, ip, ipg)
    return PointVectorValues(cv, ip)
end

"""
    PointScalarValues(cv::CellScalarValues)
    PointScalarValues(ip_f::Interpolation, ip_g::Interpolation=ip_f)

    PointVectorValues(cv::CellVectorValues)
    PointVectorValues(ip_f::Interpolation, ip_g::Interpolation=ip_f)

Similar to `CellScalarValues` and `CellVectorValues` but with a single updateable
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
const PointValues = Union{PointScalarValues, PointVectorValues}

# Functions used by function_(value|gradient)
getnbasefunctions(pv::PointValues) = getnbasefunctions(pv.cv)
@propagate_inbounds shape_value(pv::PointValues, qp::Int, i::Int) = shape_value(pv.cv, qp, i)
@propagate_inbounds shape_gradient(pv::PointValues, qp::Int, i::Int) = shape_gradient(pv.cv, qp, i)
getnquadpoints(pv::PointValues) = 1

# PointValues can default to quadrature point 1
function_value(pv::PointValues, u::AbstractVector, args...) =
    function_value(pv, 1, u, args...)
function_gradient(pv::PointValues, u::AbstractVector, args...) =
    function_gradient(pv, 1, u, args...)
function_symmetric_gradient(pv::PointValues, u::AbstractVector, args...) =
    function_symmetric_gradient(pv, 1, u, args...)

# reinit! on PointScalarValues must first update N and dNdξ for the new "quadrature point"
# and then call the regular reinit! for the wrapped CellValues to update dNdx
function reinit!(pv::PointScalarValues{D}, x::AbstractVector{V}, ξ::V) where {D,V<:Vec{D}}
    qp = 1 # PointScalarValues only have a single qp
    # TODO: Does M need to be updated too?
    for i in 1:getnbasefunctions(pv.ip)
        pv.cv.dNdξ[i, qp], pv.cv.N[i, qp] = gradient(ξ -> value(pv.ip, i, ξ), ξ, :all)
    end
    reinit!(pv.cv, x)
    return pv
end
function reinit!(pv::PointVectorValues{D,T}, x::AbstractVector{V}, ξ::V) where {D,T,V<:Vec{D}}
    qp = 1 # PointVectorValues only have a single qp
    basefunc_count = 1
    # TODO: Does M need to be updated too?
    for i in 1:getnbasefunctions(pv.ip)
        dNdξ_temp, N_temp = gradient(ξ -> value(pv.ip, i, ξ), ξ, :all)
        for comp in 1:D
            pv.cv.N[basefunc_count, qp] = Vec{D,T}(i -> i == comp ? N_temp : zero(T))
            pv.cv.dNdξ[basefunc_count, qp] = Tensor{2,D,T}((i,j) -> i == comp ? dNdξ_temp[j] : zero(T))
            basefunc_count += 1
        end
    end
    reinit!(pv.cv, x)
    return pv
end
