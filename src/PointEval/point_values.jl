
struct PointScalarValues{dim,T<:Real,refshape<:AbstractRefShape} <: CellValues{dim,T,refshape}
    N::Vector{T}
end

function PointScalarValues(quad_rule::QuadratureRule, func_interpol::Interpolation)
    PointScalarValues(Float64, quad_rule, func_interpol)
end

function PointScalarValues(::Type{T}, quad_rule::QuadratureRule{dim,shape}, func_interpol::Interpolation) where {dim,T,shape<:AbstractRefShape}

    length(getweights(quad_rule)) == 1 || error("PointScalarValues supports only a single point.")

    # Function interpolation
    n_func_basefuncs = getnbasefunctions(func_interpol)
    N    = fill(zero(T)          * T(NaN), n_func_basefuncs)

    ξ = quad_rule.points[1]
    for i in 1:n_func_basefuncs
        N[i] = value(func_interpol, i, ξ)
    end

    PointScalarValues{dim,T,shape}(N)
end

# PointScalarValues only have one quadrature point anyways
function PointScalarValues(coord::Vec{dim,T}, ip::Interpolation{dim, refshape}) where {dim,refshape,T}
    qr = QuadratureRule{dim,refshape,T}([one(T)], [coord])
    return PointScalarValues(qr, ip)
end

# allow to use function_value with any
Base.@pure _valuetype(::PointScalarValues{dim}, ::Vector{T}) where {dim, T<:AbstractTensor} = T

# allow on-the-fly updating
function reinit!(pv::PointScalarValues{dim,T,refshape}, coord::Vec{dim,T}, func_interpol::Interpolation{dim,refshape,order}) where {dim,T,refshape,order}
    n_func_basefuncs = getnbasefunctions(func_interpol)
    for i in 1:n_func_basefuncs
        pv.N[i] = value(func_interpol, i, coord)
    end
    return pv
end
# TODO: need a show method for PointScalarValues