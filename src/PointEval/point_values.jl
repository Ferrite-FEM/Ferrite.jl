
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

# allow to use function_value with any
Base.@pure _valuetype(::PointScalarValues{dim}, ::Vector{T}) where {dim, T<:AbstractTensor} = T
