# Helpers to get the correct types for FunctionValues for the given function and, if needed, geometric interpolations. 
struct SInterpolationDims{rdim,sdim} end
struct VInterpolationDims{rdim,sdim,vdim} end
function InterpolationDims(::ScalarInterpolation, ip_geo::VectorizedInterpolation{sdim}) where sdim
    return SInterpolationDims{getdim(ip_geo),sdim}()
end
function InterpolationDims(::VectorInterpolation{vdim}, ip_geo::VectorizedInterpolation{sdim}) where {vdim,sdim}
    return VInterpolationDims{getdim(ip_geo),sdim,vdim}()
end

typeof_N(::Type{T}, ::SInterpolationDims) where T = T
typeof_N(::Type{T}, ::VInterpolationDims{<:Any,dim,dim}) where {T,dim} = Vec{dim,T}
typeof_N(::Type{T}, ::VInterpolationDims{<:Any,<:Any,vdim}) where {T,vdim} = SVector{vdim,T} # Why not ::Vec here?

typeof_dNdx(::Type{T}, ::SInterpolationDims{dim,dim}) where {T,dim} = Vec{dim,T}
typeof_dNdx(::Type{T}, ::SInterpolationDims{<:Any,sdim}) where {T,sdim} = SVector{sdim,T} # Why not ::Vec here?
typeof_dNdx(::Type{T}, ::VInterpolationDims{dim,dim,dim}) where {T,dim} = Tensor{2,dim,T}
typeof_dNdx(::Type{T}, ::VInterpolationDims{<:Any,sdim,vdim}) where {T,sdim,vdim} = SMatrix{vdim,sdim,T} # If vdim=sdim!=rdim Tensor would be possible...

typeof_dNdξ(::Type{T}, ::SInterpolationDims{dim,dim}) where {T,dim} = Vec{dim,T}
typeof_dNdξ(::Type{T}, ::SInterpolationDims{rdim}) where {T,rdim} = SVector{rdim,T} # Why not ::Vec here?
typeof_dNdξ(::Type{T}, ::VInterpolationDims{dim,dim,dim}) where {T,dim} = Tensor{2,dim,T}
typeof_dNdξ(::Type{T}, ::VInterpolationDims{rdim,<:Any,vdim}) where {T,rdim,vdim} = SMatrix{vdim,rdim,T} # If vdim=rdim!=sdim Tensor would be possible...

struct FunctionValues{IP, N_t, dNdx_t, dNdξ_t}
    N::Matrix{N_t} 
    dNdx::Matrix{dNdx_t}
    dNdξ::Matrix{dNdξ_t}
    ip::IP
end
function FunctionValues(::Type{T}, ip::Interpolation, qr::QuadratureRule, ip_geo::VectorizedInterpolation) where T
    ip_dims = InterpolationDims(ip, ip_geo)
    N_t = typeof_N(T, ip_dims)
    dNdx_t = typeof_dNdx(T, ip_dims)
    dNdξ_t = typeof_dNdξ(T, ip_dims)
    n_shape = getnbasefunctions(ip)
    n_qpoints = getnquadpoints(qr)
    
    N    = zeros(N_t,    n_shape, n_qpoints)
    dNdξ = zeros(dNdξ_t, n_shape, n_qpoints)
    dNdx = fill(zero(dNdx_t) * T(NaN), n_shape, n_qpoints)
    fv = FunctionValues(N, dNdx, dNdξ, ip)
    precompute_values!(fv, qr) # Precompute N and dNdξ
    return fv
end

function precompute_values!(fv::FunctionValues, qr::QuadratureRule)
    n_shape = getnbasefunctions(fv.ip)
    for (qp, ξ) in pairs(getpoints(qr))
        for i in 1:n_shape
            fv.dNdξ[i, qp], fv.N[i, qp] = shape_gradient_and_value(fv.ip, ξ, i)
        end
    end
end

getnbasefunctions(funvals::FunctionValues) = size(funvals.N, 1)
@propagate_inbounds shape_value(funvals::FunctionValues, q_point::Int, base_func::Int) = funvals.N[base_func, q_point]
@propagate_inbounds shape_gradient(funvals::FunctionValues, q_point::Int, base_func::Int) = funvals.dNdx[base_func, q_point]
@propagate_inbounds shape_symmetric_gradient(funvals::FunctionValues, q_point::Int, base_func::Int) = symmetric(shape_gradient(funvals, q_point, base_func))


# Hotfix to get the dots right for embedded elements until mixed tensors are merged.
# Scalar/Vector interpolations with sdim == rdim (== vdim)
@inline dothelper(A, B) = A ⋅ B
# Vector interpolations with sdim == rdim != vdim
@inline dothelper(A::SMatrix{vdim, dim}, B::Tensor{2, dim}) where {vdim, dim} = A * SMatrix{dim, dim}(B)
# Scalar interpolations with sdim > rdim
@inline dothelper(A::SVector{rdim}, B::SMatrix{rdim, sdim}) where {rdim, sdim} = B' * A
# Vector interpolations with sdim > rdim
@inline dothelper(B::SMatrix{vdim, rdim}, A::SMatrix{rdim, sdim}) where {vdim, rdim, sdim} = B * A

function apply_mapping!(funvals::FunctionValues, q_point::Int, Jinv)
    @inbounds for j in 1:getnbasefunctions(funvals)
        #funvals.dNdx[j, q_point] = funvals.dNdξ[j, q_point] ⋅ Jinv # TODO via Tensors.jl
        funvals.dNdx[j, q_point] = dothelper(funvals.dNdξ[j, q_point], Jinv)
    end
    return nothing
end