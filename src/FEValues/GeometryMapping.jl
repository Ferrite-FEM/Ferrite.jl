"""
    MappingValues(J, H)

The mapping values are calculated based on a
`geometric_mapping::GeometryMapping` along with the cell coordinates,
and the stored jacobian, `J`, and potentially hessian, `H`, are
used when mapping the `FunctionValues` to the current cell during `reinit!`.
"""
MappingValues

struct MappingValues{JT, HT}
    J::JT # dx/dξ # Jacobian
    H::HT # dJ/dξ # Hessian
end

@inline getjacobian(mv::MappingValues{<:Union{AbstractTensor, SMatrix}}) = mv.J
@inline gethessian(mv::MappingValues{<:Any,<:AbstractTensor}) = mv.H


"""
    GeometryMapping{DiffOrder}(::Type{T}, ip_geo, qr::QuadratureRule)

Create a `GeometryMapping` object which contains the geometric

* shape values
* gradient values (if DiffOrder ≥ 1)
* hessians values (if DiffOrder ≥ 2)

`T<:AbstractFloat` gives the numeric type of the values.
"""
GeometryMapping

struct GeometryMapping{DiffOrder, IP, M_t, dMdξ_t, d2Mdξ2_t}
    ip::IP             # ::Interpolation                Geometric interpolation
    M::M_t             # ::AbstractMatrix{<:Number}     Values of geometric shape functions
    dMdξ::dMdξ_t       # ::AbstractMatrix{<:Vec}        Gradients of geometric shape functions in ref-domain
    d2Mdξ2::d2Mdξ2_t   # ::AbstractMatrix{<:Tensor{2}}  Hessians of geometric shape functions in ref-domain
                       # ::Nothing (dMdξ or d2Mdξ2 if not required)
    function GeometryMapping(
        ip::IP, M::M_t, ::Nothing, ::Nothing
        ) where {IP <: ScalarInterpolation, M_t<:AbstractMatrix{<:Number}}
        return new{0, IP, M_t, Nothing, Nothing}(ip, M, nothing, nothing)
    end
    function GeometryMapping(
        ip::IP, M::M_t, dMdξ::dMdξ_t, ::Nothing
        ) where {IP <: ScalarInterpolation, M_t<:AbstractMatrix{<:Number}, dMdξ_t <: AbstractMatrix{<:Vec}}
        return new{1, IP, M_t, dMdξ_t, Nothing}(ip, M, dMdξ, nothing)
    end
    function GeometryMapping(
        ip::IP, M::M_t, dMdξ::dMdξ_t, d2Mdξ2::d2Mdξ2_t) where
        {IP <: ScalarInterpolation, M_t<:AbstractMatrix{<:Number},
        dMdξ_t <: AbstractMatrix{<:Vec}, d2Mdξ2_t <: AbstractMatrix{<:Tensor{2}}}
        return new{2, IP, M_t, dMdξ_t, d2Mdξ2_t}(ip, M, dMdξ, d2Mdξ2)
    end
end
function GeometryMapping{0}(::Type{T}, ip::ScalarInterpolation, qr::QuadratureRule) where T
    n_shape = getnbasefunctions(ip)
    n_qpoints = getnquadpoints(qr)
    gm = GeometryMapping(ip, zeros(T, n_shape, n_qpoints), nothing, nothing)
    precompute_values!(gm, getpoints(qr))
    return gm
end
function GeometryMapping{1}(::Type{T}, ip::ScalarInterpolation, qr::QuadratureRule) where T
    n_shape = getnbasefunctions(ip)
    n_qpoints = getnquadpoints(qr)

    M    = zeros(T,                 n_shape, n_qpoints)
    dMdξ = zeros(Vec{getrefdim(ip),T}, n_shape, n_qpoints)

    gm = GeometryMapping(ip, M, dMdξ, nothing)
    precompute_values!(gm, getpoints(qr))
    return gm
end
function GeometryMapping{2}(::Type{T}, ip::ScalarInterpolation, qr::QuadratureRule) where T
    n_shape = getnbasefunctions(ip)
    n_qpoints = getnquadpoints(qr)

    M      = zeros(T,                      n_shape, n_qpoints)
    dMdξ   = zeros(Vec{getrefdim(ip),T},      n_shape, n_qpoints)
    d2Mdξ2 = zeros(Tensor{2,getrefdim(ip),T}, n_shape, n_qpoints)

    gm = GeometryMapping(ip, M, dMdξ, d2Mdξ2)
    precompute_values!(gm, getpoints(qr))
    return gm
end

function precompute_values!(gm::GeometryMapping{0}, qr_points::AbstractVector{<:Vec})
    reference_shape_values!(gm.M, gm.ip, qr_points)
end
function precompute_values!(gm::GeometryMapping{1}, qr_points::AbstractVector{<:Vec})
    reference_shape_gradients_and_values!(gm.dMdξ, gm.M, gm.ip, qr_points)
end
function precompute_values!(gm::GeometryMapping{2}, qr_points::AbstractVector{<:Vec})
    reference_shape_hessians_gradients_and_values!(gm.d2Mdξ2, gm.dMdξ, gm.M, gm.ip, qr_points)
end

function Base.copy(v::GeometryMapping)
    return GeometryMapping(copy(v.ip), copy(v.M), _copy_or_nothing(v.dMdξ), _copy_or_nothing(v.d2Mdξ2))
end

getngeobasefunctions(geo_mapping::GeometryMapping) = size(geo_mapping.M, 1)
@propagate_inbounds geometric_value(geo_mapping::GeometryMapping, q_point::Int, base_func::Int) = geo_mapping.M[base_func, q_point]
geometric_interpolation(geo_mapping::GeometryMapping) = geo_mapping.ip

# Hot-fixes to support embedded elements before MixedTensors are available
# See https://github.com/Ferrite-FEM/Tensors.jl/pull/188
@inline otimes_helper(x::Vec{dim}, dMdξ::Vec{dim}) where dim = x ⊗ dMdξ
@inline function otimes_helper(x::Vec{sdim}, dMdξ::Vec{rdim}) where {sdim, rdim}
    SMatrix{sdim,rdim}((x[i]*dMdξ[j] for i in 1:sdim, j in 1:rdim)...)
end
# End of embedded hot-fixes

# For creating initial value
function otimes_returntype(#=typeof(x)=#::Type{<:Vec{sdim,Tx}}, #=typeof(dMdξ)=#::Type{<:Vec{rdim,TM}}) where {sdim,rdim,Tx,TM}
    return SMatrix{sdim,rdim,promote_type(Tx,TM)}
end
function otimes_returntype(#=typeof(x)=#::Type{<:Vec{dim,Tx}}, #=typeof(dMdξ)=#::Type{<:Vec{dim,TM}}) where {dim, Tx, TM}
    return Tensor{2,dim,promote_type(Tx,TM)}
end
function otimes_returntype(#=typeof(x)=#::Type{<:Vec{dim,Tx}}, #=typeof(d2Mdξ2)=#::Type{<:Tensor{2,dim,TM}}) where {dim, Tx, TM}
    return Tensor{3,dim,promote_type(Tx,TM)}
end

@inline function calculate_mapping(::GeometryMapping{0}, q_point::Int, x::AbstractVector{<:Vec})
    return MappingValues(nothing, nothing)
end

@inline function calculate_mapping(geo_mapping::GeometryMapping{1}, q_point::Int, x::AbstractVector{<:Vec})
    J = zero(otimes_returntype(eltype(x), eltype(geo_mapping.dMdξ)))
    @inbounds for j in 1:getngeobasefunctions(geo_mapping)
        # J += x[j] ⊗ geo_mapping.dMdξ[j, q_point]
        J += otimes_helper(x[j], geo_mapping.dMdξ[j, q_point])
    end
    return MappingValues(J, nothing)
end

@inline function calculate_mapping(geo_mapping::GeometryMapping{2}, q_point::Int, x::AbstractVector{<:Vec})
    J = zero(otimes_returntype(eltype(x), eltype(geo_mapping.dMdξ)))
    sdim, rdim = size(J)
    (rdim != sdim) && error("hessian for embedded elements not implemented (rdim=$rdim, sdim=$sdim)")
    H = zero(otimes_returntype(eltype(x), eltype(geo_mapping.d2Mdξ2)))
    @inbounds for j in 1:getngeobasefunctions(geo_mapping)
        J += x[j] ⊗ geo_mapping.dMdξ[j, q_point]
        H += x[j] ⊗ geo_mapping.d2Mdξ2[j, q_point]
    end
    return MappingValues(J, H)
end

@inline function calculate_mapping(gip::ScalarInterpolation, ξ::Vec, x::AbstractVector{<:Vec}, ::Val{0})
    return MappingValues(nothing, nothing)
end

@inline function calculate_mapping(gip::ScalarInterpolation, ξ::Vec{rdim,T}, x::AbstractVector{<:Vec{sdim}}, ::Val{1}) where {T,rdim, sdim}
    n_basefuncs = getnbasefunctions(gip)
    @boundscheck checkbounds(x, Base.OneTo(n_basefuncs))

    J = zero(otimes_returntype(Vec{sdim,T}, Vec{rdim,T}))
    @inbounds for j in 1:n_basefuncs
        dMdξ = reference_shape_gradient(gip, ξ, j)
        # J += x[j] ⊗ dMdξ # https://github.com/Ferrite-FEM/Tensors.jl/pull/188
        J += otimes_helper(x[j], dMdξ)
    end
    return MappingValues(J, nothing)
end

@inline function calculate_mapping(gip::ScalarInterpolation, ξ::Vec{rdim,T}, x::AbstractVector{<:Vec{sdim}}, ::Val{2}) where {T,rdim, sdim}
    n_basefuncs = getnbasefunctions(gip)
    @boundscheck checkbounds(x, Base.OneTo(n_basefuncs))
    (rdim != sdim) && error("hessian for embedded elements not implemented (rdim=$rdim, sdim=$sdim)")
    J = zero(otimes_returntype(Vec{sdim,T}, Vec{rdim,T}))
    H = zero(otimes_returntype(eltype(x), typeof(J)))
    @inbounds for j in 1:n_basefuncs
        d2Mdξ2, dMdξ, _ = reference_shape_hessian_gradient_and_value(gip, ξ, j)
        J += x[j] ⊗ dMdξ
        H += x[j] ⊗ d2Mdξ2
    end
    return MappingValues(J, H)
end

calculate_detJ(J::Tensor{2}) = det(J)
calculate_detJ(J::SMatrix) = embedding_det(J)

function calculate_jacobian_and_spatial_coordinate(gip::ScalarInterpolation, ξ::Vec{rdim,Tξ}, x::AbstractVector{<:Vec{sdim, Tx}}) where {Tξ, Tx, rdim, sdim}
    n_basefuncs = getnbasefunctions(gip)
    @boundscheck checkbounds(x, Base.OneTo(n_basefuncs))

    fecv_J = zero(otimes_returntype(Vec{sdim,Tx}, Vec{rdim,Tξ}))
    sx = zero(Vec{sdim, Tx})
    @inbounds for j in 1:n_basefuncs
        dMdξ, M = reference_shape_gradient_and_value(gip, ξ, j)
        sx += M * x[j]
        fecv_J += otimes_helper(x[j], dMdξ)
    end
    return fecv_J, sx
end


# Embedded

"""
    embedding_det(J::SMatrix{3, 2})

Embedding determinant for surfaces in 3D.

TLDR: "det(J) =" ||∂x/∂ξ₁ × ∂x/∂ξ₂||₂

The transformation theorem for some function f on a 2D surface in 3D space leads to
  ∫ f ⋅ dS = ∫ f ⋅ (∂x/∂ξ₁ × ∂x/∂ξ₂) dξ₁dξ₂ = ∫ f ⋅ n ||∂x/∂ξ₁ × ∂x/∂ξ₂||₂ dξ₁dξ₂
where ||∂x/∂ξ₁ × ∂x/∂ξ₂||₂ is "detJ" and n is the unit normal.
See e.g. https://scicomp.stackexchange.com/questions/41741/integration-of-d-1-dimensional-functions-on-finite-element-surfaces for simple explanation.
For more details see e.g. the doctoral thesis by Mirza Cenanovic **Tangential Calculus** [Cenanovic2017](@cite).
"""
embedding_det(J::SMatrix{3,2}) = norm(J[:,1] × J[:,2])

"""
    embedding_det(J::Union{SMatrix{2, 1}, SMatrix{3, 1}})

Embedding determinant for curves in 2D and 3D.

TLDR: "det(J) =" ||∂x/∂ξ||₂

The transformation theorem for some function f on a 1D curve in 2D and 3D space leads to
  ∫ f ⋅ dE = ∫ f ⋅ ∂x/∂ξ dξ = ∫ f ⋅ t ||∂x/∂ξ||₂ dξ
where ||∂x/∂ξ||₂ is "detJ" and t is "the unit tangent".
See e.g. https://scicomp.stackexchange.com/questions/41741/integration-of-d-1-dimensional-functions-on-finite-element-surfaces for simple explanation.
"""
embedding_det(J::Union{SMatrix{2, 1}, SMatrix{3, 1}}) = norm(J)
