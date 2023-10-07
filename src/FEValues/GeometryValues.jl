struct MappingValues{JT, HT<:Union{Nothing,AbstractTensor{3}}}
    J::JT # dx/dξ # Jacobian
    H::HT # dJ/dξ # Hessian
end
@inline getjacobian(mv::MappingValues) = mv.J 
@inline gethessian(mv::MappingValues{<:Any,<:AbstractTensor}) = mv.H

@inline gethessian(::MappingValues{JT,Nothing}) where JT = _make_hessian(JT)
@inline _make_hessian(::Type{Tensor{2,dim,T}}) where {dim,T} = zero(Tensor{3,dim,T})

struct RequiresHessian{B} end
RequiresHessian(B::Bool) = RequiresHessian{B}()
function RequiresHessian(ip_fun::Interpolation, ip_geo::Interpolation)
    # Leave ip_geo as input, because for later the hessian can also be avoided 
    # for fully linear geometric elements (e.g. triangle and tetrahedron)
    # This optimization is left out for now. 
    RequiresHessian(requires_hessian(get_mapping_type(ip_fun)))
end

struct GeometryValues{dMdξ_t, GIP, T, d2Mdξ2_t}
    M::Matrix{T}                # value of geometric shape function
    dMdξ::Matrix{dMdξ_t}        # gradient of geometric shape function in ref-domain
    d2Mdξ2::Matrix{d2Mdξ2_t}    # hessian of geometric shape function in ref-domain
    ip::GIP                     # geometric interpolation 
end
function GeometryValues(::Type{T}, ip::ScalarInterpolation, qr::QuadratureRule, ::RequiresHessian{RH}) where {T,RH}
    n_shape = getnbasefunctions(ip)
    n_qpoints = getnquadpoints(qr)
    VT   = Vec{getdim(ip),T}
    M    = zeros(T,  n_shape, n_qpoints)
    dMdξ = zeros(VT, n_shape, n_qpoints)
    if RH
        HT = Tensor{2,getdim(ip),T}
        dM2dξ2 = zeros(HT, n_shape, n_qpoints)
    else
        dM2dξ2 = Matrix{Nothing}(undef,0,0)
    end
    for (qp, ξ) in pairs(getpoints(qr))
        for i in 1:n_shape
            if RH
                dM2dξ2[i, qp], dMdξ[i, qp], M[i, qp] = shape_hessian_gradient_and_value(ip, ξ, i)
            else
                dMdξ[i, qp], M[i, qp] = shape_gradient_and_value(ip, ξ, i)
            end
        end
    end
    return GeometryValues(M, dMdξ, dM2dξ2, ip)
end
RequiresHessian(::GeometryValues{<:Any,<:Any,<:Any,Nothing}) = RequiresHessian(false)
RequiresHessian(::GeometryValues) = RequiresHessian(true)

getngeobasefunctions(geovals::GeometryValues) = size(geovals.M, 1)
@propagate_inbounds geometric_value(geovals::GeometryValues, q_point::Int, base_func::Int) = geovals.M[base_func, q_point]
get_geometric_interpolation(geovals::GeometryValues) = geovals.ip

@propagate_inbounds calculate_mapping(geovals::GeometryValues, args...) = calculate_mapping(RequiresHessian(geovals), geovals, args...)

@inline function calculate_mapping(::RequiresHessian{false}, geo_values::GeometryValues{<:Vec{dim,T}}, q_point, x::AbstractVector{<:Vec{dim,T}}) where {dim,T}
    fecv_J = zero(Tensor{2,dim,T}) # zero(Tensors.getreturntype(⊗, eltype(x), eltype(geo_values.dMdξ)))
    @inbounds for j in 1:getngeobasefunctions(geo_values)
        fecv_J += x[j] ⊗ geo_values.dMdξ[j, q_point]
    end
    return MappingValues(fecv_J, nothing)
end

@inline function calculate_mapping(::RequiresHessian{true}, geo_values::GeometryValues{<:Vec{dim,T}}, q_point, x::AbstractVector{<:Vec{dim,T}}) where {dim,T}
    J = zero(Tensor{2,dim,T}) # zero(Tensors.getreturntype(⊗, eltype(x), eltype(geo_values.dMdξ)))
    H = zero(Tensor{3,dim,T})
    @inbounds for j in 1:getngeobasefunctions(geo_values)
        J += x[j] ⊗ geo_values.dMdξ[j, q_point]
        H += x[j] ⊗ geo_values.d2Mdξ2[j, q_point]
    end
    return MappingValues(J, H)
end

calculate_detJ(J::Tensor{2}) = det(J)
calculate_detJ(J::SMatrix) = embedding_det(J)

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

@inline function calculate_mapping(::RequiresHessian{false}, geo_values::GeometryValues{<:Vec{rdim,T}}, q_point, x::AbstractVector{<:Vec{sdim,T}}) where {rdim,sdim,T}
    n_geom_basefuncs = getngeobasefunctions(geo_values)
    fecv_J = zero(MMatrix{sdim, rdim, T}) # TODO replace with MixedTensor (see https://github.com/Ferrite-FEM/Tensors.jl/pull/188)
    for j in 1:n_geom_basefuncs
        #fecv_J += x[j] ⊗ geo_values.dMdξ[j, i] # TODO via Tensors.jl
        for k in 1:sdim, l in 1:rdim
            fecv_J[k, l] += x[j][k] * geo_values.dMdξ[j, q_point][l]
        end
    end
    return MappingValues(SMatrix(fecv_J), nothing)
end