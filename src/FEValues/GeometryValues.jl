struct GeometryValues{dMdξ_t, GIP, T}
    M::Matrix{T}
    dMdξ::Matrix{dMdξ_t}
    detJdV::Vector{T}
    ip::GIP
end
function GeometryValues(::Type{T}, ip::ScalarInterpolation, qr::QuadratureRule) where T
    n_shape = getnbasefunctions(ip)
    n_qpoints = getnquadpoints(qr)
    VT = Vec{getdim(ip),T}
    M    = zeros(T,  n_shape, n_qpoints)
    dMdξ = zeros(VT, n_shape, n_qpoints)
    for (qp, ξ) in pairs(getpoints(qr))
        for i in 1:n_shape
            dMdξ[i, qp], M[i, qp] = shape_gradient_and_value(ip, ξ, i)
        end
    end
    detJdV::Vector{T} = fill(T(NaN), n_qpoints)
    return GeometryValues(M, dMdξ, detJdV, ip)
end

getngeobasefunctions(geovals::GeometryValues) = size(geovals.M, 1)
@propagate_inbounds geometric_value(geovals::GeometryValues, q_point::Int, base_func::Int) = geovals.M[base_func, q_point]
@propagate_inbounds getdetJdV(geovals::GeometryValues, q_point::Int) = geovals.detJdV[q_point]

function calculate_mapping(geo_values::GeometryValues{<:Vec{dim,T}}, q_point, w, x::AbstractVector{<:Vec{dim,T}}) where {dim,T}
    fecv_J = zero(Tensor{2,dim,T}) # zero(Tensors.getreturntype(⊗, eltype(x), eltype(geo_values.dMdξ)))
    @inbounds for j in 1:getngeobasefunctions(geo_values)
        fecv_J += x[j] ⊗ geo_values.dMdξ[j, q_point]
    end
    detJ = det(fecv_J)
    detJ > 0.0 || throw_detJ_not_pos(detJ)
    @inbounds geo_values.detJdV[q_point] = detJ*w
    return inv(fecv_J)
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
For more details see e.g. the doctoral thesis by Mirza Cenanovic **Finite element methods for surface problems* (2017), Ch. 2 **Trangential Calculus**.
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

function calculate_mapping(geo_values::GeometryValues{<:Vec{rdim,T}}, q_point, w, x::AbstractVector{<:Vec{sdim,T}}) where {rdim,sdim,T}
    n_geom_basefuncs = getngeobasefunctions(geo_values)
    fecv_J = zero(MMatrix{sdim, rdim, T}) # TODO replace with MixedTensor (see https://github.com/Ferrite-FEM/Tensors.jl/pull/188)
    for j in 1:n_geom_basefuncs
        #fecv_J += x[j] ⊗ geo_values.dMdξ[j, i] # TODO via Tensors.jl
        for k in 1:sdim, l in 1:rdim
            fecv_J[k, l] += x[j][k] * geo_values.dMdξ[j, q_point][l]
        end
    end
    fecv_J = SMatrix(fecv_J)
    detJ = embedding_det(fecv_J)
    detJ > 0.0 || throw_detJ_not_pos(detJ)
    @inbounds geo_values.detJdV[q_point] = detJ * w
    # Compute "left inverse" of J
    return pinv(fecv_J)
end