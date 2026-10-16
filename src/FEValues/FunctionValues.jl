#################################################################
# Note on dimensions:                                           #
# sdim = spatial dimension (dimension of the grid nodes)        #
# rdim = reference dimension (dimension in isoparametric space) #
# vdim = vector dimension (dimension of the field)              #
#################################################################
# The following internal dispatches are used to correctly preallocate fields in `FunctionValues` below
typeof_N(::Type{T}, ::ScalarInterpolation, ::VectorizedInterpolation{sdim, <:AbstractRefShape{rdim}}) where {T, sdim, rdim} = T
typeof_dNdx(::Type{T}, ::ScalarInterpolation, ::VectorizedInterpolation{sdim, <:AbstractRefShape{rdim}}) where {T, sdim, rdim} = Vec{sdim, T}
typeof_dNdξ(::Type{T}, ::ScalarInterpolation, ::VectorizedInterpolation{sdim, <:AbstractRefShape{rdim}}) where {T, sdim, rdim} = Vec{rdim, T}
typeof_d2Ndx2(::Type{T}, ::ScalarInterpolation, ::VectorizedInterpolation{sdim, <:AbstractRefShape{rdim}}) where {T, sdim, rdim} = Tensor{2, sdim, T}
typeof_d2Ndξ2(::Type{T}, ::ScalarInterpolation, ::VectorizedInterpolation{sdim, <:AbstractRefShape{rdim}}) where {T, sdim, rdim} = Tensor{2, rdim, T}

typeof_N(::Type{T}, ::VectorInterpolation{vdim}, ::VectorizedInterpolation{sdim, <:AbstractRefShape{rdim}}) where {T, vdim, sdim, rdim} = Vec{vdim, T}
typeof_dNdx(::Type{T}, ::VectorInterpolation{vdim}, ::VectorizedInterpolation{sdim, <:AbstractRefShape{rdim}}) where {T, vdim, sdim, rdim} = Tensors.regular_if_possible(MixedTensor2{vdim, sdim, T})
typeof_dNdξ(::Type{T}, ::VectorInterpolation{vdim}, ::VectorizedInterpolation{sdim, <:AbstractRefShape{rdim}}) where {T, vdim, sdim, rdim} = Tensors.regular_if_possible(MixedTensor2{vdim, rdim, T})
typeof_d2Ndx2(::Type{T}, ::VectorInterpolation{vdim}, ::VectorizedInterpolation{sdim, <:AbstractRefShape{rdim}}) where {T, vdim, sdim, rdim} = Tensors.regular_if_possible(MixedTensor3{vdim, sdim, sdim, T})
typeof_d2Ndξ2(::Type{T}, ::VectorInterpolation{vdim}, ::VectorizedInterpolation{sdim, <:AbstractRefShape{rdim}}) where {T, vdim, sdim, rdim} = Tensors.regular_if_possible(MixedTensor3{vdim, rdim, rdim, T})

"""
    FunctionValues{DiffOrder_x, DiffOrder_s}(::Type{T}, ip_fun, qr::QuadratureRule, ip_geo::VectorizedInterpolation)
    FunctionValues{DiffOrder_x}(::Type{T}, ip_fun, qr::QuadratureRule, ip_geo::VectorizedInterpolation)

Create a `FunctionValues <: AbstractValues` object containing the shape values and derivatives for both the
reference cell (precalculated) and the real cell (updated in `reinit!`). Derivatives with respect to the
spatial coordinates, ``\\mathbf{x}``, are stored up to order `DiffOrder_x`, and derivatives with respect to
the local frame coordinates, ``\\mathbf{s}``, (see [`shape_local_gradient`](@ref)) are stored up to order
`DiffOrder_s` (defaults to 0).
The user should normally not create `FunctionValues`, these are typically only created from the constructors
of `AbstractCellValues` and `AbstractFacetValues`. However, the user will interact with `fv::FunctionValues`
when indexing e.g. `cmv::MultiFieldCellValues` (e.g. `fv = cmv.u`), as `fv` supports

* [`getnbasefunctions`](@ref)
* [`shape_value`](@ref)
* [`shape_gradient`](@ref)
* [`shape_symmetric_gradient`](@ref)
* [`shape_divergence`](@ref)
* [`shape_local_gradient`](@ref)
* [`shape_local_hessian`](@ref)
* [`function_value`](@ref)
* [`function_gradient`](@ref)
* [`function_symmetric_gradient`](@ref)
* [`function_divergence`](@ref)
* [`function_local_gradient`](@ref)
* [`function_local_hessian`](@ref)
"""
FunctionValues

struct FunctionValues{DiffOrder_x, DiffOrder_s, IP, Nx_t, Nξ_t, dNdx_t, dNdξ_t, d2Ndx2_t, d2Ndξ2_t, dNds_t, d2Nds2_t} <: AbstractValues
    ip::IP          # ::Interpolation
    # FunctionValues are only functional for the types in the comments for the fields below.
    # However, e.g. for GPU support, we allow arrays of one order higher to be passed, allowing this type to be used as a struct-of-arrays (SoA) type.
    # See `soa_utils.jl` for the SoA transformation infrastructure.
    Nx::Nx_t         # ::AbstractMatrix{Union{<:Tensor,<:Number}}
    Nξ::Nξ_t         # ::AbstractMatrix{Union{<:Tensor,<:Number}}
    dNdx::dNdx_t    # ::AbstractMatrix{Union{<:Tensor,<:StaticArray}} or Nothing
    dNdξ::dNdξ_t    # ::AbstractMatrix{Union{<:Tensor,<:StaticArray}} or Nothing
    d2Ndx2::d2Ndx2_t   # ::AbstractMatrix{<:Tensor{2}}  Hessians of geometric shape functions in ref-domain
    d2Ndξ2::d2Ndξ2_t   # ::AbstractMatrix{<:Tensor{2}}  Hessians of geometric shape functions in ref-domain
    dNds::dNds_t       # ::AbstractMatrix{<:Tensor} or Nothing   Gradients wrt. the local frame coordinates
    d2Nds2::d2Nds2_t   # ::AbstractMatrix{<:Tensor} or Nothing   Hessians wrt. the local frame coordinates

    function FunctionValues(
            ip::Interpolation,
            Nx::Nx_t,
            Nξ::Nξ_t,
            dNdx::dNdx_t = nothing,
            dNdξ::dNdξ_t = nothing,
            d2Ndx2::d2Ndx2_t = nothing,
            d2Ndξ2::d2Ndξ2_t = nothing,
            dNds::dNds_t = nothing,
            d2Nds2::d2Nds2_t = nothing,
        ) where {Nx_t <: AbstractArray, Nξ_t <: AbstractArray, dNdx_t, dNdξ_t, d2Ndx2_t, d2Ndξ2_t, dNds_t, d2Nds2_t}

        difforder_x = !isnothing(d2Ndx2) ? 2 : (!isnothing(dNdx) ? 1 : 0)
        difforder_s = !isnothing(d2Nds2) ? 2 : (!isnothing(dNds) ? 1 : 0)

        return new{difforder_x, difforder_s, typeof(ip), Nx_t, Nξ_t, dNdx_t, dNdξ_t, d2Ndx2_t, d2Ndξ2_t, dNds_t, d2Nds2_t}(
            ip, Nx, Nξ, dNdx, dNdξ, d2Ndx2, d2Ndξ2, dNds, d2Nds2
        )
    end
end
# For backwards compatibility:
function FunctionValues{DiffOrder_x}(::Type{T}, ip::Interpolation, qr::QuadratureRule, ip_geo::VectorizedInterpolation) where {DiffOrder_x, T}
    return FunctionValues{DiffOrder_x, 0}(T, ip, qr, ip_geo)
end
function FunctionValues{DiffOrder_x, DiffOrder_s}(::Type{T}, ip::Interpolation, qr::QuadratureRule, ip_geo::VectorizedInterpolation) where {DiffOrder_x, DiffOrder_s, T}
    assert_same_refshapes(qr, ip, ip_geo)
    n_shape = getnbasefunctions(ip)
    n_qpoints = getnquadpoints(qr)

    Nξ = zeros(typeof_N(T, ip, ip_geo), n_shape, n_qpoints)
    Nx = isa(mapping_type(ip), IdentityMapping) ? Nξ : similar(Nξ)
    dNdξ = dNdx = d2Ndξ2 = d2Ndx2 = dNds = d2Nds2 = nothing

    if DiffOrder_s > 0 && !isa(mapping_type(ip), IdentityMapping)
        throw(ArgumentError("Local frame derivatives are only supported for interpolations with identity mapping"))
    end
    if max(DiffOrder_x, DiffOrder_s) > 2
        throw(ArgumentError("Currently only values, gradients, and hessians can be updated in FunctionValues"))
    end

    if DiffOrder_x >= 1 || DiffOrder_s >= 1
        dNdξ = zeros(typeof_dNdξ(T, ip, ip_geo), n_shape, n_qpoints)
    end
    if DiffOrder_x >= 1
        dNdx = fill(zero(typeof_dNdx(T, ip, ip_geo)) * T(NaN), n_shape, n_qpoints)
    end
    if DiffOrder_s >= 1
        dNds = fill(zero(typeof_dNdξ(T, ip, ip_geo)) * T(NaN), n_shape, n_qpoints)
    end

    if DiffOrder_x >= 2 || DiffOrder_s >= 2
        d2Ndξ2 = zeros(typeof_d2Ndξ2(T, ip, ip_geo), n_shape, n_qpoints)
    end
    if DiffOrder_x >= 2
        d2Ndx2 = fill(zero(typeof_d2Ndx2(T, ip, ip_geo)) * T(NaN), n_shape, n_qpoints)
    end
    if DiffOrder_s >= 2
        d2Nds2 = fill(zero(typeof_d2Ndξ2(T, ip, ip_geo)) * T(NaN), n_shape, n_qpoints)
    end

    fv = FunctionValues(ip, Nx, Nξ, dNdx, dNdξ, d2Ndx2, d2Ndξ2, dNds, d2Nds2)
    precompute_values!(fv, getpoints(qr)) # Separate function for qr point update in PointValues
    return fv
end

function precompute_values!(fv::FunctionValues{DiffOrder_x, DiffOrder_s}, qr_points::AbstractVector{<:Vec}) where {DiffOrder_x, DiffOrder_s}
    max_order = max(DiffOrder_x, DiffOrder_s)
    max_order == 0 && return reference_shape_values!(fv.Nξ, fv.ip, qr_points)
    max_order == 1 && return reference_shape_gradients_and_values!(fv.dNdξ, fv.Nξ, fv.ip, qr_points)
    max_order == 2 && return reference_shape_hessians_gradients_and_values!(fv.d2Ndξ2, fv.dNdξ, fv.Nξ, fv.ip, qr_points)
    error("Unsupported derivative order: $max_order")
end

function task_local_copy(v::FunctionValues)
    Nξ = task_local_copy(v.Nξ)
    Nx = v.Nξ === v.Nx ? Nξ : task_local_copy(v.Nx) # Preserve aliasing
    return FunctionValues(
        task_local_copy(v.ip), Nx, Nξ, task_local_copy(v.dNdx), task_local_copy(v.dNdξ),
        task_local_copy(v.d2Ndx2), task_local_copy(v.d2Ndξ2), task_local_copy(v.dNds), task_local_copy(v.d2Nds2)
    )
end

getnbasefunctions(funvals::FunctionValues) = size(funvals.Nx, 1)
getnquadpoints(funvals::FunctionValues) = size(funvals.Nx, 2)
@propagate_inbounds shape_value(funvals::FunctionValues, q_point::Int, base_func::Int) = funvals.Nx[base_func, q_point]
@propagate_inbounds shape_gradient(funvals::FunctionValues, q_point::Int, base_func::Int) = funvals.dNdx[base_func, q_point]
@propagate_inbounds shape_hessian(funvals::FunctionValues{2}, q_point::Int, base_func::Int) = funvals.d2Ndx2[base_func, q_point]
@propagate_inbounds shape_local_gradient(funvals::FunctionValues, q_point::Int, base_func::Int) = funvals.dNds[base_func, q_point]
@propagate_inbounds shape_local_hessian(funvals::FunctionValues{<:Any, 2}, q_point::Int, base_func::Int) = funvals.d2Nds2[base_func, q_point]

function_interpolation(funvals::FunctionValues) = funvals.ip
function_difforder(::FunctionValues{DiffOrder}) where {DiffOrder} = DiffOrder
function_local_difforder(::FunctionValues{<:Any, DiffOrder_s}) where {DiffOrder_s} = DiffOrder_s
shape_value_type(funvals::FunctionValues) = eltype(funvals.Nx)
shape_gradient_type(funvals::FunctionValues) = eltype(funvals.dNdx)
shape_gradient_type(::FunctionValues{0}) = nothing
shape_hessian_type(funvals::FunctionValues) = eltype(funvals.d2Ndx2)
shape_hessian_type(::FunctionValues{0}) = nothing
shape_hessian_type(::FunctionValues{1}) = nothing
shape_local_gradient_type(funvals::FunctionValues) = eltype(funvals.dNds)
shape_local_gradient_type(::FunctionValues{<:Any, 0}) = nothing
shape_local_hessian_type(funvals::FunctionValues) = eltype(funvals.d2Nds2)
shape_local_hessian_type(::FunctionValues{<:Any, 0}) = nothing
shape_local_hessian_type(::FunctionValues{<:Any, 1}) = nothing


# Checks that the user provides the right dimension of coordinates to reinit! methods to ensure good error messages if not
sdim_from_gradtype(::Type{<:TT}) where {TT <: AbstractTensor} = last(size(TT))

# For performance, these must be fully inferable for the compiler.
# args: valname (:CellValues or :FacetValues), shape_gradient_type, eltype(x)
function check_reinit_sdim_consistency(fe_v::FeV, ::AbstractVector{VT}) where {FeV <: AbstractValues, VT}
    return check_reinit_sdim_consistency(nameof(FeV), shape_gradient_type(fe_v), VT)
end
function check_reinit_sdim_consistency(valname, gradtype::Type, ::Type{<:Vec{sdim}}) where {sdim}
    check_reinit_sdim_consistency(valname, Val(sdim_from_gradtype(gradtype)), Val(sdim))
    return
end
check_reinit_sdim_consistency(_, ::Nothing, ::Type{<:Vec}) = nothing # gradient not stored, cannot check
check_reinit_sdim_consistency(_, ::Val{sdim}, ::Val{sdim}) where {sdim} = nothing
function check_reinit_sdim_consistency(valname, ::Val{sdim_val}, ::Val{sdim_x}) where {sdim_val, sdim_x}
    throw(ArgumentError("The $valname (sdim=$sdim_val) and coordinates (sdim=$sdim_x) have different spatial dimensions."))
end

# Mapping types
struct IdentityMapping end
struct CovariantPiolaMapping end
struct ContravariantPiolaMapping end

mapping_type(fv::FunctionValues) = mapping_type(fv.ip)

"""
    required_geo_diff_order(fun_mapping, fun_diff_order::Int)

Return the required order of geometric derivatives to map
the function values and gradients from the reference cell
to the physical cell geometry.
"""
required_geo_diff_order(::IdentityMapping, fun_diff_order::Int) = fun_diff_order
required_geo_diff_order(::ContravariantPiolaMapping, fun_diff_order::Int) = 1 + fun_diff_order
required_geo_diff_order(::CovariantPiolaMapping, fun_diff_order::Int) = 1 + fun_diff_order

# Support for embedded elements
@inline calculate_Jinv(J::Tensor{2}) = inv(J)
@inline function calculate_Jinv(
        J::Union{ #MixedTensor2{sdim, rdim}
            MixedTensor2{2, 1}, MixedTensor2{3, 1}, MixedTensor2{3, 2},
        }
    )
    # Optimized Moore-Penrose pseudo inverse.
    # We assume that `J'⋅J` is invertible. This is a reasonable for non-degenerate elements.
    return inv(tdot(J)) ⋅ (J)'
end

# =============
# Apply mapping
# =============
@inline function apply_mapping!(funvals::FunctionValues, q_point::Int, args...)
    _apply_mapping!(funvals, mapping_type(funvals), q_point, args...)
    _apply_mapping_local_frame!(funvals, mapping_type(funvals), q_point, args...)
    return nothing
end

# Identity mapping
@inline function _apply_mapping!(::FunctionValues{0}, ::IdentityMapping, ::Int, mapping_values, args...)
    return nothing
end

@inline function _apply_mapping!(funvals::FunctionValues{1}, ::IdentityMapping, q_point::Int, mapping_values, args...)
    Jinv = calculate_Jinv(getjacobian(mapping_values))
    @inbounds for j in 1:getnbasefunctions(funvals)
        funvals.dNdx[j, q_point] = funvals.dNdξ[j, q_point] ⋅ Jinv
    end
    return nothing
end

@inline function _apply_mapping!(funvals::FunctionValues{2}, ::IdentityMapping, q_point::Int, mapping_values, args...)
    Jinv = calculate_Jinv(getjacobian(mapping_values))

    sdim, rdim = size(Jinv)
    (rdim != sdim) && error("_apply_mapping! for second order gradients and embedded elements not implemented")

    H = gethessian(mapping_values)
    is_vector_valued = first(funvals.Nx) isa Vec
    Jinv_otimesu_Jinv = is_vector_valued ? otimesu(Jinv, Jinv) : nothing
    @inbounds for j in 1:getnbasefunctions(funvals)
        dNdx = funvals.dNdξ[j, q_point] ⋅ Jinv
        if is_vector_valued
            d2Ndx2 = (funvals.d2Ndξ2[j, q_point] - dNdx ⋅ H) ⊡ Jinv_otimesu_Jinv
        else
            d2Ndx2 = Jinv' ⋅ (funvals.d2Ndξ2[j, q_point] - dNdx ⋅ H) ⋅ Jinv
        end

        funvals.dNdx[j, q_point] = dNdx
        funvals.d2Ndx2[j, q_point] = d2Ndx2
    end
    return nothing
end

"""
    gram_schmidt_frame(J)

Return the orthonormal local frame, `E`, obtained by Gram-Schmidt orthonormalization of the columns
of the jacobian `J = ∂x/∂ξ` (size `sdim × rdim`). `E` has the same size as `J`, and its columns span
the same (tangent) space as the columns of `J`. The first column of `E` is aligned with the first column of `J`.
"""
@inline gram_schmidt_frame(J::Tensor{2, dim}) where {dim} = _gram_schmidt_frame(Tensor{2, dim}, J, Val(dim))
@inline gram_schmidt_frame(J::MixedTensor2{sdim, rdim}) where {sdim, rdim} = _gram_schmidt_frame(MixedTensor2{sdim, rdim}, J, Val(rdim))

@inline function _gram_schmidt_frame(::Type{TT}, J, ::Val{1}) where {TT}
    e1 = normalize(J[:, 1])
    return TT((e1...,))
end
@inline function _gram_schmidt_frame(::Type{TT}, J, ::Val{2}) where {TT}
    e1 = normalize(J[:, 1])
    x2 = J[:, 2]
    e2 = normalize(x2 - (e1 ⋅ x2) * e1)
    return TT((e1..., e2...))
end
@inline function _gram_schmidt_frame(::Type{TT}, J, ::Val{3}) where {TT}
    e1 = normalize(J[:, 1])
    x2 = J[:, 2]
    e2 = normalize(x2 - (e1 ⋅ x2) * e1)
    x3 = J[:, 3]
    e3 = normalize(x3 - (e1 ⋅ x3) * e1 - (e2 ⋅ x3) * e2)
    return TT((e1..., e2..., e3...))
end

@inline function _apply_mapping_local_frame!(::FunctionValues{<:Any, 0}, ::Any, ::Int, mapping_values, args...)
    return nothing
end

@inline function _apply_mapping_local_frame!(funvals::FunctionValues{<:Any, 1}, ::IdentityMapping, q_point::Int, mapping_values, args...)
    J = getjacobian(mapping_values)
    E = gram_schmidt_frame(J)
    Binv = inv(E' ⋅ J)
    @inbounds for j in 1:getnbasefunctions(funvals)
        funvals.dNds[j, q_point] = funvals.dNdξ[j, q_point] ⋅ Binv
    end
    return nothing
end

@inline function _apply_mapping_local_frame!(funvals::FunctionValues{<:Any, 2}, ::IdentityMapping, q_point::Int, mapping_values, args...)
    J = getjacobian(mapping_values)
    H = gethessian(mapping_values)
    E = gram_schmidt_frame(J)
    Binv = inv(E' ⋅ J)
    Hs = E' ⋅ H
    is_vector_valued = first(funvals.Nx) isa Vec
    Binv_otimesu_Binv = is_vector_valued ? otimesu(Binv, Binv) : nothing
    @inbounds for j in 1:getnbasefunctions(funvals)
        dNds = funvals.dNdξ[j, q_point] ⋅ Binv
        if is_vector_valued
            d2Nds2 = (funvals.d2Ndξ2[j, q_point] - dNds ⋅ Hs) ⊡ Binv_otimesu_Binv
        else
            d2Nds2 = Binv' ⋅ (funvals.d2Ndξ2[j, q_point] - dNds ⋅ Hs) ⋅ Binv
        end
        funvals.dNds[j, q_point] = dNds
        funvals.d2Nds2[j, q_point] = d2Nds2
    end
    return nothing
end

# Covariant Piola Mapping
@inline function _apply_mapping!(funvals::FunctionValues{0}, ::CovariantPiolaMapping, q_point::Int, mapping_values, cell)
    Jinv = inv(getjacobian(mapping_values))
    @inbounds for j in 1:getnbasefunctions(funvals)
        d = get_direction(funvals.ip, j, cell)
        Nξ = funvals.Nξ[j, q_point]
        funvals.Nx[j, q_point] = d * (Nξ ⋅ Jinv)
    end
    return nothing
end

@inline function _apply_mapping!(funvals::FunctionValues{1}, ::CovariantPiolaMapping, q_point::Int, mapping_values, cell)
    H = gethessian(mapping_values)
    Jinv = inv(getjacobian(mapping_values))
    @inbounds for j in 1:getnbasefunctions(funvals)
        d = get_direction(funvals.ip, j, cell)
        dNdξ = funvals.dNdξ[j, q_point]
        Nξ = funvals.Nξ[j, q_point]
        funvals.Nx[j, q_point] = d * (Nξ ⋅ Jinv)
        funvals.dNdx[j, q_point] = d * (Jinv' ⋅ dNdξ ⋅ Jinv - Jinv' ⋅ (Nξ ⋅ Jinv ⋅ H ⋅ Jinv))
    end
    return nothing
end

# Contravariant Piola Mapping
@inline function _apply_mapping!(funvals::FunctionValues{0}, ::ContravariantPiolaMapping, q_point::Int, mapping_values, cell)
    J = getjacobian(mapping_values)
    detJ = det(J)
    @inbounds for j in 1:getnbasefunctions(funvals)
        d = get_direction(funvals.ip, j, cell)
        Nξ = funvals.Nξ[j, q_point]
        funvals.Nx[j, q_point] = d * (J ⋅ Nξ) / detJ
    end
    return nothing
end

@inline function _apply_mapping!(funvals::FunctionValues{1}, ::ContravariantPiolaMapping, q_point::Int, mapping_values, cell)
    H = gethessian(mapping_values)
    J = getjacobian(mapping_values)
    Jinv = inv(J)
    detJ = det(J)
    I2 = one(J)
    H_Jinv = H ⋅ Jinv
    A1 = (H_Jinv ⊡ (otimesl(I2, I2))) / detJ
    A2 = (Jinv' ⊡ H_Jinv) / detJ
    @inbounds for j in 1:getnbasefunctions(funvals)
        d = get_direction(funvals.ip, j, cell)
        dNdξ = funvals.dNdξ[j, q_point]
        Nξ = funvals.Nξ[j, q_point]
        funvals.Nx[j, q_point] = d * (J ⋅ Nξ) / detJ
        funvals.dNdx[j, q_point] = d * (J ⋅ dNdξ ⋅ Jinv / detJ + A1 ⋅ Nξ - (J ⋅ Nξ) ⊗ A2)
    end
    return nothing
end
