#################################################################
# Note on dimensions:                                           #
# sdim = spatial dimension (dimension of the grid nodes)        #
# rdim = reference dimension (dimension in isoparametric space) #
# vdim = vector dimension (dimension of the field)              #
#################################################################

# Scalar, sdim != rdim (TODO: Use Vec if (s|r)dim <= 3?)
typeof_N(::Type{T}, ::ScalarInterpolation, ::VectorizedInterpolation{sdim, <:AbstractRefShape{rdim}}) where {T, sdim, rdim} = T
typeof_dNdx(::Type{T}, ::ScalarInterpolation, ::VectorizedInterpolation{sdim, <:AbstractRefShape{rdim}}) where {T, sdim, rdim} = Vec{sdim, T}
typeof_dNdξ(::Type{T}, ::ScalarInterpolation, ::VectorizedInterpolation{sdim, <:AbstractRefShape{rdim}}) where {T, sdim, rdim} = Vec{rdim, T}
typeof_d2Ndx2(::Type{T}, ::ScalarInterpolation, ::VectorizedInterpolation{sdim, <:AbstractRefShape{rdim}}) where {T, sdim, rdim} = Tensor{2, sdim, T}
typeof_d2Ndξ2(::Type{T}, ::ScalarInterpolation, ::VectorizedInterpolation{sdim, <:AbstractRefShape{rdim}}) where {T, sdim, rdim} = Tensor{2, rdim, T}

# Vector, vdim != sdim != rdim (TODO: Use Vec/Tensor if (s|r)dim <= 3?)
typeof_N(::Type{T}, ::VectorInterpolation{vdim}, ::VectorizedInterpolation{sdim, <:AbstractRefShape{rdim}}) where {T, vdim, sdim, rdim} = Vec{vdim, T}
typeof_dNdx(::Type{T}, ::VectorInterpolation{vdim}, ::VectorizedInterpolation{sdim, <:AbstractRefShape{rdim}}) where {T, vdim, sdim, rdim} = Tensors.regular_if_possible(MixedTensor2{vdim, sdim, T})
typeof_dNdξ(::Type{T}, ::VectorInterpolation{vdim}, ::VectorizedInterpolation{sdim, <:AbstractRefShape{rdim}}) where {T, vdim, sdim, rdim} = Tensors.regular_if_possible(MixedTensor2{vdim, rdim, T})
typeof_d2Ndx2(::Type{T}, ::VectorInterpolation{vdim}, ::VectorizedInterpolation{sdim, <:AbstractRefShape{rdim}}) where {T, vdim, sdim, rdim} = Tensors.regular_if_possible(MixedTensor3{vdim, sdim, sdim, T})
typeof_d2Ndξ2(::Type{T}, ::VectorInterpolation{vdim}, ::VectorizedInterpolation{sdim, <:AbstractRefShape{rdim}}) where {T, vdim, sdim, rdim} = Tensors.regular_if_possible(MixedTensor3{vdim, rdim, rdim, T})

"""
    FunctionValues{DiffOrder}(::Type{T}, ip_fun, qr::QuadratureRule, ip_geo::VectorizedInterpolation)

Create a `FunctionValues <: AbstractValues` object containing the shape values and gradients (up to order
`DiffOrder`) for both the reference cell (precalculated) and the real cell (updated in `reinit!`).
The user should normally not create `FunctionValues`, these are typically only created from the constructors
of `AbstractCellValues` and `AbstractFacetValues`. However, the user will interact with `fv::FunctionValues`
when indexing e.g. `cmv::MultiFieldCellValues` (e.g. `fv = cmv.u`), as `fv` supports

* [`getnbasefunctions`](@ref)
* [`shape_value`](@ref)
* [`shape_gradient`](@ref)
* [`shape_symmetric_gradient`](@ref)
* [`shape_divergence`](@ref)
* [`function_value`](@ref)
* [`function_gradient`](@ref)
* [`function_symmetric_gradient`](@ref)
* [`function_divergence`](@ref)
"""
FunctionValues

struct FunctionValues{DiffOrder, IP, N_t, dNdx_t, dNdξ_t, d2Ndx2_t, d2Ndξ2_t} <: AbstractValues
    ip::IP          # ::Interpolation
    Nx::N_t         # ::AbstractMatrix{Union{<:Tensor,<:Number}}
    Nξ::N_t         # ::AbstractMatrix{Union{<:Tensor,<:Number}}
    dNdx::dNdx_t    # ::AbstractMatrix{Union{<:Tensor,<:StaticArray}} or Nothing
    dNdξ::dNdξ_t    # ::AbstractMatrix{Union{<:Tensor,<:StaticArray}} or Nothing
    d2Ndx2::d2Ndx2_t   # ::AbstractMatrix{<:Tensor{2}}  Hessians of geometric shape functions in ref-domain
    d2Ndξ2::d2Ndξ2_t   # ::AbstractMatrix{<:Tensor{2}}  Hessians of geometric shape functions in ref-domain
    function FunctionValues(ip::Interpolation, Nx::N_t, Nξ::N_t, ::Nothing, ::Nothing, ::Nothing, ::Nothing) where {N_t <: AbstractMatrix}
        return new{0, typeof(ip), N_t, Nothing, Nothing, Nothing, Nothing}(ip, Nx, Nξ, nothing, nothing, nothing, nothing)
    end
    function FunctionValues(ip::Interpolation, Nx::N_t, Nξ::N_t, dNdx::AbstractMatrix, dNdξ::AbstractMatrix, ::Nothing, ::Nothing) where {N_t <: AbstractMatrix}
        return new{1, typeof(ip), N_t, typeof(dNdx), typeof(dNdξ), Nothing, Nothing}(ip, Nx, Nξ, dNdx, dNdξ, nothing, nothing)
    end
    function FunctionValues(ip::Interpolation, Nx::N_t, Nξ::N_t, dNdx::AbstractMatrix, dNdξ::AbstractMatrix, d2Ndx2::AbstractMatrix, d2Ndξ2::AbstractMatrix) where {N_t <: AbstractMatrix}
        return new{2, typeof(ip), N_t, typeof(dNdx), typeof(dNdξ), typeof(d2Ndx2), typeof(d2Ndξ2)}(ip, Nx, Nξ, dNdx, dNdξ, d2Ndx2, d2Ndξ2)
    end
end
function FunctionValues{DiffOrder}(::Type{T}, ip::Interpolation, qr::QuadratureRule, ip_geo::VectorizedInterpolation) where {DiffOrder, T}
    assert_same_refshapes(qr, ip, ip_geo)
    n_shape = getnbasefunctions(ip)
    n_qpoints = getnquadpoints(qr)

    Nξ = zeros(typeof_N(T, ip, ip_geo), n_shape, n_qpoints)
    Nx = isa(mapping_type(ip), IdentityMapping) ? Nξ : similar(Nξ)
    dNdξ = dNdx = d2Ndξ2 = d2Ndx2 = nothing

    if DiffOrder >= 1
        dNdξ = zeros(typeof_dNdξ(T, ip, ip_geo), n_shape, n_qpoints)
        dNdx = fill(zero(typeof_dNdx(T, ip, ip_geo)) * T(NaN), n_shape, n_qpoints)
    end

    if DiffOrder >= 2
        d2Ndξ2 = zeros(typeof_d2Ndξ2(T, ip, ip_geo), n_shape, n_qpoints)
        d2Ndx2 = fill(zero(typeof_d2Ndx2(T, ip, ip_geo)) * T(NaN), n_shape, n_qpoints)
    end

    if DiffOrder > 2
        throw(ArgumentError("Currently only values, gradients, and hessians can be updated in FunctionValues"))
    end

    fv = FunctionValues(ip, Nx, Nξ, dNdx, dNdξ, d2Ndx2, d2Ndξ2)
    precompute_values!(fv, getpoints(qr)) # Separate function for qr point update in PointValues
    return fv
end

function precompute_values!(fv::FunctionValues{0}, qr_points::AbstractVector{<:Vec})
    return reference_shape_values!(fv.Nξ, fv.ip, qr_points)
end
function precompute_values!(fv::FunctionValues{1}, qr_points::AbstractVector{<:Vec})
    return reference_shape_gradients_and_values!(fv.dNdξ, fv.Nξ, fv.ip, qr_points)
end
function precompute_values!(fv::FunctionValues{2}, qr_points::AbstractVector{<:Vec})
    return reference_shape_hessians_gradients_and_values!(fv.d2Ndξ2, fv.dNdξ, fv.Nξ, fv.ip, qr_points)
end

function Base.copy(v::FunctionValues)
    Nξ_copy = copy(v.Nξ)
    Nx_copy = v.Nξ === v.Nx ? Nξ_copy : copy(v.Nx) # Preserve aliasing
    dNdx_copy = _copy_or_nothing(v.dNdx)
    dNdξ_copy = _copy_or_nothing(v.dNdξ)
    d2Ndx2_copy = _copy_or_nothing(v.d2Ndx2)
    d2Ndξ2_copy = _copy_or_nothing(v.d2Ndξ2)
    return FunctionValues(copy(v.ip), Nx_copy, Nξ_copy, dNdx_copy, dNdξ_copy, d2Ndx2_copy, d2Ndξ2_copy)
end

getnbasefunctions(funvals::FunctionValues) = size(funvals.Nx, 1)
getnquadpoints(funvals::FunctionValues) = size(funvals.Nx, 2)
@propagate_inbounds shape_value(funvals::FunctionValues, q_point::Int, base_func::Int) = funvals.Nx[base_func, q_point]
@propagate_inbounds shape_gradient(funvals::FunctionValues, q_point::Int, base_func::Int) = funvals.dNdx[base_func, q_point]
@propagate_inbounds shape_hessian(funvals::FunctionValues{2}, q_point::Int, base_func::Int) = funvals.d2Ndx2[base_func, q_point]
@propagate_inbounds shape_symmetric_gradient(funvals::FunctionValues, q_point::Int, base_func::Int) = symmetric(shape_gradient(funvals, q_point, base_func))

function_interpolation(funvals::FunctionValues) = funvals.ip
function_difforder(::FunctionValues{DiffOrder}) where {DiffOrder} = DiffOrder
shape_value_type(funvals::FunctionValues) = eltype(funvals.Nx)
shape_gradient_type(funvals::FunctionValues) = eltype(funvals.dNdx)
shape_gradient_type(::FunctionValues{0}) = nothing
shape_hessian_type(funvals::FunctionValues) = eltype(funvals.d2Ndx2)
shape_hessian_type(::FunctionValues{0}) = nothing
shape_hessian_type(::FunctionValues{1}) = nothing


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
# TODO: Have Tensors.jl support pinv?
@inline function calculate_Jinv(J::MixedTensor2{dim1, dim2}) where {dim1, dim2}
    Js = SMatrix{dim1, dim2}(J.data)
    return MixedTensor2{dim2, dim1}((pinv(Js)...,))
end

# =============
# Apply mapping
# =============
@inline function apply_mapping!(funvals::FunctionValues, q_point::Int, args...)
    return apply_mapping!(funvals, mapping_type(funvals), q_point, args...)
end

# Identity mapping
@inline function apply_mapping!(::FunctionValues{0}, ::IdentityMapping, ::Int, mapping_values, args...)
    return nothing
end

@inline function apply_mapping!(funvals::FunctionValues{1}, ::IdentityMapping, q_point::Int, mapping_values, args...)
    Jinv = calculate_Jinv(getjacobian(mapping_values))
    @inbounds for j in 1:getnbasefunctions(funvals)
        funvals.dNdx[j, q_point] = funvals.dNdξ[j, q_point] ⋅ Jinv
    end
    return nothing
end

@inline function apply_mapping!(funvals::FunctionValues{2}, ::IdentityMapping, q_point::Int, mapping_values, args...)
    Jinv = calculate_Jinv(getjacobian(mapping_values))

    sdim, rdim = size(Jinv)
    (rdim != sdim) && error("apply_mapping! for second order gradients and embedded elements not implemented")

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

# Covariant Piola Mapping
@inline function apply_mapping!(funvals::FunctionValues{0}, ::CovariantPiolaMapping, q_point::Int, mapping_values, cell)
    Jinv = inv(getjacobian(mapping_values))
    @inbounds for j in 1:getnbasefunctions(funvals)
        d = get_direction(funvals.ip, j, cell)
        Nξ = funvals.Nξ[j, q_point]
        funvals.Nx[j, q_point] = d * (Nξ ⋅ Jinv)
    end
    return nothing
end

@inline function apply_mapping!(funvals::FunctionValues{1}, ::CovariantPiolaMapping, q_point::Int, mapping_values, cell)
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
@inline function apply_mapping!(funvals::FunctionValues{0}, ::ContravariantPiolaMapping, q_point::Int, mapping_values, cell)
    J = getjacobian(mapping_values)
    detJ = det(J)
    @inbounds for j in 1:getnbasefunctions(funvals)
        d = get_direction(funvals.ip, j, cell)
        Nξ = funvals.Nξ[j, q_point]
        funvals.Nx[j, q_point] = d * (J ⋅ Nξ) / detJ
    end
    return nothing
end

@inline function apply_mapping!(funvals::FunctionValues{1}, ::ContravariantPiolaMapping, q_point::Int, mapping_values, cell)
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
