#################################################################
# Note on dimensions:                                           #
# sdim = spatial dimension (dimension of the grid nodes)        #
# rdim = reference dimension (dimension in isoparametric space) #
# vdim = vector dimension (dimension of the field)              #
#################################################################

# Scalar, sdim == rdim                                                 sdim                     rdim
typeof_N(     ::Type{T}, ::ScalarInterpolation, ::VectorizedInterpolation{dim, <: AbstractRefShape{dim}}) where {T, dim} = T
typeof_dNdx(  ::Type{T}, ::ScalarInterpolation, ::VectorizedInterpolation{dim, <: AbstractRefShape{dim}}) where {T, dim} = Vec{dim, T}
typeof_dNdξ(  ::Type{T}, ::ScalarInterpolation, ::VectorizedInterpolation{dim, <: AbstractRefShape{dim}}) where {T, dim} = Vec{dim, T}
typeof_d2Ndx2(::Type{T}, ::ScalarInterpolation, ::VectorizedInterpolation{dim, <: AbstractRefShape{dim}}) where {T, dim} = Tensor{2, dim, T}
typeof_d2Ndξ2(::Type{T}, ::ScalarInterpolation, ::VectorizedInterpolation{dim, <: AbstractRefShape{dim}}) where {T, dim} = Tensor{2, dim, T}

# Vector, vdim == sdim == rdim              vdim                            sdim                     rdim
typeof_N(     ::Type{T}, ::VectorInterpolation{dim}, ::VectorizedInterpolation{dim, <: AbstractRefShape{dim}}) where {T, dim} = Vec{dim, T}
typeof_dNdx(  ::Type{T}, ::VectorInterpolation{dim}, ::VectorizedInterpolation{dim, <: AbstractRefShape{dim}}) where {T, dim} = Tensor{2, dim, T}
typeof_dNdξ(  ::Type{T}, ::VectorInterpolation{dim}, ::VectorizedInterpolation{dim, <: AbstractRefShape{dim}}) where {T, dim} = Tensor{2, dim, T}
typeof_d2Ndx2(::Type{T}, ::VectorInterpolation{dim}, ::VectorizedInterpolation{dim, <: AbstractRefShape{dim}}) where {T, dim} = Tensor{3, dim, T}
typeof_d2Ndξ2(::Type{T}, ::VectorInterpolation{dim}, ::VectorizedInterpolation{dim, <: AbstractRefShape{dim}}) where {T, dim} = Tensor{3, dim, T}

# Scalar, sdim != rdim (TODO: Use Vec if (s|r)dim <= 3?)
typeof_N(   ::Type{T}, ::ScalarInterpolation, ::VectorizedInterpolation{sdim, <: AbstractRefShape{rdim}}) where {T, sdim, rdim} = T
typeof_dNdx(::Type{T}, ::ScalarInterpolation, ::VectorizedInterpolation{sdim, <: AbstractRefShape{rdim}}) where {T, sdim, rdim} = SVector{sdim, T}
typeof_dNdξ(::Type{T}, ::ScalarInterpolation, ::VectorizedInterpolation{sdim, <: AbstractRefShape{rdim}}) where {T, sdim, rdim} = SVector{rdim, T}
typeof_d2Ndx2(::Type{T}, ::ScalarInterpolation, ::VectorizedInterpolation{sdim, <: AbstractRefShape{rdim}}) where {T, sdim, rdim} = SMatrix{sdim, sdim, T}
typeof_d2Ndξ2(::Type{T}, ::ScalarInterpolation, ::VectorizedInterpolation{sdim, <: AbstractRefShape{rdim}}) where {T, sdim, rdim} = SMatrix{rdim, rdim, T}


# Vector, vdim != sdim != rdim (TODO: Use Vec/Tensor if (s|r)dim <= 3?)
typeof_N(   ::Type{T}, ::VectorInterpolation{vdim}, ::VectorizedInterpolation{sdim, <: AbstractRefShape{rdim}}) where {T, vdim, sdim, rdim} = SVector{vdim, T}
typeof_dNdx(::Type{T}, ::VectorInterpolation{vdim}, ::VectorizedInterpolation{sdim, <: AbstractRefShape{rdim}}) where {T, vdim, sdim, rdim} = SMatrix{vdim, sdim, T}
typeof_dNdξ(::Type{T}, ::VectorInterpolation{vdim}, ::VectorizedInterpolation{sdim, <: AbstractRefShape{rdim}}) where {T, vdim, sdim, rdim} = SMatrix{vdim, rdim, T}
typeof_d2Ndx2(::Type{T}, ::VectorInterpolation{vdim}, ::VectorizedInterpolation{sdim, <: AbstractRefShape{rdim}}) where {T, vdim, sdim, rdim} = SArray{Tuple{vdim, sdim, sdim}, T}
typeof_d2Ndξ2(::Type{T}, ::VectorInterpolation{vdim}, ::VectorizedInterpolation{sdim, <: AbstractRefShape{rdim}}) where {T, vdim, sdim, rdim} = SArray{Tuple{vdim, rdim, rdim}, T}


"""
    FunctionValues{DiffOrder}(::Type{T}, ip_fun, qr::QuadratureRule, ip_geo::VectorizedInterpolation)

Create a `FunctionValues` object containing the shape values and gradients (up to order `DiffOrder`) 
for both the reference cell (precalculated) and the real cell (updated in `reinit!`). 
"""
FunctionValues

struct FunctionValues{DiffOrder, IP, N_t, dNdx_t, dNdξ_t, d2Ndx2_t, d2Ndξ2_t}
    ip::IP          # ::Interpolation
    Nx::N_t         # ::AbstractMatrix{Union{<:Tensor,<:Number}}
    Nξ::N_t         # ::AbstractMatrix{Union{<:Tensor,<:Number}}
    dNdx::dNdx_t    # ::AbstractMatrix{Union{<:Tensor,<:StaticArray}} or Nothing
    dNdξ::dNdξ_t    # ::AbstractMatrix{Union{<:Tensor,<:StaticArray}} or Nothing
    d2Ndx2::d2Ndx2_t   # ::AbstractMatrix{<:Tensor{2}}  Hessians of geometric shape functions in ref-domain
    d2Ndξ2::d2Ndξ2_t   # ::AbstractMatrix{<:Tensor{2}}  Hessians of geometric shape functions in ref-domain
    function FunctionValues(ip::Interpolation, Nx::N_t, Nξ::N_t, ::Nothing, ::Nothing, ::Nothing, ::Nothing) where {N_t<:AbstractMatrix}
        return new{0, typeof(ip), N_t, Nothing, Nothing, Nothing, Nothing}(ip, Nx, Nξ, nothing, nothing, nothing, nothing)
    end
    function FunctionValues(ip::Interpolation, Nx::N_t, Nξ::N_t, dNdx::AbstractMatrix, dNdξ::AbstractMatrix, ::Nothing, ::Nothing) where {N_t<:AbstractMatrix}
        return new{1, typeof(ip), N_t, typeof(dNdx), typeof(dNdξ), Nothing, Nothing}(ip, Nx, Nξ, dNdx, dNdξ, nothing, nothing)
    end
    function FunctionValues(ip::Interpolation, Nx::N_t, Nξ::N_t, dNdx::AbstractMatrix, dNdξ::AbstractMatrix, d2Ndx2::AbstractMatrix, d2Ndξ2::AbstractMatrix) where {N_t<:AbstractMatrix}
        return new{2, typeof(ip), N_t, typeof(dNdx), typeof(dNdξ), typeof(d2Ndx2), typeof(d2Ndξ2)}(ip, Nx, Nξ, dNdx, dNdξ, d2Ndx2, d2Ndξ2)
    end
end
function FunctionValues{DiffOrder}(::Type{T}, ip::Interpolation, qr::QuadratureRule, ip_geo::VectorizedInterpolation) where {DiffOrder, T}
    sdim = n_components(ip_geo)
    rdim = getdim(ip_geo)
    (DiffOrder > 1 && (rdim != sdim)) && error("Higher order gradient for embedded elements not implemented")

    n_shape = getnbasefunctions(ip)
    n_qpoints = getnquadpoints(qr)
    
    Nξ = zeros(typeof_N(T, ip, ip_geo), n_shape, n_qpoints)
    Nx = isa(mapping_type(ip), IdentityMapping) ? Nξ : similar(Nξ)
    dNdξ = dNdx = d2Ndξ2 = d2Ndx2 = nothing
    
    if DiffOrder >= 1
        dNdξ = zeros(typeof_dNdξ(T, ip, ip_geo),               n_shape, n_qpoints)
        dNdx = fill(zero(typeof_dNdx(T, ip, ip_geo)) * T(NaN), n_shape, n_qpoints)
    end

    if DiffOrder >= 2
        d2Ndξ2 = zeros(typeof_d2Ndξ2(T, ip, ip_geo),               n_shape, n_qpoints)
        d2Ndx2 = fill(zero(typeof_d2Ndx2(T, ip, ip_geo)) * T(NaN), n_shape, n_qpoints)
    end

    if DiffOrder > 2
        throw(ArgumentError("Currently only values and gradients can be updated in FunctionValues"))
    end

    fv = FunctionValues(ip, Nx, Nξ, dNdx, dNdξ, d2Ndx2, d2Ndξ2)
    precompute_values!(fv, getpoints(qr)) # Separate function for qr point update in PointValues
    return fv
end

function precompute_values!(fv::FunctionValues{0}, qr_points::Vector{<:Vec})
    shape_values!(fv.Nξ, fv.ip, qr_points)
end
function precompute_values!(fv::FunctionValues{1}, qr_points::Vector{<:Vec})
    shape_gradients_and_values!(fv.dNdξ, fv.Nξ, fv.ip, qr_points)
end
function precompute_values!(fv::FunctionValues{2}, qr_points::Vector{<:Vec})
    shape_hessians_gradients_and_values!(fv.d2Ndξ2, fv.dNdξ, fv.Nξ, fv.ip, qr_points)
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
@propagate_inbounds shape_value(funvals::FunctionValues, q_point::Int, base_func::Int) = funvals.Nx[base_func, q_point]
@propagate_inbounds shape_gradient(funvals::FunctionValues, q_point::Int, base_func::Int) = funvals.dNdx[base_func, q_point]
@propagate_inbounds shape_hessian(funvals::FunctionValues{2}, q_point::Int, base_func::Int) = funvals.d2Ndx2[base_func, q_point]
@propagate_inbounds shape_symmetric_gradient(funvals::FunctionValues, q_point::Int, base_func::Int) = symmetric(shape_gradient(funvals, q_point, base_func))

function_interpolation(funvals::FunctionValues) = funvals.ip
function_difforder(::FunctionValues{DiffOrder}) where DiffOrder = DiffOrder
shape_value_type(funvals::FunctionValues) = eltype(funvals.Nx)
shape_gradient_type(funvals::FunctionValues) = eltype(funvals.dNdx)
shape_gradient_type(::FunctionValues{0}) = nothing
shape_hessian_type(funvals::FunctionValues) = eltype(funvals.d2Ndx2)
shape_hessian_type(::FunctionValues{0}) = nothing
shape_hessian_type(::FunctionValues{1}) = nothing


# Checks that the user provides the right dimension of coordinates to reinit! methods to ensure good error messages if not
sdim_from_gradtype(::Type{<:AbstractTensor{<:Any,sdim}}) where sdim = sdim
sdim_from_gradtype(::Type{<:SVector{sdim}}) where sdim = sdim
sdim_from_gradtype(::Type{<:SMatrix{<:Any,sdim}}) where sdim = sdim

# For performance, these must be fully inferable for the compiler.
# args: valname (:CellValues or :FaceValues), shape_gradient_type, eltype(x)
function check_reinit_sdim_consistency(valname, gradtype::Type, ::Type{<:Vec{sdim}}) where {sdim}
    check_reinit_sdim_consistency(valname, Val(sdim_from_gradtype(gradtype)), Val(sdim))
end
check_reinit_sdim_consistency(_, ::Nothing, ::Type{<:Vec}) = nothing # gradient not stored, cannot check
check_reinit_sdim_consistency(_, ::Val{sdim}, ::Val{sdim}) where sdim = nothing
function check_reinit_sdim_consistency(valname, ::Val{sdim_val}, ::Val{sdim_x}) where {sdim_val, sdim_x}
    throw(ArgumentError("The $valname (sdim=$sdim_val) and coordinates (sdim=$sdim_x) have different spatial dimensions."))
end

# Mapping types 
struct IdentityMapping end 
# Not yet implemented:
# struct CovariantPiolaMapping end # PR798
# struct ContravariantPiolaMapping end # PR798
# struct DoubleCovariantPiolaMapping end 
# struct DoubleContravariantPiolaMapping end 

mapping_type(fv::FunctionValues) = mapping_type(fv.ip)

"""
    required_geo_diff_order(fun_mapping, fun_diff_order::Int)

Return the required order of geometric derivatives to map 
the function values and gradients from the reference cell 
to the physical cell geometry.
"""
required_geo_diff_order(::IdentityMapping,           fun_diff_order::Int) = fun_diff_order
#required_geo_diff_order(::ContravariantPiolaMapping, fun_diff_order::Int) = 1 + fun_diff_order # PR798
#required_geo_diff_order(::CovariantPiolaMapping,     fun_diff_order::Int) = 1 + fun_diff_order # PR798


# Support for embedded elements
@inline calculate_Jinv(J::Tensor{2}) = inv(J)
@inline calculate_Jinv(J::SMatrix) = pinv(J)

# Hotfix to get the dots right for embedded elements until mixed tensors are merged.
# Scalar/Vector interpolations with sdim == rdim (== vdim)
@inline dothelper(A, B) = A ⋅ B
# Vector interpolations with sdim == rdim != vdim
@inline dothelper(A::SMatrix{vdim, dim}, B::Tensor{2, dim}) where {vdim, dim} = A * SMatrix{dim, dim}(B)
# Scalar interpolations with sdim > rdim
@inline dothelper(A::SVector{rdim}, B::SMatrix{rdim, sdim}) where {rdim, sdim} = B' * A
# Vector interpolations with sdim > rdim
@inline dothelper(B::SMatrix{vdim, rdim}, A::SMatrix{rdim, sdim}) where {vdim, rdim, sdim} = B * A

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
        #funvals.dNdx[j, q_point] = funvals.dNdξ[j, q_point] ⋅ Jinv # TODO via Tensors.jl
        funvals.dNdx[j, q_point] = dothelper(funvals.dNdξ[j, q_point], Jinv)
    end
    return nothing
end

@inline function apply_mapping!(funvals::FunctionValues{2}, ::IdentityMapping, q_point::Int, mapping_values, args...)
    Jinv = calculate_Jinv(getjacobian(mapping_values))
    H    = gethessian(mapping_values)
    @inbounds for j in 1:getnbasefunctions(funvals)
        dNdx = dothelper(funvals.dNdξ[j, q_point], Jinv)
        
        is_vector_valued = first(funvals.Nx) isa Vec
        if is_vector_valued #TODO - combine with helper function ? 
            d2Ndx2 = (funvals.d2Ndξ2[j, q_point] - dNdx⋅H) ⊡ otimesu(Jinv,Jinv)
        else
            d2Ndx2 = Jinv'⋅(funvals.d2Ndξ2[j, q_point] - dNdx⋅H)⋅Jinv
        end

        funvals.dNdx[j, q_point]   = dNdx
        funvals.d2Ndx2[j, q_point] = d2Ndx2
    end
    return nothing
end

# TODO in PR798, apply_mapping! for 
# * CovariantPiolaMapping
# * ContravariantPiolaMapping
