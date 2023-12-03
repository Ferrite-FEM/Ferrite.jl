#################################################################
# Note on dimensions:                                           #
# sdim = spatial dimension (dimension of the grid nodes)        #
# rdim = reference dimension (dimension in isoparametric space) #
# vdim = vector dimension (dimension of the field)              #
#################################################################

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

"""
    FunctionValues{DiffOrder}(::Type{T}, ip_fun, qr::QuadratureRule, ip_geo::VectorizedInterpolation)

Create a `FunctionValues` object containing the shape values and gradients (up to order `DiffOrder`) 
for both the reference cell (precalculated) and the real cell (updated in `reinit!`). 
"""
FunctionValues

struct FunctionValues{DiffOrder, IP, N_t, dNdx_t, dNdξ_t}
    ip::IP          # ::Interpolation
    N_x::N_t        # ::AbstractMatrix{Union{<:Tensor,<:Number}}
    N_ξ::N_t        # ::AbstractMatrix{Union{<:Tensor,<:Number}}
    dNdx::dNdx_t    # ::AbstractMatrix{Union{<:Tensor,<:StaticArray}} or Nothing
    dNdξ::dNdξ_t    # ::AbstractMatrix{Union{<:Tensor,<:StaticArray}} or Nothing
    function FunctionValues(ip::Interpolation, N_x::N_t, N_ξ::N_t, ::Nothing, ::Nothing) where {N_t<:AbstractMatrix}
        return new{0, typeof(ip), N_t, Nothing, Nothing}(ip, N_x, N_ξ, nothing, nothing)
    end
    function FunctionValues(ip::Interpolation, N_x::N_t, N_ξ::N_t, dNdx::AbstractMatrix, dNdξ::AbstractMatrix) where {N_t<:AbstractMatrix}
        return new{1, typeof(ip), N_t, typeof(dNdx), typeof(dNdξ)}(ip, N_x, N_ξ, dNdx, dNdξ)
    end
end
function FunctionValues{0}(::Type{T}, ip::Interpolation, qr::QuadratureRule, ip_geo::VectorizedInterpolation) where T
    ip_dims = InterpolationDims(ip, ip_geo)
    n_shape = getnbasefunctions(ip)
    n_qpoints = getnquadpoints(qr)
    
    N_ξ = zeros(typeof_N(T, ip_dims), n_shape, n_qpoints)
    N_x = isa(get_mapping_type(ip), IdentityMapping) ? N_ξ : similar(N_ξ)
        
    fv = FunctionValues(ip, N_x, N_ξ, nothing, nothing)
    precompute_values!(fv, getpoints(qr)) # Separate function for qr point update in PointValues
    return fv
end
function FunctionValues{1}(::Type{T}, ip::Interpolation, qr::QuadratureRule, ip_geo::VectorizedInterpolation) where T
    ip_dims = InterpolationDims(ip, ip_geo)
    n_shape = getnbasefunctions(ip)
    n_qpoints = getnquadpoints(qr)
    
    N_ξ = zeros(typeof_N(T, ip_dims), n_shape, n_qpoints)
    N_x = isa(get_mapping_type(ip), IdentityMapping) ? N_ξ : similar(N_ξ)
    
    dNdξ = zeros(typeof_dNdξ(T, ip_dims),               n_shape, n_qpoints)
    dNdx = fill(zero(typeof_dNdx(T, ip_dims)) * T(NaN), n_shape, n_qpoints)
    
    fv = FunctionValues(ip, N_x, N_ξ, dNdx, dNdξ)
    precompute_values!(fv, getpoints(qr)) # Separate function for qr point update in PointValues
    return fv
end

function precompute_values!(fv::FunctionValues{0}, qr_points::Vector{<:Vec})
    shape_values!(fv.N_ξ, fv.ip, qr_points)
end
function precompute_values!(fv::FunctionValues{1}, qr_points::Vector{<:Vec})
    shape_gradients_and_values!(fv.dNdξ, fv.N_ξ, fv.ip, qr_points)
end

function Base.copy(v::FunctionValues)
    N_ξ_copy = copy(v.N_ξ)
    N_x_copy = v.N_ξ === v.N_x ? N_ξ_copy : copy(v.N_x) # Preserve aliasing
    dNdx_copy = _copy_or_nothing(v.dNdx)
    dNdξ_copy = _copy_or_nothing(v.dNdξ)
    return FunctionValues(copy(v.ip), N_x_copy, N_ξ_copy, dNdx_copy, dNdξ_copy)
end

getnbasefunctions(funvals::FunctionValues) = size(funvals.N_x, 1)
@propagate_inbounds shape_value(funvals::FunctionValues, q_point::Int, base_func::Int) = funvals.N_x[base_func, q_point]
@propagate_inbounds shape_gradient(funvals::FunctionValues, q_point::Int, base_func::Int) = funvals.dNdx[base_func, q_point]
@propagate_inbounds shape_symmetric_gradient(funvals::FunctionValues, q_point::Int, base_func::Int) = symmetric(shape_gradient(funvals, q_point, base_func))

get_function_interpolation(funvals::FunctionValues) = funvals.ip
get_function_difforder(::FunctionValues{DiffOrder}) where DiffOrder = DiffOrder
shape_value_type(funvals::FunctionValues) = eltype(funvals.N_x)
shape_gradient_type(funvals::FunctionValues) = eltype(funvals.dNdx)
shape_gradient_type(::FunctionValues{0}) = nothing


# Checks that the user provides the right dimension of coordinates to reinit! methods to ensure good error messages if not
sdim_from_gradtype(::Type{<:AbstractTensor{<:Any,sdim}}) where sdim = sdim
sdim_from_gradtype(::Type{<:SVector{sdim}}) where sdim = sdim
sdim_from_gradtype(::Type{<:SMatrix{<:Any,sdim}}) where sdim = sdim

# For performance, these must be fully inferrable for the compiler.
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

get_mapping_type(fv::FunctionValues) = get_mapping_type(fv.ip)

"""
    increased_diff_order(mapping)

How many order higher geometric derivatives are required to  
to map the function values and gradients from the reference cell 
to the physical cell geometry?
"""
increased_diff_order(::IdentityMapping) = 0
# increased_diff_order(::ContravariantPiolaMapping) = 1  # PR798
# increased_diff_order(::CovariantPiolaMapping) = 1  # PR798

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
    return apply_mapping!(funvals, get_mapping_type(funvals), q_point, args...)
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

# TODO in PR798, apply_mapping! for 
# * CovariantPiolaMapping
# * ContravariantPiolaMapping
