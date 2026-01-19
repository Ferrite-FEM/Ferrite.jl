#################################################################
# Note on dimensions:                                           #
# sdim = spatial dimension (dimension of the grid nodes)        #
# rdim = reference dimension (dimension in isoparametric space) #
# vdim = vector dimension (dimension of the field)              #
#################################################################

# Scalar, sdim == rdim                                              sdim                    rdim
typeof_N(::Type{T}, ::ScalarInterpolation, ::VectorizedInterpolation{dim, <:AbstractRefShape{dim}}) where {T, dim} = T
typeof_dNdx(::Type{T}, ::ScalarInterpolation, ::VectorizedInterpolation{dim, <:AbstractRefShape{dim}}) where {T, dim} = Vec{dim, T}
typeof_dNdξ(::Type{T}, ::ScalarInterpolation, ::VectorizedInterpolation{dim, <:AbstractRefShape{dim}}) where {T, dim} = Vec{dim, T}
typeof_d2Ndx2(::Type{T}, ::ScalarInterpolation, ::VectorizedInterpolation{dim, <:AbstractRefShape{dim}}) where {T, dim} = Tensor{2, dim, T}
typeof_d2Ndξ2(::Type{T}, ::ScalarInterpolation, ::VectorizedInterpolation{dim, <:AbstractRefShape{dim}}) where {T, dim} = Tensor{2, dim, T}

# Vector, vdim == sdim == rdim           vdim                            sdim                    rdim
typeof_N(::Type{T}, ::VectorInterpolation{dim}, ::VectorizedInterpolation{dim, <:AbstractRefShape{dim}}) where {T, dim} = Vec{dim, T}
typeof_dNdx(::Type{T}, ::VectorInterpolation{dim}, ::VectorizedInterpolation{dim, <:AbstractRefShape{dim}}) where {T, dim} = Tensor{2, dim, T}
typeof_dNdξ(::Type{T}, ::VectorInterpolation{dim}, ::VectorizedInterpolation{dim, <:AbstractRefShape{dim}}) where {T, dim} = Tensor{2, dim, T}
typeof_d2Ndx2(::Type{T}, ::VectorInterpolation{dim}, ::VectorizedInterpolation{dim, <:AbstractRefShape{dim}}) where {T, dim} = Tensor{3, dim, T}
typeof_d2Ndξ2(::Type{T}, ::VectorInterpolation{dim}, ::VectorizedInterpolation{dim, <:AbstractRefShape{dim}}) where {T, dim} = Tensor{3, dim, T}

# Scalar, sdim != rdim (TODO: Use Vec if (s|r)dim <= 3?)
typeof_N(::Type{T}, ::ScalarInterpolation, ::VectorizedInterpolation{sdim, <:AbstractRefShape{rdim}}) where {T, sdim, rdim} = T
typeof_dNdx(::Type{T}, ::ScalarInterpolation, ::VectorizedInterpolation{sdim, <:AbstractRefShape{rdim}}) where {T, sdim, rdim} = SVector{sdim, T}
typeof_dNdξ(::Type{T}, ::ScalarInterpolation, ::VectorizedInterpolation{sdim, <:AbstractRefShape{rdim}}) where {T, sdim, rdim} = SVector{rdim, T}
typeof_d2Ndx2(::Type{T}, ::ScalarInterpolation, ::VectorizedInterpolation{sdim, <:AbstractRefShape{rdim}}) where {T, sdim, rdim} = SMatrix{sdim, sdim, T, sdim * sdim}
typeof_d2Ndξ2(::Type{T}, ::ScalarInterpolation, ::VectorizedInterpolation{sdim, <:AbstractRefShape{rdim}}) where {T, sdim, rdim} = SMatrix{rdim, rdim, T, rdim * rdim}


# Vector, vdim != sdim != rdim (TODO: Use Vec/Tensor if (s|r)dim <= 3?)
typeof_N(::Type{T}, ::VectorInterpolation{vdim}, ::VectorizedInterpolation{sdim, <:AbstractRefShape{rdim}}) where {T, vdim, sdim, rdim} = SVector{vdim, T}
typeof_dNdx(::Type{T}, ::VectorInterpolation{vdim}, ::VectorizedInterpolation{sdim, <:AbstractRefShape{rdim}}) where {T, vdim, sdim, rdim} = SMatrix{vdim, sdim, T, vdim * sdim}
typeof_dNdξ(::Type{T}, ::VectorInterpolation{vdim}, ::VectorizedInterpolation{sdim, <:AbstractRefShape{rdim}}) where {T, vdim, sdim, rdim} = SMatrix{vdim, rdim, T, vdim * rdim}
typeof_d2Ndx2(::Type{T}, ::VectorInterpolation{vdim}, ::VectorizedInterpolation{sdim, <:AbstractRefShape{rdim}}) where {T, vdim, sdim, rdim} = SArray{Tuple{vdim, sdim, sdim}, T, 3, vdim * sdim * sdim}
typeof_d2Ndξ2(::Type{T}, ::VectorInterpolation{vdim}, ::VectorizedInterpolation{sdim, <:AbstractRefShape{rdim}}) where {T, vdim, sdim, rdim} = SArray{Tuple{vdim, rdim, rdim}, T, 3, vdim * rdim * rdim}


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
sdim_from_gradtype(::Type{<:AbstractTensor{<:Any, sdim}}) where {sdim} = sdim
sdim_from_gradtype(::Type{<:SVector{sdim}}) where {sdim} = sdim
sdim_from_gradtype(::Type{<:SMatrix{<:Any, sdim}}) where {sdim} = sdim

# For performance, these must be fully inferable for the compiler.
# args: valname (:CellValues or :FacetValues), shape_gradient_type, eltype(x)
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
struct ArgyrisMapping end

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
required_geo_diff_order(::ArgyrisMapping, fun_diff_order::Int) = fun_diff_order 

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

    sdim, rdim = size(Jinv)
    (rdim != sdim) && error("apply_mapping! for second order gradients and embedded elements not implemented")

    H = gethessian(mapping_values)
    is_vector_valued = first(funvals.Nx) isa Vec
    Jinv_otimesu_Jinv = is_vector_valued ? otimesu(Jinv, Jinv) : nothing
    @inbounds for j in 1:getnbasefunctions(funvals)
        dNdx = dothelper(funvals.dNdξ[j, q_point], Jinv)
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
@inline function apply_mapping!(funvals::FunctionValues{0}, ::CovariantPiolaMapping, q_point::Int, mapping_values, cell, args...)
    Jinv = inv(getjacobian(mapping_values))
    @inbounds for j in 1:getnbasefunctions(funvals)
        d = get_direction(funvals.ip, j, cell)
        Nξ = funvals.Nξ[j, q_point]
        funvals.Nx[j, q_point] = d * (Nξ ⋅ Jinv)
    end
    return nothing
end

@inline function apply_mapping!(funvals::FunctionValues{1}, ::CovariantPiolaMapping, q_point::Int, mapping_values, cell, args...)
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
@inline function apply_mapping!(funvals::FunctionValues{0}, ::ContravariantPiolaMapping, q_point::Int, mapping_values, cell, args...)
    J = getjacobian(mapping_values)
    detJ = det(J)
    @inbounds for j in 1:getnbasefunctions(funvals)
        d = get_direction(funvals.ip, j, cell)
        Nξ = funvals.Nξ[j, q_point]
        funvals.Nx[j, q_point] = d * (J ⋅ Nξ) / detJ
    end
    return nothing
end

@inline function apply_mapping!(funvals::FunctionValues{1}, ::ContravariantPiolaMapping, q_point::Int, mapping_values, cell, args...)
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

# Type storing data for the mapping of argyris element
struct ArgyrisData
    t::NTuple{3,Vec{2,Float64}} # Tangents on edges
    l::NTuple{3,Float64} # Edge lengths
    B::NTuple{3,Tensor{2,2,Float64,4}} # Rotated Jacobians on edges
    J::Tensor{2,2,Float64,4} # Jacobian
end

function ArgyrisData(ip::Lagrange{RefTriangle,1}, coords)
    t1 = coords[1] - coords[2]
    t2 = coords[2] - coords[3]
    t3 = coords[3] - coords[1]
    l1,l2,l3 = l = (norm(t1), norm(t2), norm(t3))

    #TODO: For non-linear geometries, we need to compute three jacobian at each corner.
    #Current implementation only works for non-linear geometries.
    ξ = zero(Vec{2,Float64})
    J, _ = Ferrite.calculate_jacobian_and_spatial_coordinate(ip, ξ, coords)

    t = (t1/l1, t2/l2, t3/l3)
    n = ntuple(i -> rotate(t[i], pi/2), 3)

    n̂ = (Vec((1/√2, 1/√2)), Vec((-1.0, 0.0)), Vec((0.0, -1.0)))
    t̂ = (Vec((1/√2, -1/√2)), Vec((0.0, 1.0)), Vec((-1.0, 0.0)))
    B = zeros(Tensor{2,2}, 3)
    for i in 1:3
        Ĝ = Tensor{2,2}(hcat(n̂[i], t̂[i])')
        G = Tensor{2,2}(hcat(n[i], t[i])')
        B[i] = Ĝ ⋅ J' ⋅ G'
    end

    return ArgyrisData(t, l, Tuple(B), J)
end

#Apply mapping for the Argyris element.
#The shape functions on the physical element are related to the shape functions on reference element Nξ via a sparse matrix M: Nx = M*Nξ
#Since M is sparse, we avoid creating the M-matrix and compute Nx directly.
#For more information see: 
# Robert C. Kirby. A general approach to transforming finite elements. The SMAI Journal of computational mathematics (2018)
function apply_mapping!(funvals::Ferrite.FunctionValues{DO}, ::ArgyrisMapping, q_point::Int, mapping_values, cell, coords) where DO 
    ip_geo = Ferrite.geometric_interpolation(cell)
    @assert ip_geo isa Lagrange{RefTriangle, 1} "Only linear geometries allowed for Argyris interpolation"
    @assert DO < 3

    @show DO
    
    #Compute data required for the argyris mapping
    #TODO: This can be done once per element, but we have to recompute for each quadrature point with the current setup
    argyris_data = ArgyrisData(ip_geo, coords)
    
    Nx = view(funvals.Nx, :, q_point)
    Nξ = view(funvals.Nξ, :, q_point)
    dNdx = (DO > 0) ? view(funvals.dNdx, :, q_point) : nothing
    dNdξ = (DO > 0) ? view(funvals.dNdξ, :, q_point) : nothing
    d²Ndx² = (DO > 1) ? view(funvals.d2Ndx2, :, q_point) : nothing
    d²Ndξ² = (DO > 1) ? view(funvals.d2Ndξ2, :, q_point) : nothing
    
    vdim = length(first(funvals.Nx))
    _argyris_mapping!(argyris_data, Nx, Nξ, dNdx, dNdξ, d²Ndx², d²Ndξ², mapping_values, Val(DO), Val(vdim))

    #Fix directions of normal gradients dofs on edges
    for i in 19:21
        dir = Ferrite.get_direction(vdim==1 ? funvals.ip : funvals.ip.ip, i, cell)
        for d in 1:vdim
            j = (i-1)*vdim + d
            Nx[j] *= dir
            if DO > 0
                dNdx[j] *= dir
            end
            if DO > 1
                d²Ndx²[j] *= dir
            end
        end
    end

    return nothing
end

_hessian_helper(d2Ndξ2::Tensor{2,2}, dNdx, H, Jinv) = Jinv' ⋅ (d2Ndξ2 - dNdx ⋅ H) ⋅ Jinv
_hessian_helper(d2Ndξ2::Tensor{3,2}, dNdx, H, Jinv_otimesu_Jinv) = (d2Ndξ2 - dNdx ⋅ H) ⊡ Jinv_otimesu_Jinv
#Computes Nx = M*Nξ without allocating the M-matrix
function _argyris_mapping!(argyris_data::ArgyrisData, Nx, Nξ, dNdx, dNdξ, d²Ndx², d²Ndξ², mapping_values, ::Val{DO}, ::Val{vdim}) where {DO, vdim}
    
    (; l, B, t) = argyris_data
    J = argyris_data.J
    
    Jinv = DO > 0 ? inv(mapping_values.J) : nothing
    H = DO > 1 ? mapping_values.H : nothing
    Jinv2 = vdim==1 ? Jinv : otimesu(Jinv, Jinv)

    t = @SVector [SVector{2}(argyris_data.t[i]) for i in 1:3]
    τ = @SVector [SVector{3}(argyris_data.t[i][1]^2, 2*argyris_data.t[i][1]*argyris_data.t[i][2], argyris_data.t[i][2]^2) for i in 1:3]
    J = SMatrix{2,2}(argyris_data.J)
    Θ = @SMatrix [J[1,1]^2 J[1,2]*J[1,1] J[1,2]^2; 
                  2*J[1,1]*J[2,1] J[1,2]*J[2,1] + J[1,1]*J[2,2] 2*J[2,2]*J[1,2]; 
                  J[2,1]^2 J[2,1]*J[2,2] J[2,2]^2]

    edgeindeces = ((1, 3), (1, 2), (2, 3))
    edge_to_basefunc = (19, 20, 21)
    _signs = ((1,-1), (-1, 1), (-1, 1))

    #_M = zeros(Float64, 21, 21) 
    for i in 1:3 #Node loop
        e1,e2 = edgeindeces[i]
        b1_scalar,b2_scalar = edge_to_basefunc[e1], edge_to_basefunc[e2]
        offset = 6*(i-1)*vdim #Offset (6 shape functions per node)
        
        #Dofs related to values
        m1 = 15*B[e1][1, 2] / 8l[e1] * _signs[i][1]
        m2 = 15*B[e2][1, 2] / 8l[e2] * _signs[i][2]
        for d in 1:vdim
            o = offset + (d-1)
            b1, b2 = ((b1_scalar-1)*vdim + d, (b2_scalar-1)*vdim + d)
            # -- Shape value
            N1x = Nξ[1+o] + m1*Nξ[b1] + m2*Nξ[b2]
            Nx[1+o] = N1x
            if DO > 0 # -- Shape gradient
                dN1dx = dNdξ[1+o] + m1*dNdξ[b1] + m2*dNdξ[b2]
                dNdx[1+o] = dN1dx ⋅ Jinv
                if DO > 1 # -- Shape hessian
                    d²N1dx² = d²Ndξ²[1+o] + m1*d²Ndξ²[b1] + m2*d²Ndξ²[b2]
                    d²Ndx²[1+o] = _hessian_helper(d²N1dx², dNdx[1+o], H, Jinv2)
                end
            end
        end
        
        #Dof related to gradients
        m3 = -(7/16)*B[e1][1, 2]*t[e1]
        m4 = -(7/16)*B[e2][1, 2]*t[e2]
        for d in 1:vdim
            o = offset + (d-1) + 1*(vdim-1) 
            o1, o2 = (o, o + (vdim-1))
            b1, b2 = ((b1_scalar-1)*vdim + d, (b2_scalar-1)*vdim + d)

            # -- Shape value
            N23 = @SVector [Nξ[2+o1], Nξ[3+o2]]
            N23x = J*N23 + @SVector [m3[1]*Nξ[b1] + m4[1]*Nξ[b2],
                                     m3[2]*Nξ[b1] + m4[2]*Nξ[b2]] 
            Nx[2+o1] = N23x[1]
            Nx[3+o2] = N23x[2]
            if DO > 0 # -- Shape gradient
                dN23dξ = @SVector [dNdξ[2+o1], dNdξ[3+o2]]
                dN23dx = J*dN23dξ + @SVector [m3[1]*dNdξ[b1] + m4[1]*dNdξ[b2],
                                              m3[2]*dNdξ[b1] + m4[2]*dNdξ[b2]] 
                dNdx[2+o1] = dN23dx[1] ⋅ Jinv
                dNdx[3+o2] = dN23dx[2] ⋅ Jinv
                if DO > 1 # -- Shape hessian
                    d²N23dξ² = @SVector [d²Ndξ²[2+o1], d²Ndξ²[3+o2]]
                    d²N23dx² = J*d²N23dξ² + @SVector [m3[1]*d²Ndξ²[b1] + m4[1]*d²Ndξ²[b1],
                                                      m3[2]*d²Ndξ²[b2] + m4[2]*d²Ndξ²[b2]]
                    d²Ndx²[2+o1] = _hessian_helper(d²N23dx²[1], dNdx[2+o], H, Jinv2)
                    d²Ndx²[3+o2] = _hessian_helper(d²N23dx²[2], dNdx[3+o], H, Jinv2) 
                end
            end
        end
        #Dof related to hessians
        m5 = (1/32)*B[e1][1, 2] * τ[e1]*l[e1] * _signs[i][1]
        m6 = (1/32)*B[e2][1, 2] * τ[e2]*l[e2] * _signs[i][2]
        for d in 1:vdim
            # -- Shape value
            o = offset + (d-1) + 3*(vdim-1) 
            o1, o2, o3 = (o, o + (vdim-1), o + 2*(vdim-1))
            b1, b2 = ((b1_scalar-1)*vdim + d, (b2_scalar-1)*vdim + d)

            N456 = @SVector [Nξ[4+o1], Nξ[5+o2], Nξ[6+o3]]
            N456x = Θ*N456 + @SVector[m5[1]*Nξ[b1] + m6[1]*Nξ[b2],
                                    m5[2]*Nξ[b1] + m6[2]*Nξ[b2], 
                                    m5[3]*Nξ[b1] + m6[3]*Nξ[b2]]
            Nx[4 + o1] = N456x[1]
            Nx[5 + o2] = N456x[2]
            Nx[6 + o3] = N456x[3]
            if DO > 0  # -- Shape gradient
                dN456dξ = @SVector [dNdξ[4+o1], dNdξ[5+o2], dNdξ[6+o3]]
                dN456dx = Θ*dN456dξ + @SVector[m5[1]*dNdξ[b1] + m6[1]*dNdξ[b2],
                                            m5[2]*dNdξ[b1] + m6[2]*dNdξ[b2], 
                                            m5[3]*dNdξ[b1] + m6[3]*dNdξ[b2]]
                dNdx[4+o1] = dN456dx[1] ⋅ Jinv
                dNdx[5+o2] = dN456dx[2] ⋅ Jinv
                dNdx[6+o3] = dN456dx[3] ⋅ Jinv
                if DO > 1 # -- Shape hessian
                    d²N456dx² = @SVector [d²Ndξ²[4+o1], d²Ndξ²[5+o2], d²Ndξ²[6+o3]]
                    d²N456dx² = Θ*d²N456dx² + @SVector[m5[1]*d²Ndξ²[b1] + m6[1]*d²Ndξ²[b2],
                                                    m5[2]*d²Ndξ²[b1] + m6[2]*d²Ndξ²[b2], 
                                                    m5[3]*d²Ndξ²[b1] + m6[3]*d²Ndξ²[b2]]
                    d²Ndx²[4+o1] = _hessian_helper(d²N456dx²[1], dNdx[4+o], H, Jinv2)
                    d²Ndx²[5+o2] = _hessian_helper(d²N456dx²[2], dNdx[5+o], H, Jinv2)
                    d²Ndx²[6+o3] = _hessian_helper(d²N456dx²[3], dNdx[6+o], H, Jinv2)
                end
            end
        end

        # row = (i-1)*6
        # _M[1+row, 1+ row] = 1.0
        # _M[1+row, b1_scalar] = m1
        # _M[1+row, b2_scalar] = m2
        # _M[(2:3) .+ row, (2:3) .+ row] = J
        # _M[(2:3) .+ row, b1_scalar] = m3
        # _M[(2:3) .+ row, b2_scalar] = m4
        # _M[(4:6) .+ row, (4:6) .+ row] = Θ
        # _M[(4:6) .+ row, b1_scalar] = m5
        # _M[(4:6) .+ row, b2_scalar] = m6
    end

    #Transform edge functions
    for i in 1:3
        b1 = edge_to_basefunc[i]
        for d in 1:vdim
            j = (b1-1)*vdim + d
            
            Njx = Nξ[j] * B[i][1,1]
            Nx[j] = Njx
            if DO > 0
                dNjdx = dNdξ[j] * B[i][1,1]
                dNdx[j] = dNjdx ⋅ Jinv
                if DO > 1
                    d²Njdx² = d²Ndξ²[j] * B[i][1,1]
                    d²Ndx²[j] = _hessian_helper(d²Njdx², dNdx[j], H, Jinv2)
                end
            end
        end
        #_M[b1,b1] = B[i][1,1]
    end
end
