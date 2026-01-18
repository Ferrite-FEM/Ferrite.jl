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

struct FunctionValues{DiffOrder, IP, N_t, dNdx_t, dNdξ_t, d2Ndx2_t, d2Ndξ2_t, transformation_t} <: AbstractValues
    ip::IP          # ::Interpolation
    Nx::N_t         # ::AbstractMatrix{Union{<:Tensor,<:Number}}
    Nξ::N_t         # ::AbstractMatrix{Union{<:Tensor,<:Number}}
    dNdx::dNdx_t    # ::AbstractMatrix{Union{<:Tensor,<:StaticArray}} or Nothing
    dNdξ::dNdξ_t    # ::AbstractMatrix{Union{<:Tensor,<:StaticArray}} or Nothing
    d2Ndx2::d2Ndx2_t   # ::AbstractMatrix{<:Tensor{2}}  Hessians of geometric shape functions in ref-domain
    d2Ndξ2::d2Ndξ2_t   # ::AbstractMatrix{<:Tensor{2}}  Hessians of geometric shape functions in ref-domain
    transformation::transformation_t # ::BasisTransformation
    function FunctionValues(
            ip::Interpolation,
            Nx::N_t,
            Nξ::N_t,
            dNdx::dNdx_t,
            dNdξ::dNdξ_t,
            d2Ndx2::d2Ndx2_t,
            d2Ndξ2::d2Ndξ2_t,
            transformation::transformation_t
        ) where {N_t, dNdx_t, dNdξ_t, d2Ndx2_t, d2Ndξ2_t, transformation_t}

        difforder = !isnothing(d2Ndx2) ? 2 : (!isnothing(dNdx) ? 1 : 0)
        return new{difforder, typeof(ip), N_t, dNdx_t, dNdξ_t, d2Ndx2_t, d2Ndξ2_t, transformation_t}(ip, Nx, Nξ, dNdx, dNdξ, d2Ndx2, d2Ndξ2, transformation)
    end
end

function FunctionValues{DiffOrder}(::Type{T}, ip::Interpolation, qr::QuadratureRule, ip_geo::VectorizedInterpolation) where {DiffOrder, T}
    assert_same_refshapes(qr, ip, ip_geo)
    n_shape = getnbasefunctions(ip)
    n_qpoints = getnquadpoints(qr)

    Nξ = zeros(typeof_N(T, ip, ip_geo), n_shape, n_qpoints)
    Nx = reinit_needs_cell(ip) ? similar(Nξ) : Nξ

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

    transformation = nothing
    if requires_basis_transformation(ip)
        transformation = BasisTransformation(ip)
    end

    fv = FunctionValues(ip, Nx, Nξ, dNdx, dNdξ, d2Ndx2, d2Ndξ2, transformation)
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
    transformation_copy = _copy_or_nothing(v.transformation)
    return FunctionValues(copy(v.ip), Nx_copy, Nξ_copy, dNdx_copy, dNdξ_copy, d2Ndx2_copy, d2Ndξ2_copy, transformation_copy)
end

getnbasefunctions(funvals::FunctionValues) = size(funvals.Nx, 1)
getnquadpoints(funvals::FunctionValues) = size(funvals.Nx, 2)
@propagate_inbounds shape_value(funvals::FunctionValues, q_point::Int, base_func::Int) = funvals.Nx[base_func, q_point]
@propagate_inbounds shape_gradient(funvals::FunctionValues, q_point::Int, base_func::Int) = funvals.dNdx[base_func, q_point]
@propagate_inbounds shape_hessian(funvals::FunctionValues{2}, q_point::Int, base_func::Int) = funvals.d2Ndx2[base_func, q_point]

function_interpolation(funvals::FunctionValues) = funvals.ip
function_difforder(::FunctionValues{DiffOrder}) where {DiffOrder} = DiffOrder
shape_value_type(funvals::FunctionValues) = eltype(funvals.Nx)
shape_gradient_type(funvals::FunctionValues) = eltype(funvals.dNdx)
shape_gradient_type(::FunctionValues{0}) = nothing
shape_hessian_type(funvals::FunctionValues) = eltype(funvals.d2Ndx2)
shape_hessian_type(::FunctionValues{0}) = nothing
shape_hessian_type(::FunctionValues{1}) = nothing
reinit_needs_cell(funvals::FunctionValues) = reinit_needs_cell(funvals.ip)

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

# If we have transformed the basis (e.g. in Argyris element), then the transformed basis
# is stored in funvals.dNdx. If no transformation is been made, we return the funvals.dNdξ as normal.
get_dNdξ(funvals::FunctionValues) = funvals.transformation === nothing ? (funvals.dNdξ) : (funvals.dNdx)
get_dNdξ_and_d2Ndξ(funvals::FunctionValues) = funvals.transformation === nothing ? (funvals.dNdξ, funvals.d2Ndξ2) : (funvals.dNdx, funvals.d2Ndx2)

# =============
# Apply mapping
# =============
@inline function apply_mapping!(funvals::FunctionValues, q_point::Int, args...)
    return apply_mapping!(funvals, mapping_type(funvals), q_point, args...)
end

# Identity mapping
@inline function apply_mapping!(funvals::FunctionValues{0}, ::IdentityMapping, q_point::Int, mapping_values, cell)
    #Some elements also need to flip the direction of the dof, e.g. if they have normal gradient dofs.
    @inbounds for j in 1:getnbasefunctions(funvals)
        d = get_direction(funvals.ip, j, cell)
        funvals.Nx[j, q_point] *= d
    end
    return nothing
end

@inline function apply_mapping!(funvals::FunctionValues{1}, ::IdentityMapping, q_point::Int, mapping_values, cell)
    dNdξ = get_dNdξ(funvals)
    Jinv = calculate_Jinv(getjacobian(mapping_values))
    @inbounds for j in 1:getnbasefunctions(funvals)
        d = get_direction(funvals.ip, j, cell)
        funvals.dNdx[j, q_point] = d * (dNdξ[j, q_point] ⋅ Jinv)
        funvals.Nx[j, q_point] *= d
    end
    return nothing
end

@inline function apply_mapping!(funvals::FunctionValues{2}, ::IdentityMapping, q_point::Int, mapping_values, cell)
    dNdξ, dN2dξ2 = get_dNdξ_and_d2Ndξ(funvals)
    Jinv = calculate_Jinv(getjacobian(mapping_values))

    sdim, rdim = size(Jinv)
    (rdim != sdim) && error("apply_mapping! for second order gradients and embedded elements not implemented")

    H = gethessian(mapping_values)
    is_vector_valued = first(funvals.Nx) isa Vec
    Jinv_otimesu_Jinv = is_vector_valued ? otimesu(Jinv, Jinv) : nothing
    @inbounds for j in 1:getnbasefunctions(funvals)
        dNdx = dNdξ[j, q_point] ⋅ Jinv
        if is_vector_valued
            d2Ndx2 = (dN2dξ2[j, q_point] - dNdx ⋅ H) ⊡ Jinv_otimesu_Jinv
        else
            d2Ndx2 = Jinv' ⋅ (dN2dξ2[j, q_point] - dNdx ⋅ H) ⋅ Jinv
        end

        d = get_direction(funvals.ip, j, cell)
        funvals.Nx[j, q_point] *= d
        funvals.dNdx[j, q_point] = dNdx * d
        funvals.d2Ndx2[j, q_point] = d2Ndx2 * d
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

struct BasisTransformation{matrix_t <: AbstractMatrix}
    M::matrix_t
    stride::Int #Used for VectorizedInterpolations
end
Base.copy(bt::BasisTransformation) = BasisTransformation(copy(bt.M), bt.stride)

"""
    BasisTransformation(ip::Interpolation)

Creates a transformation matrix `M` for the interpolation `ip`. The
transformation is used to transform the basis (values, gradients, etc.) via
`Nx = M * Nξ` for elements/interpolations that are not equivalently mapped,
where the basis functions must be transformed as linear combinations of
one another.
"""
function BasisTransformation(ip::Interpolation)

    ip, stride = if ip isa VectorizedInterpolation
        ip.ip
    else
        ip, 1
    end

    M0 = init_basis_transformation_matrix(Float64, ip)
    return BasisTransformation(M0, stride)
end

"""
    basistransformation!(out::AbstractVecOrMat{T}, transform::BasisTransformation, in::AbstractVecOrMat) where T

Performs the basis transformation M * Nξ and stores the result in Nx, where M is stored in `transform`.
"""
function basistransformation!(Nx::AbstractVecOrMat{T}, transform::BasisTransformation, Nξ::AbstractVecOrMat) where {T}
    stride = transform.stride
    M = transform.M
    for i in axes(Nx, 2)
        for d in 1:stride
            Nx[d:stride:end, i] = M * Nξ[d:stride:end, i]
        end
    end
    return Nx
end

"""
    init_basis_transformation_matrix(T, ip::Interpolation)

Create the basis-transformation matrix for `ip`, with entries of type `T`.
The matrix can be of any ::AbstractMatrix type.

"""
function init_basis_transformation_matrix(T, ip::Interpolation) end

"""
    calculate_basis_transformation!(funvals::FunctionValues{DiffOrder}, ip_geo, coords::Vector{<:Vec}) where {DiffOrder}

Computes the basis transformation matrix `M` for the physical cell defined by `ip_geo` and the cell coorindates `coords`.
It then uses this transformation matrix to transform the 
"""
function calculate_basis_transformation!(funvals::FunctionValues{DiffOrder}, ip_geo, coords::Vector{<:Vec}) where {DiffOrder}
    funvals.transformation === nothing && return nothing

    calculate_basis_transformation_matrix!(funvals, ip_geo, coords) #Update M-matrix
    #Perform the basis transformation (Nx = M*Nξ):
    # NOTE: we store the results *temporarily* in Nx, dNdx and d2Ndx2
    basistransformation!(funvals.Nx, funvals.transformation, funvals.Nξ)
    DiffOrder >= 1 && basistransformation!(funvals.dNdx, funvals.transformation, funvals.dNdξ)
    DiffOrder >= 2 && basistransformation!(funvals.d2Ndx2, funvals.transformation, funvals.d2Ndξ2)
    return nothing
end

######################################
# Transformation for Argyris element #
######################################
function init_basis_transformation_matrix(T, ip::Argyris)
    return zeros(T, 21, 21)
end

function calculate_basis_transformation_matrix!(funvals::FunctionValues{DiffOrder, IP}, ip_geo, coords::Vector{Vec{dim, T}}) where {DiffOrder, IP <: Argyris, dim, T}
    @assert ip_geo isa Lagrange{RefTriangle, 1} "Only linear geometries allowed for Argyris interpolation"
    #Compute data required for the argyris basis transformation matrix
    (t, l, B, J) = compute_argyris_data(ip_geo, coords)

    τ = [Vec{3}((t[i][1]^2, 2 * t[i][1] * t[i][2], t[i][2]^2)) for i in 1:3] #Todo allocation free
    Θ = Tensor{2, 3}(
        [
            J[1, 1]^2 J[1, 2] * J[1, 1] J[1, 2]^2;
            2 * J[1, 1] * J[2, 1] J[1, 2] * J[2, 1] + J[1, 1] * J[2, 2] 2 * J[2, 2] * J[1, 2];
            J[2, 1]^2 J[2, 1] * J[2, 2] J[2, 2]^2
        ]
    )

    M = funvals.transformation.M
    fill!(M, zero(eltype(M)))

    edgeindeces = ((1, 3), (1, 2), (2, 3))
    edge_to_basefunc = (19, 20, 21)
    _signs = ((1, -1), (-1, 1), (-1, 1))

    for i in 1:3 #Node loop
        e1, e2 = edgeindeces[i]
        b1_scalar, b2_scalar = edge_to_basefunc[e1], edge_to_basefunc[e2]

        m1 = 15 * B[e1][1, 2] / 8l[e1] * _signs[i][1]
        m2 = 15 * B[e2][1, 2] / 8l[e2] * _signs[i][2]
        row = (i - 1) * 6
        M[1 + row, 1 + row] = 1.0
        M[1 + row, b1_scalar] = m1
        M[1 + row, b2_scalar] = m2

        m3 = -(7 / 16) * B[e1][1, 2] * t[e1]
        m4 = -(7 / 16) * B[e2][1, 2] * t[e2]
        M[(2:3) .+ row, (2:3) .+ row] = J
        M[(2:3) .+ row, b1_scalar] = m3
        M[(2:3) .+ row, b2_scalar] = m4

        m5 = (1 / 32) * B[e1][1, 2] * τ[e1] * l[e1] * _signs[i][1]
        m6 = (1 / 32) * B[e2][1, 2] * τ[e2] * l[e2] * _signs[i][2]
        M[(4:6) .+ row, (4:6) .+ row] = Θ
        M[(4:6) .+ row, b1_scalar] = m5
        M[(4:6) .+ row, b2_scalar] = m6
    end

    for i in 1:3
        b1 = edge_to_basefunc[i]
        M[b1, b1] = B[i][1, 1]
    end

    return
end

function _compute_B(t, n̂, t̂, J)
    n = Vec(-t[2], t[1]) # Rotate 90 deg
    Ĝ = Tensor{2, 2}((n̂[1], t̂[1], n̂[2], t̂[2]))
    G = Tensor{2, 2}((n[1], t[1], n[2], t[2]))
    return Ĝ ⋅ J' ⋅ G'
end

function compute_argyris_data(ip::Lagrange{RefTriangle, 1}, coords)
    t1 = coords[1] - coords[2]
    t2 = coords[2] - coords[3]
    t3 = coords[3] - coords[1]
    l1, l2, l3 = l = (norm(t1), norm(t2), norm(t3))

    #TODO: For non-linear geometries, we need to compute three jacobian at each corner.
    #Current implementation only works for linear geometries.
    ξ = zero(Vec{2, Float64})
    J, _ = calculate_jacobian_and_spatial_coordinate(ip, ξ, coords)

    t = (t1 / l1, t2 / l2, t3 / l3)

    ts = (t1 / l1, t2 / l2, t3 / l3)
    n̂s = (Vec((1 / √2, 1 / √2)), Vec((-1.0, 0.0)), Vec((0.0, -1.0)))
    t̂s = (Vec((1 / √2, -1 / √2)), Vec((0.0, 1.0)), Vec((-1.0, 0.0)))
    B = map(ts, n̂s, t̂s) do t, n̂, t̂
        _compute_B(t, n̂, t̂, J)
    end

    return (t, l, B, J)
end
