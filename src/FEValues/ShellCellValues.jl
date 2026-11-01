#################################################################
# ShellCellValues: CellValues extended with derivatives wrt.    #
# the coordinates in a local orthonormal frame, e.g. the        #
# tangent frame of shell elements.                              #
#################################################################

"""
    gram_schmidt_frame(J)

Return the orthonormal local frame, `E`, obtained by Gram-Schmidt orthonormalization of the columns
of the jacobian `J = ∂x/∂ξ` (size `sdim × rdim`). `E` has the same size as `J`, and its columns span
the same (tangent) space as the columns of `J`. The first column of `E` is aligned with the first column of `J`.
This is the local frame used by [`ShellCellValues`](@ref), see [`Ferrite.getlocalframe`](@ref).
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

# =============================================================
# LocalFunctionValues: Local frame derivatives for one interpolation
# =============================================================
struct LocalFunctionValues{DiffOrder, dNdξ_t, d2Ndξ2_t, dNds_t, d2Nds2_t}
    # As for `FunctionValues`, arrays of one order higher are allowed for the SoA (GPU) layout, see `soa_utils.jl`.
    dNdξ::dNdξ_t     # ::AbstractMatrix{<:Tensor}               Gradients in the reference domain (precomputed)
    d2Ndξ2::d2Ndξ2_t # ::AbstractMatrix{<:Tensor} or Nothing    Hessians in the reference domain (precomputed)
    dNds::dNds_t     # ::AbstractMatrix{<:Tensor}               Gradients wrt. the local frame coordinates
    d2Nds2::d2Nds2_t # ::AbstractMatrix{<:Tensor} or Nothing    Hessians wrt. the local frame coordinates
    function LocalFunctionValues(dNdξ::AbstractArray, ::Nothing, dNds::AbstractArray, ::Nothing)
        return new{1, typeof(dNdξ), Nothing, typeof(dNds), Nothing}(dNdξ, nothing, dNds, nothing)
    end
    function LocalFunctionValues(dNdξ::AbstractArray, d2Ndξ2::AbstractArray, dNds::AbstractArray, d2Nds2::AbstractArray)
        return new{2, typeof(dNdξ), typeof(d2Ndξ2), typeof(dNds), typeof(d2Nds2)}(dNdξ, d2Ndξ2, dNds, d2Nds2)
    end
end

function LocalFunctionValues{DiffOrder}(::Type{T}, ip::Interpolation, qr::QuadratureRule, ip_geo::VectorizedInterpolation) where {DiffOrder, T}
    assert_same_refshapes(qr, ip, ip_geo)
    if !isa(mapping_type(ip), IdentityMapping)
        throw(ArgumentError("Local frame derivatives are only supported for interpolations with identity mapping"))
    end
    n_shape = getnbasefunctions(ip)
    n_qpoints = getnquadpoints(qr)
    # The local frame derivatives have the same type as the reference derivatives (rdim local coordinates)
    Nξ = zeros(typeof_N(T, ip, ip_geo), n_shape, n_qpoints) # Only used for precomputing
    dNdξ = zeros(typeof_dNdξ(T, ip, ip_geo), n_shape, n_qpoints)
    dNds = fill(zero(typeof_dNdξ(T, ip, ip_geo)) * T(NaN), n_shape, n_qpoints)
    if DiffOrder == 1
        reference_shape_gradients_and_values!(dNdξ, Nξ, ip, getpoints(qr))
        return LocalFunctionValues(dNdξ, nothing, dNds, nothing)
    elseif DiffOrder == 2
        d2Ndξ2 = zeros(typeof_d2Ndξ2(T, ip, ip_geo), n_shape, n_qpoints)
        d2Nds2 = fill(zero(typeof_d2Ndξ2(T, ip, ip_geo)) * T(NaN), n_shape, n_qpoints)
        reference_shape_hessians_gradients_and_values!(d2Ndξ2, dNdξ, Nξ, ip, getpoints(qr))
        return LocalFunctionValues(dNdξ, d2Ndξ2, dNds, d2Nds2)
    else
        throw(ArgumentError("Currently only local gradients and hessians can be updated in ShellCellValues"))
    end
end

function task_local_copy(lfv::LocalFunctionValues)
    return LocalFunctionValues(
        task_local_copy(lfv.dNdξ), task_local_copy(lfv.d2Ndξ2), task_local_copy(lfv.dNds), task_local_copy(lfv.d2Nds2)
    )
end

@propagate_inbounds shape_local_gradient(lfv::LocalFunctionValues, q_point::Int, base_func::Int) = lfv.dNds[base_func, q_point]
@propagate_inbounds shape_local_hessian(lfv::LocalFunctionValues{2}, q_point::Int, base_func::Int) = lfv.d2Nds2[base_func, q_point]
function_local_difforder(::LocalFunctionValues{DiffOrder}) where {DiffOrder} = DiffOrder
shape_local_gradient_type(lfv::LocalFunctionValues) = eltype(lfv.dNds)
shape_local_hessian_type(lfv::LocalFunctionValues) = eltype(lfv.d2Nds2)
shape_local_hessian_type(::LocalFunctionValues{1}) = nothing

# Local frame derivatives, given the local frame `E` in the quadrature point
@inline function apply_local_frame!(lfv::LocalFunctionValues{1}, q_point::Int, mapping_values::MappingValues, E::AbstractTensor{2})
    Binv = inv(E' ⋅ getjacobian(mapping_values))
    @inbounds for j in 1:size(lfv.dNds, 1)
        lfv.dNds[j, q_point] = lfv.dNdξ[j, q_point] ⋅ Binv
    end
    return nothing
end

@inline function apply_local_frame!(lfv::LocalFunctionValues{2}, q_point::Int, mapping_values::MappingValues, E::AbstractTensor{2})
    J = getjacobian(mapping_values)
    H = gethessian(mapping_values)
    Binv = inv(E' ⋅ J)
    Hs = E' ⋅ H
    is_vector_valued = first(lfv.dNdξ) isa AbstractTensor{2}
    Binv_otimesu_Binv = is_vector_valued ? otimesu(Binv, Binv) : nothing
    @inbounds for j in 1:size(lfv.dNds, 1)
        dNds = lfv.dNdξ[j, q_point] ⋅ Binv
        if is_vector_valued
            d2Nds2 = (lfv.d2Ndξ2[j, q_point] - dNds ⋅ Hs) ⊡ Binv_otimesu_Binv
        else
            d2Nds2 = Binv' ⋅ (lfv.d2Ndξ2[j, q_point] - dNds ⋅ Hs) ⋅ Binv
        end
        lfv.dNds[j, q_point] = dNds
        lfv.d2Nds2[j, q_point] = d2Nds2
    end
    return nothing
end

# Equivalent to `foreach(lfv -> apply_local_frame!(lfv, q_point, mapping, E), local_values))`,
# see `apply_mapping!(::Tuple, ...)` in CellValues.jl
@generated function apply_local_frame!(local_values::Tuple{Vararg{LocalFunctionValues, N}}, q_point, mapping, E) where {N}
    expr = Expr(:block)
    for i in 1:N
        push!(expr.args, :(apply_local_frame!(local_values[$i], q_point, mapping, E)))
    end
    return quote
        $(Expr(:meta, :inline))
        @inbounds return $expr
    end
end

# =============================================================
# ShellCellValues
# =============================================================
"""
    ShellCellValues([::Type{T},] quad_rule::QuadratureRule, func_interpol::Interpolation, [geom_interpol::Interpolation]; kwargs...)
    ShellCellValues([::Type{T},] quad_rule::QuadratureRule, func_interpols::NamedTuple, [geom_interpol::Interpolation]; kwargs...)

A `ShellCellValues` object works as a [`CellValues`](@ref) (see its docstring for the arguments), but in addition
the derivatives of the shape functions wrt. the coordinates in a local orthonormal frame in each quadrature point
are computed in [`reinit!`](@ref), see [`shape_local_gradient`](@ref). This is useful for e.g. shell, plate,
and beam elements, and is also supported for embedded elements, e.g. `ShellCellValues(qr, ip, ip_geo^3)` for
surfaces in 3D. The local frame is computed by Gram-Schmidt orthonormalization of the jacobian, see
[`Ferrite.gram_schmidt_frame`](@ref), and is accessed with [`Ferrite.getlocalframe`](@ref).
Only interpolations with identity mapping are supported.

**Keyword arguments:** The following keyword arguments are experimental and may change in future minor releases
* `update_local_hessians`: Specifies if the hessians of the shape functions wrt. the local frame coordinates
  should be updated (default false). The gradients wrt. the local frame coordinates are always updated.
* The keyword arguments of [`CellValues`](@ref) (e.g. `update_gradients`, `update_hessians`, and `update_detJdV`)
  are forwarded to the wrapped `CellValues`.

**Common methods:** In addition to the methods of [`CellValues`](@ref),
* [`shape_local_gradient`](@ref)
* [`shape_local_hessian`](@ref)
* [`function_local_gradient`](@ref)
* [`function_local_hessian`](@ref)
* [`Ferrite.getlocalframe`](@ref)

# Multiple fields
Constructed with a `NamedTuple` of interpolations, the field specific functions are called on
`scv.<name>`, e.g. `shape_gradient(scv.u, q_point, i)` and `shape_local_gradient(scv.u, q_point, i)`,
as for [`CellValues`](@ref).

```julia
scv = ShellCellValues(qr, ip, ip_geo^3; update_local_hessians = true)
reinit!(scv, x)
E = Ferrite.getlocalframe(scv, q_point)
∇ₛN = shape_local_gradient(scv, q_point, i)
```
"""
struct ShellCellValues{CV <: CellValues, LFV <: Tuple{Vararg{LocalFunctionValues}}, ET} <: AbstractCellValues
    cv::CV            # CellValues (single- or multi-field)
    local_values::LFV # LocalFunctionValues collected in a tuple, one for each FunctionValues in `cv`
    # ShellCellValues are only functional for `ET <: AbstractVector{<:AbstractTensor{2}}`, see `CellValues`.
    E::ET             # Local frame in each quadrature point
end

const SingleFieldShellCellValues = ShellCellValues{<:SingleFieldCellValues}
const MultiFieldShellCellValues = ShellCellValues{<:MultiFieldCellValues}

function ShellCellValues(
        ::Type{T}, qr::QuadratureRule, ip_fun::Union{Interpolation, NamedTuple}, ip_geo::VectorizedInterpolation{sdim, <:AbstractRefShape{rdim}},
        flags::ValuesUpdateFlags, ::Val{LocalDiffOrder}
    ) where {T, sdim, rdim, LocalDiffOrder}
    cv = CellValues(T, qr, ip_fun, ip_geo, _shell_update_flags(flags, Val(LocalDiffOrder)))
    # Same ordering as the (unique) FunctionValues in `cv`
    unique_ips = ip_fun isa Interpolation ? (ip_fun,) : unique(values(ip_fun))
    local_values = tuple((LocalFunctionValues{LocalDiffOrder}(T, ip, qr, ip_geo) for ip in unique_ips)...)
    E = allocate_frames(T, Val(sdim), Val(rdim), getnquadpoints(qr))
    return ShellCellValues(cv, local_values, E)
end

# The local derivatives require the geometric derivatives up to the same order
function _shell_update_flags(::ValuesUpdateFlags{FunDiffOrder, GeoDiffOrder, DetJdV}, ::Val{LocalDiffOrder}) where {FunDiffOrder, GeoDiffOrder, DetJdV, LocalDiffOrder}
    return ValuesUpdateFlags{FunDiffOrder, max(GeoDiffOrder, LocalDiffOrder), DetJdV}()
end
_local_difforder(update_local_hessians::Bool) = Val(update_local_hessians ? 2 : 1)
_local_difforder(::Val{update_local_hessians}) where {update_local_hessians} = _local_difforder(update_local_hessians)

ShellCellValues(qr::QuadratureRule, ip::Union{Interpolation, NamedTuple}, args...; kwargs...) = ShellCellValues(Float64, qr, ip, args...; kwargs...)
function ShellCellValues(::Type{T}, qr::QuadratureRule, ip::Union{Interpolation, NamedTuple}, ip_geo::ScalarInterpolation; kwargs...) where {T}
    return ShellCellValues(T, qr, ip, VectorizedInterpolation(ip_geo); kwargs...)
end
function ShellCellValues(
        ::Type{T}, qr::QuadratureRule, ip::Union{Interpolation, NamedTuple}, ip_geo::VectorizedInterpolation = default_geometric_interpolation(ip);
        update_local_hessians = Val(false), kwargs...
    ) where {T}
    return ShellCellValues(T, qr, ip, ip_geo, ValuesUpdateFlags(ip; kwargs...), _local_difforder(update_local_hessians))
end

function task_local_copy(scv::ShellCellValues)
    return ShellCellValues(
        task_local_copy(_cell_values(scv)), map(task_local_copy, getfield(scv, :local_values)), task_local_copy(getlocalframes(scv))
    )
end

# getproperty is overloaded in multi-field mode, so the fields are accessed with getfield
@inline _cell_values(scv::ShellCellValues) = getfield(scv, :cv)
@inline _single_local_values(scv::SingleFieldShellCellValues) = getfield(scv, :local_values)[1]
getlocalframes(scv::ShellCellValues) = getfield(scv, :E)

"""
    getlocalframe(scv::ShellCellValues, q_point::Int)

Return the local orthonormal frame, ``\\mathbf{E}``, in quadrature point `q_point`, used for the
derivatives wrt. the local frame coordinates (see [`shape_local_gradient`](@ref)). The frame is
computed in [`reinit!`](@ref) by Gram-Schmidt orthonormalization of the columns of the jacobian,
``\\mathbf{J} = \\partial \\mathbf{x} / \\partial \\boldsymbol{\\xi}``, see [`Ferrite.gram_schmidt_frame`](@ref).
"""
@propagate_inbounds getlocalframe(scv::ShellCellValues, q_point::Int) = getlocalframes(scv)[q_point]

# Geometry, quadrature, and function values from the wrapped CellValues
get_geo_mapping(scv::ShellCellValues) = get_geo_mapping(_cell_values(scv))
getdetJdVs(scv::ShellCellValues) = getdetJdVs(_cell_values(scv))
get_quadrature_rule(scv::ShellCellValues) = get_quadrature_rule(_cell_values(scv))
get_fun_values(scv::ShellCellValues) = get_fun_values(_cell_values(scv))
reinit_needs_cell(scv::ShellCellValues) = reinit_needs_cell(_cell_values(scv))

# Single field
for f in (:getnbasefunctions, :function_interpolation, :function_difforder, :shape_value_type, :shape_gradient_type, :shape_hessian_type)
    @eval $f(scv::SingleFieldShellCellValues) = $f(_cell_values(scv))
end
for f in (:shape_value, :shape_gradient, :shape_hessian)
    @eval @propagate_inbounds $f(scv::SingleFieldShellCellValues, q_point::Int, i::Int) = $f(_cell_values(scv), q_point, i)
end
for f in (:function_local_difforder, :shape_local_gradient_type, :shape_local_hessian_type)
    @eval $f(scv::SingleFieldShellCellValues) = $f(_single_local_values(scv))
end
for f in (:shape_local_gradient, :shape_local_hessian)
    @eval @propagate_inbounds $f(scv::SingleFieldShellCellValues, q_point::Int, i::Int) = $f(_single_local_values(scv), q_point, i)
end

# Multiple fields: `scv.u` gives the FunctionValues and LocalFunctionValues of field `u`
"""
    ShellFunctionValues

The values of a single field in a multi-field [`ShellCellValues`](@ref), e.g. `scv.u`.
Supports the same functions as [`FunctionValues`](@ref Ferrite.FunctionValues), and in addition
[`shape_local_gradient`](@ref), [`shape_local_hessian`](@ref), [`function_local_gradient`](@ref),
and [`function_local_hessian`](@ref).
"""
struct ShellFunctionValues{FV <: FunctionValues, LFV <: LocalFunctionValues} <: AbstractValues
    fun_values::FV
    local_values::LFV
end

@inline function Base.getproperty(scv::MultiFieldShellCellValues, key::Symbol)
    idx = getproperty(getfield(_cell_values(scv), :fun_values_nt), key)
    return ShellFunctionValues(_tuple_index(get_fun_values(scv), idx), _tuple_index(getfield(scv, :local_values), idx))
end
Base.propertynames(scv::MultiFieldShellCellValues) = propertynames(_cell_values(scv))

for f in (:getnbasefunctions, :getnquadpoints, :function_interpolation, :function_difforder, :shape_value_type, :shape_gradient_type, :shape_hessian_type)
    @eval $f(sfv::ShellFunctionValues) = $f(sfv.fun_values)
end
for f in (:shape_value, :shape_gradient, :shape_hessian)
    @eval @propagate_inbounds $f(sfv::ShellFunctionValues, q_point::Int, i::Int) = $f(sfv.fun_values, q_point, i)
end
for f in (:function_local_difforder, :shape_local_gradient_type, :shape_local_hessian_type)
    @eval $f(sfv::ShellFunctionValues) = $f(sfv.local_values)
end
for f in (:shape_local_gradient, :shape_local_hessian)
    @eval @propagate_inbounds $f(sfv::ShellFunctionValues, q_point::Int, i::Int) = $f(sfv.local_values, q_point, i)
end

# Error messages for functions that should be called on the specific field
function getnbasefunctions(scv::MultiFieldShellCellValues)
    k = first(propertynames(scv)) # Pick the first field to use in example
    throw(ArgumentError("getnbasefunctions isn't applicable to a multi-field ShellCellValues. Use on the specific field, e.g. getnbasefunctions(scv.$k)"))
end
for f in (:shape_value, :shape_gradient, :shape_symmetric_gradient, :shape_divergence, :shape_local_gradient, :shape_local_hessian)
    @eval function $f(scv::MultiFieldShellCellValues, ::Int, ::Int)
        k = first(propertynames(scv)) # Pick the first field to use in example
        fun = $f                        # Make the function name available to use in the error message
        throw(ArgumentError("$fun isn't applicable to a multi-field ShellCellValues. Use on the specific field, e.g. $fun(scv.$k, q_point, shapenr)"))
    end
end
for f in (:function_value, :function_gradient, :function_symmetric_gradient, :function_divergence, :function_local_gradient, :function_local_hessian)
    @eval function $f(scv::MultiFieldShellCellValues, ::Int, ::AbstractVector, args...)
        k = first(propertynames(scv)) # Pick the first field to use in example
        fun = $f                        # Make the function name available to use in the error message
        throw(ArgumentError("$fun isn't applicable to a multi-field ShellCellValues. Use on the specific field, e.g. $fun(scv.$k, q_point, ae, [dofrange])"))
    end
end

function check_reinit_sdim_consistency(scv::ShellCellValues, x::AbstractVector)
    return check_reinit_sdim_consistency(_cell_values(scv), x)
end

function reinit!(scv::ShellCellValues, cell::Union{AbstractCell, Nothing}, x::AbstractVector{<:Vec})
    geo_mapping = get_geo_mapping(scv)
    fun_values = get_fun_values(scv)
    local_values = getfield(scv, :local_values)
    frames = getlocalframes(scv)
    n_geom_basefuncs = getngeobasefunctions(geo_mapping)

    check_reinit_sdim_consistency(scv, x)
    if cell === nothing && reinit_needs_cell(scv)
        throw(ArgumentError("The cell::AbstractCell input is required to reinit! non-identity function mappings"))
    end
    if !checkbounds(Bool, x, 1:n_geom_basefuncs) || length(x) != n_geom_basefuncs
        throw_incompatible_coord_length(length(x), n_geom_basefuncs)
    end
    @inbounds for (q_point, w) in enumerate(getweights(get_quadrature_rule(scv)))
        mapping = calculate_mapping(geo_mapping, q_point, x)
        _update_detJdV!(getdetJdVs(scv), q_point, w, mapping)
        apply_mapping!(fun_values, q_point, mapping, cell)
        # Local frame derivatives
        E = gram_schmidt_frame(getjacobian(mapping))
        apply_local_frame!(local_values, q_point, mapping, E)
        frames[q_point] = E
    end
    return nothing
end

function Base.show(io::IO, d::MIME"text/plain", scv::ShellCellValues)
    order = function_local_difforder(first(getfield(scv, :local_values)))
    print(io, "ShellCellValues with local frame ", order == 1 ? "gradients" : "gradients and hessians", " wrapping a\n")
    show(io, d, _cell_values(scv))
    return
end

# =============================================================
# Local frame derivatives: Docstrings and generic functions
# =============================================================
"""
    shape_local_gradient(fe_v::AbstractValues, q_point::Int, base_function::Int)

Return the gradient of shape function `base_function` wrt. the local frame coordinates, ``\\mathbf{s}``,
evaluated in quadrature point `q_point`. Requires `fe_v` to be a [`ShellCellValues`](@ref) (or a field of
one, e.g. `scv.u`).

The local frame, ``\\mathbf{E}``, in each quadrature point is computed in [`reinit!`](@ref) by Gram-Schmidt
orthonormalization of the columns of the jacobian, ``\\mathbf{J} = \\partial \\mathbf{x} / \\partial \\boldsymbol{\\xi}``,
(see [`Ferrite.gram_schmidt_frame`](@ref)), and is accessed with [`Ferrite.getlocalframe`](@ref). Its orthonormal
columns span the same space as the columns of ``\\mathbf{J}``, i.e. the tangent space for embedded
elements (e.g. shells). The local coordinates are defined as
``\\mathbf{s} = \\mathbf{E}^\\mathrm{T} \\cdot (\\mathbf{x} - \\mathbf{x}_q)``, with ``\\mathbf{E}`` frozen
at the quadrature point, ``\\mathbf{x}_q``, and the local gradient is
``\\mathbf{\\nabla}_s N = \\partial N / \\partial \\boldsymbol{\\xi} \\cdot \\mathbf{B}^{-1}``,
where ``\\mathbf{B} = \\mathbf{E}^\\mathrm{T} \\cdot \\mathbf{J}``. For non-embedded elements,
``\\mathbf{\\nabla}_s N = \\mathbf{\\nabla} N \\cdot \\mathbf{E}``.
"""
shape_local_gradient(fe_v::AbstractValues, q_point::Int, base_function::Int)

"""
    shape_local_hessian(fe_v::AbstractValues, q_point::Int, base_function::Int)

Return the hessian of shape function `base_function` wrt. the local frame coordinates, ``\\mathbf{s}``,
(see [`shape_local_gradient`](@ref)) evaluated in quadrature point `q_point`.
Requires `update_local_hessians = true` when creating the [`ShellCellValues`](@ref).

With ``\\mathbf{H} = \\partial^2 \\mathbf{x} / \\partial \\boldsymbol{\\xi}^2``, the local hessian of a scalar
shape function is calculated as
``\\mathbf{\\nabla}_s \\mathbf{\\nabla}_s N = \\mathbf{B}^{-\\mathrm{T}} \\cdot \\left[\\partial^2 N / \\partial \\boldsymbol{\\xi}^2 - \\mathbf{\\nabla}_s N \\cdot (\\mathbf{E}^\\mathrm{T} \\cdot \\mathbf{H})\\right] \\cdot \\mathbf{B}^{-1}``.
For non-embedded elements, ``\\mathbf{\\nabla}_s \\mathbf{\\nabla}_s N = \\mathbf{E}^\\mathrm{T} \\cdot \\mathbf{\\nabla} \\mathbf{\\nabla} N \\cdot \\mathbf{E}``.
"""
shape_local_hessian(fe_v::AbstractValues, q_point::Int, base_function::Int)

"""
    function_local_gradient(fe_v::AbstractValues, q_point::Int, u::AbstractVector, [dof_range])

Compute the gradient of the function wrt. the local frame coordinates, ``\\mathbf{s}``, in a quadrature point,
see [`shape_local_gradient`](@ref). `u` is a vector with scalar values for the degrees of freedom.
The local gradient is computed as
``\\mathbf{\\nabla}_s u = \\sum\\limits_{i = 1}^n \\mathbf{\\nabla}_s N_i u_i``,
where ``u_i`` are the nodal values of the function.
"""
function function_local_gradient(fe_v::AbstractValues, q_point::Int, u::AbstractVector, dof_range = eachindex(u))
    n_base_funcs = getnbasefunctions(fe_v)
    length(dof_range) == n_base_funcs || throw_incompatible_dof_length(length(dof_range), n_base_funcs)
    @boundscheck checkbounds(u, dof_range)
    @boundscheck checkquadpoint(fe_v, q_point)
    grad = function_local_gradient_init(fe_v, u)
    @inbounds for (i, j) in enumerate(dof_range)
        grad += shape_local_gradient(fe_v, q_point, i) * u[j]
    end
    return grad
end

"""
    shape_local_gradient_type(fe_v::AbstractValues)

Return the type of `shape_local_gradient(fe_v, q_point, base_function)`
"""
function shape_local_gradient_type(fe_v::AbstractValues)
    return typeof(shape_local_gradient(fe_v, 1, 1))
end

function function_local_gradient_init(cv::AbstractValues, ::AbstractVector{T}) where {T}
    return zero(shape_local_gradient_type(cv)) * zero(T)
end

"""
    function_local_hessian(fe_v::AbstractValues, q_point::Int, u::AbstractVector, [dof_range])

Compute the hessian of the function wrt. the local frame coordinates, ``\\mathbf{s}``, in a quadrature point,
see [`shape_local_hessian`](@ref). `u` is a vector with scalar values for the degrees of freedom.
Requires `update_local_hessians = true` when creating the [`ShellCellValues`](@ref).
"""
function function_local_hessian(fe_v::AbstractValues, q_point::Int, u::AbstractVector, dof_range = eachindex(u))
    n_base_funcs = getnbasefunctions(fe_v)
    length(dof_range) == n_base_funcs || throw_incompatible_dof_length(length(dof_range), n_base_funcs)
    @boundscheck checkbounds(u, dof_range)
    @boundscheck checkquadpoint(fe_v, q_point)
    hess = function_local_hessian_init(fe_v, u)
    @inbounds for (i, j) in enumerate(dof_range)
        hess += shape_local_hessian(fe_v, q_point, i) * u[j]
    end
    return hess
end

"""
    shape_local_hessian_type(fe_v::AbstractValues)

Return the type of `shape_local_hessian(fe_v, q_point, base_function)`
"""
function shape_local_hessian_type(fe_v::AbstractValues)
    return typeof(shape_local_hessian(fe_v, 1, 1))
end

function function_local_hessian_init(cv::AbstractValues, ::AbstractVector{T}) where {T}
    return zero(shape_local_hessian_type(cv)) * zero(T)
end
