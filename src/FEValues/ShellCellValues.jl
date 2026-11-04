#################################################################
# ShellCellValues: CellValues extended with derivatives wrt.    #
# the coordinates in a local orthonormal frame, e.g. the        #
# tangent frame of shell elements.                              #
#################################################################

# Equivalent to `foreach(fv -> apply_local_frame!(fv, q_point, mapping, E), fun_values))`,
# see `apply_mapping!(::Tuple, ...)` in CellValues.jl
@generated function apply_local_frame!(fun_values::Tuple{Vararg{FunctionValues, N}}, q_point, mapping, E) where {N}
    expr = Expr(:block)
    for i in 1:N
        push!(expr.args, :(apply_local_frame!(fun_values[$i], q_point, mapping, E)))
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
where `scv.<name>` is of type [`FunctionValues`](@ref Ferrite.FunctionValues), as for [`CellValues`](@ref).

```julia
scv = ShellCellValues(qr, ip, ip_geo^3; update_local_hessians = true)
reinit!(scv, x)
E = Ferrite.getlocalframe(scv, q_point)
∇ₛN = shape_local_gradient(scv, q_point, i)
```
"""
struct ShellCellValues{CV <: CellValues, ET} <: AbstractCellValues
    cv::CV # CellValues (single- or multi-field), with the local frame derivatives in its FunctionValues
    # ShellCellValues are only functional for `ET <: AbstractVector{<:AbstractTensor{2}}`, see `CellValues`.
    E::ET  # Local frame in each quadrature point
end

const SingleFieldShellCellValues = ShellCellValues{<:SingleFieldCellValues}
const MultiFieldShellCellValues = ShellCellValues{<:MultiFieldCellValues}

function ShellCellValues(
        ::Type{T}, qr::QuadratureRule, ip_fun::Union{Interpolation, NamedTuple}, ip_geo::VectorizedInterpolation{sdim, <:AbstractRefShape{rdim}},
        flags::ValuesUpdateFlags, ::Val{LocalDiffOrder}
    ) where {T, sdim, rdim, LocalDiffOrder}
    cv = CellValues(T, qr, ip_fun, ip_geo, _shell_update_flags(flags, Val(LocalDiffOrder)), Val(LocalDiffOrder))
    E = allocate_frames(T, Val(sdim), Val(rdim), getnquadpoints(qr))
    return ShellCellValues(cv, E)
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
    return ShellCellValues(task_local_copy(_cell_values(scv)), task_local_copy(getlocalframes(scv)))
end

# getproperty is overloaded in multi-field mode, so the fields are accessed with getfield
@inline _cell_values(scv::ShellCellValues) = getfield(scv, :cv)
@inline _single_fun_values(scv::SingleFieldShellCellValues) = _single_fun_values(_cell_values(scv))
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
    @eval $f(scv::SingleFieldShellCellValues) = $f(_single_fun_values(scv))
end
for f in (:shape_local_gradient, :shape_local_hessian)
    @eval @propagate_inbounds $f(scv::SingleFieldShellCellValues, q_point::Int, i::Int) = $f(_single_fun_values(scv), q_point, i)
end

# Multiple fields: `scv.u` gives the FunctionValues of field `u`, which includes the local frame derivatives
@inline Base.getproperty(scv::MultiFieldShellCellValues, key::Symbol) = getproperty(_cell_values(scv), key)
Base.propertynames(scv::MultiFieldShellCellValues) = propertynames(_cell_values(scv))

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
        apply_local_frame!(fun_values, q_point, mapping, E)
        frames[q_point] = E
    end
    return nothing
end

function Base.show(io::IO, d::MIME"text/plain", scv::ShellCellValues)
    order = function_local_difforder(first(get_fun_values(scv)))
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
