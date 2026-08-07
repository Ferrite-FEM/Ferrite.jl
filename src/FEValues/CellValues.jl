# AbstractCellValues common functions

getnquadpoints(cv::AbstractCellValues) = getnquadpoints(get_quadrature_rule(cv))

function getdetJdV(cv::AbstractCellValues, q_point)
    detJdVs = getdetJdVs(cv)
    detJdVs === nothing && throw(ArgumentError("detJdV is not saved in $(nameof(typeof(cv)))"))
    return detJdVs[q_point]
end

@propagate_inbounds getngeobasefunctions(cv::AbstractCellValues) = getngeobasefunctions(get_geo_mapping(cv))
@propagate_inbounds geometric_value(cv::AbstractCellValues, args...) = geometric_value(get_geo_mapping(cv), args...)
geometric_interpolation(cv::AbstractCellValues) = geometric_interpolation(get_geo_mapping(cv))

"""
    CellValues([::Type{T},] quad_rule::QuadratureRule, func_interpol::Interpolation, [geom_interpol::Interpolation])
    CellValues([::Type{T},] quad_rule::QuadratureRule, func_interpols::NamedTuple, [geom_interpol::Interpolation])

A `CellValues` object facilitates the process of evaluating values of shape functions, gradients of shape functions,
values of nodal functions, gradients and divergences of nodal functions etc. in the finite element cell.

**Arguments:**
* `T`: an optional argument (default to `Float64`) to determine the type the internal data is stored as.
* `quad_rule`: an instance of a [`QuadratureRule`](@ref)
* `func_interpol`: an instance of an [`Interpolation`](@ref) used to interpolate the approximated function
* `func_interpols`: a named tuple with entries of type `Interpolation`, one for each field, see
  *Multiple fields* below
* `geom_interpol`: an optional instance of an [`Interpolation`](@ref) which is used to interpolate the geometry.
  By default linear Lagrange interpolation is used. For embedded elements the geometric interpolations should
  be vectorized to the spatial dimension.

**Keyword arguments:** The following keyword arguments are experimental and may change in future minor releases
* `update_gradients`: Specifies if the gradients of the shape functions should be updated (default true)
* `update_hessians`: Specifies if the hessians of the shape functions should be updated (default false)
* `update_detJdV`: Specifies if the volume associated with each quadrature point should be updated (default true)

**Common methods:**

* [`reinit!`](@ref)
* [`getnquadpoints`](@ref)
* [`getdetJdV`](@ref)

* [`shape_value`](@ref)
* [`shape_gradient`](@ref)
* [`shape_symmetric_gradient`](@ref)
* [`shape_divergence`](@ref)

* [`function_value`](@ref)
* [`function_gradient`](@ref)
* [`function_symmetric_gradient`](@ref)
* [`function_divergence`](@ref)
* [`spatial_coordinate`](@ref)

# Multiple fields

Constructed with a `NamedTuple` of interpolations, `cv::CellValues` includes values associated with
multiple interpolations sharing the same quadrature points and geometric interpolation. In this case,
the functions above related to specific shape functions are called on `cv.<name>`, where `<name>` is
the key used when creating the named tuple, instead of directly on `cv`. `cv.<name>` is of type
[`FunctionValues`](@ref Ferrite.FunctionValues). Functions relating to geometric properties and
quadrature rules (e.g. `getdetJdV` and `spatial_coordinate`) are still called directly on `cv`.
No performance penalty occurs when using two equal function interpolations compared to a
single function interpolation as their `FunctionValues` are aliased.

Constructing a `CellValues` for three fields, 2nd order interpolation for displacements, `u`,
and 1st order interpolations for the pressure, `p`, and temperature, `T`:
```
qr = QuadratureRule{RefQuadrilateral}(2)
ip_geo = Lagrange{RefQuadrilateral, 1}() # Optional
ipu = Lagrange{RefQuadrilateral, 2}()^2
ipp = Lagrange{RefQuadrilateral, 1}()
ipT = Lagrange{RefQuadrilateral, 1}()
cv = CellValues(qr, (u = ipu, p = ipp, T = ipT), ip_geo)
```
After calling [`reinit!`](@ref) on `cv`, it can be used as, e.g.
```
dΩ = getdetJdV(cv, q_point)
Nu = shape_value(cv.u, q_point, base_function_nr)
∇Np = shape_gradient(cv.p, q_point, base_function_nr)
```
"""
CellValues

function default_geometric_interpolation(::Interpolation{shape}) where {dim, shape <: AbstractRefShape{dim}}
    return VectorizedInterpolation{dim}(Lagrange{shape, 1}())
end
default_geometric_interpolation(ip_funs::NamedTuple) = default_geometric_interpolation(first(ip_funs))

struct CellValues{NT, FVT, GM, QR, detT} <: AbstractCellValues
    fun_values_nt::NT # FunctionValues collected in a NamedTuple aliasing into fun_values, or Nothing (single-field mode)
    fun_values::FVT   # FunctionValues collected in a tuple (each unique)
    geo_mapping::GM   # GeometryMapping
    qr::QR            # QuadratureRule
    # CellValues are only functional for `detT <: AbstractVector{<:Number}`.
    # However, e.g. for GPU support, we allow AbstractMatrix{<:Number} to be passed, allowing this type to be used as a struct-of-arrays (SoA) type.
    # See `soa_utils.jl` for the SoA transformation infrastructure.
    detJdV::detT      # AbstractVector{<:Number} or Nothing
end

const SingleFieldCellValues = CellValues{Nothing}

"""
    MultiFieldCellValues

Alias for a [`CellValues`](@ref) constructed with a `NamedTuple` of interpolations,
e.g. `CellValues(qr, (u = ipu, p = ipp))`.
"""
const MultiFieldCellValues = CellValues{<:NamedTuple}

function CellValues(
        ::Type{T}, qr::QuadratureRule, ip_fun::Interpolation, ip_geo::VectorizedInterpolation,
        ::ValuesUpdateFlags{FunDiffOrder, GeoDiffOrder, DetJdV}
    ) where {T, FunDiffOrder, GeoDiffOrder, DetJdV}

    geo_mapping = GeometryMapping{GeoDiffOrder}(T, ip_geo.ip, qr)
    fun_values = (FunctionValues{FunDiffOrder}(T, ip_fun, qr, ip_geo),)
    detJdV = DetJdV ? fill(T(NaN), length(getweights(qr))) : nothing
    return CellValues(nothing, fun_values, geo_mapping, qr, detJdV)
end

function CellValues(
        ::Type{T}, qr::QuadratureRule, ip_funs::NamedTuple, ip_geo::VectorizedInterpolation,
        ::ValuesUpdateFlags{FunDiffOrder, GeoDiffOrder, DetJdV}
    ) where {T, FunDiffOrder, GeoDiffOrder, DetJdV}

    geo_mapping = GeometryMapping{GeoDiffOrder}(T, ip_geo.ip, qr)
    unique_ips = unique(values(ip_funs)) # Not type-stable, but ok for advanced users this should be constructed outside...
    fun_values = tuple((FunctionValues{FunDiffOrder}(T, ip_fun, qr, ip_geo) for ip_fun in unique_ips)...)
    fun_values_nt = NamedTuple((key => fun_values[findfirst(unique_ip -> ip == unique_ip, unique_ips)] for (key, ip) in pairs(ip_funs)))
    detJdV = DetJdV ? fill(T(NaN), length(getweights(qr))) : nothing
    return CellValues(fun_values_nt, fun_values, geo_mapping, qr, detJdV)
end

CellValues(qr::QuadratureRule, ip::Union{Interpolation, NamedTuple}, args...; kwargs...) = CellValues(Float64, qr, ip, args...; kwargs...)
function CellValues(::Type{T}, qr, ip::Union{Interpolation, NamedTuple}, ip_geo::ScalarInterpolation; kwargs...) where {T}
    return CellValues(T, qr, ip, VectorizedInterpolation(ip_geo); kwargs...)
end
function CellValues(::Type{T}, qr::QuadratureRule, ip::Union{Interpolation, NamedTuple}, ip_geo::VectorizedInterpolation = default_geometric_interpolation(ip); kwargs...) where {T}
    return CellValues(T, qr, ip, ip_geo, ValuesUpdateFlags(ip; kwargs...))
end

# Compat constructors for the MultiFieldCellValues alias
(::Type{MultiFieldCellValues})(qr::QuadratureRule, ip_funs::NamedTuple, args...; kwargs...) = CellValues(qr, ip_funs, args...; kwargs...)
(::Type{MultiFieldCellValues})(::Type{T}, qr::QuadratureRule, ip_funs::NamedTuple, args...; kwargs...) where {T} = CellValues(T, qr, ip_funs, args...; kwargs...)

# Given the old and new (e.g. copied) unique fun_values tuples, rebuild the NamedTuple
# such that keys with aliased values remain aliased in the new tuple.
_rebuild_fun_values_nt(::Nothing, ::Tuple, ::Tuple) = nothing
function _rebuild_fun_values_nt(nt::NamedTuple, old_fun_values::Tuple, new_fun_values::Tuple)
    return map(named_fv -> new_fun_values[findfirst(fv -> fv === named_fv, old_fun_values)], nt)
end

function Base.copy(cv::CV) where {CV <: CellValues}
    old_fun_values = get_fun_values(cv)
    fun_values = map(copy, old_fun_values)
    fun_values_nt = _rebuild_fun_values_nt(getfield(cv, :fun_values_nt), old_fun_values, fun_values)
    # Construct via the concrete type CV for an inferred return type (the NamedTuple rebuild is not)
    return CV(fun_values_nt, fun_values, copy(get_geo_mapping(cv)), copy(get_quadrature_rule(cv)), _copy_or_nothing(getdetJdVs(cv)))
end

# In multi-field mode, properties expose (only) the named FunctionValues, e.g. `cv.u`.
# Single-field mode keeps the default `getproperty` (plain field access).
@inline Base.getproperty(cv::MultiFieldCellValues, key::Symbol) = getproperty(getfield(cv, :fun_values_nt), key)
Base.propertynames(cv::MultiFieldCellValues) = propertynames(getfield(cv, :fun_values_nt))

# Access geometry values
get_geo_mapping(cv::CellValues) = getfield(cv, :geo_mapping)

getdetJdVs(cv::CellValues) = getfield(cv, :detJdV)

# Access quadrature rule values
get_quadrature_rule(cv::CellValues) = getfield(cv, :qr)

# Accessors for function values
get_fun_values(cv::CellValues) = getfield(cv, :fun_values)
@inline _single_fun_values(cv::SingleFieldCellValues) = getfield(cv, :fun_values)[1]

getnbasefunctions(cv::SingleFieldCellValues) = getnbasefunctions(_single_fun_values(cv))
function_interpolation(cv::SingleFieldCellValues) = function_interpolation(_single_fun_values(cv))
function_difforder(cv::SingleFieldCellValues) = function_difforder(_single_fun_values(cv))
shape_value_type(cv::SingleFieldCellValues) = shape_value_type(_single_fun_values(cv))
shape_gradient_type(cv::SingleFieldCellValues) = shape_gradient_type(_single_fun_values(cv))
shape_hessian_type(cv::SingleFieldCellValues) = shape_hessian_type(_single_fun_values(cv))

@propagate_inbounds shape_value(cv::SingleFieldCellValues, q_point::Int, i::Int) = shape_value(_single_fun_values(cv), q_point, i)
@propagate_inbounds shape_gradient(cv::SingleFieldCellValues, q_point::Int, i::Int) = shape_gradient(_single_fun_values(cv), q_point, i)
@propagate_inbounds shape_hessian(cv::SingleFieldCellValues, q_point::Int, i::Int) = shape_hessian(_single_fun_values(cv), q_point, i)

@inline function _update_detJdV!(detJvec::AbstractVector, q_point::Int, w, mapping)
    detJ = calculate_detJ(getjacobian(mapping))
    detJ > 0.0 || throw_detJ_not_pos(detJ)
    @inbounds detJvec[q_point] = detJ * w
    return
end
@inline _update_detJdV!(::Nothing, q_point, w, mapping) = nothing

@inline function reinit!(cv::AbstractCellValues, x::AbstractVector)
    return reinit!(cv, nothing, x)
end

@inline function reinit_needs_cell(cv::CellValues)
    return any(map(fv -> !isa(mapping_type(fv), IdentityMapping), get_fun_values(cv)))
end

function check_reinit_sdim_consistency(cv::CellValues, ::AbstractVector{VT}) where {VT}
    return map(fv -> check_reinit_sdim_consistency(:CellValues, shape_gradient_type(fv), VT), get_fun_values(cv))
end

function reinit!(cv::AbstractCellValues, cell::Union{AbstractCell, Nothing}, x::AbstractVector{<:Vec})
    geo_mapping = get_geo_mapping(cv)
    fun_values = get_fun_values(cv)
    n_geom_basefuncs = getngeobasefunctions(geo_mapping)

    check_reinit_sdim_consistency(cv, x)
    if cell === nothing && reinit_needs_cell(cv)
        throw(ArgumentError("The cell::AbstractCell input is required to reinit! non-identity function mappings"))
    end
    if !checkbounds(Bool, x, 1:n_geom_basefuncs) || length(x) != n_geom_basefuncs
        throw_incompatible_coord_length(length(x), n_geom_basefuncs)
    end
    @inbounds for (q_point, w) in enumerate(getweights(get_quadrature_rule(cv)))
        mapping = calculate_mapping(geo_mapping, q_point, x)
        _update_detJdV!(getdetJdVs(cv), q_point, w, mapping)
        apply_mapping!(fun_values, q_point, mapping, cell)
    end
    return nothing
end

# Need to manually unroll applying to each `fun_values` for maximum performance, equivalent code:
# `foreach(fv -> apply_mapping!(fv, q_point, mapping, cell), fun_values))`
@generated function apply_mapping!(fun_values::Tuple{Vararg{FunctionValues, N}}, q_point, mapping, cell) where {N}
    expr = Expr(:block)
    for i in 1:N
        push!(expr.args, :(apply_mapping!(fun_values[$i], q_point, mapping, cell)))
    end
    return quote
        $(Expr(:meta, :inline))
        @inbounds return $expr
    end
end

function Base.show(io::IO, d::MIME"text/plain", cv::SingleFieldCellValues)
    ip_geo = geometric_interpolation(cv)
    ip_fun = function_interpolation(cv)
    rdim = getrefdim(ip_geo)
    vdim = isa(shape_value(cv, 1, 1), Vec) ? length(shape_value(cv, 1, 1)) : 0
    GradT = shape_gradient_type(cv)
    sdim = GradT === nothing ? nothing : sdim_from_gradtype(GradT)
    vstr = vdim == 0 ? "scalar" : "vdim=$vdim"
    print(io, "CellValues(", vstr, ", rdim=$rdim, and sdim=$sdim): ")
    print(io, getnquadpoints(cv), " quadrature points")
    print(io, "\n Function interpolation: "); show(io, d, ip_fun)
    print(io, "\nGeometric interpolation: ")
    sdim === nothing ? show(io, d, ip_geo) : show(io, d, ip_geo^sdim)
    return
end

function Base.show(io::IO, d::MIME"text/plain", cv::MultiFieldCellValues)
    ip_geo = geometric_interpolation(cv)
    fv1 = first(get_fun_values(cv))
    GradT = shape_gradient_type(fv1)
    sdim = GradT === nothing ? nothing : sdim_from_gradtype(GradT)
    print(io, "CellValues with ", getnquadpoints(cv), " quadrature points")
    print(io, "\nGeometric interpolation: ")
    sdim === nothing ? show(io, d, ip_geo) : show(io, d, ip_geo^sdim)
    print(io, "\nFunction interpolations")
    for key in keys(getfield(cv, :fun_values_nt))
        ip = function_interpolation(getfield(cv, :fun_values_nt)[key])
        print(io, "\n  ", key, ": "); show(io, d, ip)
    end
    return
end

# Error messages for functions that should be called on individual `FunctionValues` to help users
function getnbasefunctions(cv::MultiFieldCellValues)
    k = first(propertynames(cv)) # Pick the first function values to use in example
    throw(ArgumentError("getnbasefunctions isn't applicable to a multi-field CellValues. Use on `FunctionValues` for the specific field, e.g. getnbasefunctions(cv.$k)"))
end

for f in (:shape_value, :shape_gradient, :shape_symmetric_gradient, :shape_divergence)
    @eval function $f(cv::MultiFieldCellValues, ::Int, ::Int)
        k = first(propertynames(cv)) # Pick the first function values to use in example
        fun = $f                       # Make the function name available to use in the error message
        throw(ArgumentError("$fun isn't applicable to a multi-field CellValues. Use on `FunctionValues` for the specific field, e.g. $fun(cv.$k, q_point, shapenr)"))
    end
end
for f in (:function_value, :function_gradient, :function_symmetric_gradient, :function_divergence)
    @eval function $f(cv::MultiFieldCellValues, ::Int, ::AbstractVector, args...)
        k = first(propertynames(cv)) # Pick the first function values to use in example
        fun = $f                       # Make the function name available to use in the error message
        throw(ArgumentError("$fun isn't applicable to a multi-field CellValues. Use on `FunctionValues` for the specific field, e.g. $fun(cv.$k, q_point, ae, [dofrange])"))
    end
end
