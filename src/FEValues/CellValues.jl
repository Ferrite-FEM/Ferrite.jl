"""
    CellValues([::Type{T},] quad_rule::QuadratureRule, func_interpol::Interpolation, [geom_interpol::Interpolation])

A `CellValues` object facilitates the process of evaluating values of shape functions, gradients of shape functions,
values of nodal functions, gradients and divergences of nodal functions etc. in the finite element cell.

**Arguments:**
* `T`: an optional argument (default to `Float64`) to determine the type the internal data is stored as.
* `quad_rule`: an instance of a [`QuadratureRule`](@ref)
* `func_interpol`: an instance of an [`Interpolation`](@ref) used to interpolate the approximated function
* `geom_interpol`: an optional instance of a [`Interpolation`](@ref) which is used to interpolate the geometry.
  By default linear Lagrange interpolation is used. For embedded elements the geometric interpolations should
  be vectorized to the spatial dimension.

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
"""
CellValues

function default_geometric_interpolation(::Interpolation{shape}) where {dim, shape <: AbstractRefShape{dim}}
    return VectorizedInterpolation{dim}(Lagrange{shape, 1}())
end

struct CellValues{FV, GM, QR, detT} <: AbstractCellValues
    fun_values::FV # FunctionValues
    geo_mapping::GM # GeometryMapping
    qr::QR         # QuadratureRule
    detJdV::detT   # AbstractVector{<:Number} or Nothing
end
function CellValues(::Type{T}, qr::QuadratureRule, ip_fun::Interpolation, ip_geo::VectorizedInterpolation; 
        update_gradients::Union{Bool,Nothing} = nothing, update_detJdV::Union{Bool,Nothing} = nothing) where T 
    
    _update_detJdV = update_detJdV === nothing ? true : update_detJdV
    FunDiffOrder = update_gradients === nothing ? 1 : convert(Int, update_gradients) # Logic must change when supporting update_hessian kwargs
    GeoDiffOrder = max(required_geo_diff_order(mapping_type(ip_fun), FunDiffOrder), _update_detJdV)
    geo_mapping = GeometryMapping{GeoDiffOrder}(T, ip_geo.ip, qr)
    fun_values = FunctionValues{FunDiffOrder}(T, ip_fun, qr, ip_geo)
    detJdV = _update_detJdV ? fill(T(NaN), length(getweights(qr))) : nothing
    return CellValues(fun_values, geo_mapping, qr, detJdV)
end

CellValues(qr::QuadratureRule, ip::Interpolation, args...; kwargs...) = CellValues(Float64, qr, ip, args...; kwargs...)
function CellValues(::Type{T}, qr, ip::Interpolation, ip_geo::ScalarInterpolation=default_geometric_interpolation(ip); kwargs...) where T
    return CellValues(T, qr, ip, VectorizedInterpolation(ip_geo); kwargs...)
end

function Base.copy(cv::CellValues)
    return CellValues(copy(cv.fun_values), copy(cv.geo_mapping), copy(cv.qr), _copy_or_nothing(cv.detJdV))
end

# Access geometry values
@propagate_inbounds getngeobasefunctions(cv::CellValues) = getngeobasefunctions(cv.geo_mapping)
@propagate_inbounds geometric_value(cv::CellValues, args...) = geometric_value(cv.geo_mapping, args...)
geometric_interpolation(cv::CellValues) = geometric_interpolation(cv.geo_mapping)

getdetJdV(cv::CellValues, q_point::Int) = cv.detJdV[q_point]
getdetJdV(::CellValues{<:Any, <:Any, <:Any, Nothing}, ::Int) = throw(ArgumentError("detJdV is not saved in CellValues"))

# Accessors for function values 
getnbasefunctions(cv::CellValues) = getnbasefunctions(cv.fun_values)
function_interpolation(cv::CellValues) = function_interpolation(cv.fun_values)
function_difforder(cv::CellValues) = function_difforder(cv.fun_values)
shape_value_type(cv::CellValues) = shape_value_type(cv.fun_values)
shape_gradient_type(cv::CellValues) = shape_gradient_type(cv.fun_values)

@propagate_inbounds shape_value(cv::CellValues, q_point::Int, i::Int) = shape_value(cv.fun_values, q_point, i)
@propagate_inbounds shape_gradient(cv::CellValues, q_point::Int, i::Int) = shape_gradient(cv.fun_values, q_point, i)
@propagate_inbounds shape_symmetric_gradient(cv::CellValues, q_point::Int, i::Int) = shape_symmetric_gradient(cv.fun_values, q_point, i)

# Access quadrature rule values 
getnquadpoints(cv::CellValues) = getnquadpoints(cv.qr)

@inline function _update_detJdV!(detJvec::AbstractVector, q_point::Int, w, mapping)
    detJ = calculate_detJ(getjacobian(mapping))
    detJ > 0.0 || throw_detJ_not_pos(detJ)
    @inbounds detJvec[q_point] = detJ * w
end
@inline _update_detJdV!(::Nothing, q_point, w, mapping) = nothing

@inline function reinit!(cv::CellValues, x::AbstractVector)
    return reinit!(cv, nothing, x)
end

function reinit!(cv::CellValues, cell::Union{AbstractCell, Nothing}, x::AbstractVector{<:Vec})
    geo_mapping = cv.geo_mapping
    fun_values = cv.fun_values
    n_geom_basefuncs = getngeobasefunctions(geo_mapping)
    
    check_reinit_sdim_consistency(:CellValues, shape_gradient_type(cv), eltype(x))
    if cell === nothing && !isa(mapping_type(fun_values), IdentityMapping)
        throw(ArgumentError("The cell::AbstractCell input is required to reinit! non-identity function mappings"))
    end
    if !checkbounds(Bool, x, 1:n_geom_basefuncs) || length(x) != n_geom_basefuncs
        throw_incompatible_coord_length(length(x), n_geom_basefuncs)
    end
    @inbounds for (q_point, w) in enumerate(getweights(cv.qr))
        mapping = calculate_mapping(geo_mapping, q_point, x)
        _update_detJdV!(cv.detJdV, q_point, w, mapping)
        apply_mapping!(fun_values, q_point, mapping, cell)
    end
    return nothing
end

function Base.show(io::IO, d::MIME"text/plain", cv::CellValues)
    ip_geo = geometric_interpolation(cv)
    ip_fun = function_interpolation(cv)
    rdim = getdim(ip_geo)
    vdim = isa(shape_value(cv, 1, 1), Vec) ? length(shape_value(cv, 1, 1)) : 0
    GradT = shape_gradient_type(cv)
    sdim = GradT === nothing ? nothing : sdim_from_gradtype(GradT)
    vstr = vdim==0 ? "scalar" : "vdim=$vdim"
    print(io, "CellValues(", vstr, ", rdim=$rdim, and sdim=$sdim): ")
    print(io, getnquadpoints(cv), " quadrature points")
    print(io, "\n Function interpolation: "); show(io, d, ip_fun)
    print(io, "\nGeometric interpolation: "); 
    sdim === nothing ? show(io, d, ip_geo) : show(io, d, ip_geo^sdim)
end

"""
    CellMultiValues([::Type{T},] quad_rule::QuadratureRule, func_interpols::NamedTuple, [geom_interpol::Interpolation])

A `cmv::CellMultiValues` is similar to a `CellValues` object, but includes values associated with multiple 
interpolations while sharing the same quadrature points and geometrical interpolation.

In general, functions applicable to a `CellValues` associated with the function interpolation 
in `func_interpols` with `key::Symbol` can be called on `cmv[key]`, as `cmv[key] isa FunctionValues`. 
Other functions relating to geometric properties and quadrature rules are called directly on `cmv`. 

**Arguments:**
* `T`: an optional argument (default to `Float64`) to determine the type the internal data is stored as.
* `quad_rule`: an instance of a [`QuadratureRule`](@ref)
* `func_interpols`: A named tuple with entires of type `Interpolation`, used to interpolate the approximated function identified by the key in `func_interpols`
* `geom_interpol`: an optional instance of a [`Interpolation`](@ref) which is used to interpolate the geometry.
  By default linear Lagrange interpolation is used. For embedded elements the geometric interpolations should
  be vectorized to the spatial dimension.

In general, no performance penalty for using two equal function interpolations compared to a 
single function interpolation should be expected as their `FunctionValues` are aliased.

**Examples**

Constructing a `CellMultiValues` for three fields, 2nd order interpolation for displacements, `u`,
and 1st order interpolations for the pressure, `p`, and temperature, `T`.
```
qr = QuadratureRule{RefQuadrilateral}(2)
ip_geo = Lagrange{RefQuadrilateral,1}() # Optional
ipu = Lagrange{RefQuadrilateral,2}()^2
ipp = Lagrange{RefQuadrilateral,1}()
ipT = Lagrange{RefQuadrilaterla,1}()
cmv = CellMultiValues(qr, (u = ipu, p = ipp, T = ipT), ip_geo)
```
After reinitialization, the `cmv` can be used as, e.g. 
```
dΩ = getdetJdV(cmv, q_point)
Nu = shape_value(cmv[:u], q_point, base_function_nr)
∇Np = shape_gradient(cmv[:p], q_point, base_function_nr)
```

**Common methods for `CellMultiValues`**

Applicable to `cmv` above

  * [`reinit!`](@ref)
  * [`getnquadpoints`](@ref)
  * [`getdetJdV`](@ref)
  * [`spatial_coordinate`](@ref)

**Common methods for `FunctionValues`**

E.g. applicable to `cmv[:u]` above

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
CellMultiValues

struct CellMultiValues{FVS, GM, QR, detT, FVT} <: AbstractCellValues
    fun_values::FVS         # FunctionValues collected in a named tuple (not necessarily unique)
    fun_values_tuple::FVT   # FunctionValues collected in a tuple (each unique)
    geo_mapping::GM         # GeometryMapping
    qr::QR                  # QuadratureRule
    detJdV::detT            # AbstractVector{<:Number} or Nothing
end

function CellMultiValues(::Type{T}, qr::QuadratureRule, ip_funs::NamedTuple, ip_geo::VectorizedInterpolation;
    update_gradients::Bool = true, update_detJdV::Bool = true) where T 

    FunDiffOrder = convert(Int, update_gradients) # Logic must change when supporting update_hessian kwargs
    GeoDiffOrder = max(maximum(ip_fun -> required_geo_diff_order(mapping_type(ip_fun), FunDiffOrder), values(ip_funs)), update_detJdV)
    geo_mapping = GeometryMapping{GeoDiffOrder}(T, ip_geo.ip, qr)
    unique_ips = unique(values(ip_funs))
    fun_values_tuple = tuple((FunctionValues{FunDiffOrder}(T, ip_fun, qr, ip_geo) for ip_fun in unique_ips)...)
    fun_values = NamedTuple((key => fun_values_tuple[findfirst(unique_ip -> ip == unique_ip, unique_ips)] for (key, ip) in pairs(ip_funs)))
    detJdV = update_detJdV ? fill(T(NaN), length(getweights(qr))) : nothing
    return CellMultiValues(fun_values, fun_values_tuple, geo_mapping, qr, detJdV)
end

CellMultiValues(qr::QuadratureRule, ip_funs::NamedTuple, args...; kwargs...) = CellMultiValues(Float64, qr, ip_funs, args...; kwargs...)
function CellMultiValues(::Type{T}, qr, ip_funs::NamedTuple, ip_geo::ScalarInterpolation=default_geometric_interpolation(first(ip_funs)); kwargs...) where T
    return CellMultiValues(T, qr, ip_funs, VectorizedInterpolation(ip_geo); kwargs...)
end

function Base.copy(cv::CMV) where {CMV <: CellMultiValues}
    fun_values_tuple = map(copy, cv.fun_values_tuple)
    fun_values = NamedTuple((key => fun_values_tuple[findfirst(fv -> fv === named_fv, cv.fun_values_tuple)] for (key, named_fv) in pairs(cv.fun_values)))
    return CMV(fun_values, fun_values_tuple, copy(cv.geo_mapping), copy(cv.qr), _copy_or_nothing(cv.detJdV))
end

# Access geometry values
@propagate_inbounds getngeobasefunctions(cv::CellMultiValues) = getngeobasefunctions(cv.geo_mapping)
@propagate_inbounds geometric_value(cv::CellMultiValues, args...) = geometric_value(cv.geo_mapping, args...)
geometric_interpolation(cv::CellMultiValues) = geometric_interpolation(cv.geo_mapping)

function getdetJdV(cv::CellMultiValues, q_point::Int)
    cv.detJdV === nothing && throw(ArgumentError("detJdV calculation was not requested at construction"))
    return cv.detJdV[q_point]
end

# No accessors for function values, just ability to get the stored `FunctionValues` which can be called directly. 
@inline Base.getindex(cv::CellMultiValues, key::Symbol) = cv.fun_values[key]

# Access quadrature rule values 
getnquadpoints(cv::CellMultiValues) = getnquadpoints(cv.qr)

@inline function reinit!(cv::CellMultiValues, x::AbstractVector)
    return reinit!(cv, nothing, x)
end

function reinit!(cv::CellMultiValues, cell::Union{AbstractCell, Nothing}, x::AbstractVector{<:Vec})
    geo_mapping = cv.geo_mapping
    fun_values = cv.fun_values_tuple
    n_geom_basefuncs = getngeobasefunctions(geo_mapping)
    
    map(fv -> check_reinit_sdim_consistency(:CellMultiValues, shape_gradient_type(fv), eltype(x)), fun_values)
    if cell === nothing && !all(map(fv -> isa(mapping_type(fv), IdentityMapping), fun_values))
        throw(ArgumentError("The cell::AbstractCell input is required to reinit! non-identity function mappings"))
    end
    if !checkbounds(Bool, x, 1:n_geom_basefuncs) || length(x) != n_geom_basefuncs
        throw_incompatible_coord_length(length(x), n_geom_basefuncs)
    end
    
    @inbounds for (q_point, w) in enumerate(getweights(cv.qr))
        mapping = calculate_mapping(geo_mapping, q_point, x)
        _update_detJdV!(cv.detJdV, q_point, w, mapping)
        _apply_mappings!(fun_values, q_point, mapping, cell)
    end
    return nothing
end

@inline function _apply_mappings!(fun_values::Tuple, q_point, mapping, cell)
    map(fv -> (@inbounds apply_mapping!(fv, q_point, mapping, cell)), fun_values)
end

# Slightly faster for unknown reason to write out each call, only worth it for a few unique function values. 
@inline function _apply_mappings!(fun_values::Tuple{<:FunctionValues}, q_point, mapping, cell)
    @inbounds apply_mapping!(fun_values[1], q_point, mapping, cell)
end

@inline function _apply_mappings!(fun_values::Tuple{<:FunctionValues, <:FunctionValues}, q_point, mapping, cell)
    @inbounds begin
        apply_mapping!(fun_values[1], q_point, mapping, cell)
        apply_mapping!(fun_values[2], q_point, mapping, cell)
    end
end
