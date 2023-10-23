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

struct CellValues{FV, GM, QR, detT<:AbstractVector} <: AbstractCellValues
    fun_values::FV # FunctionValues
    geo_mapping::GM # GeometryMapping
    qr::QR         # QuadratureRule
    detJdV::detT   # AbstractVector{<:Number}
end
function CellValues(::Type{T}, qr::QuadratureRule, ip_fun::Interpolation, ip_geo::VectorizedInterpolation) where T 
    geo_mapping = GeometryMapping(T, ip_geo.ip, qr, RequiresHessian(ip_fun, ip_geo))
    fun_values = FunctionValues(T, ip_fun, qr, ip_geo)
    detJdV = fill(T(NaN), length(getweights(qr)))
    return CellValues(fun_values, geo_mapping, qr, detJdV)
end

CellValues(qr::QuadratureRule, ip::Interpolation, args...) = CellValues(Float64, qr, ip, args...)
function CellValues(::Type{T}, qr, ip::Interpolation, ip_geo::ScalarInterpolation=default_geometric_interpolation(ip)) where T
    return CellValues(T, qr, ip, VectorizedInterpolation(ip_geo))
end

function Base.copy(cv::CellValues)
    return CellValues(copy(cv.fun_values), copy(cv.geo_mapping), copy(cv.qr), copy(cv.detJdV))
end

"""
    precompute_values!(cv::CellValues)

Precompute all values for the current quadrature rule in `cv`. This method allows you to modify
the quadrature positions, and then update all relevant parts of `cv` accordingly. 
Used by `PointValues`.
"""
function precompute_values!(cv::CellValues)
    precompute_values!(cv.fun_values, cv.qr)
    precompute_values!(cv.geo_mapping, cv.qr)
end

# Access geometry values
@propagate_inbounds getngeobasefunctions(cv::CellValues) = getngeobasefunctions(cv.geo_mapping)
@propagate_inbounds geometric_value(cv::CellValues, args...) = geometric_value(cv.geo_mapping, args...)
get_geometric_interpolation(cv::CellValues) = get_geometric_interpolation(cv.geo_mapping)

getdetJdV(cv::CellValues, q_point::Int) = cv.detJdV[q_point]

# Accessors for function values 
getnbasefunctions(cv::CellValues) = getnbasefunctions(cv.fun_values)
get_function_interpolation(cv::CellValues) = get_function_interpolation(cv.fun_values)
shape_value_type(cv::CellValues) = shape_value_type(cv.fun_values)
shape_gradient_type(cv::CellValues) = shape_gradient_type(cv.fun_values)

for op = (:shape_value, :shape_gradient, :shape_symmetric_gradient)
    eval(quote
        @propagate_inbounds $op(cv::CellValues, i::Int, q_point::Int) = $op(cv.fun_values, i, q_point)
    end)
end

# Access quadrature rule values 
getnquadpoints(cv::CellValues) = getnquadpoints(cv.qr)

function reinit!(cv::CellValues, x::AbstractVector{<:Vec}, cell=nothing)
    geo_mapping = cv.geo_mapping
    fun_values = cv.fun_values
    n_geom_basefuncs = getngeobasefunctions(geo_mapping)
    
    check_reinit_sdim_consistency(:CellValues, shape_gradient_type(cv), eltype(x))
    if cell === nothing && !isa(get_mapping_type(fun_values), IdentityMapping)
        throw(ArgumentError("The cell::AbstractCell input is required to reinit! non-identity function mappings"))
    end
    if !checkbounds(Bool, x, 1:n_geom_basefuncs) || length(x)!=n_geom_basefuncs
        throw_incompatible_coord_length(length(x), n_geom_basefuncs)
    end
    @inbounds for (q_point, w) in enumerate(getweights(cv.qr))
        mapping = calculate_mapping(geo_mapping, q_point, x)
        detJ = calculate_detJ(getjacobian(mapping))
        detJ > 0.0 || throw_detJ_not_pos(detJ)
        @inbounds cv.detJdV[q_point] = detJ*w
        apply_mapping!(fun_values, q_point, mapping, cell)
    end
    return nothing
end

function Base.show(io::IO, d::MIME"text/plain", cv::CellValues)
    ip_geo = get_geometric_interpolation(cv)
    ip_fun = get_function_interpolation(cv)
    rdim = getdim(ip_geo)
    vdim = isa(shape_value(cv, 1, 1), Vec) ? length(shape_value(cv, 1, 1)) : 0
    sdim = length(shape_gradient(cv, 1, 1)) ÷ length(shape_value(cv, 1, 1))
    vstr = vdim==0 ? "scalar" : "vdim=$vdim"
    print(io, "CellValues(", vstr, ", rdim=$rdim, and sdim=$sdim): ")
    print(io, getnquadpoints(cv), " quadrature points")
    print(io, "\n Function interpolation: "); show(io, d, ip_fun)
    print(io, "\nGeometric interpolation: "); show(io, d, ip_geo^sdim)
end
