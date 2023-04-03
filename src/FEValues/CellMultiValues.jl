# Idea
# Collect multiple function values inside one object, as the geometry updates will be the same.
# These FunctionScalarValues/FunctionVectorValues can also be used for FaceValues, but requires 
# face values to contain vectors of these for each face (which makes sense especially for e.g. wedge elements)
# In addition to the potential speed improvements, this structure has the following user side improvements
# - For each SubDofHandler, only one cellvalues object 
# - For the loop, only one quadrature point (nothing preventing separate quad-rules if required by using separate cellvalues)
# - Consistency between each CellValues object ensured automatically (i.e. no strange bugs from differences in e.g. quadrules)
# - Each FunctionValues object accessible via a symbol (stored in NamedTuple).

struct GeometryValues{dim,T<:Real,RefShape<:AbstractRefShape}
    detJdV::Vector{T}
    M::Matrix{T}
    dMdξ::Matrix{Vec{dim,T}}
    ip::Interpolation{dim,RefShape}
end
function GeometryValues(cv::Union{CellVectorValues,CellScalarValues})
    return GeometryValues(cv.detJdV, cv.M, cv.dMdξ, cv.geo_interp)
end

getngeobasefunctions(geovals::GeometryValues) = size(geovals.M, 1)

abstract type FunctionValues{dim,T,RefShape} <: Values{dim,T,RefShape} end

struct FunctionScalarValues{dim,T<:Real,RefShape<:AbstractRefShape} <: FunctionValues{dim,T,RefShape}
    N::Matrix{T}
    dNdx::Matrix{Vec{dim,T}}
    dNdξ::Matrix{Vec{dim,T}}
    ip::Interpolation{dim,RefShape}
end
FieldTrait(::Type{<:FunctionScalarValues}) = ScalarValued()

struct FunctionVectorValues{dim,T<:Real,RefShape<:AbstractRefShape,M} <: FunctionValues{dim,T,RefShape}
    N::Matrix{Vec{dim,T}}
    dNdx::Matrix{Tensor{2,dim,T,M}}
    dNdξ::Matrix{Tensor{2,dim,T,M}}
    ip::Interpolation{dim,RefShape}
end
FieldTrait(::Type{<:FunctionVectorValues}) = VectorValued()

# Temporary solution?
function create_function_values(cv::CellVectorValues)
    return FunctionVectorValues(cv.N, cv.dNdx, cv.dNdξ, cv.func_interp)
end

function create_function_values(cv::CellScalarValues)
    return FunctionScalarValues(cv.N, cv.dNdx, cv.dNdξ, cv.ip)
end

getnbasefunctions(funvals::FunctionValues) = size(funvals.N, 1)
@propagate_inbounds shape_value(funvals::FunctionValues, q_point::Int, base_func::Int) = funvals.N[base_func, q_point]
@propagate_inbounds shape_gradient(funvals::FunctionValues, q_point::Int, base_func::Int) = funvals.dNdx[base_func, q_point]
@propagate_inbounds shape_symmetric_gradient(funvals::FunctionVectorValues, q_point::Int, base_func::Int) = symmetric(shape_gradient(funvals, q_point, base_func))
@propagate_inbounds shape_divergence(funvals::FunctionScalarValues, q_point::Int, base_func::Int) = sum(funvals.dNdx[base_func, q_point])
@propagate_inbounds shape_divergence(funvals::FunctionVectorValues, q_point::Int, base_func::Int) = tr(funvals.dNdx[base_func, q_point])
@propagate_inbounds shape_curl(funvals::FunctionVectorValues, q_point, base_func) = curl_from_gradient(shape_gradient(funvals, q_point, base_func))


struct CellMultiValues{dim,T,RefShape,FVS<:NamedTuple} <: CellValues{dim,T,RefShape}
    geo_values::GeometryValues{dim,T,RefShape}
    fun_values::FVS
    qr::QuadratureRule{dim,RefShape,T}
end
function CellMultiValues(;cvs...)
    # cvs::Pairs{Symbol, CellValues, Tuple, NamedTuple}, cf. foo(;kwargs...) = kwargs
    @assert allequal(cv.qr for (_, cv) in cvs)
    qr = first(values(cvs)).qr
    geo_values = GeometryValues(first(cvs))
    fun_values = NamedTuple(key=>create_function_values(cv) for (key, cv) in cvs)
    return CellMultiValues(geo_values, fun_values, qr)
end

# Quadrature
# getnquadpoints(::QuadratureRule) would be nice...
getnquadpoints(cv::CellMultiValues) = length(cv.qr.weights)

# Geometric functions
getngeobasefunctions(cv::CellMultiValues) = getngeobasefunctions(cv.geo_values)
getdetJdV(cv::CellMultiValues, args...) = getdetJdV(cv.geo_values, args...)
geometric_value(cv::CellMultiValues, args...) = geometric_value(cv.geo_values, args...)

# FunctionValues functions: call like with CellValues, but foo(cv[:name], args...)

function _update_dNdx!(fv::FunctionValues{dim}, i::Int, Jinv::Tensor{2,dim}) where dim
    @inbounds for j in 1:getnbasefunctions(fv)
        cv.dNdx[j, i] = cv.dNdξ[j, i] ⋅ Jinv
    end
end

function reinit!(cv::CellMultiValues{dim}, x::AbstractVector{Vec{dim,T}}) where {dim,T}
    n_geom_basefuncs = getngeobasefunctions(cv)
    length(x) == n_geom_basefuncs || throw_incompatible_coord_length(length(x), n_geom_basefuncs)

    @inbounds for (i, w) in cv.qr.weights
        fecv_J = zero(Tensor{2,dim})
        for j in 1:n_geom_basefuncs
            fecv_J += x[j] ⊗ cv.dMdξ[j, i]
        end
        detJ = det(fecv_J)
        detJ > 0.0 || throw_detJ_not_pos(detJ)
        cv.geo_values.detJdV[i] = detJ * w
        Jinv = inv(fecv_J)
        for (_, funvals) in cv.fun_values
            _update_dNdx!(funvals, i, Jinv)
        end
    end
end