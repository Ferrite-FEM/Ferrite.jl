# Notes
# Collect multiple function values inside one object, as the geometry updates will be the same.
# These FunctionScalarValues/FunctionVectorValues can also be used inside FaceValues
# In addition to the potential speed improvements, this structure has the following user side improvements
# - For each SubDofHandler, only one cellvalues object 
# - For the loop, only one quadrature point (nothing preventing separate quad-rules if required by using separate cellvalues)
# - Consistency between each CellValues object ensured automatically (i.e. no strange bugs from differences in e.g. quadrules)
# - Each FunctionValues object accessible via a symbol (stored in NamedTuple) which works statically 


struct GeometryValues{dim,T<:Real,RefShape<:AbstractRefShape}
    M::Matrix{T}
    dMdξ::Matrix{Vec{dim,T}}
    ip::Interpolation{dim,RefShape}
end
function GeometryValues(cv::Union{CellVectorValues,CellScalarValues})
    return GeometryValues(cv.M, cv.dMdξ, cv.geo_interp)
end

getngeobasefunctions(geovals::GeometryValues) = size(geovals.M, 1)
@propagate_inbounds geometric_value(geovals::GeometryValues, q_point::Int, base_func::Int) = geovals.M[base_func, q_point]

abstract type FunctionValues{dim,T,RefShape} <: Values{dim,T,RefShape} end

struct FunctionScalarValues{dim,T<:Real,RefShape<:AbstractRefShape} <: FunctionValues{dim,T,RefShape}
    N::Matrix{T} 
    dNdx::Matrix{Vec{dim,T}}
    dNdξ::Matrix{Vec{dim,T}}
    ip::Interpolation{dim,RefShape}
end
FieldTrait(::Type{<:FunctionScalarValues}) = ScalarValued()

struct FunctionVectorValues{dim,T<:Real,RefShape<:AbstractRefShape,M} <: FunctionValues{dim,T,RefShape}
    N::Matrix{Vec{dim,T}} # For greater generality, I think Nξ (constant) and Nx are needed for non-identity mappings (but only vector values)
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
    return FunctionScalarValues(cv.N, cv.dNdx, cv.dNdξ, cv.func_interp)
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
    detJdV::Vector{T}
    qr::QuadratureRule{dim,RefShape,T}
end
function CellMultiValues(;cvs...)
    # cvs::Pairs{Symbol, CellValues, Tuple, NamedTuple}, cf. foo(;kwargs...) = kwargs
    @assert allequal(typeof(cv.qr) for (_, cv) in cvs)
    @assert allequal(length(getweights(cv.qr)) for (_, cv) in cvs)
    cv1 = first(values(cvs))
    geo_values = GeometryValues(cv1)
    fun_values = NamedTuple(key=>create_function_values(cv) for (key, cv) in cvs)
    return CellMultiValues(geo_values, fun_values, cv1.detJdV, cv1.qr)
end

# Quadrature
# getnquadpoints(::QuadratureRule) would be nice...
getnquadpoints(cv::CellMultiValues) = length(cv.qr.weights)

# Geometric functions
getngeobasefunctions(cv::CellMultiValues) = getngeobasefunctions(cv.geo_values)
getdetJdV(cv::CellMultiValues, q_point::Int) = cv.detJdV[q_point]
geometric_value(cv::CellMultiValues, q_point::Int, base_func::Int) = geometric_value(cv.geo_values, q_point, base_func)

# FunctionValues functions: call like with CellValues, but foo(cv[:name], args...)
Base.getindex(cmv::CellMultiValues, key::Symbol) = getindex(cmv.fun_values, key)

# This function can be specialized for different mappings, 
# see https://defelement.com/ciarlet.html ("Mapping finite elements")
function map_functions!(funvals::FunctionValues{dim}, i::Int, detJ, J, Jinv::Tensor{2,dim}) where dim
    @inbounds for j in 1:getnbasefunctions(funvals)
        funvals.dNdx[j, i] = funvals.dNdξ[j, i] ⋅ Jinv
    end
    return nothing
end

# Note that this function is "unsafe", as it applies inbounds. Checks in reinit!
function calculate_mapping(geo_values::GeometryValues{dim}, q_point, x) where dim
    fecv_J = zero(Tensor{2,dim})
    @inbounds for j in 1:getngeobasefunctions(geo_values)
        fecv_J += x[j] ⊗ geo_values.dMdξ[j, q_point]
    end
    detJ = det(fecv_J)
    detJ > 0.0 || throw_detJ_not_pos(detJ)
    return detJ, fecv_J, inv(fecv_J)
end

function reinit!(cv::CellMultiValues{dim}, x::AbstractVector{Vec{dim,T}}) where {dim,T}
    checkbounds(Bool, x, 1:getngeobasefunctions(cv)) || throw_incompatible_coord_length(length(x), getngeobasefunctions(cv))
    @inbounds for (q_point, w) in pairs(getweights(cv.qr))
        detJ, J, Jinv = calculate_mapping(cv.geo_values, q_point, x)
        cv.detJdV[q_point] = detJ * w # Do it here instead to avoid making calculate_mapping mutating. 
        
        # `map` required for performance, `foreach` allocates due to method lookup. 
        # `values(cv.fun_values)` returns a tuple of the content.
        # This ensures that `map` specializes for number of elements, see how Base/tuple.jl
        map(funvals->map_functions!(funvals, q_point, detJ, J, Jinv), values(cv.fun_values)) 
    end
    return nothing
end

function Base.show(io::IO, ::MIME"text/plain", fe_v::CellMultiValues)
    print(io, "$(typeof(fe_v))")
end