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
    #@assert allequal(typeof(cv.qr) for (_, cv) in cvs)
    #@assert allequal(length(getweights(cv.qr)) for (_, cv) in cvs)
    cv1 = first(values(cvs))
    @assert all(==(typeof(cv1.qr)), typeof(cv.qr) for (_, cv) in cvs)
    @assert all(==(length(getweights(cv1.qr))), length(getweights(cv.qr)) for (_, cv) in cvs)
    
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

# Standard identity mapping (scalar function / lagrange vector)
function apply_mapping!(funvals::FunctionValues{dim}, q_point::Int, Jinv::Tensor{2,dim}) where dim
    @inbounds for j in 1:getnbasefunctions(funvals)
        funvals.dNdx[j, q_point] = funvals.dNdξ[j, q_point] ⋅ Jinv
    end
    return nothing
end

#= TODO: This function could be generalized as follows, but does not make sense before we 
   implement other mappings (i.e. Piola-mappings)
"""
    apply_mapping!(funvals::FunctionValues, q_point::Int, detJ, J, Jinv, geo_values)

Apply the appropriate mapping for `funvals` for quadrature point `q_point`,
given the jacobian `J` (as well as its determinant, `detJ`, and inverse, `Jinv`)
and `geo_values::GeometricValues`. 
See (DefElement)[https://defelement.com/ciarlet.html], "Mapping finite elements"
for an overview of different mappings. 

Note that this function should only be called from `reinit!`. 
There, q_point < getnquadpoints(qr::QuadratureRule) is checked.
During construction of `CellMultiValues`, sizes of buffers are checked to match `qr`.
Hence, it is allowed to use `@inbounds` in this function.
"""
function apply_mapping! end  
=#

# Note that this function is "unsafe", as it applies inbounds. Checks in reinit!
function calculate_mapping(geo_values::GeometryValues{dim}, q_point, x) where dim
    @inbounds fecv_J = x[1] ⊗ geo_values.dMdξ[1, q_point]
    @inbounds for j in 2:getngeobasefunctions(geo_values)
        fecv_J += x[j] ⊗ geo_values.dMdξ[j, q_point]
    end
    detJ = det(fecv_J)
    detJ > 0.0 || throw_detJ_not_pos(detJ)
    return detJ, inv(fecv_J)
end

function reinit!(cv::CellMultiValues{dim}, x::AbstractVector{Vec{dim,T}}) where {dim,T}
    geo_values = cv.geo_values
    checkbounds(Bool, x, 1:getngeobasefunctions(geo_values)) || throw_incompatible_coord_length(length(x), getngeobasefunctions(geo_values))
    @inbounds for (q_point, w) in enumerate(getweights(cv.qr))
        detJ, Jinv = calculate_mapping(geo_values, q_point, x)
        cv.detJdV[q_point] = detJ*w
        # `fun_values::Tuple` makes `map` specialize for number of elements, see Base/tuple.jl vs Base/named_tuple.jl
        map(funvals->apply_mapping!(funvals, q_point, Jinv), values(cv.fun_values)) 
    end
    return nothing
end

function Base.show(io::IO, ::MIME"text/plain", fe_v::CellMultiValues)
    print(io, "$(typeof(fe_v))")
end

# ==== The remaining code is just temporary ==== #
# Temporary and incomplete implementation of CellSingleValues: Uses GeometricValues and FunctionValues internally
# If successful, could be used internally in CellScalarValues and CellVectorValues
struct CellSingleValues{dim,T,RefShape,FV<:FunctionValues{dim,T,RefShape}} <: CellValues{dim,T,RefShape}
    geo_values::GeometryValues{dim,T,RefShape}
    fun_values::FV
    detJdV::Vector{T}
    qr::QuadratureRule{dim,RefShape,T}
end
function CellSingleValues(cv)
    geo_values = GeometryValues(cv)
    fun_values = create_function_values(cv)
    return CellSingleValues(geo_values, fun_values, cv.detJdV, cv.qr)
end

getnquadpoints(cv::CellSingleValues) = length(cv.qr.weights)

getngeobasefunctions(cv::CellSingleValues) = getngeobasefunctions(cv.geo_values)
getdetJdV(cv::CellSingleValues, q_point::Int) = cv.detJdV[q_point]
geometric_value(cv::CellSingleValues, q_point::Int, base_func::Int) = geometric_value(cv.geo_values, q_point, base_func)

function reinit!(cv::CellSingleValues{dim}, x::AbstractVector{Vec{dim,T}}) where {dim,T}
    qr = cv.qr 
    geo_values = cv.geo_values
    fun_values = cv.fun_values
    detJdV = cv.detJdV
    checkbounds(Bool, x, 1:getngeobasefunctions(geo_values)) || throw_incompatible_coord_length(length(x), getngeobasefunctions(geo_values))
    @inbounds for (q_point, w) in enumerate(getweights(qr))
        detJ, Jinv = calculate_mapping(geo_values, q_point, x)
        detJdV[q_point] = detJ*w
        apply_mapping!(fun_values, q_point, Jinv)
    end
    return nothing
end

function Base.show(io::IO, ::MIME"text/plain", fe_v::CellSingleValues)
    print(io, "$(typeof(fe_v))")
end


# Temporary test for seeing the influence of just reusing the geometry calculation
# Currently, the args... approach seems to allocate, 
# hence just implementing specific cases for 1, 2, and 4 values for testing.

# This function is also used by CellValuesGroup later. 
function update_dNdX_and_detJdV!(cv, i, detJ_w, Jinv)
    @inbounds cv.detJdV[i] = detJ_w
    @inbounds for j in 1:getnbasefunctions(cv)
        cv.dNdx[j, i] = cv.dNdξ[j, i] ⋅ Jinv
    end
    return nothing
end

function reinit_multiple!(x, cv::CellValues{dim}) where dim
    n_geom_basefuncs = getngeobasefunctions(cv)
    #n_func_basefuncs = getnbasefunctions(cv)
    length(x) == n_geom_basefuncs || throw_incompatible_coord_length(length(x), n_geom_basefuncs)

    @inbounds for i in 1:length(cv.qr.weights)
        w = cv.qr.weights[i]
        fecv_J = zero(Tensor{2,dim})
        for j in 1:n_geom_basefuncs
            fecv_J += x[j] ⊗ cv.dMdξ[j, i]
        end
        detJ = det(fecv_J)
        detJ > 0.0 || throw_detJ_not_pos(detJ)
        Jinv = inv(fecv_J)
        detJ_w = detJ*w
        update_dNdX_and_detJdV!(cv, i, detJ_w, Jinv)
    end
end

function reinit_multiple!(x, cv::CellValues{dim}, cv2::CellValues{dim}) where dim
    n_geom_basefuncs = getngeobasefunctions(cv)
    #n_func_basefuncs = getnbasefunctions(cv)
    length(x) == n_geom_basefuncs || throw_incompatible_coord_length(length(x), n_geom_basefuncs)

    @inbounds for i in 1:length(cv.qr.weights)
        w = cv.qr.weights[i]
        fecv_J = zero(Tensor{2,dim})
        for j in 1:n_geom_basefuncs
            fecv_J += x[j] ⊗ cv.dMdξ[j, i]
        end
        detJ = det(fecv_J)
        detJ > 0.0 || throw_detJ_not_pos(detJ)
        Jinv = inv(fecv_J)
        detJ_w = detJ*w
        update_dNdX_and_detJdV!(cv, i, detJ_w, Jinv)
        update_dNdX_and_detJdV!(cv2, i, detJ_w, Jinv)
    end
end

function reinit_multiple!(x, cv::CellValues{dim}, cv2::CellValues{dim}, cv3::CellValues{dim}, cv4::CellValues{dim}) where dim
    n_geom_basefuncs = getngeobasefunctions(cv)
    #n_func_basefuncs = getnbasefunctions(cv)
    length(x) == n_geom_basefuncs || throw_incompatible_coord_length(length(x), n_geom_basefuncs)

    @inbounds for i in 1:length(cv.qr.weights)
        w = cv.qr.weights[i]
        fecv_J = zero(Tensor{2,dim})
        for j in 1:n_geom_basefuncs
            fecv_J += x[j] ⊗ cv.dMdξ[j, i]
        end
        detJ = det(fecv_J)
        detJ > 0.0 || throw_detJ_not_pos(detJ)
        Jinv = inv(fecv_J)
        detJ_w = detJ*w
        update_dNdX_and_detJdV!(cv, i, detJ_w, Jinv)
        update_dNdX_and_detJdV!(cv2, i, detJ_w, Jinv)
        update_dNdX_and_detJdV!(cv3, i, detJ_w, Jinv)
        update_dNdX_and_detJdV!(cv4, i, detJ_w, Jinv)
    end
end

#struct CellValuesGroup{dim,T,RefShape,NT<:NamedTuple{Any,CellValues{dim,T,RefShape}}}
struct CellValuesGroup{dim, N, NT<:NamedTuple{<:Any, <:NTuple{N,CellValues{dim}}}}
    cvs::NT
end
function CellValuesGroup(;cvs...)
    # cvs::Pairs{Symbol, CellValues, Tuple, NamedTuple}, cf. foo(;kwargs...) = kwargs
    @assert allequal(typeof(cv.qr) for (_, cv) in cvs)
    @assert allequal(length(getweights(cv.qr)) for (_, cv) in cvs)
    return CellValuesGroup(NamedTuple(cvs))
end

function reinit!(cvs::CellValuesGroup{dim}, x::AbstractVector{Vec{dim,T}}) where {dim,T}
    cvs_tuple = values(cvs.cvs)
    cv = first(cvs_tuple)
    n_geom_basefuncs = getngeobasefunctions(cv)
    #n_func_basefuncs = getnbasefunctions(cv)
    length(x) == n_geom_basefuncs || throw_incompatible_coord_length(length(x), n_geom_basefuncs)

    @inbounds for i in 1:length(cv.qr.weights)
        w = cv.qr.weights[i]
        fecv_J = zero(Tensor{2,dim})
        for j in 1:n_geom_basefuncs
            fecv_J += x[j] ⊗ cv.dMdξ[j, i]
        end
        detJ = det(fecv_J)
        detJ > 0.0 || throw_detJ_not_pos(detJ)
        Jinv = inv(fecv_J)
        detJ_w = detJ*w
        map(cvi -> update_dNdX_and_detJdV!(cvi, i, detJ_w, Jinv), cvs_tuple)
    end
end
