struct GeometryValues{GIP, T, dMdξ_t}
    M::Matrix{T}
    dMdξ::Matrix{dMdξ_t}
    ip::GIP
end
function GeometryValues(cv::CellValues)
    return GeometryValues(cv.M, cv.dMdξ, cv.gip)
end

getngeobasefunctions(geovals::GeometryValues) = size(geovals.M, 1)
@propagate_inbounds geometric_value(geovals::GeometryValues, q_point::Int, base_func::Int) = geovals.M[base_func, q_point]

struct FunctionValues{IP, N_t, dNdx_t, dNdξ_t}
    N::Matrix{N_t} 
    dNdx::Matrix{dNdx_t}
    dNdξ::Matrix{dNdξ_t}
    ip::IP
end
FunctionValues(cv::CellValues) = FunctionValues(cv.N, cv.dNdx, cv.dNdξ, cv.ip)

getnbasefunctions(funvals::FunctionValues) = size(funvals.N, 1)
@propagate_inbounds shape_value(funvals::FunctionValues, q_point::Int, base_func::Int) = funvals.N[base_func, q_point]
@propagate_inbounds shape_gradient(funvals::FunctionValues, q_point::Int, base_func::Int) = funvals.dNdx[base_func, q_point]
@propagate_inbounds shape_symmetric_gradient(funvals::FunctionValues, q_point::Int, base_func::Int) = symmetric(shape_gradient(funvals, q_point, base_func))
@propagate_inbounds shape_divergence(funvals::FunctionValues, q_point::Int, base_func::Int) = divergence_from_gradient(funvals.dNdx[base_func, q_point])
@propagate_inbounds shape_curl(funvals::FunctionValues, q_point, base_func) = curl_from_gradient(shape_gradient(funvals, q_point, base_func))

struct MultiCellValues2{T, FVS, QR, GVS} <: AbstractCellValues
    geo_values::GVS     # GeometryValues
    fun_values::FVS     # NamedTuple(function values)
    detJdV::Vector{T}   # 
    qr::QR
end
function MultiCellValues2(;cvs...)
    # cvs::Pairs{Symbol, CellValues, Tuple, NamedTuple}, cf. foo(;kwargs...) = kwargs
    #@assert allequal(typeof(cv.qr) for (_, cv) in cvs)
    #@assert allequal(length(getweights(cv.qr)) for (_, cv) in cvs)
    cv1 = first(values(cvs))
    @assert all(==(typeof(cv1.qr)), typeof(cv.qr) for (_, cv) in cvs)
    @assert all(==(length(getweights(cv1.qr))), length(getweights(cv.qr)) for (_, cv) in cvs)
    
    geo_values = GeometryValues(cv1)
    fun_values = NamedTuple(key=>FunctionValues(cv) for (key, cv) in cvs)
    return MultiCellValues2(geo_values, fun_values, cv1.detJdV, cv1.qr)
end

# Quadrature
# getnquadpoints(::QuadratureRule) would be nice...
getnquadpoints(cv::MultiCellValues2) = length(cv.qr.weights)

# Geometric functions
getngeobasefunctions(cv::MultiCellValues2) = getngeobasefunctions(cv.geo_values)
getdetJdV(cv::MultiCellValues2, q_point::Int) = cv.detJdV[q_point]
geometric_value(cv::MultiCellValues2, q_point::Int, base_func::Int) = geometric_value(cv.geo_values, q_point, base_func)

# FunctionValues functions: call like with CellValues, but foo(cv[:name], args...)
Base.getindex(cmv::MultiCellValues2, key::Symbol) = getindex(cmv.fun_values, key)

# Standard identity mapping (scalar function / lagrange vector)
function apply_mapping!(funvals::FunctionValues, q_point::Int, Jinv::Tensor{2})
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
During construction of `MultiCellValues2`, sizes of buffers are checked to match `qr`.
Hence, it is allowed to use `@inbounds` in this function.
"""
function apply_mapping! end  
=#

# Note that this function is "unsafe", as it applies inbounds. Checks in reinit!
function calculate_mapping(geo_values::GeometryValues{dim}, q_point, x) where dim
    fecv_J = zero(Tensors.getreturntype(⊗, eltype(x), eltype(geo_values.dMdξ)))
    @inbounds for j in 1:getngeobasefunctions(geo_values)
        fecv_J += x[j] ⊗ geo_values.dMdξ[j, q_point]
    end
    detJ = det(fecv_J)
    detJ > 0.0 || throw_detJ_not_pos(detJ)
    return detJ, inv(fecv_J)
end

function reinit!(cv::MultiCellValues2, x::AbstractVector{<:Vec})
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

function Base.show(io::IO, ::MIME"text/plain", fe_v::MultiCellValues2)
    print(io, "$(typeof(fe_v))")
end
