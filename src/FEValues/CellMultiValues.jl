# Notes
# Collect multiple function values inside one object, as the geometry updates will be the same.
# These FunctionScalarValues/FunctionVectorValues can also be used inside FaceValues
# In addition to the potential speed improvements, this structure has the following user side improvements
# - For each SubDofHandler, only one cellvalues object 
# - For the loop, only one quadrature point (nothing preventing separate quad-rules if required by using separate cellvalues)
# - Consistency between each CellValues object ensured automatically (i.e. no strange bugs from differences in e.g. quadrules)
# - Each FunctionValues object accessible via a symbol (stored in NamedTuple) which works statically 
#
# Additional ideas for restructuring
# Currently, all the cellvalues are made for a full cell. However, we always loop over each quadrature point, essentially 
# duplicating all values. From that perspective, it would make more sense to do `reinit!(values, cell_coords, quad_point_nr)`
# And only save for every quad point values that don't change upon reinit!. 
# This reduces the memory usage and could potentially reduce bandwidth issues and memory alignment.
# The looping structure would then becomes
#= 
for q_point in 1:getnquadpoints(cv)
    reinit!(cv, x, q_point)
    dΩ = getdetJdV(cv)
    for i in 1:getnbasefunctions(cv)
        ∇N = shape_gradient(cv, i)
        for j in 1:getnbasefunctions(cv)
            ∇δN = shape_gradient(cv, j)
            Ke[j,i] = ∇δN ⋅ ∇N * dΩ
        end
    end
end
Where the first two lines could even be written as 
for qp in QuadPointIterator(cv, x)
    dΩ = getdetJdV(qp)
    .... # replacing cv with qp
=#


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
@propagate_inbounds geometric_value(geovals::GeometryValues, q_point::Int, base_func::Int) = geovals.M[base_func, q_point]
@propagate_inbounds getdetJdV(geovals::GeometryValues, q_point::Int) = geovals.detJdV[q_point]

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
    qr::QuadratureRule{dim,RefShape,T}
end
function CellMultiValues(;cvs...)
    # cvs::Pairs{Symbol, CellValues, Tuple, NamedTuple}, cf. foo(;kwargs...) = kwargs
    @assert allequal(cv.qr for (_, cv) in cvs)
    cv1 = first(values(cvs))
    geo_values = GeometryValues(cv1)
    fun_values = NamedTuple(key=>create_function_values(cv) for (key, cv) in cvs)
    return CellMultiValues(geo_values, fun_values, cv1.qr)
end

# Quadrature
# getnquadpoints(::QuadratureRule) would be nice...
getnquadpoints(cv::CellMultiValues) = length(cv.qr.weights)

# Geometric functions
getngeobasefunctions(cv::CellMultiValues) = getngeobasefunctions(cv.geo_values)
getdetJdV(cv::CellMultiValues, q_point::Int) = getdetJdV(cv.geo_values, q_point)
geometric_value(cv::CellMultiValues, q_point::Int, base_func::Int) = geometric_value(cv.geo_values, q_point, base_func)

# FunctionValues functions: call like with CellValues, but foo(cv[:name], args...)
Base.getindex(cmv::CellMultiValues, key::Symbol) = getindex(cmv.fun_values, key)
# Note: Need to add tests that checks that type is inferred (this seems to work for mwe)

function _update_dNdx!(funvals::FunctionValues{dim}, i::Int, Jinv::Tensor{2,dim}) where dim
    @inbounds for j in 1:getnbasefunctions(funvals)
        funvals.dNdx[j, i] = funvals.dNdξ[j, i] ⋅ Jinv
    end
    return nothing
end

function reinit!(cv::CellMultiValues{dim}, x::AbstractVector{Vec{dim,T}}) where {dim,T}
    n_geom_basefuncs = getngeobasefunctions(cv)
    length(x) == n_geom_basefuncs || throw_incompatible_coord_length(length(x), n_geom_basefuncs)
    @inbounds for (i, w) in enumerate(cv.qr.weights)
        fecv_J = zero(Tensor{2,dim})
        for j in 1:n_geom_basefuncs
            fecv_J += x[j] ⊗ cv.geo_values.dMdξ[j, i]
        end
        detJ = det(fecv_J)
        detJ > 0.0 || throw_detJ_not_pos(detJ)
        cv.geo_values.detJdV[i] = detJ * w
        Jinv = inv(fecv_J)
        map(funvals->_update_dNdx!(funvals, i, Jinv), funvals_tuple) # map required for performance!
    end
end

function Base.show(io::IO, ::MIME"text/plain", fe_v::CellMultiValues)
    print(io, "$(typeof(fe_v))")
end