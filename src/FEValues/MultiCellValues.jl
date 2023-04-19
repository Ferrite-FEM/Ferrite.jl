struct MultiCellValues{dim,T,RefShape,CVS<:Tuple,NV<:NamedTuple} <: CellValues{dim,T,RefShape}
    values::CVS         # Points only to unique values
    named_values::NV    # Can point to the same value in values multiple times
end
MultiCellValues(;cvs...) = MultiCellValues(NamedTuple(cvs))
function MultiCellValues(named_values::NamedTuple)
    # Check that all are compatible 
    @assert allequal(typeof(cv.qr) for cv in named_values)
    @assert allequal(length(getweights(cv.qr)) for cv in named_values)
    # Should also check the geometric interpolation...
    
    function get_unique_values(values_all)
        # Quick and dirty for testing
        tmp = []
        for value in values_all
            any(x===value for x in tmp) || push!(tmp, value)
        end
        return tuple(tmp...)
    end

    tuple_values = get_unique_values(values(named_values))
    # return MultiCellValues(tuple_values, named_values)
    # Temp until CellValues gets different parameterization...
    get_type_params(::CellValues{dim,T,RefShape}) where {dim,T,RefShape} = (dim,T,RefShape)
    dim,T,RefShape = get_type_params(first(tuple_values))
    return MultiCellValues{dim,T,RefShape,typeof(tuple_values),typeof(named_values)}(tuple_values, named_values)
end

# Convenience constructors (Reason for including this file later)
create_qr_rule(qr::QuadratureRule, args...) = qr 
create_qr_rule(qr::Int, ::Interpolation{Dim,RefShape}) where {Dim,RefShape} = QuadratureRule{Dim,RefShape}(qr)

function MultiCellValues(dh::DofHandler; kwargs...)
    @assert length(dh.fieldhandlers)==1
    return MultiCellValues(first(dh.fieldhandlers), getcelltype(dh.grid); kwargs...)
end
function MultiCellValues(fh::FieldHandler, CT; qr=2)
    # TODO: With new SubDofHandler, CT should not be required anymore. 
    ip_geo = default_interpolation(CT)

    qr_actual = create_qr_rule(qr, ip_geo)

    # TODO: For 1-dimensional problems we cannot differentiate vector vs scalar values
    #       This can be solved with the new vectorized interpolations discussin in 
    #       Then, we also only need ip_fun as key to the dict below
    @assert getdim(ip_geo) > 1
    ip_funs = [name=>getfieldinterpolation(fh, name) for name in getfieldnames(fh)]
    values = Dict{}()
    for (name, ip_fun) in ip_funs
        dim = getfielddim(fh, name)
        if !(haskey(values, (ip_fun,dim)))
            if dim == 1
                values[(ip_fun,dim)] = CellScalarValues(qr_actual, ip_fun, ip_geo)
            else
                values[(ip_fun,dim)] = CellVectorValues(qr_actual, ip_fun, ip_geo)
            end
        end
    end

    return MultiCellValues(NamedTuple(name=>values[(ip_fun, getfielddim(fh, name))] for (name, ip_fun) in ip_funs))
end

# Not sure if aggressive constprop is required, but is intended so use to ensure?
Base.@constprop :aggressive Base.getindex(mcv::MultiCellValues, key::Symbol) = getindex(mcv.named_values, key)

# Geometric values should all be equal and hence can be queried from ::MultiCellValues
@propagate_inbounds getngeobasefunctions(mcv::MultiCellValues) = getngeobasefunctions(first(mcv.values))
@propagate_inbounds function geometric_value(mcv::MultiCellValues, q_point::Int, base_func::Int)
    return geometric_value(first(mcv.values), q_point, base_func)
end

# Quadrature
getnquadpoints(mcv::MultiCellValues) = getnquadpoints(first(mcv.values))
# @propagate_inbounds getdetJdV ? 
getdetJdV(mcv::MultiCellValues, q_point::Int) = getdetJdV(first(mcv.values), q_point)

@inline function _unsafe_calculate_mapping(cv::CellValues{dim,T}, q_point, x) where {dim,T}
    #@inbounds fecv_J = x[1] ⊗ cv.dMdξ[1, q_point]
    fecv_J = zero(Tensor{2,dim,T})
    @inbounds for j in 1:getngeobasefunctions(cv)
        fecv_J += x[j] ⊗ cv.dMdξ[j, q_point]
    end
    #return fecv_J
    detJ = det(fecv_J)
    detJ > 0.0 || throw_detJ_not_pos(detJ)
    Jinv = inv(fecv_J)
    return detJ, Jinv 
end

@inline function _unsafe_apply_mapping!(cv, q_point, detJ_w, Jinv)
    @inbounds cv.detJdV[q_point] = detJ_w
    @inbounds for j in 1:getnbasefunctions(cv)
        cv.dNdx[j, q_point] = cv.dNdξ[j, q_point] ⋅ Jinv
    end
    return nothing
end

# The following is quite fast, a few percent is saved by specialized versions below
function apply_mapping!(cvs_values, q_point, detJ_w, Jinv)
    map(cvi -> _unsafe_apply_mapping!(cvi, q_point, detJ_w, Jinv), cvs_values)
end

# Specialized, should probably do with generated function for the cases below.
# For performance, it is mostly relevant for the length(cvs_values)=1 though...
#@generated function apply_mapping!(cvs_values::NTuple{CellValues,N}, q_point, detJ_w, Jinv) where N
#    Base.Cartesian.@nexprs N i -> _unsafe_apply_mapping!(cvs_values[i], q_point, detJ_w, Jinv)
#end

function apply_mapping!(cvs_values::Tuple{<:CellValues}, q_point, detJ_w, Jinv)
    _unsafe_apply_mapping!(cvs_values[1], q_point, detJ_w, Jinv)
end
function apply_mapping!(cvs_values::NTuple{2,CellValues}, q_point, detJ_w, Jinv)
    _unsafe_apply_mapping!(cvs_values[1], q_point, detJ_w, Jinv)
    _unsafe_apply_mapping!(cvs_values[2], q_point, detJ_w, Jinv)
end
function apply_mapping!(cvs_values::NTuple{3,CellValues}, q_point, detJ_w, Jinv)
    _unsafe_apply_mapping!(cvs_values[1], q_point, detJ_w, Jinv)
    _unsafe_apply_mapping!(cvs_values[2], q_point, detJ_w, Jinv)
    _unsafe_apply_mapping!(cvs_values[3], q_point, detJ_w, Jinv)
end

function reinit!(cvs::MultiCellValues, x::AbstractVector{Vec{dim,T}}) where {dim,T}
    n_geom_basefuncs = getngeobasefunctions(cvs)
    length(x) == n_geom_basefuncs || throw_incompatible_coord_length(length(x), n_geom_basefuncs)
    
    cv_ref = first(cvs.values) # Reference value for geometric interpolation and quadrature
    @inbounds for (q_point, w) in pairs(cv_ref.qr.weights)
        detJ, Jinv = _unsafe_calculate_mapping(cv_ref, q_point, x)
        detJ > 0.0 || throw_detJ_not_pos(detJ)
        detJ_w = detJ*w
        apply_mapping!(cvs.values, q_point, detJ_w, Jinv)
    end
end

function Base.show(io::IO, ::MIME"text/plain", fe_v::MultiCellValues)
    print(io, "MultiCellValues with ", length(fe_v.values), " unique values. Access names:")
    for (name, cv) in pairs(fe_v.named_values)
        println(io)
        print(io, "$name: ")
        show(io, MIME"text/plain"(), cv)
    end
end