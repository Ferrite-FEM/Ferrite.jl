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

# Convenience
create_qr_rule(qr::QuadratureRule, args...) = qr 
create_qr_rule(qr::Int, ::Interpolation{Dim,RefShape}) where {Dim,RefShape} = QuadratureRule{Dim,RefShape}(qr)

function MultiCellValues(dh::DofHandler; kwargs...)
    @assert length(dh.fieldhandlers)==1
    return MultiCellValues(first(dh.fieldhandlers), getcelltype(dh.grid); kwargs...)
end
function MultiCellValues(fh::FieldHandler, CT; qr=2)
    # TODO: With new SubDofHandler, CT should not be required anymore. 
    ip_geo = default_interpolation(CT)
    @assert getdim(ip_geo) > 1 # For 1-dimensional problems we cannot differentiate vector vs scalar values
    qr_actual = create_qr_rule(qr, ip_geo)
    ip_funs = [name=>getfieldinterpolation(fh, name) for name in getfieldnames(fh)]
    val_type = Dict(name=> 1==getfielddim(fh, name) ? CellScalarValues : CellVectorValues for name in getfieldnames(fh))
    # TODO: Create only the minimum required unique values: I.e. if ip_fun is the same, we can use the same value 
    return MultiCellValues(NamedTuple(key=> val_type[key](qr_actual, ip_fun, ip_geo) for (key, ip_fun) in ip_funs))
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

function _unsafe_calculate_mapping(cv::CellValues, q_point, x)
    @inbounds fecv_J = x[1] ⊗ cv.dMdξ[1, q_point]
    @inbounds for j in 2:getngeobasefunctions(cv)
        fecv_J += x[j] ⊗ cv.dMdξ[j, q_point]
    end
    detJ = det(fecv_J)
    detJ > 0.0 || throw_detJ_not_pos(detJ)
    Jinv = inv(fecv_J)
    return detJ, Jinv 
end

function _unsafe_apply_mapping!(cv, q_point, detJ_w, Jinv)
    @inbounds cv.detJdV[q_point] = detJ_w
    @inbounds for j in 1:getnbasefunctions(cv)
        cv.dNdx[j, q_point] = cv.dNdξ[j, q_point] ⋅ Jinv
    end
    return nothing
end

function reinit!(cvs::MultiCellValues, x::AbstractVector{Vec{dim,T}}) where {dim,T}
    if length(cvs.values) == 1
        # Short-circuit when all cellvalues are the same
        return reinit!(first(cvs.values), x)
    end 
    n_geom_basefuncs = getngeobasefunctions(cvs)
    length(x) == n_geom_basefuncs || throw_incompatible_coord_length(length(x), n_geom_basefuncs)

    cv_ref = first(cvs.values) # Reference value for geometric interpolation and quadrature
    @inbounds for (q_point, w) in pairs(cv_ref.qr.weights)
        detJ, Jinv = _unsafe_calculate_mapping(cv_ref, q_point, x)
        detJ_w = detJ*w
        map(cvi -> _unsafe_apply_mapping!(cvi, q_point, detJ_w, Jinv), cvs.values)
    end
end

function Base.show(io::IO, ::MIME"text/plain", fe_v::MultiCellValues)
    print(io, "$(typeof(fe_v))")
end
