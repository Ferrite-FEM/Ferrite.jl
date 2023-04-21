"""
    MultiCellValues(;kwargs...)

Create `MultiCellValues` that contains the cellvalues supplied via keyword arguments

```
cv_vector = CellVectorValues(...)
cv_scalar = CellScalarValues(...)
cvs = MultiCellValues(;u=cv_vector, p=cv_scalar, T=cv_scalar)
```

`cvs` is `reinit!`:ed as regular cellvalues. 
Functions for getting information about quadrature points and geometric interpolation 
accept `cvs` directly. Functions to access the specific function interpolation values 
are called as `foo(cvs[:u], args...)` for `u`, and equivalent for other keys. 
"""
struct MultiCellValues{dim,T,RefShape,CVS<:Tuple,NV<:NamedTuple} <: CellValues{dim,T,RefShape}
    values::CVS         # Points only to unique CellValues
    named_values::NV    # Can point to the same CellValues in values multiple times
end
MultiCellValues(;cvs...) = MultiCellValues(NamedTuple(cvs))
function MultiCellValues(named_values::NamedTuple)
    # Extract the unique CellValues checked by ===
    tuple_values = tuple(unique(objectid, values(named_values))...)

    # Check that all values are compatible with eachother
    # allequal julia>=1.8
    cv_ref = first(named_values)
    @assert all( getpoints(cv.qr) ==  getpoints(cv_ref.qr) for cv in tuple_values)
    @assert all(getweights(cv.qr) == getweights(cv_ref.qr) for cv in tuple_values)
    # Note: The following only works while isbitstype(Interpolation)
    @assert all(cv.geo_interp == cv_ref.geo_interp for cv in tuple_values) 

    # getrefshape only defined for ip, and ip is type-unstable with current parameterization
    get_type_params(::CellValues{dim,T,RefShape}) where {dim,T,RefShape} = (dim,T,RefShape)
    dim,T,RefShape = get_type_params(cv_ref)
    return MultiCellValues{dim,T,RefShape,typeof(tuple_values),typeof(named_values)}(tuple_values, named_values)
end

# Not sure if aggressive constprop is required, but is intended so use to ensure? (Not supported on v1.6)
# Base.@constprop :aggressive Base.getindex(mcv::MultiCellValues, key::Symbol) = getindex(mcv.named_values, key)
Base.getindex(mcv::MultiCellValues, key::Symbol) = getindex(mcv.named_values, key)

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
    fecv_J = zero(Tensor{2,dim,T})
    @inbounds for j in 1:getngeobasefunctions(cv)
        fecv_J += x[j] ⊗ cv.dMdξ[j, q_point]
    end
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
# Specialized, versions for 1-3 values: Not often more than 3 unique values,
# and also not that much to gain for those problem sizes. Only 1s really important.
# Alternative would be @generated + Base.Cartesian.@nexprs
function apply_mapping!(cvs_values::NTuple{1,CellValues}, q_point, detJ_w, Jinv)
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