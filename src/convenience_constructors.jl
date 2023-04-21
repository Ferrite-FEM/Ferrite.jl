# Convenience constructors often require multiple types to be defined.
# Where necessary, these can be put in this file.

# MultiCellValues
create_qr_rule(qr::QuadratureRule, args...) = qr 
create_qr_rule(qr::Int, ::Interpolation{Dim,RefShape}) where {Dim,RefShape} = QuadratureRule{Dim,RefShape}(qr)


"""
    MultiCellValues(dh::DofHandler; qr::Union{Int,QuadratureRule}=2)
    MultiCellValues(fh::FieldHandler, CT::Type{<:AbstractCell}; qr::Union{Int,QuadratureRule}=2)

Automatically create MultiCellValues where the values are accessed by the fieldname.
If `qr::Int` is given, create `QuadratureRule{dim,RefShape}(qr)` where `dim` and `RefShape` 
is taken from `dh` or `fh`. 
If fields have the same interpolation, only one `CellValues` object will be created for these fields,
but this value can be accessed by each of the fields' names.

Note: 1D cases are currently not supported as it is not possible to differentiate between 
scalar and vectorial fields in this case. 
"""
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