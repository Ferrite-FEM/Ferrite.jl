# Convenience constructors often require multiple types to be defined.
# Where necessary, these can be put in this file.

# MultiCellValues
create_qr_rule(qr::QuadratureRule, args...) = qr 
create_qr_rule(qr::Int, ::Interpolation{RefShape}) where {RefShape} = QuadratureRule{RefShape}(qr)

"""
    MultiCellValues(dh::DofHandler; qr::Union{Int,QuadratureRule})
    MultiCellValues(sdh::SubDofHandler; qr::Union{Int,QuadratureRule})

Automatically create MultiCellValues where the values are accessed by the fieldname.
Giving a `DofHandler` is only possible when there is only one `SubDofHandler` in `dh`.

The quadrature rule is created automatically as `QuadratureRule{dim,RefShape}(qr)`,
where `dim` and `RefShape` are taken from `dh` or `sdh`, if `qr::Int` is given. 

For fields that have the same interpolation, only one `CellValues` object will be 
created, but this object can be accessed by the names of all of those fields.
"""
function MultiCellValues(dh::DofHandler; kwargs...)
    length(dh.subdofhandlers)==1 || throw(ArgumentError("Multiple SubDofHandlers are not supported, give a single SubDofHandler instead"))
    return MultiCellValues(first(dh.subdofhandlers); kwargs...)
end
function MultiCellValues(sdh::SubDofHandler; qr)
    CT = getcelltype(get_grid(sdh.dh), sdh)
    ip_geo = default_interpolation(CT)

    qr_actual = create_qr_rule(qr, ip_geo)

    ip_funs = [name=>getfieldinterpolation(sdh, name) for name in getfieldnames(sdh)]
    values = Dict{Interpolation,CellValues}()
    for (_, ip_fun) in ip_funs
        haskey(values, ip_fun) || (values[ip_fun] = CellValues(qr_actual, ip_fun, ip_geo))
    end
    return MultiCellValues(NamedTuple(name=>values[ip_fun] for (name, ip_fun) in ip_funs))
end

"""
    CellValues(dh::DofHandler, fieldname::Symbol; qr::Union{Int,QuadratureRule})
    CellValues(sdh::SubDofHandler, fieldname::Symbol; qr::Union{Int,QuadratureRule})

Automatically create a CellValues, using the interpolation of `fieldname` in `dh` or `sdh`.
Giving a `DofHandler` is only possible when there is only one `SubDofHandler` in `dh`.

The quadrature rule is created automatically as `QuadratureRule{dim,RefShape}(qr)`,
where `dim` and `RefShape` are taken from `dh` or `sdh`, if `qr::Int` is given. 
"""
function CellValues(dh::DofHandler, fieldname::Symbol; kwargs...)
    length(dh.subdofhandlers)==1 || throw(ArgumentError("Multiple SubDofHandlers are not supported, give a single SubDofHandler instead"))
    return CellValues(first(dh.subdofhandlers), fieldname; kwargs...)
end
function CellValues(sdh::SubDofHandler, fieldname::Symbol; qr)
    CT = getcelltype(get_grid(sdh.dh), sdh)
    ip_geo = default_interpolation(CT)
    qr_actual = create_qr_rule(qr, ip_geo)
    ip = getfieldinterpolation(sdh, fieldname)
    return CellValues(qr_actual, ip, ip_geo)
end