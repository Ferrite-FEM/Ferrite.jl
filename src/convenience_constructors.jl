# Convenience constructors often require multiple types to be defined.
# Where necessary, these can be put in this file.

_create_qr_rule(qr::QuadratureRule, args...) = qr 
_create_qr_rule(order::Int, ::Interpolation{RefShape}) where {RefShape} = QuadratureRule{RefShape}(order)

"""
    CellValues(dh::DofHandler, fieldname::Symbol; qr::Union{Int,QuadratureRule})
    CellValues(sdh::SubDofHandler, fieldname::Symbol; qr::Union{Int,QuadratureRule})

Create a `CellValues` object by using the interpolation of `fieldname` in `dh` 
or `sdh`, as well as the default geometric interpolation based on the cell type.
A `DofHandler` input is only allowed with a single `SubDofHandler` in `dh`.
The quadrature rule is created automatically as `QuadratureRule{dim,RefShape}(qr)`,
where `dim` and `RefShape` are taken from `dh` or `sdh`, if `qr::Int` is given.
"""
function CellValues(dh::DofHandler, fieldname::Symbol; kwargs...)
    length(dh.subdofhandlers)==1 || throw(ArgumentError("Multiple SubDofHandlers are not supported, give a single SubDofHandler instead"))
    return CellValues(first(dh.subdofhandlers), fieldname; kwargs...)
end
function CellValues(sdh::SubDofHandler, fieldname::Symbol; qr)
    CT = getcelltype(sdh)
    ip_geo = default_interpolation(CT)
    qr_actual = _create_qr_rule(qr, ip_geo)
    ip = getfieldinterpolation(sdh, fieldname)
    return CellValues(qr_actual, ip, ip_geo)
end

_create_fqr_rule(qr::FaceQuadratureRule, args...) = qr 
_create_fqr_rule(order::Int, ::Interpolation{RefShape}) where {RefShape} = FaceQuadratureRule{RefShape}(order)

"""
    FaceValues(dh::DofHandler, fieldname::Symbol; qr::Union{Int,FaceQuadratureRule})
    FaceValues(sdh::SubDofHandler, fieldname::Symbol; qr::Union{Int,FaceQuadratureRule})

Create a `FaceValues` object by using the interpolation of `fieldname` in `dh` 
or `sdh`, as well as the default geometric interpolation based on the cell type.
A `DofHandler` input is only allowed with a single `SubDofHandler` in `dh`.
The quadrature rule is created automatically as `FaceQuadratureRule{dim,RefShape}(qr)`,
where `dim` and `RefShape` are taken from `dh` or `sdh`, if `qr::Int` is given.
"""
function FaceValues(dh::DofHandler, fieldname::Symbol; kwargs...)
    length(dh.subdofhandlers)==1 || throw(ArgumentError("Multiple SubDofHandlers are not supported, give a single SubDofHandler instead"))
    return FaceValues(first(dh.subdofhandlers), fieldname; kwargs...)
end
function FaceValues(sdh::SubDofHandler, fieldname::Symbol; qr)
    CT = getcelltype(sdh)
    ip_geo = default_interpolation(CT)
    qr_actual = _create_fqr_rule(qr, ip_geo)
    ip = getfieldinterpolation(sdh, fieldname)
    return FaceValues(qr_actual, ip, ip_geo)
end
