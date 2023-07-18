struct FaceGeometryValues{dMdξ_t, GIP, T, Normal_t}
    M::Matrix{T}
    dMdξ::Matrix{dMdξ_t}
    detJdV::Vector{T}
    normals::Vector{Normal_t}
    ip::GIP
end
function FaceGeometryValues(::Type{T}, ip_vec::VectorizedInterpolation{sdim}, qr::QuadratureRule) where {T,sdim}
    ip = ip_vec.ip
    n_shape = getnbasefunctions(ip)
    n_qpoints = getnquadpoints(qr)
    M    = zeros(T,  n_shape, n_qpoints)
    dMdξ = zeros(Vec{getdim(ip),T}, n_shape, n_qpoints)
    for (qp, ξ) in pairs(getpoints(qr))
        for i in 1:n_shape
            dMdξ[i, qp], M[i, qp] = shape_gradient_and_value(ip, ξ, i)
        end
    end
    normals = fill(zero(Vec{sdim,T})*T(NaN), n_qpoints)
    detJdV = fill(T(NaN), n_qpoints)
    return FaceGeometryValues(M, dMdξ, detJdV, normals, ip)
end

getngeobasefunctions(geovals::FaceGeometryValues) = size(geovals.M, 1)
@propagate_inbounds geometric_value(geovals::FaceGeometryValues, q_point::Int, base_func::Int) = geovals.M[base_func, q_point]
@propagate_inbounds getdetJdV(geovals::FaceGeometryValues, q_point::Int) = geovals.detJdV[q_point]
@propagate_inbounds getnormal(geovals::FaceGeometryValues, q_point::Int) = geovals.normals[q_point]

@inline function calculate_mapping(geo_values::FaceGeometryValues{<:Vec{dim,T}}, face_nr::Int, q_point, w, x::AbstractVector{<:Vec{dim,T}}) where {dim,T}
    fefv_J = zero(Tensor{2,dim,T}) # zero(Tensors.getreturntype(⊗, eltype(x), eltype(geo_values.dMdξ)))
    @inbounds for j in 1:getngeobasefunctions(geo_values)
        fefv_J += x[j] ⊗ geo_values.dMdξ[j, q_point]
    end
    weight_norm = weighted_normal(fefv_J, getrefshape(geo_values.ip), face_nr)
    detJ = norm(weight_norm)
    detJ > 0.0 || throw_detJ_not_pos(detJ)
    @inbounds geo_values.detJdV[q_point] = detJ*w
    @inbounds geo_values.normals[q_point] = weight_norm / norm(weight_norm)
    return inv(fefv_J)
end

struct FaceValues{IP, N_t, dNdx_t, dNdξ_t, T, dMdξ_t, QR, Normal_t, GIP} <: AbstractFaceValues
    geo_values::Vector{FaceGeometryValues{dMdξ_t, GIP, T, Normal_t}}
    fun_values::Vector{FunctionValues{IP, N_t, dNdx_t, dNdξ_t}}
    qr::QR # FaceQuadratureRule
    current_face::ScalarWrapper{Int}
end

function FaceValues(::Type{T}, fqr::FaceQuadratureRule, ip_fun::Interpolation, ip_geo::VectorizedInterpolation=default_geometric_interpolation(ip_fun)) where T 
    geo_values = [FaceGeometryValues(T, ip_geo, qr) for qr in fqr.face_rules]
    fun_values = [FunctionValues(T, ip_fun, qr, ip_geo) for qr in fqr.face_rules]
    return FaceValues(geo_values, fun_values, fqr, ScalarWrapper(1))
end

FaceValues(qr::FaceQuadratureRule, ip::Interpolation, args...) = FaceValues(Float64, qr, ip, args...)
function FaceValues(::Type{T}, qr::FaceQuadratureRule, ip::Interpolation, ip_geo::ScalarInterpolation) where T
    return FaceValues(T, qr, ip, VectorizedInterpolation(ip_geo))
end

getngeobasefunctions(fv::FaceValues) = getngeobasefunctions(get_geo_values(fv))
getnbasefunctions(fv::FaceValues) = getnbasefunctions(get_fun_values(fv))
getnquadpoints(fv::FaceValues) = getnquadpoints(fv.qr, getcurrentface(fv))

get_geo_values(fv::FaceValues) = @inbounds fv.geo_values[getcurrentface(fv)]
for op = (:getdetJdV, :getngeobasefunctions, :geometric_value)
    eval(quote
        @propagate_inbounds $op(fv::FaceValues, args...) = $op(get_geo_values(fv), args...)
    end)
end

get_fun_values(fv::FaceValues) = @inbounds fv.fun_values[getcurrentface(fv)]
for op = (:shape_value, :shape_gradient, :shape_symmetric_gradient, :shape_curl)
    eval(quote
        @propagate_inbounds $op(fv::FaceValues, i::Int, q_point::Int) = $op(get_fun_values(fv), i, q_point)
    end)
end

"""
    getcurrentface(fv::FaceValues)

Return the current active face of the `FaceValues` object (from last `reinit!`).

"""
getcurrentface(fv::FaceValues) = fv.current_face[]

"""
    getnormal(fv::FaceValues, qp::Int)

Return the normal at the quadrature point `qp` for the active face of the
`FaceValues` object(from last `reinit!`).
"""
getnormal(fv::FaceValues, qp::Int) = getnormal(get_geo_values(fv), qp)

nfaces(fv::FaceValues) = length(fv.geo_values)

function checkface(fv::FaceValues, face::Int)
    0 < face <= nfaces(fv) || error("Face index out of range.")
    return nothing
end

function reinit!(fv::FaceValues, x::AbstractVector{Vec{dim,T}}, face_nr::Int) where {dim, T}
    @boundscheck checkface(fv, face_nr)
    n_geom_basefuncs = getngeobasefunctions(fv)
    length(x) == n_geom_basefuncs || throw_incompatible_coord_length(length(x), n_geom_basefuncs)
    
    fv.current_face[] = face_nr

    geo_values = get_geo_values(fv)
    fun_values = get_fun_values(fv)
    @inbounds for (q_point, w) in pairs(getweights(fv.qr, face_nr))
        Jinv = calculate_mapping(geo_values, face_nr, q_point, w, x)
        apply_mapping!(fun_values, q_point, Jinv)
    end
end

"""
    BCValues(func_interpol::Interpolation, geom_interpol::Interpolation, boundary_type::Union{Type{<:BoundaryIndex}})

`BCValues` stores the shape values at all faces/edges/vertices (depending on `boundary_type`) for the geomatric interpolation (`geom_interpol`),
for each dof-position determined by the `func_interpol`. Used mainly by the `ConstrainHandler`.
"""
struct BCValues{T}
    M::Array{T,3}
    nqp::Array{Int}
    current_entity::ScalarWrapper{Int}
end

BCValues(func_interpol::Interpolation, geom_interpol::Interpolation, boundary_type::Type{<:BoundaryIndex} = Ferrite.FaceIndex) =
    BCValues(Float64, func_interpol, geom_interpol, boundary_type)

function BCValues(::Type{T}, func_interpol::Interpolation{refshape}, geom_interpol::Interpolation{refshape}, boundary_type::Type{<:BoundaryIndex} = Ferrite.FaceIndex) where {T,dim,refshape <: AbstractRefShape{dim}}
    # set up quadrature rules for each boundary entity with dof-positions
    # (determined by func_interpol) as the quadrature points
    interpolation_coords = reference_coordinates(func_interpol)

    qrs = QuadratureRule{refshape,T,dim}[]
    for boundarydofs in dirichlet_boundarydof_indices(boundary_type)(func_interpol)
        dofcoords = Vec{dim,T}[]
        for boundarydof in boundarydofs
            push!(dofcoords, interpolation_coords[boundarydof])
        end
        qrf = QuadratureRule{refshape,T}(fill(T(NaN), length(dofcoords)), dofcoords) # weights will not be used
        push!(qrs, qrf)
    end

    n_boundary_entities = length(qrs)
    n_qpoints = n_boundary_entities == 0 ? 0 : maximum(qr->length(getweights(qr)), qrs) # Bound number of qps correctly.
    n_geom_basefuncs = getnbasefunctions(geom_interpol)
    M   = fill(zero(T) * T(NaN), n_geom_basefuncs, n_qpoints, n_boundary_entities)
    nqp = zeros(Int,n_boundary_entities)

    for n_boundary_entity in 1:n_boundary_entities
        for (qp, ξ) in enumerate(qrs[n_boundary_entity].points), i in 1:n_geom_basefuncs
            M[i, qp, n_boundary_entity] = shape_value(geom_interpol, ξ, i)
        end
        nqp[n_boundary_entity] = length(qrs[n_boundary_entity].points)
    end

    BCValues{T}(M, nqp, ScalarWrapper(0))
end

getnquadpoints(bcv::BCValues) = bcv.nqp[bcv.current_entity.x]
function spatial_coordinate(bcv::BCValues, q_point::Int, xh::AbstractVector{Vec{dim,T}}) where {dim,T}
    n_base_funcs = size(bcv.M, 1)
    length(xh) == n_base_funcs || throw_incompatible_coord_length(length(xh), n_base_funcs)
    x = zero(Vec{dim,T})
    face = bcv.current_entity[]
    @inbounds for i in 1:n_base_funcs
        x += bcv.M[i,q_point,face] * xh[i] # geometric_value(fe_v, q_point, i) * xh[i]
    end
    return x
end

include("face_values.jl")