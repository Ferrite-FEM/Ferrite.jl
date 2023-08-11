struct FaceValues{IP, N_t, dNdx_t, dNdξ_t, T, dMdξ_t, QR, Normal_t, GIP} <: AbstractFaceValues
    geo_values::Vector{GeometryValues{dMdξ_t, GIP, T}}
    detJdV::Vector{T}
    normals::Vector{Normal_t}
    fun_values::Vector{FunctionValues{IP, N_t, dNdx_t, dNdξ_t}}
    qr::QR # FaceQuadratureRule
    current_face::ScalarWrapper{Int}
end

function FaceValues(::Type{T}, fqr::FaceQuadratureRule, ip_fun::Interpolation, ip_geo::VectorizedInterpolation{sdim}=default_geometric_interpolation(ip_fun)) where {T,sdim} 
    geo_values = [GeometryValues(T, ip_geo.ip, qr) for qr in fqr.face_rules]
    fun_values = [FunctionValues(T, ip_fun, qr, ip_geo) for qr in fqr.face_rules]
    detJdV = fill(T(NaN), maximum(qr->length(getweights(qr)), fqr.face_rules))
    normals = fill(zero(Vec{sdim,T})*T(NaN), length(geo_values))
    return FaceValues(geo_values, detJdV, normals, fun_values, fqr, ScalarWrapper(1))
end

FaceValues(qr::FaceQuadratureRule, ip::Interpolation, args...) = FaceValues(Float64, qr, ip, args...)
function FaceValues(::Type{T}, qr::FaceQuadratureRule, ip::Interpolation, ip_geo::ScalarInterpolation) where T
    return FaceValues(T, qr, ip, VectorizedInterpolation(ip_geo))
end

getngeobasefunctions(fv::FaceValues) = getngeobasefunctions(get_geo_values(fv))
getnbasefunctions(fv::FaceValues) = getnbasefunctions(get_fun_values(fv))
getnquadpoints(fv::FaceValues) = getnquadpoints(fv.qr, getcurrentface(fv))
getdetJdV(fv::FaceValues, q_point) = fv.detJdV[q_point] 

get_geo_values(fv::FaceValues) = @inbounds fv.geo_values[getcurrentface(fv)]
for op = (:getngeobasefunctions, :geometric_value)
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
getnormal(fv::FaceValues, qp::Int) = fv.normals[qp]

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
        mapping = calculate_mapping(geo_values, q_point, x)
        J = getjacobian(mapping)
        weight_norm = weighted_normal(J, getrefshape(geo_values.ip), face_nr)
        detJ = norm(weight_norm)
        detJ > 0.0 || throw_detJ_not_pos(detJ)
        @inbounds fv.detJdV[q_point] = detJ*w
        @inbounds fv.normals[q_point] = weight_norm / norm(weight_norm)       
        apply_mapping!(fun_values, q_point, mapping)
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