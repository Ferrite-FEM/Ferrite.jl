"""
    FaceValues([::Type{T}], quad_rule::FaceQuadratureRule, func_interpol::Interpolation, [geom_interpol::Interpolation])

A `FaceValues` object facilitates the process of evaluating values of shape functions, gradients of shape functions,
values of nodal functions, gradients and divergences of nodal functions etc. on the faces of finite elements.

**Arguments:**

* `T`: an optional argument to determine the type the internal data is stored as.
* `quad_rule`: an instance of a [`FaceQuadratureRule`](@ref)
* `func_interpol`: an instance of an [`Interpolation`](@ref) used to interpolate the approximated function
* `geom_interpol`: an optional instance of an [`Interpolation`](@ref) which is used to interpolate the geometry.
  By default linear Lagrange interpolation is used.

**Common methods:**

* [`reinit!`](@ref)
* [`getnquadpoints`](@ref)
* [`getdetJdV`](@ref)

* [`shape_value`](@ref)
* [`shape_gradient`](@ref)
* [`shape_symmetric_gradient`](@ref)
* [`shape_divergence`](@ref)

* [`function_value`](@ref)
* [`function_gradient`](@ref)
* [`function_symmetric_gradient`](@ref)
* [`function_divergence`](@ref)
* [`spatial_coordinate`](@ref)
"""
FaceValues

struct FaceValues{IP, N_t, dNdx_t, dNdξ_t, T, dMdξ_t, QR, Normal_t, GIP} <: AbstractFaceValues
    N::Array{N_t, 3}
    dNdx::Array{dNdx_t, 3}
    dNdξ::Array{dNdξ_t, 3}
    detJdV::Matrix{T}
    normals::Vector{Normal_t}
    M::Array{T, 3}
    dMdξ::Array{dMdξ_t, 3}
    qr::QR
    current_face::ScalarWrapper{Int}
    func_interp::IP
    geo_interp::GIP
end

# Common initializer code for constructing FaceValues after the types have been determined
function FaceValues{IP, N_t, dNdx_t, dNdξ_t, T, dMdξ_t, QR, Normal_t, GIP}(qr::QR, ip::IP, gip::GIP) where {
    IP, N_t, dNdx_t, dNdξ_t, T, dMdξ_t, QR, Normal_t, GIP,
}
    @assert isconcretetype(IP)     && isconcretetype(N_t)      && isconcretetype(dNdx_t) &&
            isconcretetype(dNdξ_t) && isconcretetype(T)        && isconcretetype(dMdξ_t) &&
            isconcretetype(QR)     && isconcretetype(Normal_t) && isconcretetype(GIP)
    # Quadrature
    max_n_qpoints = maximum(getnquadpoints, qr.face_rules; init = 0)
    n_faces = length(qr.face_rules)

    # Normals
    normals = zeros(Normal_t, max_n_qpoints)

    # Field interpolation
    n_func_basefuncs = getnbasefunctions(ip)
    N    = fill(zero(N_t)    * T(NaN), n_func_basefuncs, max_n_qpoints, n_faces)
    dNdx = fill(zero(dNdx_t) * T(NaN), n_func_basefuncs, max_n_qpoints, n_faces)
    dNdξ = fill(zero(dNdξ_t) * T(NaN), n_func_basefuncs, max_n_qpoints, n_faces)

    # Geometry interpolation
    n_geom_basefuncs = getnbasefunctions(gip)
    M    = fill(zero(T)      * T(NaN), n_geom_basefuncs, max_n_qpoints, n_faces)
    dMdξ = fill(zero(dMdξ_t) * T(NaN), n_geom_basefuncs, max_n_qpoints, n_faces)

    for face in 1:n_faces, (qp, ξ) in pairs(getpoints(qr, face))
        for basefunc in 1:n_func_basefuncs
            dNdξ[basefunc, qp, face], N[basefunc, qp, face] = shape_gradient_and_value(ip, ξ, basefunc)
        end
        for basefunc in 1:n_geom_basefuncs
            dMdξ[basefunc, qp, face], M[basefunc, qp, face] = shape_gradient_and_value(gip, ξ, basefunc)
        end
    end

    detJdV = fill(T(NaN), max_n_qpoints, n_faces)

    FaceValues{IP, N_t, dNdx_t, dNdξ_t, T, dMdξ_t, QR, Normal_t, GIP}(N, dNdx, dNdξ, detJdV, normals, M, dMdξ, qr, ScalarWrapper(0), ip, gip)
end

# Common entry point that fills in the numeric type and geometric interpolation
function FaceValues(qr::FaceQuadratureRule, ip::Interpolation,
        gip::Interpolation = default_geometric_interpolation(ip))
    return FaceValues(Float64, qr, ip, gip)
end

# Common entry point that fills in the geometric interpolation
function FaceValues(::Type{T}, qr::FaceQuadratureRule, ip::Interpolation) where {T}
    return FaceValues(T, qr, ip, default_geometric_interpolation(ip))
end

# Common entry point that vectorizes an input scalar geometric interpolation
function FaceValues(::Type{T}, qr::FaceQuadratureRule, ip::Interpolation, sgip::ScalarInterpolation) where {T}
    return FaceValues(T, qr, ip, VectorizedInterpolation(sgip))
end

# Entrypoint for `ScalarInterpolation`s (rdim == sdim)
function FaceValues(::Type{T}, qr::QR, ip::IP, gip::VGIP) where {
    dim, shape <: AbstractRefShape{dim}, T,
    QR  <: FaceQuadratureRule{shape},
    IP  <: ScalarInterpolation{shape},
    GIP <: ScalarInterpolation{shape},
    VGIP <: VectorizedInterpolation{dim, shape, <:Any, GIP},
}
    # Normals
    Normal_t = Vec{dim, T}
    # Function interpolation
    N_t = T
    dNdx_t = dNdξ_t = Vec{dim, T}
    # Geometry interpolation
    M_t    = T
    dMdξ_t = Vec{dim, T}
    return FaceValues{IP, N_t, dNdx_t, dNdξ_t, M_t, dMdξ_t, QR, Normal_t, GIP}(qr, ip, gip.ip)
end

# Entrypoint for `VectorInterpolation`s (vdim == rdim == sdim)
function FaceValues(::Type{T}, qr::QR, ip::IP, gip::VGIP) where {
    dim, shape <: AbstractRefShape{dim}, T,
    QR  <: FaceQuadratureRule{shape},
    IP  <: VectorInterpolation{dim, shape},
    GIP <: ScalarInterpolation{shape},
    VGIP <: VectorizedInterpolation{dim, shape, <:Any, GIP},
}
    # Normals
    Normal_t = Vec{dim, T}
    # Function interpolation
    N_t    = Vec{dim, T}
    dNdx_t = dNdξ_t = Tensor{2, dim, T, Tensors.n_components(Tensor{2,dim})}
    # Geometry interpolation
    M_t    = T
    dMdξ_t = Vec{dim, T}
    return FaceValues{IP, N_t, dNdx_t, dNdξ_t, M_t, dMdξ_t, QR, Normal_t, GIP}(qr, ip, gip.ip)
end

function reinit!(fv::FaceValues{<:Any, N_t, dNdx_t}, x::AbstractVector{Vec{dim,T}}, face::Int, reinit_values::Bool = false) where {
    dim, T,
    N_t    <: Union{Number,   Vec{dim}},
    dNdx_t <: Union{Vec{dim}, Tensor{2,dim}}
}
    n_geom_basefuncs = getngeobasefunctions(fv)
    n_func_basefuncs = getnbasefunctions(fv)
    length(x) == n_geom_basefuncs || throw_incompatible_coord_length(length(x), n_geom_basefuncs)
    @boundscheck checkface(fv, face)

    fv.current_face[] = face
    cb = getcurrentface(fv)
    if reinit_values
        n_faces = length(fv.qr.face_rules)
        for face in 1:n_faces, (qp, ξ) in pairs(getpoints(fv.qr, face))
            for basefunc in 1:n_func_basefuncs
                fv.dNdξ[basefunc, qp, face], fv.N[basefunc, qp, face] = shape_gradient_and_value(fv.func_interp, ξ, basefunc)
            end
            for basefunc in 1:n_geom_basefuncs
                fv.dMdξ[basefunc, qp, face], fv.M[basefunc, qp, face] = shape_gradient_and_value(fv.geo_interp, ξ, basefunc)
            end
        end
    end
    @inbounds for (i, w) in pairs(getweights(fv.qr, cb))
        fefv_J = zero(Tensor{2,dim})
        for j in 1:n_geom_basefuncs
            fefv_J += x[j] ⊗ fv.dMdξ[j, i, cb]
        end
        weight_norm = weighted_normal(fefv_J, fv, cb)
        fv.normals[i] = weight_norm / norm(weight_norm)
        detJ = norm(weight_norm)

        detJ > 0.0 || throw_detJ_not_pos(detJ)
        fv.detJdV[i, cb] = detJ * w
        Jinv = inv(fefv_J)
        for j in 1:n_func_basefuncs
            fv.dNdx[j, i, cb] = fv.dNdξ[j, i, cb] ⋅ Jinv
        end
    end
    return nothing
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

nfaces(fv::FaceValues) = size(fv.N, 3)

function checkface(fv::FaceValues, face::Int)
    0 < face <= nfaces(fv) || error("Face index out of range.")
    return nothing
end
