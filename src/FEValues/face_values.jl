# Defines FaceScalarValues and FaceVectorValues and common methods
"""
    FaceScalarValues([::Type{T}], quad_rule::QuadratureRule, func_interpol::Interpolation, [geom_interpol::Interpolation])
    FaceVectorValues([::Type{T}], quad_rule::QuadratureRule, func_interpol::Interpolation, [geom_interpol::Interpolation])

A `FaceValues` object facilitates the process of evaluating values of shape functions, gradients of shape functions,
values of nodal functions, gradients and divergences of nodal functions etc. on the faces of finite elements. There are
two different types of `FaceValues`: `FaceScalarValues` and `FaceVectorValues`. As the names suggest,
`FaceScalarValues` utilizes scalar shape functions and `FaceVectorValues` utilizes vectorial shape functions.
For a scalar field, the `FaceScalarValues` type should be used. For vector field, both subtypes can be used.

!!! note
    The quadrature rule for the face should be given with one dimension lower.
    I.e. for a 3D case, the quadrature rule should be in 2D.

**Arguments:**

* `T`: an optional argument to determine the type the internal data is stored as.
* `quad_rule`: an instance of a [`QuadratureRule`](@ref)
* `func_interpol`: an instance of an [`Interpolation`](@ref) used to interpolate the approximated function
* `geom_interpol`: an optional instance of an [`Interpolation`](@ref) which is used to interpolate the geometry

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
FaceValues, FaceScalarValues, FaceVectorValues

# FaceScalarValues
struct FaceScalarValues{dim,T<:Real,refshape<:AbstractRefShape} <: FaceValues{dim,T,refshape}
    N::Array{T,3}
    dNdx::Array{Vec{dim,T},3}
    dNdξ::Array{Vec{dim,T},3}
    detJdV::Matrix{T}
    normals::Vector{Vec{dim,T}}
    M::Array{T,3}
    dMdξ::Array{Vec{dim,T},3}
    # 'Any' is 'dim-1' here -- this is deliberately abstractly typed. Only qr.weights is
    # accessed in performance critical code so this doesn't seem to be a problem in
    # practice since qr.weights is correctly inferred as Vector{T}, and T is a parameter
    # of the struct.
    qr::QuadratureRule{<:Any,refshape,T}
    current_face::ScalarWrapper{Int}
    # The following fields are deliberately abstract -- they are never used in
    # performance critical code, just stored here for convenience.
    func_interp::Interpolation{dim,refshape}
    geo_interp::Interpolation{dim,refshape}
end

function FaceScalarValues(quad_rule::QuadratureRule, func_interpol::Interpolation,
                          geom_interpol::Interpolation=func_interpol)
    FaceScalarValues(Float64, quad_rule, func_interpol, geom_interpol)
end

function FaceScalarValues(::Type{T}, quad_rule::QuadratureRule{dim_qr,shape}, func_interpol::Interpolation,
        geom_interpol::Interpolation=func_interpol) where {dim_qr,T,shape<:AbstractRefShape}

    @assert getdim(func_interpol) == getdim(geom_interpol)
    @assert getrefshape(func_interpol) == getrefshape(geom_interpol) == shape
    n_qpoints = length(getweights(quad_rule))
    dim = dim_qr + 1

    face_quad_rule = create_face_quad_rule(quad_rule, func_interpol)
    n_faces = length(face_quad_rule)

    # Normals
    normals = zeros(Vec{dim,T}, n_qpoints)

    # Function interpolation
    n_func_basefuncs = getnbasefunctions(func_interpol)
    N =    fill(zero(T)          * T(NaN), n_func_basefuncs, n_qpoints, n_faces)
    dNdx = fill(zero(Vec{dim,T}) * T(NaN), n_func_basefuncs, n_qpoints, n_faces)
    dNdξ = fill(zero(Vec{dim,T}) * T(NaN), n_func_basefuncs, n_qpoints, n_faces)

    # Geometry interpolation
    n_geom_basefuncs = getnbasefunctions(geom_interpol)
    M =    fill(zero(T)          * T(NaN), n_geom_basefuncs, n_qpoints, n_faces)
    dMdξ = fill(zero(Vec{dim,T}) * T(NaN), n_geom_basefuncs, n_qpoints, n_faces)

    for face in 1:n_faces, (qp, ξ) in enumerate(face_quad_rule[face].points)
        for i in 1:n_func_basefuncs
            dNdξ[i, qp, face], N[i, qp, face] = gradient(ξ -> value(func_interpol, i, ξ), ξ, :all)
        end
        for i in 1:n_geom_basefuncs
            dMdξ[i, qp, face], M[i, qp, face] = gradient(ξ -> value(geom_interpol, i, ξ), ξ, :all)
        end
    end

    detJdV = fill(T(NaN), n_qpoints, n_faces)

    FaceScalarValues{dim,T,shape}(N, dNdx, dNdξ, detJdV, normals, M, dMdξ, quad_rule, ScalarWrapper(0), func_interpol, geom_interpol)
end

# FaceVectorValues
struct FaceVectorValues{dim,T<:Real,refshape<:AbstractRefShape,M} <: FaceValues{dim,T,refshape}
    N::Array{Vec{dim,T},3}
    dNdx::Array{Tensor{2,dim,T,M},3}
    dNdξ::Array{Tensor{2,dim,T,M},3}
    detJdV::Matrix{T}
    normals::Vector{Vec{dim,T}}
    M::Array{T,3}
    dMdξ::Array{Vec{dim,T},3}
    # 'Any' is 'dim-1' here -- this is deliberately abstractly typed. Only qr.weights is
    # accessed in performance critical code so this doesn't seem to be a problem in
    # practice since qr.weights is correctly inferred as Vector{T}, and T is a parameter
    # of the struct.
    qr::QuadratureRule{<:Any,refshape,T}
    current_face::ScalarWrapper{Int}
    # The following fields are deliberately abstract -- they are never used in
    # performance critical code, just stored here for convenience.
    func_interp::Interpolation{dim,refshape}
    geo_interp::Interpolation{dim,refshape}
end

function FaceVectorValues(quad_rule::QuadratureRule, func_interpol::Interpolation, geom_interpol::Interpolation=func_interpol)
    FaceVectorValues(Float64, quad_rule, func_interpol, geom_interpol)
end

function FaceVectorValues(::Type{T}, quad_rule::QuadratureRule{dim_qr,shape}, func_interpol::Interpolation,
        geom_interpol::Interpolation=func_interpol) where {dim_qr,T,shape<:AbstractRefShape}

    @assert getdim(func_interpol) == getdim(geom_interpol)
    @assert getrefshape(func_interpol) == getrefshape(geom_interpol) == shape
    n_qpoints = length(getweights(quad_rule))
    dim = dim_qr + 1

    face_quad_rule = create_face_quad_rule(quad_rule, func_interpol)
    n_faces = length(face_quad_rule)

    # Normals
    normals = zeros(Vec{dim,T}, n_qpoints)

    # Function interpolation
    n_func_basefuncs = getnbasefunctions(func_interpol) * dim
    N    = fill(zero(Vec{dim,T})      * T(NaN), n_func_basefuncs, n_qpoints, n_faces)
    dNdx = fill(zero(Tensor{2,dim,T}) * T(NaN), n_func_basefuncs, n_qpoints, n_faces)
    dNdξ = fill(zero(Tensor{2,dim,T}) * T(NaN), n_func_basefuncs, n_qpoints, n_faces)

    # Geometry interpolation
    n_geom_basefuncs = getnbasefunctions(geom_interpol)
    M    = fill(zero(T)          * T(NaN), n_geom_basefuncs, n_qpoints, n_faces)
    dMdξ = fill(zero(Vec{dim,T}) * T(NaN), n_geom_basefuncs, n_qpoints, n_faces)

    for face in 1:n_faces, (qp, ξ) in enumerate(face_quad_rule[face].points)
        basefunc_count = 1
        for basefunc in 1:getnbasefunctions(func_interpol)
            dNdξ_temp, N_temp = gradient(ξ -> value(func_interpol, basefunc, ξ), ξ, :all)
            for comp in 1:dim
                N_comp = zeros(T, dim)
                N_comp[comp] = N_temp
                N[basefunc_count, qp, face] = Vec{dim,T}((N_comp...,))

                dN_comp = zeros(T, dim, dim)
                dN_comp[comp, :] = dNdξ_temp
                dNdξ[basefunc_count, qp, face] = Tensor{2,dim,T}((dN_comp...,))
                basefunc_count += 1
            end
        end
        for basefunc in 1:n_geom_basefuncs
            dMdξ[basefunc, qp, face], M[basefunc, qp, face] = gradient(ξ -> value(geom_interpol, basefunc, ξ), ξ, :all)
        end
    end

    detJdV = fill(T(NaN), n_qpoints, n_faces)
    MM = Tensors.n_components(Tensors.get_base(eltype(dNdx)))

    FaceVectorValues{dim,T,shape,MM}(N, dNdx, dNdξ, detJdV, normals, M, dMdξ, quad_rule, ScalarWrapper(0), func_interpol, geom_interpol)
end

function reinit!(fv::FaceValues{dim}, x::AbstractVector{Vec{dim,T}}, face::Int) where {dim,T}
    n_geom_basefuncs = getngeobasefunctions(fv)
    n_func_basefuncs = getnbasefunctions(fv)
    length(x) == n_geom_basefuncs || throw_incompatible_coord_length(length(x), n_geom_basefuncs)
    @boundscheck checkface(fv, face)

    fv.current_face[] = face
    cb = getcurrentface(fv)

    @inbounds for i in 1:length(fv.qr.weights)
        w = fv.qr.weights[i]
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
    current_face::ScalarWrapper{Int}
end

BCValues(func_interpol::Interpolation, geom_interpol::Interpolation, boundary_type::Type{<:BoundaryIndex} = Ferrite.FaceIndex) =
    BCValues(Float64, func_interpol, geom_interpol, boundary_type)

function BCValues(::Type{T}, func_interpol::Interpolation{dim,refshape}, geom_interpol::Interpolation{dim,refshape}, boundary_type::Type{<:BoundaryIndex} = Ferrite.FaceIndex) where {T,dim,refshape}
    # set up quadrature rules for each face with dof-positions
    # (determined by func_interpol) as the quadrature points
    interpolation_coords = reference_coordinates(func_interpol)

    qrs = QuadratureRule{dim,refshape,T}[]
    faces = boundaryfunction(boundary_type) # faces, edges or vertices
    for face in faces(func_interpol)
        dofcoords = Vec{dim,T}[]
        for facedof in face
            push!(dofcoords, interpolation_coords[facedof])
        end
        qrf = QuadratureRule{dim,refshape,T}(fill(T(NaN), length(dofcoords)), dofcoords) # weights will not be used
        push!(qrs, qrf)
    end

    n_faces = length(qrs)
    n_qpoints = n_faces == 0 ? 0 : length(getweights(qrs[1])) # assume same in all
    n_geom_basefuncs = getnbasefunctions(geom_interpol)
    M =    fill(zero(T)           * T(NaN), n_geom_basefuncs, n_qpoints, n_faces)

    for face in 1:n_faces, (qp, ξ) in enumerate(qrs[face].points)
        for i in 1:n_geom_basefuncs
            M[i, qp, face] = value(geom_interpol, i, ξ)
        end
    end

    BCValues{T}(M, ScalarWrapper(0))
end

getnquadpoints(bcv::BCValues) = size(bcv.M, 2)
function spatial_coordinate(bcv::BCValues, q_point::Int, xh::AbstractVector{Vec{dim,T}}) where {dim,T}
    n_base_funcs = size(bcv.M, 1)
    length(xh) == n_base_funcs || throw_incompatible_coord_length(length(xh), n_base_funcs)
    x = zero(Vec{dim,T})
    face = bcv.current_face[]
    @inbounds for i in 1:n_base_funcs
        x += bcv.M[i,q_point,face] * xh[i] # geometric_value(fe_v, q_point, i) * xh[i]
    end
    return x
end

nfaces(fv) = size(fv.N, 3)

function checkface(fv::FaceValues, face::Int)
    0 < face <= nfaces(fv) || error("Face index out of range.")
    return nothing
end
