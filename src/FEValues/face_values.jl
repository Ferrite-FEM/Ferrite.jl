# Defines FaceScalarValues and FaceVectorValues and common methods
"""
A `FaceValues` object facilitates the process of evaluating values of shape functions, gradients of shape functions,
values of nodal functions, gradients and divergences of nodal functions etc. on the faces of finite elements. There are
two different types of `FaceValues`: `FaceScalarValues` and `FaceVectorValues`. As the names suggest,
`FaceScalarValues` utilizes scalar shape functions and `FaceVectorValues` utilizes vectorial shape functions.
For a scalar field, the `FaceScalarValues` type should be used. For vector field, both subtypes can be used.

**Constructors:**

Note: The quadrature rule for the face should be given with one dimension lower. I.e. for a 3D case, the quadrature rule
should be in 2D.

```julia
FaceScalarValues([::Type{T}], quad_rule::QuadratureRule, func_interpol::Interpolation, [geom_interpol::Interpolation])
FaceVectorValues([::Type{T}], quad_rule::QuadratureRule, func_interpol::Interpolation, [geom_interpol::Interpolation])
```

**Arguments:**

* `T`: an optional argument to determine the type the internal data is stored as.
* `quad_rule`: an instance of a [`QuadratureRule`](@ref)
* `func_interpol`: an instance of an [`Interpolation`](@ref) used to interpolate the approximated function
* `geom_interpol`: an optional instance of an [`Interpolation`](@ref) which is used to interpolate the geometry

**Common methods:**

* [`reinit!`](@ref)
* [`getfacenumber`](@ref)
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

# FaceScalarValues
immutable FaceScalarValues{dim, T <: Real, refshape <: AbstractRefShape} <: FaceValues{dim, T, refshape}
    N::Array{T, 3}
    dNdx::Array{Vec{dim, T}, 3}
    dNdξ::Array{Vec{dim, T}, 3}
    detJdV::Matrix{T}
    normals::Vector{Vec{dim, T}}
    M::Array{T, 3}
    dMdξ::Array{Vec{dim, T}, 3}
    qr_weights::Vector{T}
    current_face::ScalarWrapper{Int}
end

function FaceScalarValues(quad_rule::QuadratureRule, func_interpol::Interpolation,
                          geom_interpol::Interpolation=func_interpol)
    FaceScalarValues(Float64, quad_rule, func_interpol, geom_interpol)
end

function FaceScalarValues{dim, dim_qr, T, shape <: AbstractRefShape}(::Type{T}, quad_rule::QuadratureRule{dim_qr, shape},
        func_interpol::Interpolation{dim, shape}, geom_interpol::Interpolation{dim, shape}=func_interpol)

    n_qpoints = length(getweights(quad_rule))

    face_quad_rule = create_face_quad_rule(quad_rule, func_interpol)
    n_faces = length(face_quad_rule)

    # Normals
    normals = zeros(Vec{dim, T}, n_qpoints)

    # Function interpolation
    n_func_basefuncs = getnbasefunctions(func_interpol)
    N =    zeros(T, n_func_basefuncs, n_qpoints, n_faces)
    dNdx = zeros(Vec{dim, T}, n_func_basefuncs, n_qpoints, n_faces)
    dNdξ = zeros(Vec{dim, T}, n_func_basefuncs, n_qpoints, n_faces)

    # Geometry interpolation
    n_geom_basefuncs = getnbasefunctions(geom_interpol)
    M =    zeros(T, n_geom_basefuncs, n_qpoints, n_faces)
    dMdξ = zeros(Vec{dim, T}, n_geom_basefuncs, n_qpoints, n_faces)

    for k in 1:n_faces, (i, ξ) in enumerate(face_quad_rule[k].points)
        value!(func_interpol, view(N, :, i, k), ξ)
        derivative!(func_interpol, view(dNdξ, :, i, k), ξ)
        value!(geom_interpol, view(M, :, i, k), ξ)
        derivative!(geom_interpol, view(dMdξ, :, i, k), ξ)
    end

    detJdV = zeros(T, n_qpoints, n_faces)

    FaceScalarValues{dim, T, shape}(N, dNdx, dNdξ, detJdV, normals, M, dMdξ, quad_rule.weights, ScalarWrapper(0))
end

# FaceVectorValues
immutable FaceVectorValues{dim, T <: Real, refshape <: AbstractRefShape, M} <: FaceValues{dim, T, refshape}
    N::Array{Vec{dim, T}, 3}
    dNdx::Array{Tensor{2, dim, T, M}, 3}
    dNdξ::Array{Tensor{2, dim, T, M}, 3}
    detJdV::Matrix{T}
    normals::Vector{Vec{dim, T}}
    M::Array{T, 3}
    dMdξ::Array{Vec{dim, T}, 3}
    qr_weights::Vector{T}
    current_face::ScalarWrapper{Int}
end

function FaceVectorValues(quad_rule::QuadratureRule, func_interpol::Interpolation, geom_interpol::Interpolation=func_interpol)
    FaceVectorValues(Float64, quad_rule, func_interpol, geom_interpol)
end

function FaceVectorValues{QR <: QuadratureRule}(quad_rules::Vector{QR}, func_interpol::Interpolation, geom_interpol::Interpolation=func_interpol)
    FaceVectorValues(Float64, quad_rules, func_interpol, geom_interpol)
end


function FaceVectorValues{dim_qr}(quad_rule::QuadratureRule{dim_qr}, func_interpol::Interpolation, geom_interpol::Interpolation=func_interpol)
    face_quad_rule = create_face_quad_rule(quad_rule, func_interpol)
    FaceVectorValues(Float64, face_quad_rule, func_interpol, geom_interpol)
end

function FaceVectorValues{T, dim, refshape}(::Type{T}, face_quad_rule::Vector, # TODO: Fix type
        func_interpol::Interpolation{dim, refshape}, geom_interpol::Interpolation=func_interpol)
    n_qpoints = length(getweights(face_quad_rule[1]))

    n_faces = length(face_quad_rule)

    # Normals
    normals = zeros(Vec{dim, T}, n_qpoints)

    # Function interpolation
    n_func_basefuncs = getnbasefunctions(func_interpol) * dim
    N = zeros(Vec{dim, T}, n_func_basefuncs, n_qpoints, n_faces)
    dNdx = [zero(Tensor{2, dim, T}) for i in 1:n_func_basefuncs, j in 1:n_qpoints, k in 1:n_faces]
    dNdξ = [zero(Tensor{2, dim, T}) for i in 1:n_func_basefuncs, j in 1:n_qpoints, k in 1:n_faces]

    # Geometry interpolation
    n_geom_basefuncs = getnbasefunctions(geom_interpol)
    M = zeros(T, n_geom_basefuncs, n_qpoints, n_faces)
    dMdξ = zeros(Vec{dim, T}, n_geom_basefuncs, n_qpoints, n_faces)

    for k in 1:n_faces
        N_temp = zeros(getnbasefunctions(func_interpol))
        dNdξ_temp = zeros(Vec{dim, T}, getnbasefunctions(func_interpol))
        for (i, ξ) in enumerate(face_quad_rule[k].points)
            value!(func_interpol, N_temp, ξ)
            derivative!(func_interpol, dNdξ_temp, ξ)
            basefunc_count = 1
            for basefunc in 1:getnbasefunctions(func_interpol)
                for comp in 1:dim
                    N_comp = zeros(T, dim)
                    N_comp[comp] = N_temp[basefunc]
                    N[basefunc_count, i, k] = Vec{dim, T}((N_comp...))

                    dN_comp = zeros(T, dim, dim)
                    dN_comp[comp, :] = dNdξ_temp[basefunc]
                    dNdξ[basefunc_count, i, k] = Tensor{2, dim, T}((dN_comp...))
                    basefunc_count += 1
                end
            end
        value!(geom_interpol, view(M, :, i, k), ξ)
        derivative!(geom_interpol, view(dMdξ, :, i, k), ξ)
        end
    end

    detJdV = zeros(T, n_qpoints, n_faces)
    MM = Tensors.n_components(Tensors.get_base(eltype(dNdx)))


    FaceVectorValues{dim, T, refshape, MM}(N, dNdx, dNdξ, detJdV, normals, M, dMdξ, face_quad_rule[1].weights, ScalarWrapper(0))
end

function reinit!{dim, T}(fv::FaceValues{dim}, x::AbstractVector{Vec{dim, T}}, face::Int)
    n_geom_basefuncs = getngeobasefunctions(fv)
    n_func_basefuncs = getn_scalarbasefunctions(fv)
    @assert length(x) == n_geom_basefuncs
    isa(fv, FaceVectorValues) && (n_func_basefuncs *= dim)


    fv.current_face[] = face
    cb = getcurrentface(fv)

    @inbounds for i in 1:length(fv.qr_weights)
        w = fv.qr_weights[i]
        fefv_J = zero(Tensor{2, dim})
        for j in 1:n_geom_basefuncs
            fefv_J += x[j] ⊗ fv.dMdξ[j, i, cb]
        end
        weight_norm = weighted_normal(fefv_J, fv, cb)
        fv.normals[i] = weight_norm / norm(weight_norm)
        detJ = norm(weight_norm)

        detJ > 0.0 || throw(ArgumentError("det(J) is not positive: det(J) = $(detJ)"))
        fv.detJdV[i, cb] = detJ * w
        Jinv = inv(fefv_J)
        for j in 1:n_func_basefuncs
            fv.dNdx[j, i, cb] = fv.dNdξ[j, i, cb] ⋅ Jinv
        end
    end
end

"""
The current active face of the `FaceValues` type.

    getcurrentface(fv::FaceValues)

** Arguments **

* `fv`: the `FaceValues` object

** Results **

* `::Int`: the current active face (from last `reinit!`).

"""
getcurrentface(fv::FaceValues) = fv.current_face[]

"""
The normal at the quadrature point `qp` for the active face of the `FaceValues` type.

    getnormal(fv::FaceValues, qp::Int)

** Arguments **

* `fv`: the `FaceValues` object
* `qp`: the quadrature point

** Results **

* `::Vec{dim}`: the normal of the current active face (from last `reinit!`).

"""
getnormal(fv::FaceValues, qp::Int) = fv.normals[qp]
