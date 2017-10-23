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
struct FaceScalarValues{dim, T <: Real, refshape <: AbstractRefShape} <: FaceValues{dim, T, refshape}
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

function FaceScalarValues(::Type{T}, quad_rule::QuadratureRule{dim_qr, shape}, func_interpol::Interpolation,
        geom_interpol::Interpolation=func_interpol) where {dim_qr, T, shape <: AbstractRefShape}

    @assert getdim(func_interpol) == getdim(geom_interpol)
    @assert getrefshape(func_interpol) == getrefshape(geom_interpol) == shape
    n_qpoints = length(getweights(quad_rule))
    dim = dim_qr + 1

    face_quad_rule = create_face_quad_rule(quad_rule, func_interpol)
    n_faces = length(face_quad_rule)

    # Normals
    normals = zeros(Vec{dim, T}, n_qpoints)

    # Function interpolation
    n_func_basefuncs = getnbasefunctions(func_interpol)
    N =    fill(zero(T)           * T(NaN), n_func_basefuncs, n_qpoints, n_faces)
    dNdx = fill(zero(Vec{dim, T}) * T(NaN), n_func_basefuncs, n_qpoints, n_faces)
    dNdξ = fill(zero(Vec{dim, T}) * T(NaN), n_func_basefuncs, n_qpoints, n_faces)

    # Geometry interpolation
    n_geom_basefuncs = getnbasefunctions(geom_interpol)
    M =    fill(zero(T)           * T(NaN), n_geom_basefuncs, n_qpoints, n_faces)
    dMdξ = fill(zero(Vec{dim, T}) * T(NaN), n_geom_basefuncs, n_qpoints, n_faces)

    for face in 1:n_faces, (qp, ξ) in enumerate(face_quad_rule[face].points)
        for i in 1:n_func_basefuncs
            dNdξ[i, qp, face], N[i, qp, face] = gradient(ξ -> value(func_interpol, i, ξ), ξ, :all)
        end
        for i in 1:n_geom_basefuncs
            dMdξ[i, qp, face], M[i, qp, face] = gradient(ξ -> value(geom_interpol, i, ξ), ξ, :all)
        end
    end

    detJdV = fill(T(NaN), n_qpoints, n_faces)

    FaceScalarValues{dim, T, shape}(N, dNdx, dNdξ, detJdV, normals, M, dMdξ, quad_rule.weights, ScalarWrapper(0))
end

# FaceVectorValues
struct FaceVectorValues{dim, T <: Real, refshape <: AbstractRefShape, M} <: FaceValues{dim, T, refshape}
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

function FaceVectorValues(::Type{T}, quad_rule::QuadratureRule{dim_qr, shape}, func_interpol::Interpolation,
        geom_interpol::Interpolation=func_interpol) where {dim_qr, T, shape <: AbstractRefShape}

    @assert getdim(func_interpol) == getdim(geom_interpol)
    @assert getrefshape(func_interpol) == getrefshape(geom_interpol) == shape
    n_qpoints = length(getweights(quad_rule))
    dim = dim_qr + 1

    face_quad_rule = create_face_quad_rule(quad_rule, func_interpol)
    n_faces = length(face_quad_rule)

    # Normals
    normals = zeros(Vec{dim, T}, n_qpoints)

    # Function interpolation
    n_func_basefuncs = getnbasefunctions(func_interpol) * dim
    N    = fill(zero(Vec{dim, T})       * T(NaN), n_func_basefuncs, n_qpoints, n_faces)
    dNdx = fill(zero(Tensor{2, dim, T}) * T(NaN), n_func_basefuncs, n_qpoints, n_faces)
    dNdξ = fill(zero(Tensor{2, dim, T}) * T(NaN), n_func_basefuncs, n_qpoints, n_faces)

    # Geometry interpolation
    n_geom_basefuncs = getnbasefunctions(geom_interpol)
    M    = fill(zero(T)           * T(NaN), n_geom_basefuncs, n_qpoints, n_faces)
    dMdξ = fill(zero(Vec{dim, T}) * T(NaN), n_geom_basefuncs, n_qpoints, n_faces)

    for face in 1:n_faces, (qp, ξ) in enumerate(face_quad_rule[face].points)
        basefunc_count = 1
        for basefunc in 1:getnbasefunctions(func_interpol)
            dNdξ_temp, N_temp = gradient(ξ -> value(func_interpol, basefunc, ξ), ξ, :all)
            for comp in 1:dim
                N_comp = zeros(T, dim)
                N_comp[comp] = N_temp
                N[basefunc_count, qp, face] = Vec{dim, T}((N_comp...))

                dN_comp = zeros(T, dim, dim)
                dN_comp[comp, :] = dNdξ_temp
                dNdξ[basefunc_count, qp, face] = Tensor{2, dim, T}((dN_comp...))
                basefunc_count += 1
            end
        end
        for basefunc in 1:n_geom_basefuncs
            dMdξ[basefunc, qp, face], M[basefunc, qp, face] = gradient(ξ -> value(geom_interpol, basefunc, ξ), ξ, :all)
        end
    end

    detJdV = fill(T(NaN), n_qpoints, n_faces)
    MM = Tensors.n_components(Tensors.get_base(eltype(dNdx)))

    FaceVectorValues{dim, T, shape, MM}(N, dNdx, dNdξ, detJdV, normals, M, dMdξ, quad_rule.weights, ScalarWrapper(0))
end

function reinit!(fv::FaceValues{dim}, x::AbstractVector{Vec{dim, T}}, face::Int) where {dim, T}
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

"""
The face number for a cell, typically used to get the face number which is needed
to `reinit!` a `FaceValues` object for  face integration

    getfacenumber(face_nodes, cell_nodes, ip::Interpolation)

** Arguments **

* `face_nodes`: the node numbers of the nodes on the face of the cell
* `cell_nodes`: the node numbers of the cell
* `ip`: the `Interpolation` for the cell

** Results **

* `::Int`: the corresponding face
"""
function getfacenumber(face_nodes::Vector{Int}, cell_nodes::Vector{Int}, ip::Interpolation)
    @assert length(face_nodes) == getnfacenodes(ip)
    @assert length(cell_nodes) == getnbasefunctions(ip)

    tmp = zeros(face_nodes)
    for i in 1:length(face_nodes)
        tmp[i] = findfirst(j -> j == face_nodes[i], cell_nodes)
    end

    if 0 in tmp
        throw(ArgumentError("at least one face node: $face_nodes not in cell nodes: $cell_nodes"))
    end
    sort!(tmp)
    face_nodes_sorted = ntuple(i -> tmp[i], Val{getnfacenodes(ip)})
    for (i, face) in enumerate(getfacelist(ip))
        face_nodes_sorted == face && return i
    end

    throw(ArgumentError("invalid node numbers for face"))
end
