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
* [`getquadrule`](@ref)
* [`getfunctioninterpolation`](@ref)
* [`getgeometryinterpolation`](@ref)
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
immutable FaceScalarValues{dim, T <: Real, FIP <: Interpolation, GIP <: Interpolation, shape <: AbstractRefShape} <: FaceValues{dim, T, FIP, GIP}
    N::Array{T, 3}
    dNdx::Array{Vec{dim, T}, 3}
    dNdξ::Array{Vec{dim, T}, 3}
    detJdV::Matrix{T}
    quad_rule::Vector{QuadratureRule{dim, shape, T}}
    func_interpol::FIP
    M::Array{T, 3}
    dMdξ::Array{Vec{dim, T}, 3}
    geom_interpol::GIP
    current_face::Ref{Int}
end

FaceScalarValues{dim_qr, FIP <: Interpolation, GIP <: Interpolation}(quad_rule::QuadratureRule{dim_qr}, func_interpol::FIP, geom_interpol::GIP=func_interpol) =
    FaceScalarValues(Float64, quad_rule, func_interpol, geom_interpol)

getnbasefunctions(bvv::FaceScalarValues) = getnbasefunctions(bvv.func_interpol)

function FaceScalarValues{dim_qr, T, FIP <: Interpolation, GIP <: Interpolation, shape <: AbstractRefShape}(
                        ::Type{T}, quad_rule::QuadratureRule{dim_qr, shape}, func_interpol::FIP, geom_interpol::GIP=func_interpol)

    @assert getdim(func_interpol) == getdim(geom_interpol)
    @assert getrefshape(func_interpol) == getrefshape(geom_interpol) == shape
    n_qpoints = length(getweights(quad_rule))
    dim = dim_qr + 1

    face_quad_rule = create_face_quad_rule(quad_rule, func_interpol)
    n_bounds = length(face_quad_rule)

    # Function interpolation
    n_func_basefuncs = getnbasefunctions(func_interpol)
    N =    zeros(T, n_func_basefuncs, n_qpoints, n_bounds)
    dNdx = zeros(Vec{dim, T}, n_func_basefuncs, n_qpoints, n_bounds)
    dNdξ = zeros(Vec{dim, T}, n_func_basefuncs, n_qpoints, n_bounds)

    # Geometry interpolation
    n_geom_basefuncs = getnbasefunctions(geom_interpol)
    M =    zeros(T, n_geom_basefuncs, n_qpoints, n_bounds)
    dMdξ = zeros(Vec{dim, T}, n_geom_basefuncs, n_qpoints, n_bounds)

    for k in 1:n_bounds, (i, ξ) in enumerate(face_quad_rule[k].points)
        value!(func_interpol, view(N, :, i, k), ξ)
        derivative!(func_interpol, view(dNdξ, :, i, k), ξ)
        value!(geom_interpol, view(M, :, i, k), ξ)
        derivative!(geom_interpol, view(dMdξ, :, i, k), ξ)
    end

    detJdV = zeros(T, n_qpoints, n_bounds)

    FaceScalarValues(N, dNdx, dNdξ, detJdV, face_quad_rule, func_interpol, M, dMdξ, geom_interpol, Ref(0))
end

# FaceVectorValues
immutable FaceVectorValues{dim, T <: Real, FIP <: Interpolation, GIP <: Interpolation, shape <: AbstractRefShape, M} <: FaceValues{dim, T, FIP, GIP}
    N::Array{Vec{dim, T}, 3}
    dNdx::Array{Tensor{2, dim, T, M}, 3}
    dNdξ::Array{Tensor{2, dim, T, M}, 3}
    detJdV::Matrix{T}
    quad_rule::Vector{QuadratureRule{dim, shape, T}}
    func_interpol::FIP
    M::Array{T, 3}
    dMdξ::Array{Vec{dim, T}, 3}
    geom_interpol::GIP
    current_face::Ref{Int}
end

FaceVectorValues{dim_qr, FIP <: Interpolation, GIP <: Interpolation}(quad_rule::QuadratureRule{dim_qr}, func_interpol::FIP, geom_interpol::GIP=func_interpol) =
    FaceVectorValues(Float64, quad_rule, func_interpol, geom_interpol)

getnbasefunctions{dim}(bvv::FaceVectorValues{dim}) = getnbasefunctions(bvv.func_interpol) * dim

function FaceVectorValues{dim_qr, T, FIP <: Interpolation, GIP <: Interpolation, shape <: AbstractRefShape}(
                        ::Type{T}, quad_rule::QuadratureRule{dim_qr, shape}, func_interpol::FIP, geom_interpol::GIP=func_interpol)

    @assert getdim(func_interpol) == getdim(geom_interpol)
    @assert getrefshape(func_interpol) == getrefshape(geom_interpol) == shape
    n_qpoints = length(getweights(quad_rule))
    dim = dim_qr + 1

    face_quad_rule = create_face_quad_rule(quad_rule, func_interpol)
    n_bounds = length(face_quad_rule)

    # Function interpolation
    n_func_basefuncs = getnbasefunctions(func_interpol) * dim
    N = zeros(Vec{dim, T}, n_func_basefuncs, n_qpoints, n_bounds)
    dNdx = [zero(Tensor{2, dim, T}) for i in 1:n_func_basefuncs, j in 1:n_qpoints, k in 1:n_bounds]
    dNdξ = [zero(Tensor{2, dim, T}) for i in 1:n_func_basefuncs, j in 1:n_qpoints, k in 1:n_bounds]

    # Geometry interpolation
    n_geom_basefuncs = getnbasefunctions(geom_interpol)
    M = zeros(T, n_geom_basefuncs, n_qpoints, n_bounds)
    dMdξ = zeros(Vec{dim, T}, n_geom_basefuncs, n_qpoints, n_bounds)

    for k in 1:n_bounds
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

    detJdV = zeros(T, n_qpoints, n_bounds)

    FaceVectorValues(N, dNdx, dNdξ, detJdV, face_quad_rule, func_interpol, M, dMdξ, geom_interpol, Ref(0))
end

function reinit!{dim, T}(bv::FaceValues{dim}, x::AbstractVector{Vec{dim, T}}, face::Int)
    n_geom_basefuncs = getnbasefunctions(getgeometryinterpolation(bv))
    n_func_basefuncs = getnbasefunctions(getfunctioninterpolation(bv))
    @assert length(x) == n_geom_basefuncs
    isa(bv, FaceVectorValues) && (n_func_basefuncs *= dim)

    bv.current_face[] = face
    cb = getcurrentface(bv)

    @inbounds for i in 1:length(getpoints(bv.quad_rule[cb]))
        w = getweights(bv.quad_rule[cb])[i]
        febv_J = zero(Tensor{2, dim})
        for j in 1:n_geom_basefuncs
            febv_J += x[j] ⊗ bv.dMdξ[j, i, cb]
        end
        detJ = detJ_face(febv_J, getgeometryinterpolation(bv), cb)
        detJ > 0.0 || throw(ArgumentError("detJ is not positive: detJ = $(detJ)"))
        bv.detJdV[i, cb] = detJ * w
        Jinv = inv(febv_J)
        for j in 1:n_func_basefuncs
            bv.dNdx[j, i, cb] = bv.dNdξ[j, i, cb] ⋅ Jinv
        end
    end
end

"""
The current active face of the `FaceValues` type.

    getcurrentface(bv::FaceValues)

** Arguments **

* `bv`: the `FaceValues` object

** Results **

* `::Int`: the current active face (from last `reinit!`).

"""
getcurrentface(bv::FaceValues) = bv.current_face[]

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
