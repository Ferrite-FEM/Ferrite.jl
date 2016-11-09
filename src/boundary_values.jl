# Defines BoundaryScalarValues and BoundaryVectorValues and common methods
"""
A `BoundaryValues` object facilitates the process of evaluating values of shape functions, gradients of shape functions,
values of nodal functions, gradients and divergences of nodal functions etc. on the finite element boundary. There are
two different types of `BoundaryValues`: `BoundaryScalarValues` and `BoundaryVectorValues`. As the names suggest,
`BoundaryScalarValues` utilizes scalar shape functions and `BoundaryVectorValues` utilizes vectorial shape functions.
For a scalar field, the `BoundaryScalarValues` type should be used. For vector field, both subtypes can be used.

**Constructors:**

Note: The quadrature rule for the boundary should be given with one dimension lower. I.e. for a 3D case, the quadrature rule
should be in 2D.

```julia
BoundaryScalarValues([::Type{T}], quad_rule::QuadratureRule, func_interpol::Interpolation, [geom_interpol::Interpolation])
BoundaryVectorValues([::Type{T}], quad_rule::QuadratureRule, func_interpol::Interpolation, [geom_interpol::Interpolation])
```

**Arguments:**

* `T`: an optional argument to determine the type the internal data is stored as.
* `quad_rule`: an instance of a [`QuadratureRule`](@ref)
* `func_interpol`: an instance of an [`Interpolation`](@ref) used to interpolate the approximated function
* `geom_interpol`: an optional instance of an [`Interpolation`](@ref) which is used to interpolate the geometry

**Common methods:**

* [`reinit!`](@ref)
* [`getboundarynumber`](@ref)
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
BoundaryValues

# BoundaryScalarValues
immutable BoundaryScalarValues{dim, T <: Real, FIP <: Interpolation, GIP <: Interpolation, shape <: AbstractRefShape} <: BoundaryValues{dim, T, FIP, GIP}
    N::Array{T, 3}
    dNdx::Array{Vec{dim, T}, 3}
    dNdξ::Array{Vec{dim, T}, 3}
    detJdV::Matrix{T}
    quad_rule::Vector{QuadratureRule{dim, shape, T}}
    func_interpol::FIP
    M::Array{T, 3}
    dMdξ::Array{Vec{dim, T}, 3}
    geom_interpol::GIP
    current_boundary::Ref{Int}
end

BoundaryScalarValues{dim_qr, FIP <: Interpolation, GIP <: Interpolation}(quad_rule::QuadratureRule{dim_qr}, func_interpol::FIP, geom_interpol::GIP=func_interpol) =
    BoundaryScalarValues(Float64, quad_rule, func_interpol, geom_interpol)

getnbasefunctions(bvv::BoundaryScalarValues) = getnbasefunctions(bvv.func_interpol)

function BoundaryScalarValues{dim_qr, T, FIP <: Interpolation, GIP <: Interpolation, shape <: AbstractRefShape}(
                        ::Type{T}, quad_rule::QuadratureRule{dim_qr, shape}, func_interpol::FIP, geom_interpol::GIP=func_interpol)

    @assert getdim(func_interpol) == getdim(geom_interpol)
    @assert getrefshape(func_interpol) == getrefshape(geom_interpol) == shape
    n_qpoints = length(getweights(quad_rule))
    dim = dim_qr + 1

    boundary_quad_rule = create_boundary_quad_rule(quad_rule, func_interpol)
    n_bounds = length(boundary_quad_rule)

    # Function interpolation
    n_func_basefuncs = getnbasefunctions(func_interpol)
    N =    zeros(T, n_func_basefuncs, n_qpoints, n_bounds)
    dNdx = zeros(Vec{dim, T}, n_func_basefuncs, n_qpoints, n_bounds)
    dNdξ = zeros(Vec{dim, T}, n_func_basefuncs, n_qpoints, n_bounds)

    # Geometry interpolation
    n_geom_basefuncs = getnbasefunctions(geom_interpol)
    M =    zeros(T, n_geom_basefuncs, n_qpoints, n_bounds)
    dMdξ = zeros(Vec{dim, T}, n_geom_basefuncs, n_qpoints, n_bounds)

    for k in 1:n_bounds, (i, ξ) in enumerate(boundary_quad_rule[k].points)
        value!(func_interpol, view(N, :, i, k), ξ)
        derivative!(func_interpol, view(dNdξ, :, i, k), ξ)
        value!(geom_interpol, view(M, :, i, k), ξ)
        derivative!(geom_interpol, view(dMdξ, :, i, k), ξ)
    end

    detJdV = zeros(T, n_qpoints, n_bounds)

    BoundaryScalarValues(N, dNdx, dNdξ, detJdV, boundary_quad_rule, func_interpol, M, dMdξ, geom_interpol, Ref(0))
end

# BoundaryVectorValues
immutable BoundaryVectorValues{dim, T <: Real, FIP <: Interpolation, GIP <: Interpolation, shape <: AbstractRefShape, M} <: BoundaryValues{dim, T, FIP, GIP}
    N::Array{Vec{dim, T}, 3}
    dNdx::Array{Tensor{2, dim, T, M}, 3}
    dNdξ::Array{Tensor{2, dim, T, M}, 3}
    detJdV::Matrix{T}
    quad_rule::Vector{QuadratureRule{dim, shape, T}}
    func_interpol::FIP
    M::Array{T, 3}
    dMdξ::Array{Vec{dim, T}, 3}
    geom_interpol::GIP
    current_boundary::Ref{Int}
end

BoundaryVectorValues{dim_qr, FIP <: Interpolation, GIP <: Interpolation}(quad_rule::QuadratureRule{dim_qr}, func_interpol::FIP, geom_interpol::GIP=func_interpol) =
    BoundaryVectorValues(Float64, quad_rule, func_interpol, geom_interpol)

getnbasefunctions{dim}(bvv::BoundaryVectorValues{dim}) = getnbasefunctions(bvv.func_interpol) * dim

function BoundaryVectorValues{dim_qr, T, FIP <: Interpolation, GIP <: Interpolation, shape <: AbstractRefShape}(
                        ::Type{T}, quad_rule::QuadratureRule{dim_qr, shape}, func_interpol::FIP, geom_interpol::GIP=func_interpol)

    @assert getdim(func_interpol) == getdim(geom_interpol)
    @assert getrefshape(func_interpol) == getrefshape(geom_interpol) == shape
    n_qpoints = length(getweights(quad_rule))
    dim = dim_qr + 1

    boundary_quad_rule = create_boundary_quad_rule(quad_rule, func_interpol)
    n_bounds = length(boundary_quad_rule)

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
        for (i, ξ) in enumerate(boundary_quad_rule[k].points)
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

    BoundaryVectorValues(N, dNdx, dNdξ, detJdV, boundary_quad_rule, func_interpol, M, dMdξ, geom_interpol, Ref(0))
end

function reinit!{dim, T}(bv::BoundaryValues{dim}, x::Vector{Vec{dim, T}}, boundary::Int)
    n_geom_basefuncs = getnbasefunctions(getgeometryinterpolation(bv))
    n_func_basefuncs = getnbasefunctions(getfunctioninterpolation(bv))
    @assert length(x) == n_geom_basefuncs
    isa(bv, BoundaryVectorValues) && (n_func_basefuncs *= dim)

    bv.current_boundary[] = boundary
    cb = getcurrentboundary(bv)

    @inbounds for i in 1:length(getpoints(bv.quad_rule[cb]))
        w = getweights(bv.quad_rule[cb])[i]
        febv_J = zero(Tensor{2, dim})
        for j in 1:n_geom_basefuncs
            febv_J += x[j] ⊗ bv.dMdξ[j, i, cb]
        end
        Jinv = inv(febv_J)
        for j in 1:n_func_basefuncs
            bv.dNdx[j, i, cb] = bv.dNdξ[j, i, cb] ⋅ Jinv
        end
        detJ = detJ_boundary(febv_J, getgeometryinterpolation(bv), cb)
        detJ <= 0.0 && throw(ArgumentError("detJ is not positive: detJ = $(detJ)"))
        bv.detJdV[i, cb] = detJ * w
    end
end

"""
The current active boundary of the `BoundaryValues` type.

    getcurrentboundary(bv::BoundaryValues)

** Arguments **

* `bv`: the `BoundaryValues` object

** Results **

* `::Int`: the current active boundary (from last `reinit!`).

"""
getcurrentboundary(bv::BoundaryValues) = bv.current_boundary[]

"""
The boundary number for a cell, typically used to get the boundary number which is needed
to `reinit!` a `BoundaryValues` object for  boundary integration

    getboundarynumber(boundary_nodes, cell_nodes, ip::Interpolation)

** Arguments **

* `boundary_nodes`: the node numbers of the nodes on the boundary of the cell
* `cell_nodes`: the node numbers of the cell
* `ip`: the `Interpolation` for the cell

** Results **

* `::Int`: the corresponding boundary
"""
function getboundarynumber(boundary_nodes::Vector{Int}, cell_nodes::Vector{Int}, ip::Interpolation)
    @assert length(boundary_nodes) == getnboundarynodes(ip)
    @assert length(cell_nodes) == getnbasefunctions(ip)

    tmp = zeros(boundary_nodes)
    for i in 1:length(boundary_nodes)
        tmp[i] = findfirst(j -> j == boundary_nodes[i], cell_nodes)
    end

    if 0 in tmp
        throw(ArgumentError("at least one boundary node: $boundary_nodes not in cell nodes: $cell_nodes"))
    end
    sort!(tmp)
    boundary_nodes_sorted = ntuple(i -> tmp[i], Val{getnboundarynodes(ip)})
    for (i, boundary) in enumerate(getboundarylist(ip))
        boundary_nodes_sorted == boundary && return i
    end

    throw(ArgumentError("invalid node numbers for boundary"))
end
