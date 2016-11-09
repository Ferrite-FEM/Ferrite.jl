# Defines CellScalarValues and CellVectorValues and common methods
"""
A `CellValues` object facilitates the process of evaluating values shape functions, gradients of shape functions,
values of nodal functions, gradients and divergences of nodal functions etc. in the finite element cell. There are
two different types of `CellValues`: `CellScalarValues` and `CellVectorValues`. As the names suggest, `CellScalarValues`
utilizes scalar shape functions and `CellVectorValues` utilizes vectorial shape functions. For a scalar field, the
`CellScalarValues` type should be used. For vector field, both subtypes can be used.

**Constructors:**

```julia
CellScalarValues([::Type{T}], quad_rule::QuadratureRule, func_interpol::Interpolation, [geom_interpol::Interpolation])
CellVectorValues([::Type{T}], quad_rule::QuadratureRule, func_interpol::Interpolation, [geom_interpol::Interpolation])
```

**Arguments:**

* `T`: an optional argument to determine the type the internal data is stored as.
* `quad_rule`: an instance of a [`QuadratureRule`](@ref)
* `func_interpol`: an instance of an [`Interpolation`](@ref) used to interpolate the approximated function
* `geom_interpol`: an optional instance of a [`Interpolation`](@ref) which is used to interpolate the geometry

**Common methods:**

* [`reinit!`](@ref)
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
CellValues

# CellScalarValues
immutable CellScalarValues{dim, T <: Real, FIP <: Interpolation, GIP <: Interpolation, shape <: AbstractRefShape} <: CellValues{dim, T, FIP, GIP}
    N::Matrix{T}
    dNdx::Matrix{Vec{dim, T}}
    dNdξ::Matrix{Vec{dim, T}}
    detJdV::Vector{T}
    quad_rule::QuadratureRule{dim, shape, T}
    func_interpol::FIP
    M::Matrix{T}
    dMdξ::Matrix{Vec{dim, T}}
    geom_interpol::GIP
end

CellScalarValues{dim, FIP <: Interpolation, GIP <: Interpolation}(quad_rule::QuadratureRule{dim}, func_interpol::FIP, geom_interpol::GIP=func_interpol) =
    CellScalarValues(Float64, quad_rule, func_interpol, geom_interpol)

getnbasefunctions(cv::CellScalarValues) = getnbasefunctions(cv.func_interpol)

function CellScalarValues{dim, T, FIP <: Interpolation, GIP <: Interpolation, shape <: AbstractRefShape}(
    ::Type{T}, quad_rule::QuadratureRule{dim, shape}, func_interpol::FIP, geom_interpol::GIP=func_interpol)

    @assert getdim(func_interpol) == getdim(geom_interpol)
    @assert getrefshape(func_interpol) == getrefshape(geom_interpol) == shape
    n_qpoints = length(getweights(quad_rule))

    # Function interpolation
    n_func_basefuncs = getnbasefunctions(func_interpol)
    N = zeros(T, n_func_basefuncs, n_qpoints)
    dNdx = zeros(Vec{dim, T}, n_func_basefuncs, n_qpoints)
    dNdξ = zeros(Vec{dim, T}, n_func_basefuncs, n_qpoints)

    # Geometry interpolation
    n_geom_basefuncs = getnbasefunctions(geom_interpol)
    M = zeros(T, n_geom_basefuncs, n_qpoints)
    dMdξ = zeros(Vec{dim, T}, n_geom_basefuncs, n_qpoints)

    for (i, ξ) in enumerate(quad_rule.points)
        value!(func_interpol,  view(N, :, i), ξ)
        derivative!(func_interpol,  view(dNdξ, :, i), ξ)
        value!(geom_interpol,  view(M, :, i), ξ)
        derivative!(geom_interpol,  view(dMdξ, :, i), ξ)
    end

    detJdV = zeros(T, n_qpoints)

    CellScalarValues(N, dNdx, dNdξ, detJdV, quad_rule, func_interpol, M, dMdξ, geom_interpol)
end

# CellVectorValues
immutable CellVectorValues{dim, T <: Real, FIP <: Interpolation, GIP <: Interpolation, shape <: AbstractRefShape, M} <: CellValues{dim, T, FIP, GIP}
    N::Matrix{Vec{dim, T}}
    dNdx::Matrix{Tensor{2, dim, T, M}}
    dNdξ::Matrix{Tensor{2, dim, T, M}}
    detJdV::Vector{T}
    quad_rule::QuadratureRule{dim, shape, T}
    func_interpol::FIP
    M::Matrix{T}
    dMdξ::Matrix{Vec{dim, T}}
    geom_interpol::GIP
end

CellVectorValues{dim, FIP <: Interpolation, GIP <: Interpolation}(quad_rule::QuadratureRule{dim}, func_interpol::FIP, geom_interpol::GIP=func_interpol) =
    CellVectorValues(Float64, quad_rule, func_interpol, geom_interpol)

getnbasefunctions{dim}(cvv::CellVectorValues{dim}) = getnbasefunctions(cvv.func_interpol) * dim

function CellVectorValues{dim, T, FIP <: Interpolation, GIP <: Interpolation, shape <: AbstractRefShape}(
                            ::Type{T}, quad_rule::QuadratureRule{dim, shape}, func_interpol::FIP, geom_interpol::GIP=func_interpol)
    @assert getdim(func_interpol) == getdim(geom_interpol)
    @assert getrefshape(func_interpol) == getrefshape(geom_interpol) == shape
    n_qpoints = length(getweights(quad_rule))

    # Function interpolation
    n_func_basefuncs = getnbasefunctions(func_interpol) * dim
    N = zeros(Vec{dim, T}, n_func_basefuncs, n_qpoints)
    dNdx = [zero(Tensor{2, dim, T}) for i in 1:n_func_basefuncs, j in 1:n_qpoints]
    dNdξ = [zero(Tensor{2, dim, T}) for i in 1:n_func_basefuncs, j in 1:n_qpoints]

    # Geometry interpolation
    n_geom_basefuncs = getnbasefunctions(geom_interpol)
    M = zeros(T, n_geom_basefuncs, n_qpoints)
    dMdξ = zeros(Vec{dim, T}, n_geom_basefuncs, n_qpoints)

    N_temp = zeros(getnbasefunctions(func_interpol))
    dNdξ_temp = zeros(Vec{dim, T}, getnbasefunctions(func_interpol))
    for (i, ξ) in enumerate(quad_rule.points)
        value!(func_interpol, N_temp, ξ)
        derivative!(func_interpol, dNdξ_temp, ξ)
        basefunc_count = 1
        for basefunc in 1:getnbasefunctions(func_interpol)
            for comp in 1:dim
                N_comp = zeros(T, dim)
                N_comp[comp] = N_temp[basefunc]
                N[basefunc_count, i] = Vec{dim, T}((N_comp...))

                dN_comp = zeros(T, dim, dim)
                dN_comp[comp, :] = dNdξ_temp[basefunc]
                dNdξ[basefunc_count, i] = Tensor{2, dim, T}((dN_comp...))
                basefunc_count += 1
            end
        end
        value!(geom_interpol, view(M, :, i), ξ)
        derivative!(geom_interpol, view(dMdξ, :, i), ξ)
    end

    detJdV = zeros(T, n_qpoints)

    CellVectorValues(N, dNdx, dNdξ, detJdV, quad_rule, func_interpol, M, dMdξ, geom_interpol)
end

function reinit!{dim, T}(cv::CellValues{dim}, x::Vector{Vec{dim, T}})
    n_geom_basefuncs = getnbasefunctions(getgeometryinterpolation(cv))
    n_func_basefuncs = getnbasefunctions(getfunctioninterpolation(cv))
    @assert length(x) == n_geom_basefuncs
    isa(cv, CellVectorValues) && (n_func_basefuncs *= dim)

    @inbounds for i in 1:length(getpoints(cv.quad_rule))
        w = getweights(cv.quad_rule)[i]
        fecv_J = zero(Tensor{2, dim})
        for j in 1:n_geom_basefuncs
            fecv_J += x[j] ⊗ cv.dMdξ[j, i]
        end
        Jinv = inv(fecv_J)
        for j in 1:n_func_basefuncs
            cv.dNdx[j,i] = cv.dNdξ[j, i] ⋅ Jinv
        end
        detJ = det(fecv_J)
        detJ <= 0.0 && throw(ArgumentError("detJ is not positive: detJ = $(detJ)"))
        cv.detJdV[i] = detJ * w
    end
end
