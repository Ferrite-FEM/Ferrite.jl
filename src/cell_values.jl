# Defines CellScalarValues and CellVectorValues and common methods
"""
A `CellValues` object facilitates the process of evaluating values shape functions, gradients of shape functions,
values of nodal functions, gradients and divergences of nodal functions etc. in the finite element cell. There are
two different types of `CellValues`: `CellScalarValues` and `CellVectorValues`. As the names suggest, `CellScalarValues`
utilizes scalar shape functions and `CellVectorValues` utilizes vectorial shape functions. For a scalar field, the
`CellScalarValues` type should be used. For vector field, both subtypes can be used.

**Constructors:**

```julia
CellScalarValues([::Type{T}], quad_rule::QuadratureRule, function_space::FunctionSpace, [geometric_space::FunctionSpace])
CellVectorValues([::Type{T}], quad_rule::QuadratureRule, function_space::FunctionSpace, [geometric_space::FunctionSpace])
```

**Arguments:**

* `T`: an optional argument to determine the type the internal data is stored as.
* `quad_rule`: an instance of a [`QuadratureRule`](@ref)
* `function_space`: an instance of a [`FunctionSpace`](@ref) used to interpolate the approximated function
* `geometric_space`: an optional instance of a [`FunctionSpace`](@ref) which is used to interpolate the geometry

**Common methods:**

* [`reinit!`](@ref)
* [`getnquadpoints`](@ref)
* [`getquadrule`](@ref)
* [`getfunctionspace`](@ref)
* [`getgeometricspace`](@ref)
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
immutable CellScalarValues{dim, T <: Real, FS <: FunctionSpace, GS <: FunctionSpace, shape <: AbstractRefShape} <: CellValues{dim, T, FS, GS}
    N::Matrix{T}
    dNdx::Matrix{Vec{dim, T}}
    dNdξ::Matrix{Vec{dim, T}}
    detJdV::Vector{T}
    quad_rule::QuadratureRule{dim, shape, T}
    function_space::FS
    M::Matrix{T}
    dMdξ::Matrix{Vec{dim, T}}
    geometric_space::GS
end

CellScalarValues{dim, FS <: FunctionSpace, GS <: FunctionSpace}(quad_rule::QuadratureRule{dim}, func_space::FS, geom_space::GS=func_space) =
    CellScalarValues(Float64, quad_rule, func_space, geom_space)

getnbasefunctions(cv::CellScalarValues) = getnbasefunctions(cv.function_space)

function CellScalarValues{dim, T, FS <: FunctionSpace, GS <: FunctionSpace, shape <: AbstractRefShape}(
    ::Type{T}, quad_rule::QuadratureRule{dim, shape}, func_space::FS, geom_space::GS=func_space)

    @assert getdim(func_space) == getdim(geom_space)
    @assert getrefshape(func_space) == getrefshape(geom_space) == shape
    n_qpoints = length(getweights(quad_rule))

    # Function interpolation
    n_func_basefuncs = getnbasefunctions(func_space)
    N = zeros(T, n_func_basefuncs, n_qpoints)
    dNdx = zeros(Vec{dim, T}, n_func_basefuncs, n_qpoints)
    dNdξ = zeros(Vec{dim, T}, n_func_basefuncs, n_qpoints)

    # Geometry interpolation
    n_geom_basefuncs = getnbasefunctions(geom_space)
    M = zeros(T, n_geom_basefuncs, n_qpoints)
    dMdξ = zeros(Vec{dim, T}, n_geom_basefuncs, n_qpoints)

    for (i, ξ) in enumerate(quad_rule.points)
        value!(func_space,  view(N, :, i), ξ)
        derivative!(func_space,  view(dNdξ, :, i), ξ)
        value!(geom_space,  view(M, :, i), ξ)
        derivative!(geom_space,  view(dMdξ, :, i), ξ)
    end

    detJdV = zeros(T, n_qpoints)

    CellScalarValues(N, dNdx, dNdξ, detJdV, quad_rule, func_space, M, dMdξ, geom_space)
end

# CellVectorValues
immutable CellVectorValues{dim, T <: Real, FS <: FunctionSpace, GS <: FunctionSpace, shape <: AbstractRefShape, M} <: CellValues{dim, T, FS, GS}
    N::Matrix{Vec{dim, T}}
    dNdx::Matrix{Tensor{2, dim, T, M}}
    dNdξ::Matrix{Tensor{2, dim, T, M}}
    detJdV::Vector{T}
    quad_rule::QuadratureRule{dim, shape, T}
    function_space::FS
    M::Matrix{T}
    dMdξ::Matrix{Vec{dim, T}}
    geometric_space::GS
end

CellVectorValues{dim, FS <: FunctionSpace, GS <: FunctionSpace}(quad_rule::QuadratureRule{dim}, func_space::FS, geom_space::GS=func_space) =
    CellVectorValues(Float64, quad_rule, func_space, geom_space)

getnbasefunctions{dim}(cvv::CellVectorValues{dim}) = getnbasefunctions(cvv.function_space) * dim

function CellVectorValues{dim, T, FS <: FunctionSpace, GS <: FunctionSpace, shape <: AbstractRefShape}(
                            ::Type{T}, quad_rule::QuadratureRule{dim, shape}, func_space::FS, geom_space::GS=func_space)
    @assert getdim(func_space) == getdim(geom_space)
    @assert getrefshape(func_space) == getrefshape(geom_space) == shape
    n_qpoints = length(getweights(quad_rule))

    # Function interpolation
    n_func_basefuncs = getnbasefunctions(func_space) * dim
    N = zeros(Vec{dim, T}, n_func_basefuncs, n_qpoints)
    dNdx = [zero(Tensor{2, dim, T}) for i in 1:n_func_basefuncs, j in 1:n_qpoints]
    dNdξ = [zero(Tensor{2, dim, T}) for i in 1:n_func_basefuncs, j in 1:n_qpoints]

    # Geometry interpolation
    n_geom_basefuncs = getnbasefunctions(geom_space)
    M = zeros(T, n_geom_basefuncs, n_qpoints)
    dMdξ = zeros(Vec{dim, T}, n_geom_basefuncs, n_qpoints)

    N_temp = zeros(getnbasefunctions(func_space))
    dNdξ_temp = zeros(Vec{dim, T}, getnbasefunctions(func_space))
    for (i, ξ) in enumerate(quad_rule.points)
        value!(func_space, N_temp, ξ)
        derivative!(func_space, dNdξ_temp, ξ)
        basefunc_count = 1
        for basefunc in 1:getnbasefunctions(func_space)
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
        value!(geom_space, view(M, :, i), ξ)
        derivative!(geom_space, view(dMdξ, :, i), ξ)
    end

    detJdV = zeros(T, n_qpoints)

    CellVectorValues(N, dNdx, dNdξ, detJdV, quad_rule, func_space, M, dMdξ, geom_space)
end

function reinit!{dim, T}(cv::CellValues{dim}, x::Vector{Vec{dim, T}})
    n_geom_basefuncs = getnbasefunctions(getgeometricspace(cv))
    n_func_basefuncs = getnbasefunctions(getfunctionspace(cv))
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
