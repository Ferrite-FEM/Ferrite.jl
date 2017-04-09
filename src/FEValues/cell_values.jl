# Defines CellScalarValues and CellVectorValues and common methods
"""
A `CellValues` object facilitates the process of evaluating values of shape functions, gradients of shape functions,
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
immutable CellScalarValues{dim, T <: Real, refshape <: AbstractRefShape} <: CellValues{dim, T, refshape}
    N::Matrix{T}
    dNdx::Matrix{Vec{dim, T}}
    dNdξ::Matrix{Vec{dim, T}}
    detJdV::Vector{T}
    M::Matrix{T}
    dMdξ::Matrix{Vec{dim, T}}
    qr_weights::Vector{T}
end

function CellScalarValues{dim}(quad_rule::QuadratureRule{dim}, func_interpol::Interpolation,
        geom_interpol::Interpolation=func_interpol)
    CellScalarValues(Float64, quad_rule, func_interpol, geom_interpol)
end

function CellScalarValues{dim, T, shape <: AbstractRefShape}(::Type{T}, quad_rule::QuadratureRule{dim, shape},
        func_interpol::Interpolation, geom_interpol::Interpolation=func_interpol)

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

    for (qp, ξ) in enumerate(quad_rule.points)
        for i in 1:n_func_basefuncs
            dNdξ[i, qp], N[i, qp] = gradient(ξ -> value(func_interpol, i, ξ), ξ, :all)
        end
        for i in 1:n_geom_basefuncs
            dMdξ[i, qp], M[i, qp] = gradient(ξ -> value(func_interpol, i, ξ), ξ, :all)
        end
    end

    detJdV = zeros(T, n_qpoints)

    CellScalarValues{dim, T, shape}(N, dNdx, dNdξ, detJdV, M, dMdξ, quad_rule.weights)
end

# CellVectorValues
immutable CellVectorValues{dim, T <: Real, refshape <: AbstractRefShape, M} <: CellValues{dim, T, refshape}
    N::Matrix{Vec{dim, T}}
    dNdx::Matrix{Tensor{2, dim, T, M}}
    dNdξ::Matrix{Tensor{2, dim, T, M}}
    detJdV::Vector{T}
    M::Matrix{T}
    dMdξ::Matrix{Vec{dim, T}}
    qr_weights::Vector{T}
end

function CellVectorValues(quad_rule::QuadratureRule, func_interpol::Interpolation, geom_interpol::Interpolation=func_interpol)
    CellVectorValues(Float64, quad_rule, func_interpol, geom_interpol)
end

function CellVectorValues{dim, T, shape <: AbstractRefShape}(::Type{T}, quad_rule::QuadratureRule{dim, shape}, func_interpol::Interpolation,
        geom_interpol::Interpolation=func_interpol)

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

    for (qp, ξ) in enumerate(quad_rule.points)
        basefunc_count = 1
        for basefunc in 1:getnbasefunctions(func_interpol)
            dNdξ_temp, N_temp = gradient(ξ -> value(func_interpol, basefunc, ξ), ξ, :all)
            for comp in 1:dim
                N_comp = zeros(T, dim)
                N_comp[comp] = N_temp
                N[basefunc_count, qp] = Vec{dim, T}((N_comp...))

                dN_comp = zeros(T, dim, dim)
                dN_comp[comp, :] = dNdξ_temp
                dNdξ[basefunc_count, qp] = Tensor{2, dim, T}((dN_comp...))
                basefunc_count += 1
            end
        end
        for basefunc in 1:n_geom_basefuncs
            dMdξ[basefunc, qp], M[basefunc, qp] = gradient(ξ -> value(geom_interpol, basefunc, ξ), ξ, :all)
        end
    end

    detJdV = zeros(T, n_qpoints)
    MM = Tensors.n_components(Tensors.get_base(eltype(dNdx)))

    CellVectorValues{dim, T, shape, MM}(N, dNdx, dNdξ, detJdV, M, dMdξ, quad_rule.weights)
end

function reinit!{dim, T}(cv::CellValues{dim}, x::AbstractVector{Vec{dim, T}})
    n_geom_basefuncs = getngeobasefunctions(cv)
    n_func_basefuncs = getn_scalarbasefunctions(cv)
    @assert length(x) == n_geom_basefuncs
    isa(cv, CellVectorValues) && (n_func_basefuncs *= dim)


    @inbounds for i in 1:length(cv.qr_weights)
        w = cv.qr_weights[i]
        fecv_J = zero(Tensor{2, dim})
        for j in 1:n_geom_basefuncs
            fecv_J += x[j] ⊗ cv.dMdξ[j, i]
        end
        detJ = det(fecv_J)
        detJ > 0.0 || throw(ArgumentError("det(J) is not positive: det(J) = $(detJ)"))
        cv.detJdV[i] = detJ * w
        Jinv = inv(fecv_J)
        for j in 1:n_func_basefuncs
            cv.dNdx[j, i] = cv.dNdξ[j, i] ⋅ Jinv
        end
    end
end
