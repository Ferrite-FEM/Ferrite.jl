# Defines CellScalarValues and CellVectorValues and common methods
"""
    CellScalarValues([::Type{T}], quad_rule::QuadratureRule, func_interpol::Interpolation, [geom_interpol::Interpolation])
    CellVectorValues([::Type{T}], quad_rule::QuadratureRule, func_interpol::Interpolation, [geom_interpol::Interpolation])

A `CellValues` object facilitates the process of evaluating values of shape functions, gradients of shape functions,
values of nodal functions, gradients and divergences of nodal functions etc. in the finite element cell. There are
two different types of `CellValues`: `CellScalarValues` and `CellVectorValues`. As the names suggest, `CellScalarValues`
utilizes scalar shape functions and `CellVectorValues` utilizes vectorial shape functions. For a scalar field, the
`CellScalarValues` type should be used. For vector field, both subtypes can be used.

**Arguments:**
* `T`: an optional argument (default to `Float64`) to determine the type the internal data is stored as.
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

"""
    CellScalarValues{ξdim,xdim,T<:Real,refshape<:AbstractRefShape} <: CellValues{ξdim,xdim,T,refshape}

`ξdim` : reference domain dimension \\
`xdim` : node coordinate dimension
"""
struct CellScalarValues{ξdim,xdim,T<:Real,refshape<:AbstractRefShape} <: CellValues{ξdim,xdim,T,refshape}
    N::Matrix{T}
    dNdx::Matrix{Vec{xdim,T}}
    dNdξ::Matrix{Vec{ξdim,T}}
    detJdV::Vector{T}
    M::Matrix{T}
    dMdξ::Matrix{Vec{ξdim,T}}
    qr_weights::Vector{T}
end

function CellScalarValues(quad_rule::QuadratureRule, func_interpol::Interpolation,
        geom_interpol::Interpolation=func_interpol)
    CellScalarValues(Float64, quad_rule, func_interpol, geom_interpol)
end

function CellScalarValues(::Type{T}, quad_rule::QuadratureRule{ξdim,shape}, func_interpol::Interpolation,
        geom_interpol::Interpolation=func_interpol; xdim=ξdim) where {ξdim,T,shape<:AbstractRefShape}
    # There is no way to infer xdim from ξdim
    # In most solid mechanics cases, xdim = ξdim
    # But in a truss element (line element with nodes in 2D or 3D), 
    # xdim = 2 or 3, ξdim = 1. Users should pass in proper xdim value in the call site.

    @assert getdim(func_interpol) == getdim(geom_interpol)
    @assert getrefshape(func_interpol) == getrefshape(geom_interpol) == shape
    n_qpoints = length(getweights(quad_rule))

    # Function interpolation
    n_func_basefuncs = getnbasefunctions(func_interpol)
    N    = fill(zero(T)           * T(NaN), n_func_basefuncs, n_qpoints)
    dNdx = fill(zero(Vec{xdim,T}) * T(NaN), n_func_basefuncs, n_qpoints)
    dNdξ = fill(zero(Vec{ξdim,T}) * T(NaN), n_func_basefuncs, n_qpoints)

    # Geometry interpolation
    n_geom_basefuncs = getnbasefunctions(geom_interpol)
    M    = fill(zero(T)           * T(NaN), n_geom_basefuncs, n_qpoints)
    dMdξ = fill(zero(Vec{ξdim,T}) * T(NaN), n_geom_basefuncs, n_qpoints)

    for (qp, ξ) in enumerate(quad_rule.points)
        for i in 1:n_func_basefuncs
            dNdξ[i, qp], N[i, qp] = gradient(ξ -> value(func_interpol, i, ξ), ξ, :all)
        end
        for i in 1:n_geom_basefuncs
            dMdξ[i, qp], M[i, qp] = gradient(ξ -> value(geom_interpol, i, ξ), ξ, :all)
        end
    end

    detJdV = fill(T(NaN), n_qpoints)

    CellScalarValues{ξdim,xdim,T,shape}(N, dNdx, dNdξ, detJdV, M, dMdξ, quad_rule.weights)
end

# CellVectorValues
# ? Vector dimension is assumed to be xdim ?
# related: https://github.com/KristofferC/JuAFEM.jl/issues/193#issuecomment-502247133
struct CellVectorValues{ξdim,xdim,T<:Real,refshape<:AbstractRefShape,M} <: CellValues{ξdim,xdim,T,refshape}
    N::Matrix{Vec{xdim,T}}
    dNdx::Matrix{Tensor{2,xdim,T,M}}
    dNdξ::Matrix{Tensor{2,ξdim,T,M}}
    detJdV::Vector{T}
    M::Matrix{T}
    dMdξ::Matrix{Vec{ξdim,T}}
    qr_weights::Vector{T}
end

function CellVectorValues(quad_rule::QuadratureRule, func_interpol::Interpolation, geom_interpol::Interpolation=func_interpol)
    CellVectorValues(Float64, quad_rule, func_interpol, geom_interpol)
end

function CellVectorValues(::Type{T}, quad_rule::QuadratureRule{ξdim,shape}, func_interpol::Interpolation,
        geom_interpol::Interpolation=func_interpol; xdim=ξdim) where {ξdim,T,shape<:AbstractRefShape}
    # There is no way to infer xdim from ξdim
    # In most solid mechanics cases, xdim = ξdim
    # But in a truss element (line element with nodes in 2D or 3D), 
    # xdim = 2 or 3, ξdim = 1. Users should pass in proper xdim value in the call site.

    @assert getdim(func_interpol) == getdim(geom_interpol)
    @assert getrefshape(func_interpol) == getrefshape(geom_interpol) == shape
    n_qpoints = length(getweights(quad_rule))

    # Function interpolation
    n_func_basefuncs = getnbasefunctions(func_interpol) * xdim
    N    = fill(zero(Vec{xdim,T})      * T(NaN), n_func_basefuncs, n_qpoints)
    dNdx = fill(zero(Tensor{2,xdim,T}) * T(NaN), n_func_basefuncs, n_qpoints)
    # TODO: each entry should be a second order tensor with shape xdim x ξdim
    dNdξ = fill(zero(Tensor{2,ξdim,T}) * T(NaN), n_func_basefuncs, n_qpoints)

    # Geometry interpolation
    n_geom_basefuncs = getnbasefunctions(geom_interpol)
    M    = fill(zero(T)          * T(NaN), n_geom_basefuncs, n_qpoints)
    dMdξ = fill(zero(Vec{ξdim,T}) * T(NaN), n_geom_basefuncs, n_qpoints)

    for (qp, ξ) in enumerate(quad_rule.points)
        basefunc_count = 1
        for basefunc in 1:getnbasefunctions(func_interpol)
            dNdξ_temp, N_temp = gradient(ξ -> value(func_interpol, basefunc, ξ), ξ, :all)
            for comp in 1:xdim
                N_comp = zeros(T, xdim)
                N_comp[comp] = N_temp
                N[basefunc_count, qp] = Vec{xdim,T}((N_comp...,))

                dN_comp = zeros(T, ξdim, ξdim)
                dN_comp[comp, :] = dNdξ_temp
                dNdξ[basefunc_count, qp] = Tensor{2,ξdim,T}((dN_comp...,))
                basefunc_count += 1
            end
        end
        for basefunc in 1:n_geom_basefuncs
            dMdξ[basefunc, qp], M[basefunc, qp] = gradient(ξ -> value(geom_interpol, basefunc, ξ), ξ, :all)
        end
    end

    detJdV = fill(T(NaN), n_qpoints)
    MM = Tensors.n_components(Tensors.get_base(eltype(dNdx)))

    CellVectorValues{ξdim,xdim,T,shape,MM}(N, dNdx, dNdξ, detJdV, M, dMdξ, quad_rule.weights)
end

function reinit!(cv::CellValues{ξdim,xdim}, x::AbstractVector{Vec{xdim,T}}) where {ξdim,xdim,T}
    n_geom_basefuncs = getngeobasefunctions(cv)
    n_func_basefuncs = getn_scalarbasefunctions(cv)
    @assert length(x) == n_geom_basefuncs
    isa(cv, CellVectorValues) && (n_func_basefuncs *= xdim)

    @inbounds for i in 1:length(cv.qr_weights)
        w = cv.qr_weights[i]
        fecv_J = zero(Tensor{2,xdim})
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