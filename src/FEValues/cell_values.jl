# Defines CellScalarValues and CellVectorValues and common methods
"""
    CellScalarValues([::Type{T}], quad_rule::QuadratureRule, func_interp::Interpolation, [geo_interp::Interpolation])
    CellVectorValues([::Type{T}], quad_rule::QuadratureRule, func_interp::Interpolation, [geo_interp::Interpolation])

A `CellValues` object facilitates the process of evaluating values of shape functions, gradients of shape functions,
values of nodal functions, gradients and divergences of nodal functions etc. in the finite element cell. There are
two different types of `CellValues`: `CellScalarValues` and `CellVectorValues`. As the names suggest, `CellScalarValues`
utilizes scalar shape functions and `CellVectorValues` utilizes vectorial shape functions. For a scalar field, the
`CellScalarValues` type should be used. For vector field, both subtypes can be used.

**Arguments:**
* `T`: an optional argument (default to `Float64`) to determine the type the internal data is stored as.
* `quad_rule`: an instance of a [`QuadratureRule`](@ref)
* `func_interp`: an instance of an [`Interpolation`](@ref) used to interpolate the approximated function
* `geo_interp`: an optional instance of a [`Interpolation`](@ref) which is used to interpolate the geometry

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
struct CellScalarValues{dim,T<:Real,refshape<:AbstractRefShape,FI,GI} <: CellValues{dim,T,refshape,FI,GI}
    N::Matrix{T}
    dNdx::Matrix{Vec{dim,T}}
    dNdξ::Matrix{Vec{dim,T}}
    detJdV::Vector{T}
    M::Matrix{T}
    dMdξ::Matrix{Vec{dim,T}}
    quad_rule::QuadratureRule{dim,refshape}
    func_interp::FI
    geo_interp::GI
end

function CellScalarValues(quad_rule::QuadratureRule, func_interp::Interpolation,
        geo_interp::Interpolation=func_interp)
    CellScalarValues(Float64, quad_rule, func_interp, geo_interp)
end

function CellScalarValues(::Type{T}, quad_rule::QuadratureRule{dim,shape,}, func_interp::Interpolation,
        geo_interp::Interpolation=func_interp) where {dim,T,shape<:AbstractRefShape}

    @assert getdim(func_interp) == getdim(geo_interp)
    @assert getrefshape(func_interp) == getrefshape(geo_interp) == shape
    n_qpoints = length(getweights(quad_rule))

    # Function interpolation
    n_func_basefuncs = getnbasefunctions(func_interp)
    N    = fill(zero(T)          * T(NaN), n_func_basefuncs, n_qpoints)
    dNdx = fill(zero(Vec{dim,T}) * T(NaN), n_func_basefuncs, n_qpoints)
    dNdξ = fill(zero(Vec{dim,T}) * T(NaN), n_func_basefuncs, n_qpoints)

    # Geometry interpolation
    n_geom_basefuncs = getnbasefunctions(geo_interp)
    M    = fill(zero(T)          * T(NaN), n_geom_basefuncs, n_qpoints)
    dMdξ = fill(zero(Vec{dim,T}) * T(NaN), n_geom_basefuncs, n_qpoints)

    for (qp, ξ) in enumerate(quad_rule.points)
        for i in 1:n_func_basefuncs
            dNdξ[i, qp], N[i, qp] = gradient(ξ -> value(func_interp, i, ξ), ξ, :all)
        end
        for i in 1:n_geom_basefuncs
            dMdξ[i, qp], M[i, qp] = gradient(ξ -> value(geo_interp, i, ξ), ξ, :all)
        end
    end

    detJdV = fill(T(NaN), n_qpoints)

    CellScalarValues{dim,T,shape,typeof(func_interp),typeof(geo_interp)}(N, dNdx, dNdξ, detJdV, M, dMdξ, quad_rule, func_interp, geo_interp)
end

# CellVectorValues
struct CellVectorValues{dim,T<:Real,refshape<:AbstractRefShape,FI,GI,M} <: CellValues{dim,T,refshape,FI,GI}
    N::Matrix{Vec{dim,T}}
    dNdx::Matrix{Tensor{2,dim,T,M}}
    dNdξ::Matrix{Tensor{2,dim,T,M}}
    detJdV::Vector{T}
    M::Matrix{T}
    dMdξ::Matrix{Vec{dim,T}}
    quad_rule::QuadratureRule{dim,refshape}
    func_interp::FI
    geo_interp::GI
end

function CellVectorValues(quad_rule::QuadratureRule, func_interp::Interpolation, geo_interp::Interpolation=func_interp)
    CellVectorValues(Float64, quad_rule, func_interp, geo_interp)
end

function CellVectorValues(::Type{T}, quad_rule::QuadratureRule{dim,shape}, func_interp::Interpolation,
        geo_interp::Interpolation=func_interp) where {dim,T,shape<:AbstractRefShape}

    @assert getdim(func_interp) == getdim(geo_interp)
    @assert getrefshape(func_interp) == getrefshape(geo_interp) == shape
    n_qpoints = length(getweights(quad_rule))

    # Function interpolation
    n_func_basefuncs = getnbasefunctions(func_interp) * dim
    N    = fill(zero(Vec{dim,T})      * T(NaN), n_func_basefuncs, n_qpoints)
    dNdx = fill(zero(Tensor{2,dim,T}) * T(NaN), n_func_basefuncs, n_qpoints)
    dNdξ = fill(zero(Tensor{2,dim,T}) * T(NaN), n_func_basefuncs, n_qpoints)

    # Geometry interpolation
    n_geom_basefuncs = getnbasefunctions(geo_interp)
    M    = fill(zero(T)          * T(NaN), n_geom_basefuncs, n_qpoints)
    dMdξ = fill(zero(Vec{dim,T}) * T(NaN), n_geom_basefuncs, n_qpoints)

    for (qp, ξ) in enumerate(quad_rule.points)
        basefunc_count = 1
        for basefunc in 1:getnbasefunctions(func_interp)
            dNdξ_temp, N_temp = gradient(ξ -> value(func_interp, basefunc, ξ), ξ, :all)
            for comp in 1:dim
                N_comp = zeros(T, dim)
                N_comp[comp] = N_temp
                N[basefunc_count, qp] = Vec{dim,T}((N_comp...,))

                dN_comp = zeros(T, dim, dim)
                dN_comp[comp, :] = dNdξ_temp
                dNdξ[basefunc_count, qp] = Tensor{2,dim,T}((dN_comp...,))
                basefunc_count += 1
            end
        end
        for basefunc in 1:n_geom_basefuncs
            dMdξ[basefunc, qp], M[basefunc, qp] = gradient(ξ -> value(geo_interp, basefunc, ξ), ξ, :all)
        end
    end

    detJdV = fill(T(NaN), n_qpoints)
    MM = Tensors.n_components(Tensors.get_base(eltype(dNdx)))

    CellVectorValues{dim,T,shape,typeof(func_interp),typeof(geo_interp),MM}(N, dNdx, dNdξ, detJdV, M, dMdξ, quad_rule, func_interp, geo_interp)
end

function reinit!(cv::CellValues{dim}, x::AbstractVector{Vec{dim,T}}) where {dim,T}
    n_geom_basefuncs = getngeobasefunctions(cv)
    n_func_basefuncs = getn_scalarbasefunctions(cv)
    @assert length(x) == n_geom_basefuncs
    isa(cv, CellVectorValues) && (n_func_basefuncs *= dim)


    @inbounds for i in 1:length(cv.quad_rule.weights)
        w = cv.quad_rule.weights[i]
        fecv_J = zero(Tensor{2,dim})
        for j in 1:n_geom_basefuncs
            fecv_J += x[j] ⊗ cv.dMdξ[j, i]
        end
        detJ = det(fecv_J)
        detJ > 0.0 || throw_detJ_not_pos(detJ)
        cv.detJdV[i] = detJ * w
        Jinv = inv(fecv_J)
        for j in 1:n_func_basefuncs
            cv.dNdx[j, i] = cv.dNdξ[j, i] ⋅ Jinv
        end
    end
end
