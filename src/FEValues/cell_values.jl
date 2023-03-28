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
CellValues, CellScalarValues, CellVectorValues

# CellScalarValues
struct CellScalarValues{sdim,rdim,T<:Real,refshape<:AbstractRefShape} <: CellValues{sdim,rdim,T,refshape}
    N::Matrix{T}
    dNdx::Matrix{Vec{sdim,T}}
    dNdξ::Matrix{Vec{rdim,T}}
    detJdV::Vector{T}
    M::Matrix{T}
    dMdξ::Matrix{Vec{rdim,T}}
    qr::QuadratureRule{rdim,refshape,T}
    # The following fields are deliberately abstract -- they are never used in
    # performance critical code, just stored here for convenience.
    func_interp::Interpolation{rdim,refshape}
    geo_interp::Interpolation{rdim,refshape}
end

# FIXME sdim should be something like `getdim(value(geo_interpol))``
function CellScalarValues(quad_rule::QuadratureRule, func_interpol::Interpolation,
        geom_interpol::Interpolation=func_interpol, sdim::Int=getdim(func_interpol))
    CellScalarValues(Float64, quad_rule, func_interpol, geom_interpol, sdim)
end

# FIXME sdim should be something like `getdim(value(geo_interpol))`
function CellScalarValues(::Type{T}, quad_rule::QuadratureRule{rdim,shape}, func_interpol::Interpolation{rdim,shape},
        geom_interpol::Interpolation{rdim,shape}=func_interpol, sdim::Int=getdim(func_interpol)) where {rdim,T,shape<:AbstractRefShape}

    n_qpoints = length(getweights(quad_rule))

    # Function interpolation
    n_func_basefuncs = getnbasefunctions(func_interpol)
    N    = fill(zero(T)          * T(NaN), n_func_basefuncs, n_qpoints)
    dNdx = fill(zero(Vec{sdim,T}) * T(NaN), n_func_basefuncs, n_qpoints)
    dNdξ = fill(zero(Vec{rdim,T}) * T(NaN), n_func_basefuncs, n_qpoints)

    # Geometry interpolation
    n_geom_basefuncs = getnbasefunctions(geom_interpol)
    M    = fill(zero(T)          * T(NaN), n_geom_basefuncs, n_qpoints)
    dMdξ = fill(zero(Vec{rdim,T}) * T(NaN), n_geom_basefuncs, n_qpoints)

    for (qp, ξ) in enumerate(quad_rule.points)
        for i in 1:n_func_basefuncs
            dNdξ[i, qp], N[i, qp] = gradient(ξ -> value(func_interpol, i, ξ), ξ, :all)
        end
        for i in 1:n_geom_basefuncs
            dMdξ[i, qp], M[i, qp] = gradient(ξ -> value(geom_interpol, i, ξ), ξ, :all)
        end
    end

    detJdV = fill(T(NaN), n_qpoints)

    CellScalarValues{sdim,rdim,T,shape}(N, dNdx, dNdξ, detJdV, M, dMdξ, quad_rule, func_interpol, geom_interpol)
end

# CellVectorValues
# TODO remove the assumption that all dimensions have to match.
#   rdim = reference element dimension
#   sdim = spatial dimension
#   vdim = vector dimension (i.e. dimension of evaluation of what value should return)
struct CellVectorValues{dim,T<:Real,refshape<:AbstractRefShape,M} <: CellValues{dim,dim,T,refshape}
    N::Matrix{Vec{dim,T}} # vdim
    dNdx::Matrix{Tensor{2,dim,T,M}} # vdim × sdim
    dNdξ::Matrix{Tensor{2,dim,T,M}} # vdim × rdim
    detJdV::Vector{T}
    M::Matrix{T}
    dMdξ::Matrix{Vec{dim,T}} # rdim
    qr::QuadratureRule{dim,refshape,T} #rdim
    # The following fields are deliberately abstract -- they are never used in
    # performance critical code, just stored here for convenience.
    func_interp::Interpolation{dim,refshape} # rdim
    geo_interp::Interpolation{dim,refshape} # rdim
end

function CellVectorValues(quad_rule::QuadratureRule, func_interpol::Interpolation, geom_interpol::Interpolation=func_interpol)
    CellVectorValues(Float64, quad_rule, func_interpol, geom_interpol)
end

function CellVectorValues(::Type{T}, quad_rule::QuadratureRule{dim,shape}, func_interpol::Interpolation,
        geom_interpol::Interpolation=func_interpol) where {dim,T,shape<:AbstractRefShape}

    @assert getdim(func_interpol) == getdim(geom_interpol)
    @assert getrefshape(func_interpol) == getrefshape(geom_interpol) == shape
    n_qpoints = length(getweights(quad_rule))

    # Function interpolation
    n_func_basefuncs = getnbasefunctions(func_interpol) * dim
    N    = fill(zero(Vec{dim,T})      * T(NaN), n_func_basefuncs, n_qpoints)
    dNdx = fill(zero(Tensor{2,dim,T}) * T(NaN), n_func_basefuncs, n_qpoints)
    dNdξ = fill(zero(Tensor{2,dim,T}) * T(NaN), n_func_basefuncs, n_qpoints)

    # Geometry interpolation
    n_geom_basefuncs = getnbasefunctions(geom_interpol)
    M    = fill(zero(T)          * T(NaN), n_geom_basefuncs, n_qpoints)
    dMdξ = fill(zero(Vec{dim,T}) * T(NaN), n_geom_basefuncs, n_qpoints)

    for (qp, ξ) in enumerate(quad_rule.points)
        basefunc_count = 1
        for basefunc in 1:getnbasefunctions(func_interpol)
            dNdξ_temp, N_temp = gradient(ξ -> value(func_interpol, basefunc, ξ), ξ, :all)
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
            dMdξ[basefunc, qp], M[basefunc, qp] = gradient(ξ -> value(geom_interpol, basefunc, ξ), ξ, :all)
        end
    end

    detJdV = fill(T(NaN), n_qpoints)
    MM = Tensors.n_components(Tensors.get_base(eltype(dNdx)))

    CellVectorValues{dim,T,shape,MM}(N, dNdx, dNdξ, detJdV, M, dMdξ, quad_rule, func_interpol, geom_interpol)
end

function reinit!(cv::CellValues{dim}, x::AbstractVector{Vec{dim,T}}) where {dim,T}
    n_geom_basefuncs = getngeobasefunctions(cv)
    n_func_basefuncs = getnbasefunctions(cv)
    length(x) == n_geom_basefuncs || throw_incompatible_coord_length(length(x), n_geom_basefuncs)

    @inbounds for i in 1:length(cv.qr.weights)
        w = cv.qr.weights[i]
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

# Reinit for embedded surfaces.
#
# The transformation theorem for some function f on a 2D surface in 3D space leads to
#   ∫ f ⋅ dS = ∫ f ⋅ (∂x/∂ξ₁ × ∂x/∂ξ₂) dξ₁dξ₂ = ∫ f ⋅ n ||∂x/∂ξ₁ × ∂x/∂ξ₂||₂ dξ₁dξ₂
# where ||∂x/∂ξ₁ × ∂x/∂ξ₂||₂ is "detJ" and n is the unit normal.
# See e.g. https://scicomp.stackexchange.com/questions/41741/integration-of-d-1-dimensional-functions-on-finite-element-surfaces for simple explanation.
# For more details see e.g. the doctoral thesis by Mirza Cenanovic **Finite element methods for surface problems* (2017), Ch. 2 **Trangential Calculus**.
function reinit!(cv::CellValues{3,2}, x::AbstractVector{Vec{3,T}}) where {T}
    n_geom_basefuncs = getngeobasefunctions(cv)
    n_func_basefuncs = getnbasefunctions(cv)
    length(x) == n_geom_basefuncs || throw_incompatible_coord_length(length(x), n_geom_basefuncs)

    @inbounds for i in 1:length(cv.qr.weights)
        w = cv.qr.weights[i]
        fecv_J = zeros(Float64,3,2) # TODO replace with MixedTensor (see https://github.com/Ferrite-FEM/Tensors.jl/pull/188)
        for j in 1:n_geom_basefuncs
            #fecv_J += x[j] ⊗ cv.dMdξ[j, i] # TODO via Tensors.jl
            for k in 1:3, l in 1:2
                fecv_J[k,l] += x[j][k] * cv.dMdξ[j, i][l]
            end
        end
        # "det(J) =" ||∂x/∂ξ₁ × ∂x/∂ξ₂||₂
        detJ = norm(fecv_J[:,1] × fecv_J[:,2])
        detJ > 0.0 || throw_detJ_not_pos(detJ)
        cv.detJdV[i] = detJ * w
        # Compute left inverse of J
        Jinv = pinv(fecv_J)
        for j in 1:n_func_basefuncs
            #cv.dNdx[j, i] = cv.dNdξ[j, i] ⋅ Jinv # TODO via Tensors.jl
            cv.dNdx[j, i] = Vec{3}(Jinv' * cv.dNdξ[j, i])
        end
    end
end

# Reinit for embedded curves.
#
# The transformation theorem for some function f on a 1D curve in 2D and 3D space leads to
#   ∫ f ⋅ dE = ∫ f ⋅ ∂x/∂ξ dξ = ∫ f ⋅ t ||∂x/∂ξ||₂ dξ
# where ||∂x/∂ξ||₂ is "detJ" and t is "the unit tangent".
# See e.g. https://scicomp.stackexchange.com/questions/41741/integration-of-d-1-dimensional-functions-on-finite-element-surfaces for simple explanation.
function reinit!(cv::CellValues{2,1}, x::AbstractVector{Vec{2,T}}) where {T}
    n_geom_basefuncs = getngeobasefunctions(cv)
    n_func_basefuncs = getnbasefunctions(cv)
    length(x) == n_geom_basefuncs || throw_incompatible_coord_length(length(x), n_geom_basefuncs)

    @inbounds for i in 1:length(cv.qr.weights)
        w = cv.qr.weights[i]
        fecv_J = zeros(Float64,2,1) # TODO replace with MixedTensor (see https://github.com/Ferrite-FEM/Tensors.jl/pull/188)
        for j in 1:n_geom_basefuncs
            #fecv_J += x[j] ⊗ cv.dMdξ[j, i] # TODO via Tensors.jl
            for k in 1:2, l in 1:1
                fecv_J[k,l] += x[j][k] * cv.dMdξ[j, i][l]
            end
        end
        # "det(J) =" ||∂x/∂ξ||₂
        detJ = norm(fecv_J)
        detJ > 0.0 || throw_detJ_not_pos(detJ)
        cv.detJdV[i] = detJ * w
        # Compute left inverse of J
        Jinv = pinv(fecv_J)
        for j in 1:n_func_basefuncs
            #cv.dNdx[j, i] = cv.dNdξ[j, i] ⋅ Jinv # TODO via Tensors.jl
            cv.dNdx[j, i] = Vec{2}(Jinv' * cv.dNdξ[j, i])
        end
    end
end

function reinit!(cv::CellValues{3,1}, x::AbstractVector{Vec{3,T}}) where {T}
    n_geom_basefuncs = getngeobasefunctions(cv)
    n_func_basefuncs = getnbasefunctions(cv)
    length(x) == n_geom_basefuncs || throw_incompatible_coord_length(length(x), n_geom_basefuncs)

    @inbounds for i in 1:length(cv.qr.weights)
        w = cv.qr.weights[i]
        fecv_J = zeros(Float64,3,1) # TODO replace with MixedTensor (see https://github.com/Ferrite-FEM/Tensors.jl/pull/188)
        for j in 1:n_geom_basefuncs
            #fecv_J += x[j] ⊗ cv.dMdξ[j, i] # TODO via Tensors.jl
            for k in 1:3, l in 1:1
                fecv_J[k,l] += x[j][k] * cv.dMdξ[j, i][l]
            end
        end
        # "det(J) =" ||∂x/∂ξ||₂
        detJ = norm(fecv_J)
        detJ > 0.0 || throw_detJ_not_pos(detJ)
        cv.detJdV[i] = detJ * w
        # Compute left inverse of J
        Jinv = pinv(fecv_J)
        for j in 1:n_func_basefuncs
            #cv.dNdx[j, i] = cv.dNdξ[j, i] ⋅ Jinv # TODO via Tensors.jl
            cv.dNdx[j, i] = Vec{3}(Jinv' * cv.dNdξ[j, i])
        end
    end
end
