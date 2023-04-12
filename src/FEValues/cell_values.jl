# Defines CellScalarValues and CellVectorValues and common methods
"""
    CellScalarValues([::Type{T}], quad_rule::QuadratureRule, func_interpol::Interpolation, [geo_interpol::Interpolation])
    CellVectorValues([::Type{T}], quad_rule::QuadratureRule, func_interpol::Interpolation, [geo_interpol::Interpolation])

A `CellValues` object facilitates the process of evaluating values of shape functions, gradients of shape functions,
values of nodal functions, gradients and divergences of nodal functions etc. in the finite element cell. There are
two different types of `CellValues`: `CellScalarValues` and `CellVectorValues`. As the names suggest, `CellScalarValues`
utilizes scalar shape functions and `CellVectorValues` utilizes vectorial shape functions. For a scalar field, the
`CellScalarValues` type should be used. For vector field, both subtypes can be used.

**Arguments:**
* `T`: an optional argument (default to `Float64`) to determine the type the internal data is stored as.
* `quad_rule`: an instance of a [`QuadratureRule`](@ref)
* `func_interpol`: an instance of an [`Interpolation`](@ref) used to interpolate the approximated function
* `geo_interpol`: an optional instance of a [`Interpolation`](@ref) which is used to interpolate the geometry

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
    dNdx::Matrix{SVector{sdim,T}}
    dNdξ::Matrix{SVector{rdim,T}}
    detJdV::Vector{T}
    M::Matrix{T}
    dMdξ::Matrix{SVector{rdim,T}}
    qr::QuadratureRule{rdim,refshape,T}
    # The following fields are deliberately abstract -- they are never used in
    # performance critical code, just stored here for convenience.
    func_interp::Interpolation{rdim,refshape}
    geo_interp::Interpolation{rdim,refshape}
end

# FIXME sdim should be something like `getdim(value(geo_interpol))``
function CellScalarValues(quad_rule::QuadratureRule, func_interpol::Interpolation,
        geo_interpol::Interpolation=func_interpol, sdim::Int=getdim(func_interpol))
    CellScalarValues(Float64, quad_rule, func_interpol, geo_interpol, sdim)
end

# FIXME sdim should be something like `length(value(geo_interpol))`
function CellScalarValues(::Type{T}, quad_rule::QuadratureRule{rdim,shape}, func_interpol::Interpolation{rdim,shape},
        geo_interpol::Interpolation{rdim,shape}=func_interpol, sdim::Int=getdim(func_interpol)) where {rdim,T,shape<:AbstractRefShape}

    n_qpoints = length(getweights(quad_rule))

    # Function interpolation
    n_func_basefuncs = getnbasefunctions(func_interpol)
    N    = fill(zero(T)          * T(NaN), n_func_basefuncs, n_qpoints)
    dNdx = fill(zero(SVector{sdim,T}) * T(NaN), n_func_basefuncs, n_qpoints)
    dNdξ = fill(zero(SVector{rdim,T}) * T(NaN), n_func_basefuncs, n_qpoints)

    # Geometry interpolation
    n_geom_basefuncs = getnbasefunctions(geo_interpol)
    M    = fill(zero(T)          * T(NaN), n_geom_basefuncs, n_qpoints)
    dMdξ = fill(zero(SVector{rdim,T}) * T(NaN), n_geom_basefuncs, n_qpoints)

    for (qp, ξ) in enumerate(quad_rule.points)
        for i in 1:n_func_basefuncs
            dNdξ[i, qp], N[i, qp] = gradient(ξ -> value(func_interpol, i, ξ), ξ, :all)
        end
        for i in 1:n_geom_basefuncs
            dMdξ[i, qp], M[i, qp] = gradient(ξ -> value(geo_interpol, i, ξ), ξ, :all)
        end
    end

    detJdV = fill(T(NaN), n_qpoints)

    CellScalarValues{sdim,rdim,T,shape}(N, dNdx, dNdξ, detJdV, M, dMdξ, quad_rule, func_interpol, geo_interpol)
end

# CellVectorValues
# TODO remove the assumption that all dimensions have to match.
#   rdim = reference element dimension
#   sdim = spatial dimension
#   vdim = vector dimension (i.e. dimension of evaluation of what `value` should return)
struct CellVectorValues{sdim,rdim,T<:Real,refshape<:AbstractRefShape,vdim} <: CellValues{sdim,rdim,T,refshape}
    N::Matrix{SVector{vdim,T}} # vdim
    dNdx::Matrix{SMatrix{vdim,sdim,T}} # vdim × sdim
    dNdξ::Matrix{SMatrix{vdim,rdim,T}} # vdim × rdim
    detJdV::Vector{T}
    M::Matrix{T}
    dMdξ::Matrix{SVector{rdim,T}} # rdim
    qr::QuadratureRule{rdim,refshape,T} #rdim
    # The following fields are deliberately abstract -- they are never used in
    # performance critical code, just stored here for convenience.
    func_interp::Interpolation{rdim,refshape} # rdim
    geo_interp::Interpolation{rdim,refshape} # rdim
end

function CellVectorValues(quad_rule::QuadratureRule, func_interpol::Interpolation, geo_interpol::Interpolation=func_interpol; sdim::Int=getdim(geo_interpol), vdim::Int=getdim(func_interpol))
    CellVectorValues(Float64, quad_rule, func_interpol, geo_interpol, sdim, vdim)
end

# FIXME sdim should be something like `length(value(geo_interpol))`
# FIXME vdim should be something like `length(value(func_interpol))`
function CellVectorValues(::Type{T}, quad_rule::QuadratureRule{rdim,shape}, func_interpol::Interpolation,
        geo_interpol::Interpolation=func_interpol, sdim::Int=getdim(geo_interpol), vdim::Int=getdim(func_interpol)) where {rdim,T,shape<:AbstractRefShape}
    @assert getrefshape(func_interpol) == getrefshape(geo_interpol) == shape
    n_qpoints = length(getweights(quad_rule))

    # Function interpolation
    n_func_basefuncs = getnbasefunctions(func_interpol) * vdim
    N    = fill(zero(SVector{vdim,T})      * T(NaN), n_func_basefuncs, n_qpoints)
    dNdx = fill(zero(SMatrix{vdim,sdim,T}) * T(NaN), n_func_basefuncs, n_qpoints)
    dNdξ = fill(zero(SMatrix{vdim,rdim,T}) * T(NaN), n_func_basefuncs, n_qpoints)

    # Geometry interpolation
    n_geom_basefuncs = getnbasefunctions(geo_interpol)
    M    = fill(zero(T)          * T(NaN), n_geom_basefuncs, n_qpoints)
    dMdξ = fill(zero(SVector{rdim,T}) * T(NaN), n_geom_basefuncs, n_qpoints)

    for (qp, ξ) in enumerate(quad_rule.points)
        basefunc_count = 1
        for basefunc in 1:getnbasefunctions(func_interpol)
            dNdξ_temp, N_temp = gradient(ξ -> value(func_interpol, basefunc, ξ), ξ, :all)
            for comp in 1:vdim
                N_comp = zeros(T, vdim)
                N_comp[comp] = N_temp
                N[basefunc_count, qp] = SVector{vdim,T}((N_comp...,))

                dN_comp = zeros(T, vdim, rdim)
                dN_comp[comp, :] = dNdξ_temp
                dNdξ[basefunc_count, qp] = SMatrix{vdim,rdim,T}((dN_comp...,))
                basefunc_count += 1
            end
        end
        for basefunc in 1:n_geom_basefuncs
            dMdξ[basefunc, qp], M[basefunc, qp] = gradient(ξ -> value(geo_interpol, basefunc, ξ), ξ, :all)
        end
    end

    detJdV = fill(T(NaN), n_qpoints)

    CellVectorValues{sdim,rdim,T,shape,vdim}(N, dNdx, dNdξ, detJdV, M, dMdξ, quad_rule, func_interpol, geo_interpol)
end

function reinit!(cv::CellValues{sdim,sdim}, x::AbstractVector{Vec{sdim,T}}) where {sdim,T}
    n_geom_basefuncs = getngeobasefunctions(cv)
    n_func_basefuncs = getnbasefunctions(cv)
    length(x) == n_geom_basefuncs || throw_incompatible_coord_length(length(x), n_geom_basefuncs)
    TdNdx = typeof(cv.dNdx[1, 1])

    @inbounds for i in 1:length(cv.qr.weights)
        w = cv.qr.weights[i]
        fecv_J = zero(Tensor{2,sdim,T})
        # fecv_J = zero(MMatrix{sdim,sdim,T})
        for j in 1:n_geom_basefuncs
            fecv_J += x[j] ⊗ tensor_cast(cv.dMdξ[j, i])
            # for k in 1:sdim, l in 1:sdim
            #     fecv_J[k,l] += x[j][k] * cv.dMdξ[j, i][l]
            # end
        end
        detJ = det(fecv_J)
        detJ > 0.0 || throw_detJ_not_pos(detJ)
        cv.detJdV[i] = detJ * w
        Jinv = inv(fecv_J)
        for j in 1:n_func_basefuncs
            cv.dNdx[j, i] = TdNdx((tensor_cast(cv.dNdξ[j, i]) ⋅ Jinv).data) # TODO via Tensors.jl
            # cv.dNdx[j, i] = Jinv' * cv.dNdξ[j, i]'
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
        fecv_J = zero(MMatrix{3,2,T}) # TODO replace with MixedTensor (see https://github.com/Ferrite-FEM/Tensors.jl/pull/188)
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
            cv.dNdx[j, i] = dothelper(cv.dNdξ[j, i], Jinv)
        end
    end
end

# Hotfix to get the dots right.
@inline dothelper(x::V,A::M) where {V<:SVector,M<:Union{SMatrix,MMatrix}} = A'*x
@inline dothelper(B::M1,A::M2) where {M1<:SMatrix,M2<:Union{SMatrix,MMatrix}} = B*A

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
        fecv_J = zero(MMatrix{2,1,T}) # TODO replace with MixedTensor (see https://github.com/Ferrite-FEM/Tensors.jl/pull/188)
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
            cv.dNdx[j, i] = dothelper(cv.dNdξ[j, i], Jinv)
        end
    end
end

function reinit!(cv::CellValues{3,1}, x::AbstractVector{Vec{3,T}}) where {T}
    n_geom_basefuncs = getngeobasefunctions(cv)
    n_func_basefuncs = getnbasefunctions(cv)
    length(x) == n_geom_basefuncs || throw_incompatible_coord_length(length(x), n_geom_basefuncs)

    @inbounds for i in 1:length(cv.qr.weights)
        w = cv.qr.weights[i]
        fecv_J = zero(MMatrix{3,1,T}) # TODO replace with MixedTensor (see https://github.com/Ferrite-FEM/Tensors.jl/pull/188)
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
            cv.dNdx[j, i] = dothelper(cv.dNdξ[j, i], Jinv)
        end
    end
end
