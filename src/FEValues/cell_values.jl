# Defines CellScalarValues and CellVectorValues and common methods
"""
    CellScalarValues([::Type{T}], quad_rule::QuadratureRule, func_interpol::ScalarInterpolation, [geom_interpol::Interpolation])
    CellVectorValues([::Type{T}], quad_rule::QuadratureRule, func_interpol::VectorInterpolation, [geom_interpol::Interpolation])

A `CellValues` object facilitates the process of evaluating values of shape functions, gradients of shape functions,
values of nodal functions, gradients and divergences of nodal functions etc. in the finite element cell. There are
two different types of `CellValues`: `CellScalarValues` and `CellVectorValues`. As the names suggest, `CellScalarValues`
utilizes scalar shape functions and `CellVectorValues` utilizes vectorial shape functions. For a scalar field, the
`CellScalarValues` type should be used. For vector field, both subtypes can be used.

**Arguments:**
* `T`: an optional argument (default to `Float64`) to determine the type the internal data is stored as.
* `quad_rule`: an instance of a [`QuadratureRule`](@ref)
* `func_interpol`: an instance of an [`Interpolation`](@ref) used to interpolate the approximated function
* `geom_interpol`: an optional instance of a [`Interpolation`](@ref) which is used to interpolate the geometry.
  By default linear Lagrange interpolation is used.

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

function default_geometric_interpolation(::Interpolation{dim,shape}) where {dim, shape}
    return Lagrange{dim,shape,1}()
end

# CellScalarValues
#   sdim = spatial dimension
#   rdim = reference element dimension
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

# FIXME sdim should be something like `getdim(value(geom_interpol))``
function CellScalarValues(quad_rule::QuadratureRule, func_interpol::ScalarInterpolation,
        geom_interpol::Interpolation=default_geometric_interpolation(func_interpol); sdim::Int=getdim(func_interpol))
    CellScalarValues(Float64, quad_rule, func_interpol, geom_interpol, Val(sdim))
end

# FIXME sdim should be something like `length(value(geom_interpol))`
function CellScalarValues(valtype::Type{T}, quad_rule::QuadratureRule, func_interpol::ScalarInterpolation,
        geom_interpol::Interpolation=default_geometric_interpolation(func_interpol); sdim::Int=getdim(func_interpol))  where {T}
    CellScalarValues(valtype, quad_rule, func_interpol, geom_interpol, Val(sdim))
end

function CellScalarValues(::Type{T}, quad_rule::QuadratureRule{rdim,shape}, func_interpol::ScalarInterpolation{rdim,shape},
        geom_interpol::Interpolation{rdim,shape}, ::Val{sdim}) where {rdim,T,shape<:AbstractRefShape,sdim}

    n_qpoints = length(getweights(quad_rule))

    # Function interpolation
    n_func_basefuncs = getnbasefunctions(func_interpol)
    N    = fill(zero(T)          * T(NaN), n_func_basefuncs, n_qpoints)
    dNdx = fill(zero(SVector{sdim,T}) * T(NaN), n_func_basefuncs, n_qpoints)
    dNdξ = fill(zero(SVector{rdim,T}) * T(NaN), n_func_basefuncs, n_qpoints)

    # Geometry interpolation
    n_geom_basefuncs = getnbasefunctions(geom_interpol)
    M    = fill(zero(T)          * T(NaN), n_geom_basefuncs, n_qpoints)
    dMdξ = fill(zero(SVector{rdim,T}) * T(NaN), n_geom_basefuncs, n_qpoints)

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
#   vdim = vector dimension (i.e. dimension of evaluation of what `value` should return)
#   sdim = spatial dimension
#   rdim = reference element dimension
#   M1   = number of elements in the matrix dNdx (should be vdim × sdim)
#   M2   = number of elements in the matrix dNdξ (should be vdim × rdim)
struct CellVectorValues{vdim,sdim,rdim,T<:Real,refshape<:AbstractRefShape,M1,M2} <: CellValues{sdim,rdim,T,refshape}
    N::Matrix{SVector{vdim,T}} # vdim
    dNdx::Matrix{SMatrix{vdim,sdim,T,M1}} # vdim × sdim
    dNdξ::Matrix{SMatrix{vdim,rdim,T,M2}} # vdim × rdim
    detJdV::Vector{T}
    M::Matrix{T}
    dMdξ::Matrix{SVector{rdim,T}} # rdim
    qr::QuadratureRule{rdim,refshape,T} #rdim
    # The following fields are deliberately abstract -- they are never used in
    # performance critical code, just stored here for convenience.
    func_interp::VectorInterpolation{vdim,rdim,refshape} # rdim
    geo_interp::Interpolation{rdim,refshape} # rdim
end

# FIXME sdim should be something like `length(value(geom_interpol))`
function CellVectorValues(quad_rule::QuadratureRule, func_interpol::Interpolation, 
        geom_interpol::Interpolation=default_geometric_interpolation(func_interpol); sdim::Int=getdim(geom_interpol), vdim::Int=getdim(func_interpol))
    CellVectorValues(Float64, quad_rule, func_interpol, geom_interpol, Val(sdim))
end

# FIXME sdim should be something like `length(value(geom_interpol))`
function CellVectorValues(valuetype::Type{T}, quad_rule::QuadratureRule, func_interpol::Interpolation, 
        geom_interpol::Interpolation=default_geometric_interpolation(func_interpol); sdim::Int=getdim(geom_interpol), vdim::Int=getdim(func_interpol)) where {T}
    CellVectorValues(valuetype, quad_rule, func_interpol, geom_interpol, Val(sdim))
end

function CellVectorValues(::Type{T}, quad_rule::QuadratureRule{rdim,shape}, func_interpol::VectorInterpolation{vdim,rdim,shape},
        geom_interpol::Interpolation{rdim,shape}, ::Val{sdim}) where {rdim,T,shape<:AbstractRefShape,sdim,vdim}
    n_qpoints = length(getweights(quad_rule))

    M1 = vdim*sdim
    M2 = vdim*rdim

    # Function interpolation
    n_func_basefuncs = getnbasefunctions(func_interpol)
    N    = fill(zero(SVector{vdim,T})         * T(NaN), n_func_basefuncs, n_qpoints)
    dNdx = fill(zero(SMatrix{vdim,sdim,T,M1}) * T(NaN), n_func_basefuncs, n_qpoints)
    dNdξ = fill(zero(SMatrix{vdim,rdim,T,M2}) * T(NaN), n_func_basefuncs, n_qpoints)

    # Geometry interpolation
    n_geom_basefuncs = getnbasefunctions(geom_interpol)
    M    = fill(zero(T)          * T(NaN), n_geom_basefuncs, n_qpoints)
    dMdξ = fill(zero(SVector{rdim,T}) * T(NaN), n_geom_basefuncs, n_qpoints)

    for (qp, ξ) in enumerate(quad_rule.points)
        for basefunc in 1:n_func_basefuncs
            dNdξ[basefunc, qp], N[basefunc, qp] = gradient(ξ -> value(func_interpol, basefunc, ξ), ξ, :all)
        end
        for basefunc in 1:n_geom_basefuncs
            dMdξ[basefunc, qp], M[basefunc, qp] = gradient(ξ -> value(geom_interpol, basefunc, ξ), ξ, :all)
        end
    end

    detJdV = fill(T(NaN), n_qpoints)

    CellVectorValues{vdim,sdim,rdim,T,shape,M1,M2}(N, dNdx, dNdξ, detJdV, M, dMdξ, quad_rule, func_interpol, geom_interpol)
end

"""
Reinit for volumetric elements, i.e. elements whose reference dimension match the spatial dimension.
"""
function reinit!(cv::CellValues{sdim,sdim}, x::AbstractVector{Vec{sdim,T}}) where {sdim,T}
    n_geom_basefuncs = getngeobasefunctions(cv)
    n_func_basefuncs = getnbasefunctions(cv)
    length(x) == n_geom_basefuncs || throw_incompatible_coord_length(length(x), n_geom_basefuncs)
    TdNdx = typeof(cv.dNdx[1, 1])

    @inbounds for i in 1:length(cv.qr.weights)
        w = cv.qr.weights[i]
        fecv_J = zero(Tensor{2,sdim,T})
        for j in 1:n_geom_basefuncs
            fecv_J += x[j] ⊗ tensor_cast(cv.dMdξ[j, i]) # TODO cleanup?
        end
        detJ = det(fecv_J)
        detJ > 0.0 || throw_detJ_not_pos(detJ)
        cv.detJdV[i] = detJ * w
        Jinv = inv(fecv_J)
        for j in 1:n_func_basefuncs
            cv.dNdx[j, i] = TdNdx((tensor_cast(cv.dNdξ[j, i]) ⋅ Jinv).data) # TODO cleanup?
        end
    end
end

# Hotfix to get the dots right for embedded elements.
@inline dothelper(x::V,A::M) where {V<:SVector,M<:Union{SMatrix,MMatrix}} = A'*x
@inline dothelper(B::M1,A::M2) where {M1<:SMatrix,M2<:Union{SMatrix,MMatrix}} = B*A

"""
Embedding determinant for surfaces in 3D.

TLDR: "det(J) =" ||∂x/∂ξ₁ × ∂x/∂ξ₂||₂

The transformation theorem for some function f on a 2D surface in 3D space leads to
  ∫ f ⋅ dS = ∫ f ⋅ (∂x/∂ξ₁ × ∂x/∂ξ₂) dξ₁dξ₂ = ∫ f ⋅ n ||∂x/∂ξ₁ × ∂x/∂ξ₂||₂ dξ₁dξ₂
where ||∂x/∂ξ₁ × ∂x/∂ξ₂||₂ is "detJ" and n is the unit normal.
See e.g. https://scicomp.stackexchange.com/questions/41741/integration-of-d-1-dimensional-functions-on-finite-element-surfaces for simple explanation.
For more details see e.g. the doctoral thesis by Mirza Cenanovic **Finite element methods for surface problems* (2017), Ch. 2 **Trangential Calculus**.
"""
edet(J::MMatrix{3,2,T}) where {T} = norm(J[:,1] × J[:,2])

"""
Embedding determinant for curves in 2D and 3D.

TLDR: "det(J) =" ||∂x/∂ξ||₂

The transformation theorem for some function f on a 1D curve in 2D and 3D space leads to
  ∫ f ⋅ dE = ∫ f ⋅ ∂x/∂ξ dξ = ∫ f ⋅ t ||∂x/∂ξ||₂ dξ
where ||∂x/∂ξ||₂ is "detJ" and t is "the unit tangent".
See e.g. https://scicomp.stackexchange.com/questions/41741/integration-of-d-1-dimensional-functions-on-finite-element-surfaces for simple explanation.
"""
edet(J::Union{MMatrix{2,1,T},MMatrix{3,1,T}}) where {T} = norm(J)

"""
Reinit for embedded elements, i.e. elements whose reference dimension is smaller than the spatial dimension.
"""
function reinit!(cv::CellValues{sdim,rdim}, x::AbstractVector{Vec{sdim,T}}) where {sdim,rdim,T}
    @assert sdim > rdim "This reinit only works for embedded elements. Maybe you swapped the reference and spatial dimensions?"
    n_geom_basefuncs = getngeobasefunctions(cv)
    n_func_basefuncs = getnbasefunctions(cv)
    length(x) == n_geom_basefuncs || throw_incompatible_coord_length(length(x), n_geom_basefuncs)

    @inbounds for i in 1:length(cv.qr.weights)
        w = cv.qr.weights[i]
        fecv_J = zero(MMatrix{sdim,rdim,T}) # TODO replace with MixedTensor (see https://github.com/Ferrite-FEM/Tensors.jl/pull/188)
        for j in 1:n_geom_basefuncs
            #fecv_J += x[j] ⊗ cv.dMdξ[j, i] # TODO via Tensors.jl
            for k in 1:sdim, l in 1:rdim
                fecv_J[k,l] += x[j][k] * cv.dMdξ[j, i][l]
            end
        end
        detJ = edet(fecv_J)
        detJ > 0.0 || throw_detJ_not_pos(detJ)
        cv.detJdV[i] = detJ * w
        # Compute "left inverse" of J
        Jinv = pinv(fecv_J)
        for j in 1:n_func_basefuncs
            #cv.dNdx[j, i] = cv.dNdξ[j, i] ⋅ Jinv # TODO via Tensors.jl
            cv.dNdx[j, i] = dothelper(cv.dNdξ[j, i], Jinv)
        end
    end
end
