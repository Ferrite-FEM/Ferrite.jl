"""
    CellValues([::Type{T},] quad_rule::QuadratureRule, func_interpol::Interpolation, [geom_interpol::Interpolation])

A `CellValues` object facilitates the process of evaluating values of shape functions, gradients of shape functions,
values of nodal functions, gradients and divergences of nodal functions etc. in the finite element cell.

**Arguments:**
* `T`: an optional argument (default to `Float64`) to determine the type the internal data is stored as.
* `quad_rule`: an instance of a [`QuadratureRule`](@ref)
* `func_interpol`: an instance of an [`Interpolation`](@ref) used to interpolate the approximated function
* `geom_interpol`: an optional instance of a [`Interpolation`](@ref) which is used to interpolate the geometry.
  By default linear Lagrange interpolation is used. For embedded elements the geometric interpolations should
  be vectorized to the spatial dimension.

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

function default_geometric_interpolation(::Interpolation{shape}) where {dim, shape <: AbstractRefShape{dim}}
    return VectorizedInterpolation{dim}(Lagrange{shape, 1}())
end

struct CellValues{IP, N_t, dNdx_t, dNdξ_t, T, dMdξ_t, QR, GIP} <: AbstractCellValues
    N::Matrix{N_t}
    dNdx::Matrix{dNdx_t}
    dNdξ::Matrix{dNdξ_t}
    detJdV::Vector{T}
    M::Matrix{T}
    dMdξ::Matrix{dMdξ_t}
    qr::QR
    ip::IP
    gip::GIP
end

# Common initializer code for constructing CellValues after the types have been determined
function CellValues{IP, N_t, dNdx_t, dNdξ_t, T, dMdξ_t, QR, GIP}(qr::QR, ip::IP, gip::GIP) where {
    IP, N_t, dNdx_t, dNdξ_t, T, dMdξ_t, QR, GIP,
}
    @assert isconcretetype(IP)     && isconcretetype(N_t) && isconcretetype(dNdx_t) &&
            isconcretetype(dNdξ_t) && isconcretetype(T)   && isconcretetype(dMdξ_t) &&
            isconcretetype(QR)     && isconcretetype(GIP)
    n_qpoints = getnquadpoints(qr)

    # Field interpolation
    n_func_basefuncs = getnbasefunctions(ip)
    N    = fill(zero(N_t)    * T(NaN), n_func_basefuncs, n_qpoints)
    dNdx = fill(zero(dNdx_t) * T(NaN), n_func_basefuncs, n_qpoints)
    dNdξ = fill(zero(dNdξ_t) * T(NaN), n_func_basefuncs, n_qpoints)

    # Geometry interpolation
    n_geom_basefuncs = getnbasefunctions(gip)
    M    = fill(zero(T)      * T(NaN), n_geom_basefuncs, n_qpoints)
    dMdξ = fill(zero(dMdξ_t) * T(NaN), n_geom_basefuncs, n_qpoints)

    for (qp, ξ) in pairs(getpoints(qr))
        for basefunc in 1:n_func_basefuncs
            dNdξ[basefunc, qp], N[basefunc, qp] = shape_gradient_and_value(ip, ξ, basefunc)
        end
        for basefunc in 1:n_geom_basefuncs
            dMdξ[basefunc, qp], M[basefunc, qp] = shape_gradient_and_value(gip, ξ, basefunc)
        end
    end

    detJdV = fill(T(NaN), n_qpoints)

    CellValues{IP, N_t, dNdx_t, dNdξ_t, T, dMdξ_t, QR, GIP}(N, dNdx, dNdξ, detJdV, M, dMdξ, qr, ip, gip)
end

# Common entry point that fills in the numeric type and geometric interpolation
function CellValues(qr::QuadratureRule, ip::Interpolation,
        gip::Interpolation = default_geometric_interpolation(ip))
    return CellValues(Float64, qr, ip, gip)
end

# Common entry point that fills in the geometric interpolation
function CellValues(::Type{T}, qr::QuadratureRule, ip::Interpolation) where {T}
    return CellValues(T, qr, ip, default_geometric_interpolation(ip))
end

# Common entry point that vectorizes an input scalar geometric interpolation
function CellValues(::Type{T}, qr::QuadratureRule, ip::Interpolation, sgip::ScalarInterpolation) where {T}
    return CellValues(T, qr, ip, VectorizedInterpolation(sgip))
end

# Entrypoint for `ScalarInterpolation`s (rdim == sdim)
function CellValues(::Type{T}, qr::QR, ip::IP, gip::VGIP) where {
    dim, shape <: AbstractRefShape{dim}, T,
    QR   <: QuadratureRule{shape},
    IP   <: ScalarInterpolation{shape},
    GIP  <: ScalarInterpolation{shape},
    VGIP <: VectorizedInterpolation{dim, shape, <:Any, GIP},
}
    # Function interpolation
    N_t    = T
    dNdx_t = dNdξ_t = Vec{dim, T}
    # Geometry interpolation
    M_t    = T
    dMdξ_t = Vec{dim, T}
    return CellValues{IP, N_t, dNdx_t, dNdξ_t, M_t, dMdξ_t, QR, GIP}(qr, ip, gip.ip)
end

# Entrypoint for `VectorInterpolation`s (vdim == rdim == sdim)
function CellValues(::Type{T}, qr::QR, ip::IP, gip::VGIP) where {
    dim, shape <: AbstractRefShape{dim}, T,
    QR  <: QuadratureRule{shape},
    IP  <: VectorInterpolation{dim, shape},
    GIP <: ScalarInterpolation{shape},
    VGIP <: VectorizedInterpolation{dim, shape, <:Any, GIP},
}
    # Field interpolation
    N_t    = Vec{dim, T}
    dNdx_t = dNdξ_t = Tensor{2, dim, T, Tensors.n_components(Tensor{2,dim})}
    # Geometry interpolation
    M_t    = T
    dMdξ_t = Vec{dim, T}
    return CellValues{IP, N_t, dNdx_t, dNdξ_t, M_t, dMdξ_t, QR, GIP}(qr, ip, gip.ip)
end

# Entrypoint for `VectorInterpolation`s (vdim != rdim == sdim)
function CellValues(::Type{T}, qr::QR, ip::IP, vgip::VGIP) where {
    vdim, dim, shape <: AbstractRefShape{dim}, T,
    QR  <: QuadratureRule{shape},
    IP  <: VectorInterpolation{vdim, shape},
    GIP <: ScalarInterpolation{shape},
    VGIP <: VectorizedInterpolation{dim, shape, <:Any, GIP},
}
    # Field interpolation
    N_t    = SVector{vdim, T}
    dNdx_t = dNdξ_t = SMatrix{vdim, dim, T, vdim*dim}
    # Geometry interpolation
    M_t    = T
    dMdξ_t = Vec{dim, T}
    return CellValues{IP, N_t, dNdx_t, dNdξ_t, M_t, dMdξ_t, QR, GIP}(qr, ip, vgip.ip)
end

# reinit! for regular (non-embedded) elements (rdim == sdim)
function reinit!(cv::CellValues{<:Any, N_t, dNdx_t, dNdξ_t}, x::AbstractVector{Vec{dim,T}}) where {
    dim, T, vdim,
    N_t    <: Union{Number,   Vec{dim},       SVector{vdim}     },
    dNdx_t <: Union{Vec{dim}, Tensor{2, dim}, SMatrix{vdim, dim}},
    dNdξ_t <: Union{Vec{dim}, Tensor{2, dim}, SMatrix{vdim, dim}},
}
    n_geom_basefuncs = getngeobasefunctions(cv)
    n_func_basefuncs = getnbasefunctions(cv)
    length(x) == n_geom_basefuncs || throw_incompatible_coord_length(length(x), n_geom_basefuncs)

    @inbounds for (i, w) in pairs(getweights(cv.qr))
        fecv_J = zero(Tensor{2,dim,T})
        for j in 1:n_geom_basefuncs
            fecv_J += x[j] ⊗ cv.dMdξ[j, i]
        end
        detJ = det(fecv_J)
        detJ > 0.0 || throw_detJ_not_pos(detJ)
        cv.detJdV[i] = detJ * w
        Jinv = inv(fecv_J)
        for j in 1:n_func_basefuncs
            # cv.dNdx[j, i] = cv.dNdξ[j, i] ⋅ Jinv
            cv.dNdx[j, i] = dothelper(cv.dNdξ[j, i], Jinv)
        end
    end
end

# Hotfix to get the dots right for embedded elements until mixed tensors are merged.
# Scalar/Vector interpolations with sdim == rdim (== vdim)
@inline dothelper(A, B) = A ⋅ B
# Vector interpolations with sdim == rdim != vdim
@inline dothelper(A::SMatrix{vdim, dim}, B::Tensor{2, dim}) where {vdim, dim} = A * SMatrix{dim, dim}(B)
# Scalar interpolations with sdim > rdim
@inline dothelper(A::SVector{rdim}, B::SMatrix{rdim, sdim}) where {rdim, sdim} = B' * A
# Vector interpolations with sdim > rdim
@inline dothelper(B::SMatrix{vdim, rdim}, A::SMatrix{rdim, sdim}) where {vdim, rdim, sdim} = B * A

# Entrypoint for embedded `ScalarInterpolation`s (rdim < sdim)
function CellValues(::Type{T}, qr::QR, ip::IP, gip::VGIP) where {
    sdim, rdim, shape <: AbstractRefShape{rdim}, T,
    QR  <: QuadratureRule{shape},
    IP  <: ScalarInterpolation{shape},
    GIP <: ScalarInterpolation{shape},
    VGIP <: VectorizedInterpolation{sdim, shape, <:Any, GIP},
}
    @assert sdim > rdim
    # Function interpolation
    N_t    = T
    dNdx_t = SVector{sdim, T}
    dNdξ_t = SVector{rdim, T}
    # Geometry interpolation
    M_t    = T
    dMdξ_t = Vec{rdim, T}
    return CellValues{IP, N_t, dNdx_t, dNdξ_t, M_t, dMdξ_t, QR, GIP}(qr, ip, gip.ip)
end

# Entrypoint for embedded `VectorInterpolation`s (rdim < sdim)
function CellValues(::Type{T}, qr::QR, ip::IP, gip::VGIP) where {
    sdim, vdim, rdim, shape <: AbstractRefShape{rdim}, T,
    QR  <: QuadratureRule{shape},
    IP  <: VectorInterpolation{vdim, shape},
    GIP <: ScalarInterpolation{shape},
    VGIP <: VectorizedInterpolation{sdim, shape, <:Any, GIP},
}
    @assert sdim > rdim
    # Function interpolation
    N_t    = SVector{vdim, T}
    dNdx_t = SMatrix{vdim, sdim, T, vdim*sdim}
    dNdξ_t = SMatrix{vdim, rdim, T, vdim*rdim}
    # Geometry interpolation
    M_t    = T
    dMdξ_t = Vec{rdim, T}
    return CellValues{IP, N_t, dNdx_t, dNdξ_t, M_t, dMdξ_t, QR, GIP}(qr, ip, gip.ip)
end

"""
    embedding_det(J::SMatrix{3, 2})

Embedding determinant for surfaces in 3D.

TLDR: "det(J) =" ||∂x/∂ξ₁ × ∂x/∂ξ₂||₂

The transformation theorem for some function f on a 2D surface in 3D space leads to
  ∫ f ⋅ dS = ∫ f ⋅ (∂x/∂ξ₁ × ∂x/∂ξ₂) dξ₁dξ₂ = ∫ f ⋅ n ||∂x/∂ξ₁ × ∂x/∂ξ₂||₂ dξ₁dξ₂
where ||∂x/∂ξ₁ × ∂x/∂ξ₂||₂ is "detJ" and n is the unit normal.
See e.g. https://scicomp.stackexchange.com/questions/41741/integration-of-d-1-dimensional-functions-on-finite-element-surfaces for simple explanation.
For more details see e.g. the doctoral thesis by Mirza Cenanovic **Finite element methods for surface problems* (2017), Ch. 2 **Trangential Calculus**.
"""
embedding_det(J::SMatrix{3,2}) = norm(J[:,1] × J[:,2])

"""
    embedding_det(J::Union{SMatrix{2, 1}, SMatrix{3, 1}})

Embedding determinant for curves in 2D and 3D.

TLDR: "det(J) =" ||∂x/∂ξ||₂

The transformation theorem for some function f on a 1D curve in 2D and 3D space leads to
  ∫ f ⋅ dE = ∫ f ⋅ ∂x/∂ξ dξ = ∫ f ⋅ t ||∂x/∂ξ||₂ dξ
where ||∂x/∂ξ||₂ is "detJ" and t is "the unit tangent".
See e.g. https://scicomp.stackexchange.com/questions/41741/integration-of-d-1-dimensional-functions-on-finite-element-surfaces for simple explanation.
"""
embedding_det(J::Union{SMatrix{2, 1}, SMatrix{3, 1}}) = norm(J)

# reinit! for embedded elements, rdim < sdim
function reinit!(cv::CellValues{<:Any, N_t, dNdx_t, dNdξ_t}, x::AbstractVector{Vec{sdim,T}}) where {
    rdim, sdim, vdim, T,
    N_t    <: Union{Number,           SVector{vdim}},
    dNdx_t <: Union{SVector{sdim, T}, SMatrix{vdim, sdim, T}},
    dNdξ_t <: Union{SVector{rdim, T}, SMatrix{vdim, rdim, T}},
}
    @assert sdim > rdim "This reinit only works for embedded elements. Maybe you swapped the reference and spatial dimensions?"
    n_geom_basefuncs = getngeobasefunctions(cv)
    n_func_basefuncs = getnbasefunctions(cv)
    length(x) == n_geom_basefuncs || throw_incompatible_coord_length(length(x), n_geom_basefuncs)

    @inbounds for (i, w) in pairs(getweights(cv.qr))
        fecv_J = zero(MMatrix{sdim, rdim, T}) # TODO replace with MixedTensor (see https://github.com/Ferrite-FEM/Tensors.jl/pull/188)
        for j in 1:n_geom_basefuncs
            #fecv_J += x[j] ⊗ cv.dMdξ[j, i] # TODO via Tensors.jl
            for k in 1:sdim, l in 1:rdim
                fecv_J[k, l] += x[j][k] * cv.dMdξ[j, i][l]
            end
        end
        fecv_J = SMatrix(fecv_J)
        detJ = embedding_det(fecv_J)
        detJ > 0.0 || throw_detJ_not_pos(detJ)
        cv.detJdV[i] = detJ * w
        # Compute "left inverse" of J
        Jinv = pinv(fecv_J)
        for j in 1:n_func_basefuncs
            #cv.dNdx[j, i] = cv.dNdξ[j, i] ⋅ Jinv # TODO via Tensors.jl
            cv.dNdx[j, i] = dothelper(cv.dNdξ[j, i], Jinv)
        end
    end
    return nothing
end

function Base.show(io::IO, m::MIME"text/plain", cv::CellValues)
    println(io, "CellValues with")
    println(io, "- Quadrature rule with ", getnquadpoints(cv), " points")
    print(io, "- Function interpolation: "); show(io, m, cv.ip)
    println(io)
    print(io, "- Geometric interpolation: "); show(io, m, cv.gip)
end