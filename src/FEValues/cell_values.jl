"""
    CellValues([::Type{T}], quad_rule::QuadratureRule, func_interpol::Interpolation, [geom_interpol::Interpolation])

A `CellValues` object facilitates the process of evaluating values of shape functions, gradients of shape functions,
values of nodal functions, gradients and divergences of nodal functions etc. in the finite element cell.

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
CellValues

function default_geometric_interpolation(::Interpolation{shape}) where {shape}
    return Lagrange{shape,1}()
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

# (Scalar|Vector)Interpolation, (vdim ==) refdim == spacedim -> Tensors
function CellValues(qr::QuadratureRule, ip::Interpolation,
                    gip::Interpolation = default_geometric_interpolation(ip))
    return CellValues(Float64, qr, ip, gip)
end
# TODO: This doesn't actually work for T != Float64
function CellValues(::Type{T}, qr::QR, ip::IP, gip::GIP = default_geometric_interpolation(ip)) where {
    dim, shape <: AbstractRefShape{dim}, T,
    QR  <: QuadratureRule{dim, shape},
    IP  <: Union{ScalarInterpolation{shape}, VectorInterpolation{dim, shape}},
    GIP <: ScalarInterpolation{shape}
}
    n_qpoints = length(getweights(qr))

    # Function interpolation
    if IP <: ScalarInterpolation
        N_t    = T
        dNdx_t = dNdξ_t = Vec{dim, T}
    else # IP <: VectorInterpolation
        N_t    = Vec{dim, T}
        dNdx_t = dNdξ_t = Tensor{2, dim, T, Tensors.n_components(Tensor{2,dim})}
    end
    n_func_basefuncs = getnbasefunctions(ip)
    N    = fill(zero(N_t)    * T(NaN), n_func_basefuncs, n_qpoints)
    dNdx = fill(zero(dNdx_t) * T(NaN), n_func_basefuncs, n_qpoints)
    dNdξ = fill(zero(dNdξ_t) * T(NaN), n_func_basefuncs, n_qpoints)

    # Geometry interpolation
    M_t    = T
    dMdξ_t = Vec{dim, T}
    n_geom_basefuncs = getnbasefunctions(gip)
    M    = fill(zero(M_t)    * T(NaN), n_geom_basefuncs, n_qpoints)
    dMdξ = fill(zero(dMdξ_t) * T(NaN), n_geom_basefuncs, n_qpoints)

    for (qp, ξ) in pairs(getpoints(qr))
        for basefunc in 1:n_func_basefuncs
            dNdξ[basefunc, qp], N[basefunc, qp] = gradient(ξ -> value(ip, basefunc, ξ), ξ, :all)
        end
        for basefunc in 1:n_geom_basefuncs
            dMdξ[basefunc, qp], M[basefunc, qp] = gradient(ξ -> value(gip, basefunc, ξ), ξ, :all)
        end
    end

    detJdV = fill(T(NaN), n_qpoints)

    return CellValues{IP, N_t, dNdx_t, dNdξ_t, T, dMdξ_t, QR, GIP}(N, dNdx, dNdξ, detJdV, M, dMdξ, qr, ip, gip)
end

function reinit!(cv::CellValues{<:Any, N_t, dNdx_t}, x::AbstractVector{Vec{dim,T}}) where {
    dim, T,
    N_t    <: Union{Number,   Vec{dim}},
    dNdx_t <: Union{Vec{dim}, Tensor{2, dim}}
}
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
    return nothing
end
