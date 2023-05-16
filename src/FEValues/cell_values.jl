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

function default_geometric_interpolation(::Interpolation{dim,shape}) where {dim, shape}
    return Lagrange{dim,shape,1}()
end

"""
TODO docstring.
"""
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

    """
        CellValues{T, N_t, dNdx_t, dNdξ_t, T, dMdξ_t}(qr::QR, ip::IP, gip::GIP)
    Common initializer code for constructing cell values after the types have been determined.
    """
    function CellValues{N_t, dNdx_t, dNdξ_t, T, dMdξ_t}(qr::QR, ip::IP, gip::GIP) where {QR, IP, GIP, N_t, dNdx_t, dNdξ_t, T, dMdξ_t}
        n_qpoints = length(getweights(qr))

        # Field interpolation
        n_func_basefuncs = getnbasefunctions(ip)
        N    = fill(zero(N_t)    * T(NaN), n_func_basefuncs, n_qpoints)
        dNdx = fill(zero(dNdx_t) * T(NaN), n_func_basefuncs, n_qpoints)
        dNdξ = fill(zero(dNdξ_t) * T(NaN), n_func_basefuncs, n_qpoints)

        # Geometry interpolation
        n_geom_basefuncs = getnbasefunctions(gip)
        M    = fill(zero(T)    * T(NaN), n_geom_basefuncs, n_qpoints)
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

        new{IP, N_t, dNdx_t, dNdξ_t, T, dMdξ_t, QR, GIP}(N, dNdx, dNdξ, detJdV, M, dMdξ, qr, ip, gip)
    end

    # hotfix for copy construction
    function CellValues{IP, N_t, dNdx_t, dNdξ_t, T, dMdξ_t, QR, GIP}(N::Matrix{N_t}, dNdx::Matrix{dNdx_t}, dNdξ::Matrix{dNdξ_t}, detJdV::Vector{T}, M::Matrix{T}, dMdξ::Matrix{dMdξ_t}, qr::QR, ip::IP, gip::GIP) where {IP, N_t, dNdx_t, dNdξ_t, T, dMdξ_t, QR, GIP}
        new{IP, N_t, dNdx_t, dNdξ_t, T, dMdξ_t, QR, GIP}(N, dNdx, dNdξ, detJdV, M, dMdξ, qr, ip, gip)
    end
end

"""
    CellValues(qr::QuadratureRule, ip::Interpolation, gip::Interpolation = default_geometric_interpolation(ip))
Constructs an evaluation helper for the most common use cases when we can use Tensors.jl for all geometric and 
function interpolations.
"""
function CellValues(qr::QuadratureRule, ip::Interpolation,
        gip::Interpolation = default_geometric_interpolation(ip))
    return CellValues(Float64, qr, ip, gip)
end

"""
    CellValues(sdim::Int, qr::QuadratureRule, ip::Interpolation, gip::Interpolation = default_geometric_interpolation(ip))
Constructs an evaluation helper for embedded elements. We will use StaticArrays.jl in this case to store everything.
"""
function CellValues(sdim::Int, qr::QuadratureRule, ip::Interpolation,
        gip::Interpolation = default_geometric_interpolation(ip))
    return CellValues(Float64, qr, ip, gip, Val(sdim))
end

"""
    CellValues(::Type{T}, qr::QuadratureRule, ip::Interpolation, gip::ScalarInterpolation = default_geometric_interpolation(ip))
Generic constructor for cell values in the case when we either have a scalar-valued interpolation or
a vector-valued interpolation where the vector dimension matches the reference dimension.
It is also assumed that the spatial dimension is the same as the reference dimension.

    !!!NOTE This doesn't actually work for T != Float64. We have to refactor the constructor to 
            separate the geometry data type from the data type of the approximation evaluation.
"""
function CellValues(::Type{T}, qr::QR, ip::IP, gip::GIP = default_geometric_interpolation(ip)) where {
    dim, shape <: AbstractRefShape#={dim}=#, T,
    QR  <: QuadratureRule{dim, shape},
    IP  <: ScalarInterpolation{dim, shape},
    GIP <: ScalarInterpolation{dim, shape}
}
    # Function interpolation
    N_t    = T
    dNdx_t = dNdξ_t = Vec{dim, T}

    # Geometry interpolation
    #M_t    = T
    dMdξ_t = Vec{dim, T}

    return CellValues{N_t, dNdx_t, dNdξ_t, T, dMdξ_t}(qr, ip, gip)
end

function CellValues(::Type{T}, qr::QR, ip::IP, gip::GIP = default_geometric_interpolation(ip)) where {
    dim, shape <: AbstractRefShape#={dim}=#, T,
    QR  <: QuadratureRule{dim, shape},
    IP  <: VectorInterpolation{dim, dim, shape},
    GIP <: ScalarInterpolation{dim, shape}
}
    # Field interpolation
    N_t    = Vec{dim, T}
    dNdx_t = dNdξ_t = Tensor{2, dim, T, Tensors.n_components(Tensor{2,dim})}

    # Geometry interpolation
    #M_t    = T
    dMdξ_t = Vec{dim, T}

    return CellValues{N_t, dNdx_t, dNdξ_t, T, dMdξ_t}(qr, ip, gip)
end

function CellValues(::Type{T}, qr::QR, ip::IP, gip::GIP = default_geometric_interpolation(ip)) where {
    vdim, dim, shape <: AbstractRefShape#={dim}=#, T,
    QR  <: QuadratureRule{dim, shape},
    IP  <: VectorInterpolation{vdim, dim, shape},
    GIP <: ScalarInterpolation{dim, shape}
}
    # Field interpolation
    N_t    = SVector{vdim, T}
    dNdx_t = MMatrix{vdim, dim, T, vdim*dim}
    dNdξ_t = SMatrix{vdim, dim, T, vdim*dim}

    # Geometry interpolation
    #M_t    = T
    dMdξ_t = Vec{dim, T}

    return CellValues{N_t, dNdx_t, dNdξ_t, T, dMdξ_t}(qr, ip, gip)
end

function reinit!(cv::CellValues{<:Any, N_t, dNdx_t}, x::AbstractVector{Vec{dim,T}}) where {
    dim, T,
    N_t    <: Union{Number,   Vec{dim}},
    dNdx_t <: Union{Vec{dim}, Tensor{2, dim}}
}
    n_geom_basefuncs = getngeobasefunctions(cv)
    n_func_basefuncs = getnbasefunctions(cv)
    length(x) == n_geom_basefuncs || throw_incompatible_coord_length(length(x), n_geom_basefuncs)
    TdNdx = typeof(cv.dNdx[1, 1])

    @inbounds for i in 1:length(cv.qr.weights)
        w = cv.qr.weights[i]
        fecv_J = zero(Tensor{2,dim,T})
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

# Hotfix to get the dots right for embedded elements until mixed tensors are merged.
@inline dothelper(x::V,A::M) where {V<:SVector,M<:Union{SMatrix,MMatrix}} = A'*x
@inline dothelper(B::M1,A::M2) where {M1<:SMatrix,M2<:Union{SMatrix,MMatrix}} = B*A

"""
Generic constructor for cell values for embedded elements.
"""
function CellValues(::Type{T}, qr::QR, ip::IP, gip::GIP, ::Val{sdim}) where {
    sdim, rdim, shape <: AbstractRefShape#={rdim}=#, T,
    QR  <: QuadratureRule{rdim, shape},
    IP  <: ScalarInterpolation{rdim, shape},
    GIP <: ScalarInterpolation{rdim, shape}
}
    # Function interpolation
    N_t    = T
    dNdx_t = MVector{sdim, T}
    dNdξ_t = SVector{rdim, T}

    # Geometry interpolation
    #M_t    = T
    dMdξ_t = Vec{rdim, T}

    return CellValues{N_t, dNdx_t, dNdξ_t, T, dMdξ_t}(qr, ip, gip)
end

function CellValues(::Type{T}, qr::QR, ip::IP, gip::GIP, ::Val{sdim}) where {
    sdim, vdim, rdim, shape <: AbstractRefShape#={rdim}=#, T,
    QR  <: QuadratureRule{rdim, shape},
    IP  <: VectorInterpolation{vdim, rdim, shape},
    GIP <: ScalarInterpolation{rdim, shape}
}
    # Function interpolation
    N_t    = SVector{vdim, T}
    dNdx_t = MMatrix{vdim, sdim, T, vdim*sdim}
    dNdξ_t = SMatrix{vdim, rdim, T, rdim*sdim}

    # Geometry interpolation
    #M_t    = T
    dMdξ_t = Vec{rdim, T}

    return CellValues{N_t, dNdx_t, dNdξ_t, T, dMdξ_t}(qr, ip, gip)
end

"""
Embedding determinant for surfaces in 3D.

TLDR: "det(J) =" ||∂x/∂ξ₁ × ∂x/∂ξ₂||₂

The transformation theorem for some function f on a 2D surface in 3D space leads to
  ∫ f ⋅ dS = ∫ f ⋅ (∂x/∂ξ₁ × ∂x/∂ξ₂) dξ₁dξ₂ = ∫ f ⋅ n ||∂x/∂ξ₁ × ∂x/∂ξ₂||₂ dξ₁dξ₂
where ||∂x/∂ξ₁ × ∂x/∂ξ₂||₂ is "detJ" and n is the unit normal.
See e.g. https://scicomp.stackexchange.com/questions/41741/integration-of-d-1-dimensional-functions-on-finite-element-surfaces for simple explanation.
For more details see e.g. the doctoral thesis by Mirza Cenanovic **Finite element methods for surface problems* (2017), Ch. 2 **Trangential Calculus**.
"""
edet(J::MMatrix{3,2,T,6}) where {T} = norm(J[:,1] × J[:,2])

"""
Embedding determinant for curves in 2D and 3D.

TLDR: "det(J) =" ||∂x/∂ξ||₂

The transformation theorem for some function f on a 1D curve in 2D and 3D space leads to
  ∫ f ⋅ dE = ∫ f ⋅ ∂x/∂ξ dξ = ∫ f ⋅ t ||∂x/∂ξ||₂ dξ
where ||∂x/∂ξ||₂ is "detJ" and t is "the unit tangent".
See e.g. https://scicomp.stackexchange.com/questions/41741/integration-of-d-1-dimensional-functions-on-finite-element-surfaces for simple explanation.
"""
edet(J::Union{MMatrix{2,1,T,2},MMatrix{3,1,T,3}}) where {T} = norm(J)

"""
Reinit for embedded elements, i.e. elements whose reference dimension is smaller than the spatial dimension.
"""
function reinit!(cv::CellValues{<:Any, N_t, dNdx_t, dNdξ_t}, x::AbstractVector{Vec{sdim,T}}) where {
    rdim, sdim, vdim, T, matsize1, matsize2,
    N_t    <: Union{Number,   Vec{vdim}},
    dNdx_t <: Union{MVector{sdim, T}, MMatrix{vdim, sdim, T, matsize1}},
    dNdξ_t <: Union{SVector{rdim, T}, SMatrix{vdim, rdim, T, matsize2}}
}
    @assert sdim > rdim "This reinit only works for embedded elements. Maybe you swapped the reference and spatial dimensions?"
    n_geom_basefuncs = getngeobasefunctions(cv)
    n_func_basefuncs = getnbasefunctions(cv)
    length(x) == n_geom_basefuncs || throw_incompatible_coord_length(length(x), n_geom_basefuncs)

    @inbounds for i in 1:length(cv.qr.weights)
        w = cv.qr.weights[i]
        fecv_J = zero(MMatrix{sdim,rdim,T,sdim*rdim}) # TODO replace with MixedTensor (see https://github.com/Ferrite-FEM/Tensors.jl/pull/188)
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
    return nothing
end
