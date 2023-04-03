# Idea
# Collect multiple function values inside one object, as the geometry updates will be the same.
# These FunctionScalarValues/FunctionVectorValues can also be used for FaceValues, but requires 
# face values to contain vectors of these for each face (which makes sense especially for e.g. wedge elements)
# In addition to the potential speed improvements, this structure has the following user side improvements
# - For each SubDofHandler, only one cellvalues object 
# - For the loop, only one quadrature point (nothing preventing separate quad-rules if required by using separate cellvalues)
# - Consistency between each CellValues object ensured automatically (i.e. no strange bugs from differences in e.g. quadrules)
# - Each FunctionValues object accessible via a symbol (stored in NamedTuple).

struct GeometryValues{sdim,T<:Real,RefShape<:AbstractRefShape}
    detJdV::Vector{T}
    M::Matrix{T}
    dMdξ::Matrix{Vec{dim,T}}
    ip::Interpolation{sdim,RefShape}
end

abstract type FunctionValues{dim,T,RefShape} end

struct FunctionScalarValues{dim,T<:Real,RefShape<:AbstractRefShape} <: FunctionValues{dim,T,RefShape}
    N::Matrix{T}
    dNdx::Matrix{Vec{dim,T}}
    dNdξ::Matrix{Vec{dim,T}}
    ip::Interpolation{dim,RefShape}
end

struct FunctionVectorValues{dim,T<:Real,RefShape<:AbstractRefShape,M} <: FunctionValues{dim,T,RefShape}
    N::Matrix{Vec{dim,T}}
    dNdx::Matrix{Tensor{2,dim,T,M}}
    dNdξ::Matrix{Tensor{2,dim,T,M}}
    ip::Interpolation{dim,RefShape}
end

struct CellMultiValues{dim,T,RefShape,FVS<:NamedTuple}
    geo_values::GeometryValues{dim,T,RefShape}
    fun_values::FVS
    qr::QuadratureRule{dim,RefShape,T}
end

function reinit!(cv::CellMultiValues{dim}, x::AbstractVector{Vec{dim,T}}) where {dim,T}
    n_geom_basefuncs = getngeobasefunctions(cv)
    length(x) == n_geom_basefuncs || throw_incompatible_coord_length(length(x), n_geom_basefuncs)

    @inbounds for (i, w) in cv.qr.weights
        fecv_J = zero(Tensor{2,dim})
        for j in 1:n_geom_basefuncs
            fecv_J += x[j] ⊗ cv.dMdξ[j, i]
        end
        detJ = det(fecv_J)
        detJ > 0.0 || throw_detJ_not_pos(detJ)
        cv.geo_values.detJdV[i] = detJ * w
        Jinv = inv(fecv_J)
        for fv in cv.fun_values
            _update_dNdx!(fv, i, Jinv)
        end
    end
end

function _update_dNdx!(fv::FunctionValues{dim}, i::Int, Jinv::Tensor{2,dim}) where dim
    @inbounds for j in 1:getnbasefunctions(fv)
        cv.dNdx[j, i] = cv.dNdξ[j, i] ⋅ Jinv
    end
end
