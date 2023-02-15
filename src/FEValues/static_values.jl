using StaticArrays

struct StaticQuadratureRule{dim,RefShape,T,ngp}
    weights::NTuple{ngp,T}
    points::NTuple{ngp, Vec{dim,T}}
end
function StaticQuadratureRule(qr::QuadratureRule{dim,RefShape,T}) where {dim,RefShape,T}
    weights = tuple(getweights(qr)...)
    points = tuple(getpoints(qr)...)
    ngp = length(weights)
    return StaticQuadratureRule{dim,RefShape,T,ngp}(weights, points)
end

struct QuadPointValuesAll{T,TT,NS}
    N::NTuple{NS,T}
    dNdξ::NTuple{NS,TT}
end

struct CellValuesAll{T<:Number,TT<:AbstractTensor,RefShape<:AbstractRefShape,nip_f,nip_g,ngp}
    qp_fun::NTuple{ngp,QuadPointValuesAll{T,TT,nip_f}}
    qp_geo::NTuple{ngp,QuadPointValuesAll{T,TT,nip_g}}
    qr::StaticQuadratureRule{dim,RefShape,T,ngp}
end

struct QuadPointValuesEach{T,TT,NS}
    detJdV::T
    dNdx::NTuple{NS,TT}
end

struct CellValuesEach{T<:Number,TT<:AbstractTensor,nip_f,ngp}
    gp_fun::NTuple{ngp, QuadPointValuesEach{T,TT,nip_f}}
end

