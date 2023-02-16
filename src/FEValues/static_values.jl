# To be created at start of simulation 
struct StaticQuadratureRule{dim,RefShape,T,Nqp}
    weights::NTuple{Nqp,T}
    points::NTuple{Nqp, Vec{dim,T}}
end
function StaticQuadratureRule(qr::QuadratureRule{dim,RefShape,T}) where {dim,RefShape,T}
    weights = tuple(getweights(qr)...)
    points = tuple(getpoints(qr)...)
    Nqp = length(weights)
    return StaticQuadratureRule{dim,RefShape,T,Nqp}(weights, points)
end

getpoints(qr::StaticQuadratureRule) = qr.points
getweights(qr::StaticQuadratureRule) = qr.weights
getnquadpoints(qr::StaticQuadratureRule) = length(qr.weights)

struct QuadPointValuesAll{T,TT,NS}
    N::NTuple{NS,T}
    dNdξ::NTuple{NS,TT}
end
function QuadPointValuesAll(::Val{NS}, ::Val{:scalar}, ip::Interpolation, ξ::TX) where {NS,TX<:Vec{<:Any,T}} where T
    @assert NS == getnbasefunctions(ip)
    N = ntuple(i->value(ip, i, ξ), NS)
    dNdξ = ntuple(i->gradient(z->value(ip, i, z), ξ), NS)
    return QuadPointValuesAll{T,TX,NS}(N, dNdξ)
end

function QuadPointValuesAll(::Val{NS}, ::Val{:vector}, ip::Interpolation, ξ::TX) where {NS,TX<:Vec{dim}} where dim
    @assert NS == getnbasefunctions(ip)*dim
    N_scalar = ntuple(i->value(ip, i, ξ), getnbasefunctions(ip))
    dNdξ_scalar = ntuple(i->gradient(z->value(ip, i, z), ξ), getnbasefunctions(ip))

    function construct_vectorspace(scalar_var::Tuple, k::Int)
        i = rem(k-1, dim) + 1   # index in 1:getnbasefunctions(ip)
        j = div(k-1, dim) + 1   # index in 1:dim
        ej = Vec{dim}(l->l==j ? 1.0 : 0.0)
        return ej ⊗ scalar_var[i]
    end
    N = ntuple(k->construct_vectorspace(N_scalar, k), NS)
    dNdξ = ntuple(k->construct_vectorspace(dNdξ_scalar, k), NS)
    return QuadPointValuesAll{TX,eltype(dNdξ),NS}(N, dNdξ)
end

shape_value(qp::QuadPointValuesAll, base_function::Int) = qp.N[base_function]
getnbasefunctions(::QuadPointValuesAll{<:Any,<:Any,N}) where N = N

struct CellValuesAll{dim, T<:Number, RefShape<:AbstractRefShape, Nqp, QPF<:QuadPointValuesAll, QPG<:QuadPointValuesAll}
    qp_fun::NTuple{Nqp,QPF}
    qp_geo::NTuple{Nqp,QPG}
    qr::StaticQuadratureRule{dim,RefShape,T,Nqp}
end
function CellValuesAll(type::Val{:scalar}, qr::StaticQuadratureRule, ip_fun::Interpolation, ip_geo::Interpolation=ip_fun)
    return CellValuesAll(type, Val(getnbasefunctions(ip_fun)), Val(getnbasefunctions(ip_geo)), qr, ip_fun, ip_geo)
end
function CellValuesAll(type::Val{:vector}, qr::StaticQuadratureRule{dim}, ip_fun::Interpolation, ip_geo::Interpolation=ip_fun) where dim
    return CellValuesAll(type, Val(dim*getnbasefunctions(ip_fun)), Val(getnbasefunctions(ip_geo)), qr, ip_fun, ip_geo)
end
function CellValuesAll(type::Val, ::Val{Nfun}, ::Val{Ngeo}, qr, ip_fun, ip_geo) where {Nfun, Ngeo}
    qp_fun = map(ξ -> QuadPointValuesAll(Val(Nfun), type, ip_fun, ξ), getpoints(qr))
    qp_geo = map(ξ -> QuadPointValuesAll(Val(Ngeo), Val(:scalar), ip_geo, ξ), getpoints(qr))
    return CellValuesAll(qp_fun, qp_geo, qr)
end

shape_value(cva::CellValuesAll, q_point, base_function) = shape_value(cva.qp_fun[q_point], base_function)
getnbasefunctions(cva::CellValuesAll) = getnbasefunctions(first(cva.qp_fun))
getngeobasefunctions(cva::CellValuesAll) = getnbasefunctions(first(cva.qp_geo))
getnquadpoints(cva::CellValuesAll) = getnquadpoints(cva.qr)

# To be created for each new cell
struct QuadPointValuesEach{T,TT,NS}
    detJdV::T
    dNdx::NTuple{NS,TT}
end
function QuadPointValuesEach(qp_fun, qp_geo, w::Number, x::AbstractVector{<:Vec{dim,T}}) where {dim,T}

    fecv_J = zero(Tensor{2,dim,T})
    for (xj, dMdξ) in zip(x, qp_geo.dNdξ)
        fecv_J += xj ⊗ dMdξ
    end
    detJ = det(fecv_J)
    detJ > 0.0 || throw_detJ_not_pos(detJ)
    Jinv = inv(fecv_J)
    dNdx = map(dNdξ -> dNdξ ⋅ Jinv, qp_fun.dNdξ)

    return QuadPointValuesEach(detJ * w, dNdx)
end

getdetJdV(qpv::QuadPointValuesEach) = qpv.detJdV
shape_gradient(qpv::QuadPointValuesEach, base_function::Int) = qpv.dNdx[base_function]

struct CellValuesEach{T<:Number,TT<:AbstractTensor,Nfun,Nqp}
    qp_fun::NTuple{Nqp, QuadPointValuesEach{T,TT,Nfun}}
end

function CellValuesEach(cva::CellValuesAll, x::AbstractVector)
    return CellValuesEach(QuadPointValuesEach.(cva.qp_fun, cva.qp_geo, getweights(cva.qr), (x,)))
end
shape_gradient(cve::CellValuesEach, q_point, base_function) = shape_gradient(cve.qp_fun[q_point], base_function)
getdetJdV(cve::CellValuesEach, q_point) = getdetJdV(cve.qp_fun[q_point])

struct StaticCellValues{dim, T, RefShape, CVA<:CellValuesAll, CVE} <: CellValues{dim,T,RefShape}
    cv_all::CVA
    cv_each::CVE
end

function StaticCellValues(cv::CellScalarValues{dim,T,RefShape}) where {dim,T,RefShape}
    qr = StaticQuadratureRule(cv.qr)
    cv_all = CellValuesAll(Val(:scalar), qr, cv.func_interp, cv.geo_interp) # Inefficient, but only during setup
    cv_each = CellValuesEach(cv_all, zeros(Vec{dim}, getngeobasefunctions(cv)))
    return StaticCellValues{dim,T,RefShape,typeof(cv_all),typeof(cv_each)}(cv_all, cv_each)
end

function StaticCellValues(cv::CellVectorValues{dim,T,RefShape}) where {dim,T,RefShape}
    qr = StaticQuadratureRule(cv.qr)
    cv_all = CellValuesAll(Val(:vector), qr, cv.func_interp, cv.geo_interp) # Inefficient, but only during setup
    cv_each = CellValuesEach(cv_all, reference_coordinates(cv.geo_interp))
    return StaticCellValues{dim,T,RefShape,typeof(cv_all),typeof(cv_each)}(cv_all, cv_each)
end

getdetJdV(cv::StaticCellValues, qp_point::Int) = getdetJdV(cv.cv_each, qp_point)
shape_gradient(cv::StaticCellValues, qp_point::Int, basefun::Int) = shape_gradient(cv.cv_each, qp_point, basefun)
shape_value(cv::StaticCellValues, qp_point::Int, basefun::Int) = shape_value(cv.cv_all, qp_point, basefun)

getnbasefunctions(cv::StaticCellValues) = getnbasefunctions(cv.cv_all)
getnquadpoints(cv::StaticCellValues) = getnquadpoints(cv.cv_all)


function reinit(cv::CV, x::AbstractVector{<:Vec}) where {CV<:StaticCellValues}
    return CV(cv.cv_all, CellValuesEach(cv.cv_all, x))
end

    

#= Methods to be implemented
[`getnquadpoints`](@ref)
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
=#
