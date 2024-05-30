struct StaticQuadratureValues{T, N_t, dNdx_t, M_t, NumN, NumM} <: AbstractQuadratureValues
    detJdV::T
    N::SVector{NumN, N_t}
    dNdx::SVector{NumN, dNdx_t}
    M::SVector{NumM, M_t}
end

@propagate_inbounds getngeobasefunctions(qv::StaticQuadratureValues) = length(qv.M)
@propagate_inbounds geometric_value(qv::StaticQuadratureValues, i) = qv.M[i]
# geometric_interpolation(qv::StaticQuadratureValues) = geometric_interpolation(qv.v) # Not included

getdetJdV(qv::StaticQuadratureValues) = qv.detJdV

# Accessors for function values 
getnbasefunctions(qv::StaticQuadratureValues) = length(qv.N)
# function_interpolation(qv::StaticQuadratureValues) = function_interpolation(qv.v) # Not included
shape_value_type(::StaticQuadratureValues{<:Any, N_t}) where N_t = N_t
shape_gradient_type(::StaticQuadratureValues{<:Any, <:Any, dNdx_t}) where dNdx_t = dNdx_t

@propagate_inbounds shape_value(qv::StaticQuadratureValues, i::Int) = qv.N[i]
@propagate_inbounds shape_gradient(qv::StaticQuadratureValues, i::Int) = qv.dNdx[i]
@propagate_inbounds shape_gradient(qv::StaticQuadratureValues, i::Int32) = qv.dNdx[i] # Needed for GPU (threadIdx is Int32), otherwise it will throw a dynamic function invokation error
@propagate_inbounds shape_symmetric_gradient(qv::StaticQuadratureValues, i::Int) = symmetric(qv.dNdx[i])

@propagate_inbounds geometric_value(qv::StaticQuadratureValues, i::Int) = qv.M[i]

# StaticInterpolationValues: interpolation and precalculated values for all quadrature points
# Can be both for function and geometric shape functions. 
# DiffOrder parameter?
# TODO: Could perhaps denote this just InterpolationValues and replace GeometryMapping
# Just need to make Nξ::AbstractMatrix instead as in GeometryMapping to make it equivalent (except fieldnames)
struct StaticInterpolationValues{IP, N, Nqp, N_et, dNdξ_t, Nall}
    ip::IP
    Nξ::SMatrix{N, Nqp, N_et, Nall}
    dNdξ::dNdξ_t        # Union{SMatrix{N, Nqp}, Nothing}
    #dN2dξ2::dN2dξ2_t   # Union{SMatrix{N, Nqp}, Nothing}
end
function StaticInterpolationValues(fv::FunctionValues)
    N = getnbasefunctions(fv.ip)
    Nq = size(fv.Nξ, 2)
    Nξ = SMatrix{N, Nq}(fv.Nξ)
    dNdξ = SMatrix{N, Nq}(fv.dNdξ)
    return StaticInterpolationValues(fv.ip, Nξ, dNdξ)
end
function StaticInterpolationValues(gm::GeometryMapping)
    N = getnbasefunctions(gm.ip)
    Nq = size(gm.M, 2)
    M = SMatrix{N, Nq}(gm.M)
    dMdξ = SMatrix{N, Nq}(gm.dMdξ)
    return StaticInterpolationValues(gm.ip, M, dMdξ)
end

getnbasefunctions(siv::StaticInterpolationValues) = getnbasefunctions(siv.ip)

# Dispatch on DiffOrder parameter? 
# Reuse functions for GeometryMapping - same signature but need access functions
# Or merge GeometryMapping and StaticInterpolationValues => InterpolationValues
@propagate_inbounds @inline function calculate_mapping(ip_values::StaticInterpolationValues{<:Any, N}, q_point, x) where N
    fecv_J = zero(otimes_returntype(eltype(x), eltype(ip_values.dNdξ)))
    @inbounds for j in 1:N
        #fecv_J += x[j] ⊗ geo_mapping.dMdξ[j, q_point]
        fecv_J += otimes_helper(x[j], ip_values.dNdξ[j, q_point])
    end
    return MappingValues(fecv_J, nothing)
end

@propagate_inbounds @inline function calculate_mapped_values(funvals::StaticInterpolationValues, q_point, mapping_values, args...)
    return calculate_mapped_values(funvals, mapping_type(funvals.ip), q_point, mapping_values, args...)
end

@propagate_inbounds @inline function calculate_mapped_values(funvals::StaticInterpolationValues, ::IdentityMapping, q_point, mapping_values, args...)
    Jinv = calculate_Jinv(getjacobian(mapping_values))
    Nx = funvals.Nξ[:, q_point]
    dNdx = map(dNdξ -> dothelper(dNdξ, Jinv), funvals.dNdξ[:, q_point])
    return Nx, dNdx
end

struct StaticCellValues{FV, GM, Nqp, T}
    fv::FV # StaticInterpolationValues
    gm::GM # StaticInterpolationValues
    #x::Tx  # AbstractVector{<:Vec} or Nothing
    weights::NTuple{Nqp, T}
end
function StaticCellValues(cv::CellValues) 
    fv = StaticInterpolationValues(cv.fun_values)
    gm = StaticInterpolationValues(cv.geo_mapping)
    sdim = sdim_from_gradtype(shape_gradient_type(cv))
    #x = SaveCoords ? fill(zero(Vec{sdim}), getngeobasefunctions(cv)) : nothing
    weights = ntuple(i -> getweights(cv.qr)[i], getnquadpoints(cv))
    return StaticCellValues(fv, gm, weights)
end

getnquadpoints(cv::StaticCellValues) = length(cv.weights)
getnbasefunctions(cv::StaticCellValues) = getnbasefunctions(cv.fv)
getngeobasefunctions(cv::StaticCellValues) = getnbasefunctions(cv.gm)

@inline function reinit!(cv::StaticCellValues{<:Any, <:Any, <:AbstractVector}, cell_coords::AbstractVector)
    copyto!(cv.x, cell_coords)
    #TODO: Also allow the cell::AbstracCell to be given and updated
end
@inline function reinit!(::StaticCellValues{<:Any, <:Any, Nothing}, ::AbstractVector)
    nothing # Nothing to do on reinit if x is not saved.
end

@inline function quadrature_point_values(fe_v::StaticCellValues{<:Any, <:Any, <:AbstractVector}, q_point::Int)
    return _quadrature_point_values(fe_v, q_point, fe_v.x,detJ->throw_detJ_not_pos(detJ))
end

@inline function quadrature_point_values(fe_v::StaticCellValues{<:Any, <:Any}, q_point::Int, cell_coords::AbstractVector)
    return _quadrature_point_values(fe_v, q_point, cell_coords,detJ->throw_detJ_not_pos(detJ))
end

@inline function quadrature_point_values(fe_v::StaticCellValues{<:Any, <:Any}, q_point::Int, cell_coords::SVector)
    return _quadrature_point_values(fe_v, q_point, cell_coords,detJ->-1)
end



function _quadrature_point_values(fe_v::StaticCellValues, q_point::Int, cell_coords::AbstractVector,neg_detJ_err_fun::Function)
    #q_point bounds checked, ok to use @inbounds
    @inbounds begin
         mapping = calculate_mapping(fe_v.gm, q_point, cell_coords)

         detJ = calculate_detJ(getjacobian(mapping))
         detJ > 0.0 || neg_detJ_err_fun(detJ) # Cannot throw error on GPU, TODO: return error code instead
         detJdV = detJ * fe_v.weights[q_point]

         Nx, dNdx = calculate_mapped_values(fe_v.fv, q_point, mapping)
         M = fe_v.gm.Nξ[:, q_point]
    end
     return StaticQuadratureValues(detJdV, Nx, dNdx, M)
end


