# QuadratureValuesIterator
struct QuadratureValuesIterator{VT,XT}
    v::VT
    cell_coords::XT # Union{AbstractArray{<:Vec}, Nothing}
    function QuadratureValuesIterator(v::V) where V
        return new{V, Nothing}(v, nothing)
    end
    function QuadratureValuesIterator(v::V, cell_coords::VT) where {V, VT <: AbstractArray}
        #reinit!(v, cell_coords) # Why we need that ?
        return new{V, VT}(v, cell_coords)
    end
end

function Base.iterate(iterator::QuadratureValuesIterator{<:Any, Nothing}, q_point=1)
    checkbounds(Bool, 1:getnquadpoints(iterator.v), q_point) || return nothing
    qp_v = @inbounds quadrature_point_values(iterator.v, q_point) 
    return (qp_v, q_point+1)
end

function Base.iterate(iterator::QuadratureValuesIterator{<:Any, <:StaticVector}, q_point=1)
    checkbounds(Bool, 1:getnquadpoints(iterator.v), q_point) || return nothing
    #q_point < 5 || return nothing
    qp_v = @inbounds quadrature_point_values(iterator.v, q_point, iterator.cell_coords)
    return (qp_v, q_point+1)
    #return (1, q_point+1)
end
Base.IteratorEltype(::Type{<:QuadratureValuesIterator}) = Base.EltypeUnknown()
Base.length(iterator::QuadratureValuesIterator) = getnquadpoints(iterator.v)

# AbstractQuadratureValues
abstract type AbstractQuadratureValues end

function function_value(qp_v::AbstractQuadratureValues, u::AbstractVector, dof_range = eachindex(u))
    n_base_funcs = getnbasefunctions(qp_v)
    length(dof_range) == n_base_funcs || throw_incompatible_dof_length(length(dof_range), n_base_funcs)
    @boundscheck checkbounds(u, dof_range)
    val = function_value_init(qp_v, u)
    @inbounds for (i, j) in pairs(dof_range)
        val += shape_value(qp_v, i) * u[j]
    end
    return val
end

function function_gradient(qp_v::AbstractQuadratureValues, u::AbstractVector, dof_range = eachindex(u))
    n_base_funcs = getnbasefunctions(qp_v)
    length(dof_range) == n_base_funcs || throw_incompatible_dof_length(length(dof_range), n_base_funcs)
    @boundscheck checkbounds(u, dof_range)
    grad = function_gradient_init(qp_v, u)
    @inbounds for (i, j) in pairs(dof_range)
        grad += shape_gradient(qp_v, i) * u[j]
    end
    return grad
end

function function_symmetric_gradient(qp_v::AbstractQuadratureValues, u::AbstractVector, dof_range)
    grad = function_gradient(qp_v, u, dof_range)
    return symmetric(grad)
end

function function_symmetric_gradient(qp_v::AbstractQuadratureValues, u::AbstractVector)
    grad = function_gradient(qp_v, u)
    return symmetric(grad)
end

function function_divergence(qp_v::AbstractQuadratureValues, u::AbstractVector, dof_range = eachindex(u))
    return divergence_from_gradient(function_gradient(qp_v, u, dof_range))
end

function function_curl(qp_v::AbstractQuadratureValues, u::AbstractVector, dof_range = eachindex(u))
    return curl_from_gradient(function_gradient(qp_v, u, dof_range))
end

function spatial_coordinate(qp_v::AbstractQuadratureValues, x::AbstractVector{<:Vec})
    n_base_funcs = getngeobasefunctions(qp_v)
    length(x) == n_base_funcs || throw_incompatible_coord_length(length(x), n_base_funcs)
    vec = zero(eltype(x))
    @inbounds for i in 1:n_base_funcs
        vec += geometric_value(qp_v, i) * x[i]
    end
    return vec
end

# Specific design for QuadratureValues <: AbstractQuadratureValues
# which contains standard AbstractValues
struct QuadratureValues{VT<:AbstractValues} <: AbstractQuadratureValues
    v::VT
    q_point::Int
    Base.@propagate_inbounds function QuadratureValues(v::AbstractValues, q_point::Int)
        @boundscheck checkbounds(1:getnbasefunctions(v), q_point)
        return new{typeof(v)}(v, q_point)
    end
end

@inline quadrature_point_values(fe_v::AbstractValues, q_point, args...) = QuadratureValues(fe_v, q_point)

@propagate_inbounds getngeobasefunctions(qv::QuadratureValues) = getngeobasefunctions(qv.v)
@propagate_inbounds geometric_value(qv::QuadratureValues, i) = geometric_value(qv.v, qv.q_point, i)
geometric_interpolation(qv::QuadratureValues) = geometric_interpolation(qv.v)

getdetJdV(qv::QuadratureValues) = @inbounds getdetJdV(qv.v, qv.q_point)

# Accessors for function values 
getnbasefunctions(qv::QuadratureValues) = getnbasefunctions(qv.v)
function_interpolation(qv::QuadratureValues) = function_interpolation(qv.v)
function_difforder(qv::QuadratureValues) = function_difforder(qv.v)
shape_value_type(qv::QuadratureValues) = shape_value_type(qv.v)
shape_gradient_type(qv::QuadratureValues) = shape_gradient_type(qv.v)

@propagate_inbounds shape_value(qv::QuadratureValues, i::Int) = shape_value(qv.v, qv.q_point, i)
@propagate_inbounds shape_gradient(qv::QuadratureValues, i::Int) = shape_gradient(qv.v, qv.q_point, i)
@propagate_inbounds shape_symmetric_gradient(qv::QuadratureValues, i::Int) = shape_symmetric_gradient(qv.v, qv.q_point, i)



#= Proposed syntax, for heatflow in general 
function assemble_element!(Ke::Matrix, fe::Vector, cellvalues)
    n_basefuncs = getnbasefunctions(cellvalues)
    for qv in Ferrite.QuadratureValuesIterator(cellvalues)
        dΩ = getdetJdV(qv)
        for i in 1:n_basefuncs
            δu  = shape_value(qv, i)
            ∇δu = shape_gradient(qv, i)
            fe[i] += δu * dΩ
            for j in 1:n_basefuncs
                ∇u = shape_gradient(qv, j)
                Ke[i, j] += (∇δu ⋅ ∇u) * dΩ
            end
        end
    end
    return Ke, fe
end

Where the default for a QuadratureValuesIterator would be to return a 
`QuadratureValues` as above, but custom `AbstractValues` can be created where 
for example the element type would be a static QuadPointValue type which doesn't 
use heap allocated buffers, e.g. by only saving the cell and coordinates during reinit, 
and then calculating all values for each element in the iterator. 

References: 
https://github.com/termi-official/Thunderbolt.jl/pull/53/files#diff-2b486be5a947c02ef2a38ff3f82af3141193af0b6f01ed9d5129b914ed1d84f6
https://github.com/Ferrite-FEM/Ferrite.jl/compare/master...kam/StaticValues2
=#