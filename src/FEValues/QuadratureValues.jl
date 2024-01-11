struct QuadratureValuesIterator{VT<:AbstractValues}
    v::VT
end

function Base.iterate(iterator::QuadratureValuesIterator, q_point=1)
    checkbounds(Bool, 1:getnquadpoints(iterator.v), q_point) || return nothing
    qp_v = @inbounds quadrature_point_values(iterator.v, q_point)
    return (qp_v, q_point+1)
end
Base.IteratorEltype(::Type{<:QuadratureValuesIterator}) = Base.EltypeUnknown()
Base.length(iterator::QuadratureValuesIterator) = getnquadpoints(iterator.v)


struct QuadratureValues{VT<:AbstractValues}
    v::VT
    q_point::Int
    Base.@propagate_inbounds function QuadratureValues(v::AbstractValues, q_point::Int)
        @boundscheck checkbounds(1:getnbasefunctions(v), q_point)
        return new{typeof(v)}(v, q_point)
    end
end

@inline quadrature_point_values(fe_v::AbstractValues, q_point) = QuadratureValues(fe_v, q_point)

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

# function_<something> overloads without q_point input
@inline function_value(qv::QuadratureValues, args...) = function_value(qv.v, qv.q_point, args...)
@inline function_gradient(qv::QuadratureValues, args...) = function_gradient(qv.v, qv.q_point, args...)
@inline function_symmetric_gradient(qv::QuadratureValues, args...) = function_symmetric_gradient(qv.v, qv.q_point, args...)
@inline function_divergence(qv::QuadratureValues, args...) = function_divergence(qv.v, qv.q_point, args...)
@inline function_curl(qv::QuadratureValues, args...) = function_curl(qv.v, qv.q_point, args...)

# TODO: Interface things not included yet

@inline spatial_coordinate(qv::QuadratureValues, x) = spatial_coordinate(qv.v, qv.q_point, x)


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