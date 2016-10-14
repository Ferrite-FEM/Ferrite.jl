# Issue #66, PR #73
@deprecate FEValues FECellValues

# Issue #74, PR #76
immutable Dim{T} end
export Dim

@deprecate QuadratureRule{dim}(::Type{Dim{dim}}, shape::AbstractRefShape, order::Int) QuadratureRule{dim, typeof(shape)}(order)
@deprecate QuadratureRule{dim}(quad_type::Symbol, ::Type{Dim{dim}}, shape::AbstractRefShape, order::Int) QuadratureRule{dim, typeof(shape)}(quad_type, order)

# Issue #78, PR #80
@deprecate function_scalar_value function_value
@deprecate function_vector_value function_value
@deprecate function_scalar_gradient function_gradient
@deprecate function_vector_gradient function_gradient
@deprecate function_vector_symmetric_gradient function_symmetric_gradient
@deprecate function_vector_divergence function_divergence
