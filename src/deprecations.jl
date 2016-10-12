# PR #73
@deprecate FEValues FECellValues

# Issue #74
immutable Dim{T} end
export Dim

@deprecate QuadratureRule{dim}(::Type{Dim{dim}}, shape::AbstractRefShape, order::Int) QuadratureRule{dim, typeof(shape)}(order)
@deprecate QuadratureRule{dim}(quad_type::Symbol, ::Type{Dim{dim}}, shape::AbstractRefShape, order::Int) QuadratureRule{dim, typeof(shape)}(quad_type, order)
