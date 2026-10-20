# [Matrix-free operator evaluation](@id devdocs-matrix-free)

Experimental infrastructure for evaluating operators matrix-free with sum factorization,
see the [matrix-free how-to](../howto/matrix_free.md) for background, usage, and
references. The API is limited to (vectorized) Lagrange interpolations on hexahedra with
tensor product quadrature rules, and is expected to change between releases.

## Tensor product structure of interpolations

The evaluator builds on the tensor product structure of the interpolations, exposed by
[`Ferrite.tensor_product_interpolation`](@ref) and
[`Ferrite.tensor_product_indices`](@ref) (documented with the
[interpolation devdocs](interpolations.md)).

## The evaluator

```@docs
Ferrite.TensorProductEvaluator
Ferrite.read_dof_values!
Ferrite.evaluate_gradients!
Ferrite.get_gradient
Ferrite.submit_gradient!
Ferrite.integrate_gradients!
Ferrite.distribute_local_to_global!
```

## Setup helpers

```@docs
Ferrite.lexicographic_numbering
Ferrite.lexicographic_dofmap
Ferrite.ConstrainedDofMap
Ferrite.quadrature_point_data
```
