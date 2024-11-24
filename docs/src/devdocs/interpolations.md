# [Interpolations](@id devdocs-interpolations)

## Type definitions

Interpolations are subtypes of `Interpolation{shape, order}`, i.e. they are
parametrized by the reference element and its characteristic order.

### Fallback methods applicable for all subtypes of `Interpolation`

```@docs
Ferrite.getrefshape(::Interpolation)
Ferrite.getorder(::Interpolation)
Ferrite.reference_shape_gradient(::Interpolation, ::Vec, ::Int)
Ferrite.reference_shape_gradient_and_value(::Interpolation, ::Vec, ::Int)
Ferrite.reference_shape_hessian_gradient_and_value(::Interpolation, ::Vec, ::Int)
Ferrite.boundarydof_indices
Ferrite.dirichlet_boundarydof_indices
Ferrite.reference_shape_values!
Ferrite.reference_shape_gradients!
Ferrite.reference_shape_gradients_and_values!
Ferrite.reference_shape_hessians_gradients_and_values!
Ferrite.shape_value_type(ip::Interpolation, ::Type{T}) where T<:Number
```

### Required methods to implement for all subtypes of `Interpolation` to define a new finite element

Depending on the dimension of the reference element the following functions have to be implemented

```@docs
Ferrite.reference_shape_value(::Interpolation, ::Vec, ::Int)
Ferrite.vertexdof_indices(::Interpolation)
Ferrite.dirichlet_vertexdof_indices(::Interpolation)
Ferrite.facedof_indices(::Interpolation)
Ferrite.dirichlet_facedof_indices(::Interpolation)
Ferrite.facedof_interior_indices(::Interpolation)
Ferrite.edgedof_indices(::Interpolation)
Ferrite.dirichlet_edgedof_indices(::Interpolation)
Ferrite.edgedof_interior_indices(::Interpolation)
Ferrite.volumedof_interior_indices(::Interpolation)
Ferrite.getnbasefunctions(::Interpolation)
Ferrite.reference_coordinates(::Interpolation)
Ferrite.is_discontinuous(::Interpolation)
Ferrite.adjust_dofs_during_distribution(::Interpolation)
Ferrite.mapping_type
```

for all entities which exist on that reference element. The dof functions default to having no
dofs defined on a specific entity. Hence, not overloading of the dof functions will result in an
element with zero dofs. Also, it should always be double checked that everything is consistent as
specified in the docstring of the corresponding function, as inconsistent implementations can
lead to bugs which are really difficult to track down.

## Vector interpolation properties
### Hdiv interpolations

On a facet, ``\Gamma``, with normal, ``\boldsymbol{n}``,
the set of ``H(\mathrm{div})`` interpolation functions,
``\boldsymbol{N}_i(\boldsymbol{\xi})``, should fullfill
```math
\begin{align*}
\sum_{i = 1}^N \int_\Gamma \boldsymbol{N}_i(\boldsymbol{\xi}) \cdot \boldsymbol{n} &= 1 \\
\sum_{i = 1}^N \int_\Gamma f_i(\boldsymbol{\xi}) \boldsymbol{N}_i(\boldsymbol{\xi}) \cdot \boldsymbol{n} &= 1 \\
\int_\Gamma f_i(\boldsymbol{\xi}) \boldsymbol{N}_j(\boldsymbol{\xi}) \cdot \boldsymbol{n} &= 0, \quad i \neq j
```
The moment-weighting functions ``f_i(\boldsymbol{\xi})`` depend on how many base functions there are per
facet and the reference shape of the facet (`RefLine`, `RefTriangle`, or `RefQuadrilateral`).

These integral quantities apply to arbitrarily sized cell facets, and hence the actual value of the base functions
will scale depending on the size (smaller facet ``\rightarrow`` higher values). Consequently, when applying BCs,
we need to consider the actual facet size to be able to prescribe the average flux. Therefore, we include the
`GeometryMapping` values in the `BCValues` object.
