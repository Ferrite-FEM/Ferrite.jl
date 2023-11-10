# [Interpolations](@id devdocs-interpolations)

## Type definitions

Interpolations are subtypes of `Interpolation{shape, order}`, i.e. they are
parametrized by the reference element and its characteristic order.

### Fallback methods applicable for all subtypes of `Interpolation`

```@docs
Ferrite.getdim(::Interpolation)
Ferrite.getrefshape(::Interpolation)
Ferrite.getorder(::Interpolation)
Ferrite.shape_gradient(::Interpolation, ::Vec, ::Int)
Ferrite.shape_gradient_and_value
Ferrite.boundarydof_indices
Ferrite.dirichlet_boundarydof_indices
Ferrite.shape_values!
Ferrite.shape_gradients!
Ferrite.shape_gradients_and_values!
```

### Required methods to implement for all subtypes of `Interpolation` to define a new finite element

Depending on the dimension of the reference element the following functions have to be implemented

```@docs
Ferrite.shape_value(::Interpolation, ::Vec, ::Int)
Ferrite.vertexdof_indices(::Interpolation)
Ferrite.dirichlet_vertexdof_indices(::Interpolation)
Ferrite.facedof_indices(::Interpolation)
Ferrite.dirichlet_facedof_indices(::Interpolation)
Ferrite.facedof_interior_indices(::Interpolation)
Ferrite.edgedof_indices(::Interpolation)
Ferrite.dirichlet_edgedof_indices(::Interpolation)
Ferrite.edgedof_interior_indices(::Interpolation)
Ferrite.celldof_interior_indices(::Interpolation)
Ferrite.getnbasefunctions(::Interpolation)
Ferrite.reference_coordinates(::Interpolation)
Ferrite.is_discontinuous(::Interpolation)
Ferrite.adjust_dofs_during_distribution(::Interpolation)
```

for all entities which exist on that reference element. The dof functions default to having no
dofs defined on a specific entity. Hence, not overloading of the dof functions will result in an 
element with zero dofs. Also, it should always be double checked that everything is consistent as 
specified in the docstring of the corresponding function, as inconsistent implementations can
lead to bugs which are really difficult to track down.
