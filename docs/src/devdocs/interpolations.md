# [Interpolations](@id devdocs-interpolations)

## Type definitions

Interpolations are subtypes of `Interpolation{dim, shape, order}`, i.e. they are
parametrized by the (reference element) dimension, reference shape and order.

### Fallback methods applicable for all subtypes of `Interpolation`

```@docs
Ferrite.getdim(::Interpolation)
Ferrite.getrefshape(::Interpolation)
Ferrite.getorder(::Interpolation)
Ferrite.value(::Interpolation{dim}, ::Vec{dim,T}) where {dim,T}
Ferrite.derivative(::Interpolation{dim}, ::Vec{dim}) where {dim}
Ferrite.boundarydof_indices
```

### Required methods to implement for all subtypes of `Interpolation` to define a new finite element

Depending on the dimension of the reference element the following functions have to be implemented

```@docs
Ferrite.value(::Interpolation, ::Int, ::Vec)
Ferrite.vertexdof_indices(::Interpolation)
Ferrite.facedof_indices(::Interpolation)
Ferrite.facedof_interior_indices(::Interpolation)
Ferrite.edgedof_indices(::Interpolation)
Ferrite.edgedof_interior_indices(::Interpolation)
Ferrite.celldof_interior_indices(::Interpolation)
Ferrite.getnbasefunctions(::Interpolation)
Ferrite.reference_coordinates(::Interpolation)
Ferrite.IsDiscontinuous(::Interpolation)
Ferrite.adjust_dofs_during_distribution(::Interpolation)
```

for all entities which exist on that reference element. The dof functions default to having no
dofs defined on a specific entity. Hence, not overloading of the dof functions will result in an 
element with zero dofs. Also, it should always be double checked that everything is consistent as 
specified in the docstring of the corresponding function, as inconsistent implementations can
lead to bugs which are really difficult to track down.
