```@meta
DocTestSetup = :(using Ferrite)
```

# Degrees of freedom
Degrees of freedom (dofs) are distributed by the [`DofHandler`](@ref).
```@docs
DofHandler
SubDofHandler
```

## Adding fields to the DofHandlers
```@docs
add!(::DofHandler, ::Symbol, ::Interpolation)
add!(::SubDofHandler, ::Symbol, ::Interpolation)
close!(::DofHandler)
```

## Algebraic variables and coupling descriptors

See the topic section on [Algebraic variables](@ref topic-algebraic-variables).

```@docs
AlgebraicVariable
add!(::DofHandler, ::Symbol, ::AlgebraicVariable)
getalgebraicvariablenames
algebraic_variable
algebraic_dofs
active_components
AlgebraicValues
algebraic_value
algebraic_basis_value
CellCoupling
FacetCoupling
AlgebraicCoupling
LocalDofLayout
local_dofs
local_dofs!
dof_range(::LocalDofLayout, ::Symbol)
```

## DoF renumbering
```@docs
renumber!
DofOrder.FieldWise
DofOrder.ComponentWise
```

## Common methods
```@docs
ndofs
ndofs_per_cell
dof_range
celldofs
celldofs!
```

## Grid iterators
```@docs
CellCache
CellIterator
FacetCache
FacetIterator
InterfaceCache
InterfaceIterator
```
