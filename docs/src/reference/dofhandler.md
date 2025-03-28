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

## Dof renumbering
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

# Grid iterators
```@docs
CellCache
CellIterator
FacetCache
FacetIterator
InterfaceCache
InterfaceIterator
```
