```@meta
DocTestSetup = :(using Ferrite)
```

# Degrees of Freedom
Degrees of freedom (dofs) are distributed by the [`DofHandler`](@ref) or the [`MixedDofHandler`](@ref).
```@docs
DofHandler
MixedDofHandler
```

## Adding fields to the DofHandlers
```@docs
add!(::DofHandler, ::Symbol, ::Int, ::Interpolation)
add!(::MixedDofHandler, ::FieldHandler)
Field
FieldHandler
close!(::MixedDofHandler)
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

# CellIterator
```@docs
CellCache
CellIterator
```
