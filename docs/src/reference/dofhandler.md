```@meta
DocTestSetup = :(using Ferrite)
```

# Degrees of Freedom
Degrees of freedom (dofs) are distributed by the [`DofHandler`](@ref).
```@docs
DofHandler
```

## Adding fields to the DofHandlers
```@docs
add!(::DofHandler, ::Symbol, ::Interpolation)
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
FaceCache
FaceIterator
InterfaceCache
InterfaceIterator
```