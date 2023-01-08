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
push!(::DofHandler, ::Symbol, ::Int, ::Interpolation)
push!(::MixedDofHandler, ::FieldHandler)
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

## `AbstractDofHandler` interface
The following are *internal* methods that need to be implemented to fulfill the `AbstractDofHandler`
interface.
```@docs
Ferrite.find_field
Ferrite.field_offset
Ferrite.getfieldinterpolation
Ferrite.getfielddim
```

# CellIterator
```@docs
CellCache
CellIterator
```
