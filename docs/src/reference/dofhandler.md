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

## Common methods
```@docs
ndofs
dof_range
Ferrite.nfields(::MixedDofHandler)
Ferrite.getfieldnames(::MixedDofHandler)
Ferrite.getfielddim(::MixedDofHandler, ::Symbol)
```

# CellIterator
```@docs
CellIterator
```
