```@meta
DocTestSetup = :(using Ferrite)
```

# Assembly

```@docs
start_assemble
assemble!
finish_assemble
```

## Interface assembly

Local matrices/vectors for interface terms are computed in the *stacked* dof layout of
[`interfacedofs`](@ref), in which dofs shared between the two cells (e.g. for fields with
continuous interpolations) appear once per side. Before assembly they must be condensed
onto the unique dofs:

```@docs
InterfaceAssemblyBuffer
condense_interface!
```
