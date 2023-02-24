```@meta
DocTestSetup = :(using Ferrite)
```

# Boundary Conditions

```@index
Pages = ["initial_and_boundary_conditions.md"]
```

```@docs
ConstraintHandler
Dirichlet
PeriodicDirichlet
collect_periodic_faces
collect_periodic_faces!
add!
close!
update!
apply!
apply_zero!
apply_local!
apply_assemble!
get_rhs_data
apply_rhs!
Ferrite.RHSData
```

# Initial conditions

```@docs
apply_analytical!
```
