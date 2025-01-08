```@meta
DocTestSetup = :(using Ferrite)
```

# Boundary conditions

```@index
Pages = ["boundary_conditions.md"]
```

```@docs
ConstraintHandler
Dirichlet
IntegrateableDirichlet
PeriodicDirichlet
collect_periodic_facets
collect_periodic_facets!
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
