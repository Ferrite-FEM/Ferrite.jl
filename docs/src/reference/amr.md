```@meta
DocTestSetup = :(using Ferrite)
```

# Adaptive mesh refinement

The adaptive mesh refinement (AMR) functionality is built on a forest of octrees, following
the algorithms of `p4est`. For a conceptual introduction see the [AMR topic
guide](../topics/amr.md); for the internals see the [AMR developer
documentation](../devdocs/AMR.md).

## Forest

```@docs
ForestBWG
```

## Refinement and coarsening

```@docs
refine!
refine_all!
coarsen!
refine_and_coarsen!
balanceforest!
```

## Materialization

```@docs
creategrid
```

## Constraints

```@docs
ConformityConstraint
```
