```@meta
DocTestSetup = :(using Ferrite)
```

# Adaptive mesh refinement

!!! warning "Experimental feature"
    Adaptive mesh refinement is a new and experimental feature. The API documented here may
    change, gain capabilities, or be restructured in minor releases without following semantic
    versioning. Feedback on it is very welcome.

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

## The grid facade

A `ForestBWG` answers all `AbstractGrid` queries for its current refinement state through an
epoch-guarded materialization cache; a `DofHandler` built on it detects staleness after
refinement and is re-distributed with [`reclose!`](@ref).

```@docs
Ferrite.grid_epoch
Ferrite.has_hanging_nodes
Ferrite.AMR.conformity_info
Ferrite.AMR.MaterializedForest
Ferrite.AMR.ForestSnapshot
Ferrite.AMR.getleaves
```

## Materialization

```@docs
creategrid
```

## Constraints

```@docs
ConformityConstraint
```
