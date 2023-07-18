```@meta
DocTestSetup = :(using Ferrite)
```
# Postprocessing

## Project to nodes
```@docs
L2Projector
project
```


# Postprocessing
```@docs
PointEvalHandler
evaluate_at_points
Ferrite.PointValues
PointIterator
PointLocation
```

```@docs
evaluate_at_grid_nodes
```

## VTK Export

```@docs
VTKStream
write_solution
write_projected
write_celldata
write_nodedata
write_cellset
write_nodeset
write_dirichlet
write_cell_colors
```
