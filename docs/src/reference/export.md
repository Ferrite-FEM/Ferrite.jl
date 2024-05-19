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
VTKFile
VTKFileCollection
addstep!
write_solution
write_projection
write_cell_data
write_node_data
Ferrite.write_cellset
Ferrite.write_nodeset
Ferrite.write_constraints
Ferrite.write_cell_colors
```
