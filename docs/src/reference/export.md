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
vtk_grid(filename::AbstractString, grid::Grid{dim,C,T}; compress::Bool) where {dim,C,T} 
vtk_point_data
vtk_cellset
vtk_cell_data_colors
```
