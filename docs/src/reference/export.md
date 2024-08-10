```@meta
DocTestSetup = :(using Ferrite)
```
# Postprocessing

## Projection of quadrature point data
```@docs
L2Projector(::Ferrite.AbstractGrid)
add!(::L2Projector, ::Ferrite.AbstractVecOrSet{Int}, ::Interpolation; kwargs...)
close!(::L2Projector)
L2Projector(::Interpolation, ::Ferrite.AbstractGrid; kwargs...)
project
```

## Evaluation at points
```@docs
evaluate_at_grid_nodes
PointEvalHandler
evaluate_at_points
PointValues
PointIterator
PointLocation
```

## VTK Export
```@docs
VTKGridFile
write_solution
write_projection
write_cell_data
write_node_data
Ferrite.write_cellset
Ferrite.write_nodeset
Ferrite.write_constraints
Ferrite.write_cell_colors
```
