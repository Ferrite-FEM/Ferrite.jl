```@meta
DocTestSetup = :(using JuAFEM)
```

# Grid

```@docs
Node
Cell
Grid
```

## Utility Functions

```@docs
getcells
getncells
getnodes
getnnodes
nnodes_per_cell
getcellset
getnodeset
getfaceset
compute_vertex_values
transform!
getcoordinates
getcoordinates!
```

## Grid Sets Utility

```@docs
addcellset!
addfaceset!
addnodeset!
```
