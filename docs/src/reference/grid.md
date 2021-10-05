```@meta
DocTestSetup = :(using Ferrite)
```

# Grid

```@docs
Node
Cell
CellIndex
VertexIndex
EdgeIndex
FaceIndex
Grid
```

## Utility Functions

```@docs
getcells
getncells
getnodes
getnnodes
Ferrite.nnodes_per_cell
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
