```@meta
DocTestSetup = :(using Ferrite)
```

# Grid & AbstractGrid

## Grid

```@docs
Node
Cell
CellIndex
VertexIndex
EdgeIndex
FaceIndex
Grid
```

### Utility Functions

```@docs
getcells
getncells
getnodes
getnnodes
Ferrite.nnodes_per_cell
getcellset
getcellsets
getnodeset
getnodesets
getfaceset
getfacesets
getedgeset
getedgesets
getvertexset
getvertexsets
compute_vertex_values
transform!
getcoordinates
getcoordinates!
Ferrite.ExclusiveTopology
Ferrite.getneighborhood
Ferrite.faceskeleton
```

### Grid Sets Utility

```@docs
addcellset!
addfaceset!
addnodeset!
```

### Multithreaded Assembly
```@docs
create_coloring
```
