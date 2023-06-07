```@meta
DocTestSetup = :(using Ferrite)
```

# Grid & AbstractGrid

## Grid

```@docs
generate_grid
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
transform!
getcoordinates
getcoordinates!
```

### Topology

```@docs
Ferrite.ExclusiveTopology
Ferrite.getneighborhood
Ferrite.faceskeleton
```

### Grid Sets Utility

```@docs
addcellset!
addfaceset!
addboundaryfaceset!
addboundaryedgeset!
addboundaryvertexset!
addnodeset!
```

### Multithreaded Assembly
```@docs
create_coloring
```
