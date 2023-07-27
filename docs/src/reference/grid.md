```@meta
DocTestSetup = :(using Ferrite)
```

# Grid & AbstractGrid

## Grid

```@docs
generate_grid
Node
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
getnodeset
getfaceset
getedgeset
getvertexset
transform_coordinates!
get_cell_coordinates
get_cell_coordinates!
```

### Topology

```@docs
Ferrite.ExclusiveTopology
Ferrite.getneighborhood
Ferrite.faceskeleton
Ferrite.vertex_star_stencils
Ferrite.getstencil
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
