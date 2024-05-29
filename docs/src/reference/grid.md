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
getfacetset
getvertexset
transform_coordinates!
getcoordinates
getcoordinates!
Ferrite.get_node_coordinate
Ferrite.getspatialdim(::Ferrite.AbstractGrid)
Ferrite.getrefdim
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
addfacetset!
addboundaryfacetset!
addvertexset!
addboundaryvertexset!
addnodeset!
```

### Multithreaded Assembly
```@docs
create_coloring
```
