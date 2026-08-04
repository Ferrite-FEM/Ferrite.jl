```@meta
DocTestSetup = :(using Ferrite)
```

# Grid

## Grid

```@docs
generate_grid
Node
CellIndex
VertexIndex
EdgeIndex
FaceIndex
FacetIndex
Grid
```

### Utility functions

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
geometric_interpolation(::Ferrite.AbstractCell)
get_node_coordinate
Ferrite.getspatialdim(::Ferrite.AbstractGrid)
Ferrite.getrefdim(::Ferrite.AbstractCell)
```

### Topology

```@docs
ExclusiveTopology
getneighborhood
facetskeleton
vertex_star_stencils
getstencil
```

### Grid sets utility

```@docs
addcellset!
addfacetset!
addboundaryfacetset!
addvertexset!
addboundaryvertexset!
addnodeset!
```

### Multithreaded assembly
```@docs
create_coloring
```
