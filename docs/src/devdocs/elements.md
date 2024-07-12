# [Elements and cells](@id devdocs-elements)

## Type definitions

Elements or cells are subtypes of `AbstractCell{<:AbstractRefShape}`. As shown, they are parametrized
by the associated reference element.

### Required methods to implement for all subtypes of `AbstractCell` to define a new element

```@docs
Ferrite.get_node_ids
```

### Common utilities and definitions when working with grids internally.

First we have some topological queries on the element

```@docs
Ferrite.vertices(::Ferrite.AbstractCell)
Ferrite.edges(::Ferrite.AbstractCell)
Ferrite.faces(::Ferrite.AbstractCell)
Ferrite.facets(::Ferrite.AbstractCell)
Ferrite.boundaryfunction(::Type{<:Ferrite.BoundaryIndex})
Ferrite.reference_vertices(::Ferrite.AbstractCell)
Ferrite.reference_edges(::Ferrite.AbstractCell)
Ferrite.reference_faces(::Ferrite.AbstractCell)
```

and some generic utils which are commonly found in finite element codes

```@docs
Ferrite.BoundaryIndex
Ferrite.get_coordinate_eltype(::Ferrite.AbstractGrid)
Ferrite.get_coordinate_eltype(::Node)
Ferrite.toglobal
Ferrite.sortface
Ferrite.sortface_fast
Ferrite.sortedge
Ferrite.sortedge_fast
Ferrite.element_to_facet_transformation
Ferrite.facet_to_element_transformation
Ferrite.InterfaceOrientationInfo
Ferrite.transform_interface_points!
Ferrite.get_transformation_matrix
```
