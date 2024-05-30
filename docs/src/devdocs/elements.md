# [Elements and cells](@id devdocs-elements)

## Type definitions

Elements or cells are subtypes of `AbstractCell{dim,N,M}`. They are parametrized by
the dimension of their nodes via `dim`, the number of nodes `N` and the number
of faces `M`.

### Required methods to implement for all subtypes of `AbstractCell` to define a new element

```@docs
Ferrite.vertices(::Ferrite.AbstractCell)
Ferrite.edges(::Ferrite.AbstractCell)
Ferrite.reference_faces(::Ferrite.AbstractRefShape)
Ferrite.faces(::Ferrite.AbstractCell)
Ferrite.geometric_interpolation(::Ferrite.AbstractCell)
```

### Common utilities and definitions when working with grids internally.

```@docs
Ferrite.BoundaryIndex
Ferrite.boundaryfunction(::Type{<:Ferrite.BoundaryIndex})
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
