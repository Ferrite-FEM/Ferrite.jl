# Reference cells

The reference cells are used to i) define grid cells, ii) define shape functions, and iii)
define quadrature rules. The numbering of vertices, edges, faces are visualized below. See also
[`FerriteViz.elementinfo`](https://ferrite-fem.github.io/FerriteViz.jl/dev/api/#FerriteViz.elementinfo).

### `AbstractRefShape` subtypes

```@docs
Ferrite.AbstractRefShape
Ferrite.RefLine
Ferrite.RefTriangle
Ferrite.RefQuadrilateral
Ferrite.RefTetrahedron
Ferrite.RefHexahedron
Ferrite.RefPrism
```

### Required methods to implement for all subtypes of `AbstractRefShape` to define a new reference shape

```@docs
Ferrite.reference_vertices(::Type{<:Ferrite.AbstractRefShape})
Ferrite.reference_edges(::Type{<:Ferrite.AbstractRefShape})
Ferrite.reference_faces(::Type{<:Ferrite.AbstractRefShape})
```

which automatically defines


```@docs
Ferrite.reference_facets(::Type{<:Ferrite.AbstractRefShape})
```

### Applicable methods to `AbstractRefShape`s
```@docs
getrefdim(::Type{<:AbstractRefShape})
```
