# [Interpolations](@id devdocs-interpolations)

## Type definitions

Interpolations are subtypes of `Interpolation{dim, shape, order}`, i.e. they are
parametrized by the (reference element) dimension, reference shape and order.

### Fallback methods applicable for all subtypes of `Interpolation`

```@docs
Ferrite.getdim(::Interpolation)
Ferrite.getrefshape(::Interpolation)
Ferrite.getorder(::Interpolation)
Ferrite.value(::Interpolation{dim}, ::Vec{dim,T}) where {dim,T}
Ferrite.derivative(::Interpolation{dim}, ::Vec{dim}) where {dim}
```

### Required methods to implement for all subtypes of `Interpolation` to define a new finite element

```@docs
Ferrite.value(::Interpolation, ::Int, ::Vec)
Ferrite.vertices(::Interpolation)
Ferrite.nvertexdofs(::Interpolation)
Ferrite.faces(::Interpolation)
Ferrite.nfacedofs(::Interpolation)
Ferrite.edges(::Interpolation)
Ferrite.nedgedofs(::Interpolation)
Ferrite.ncelldofs(::Interpolation)
Ferrite.getnbasefunctions(::Interpolation)
Ferrite.reference_coordinates(::Interpolation)
```
