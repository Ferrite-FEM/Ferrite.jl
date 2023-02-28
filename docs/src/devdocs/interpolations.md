# [Interpolations](@id devdocs-interpolations)

## Type definitions

Interpolations are subtypes of `Interpolation{dim, shape, order}`, i.e. they are
parametrized by the (spatial) dimension, reference shape and order.

### Fallback methods applicable for all subtypes of `Interpolation`

```@docs
Ferrite.getdim(::Interpolation)
Ferrite.getrefshape(::Interpolation)
Ferrite.getorder(::Interpolation)
Ferrite.value(::Interpolation{dim}, ::Vec{dim}) where {dim}
Ferrite.derivative(::Interpolation{dim}, ::Vec{dim}) where {dim}
```

### Required methods to implement for all subtypes of `Interpolation`

```@docs
Ferrite.getnbasefunctions(::Interpolation)
```
