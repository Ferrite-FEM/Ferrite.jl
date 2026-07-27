```@meta
CurrentModule = Ferrite
DocTestSetup = :(using Ferrite)
```

# [Interpolation](@id reference-interpolation)

```@docs
Interpolation
getnbasefunctions(::Interpolation)
getrefdim(::Interpolation)
getrefshape
getorder
```

Implemented interpolations:

```@docs
Lagrange
Serendipity
DiscontinuousLagrange
BubbleEnrichedLagrange
CrouzeixRaviart
RannacherTurek
```

Scalar interpolations can be used for vector-valued fields by vectorizing them with
`ip ^ dim`, and for second order tensor-valued fields by wrapping them in a
[`TensorizedInterpolation`](@ref):

```@docs
TensorizedInterpolation
```
