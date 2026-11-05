```@meta
CurrentModule = Ferrite
DocTestSetup = :(using Ferrite)
```

# [Interpolations](@id reference-interpolation)

```@docs
Interpolation
getnbasefunctions(::Interpolation)
getrefdim(::Interpolation)
getrefshape
getorder
```

## Scalar interpolations

```@docs
Lagrange
DiscontinuousLagrange
Serendipity
BubbleEnrichedLagrange
CrouzeixRaviart
RannacherTurek
```

## Vector interpolations

```@docs
VectorizedInterpolation
RaviartThomas
BrezziDouglasMarini
Nedelec
```
