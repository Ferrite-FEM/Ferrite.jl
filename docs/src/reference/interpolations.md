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

Implemented interpolations:

```@docs
Lagrange
Serendipity
DiscontinuousLagrange
BubbleEnrichedLagrange
CrouzeixRaviart
RannacherTurek
```

## [Dof functionals](@id dof-functionals)

Each local dof of an interpolation has a *dof functional* describing how it evaluates a
function. Most interpolations have only [`PointValue`](@ref) dofs, but e.g. H(div)/H(curl)
interpolations have integral moment dofs, and Hermite-type interpolations have derivative
dofs. The functional can be used to select which dofs a [`Dirichlet`](@ref) condition
constrains.

```@docs
Ferrite.DofFunctional
PointValue
PointDerivative
IntegralMoment
NormalMoment
TangentialMoment
InteriorMoment
Ferrite.dof_functionals
```
