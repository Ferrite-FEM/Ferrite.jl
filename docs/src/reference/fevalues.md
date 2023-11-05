```@meta
CurrentModule = Ferrite
DocTestSetup = :(using Ferrite)
```

# FEValues

## Main types
[`CellValues`](@ref) and [`FaceValues`](@ref) are the most common 
subtypes of `Ferrite.AbstractValues`. For more details about how 
these work, please see the related [topic guide](@ref fevalues_topicguide).

```@docs
CellValues
FaceValues
```

## Applicable functions
The following functions are applicable to both `CellValues`
and `FaceValues`.

```@docs
reinit!
getnquadpoints
getdetJdV

shape_value(::Ferrite.AbstractValues, ::Int, ::Int)
shape_gradient(::Ferrite.AbstractValues, ::Int, ::Int)
shape_symmetric_gradient
shape_divergence

function_value
function_gradient
function_symmetric_gradient
function_divergence
spatial_coordinate
```

In addition, there are some methods that are unique for `FaceValues`.

```@docs
Ferrite.getcurrentface
getnormal
```
