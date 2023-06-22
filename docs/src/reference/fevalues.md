```@meta
CurrentModule = Ferrite
DocTestSetup = :(using Ferrite)
```

# FEValues

## [CellValues](@id reference-cellvalues)

```@docs
CellValues
reinit!
getnquadpoints(::CellValues)
getdetJdV

shape_value
shape_gradient
shape_symmetric_gradient
shape_divergence

function_value
function_gradient
function_symmetric_gradient
function_divergence
spatial_coordinate
```

## [FaceValues](@id reference-facevalues)

All of the methods for [`CellValues`](@ref) apply for `FaceValues` as well.
In addition, there are some methods that are unique for `FaecValues`:

```@docs
FaceValues
getcurrentface
getnquadpoints(::FaceValues)
```
