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
In addition, there are some methods that are unique for `FaceValues`:

```@docs
FaceValues
getcurrentface
getnquadpoints(::FaceValues)
```

## [InterfaceValues](@id reference-interfacevalues)

All of the methods for [`FaceValues`](@ref) apply for `InterfaceValues` as well.
In addition, there are some methods that are unique for `InterfaceValues`:

```@docs
InterfaceValues
shape_value_average
shape_value_jump
shape_gradient_average
shape_gradient_jump
function_value_average
function_value_jump
function_gradient_average
function_gradient_jump
transform_interface_points!
```
