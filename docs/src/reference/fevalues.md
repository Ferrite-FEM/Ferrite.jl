```@meta
CurrentModule = Ferrite
DocTestSetup = :(using Ferrite)
```

# FEValues

## Main types
[`CellValues`](@ref), [`CellMultiValues`](@ref), and [`FaceValues`](@ref) are the most common 
subtypes of `Ferrite.AbstractValues`. For more details about how 
these work, please see the related [topic guide](@ref fevalues_topicguide).

```@docs
CellValues
CellMultiValues
FaceValues
```

## Applicable functions
The following functions are applicable 
`CellValues`, `FaceValues`, and `CellMultiValues`

```@docs
reinit!
getnquadpoints
getdetJdV
spatial_coordinate
geometric_value
```

Furthermore, the following functions are applicable to 
`CellValues`, `FaceValues`, and `FunctionValues` (obtained from [`CellMultiValues`](@ref))
```@docs
shape_value(::Ferrite.AbstractValues, ::Int, ::Int)
shape_gradient(::Ferrite.AbstractValues, ::Int, ::Int)
shape_symmetric_gradient
shape_divergence
shape_curl
getnbasefunctions

function_value
function_gradient
function_symmetric_gradient
function_divergence
function_curl
```

In addition, there are some methods that are unique for `FaceValues`.

```@docs
Ferrite.getcurrentface
getnormal
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
```
