```@meta
CurrentModule = Ferrite
DocTestSetup = :(using Ferrite)
```

# FEValues

## Main types
[`CellValues`](@ref), [`MultiFieldCellValues`](@ref), and [`FacetValues`](@ref) are the most common
subtypes of `Ferrite.AbstractValues`. For more details about how
these work, please see the related [topic guide](@ref fevalues_topicguide).

```@docs
CellValues
MultiFieldCellValues
FacetValues
```

## Applicable functions
The following functions are applicable to
`CellValues`, `FacetValues`, and `MultiFieldCellValues`

```@docs
reinit!
getnquadpoints
getdetJdV
spatial_coordinate
geometric_value
```

Furthermore, the following functions are applicable to
`CellValues`, `FacetValues`, and `FunctionValues` (obtained from [`MultiFieldCellValues`](@ref))
```@docs
shape_value(::Ferrite.AbstractValues, ::Int, ::Int)
shape_gradient(::Ferrite.AbstractValues, ::Int, ::Int)
shape_symmetric_gradient
shape_divergence
shape_curl
getnbasefunctions(::Ferrite.AbstractValues)
function_value
function_gradient
function_symmetric_gradient
function_divergence
function_curl
```

### Local frame derivatives
For e.g. shell elements, it is convenient to work with derivatives wrt. coordinates in a local
orthonormal frame that is tangent to the element. This frame is obtained by Gram-Schmidt
orthonormalization of the columns of the jacobian, ``\mathbf{J} = \partial \mathbf{x} / \partial \boldsymbol{\xi}``,
in each quadrature point, see [`shape_local_gradient`](@ref) for details. The gradients and hessians
wrt. the local frame coordinates are updated if the keyword arguments `update_local_gradients = true`
and `update_local_hessians = true`, respectively, are given when constructing the values object.
This is supported for interpolations with identity mapping, also for embedded elements
(e.g. `CellValues(qr, ip, ip_geo^3)` for surfaces in 3D). Furthermore, the jacobians of the geometric mapping
are stored if `update_jacobians = true` is given.

```@docs
shape_local_gradient
shape_local_hessian
function_local_gradient
function_local_hessian
Ferrite.getjacobian
```

In addition, there are some methods that are unique for `FacetValues`.

```@docs
Ferrite.getcurrentfacet
getnormal
```

## [InterfaceValues](@id reference-interfacevalues)

All of the methods for [`FacetValues`](@ref) apply for `InterfaceValues` as well.
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
