# [FEValues](@id devdocs-fevalues)

## Type definitions
* `AbstractValues`
  * `AbstractCellValues`
    * [`CellValues`](@ref)
  * `AbstractFaceValues`
    * [`FaceValues`](@ref)
  * [`PointValues`](@ref)
  * `PointValuesInternal` (Optimized version of PointValues)

## Internal types
```@docs
Ferrite.GeometryMapping
Ferrite.MappingValues
Ferrite.FunctionValues
```

## Interface
* The function interpolation, `ip_fun`, determines how it should be mapped, by defining `get_mapping_type(ip_fun)` for its type.
* The mapping type, e.g. `IdentityMapping`, decides the requirements for `GeometryValues`, specifically if the `hessian` $\partial^2M/\partial\xi^2$,
  of the geometric shape functions, $M(\xi)$, on the reference cell should be precalculated or not. 
  ***Note:*** *This should also in the future be determined by the required order of derivatives to be mapped in `FunctionValues`*
* As the first part of `reinit!`, the `MappingValues` are calculated based on the cell's coordinates. If the `GeometricMapping` contains the hessian 
  on the reference cell, the `hessian` on the actual cell, $\partial^2M/\partial x^2$, is calculated and returned in `MappingValues`. Otherwise, only
  the jacobian, $\partial M/\partial x$, is calculated. 
* In the second part of `reinit!`, The `MappingValues`, containing sufficient information for the current quadrature point, is given to `apply_mapping!`. 
  This allows the shape values and gradients stored in `FunctionValues`, to be mapped to the current cell by calling `apply_mapping!`.
