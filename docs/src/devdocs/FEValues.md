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

## How CellValues and FaceValues works
* The function interpolation, `ip_fun`, determines how it should be mapped, by defining `get_mapping_type(ip_fun)` for its type.
* The mapping type, e.g. `IdentityMapping`, decides the requirements for `GeometryValues`, specifically if the `hessian` $\partial^2M/\partial\xi^2$,
  of the geometric shape functions, $M(\xi)$, on the reference cell should be precalculated or not. 
  ***Note:*** *This should also in the future be determined by the required order of derivatives to be mapped in `FunctionValues`*
* As the first part of `reinit!`, the `MappingValues` are calculated based on the cell's coordinates. If the `GeometricMapping` contains the hessian 
  on the reference cell, the `hessian` on the actual cell, $\partial^2M/\partial x^2$, is calculated and returned in `MappingValues`. Otherwise, only
  the jacobian, $\partial M/\partial x$, is calculated. 
* In the second part of `reinit!`, The `MappingValues`, containing sufficient information for the current quadrature point, is given to `apply_mapping!`. 
  This allows the shape values and gradients stored in `FunctionValues`, to be mapped to the current cell by calling `apply_mapping!`.

## Custom FEValues
Custom FEValues, `fe_v`, should normally implement the `reinit!` method.
Additionally, for normal functionality the `getnquadpoints` should be implemented.
Note that asking for the `n`th quadrature point must be inside array bounds if 
`1<=n<:getnquadpoints(fe_v)`
(`checkquadpoint` can, alternatively, be dispatched to check that `n` is inbounds.)

Supporting `function_value`, `function_gradient`, `function_symmetric_gradient`, `function_divergence`, and `function_curl`,
requires implementing `getnbasefunctions`, `shape_value`, and `shape_gradient`. 
Note that asking for the `i`th shape value or gradient must be inside array bounds if `1<=i<:getnbasefunctions(fe_v)`

Supporting `spatial_coordinate` requires implementing `getngeobasefunctions` and `geometric_value`.
Note that asking for the `i`th geometric value must be inside array bounds if `1<=i<:getngeobasefunctions(fe_v)`