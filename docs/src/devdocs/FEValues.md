# [FEValues](@id devdocs-fevalues)

## Type definitions
* `AbstractValues`
  * `AbstractCellValues`
    * [`CellValues`](@ref)
  * `AbstractFacetValues`
    * [`FacetValues`](@ref)
    * [`BCValues`](@ref Ferrite.BCValues)
  * [`PointValues`](@ref)
  * [`InterfaceValues`](@ref)


## Internal types
```@docs
Ferrite.GeometryMapping
Ferrite.MappingValues
Ferrite.FunctionValues
Ferrite.BCValues
```

## Internal utilities
```@docs
Ferrite.embedding_det
Ferrite.shape_value_type(::Ferrite.AbstractValues)
Ferrite.shape_gradient_type
Ferrite.ValuesUpdateFlags
```

## Custom FEValues
Custom FEValues, `fe_v::AbstractValues`, should normally implement the [`reinit!`](@ref) method. Subtypes of `AbstractValues` have default implementations for some functions, but require some lower-level access functions, specifically

* [`function_value`](@ref), requires
  * [`shape_value`](@ref)
  * [`getnquadpoints`](@ref)
  * [`getnbasefunctions`](@ref)
* [`function_gradient`](@ref), [`function_divergence`](@ref), [`function_symmetric_gradient`](@ref), and [`function_curl`](@ref) requires
  * [`shape_gradient`](@ref)
  * [`getnquadpoints`](@ref)
  * [`getnbasefunctions`](@ref)
* [`spatial_coordinate`](@ref), requires
  * [`geometric_value`](@ref Ferrite.geometric_value)
  * `getngeobasefunctions`
  * [`getnquadpoints`](@ref)


### Array bounds
* Asking for the `n`th quadrature point must be inside array bounds if `1 <= n <= getnquadpoints(fe_v)`. (`checkquadpoint` can, alternatively, be dispatched to check that `n` is inbounds.)
* Asking for the `i`th shape value or gradient must be inside array bounds if `1 <= i <= getnbasefunctions(fe_v)`
* Asking for the `i`th geometric value must be inside array bounds if `1 <= i <= getngeobasefunctions(fe_v)`
