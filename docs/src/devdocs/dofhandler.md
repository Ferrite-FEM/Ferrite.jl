# [Dof Handler](@id dofhandler-interpolations)

## Type definitions

Dof handlers are subtypes of `AbstractDofhandler{sdim}`, i.e. they are
parametrized by the spatial dimension. Internally a helper struct
[`InterpolationInfo`](@ref Ferrite.InterpolationInfo) is utilized to enforce type stability
during dof distribution, because the interpolations are not available as concrete types.

```@docs
Ferrite.InterpolationInfo
Ferrite.PathOrientationInfo
Ferrite.SurfaceOrientationInfo
```


## Internal API

The main entry point for dof distribution is [`__close!`](@ref Ferrite.__close!).

```@docs
Ferrite.__close!
Ferrite.get_grid
Ferrite.find_field
Ferrite._find_field
Ferrite._close_subdofhandler!
Ferrite._distribute_dofs_for_cell!
Ferrite.permute_and_push!
```
