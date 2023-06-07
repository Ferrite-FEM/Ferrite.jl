# [Dof Handler](@id dofhandler-interpolations)

## Type definitions

Dof handlers are subtypes of `AbstractDofhandler{sdim}`, i.e. they are
parametrized by the spatial dimension. Internally a helper struct [`InterpolationInfo`](@ref) is utilized to enforce type stability during
dof distribution, because the interpolations are not available as concrete
types.

```@docs
Ferrite.InterpolationInfo
Ferrite.PathOrientationInfo
Ferrite.SurfaceOrientationInfo
```


## Internal API

The main entry point for dof distribution is [`__close!`](@ref).

```@docs
Ferrite.getgrid
Ferrite.find_field(dh::DofHandler, field_name::Symbol)
Ferrite._find_field(fh::FieldHandler, field_name::Symbol)
Ferrite._close_fieldhandler!
Ferrite._distribute_dofs_for_cell!
Ferrite.permute_and_push!
```
