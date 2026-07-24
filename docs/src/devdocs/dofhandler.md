# [Dof handler](@id dofhandler-interpolations)

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


## Dof distribution

Dofs are distributed cell by cell as documented in [`__close!`](@ref Ferrite.__close!):
for each cell, and for each field in the order they were added, dofs are created on (or
reused from) the vertices, edge interiors, face interiors, and the volume interior of the
cell. Dof creation for entities shared with already visited cells is tracked with
per-field dictionaries keyed by the (sorted) global vertex numbers of the entity.

Global fields ([`GlobalConstant`](@ref)) are a special case: the field's dofs are created
once, when the first cell carrying the field is visited, and the *same* dof numbers are
appended (in the regular field position) to `cell_dofs` for every cell of the
SubDofHandler(s) with the field. Every base function index of an interpolation must be
classified exactly once as vertex-, edge-, face-, volume-, or global-owned, which is
asserted during dof distribution.

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
