# [Dof Handler](@id dofhandler-interpolations)

## Type definitions

Dof handlers are subtypes of `AbstractDofhandler{sdim}`, i.e. they are
parametrized by the spatial dimension. Internally a helper struct [`InterpolationInfo`](@ref) is utilized to enforce type stability during
dof distribution, because the interpolations are not available as concrete
types.

```@docs
Ferrite.AbstractDofhandler
Ferrite.InterpolationInfo
Ferrite.PathOrientationInfo
Ferrite.SurfaceOrientationInfo
```


## Internal API

The main entry point for dof distribution is [`__close!`](@ref).

```@docs
Ferrite.find_field(dh::DofHandler, field_name::Symbol)
Ferrite.find_field(fh::FieldHandler, field_name::Symbol)::Int
Ferrite._find_field(fh::FieldHandler, field_name::Symbol)
Ferrite.__close(::DofHandler)
Ferrite._close_fieldhandler!
Ferrite._close_fieldhandler_on_cell!
Ferrite._check_cellset_intersections
Ferrite.dof_correction!(cell_dofs::Vector{Int}, dofs::StepRange{Int,Int}, orientation::Ferrite.PathOrientationInfo, correction_info::Bool)
Ferrite.dof_correction!(cell_dofs::Vector{Int}, dofs::StepRange{Int,Int}, orientation::Ferrite.SurfaceOrientationInfo, correction_info::Bool)
Ferrite._dof_correction!(cell_dofs::Vector{Int}, dofs::StepRange{Int,Int})
Ferrite.add_vertex_dofs
Ferrite.add_face_dofs
Ferrite.add_edge_dofs
Ferrite.add_cell_dofs
```
