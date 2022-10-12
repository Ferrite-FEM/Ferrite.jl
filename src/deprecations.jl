Base.@deprecate_binding DirichletBoundaryConditions ConstraintHandler
Base.@deprecate_binding DirichletBoundaryCondition Dirichlet
@deprecate ndim(dh::AbstractDofHandler, field_name::Symbol) getfielddim(dh::AbstractDofHandler, field_name::Symbol)
@deprecate nfields(dh::AbstractDofHandler) num_fields(dh::AbstractDofHandler)
