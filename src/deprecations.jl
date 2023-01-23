Base.@deprecate_binding DirichletBoundaryConditions ConstraintHandler
Base.@deprecate_binding DirichletBoundaryCondition Dirichlet
import Base: push!
@deprecate push!(dh::AbstractDofHandler, args...) add!(dh, args...)
