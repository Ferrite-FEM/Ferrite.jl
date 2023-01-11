Base.@deprecate_binding DirichletBoundaryConditions ConstraintHandler
Base.@deprecate_binding DirichletBoundaryCondition Dirichlet

import Base: push!
@deprecate push!(dh::AbstractDofHandler, args...) add!(dh, args...)

@deprecate vertices(ip::Interpolation) vertexdof_indices(ip)
@deprecate faces(ip::Interpolation) facedof_indices(ip)
@deprecate edges(ip::Interpolation) edgedof_indices(ip)
