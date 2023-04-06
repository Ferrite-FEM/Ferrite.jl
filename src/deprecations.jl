Base.@deprecate_binding DirichletBoundaryConditions ConstraintHandler
Base.@deprecate_binding DirichletBoundaryCondition Dirichlet

import Base: push!
@deprecate push!(dh::AbstractDofHandler, args...) add!(dh, args...)

@deprecate vertices(ip::Interpolation) vertexdof_indices(ip) false
@deprecate faces(ip::Interpolation) facedof_indices(ip) false
@deprecate edges(ip::Interpolation) edgedof_indices(ip) false
@deprecate nfields(dh::AbstractDofHandler) length(getfieldnames(dh)) false
@deprecate getcoordinates(node::Node) get_node_coordinate(node) true
@deprecate getcoordinates(args...) get_cell_coordinates(args...) true
@deprecate getcoordinates!(args...) get_cell_coordinates!(args...) true
@deprecate cellcoords!(x::Vector, dh::DofHandler, args...) get_cell_coordinates!(x, dh.grid, args...) false