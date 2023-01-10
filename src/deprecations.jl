Base.@deprecate_binding DirichletBoundaryConditions ConstraintHandler
Base.@deprecate_binding DirichletBoundaryCondition Dirichlet
function end_assemble(a::IJVAssembler)
    Base.depwarn("end_assemble have been renamed to finish_assemble and now " *
                 "also return the global vector", :end_assemble)
    A, _ = finish_assemble(a)
    return A
end
Base.@deprecate_binding Assembler IJVAssembler
