# TODO document this
immutable FElement{FS <: FunctionSpace}
    name::Symbol
    function_space::FS
    inits::Dict{Symbol, NTuple{2, Int}}
    nnodes::Int
    dofs_per_node::Int
    flux_size::Int
    grad_kernel::Function
    source_kernel::Function
    flux_kernel::Function
    intf_kernel::Function
    default_intorder::Int # Should give a full rank stiffness matrix
end

n_dim(fele::FElement) = n_dim(fele.function_space)
n_flux(fele::FElement) = fele.flux_size

function show(io::IO, fele::FElement)
    print(io, "Name: $(fele.name)\n")
    print(io, "Shape function: $(typeof(fele.shape_funcs))\n")
    print(io, "Number of nodes: $(fele.nnodes)\n")
    print(io, "Dofs per node: $(fele.dofs_per_node)")
end
