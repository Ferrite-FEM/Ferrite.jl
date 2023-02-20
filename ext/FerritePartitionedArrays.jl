"""
Module containing the code for distributed assembly via PartitionedArrays.jl
"""
module FerritePartitionedArrays

using Ferrite
# TODO remove me. These are merely hotfixes to split the extensions trasiently via an internal API.
import Ferrite: getglobalgrid, num_global_dofs, num_global_dofs, num_local_true_dofs, num_local_dofs, global_comm, interface_comm, global_rank, compute_owner, remote_entities
using MPI
using PartitionedArrays
using Base: @propagate_inbounds

include("FerritePartitionedArrays/assembler.jl")
include("FerritePartitionedArrays/constraints.jl")
include("FerritePartitionedArrays/export-vtk.jl")

function __init__()
    @info "FerritePartitionedArrays extension loaded."
end

end # module FerritePartitionedArrays
