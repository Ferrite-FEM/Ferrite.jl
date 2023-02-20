"""
Module containing the code for a non-overlapping grid and the corresponding dof management via MPI.
"""
module FerriteMPI

using Ferrite
# TODO remove me. These are merely hotfixes to split the extensions trasiently via an internal API.
import Ferrite: getglobalgrid, num_global_dofs, num_global_dofs, num_local_dofs, num_local_true_dofs, global_comm, interface_comm, global_rank, compute_owner, local_dof_range, remote_entities
using Metis
using MPI
using Base: @propagate_inbounds

include("FerriteMPI/DistributedGrid.jl")
include("FerriteMPI/DistributedDofHandler.jl")
include("FerriteMPI/iterators.jl")
include("FerriteMPI/vtk-export.jl")

function __init__()
    @info "FerriteMPI extension loaded."
end

end # module FerriteMPI
