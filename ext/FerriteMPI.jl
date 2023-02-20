"""
Module containing the code for a non-overlapping grid and the corresponding dof management via MPI.
"""
module FerriteMPI

using Ferrite
using Metis
using MPI
using Base: @propagate_inbounds

include("FerriteMPI/grid.jl")
include("FerriteMPI/DistributedDofHandler.jl")
include("FerriteMPI/iterators.jl")
include("FerriteMPI/vtk-export.jl")
    
end # module FerriteMPI
