"""
Module containing the code for distributed assembly via PartitionedArrays.jl
"""
module FerritePartitionedArrays

using Ferrite
using Metis
using MPI
using PartitionedArrays

include("FerritePartitionedArrays/assembler.jl")
include("FerritePartitionedArrays/constraints.jl")
include("FerritePartitionedArrays/DistributedDofHandler.jl")
include("FerritePartitionedArrays/grid.jl")
include("FerritePartitionedArrays/vtk-export.jl")

export 
    # assembler
    COOAssembler,
    # grid
    DistributedGrid,
    # vtk-export
    vtk_shared_vertices,
    vtk_shared_faces,
    vtk_shared_edges,
    vtk_partitioning,
end # module FerritePartitionedArrays
