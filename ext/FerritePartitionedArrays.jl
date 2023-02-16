"""
Module containing the code for distributed assembly via PartitionedArrays.jl
"""
module FerritePartitionedArrays

using Ferrite
using Metis
using MPI
using PartitionedArrays
using Base: @propagate_inbounds

include("FerritePartitionedArrays/grid.jl")
include("FerritePartitionedArrays/DistributedDofHandler.jl")
include("FerritePartitionedArrays/assembler.jl")
include("FerritePartitionedArrays/constraints.jl")
include("FerritePartitionedArrays/iterators.jl")
include("FerritePartitionedArrays/vtk-export.jl")

export 
    # assembler
    COOAssembler,
    # dofhandler
    DistributedDofHandler,
    # grid
    DistributedGrid,
    generate_distributed_grid,
    # vtk-export
    vtk_shared_vertices,
    vtk_shared_faces,
    vtk_shared_edges,
    vtk_partitioning    
    
end # module FerritePartitionedArrays
