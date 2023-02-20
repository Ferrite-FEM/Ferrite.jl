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
include("FerritePartitionedArrays/iterators.jl")
include("FerritePartitionedArrays/assembler.jl")
include("FerritePartitionedArrays/constraints.jl")
include("FerritePartitionedArrays/vtk-export.jl")
    
end # module FerritePartitionedArrays
