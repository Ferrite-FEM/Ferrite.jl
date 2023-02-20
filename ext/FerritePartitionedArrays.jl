"""
Module containing the code for distributed assembly via PartitionedArrays.jl
"""
module FerritePartitionedArrays

using Ferrite
using MPI
using PartitionedArrays
using Base: @propagate_inbounds

include("FerritePartitionedArrays/assembler.jl")
include("FerritePartitionedArrays/constraints.jl")
    
end # module FerritePartitionedArrays
