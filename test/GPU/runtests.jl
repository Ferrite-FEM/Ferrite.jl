using Ferrite
using CUDA
using Test
using SparseArrays

include("test_utils.jl")

# Unit tests
include("test_assemble.jl")
include("test_iterator.jl")
include("test_kernellauncher.jl")
include("test_adapt.jl")
