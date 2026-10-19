using CUDA
using Ferrite
using Test

@test CUDA.functional()

include("heat_assembly.jl")
include("../multifield_cellvalues_utils.jl")
test_multifield_cellvalues(CUDABackend())
