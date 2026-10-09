using CUDA
using Ferrite
using Test

@test CUDA.functional()

include("heat_assembly.jl")
include("../test_multifield_cellvalues.jl")
test_multifield_cellvalues(CUDABackend())
