using CUDA
using Ferrite
using Test

@test CUDA.functional()

include("heat_assembly.jl")
