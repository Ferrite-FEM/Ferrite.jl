module FerriteGPU
using Ferrite
using CUDA
using Adapt
using Base:
    @propagate_inbounds
using SparseArrays:
    AbstractSparseArray
using StaticArrays:
    SVector, MVector


include("GPU/gpu_assembler.jl")
include("GPU/CUDAKernelLauncher.jl")
include("GPU/cuda_iterator.jl")
include("GPU/adapt.jl")

end
