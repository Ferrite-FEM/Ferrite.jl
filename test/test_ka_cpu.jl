# The KernelAbstractions GPU layer run on the CPU backend, so that the backend-agnostic
# parts are tested without GPU hardware. The vendor backends run the same file from
# test/GPU/runtests.jl.
using SparseArrays
import KernelAbstractions as KA
import GPUArrays # loads GPUArraysCore too, which FerriteKAExt requires

include(joinpath(@__DIR__, "GPU", "ka_common.jl"))

backend = KA.CPU()
sparse_type(Tv, Ti) = SparseMatrixCSC{Tv, Ti}

include(joinpath(@__DIR__, "GPU", "heat_assembly.jl"))
