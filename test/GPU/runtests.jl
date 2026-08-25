# Entry point for the GPU tests, run from `.buildkite/pipeline.yml` with
# `--project=test/GPU/<backend>` and `FERRITE_GPU_BACKEND=<backend>`.
using Test
using SparseArrays
using SparseMatricesCSR
import KernelAbstractions as KA

const gpu_backend = get(ENV, "FERRITE_GPU_BACKEND", "cuda")

if gpu_backend == "cuda"
    using CUDA
    import CUDA.CUSPARSE: CuSparseMatrixCSC, CuSparseMatrixCSR
    @test CUDA.functional()
    backend = CUDABackend()
    sparse_type(Tv, Ti) = CuSparseMatrixCSC{Tv, Ti}
    sparse_type_csr(Tv, Ti) = CuSparseMatrixCSR{Tv, Ti}
elseif gpu_backend == "amdgpu"
    using AMDGPU
    @test AMDGPU.functional()
    backend = ROCBackend()
elseif gpu_backend == "oneapi"
    using oneAPI
    @test oneAPI.functional()
    backend = oneAPIBackend()
elseif gpu_backend == "metal"
    using Metal
    @test Metal.functional()
    backend = MetalBackend()
else
    error("unknown FERRITE_GPU_BACKEND=$gpu_backend")
end

include("ka_common.jl")

if gpu_backend == "cuda"
    include("howto.jl")
    include("heat_assembly.jl")
else
    # Sparse matrix allocation and the KernelAbstractions assembler for the other vendors
    # are not implemented yet. Until then this only verifies that the backend loads and is
    # functional on the CI agent, and runs the KA test suite on the CPU backend there.
    backend = KA.CPU()
    sparse_type(Tv, Ti) = SparseMatrixCSC{Tv, Ti}
    sparse_type_csr(Tv, Ti) = SparseMatrixCSR{1, Tv, Ti}
    include("heat_assembly.jl")
end
