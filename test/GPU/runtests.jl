# Entry point for the GPU tests, run from `.buildkite/pipeline.yml` with
# `--project=test/GPU/<backend>` and `FERRITE_GPU_BACKEND=<backend>`.
using Test
using SparseArrays
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
    import AMDGPU.rocSPARSE: ROCSparseMatrixCSC, ROCSparseMatrixCSR
    @test AMDGPU.functional()
    backend = ROCBackend()
    sparse_type(Tv, Ti) = ROCSparseMatrixCSC{Tv, Ti}
    sparse_type_csr(Tv, Ti) = ROCSparseMatrixCSR{Tv, Ti}
elseif gpu_backend == "oneapi"
    using oneAPI
    import oneAPI.oneMKL: oneSparseMatrixCSC, oneSparseMatrixCSR
    @test oneAPI.functional()
    backend = oneAPIBackend()
    sparse_type(Tv, Ti) = oneSparseMatrixCSC{Tv, Ti}
    sparse_type_csr(Tv, Ti) = oneSparseMatrixCSR{Tv, Ti}
elseif gpu_backend == "metal"
    # Metal has no sparse matrix type of its own, the backend agnostic matrices from
    # GenericSparseArrays.jl are used instead.
    using Metal
    using GenericSparseArrays
    @test Metal.functional()
    backend = MetalBackend()
    sparse_type(Tv, Ti) = GenericSparseMatrixCSC{Tv, Ti}
    sparse_type_csr(Tv, Ti) = GenericSparseMatrixCSR{Tv, Ti}
else
    error("unknown FERRITE_GPU_BACKEND=$gpu_backend")
end

include("ka_common.jl")

gpu_backend == "cuda" && include("howto.jl")

include("heat_assembly.jl")
