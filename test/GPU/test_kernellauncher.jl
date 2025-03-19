# Sample CUDA Kernel (adding two vectors)
function kernel_add(A, B, C, n; mem_alloc)
    i = threadIdx().x + (blockIdx().x - 1) * blockDim().x
    if i <= n
        C[i] = A[i] + B[i]
    end
    return
end

# Helper function to launch the kernel with CUDAKernelLauncher
function test_launch_kernel!(n_cells::Integer, n_basefuncs::Integer, args...)
    return init_kernel(BackendCUDA, n_cells, n_basefuncs, kernel_add, args) |> launch!
end

# Testing for different integer types
@testset "Testing CUDAKernelLauncher with different integer types" begin
    # Arrays for testing
    N = 10
    A = CUDA.fill(1.0f0, N)
    B = CUDA.fill(2.0f0, N)
    C = CUDA.fill(0.0f0, N)

    # Test with Int32
    test_launch_kernel!(Int32(N), Int32(2), A, B, C, N)
    CUDA.synchronize()
    @test all(Array(C) .== 3.0f0)

    # Test with Int64
    fill!(C, 0.0f0)  # reset C array
    test_launch_kernel!(Int64(N), Int64(2), A, B, C, N)
    CUDA.synchronize()
    @test all(Array(C) .== 3.0f0)
end
