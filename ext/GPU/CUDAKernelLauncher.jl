function Ferrite.init_kernel(::Type{BackendCUDA}, n_cells::Ti, n_basefuncs::Ti, kernel::Function, args::Tuple) where {Ti <: Integer}
    if CUDA.functional()
        return LazyKernel(n_cells, n_basefuncs, kernel, args, BackendCUDA)
    else
        throw(ArgumentError("CUDA is not functional, please check your GPU driver and CUDA installation"))
    end
end


"""
    Ferrite.launch_kernel!(kernel_config::CUDAKernelLauncher{Ti}) where Ti

Launch a CUDA kernel with the given configuration.

Arguments:
- `kernel_config`: The `CUDAKernelLauncher` object containing a higher level fields for kernel configuration.
"""
function Ferrite.launch!(kernel::LazyKernel{Ti, BackendCUDA}) where {Ti}
    n_cells = kernel.n_cells
    n_basefuncs = kernel.n_basefuncs
    ker = kernel.kernel
    args = kernel.args
    kernel = @cuda launch = false ker(args...)
    config = launch_configuration(kernel.fun)
    threads = convert(Ti, min(n_cells, config.threads, 256))
    shared_mem = _calculate_shared_memory(threads, n_basefuncs)
    blocks = _calculate_nblocks(threads, n_cells)
    return kernel(args...; threads, blocks, shmem = shared_mem)
end


function _calculate_shared_memory(threads::Integer, n_basefuncs::Integer)
    return sizeof(Float32) * (threads) * (n_basefuncs) * n_basefuncs + sizeof(Float32) * (threads) * n_basefuncs
end


# function optimize_nthreads_for_dynshmem(max_threads::Int32, n_basefuncs::Int32)
#     dev = device()
#     MAX_DYN_SHMEM = CUDA.attribute(dev, CUDA.CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK_OPTIN) #size of dynamic shared memory
#     shmem_needed = sizeof(Float32) * (max_threads) * ( n_basefuncs) * n_basefuncs + sizeof(Float32) * (max_threads) * n_basefuncs
#     if(shmem_needed < MAX_DYN_SHMEM)
#         return max_threads, shmem_needed
#     else
#         # solve for threads
#         max_possible = Int32(MAX_DYN_SHMEM ÷ (sizeof(Float32) * ( n_basefuncs) * n_basefuncs + sizeof(Float32) * n_basefuncs))
#         warp_size = CUDA.attribute(dev, CUDA.CU_DEVICE_ATTRIBUTE_WARP_SIZE)
#         # approximate the number of threads to be a multiple of warp size (mostly 32)
#         nearest_no_warps = max_possible ÷ warp_size
#         if(nearest_no_warps < 4)
#             throw(ArgumentError("Bad implementation (less than 4 warps per block, wasted resources)"))
#         else
#             possiple_threads = nearest_no_warps * warp_size
#             shmem_needed = sizeof(Float32) * (possiple_threads) * ( n_basefuncs) * n_basefuncs + sizeof(Float32) * (possiple_threads) * n_basefuncs
#             return possiple_threads, shmem_needed
#         end
#     end
# end

"""
    _calculate_nblocks(threads::Int, n_cells::Int)

Calculate the number of blocks to be used in the kernel launch.
"""
function _calculate_nblocks(threads::Integer, n_cells::Integer)
    dev = device()
    no_sms = CUDA.attribute(dev, CUDA.CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT)
    # number of blocks is usually multiple of number of SMs
    # occupancy test should be done on threads and blocks
    # the goal is to calculate how many active block per SM and multiply it by the number of SMs
    required_blocks = cld(n_cells, threads)
    required_blocks < 2 * no_sms || return 2 * no_sms
    return required_blocks
end
