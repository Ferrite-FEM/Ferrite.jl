function launch_kernel(f::Function , args::Tuple,n_cells::Int, n_basefuncs::Int)
    kernel = @cuda launch=false f(args...)
    config = launch_configuration(kernel.fun)
    max_threads = min(n_cells, config.threads)
    max_threads = 256 # for now (occupancy test should be done on threads and blocks)
    threads, shared_mem = optimize_nthreads_for_dynshmem(max_threads, n_basefuncs)
    max_threads = min(threads, config.threads)
    #blocks =  cld(n_cells, max_threads)
    blocks = optimize_nblocks()
    kernel(args...;  threads, blocks, shmem=shared_mem)
end


function optimize_nthreads_for_dynshmem(max_threads, n_basefuncs)
    MAX_DYN_SHMEM = 48 * 1024 # TODO: get the maximum shared memory per block from the device (48KB for now, currently I don't know how to get this value)
    shmem_needed = sizeof(Float32) * (max_threads) * ( n_basefuncs) * n_basefuncs + sizeof(Float32) * (max_threads) * n_basefuncs
    if(shmem_needed < MAX_DYN_SHMEM)
        return max_threads, shmem_needed
    else
        # solve for threads
        max_possible = Int32(MAX_DYN_SHMEM ÷ (sizeof(Float32) * ( n_basefuncs) * n_basefuncs + sizeof(Float32) * n_basefuncs))
        dev = device()
        warp_size = CUDA.attribute(dev, CUDA.CU_DEVICE_ATTRIBUTE_WARP_SIZE)
        # approximate the number of threads to be a multiple of warp size (mostly 32)
        nearest_no_warps = max_possible ÷ warp_size
        if(nearest_no_warps < 4)
            throw(ArgumentError("Bad implementation (less than 4 warps per block, wasted resources)"))
        else
            possiple_threads = nearest_no_warps * warp_size
            shmem_needed = sizeof(Float32) * (possiple_threads) * ( n_basefuncs) * n_basefuncs + sizeof(Float32) * (possiple_threads) * n_basefuncs
            return possiple_threads, shmem_needed
        end
    end
end

function optimize_nblocks()
    dev = device()
    no_sms = CUDA.attribute(dev, CUDA.CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT)
    # number of blocks is usually multiple of number of SMs
    # occupancy test should be done on threads and blocks
    # the goal is to calculate how many active block per SM and multiply it by the number of SMs
    return 2 #* no_sms
end
