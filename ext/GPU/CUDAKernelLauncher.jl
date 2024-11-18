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

    ## use dynamic shared memory if possible
    _can_use_dynshmem(shared_mem) && return kernel(args...; threads, blocks, shmem = shared_mem)

    ## otherwise use global memory
    nes = blocks * threads
    kes = CUDA.zeros(Float32, nes, n_basefuncs, n_basefuncs)
    fes = CUDA.zeros(Float32, nes, n_basefuncs)
    args = _to_localdh(args, kes, fes)
    @cuda blocks = blocks threads = threads ker(args...)
    return nothing
end


function _to_localdh(args::Tuple, kes::AbstractArray, fes::AbstractArray)
    dh_index = findfirst(x -> x isa Ferrite.AbstractDofHandler, args)
    dh_index !== nothing || throw(ErrorException("No subtype of AbstractDofHandler found in the arguments"))
    arr = args |> collect
    local_dh = LocalsGPUDofHandler(arr[dh_index], kes, fes)
    arr[dh_index] = local_dh
    return Tuple(arr)
end

function _calculate_shared_memory(threads::Integer, n_basefuncs::Integer)
    return sizeof(Float32) * (threads) * (n_basefuncs) * n_basefuncs + sizeof(Float32) * (threads) * n_basefuncs
end


function _can_use_dynshmem(required_shmem::Integer)
    dev = device()
    MAX_DYN_SHMEM = CUDA.attribute(dev, CUDA.CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK) #size of dynamic shared memory
    return required_shmem < MAX_DYN_SHMEM
end


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
