## This file manifsts the launch of GPU kernel on CUDA backend ##
abstract type AbstractCudaKernel <: Ferrite.AbstractKernel{BackendCUDA} end


struct CudaKernel{MemAlloc <: AbstractCudaMemAlloc, Ti <: Integer} <: AbstractCudaKernel
    n_cells::Ti
    n_basefuncs::Ti
    kernel::Function
    args::Tuple
    mem_alloc::MemAlloc
    threads::Ti
    blocks::Ti
end


function Ferrite.init_kernel(::Type{BackendCUDA}, n_cells::Ti, n_basefuncs::Ti, kernel::Function, args::Tuple) where {Ti <: Integer}
    if CUDA.functional()
        threads = convert(Ti, min(n_cells, 256))
        blocks = _calculate_nblocks(threads, n_cells)
        adapted_args = _adapt_args(args)

        mem_alloc = try_allocate_shared_mem(Float32, threads, n_basefuncs)
        mem_alloc isa Nothing || return CudaKernel(n_cells, n_basefuncs, kernel, adapted_args, mem_alloc, threads, blocks)

        # FIXME: mem_alloc adapted twice here and in launch!
        mem_alloc = allocate_global_mem(Float32, n_cells, n_basefuncs) |> _adapt
        return CudaKernel(n_cells, n_basefuncs, kernel, adapted_args, mem_alloc, threads, blocks)
    else
        throw(ArgumentError("CUDA is not functional, please check your GPU driver and CUDA installation"))
    end
end

"""
    Ferrite.launch!(kernel::LazyKernel{Ti, BackendCUDA}) where {Ti}

Launch a CUDA kernel encapsulated in a `LazyKernel` object.

# Arguments
- `kernel::LazyKernel`: The kernel to be launched, along with its configuration.

# Returns
- `nothing`: Indicates that the kernel was launched and synchronized successfully.
"""
function Ferrite.launch!(kernel::CudaKernel{SharedMemAlloc{N, M, Tv, Ti}, Ti}) where {N, M, Tv, Ti}
    ker = kernel.kernel
    args = kernel.args
    blocks = kernel.blocks
    threads = kernel.threads
    shmem_size = mem_size(kernel.mem_alloc)
    kwargs = (mem_alloc = kernel.mem_alloc,)
    kernel_fun = () -> ker(args...; kwargs...)
    CUDA.@sync @cuda blocks = blocks threads = threads shmem = shmem_size kernel_fun()

    return nothing
end


function Ferrite.launch!(kernel::CudaKernel{GlobalMemAlloc{LOCAL_MATRICES, LOCAL_VECTORS}, Ti}) where {LOCAL_MATRICES, LOCAL_VECTORS, Ti}
    ker = kernel.kernel
    args = kernel.args
    blocks = kernel.blocks
    threads = kernel.threads
    kwargs = (mem_alloc = kernel.mem_alloc,)
    kernel_fun = () -> ker(args...; kwargs...)
    CUDA.@sync @cuda blocks = blocks threads = threads kernel_fun()
    return nothing
end

"""
    _calculate_nblocks(threads::Integer, n_cells::Integer)

Calculate the number of blocks required for kernel execution.

# Arguments
- `threads::Integer`: Number of threads per block.
- `n_cells::Integer`: Total number of cells to process.

# Returns
- `Integer`: Number of blocks to launch.
"""
function _calculate_nblocks(threads::Ti, n_cells::Ti) where {Ti <: Integer}
    dev = device()
    no_sms = CUDA.attribute(dev, CUDA.CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT)
    required_blocks = cld(n_cells, threads)
    required_blocks < 2 * no_sms || return convert(Ti, 2 * no_sms)
    return convert(Ti, required_blocks)
end
