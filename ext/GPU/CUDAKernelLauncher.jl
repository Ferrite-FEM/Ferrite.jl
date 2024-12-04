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
    return if CUDA.functional()
        threads = convert(Ti, min(n_cells, 256))
        shared_mem = _calculate_shared_memory(threads, n_basefuncs)
        blocks = _calculate_nblocks(threads, n_cells)
        _adapted_args = _adapt_args(CuArray, args)

        if (_can_use_dynshmem(shared_mem) && false)
            Ke = DynamicSharedMemFunction{3, Float32, Int32}((threads, n_basefuncs, n_basefuncs), Int32(0))
            fe = DynamicSharedMemFunction{2, Float32, Int32}((threads, n_basefuncs), sizeof(Float32) * threads * n_basefuncs * n_basefuncs |> Int32)
            mem_alloc = SharedMemAlloc(Ke, fe, shared_mem)
            return CudaKernel(n_cells, n_basefuncs, kernel, _adapted_args, mem_alloc, threads, blocks)
        else
            Kes = CUDA.zeros(Float32, n_cells, n_basefuncs, n_basefuncs)
            fes = CUDA.zeros(Float32, n_cells, n_basefuncs)
            mem_alloc = GlobalMemAlloc(Kes, fes)
            return CudaKernel(n_cells, n_basefuncs, kernel, _adapted_args, mem_alloc, threads, blocks)
        end
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
    _calculate_shared_memory(threads::Integer, n_basefuncs::Integer)

Calculate the shared memory required for kernel execution.

# Arguments
- `threads::Integer`: Number of threads per block.
- `n_basefuncs::Integer`: Number of basis functions per cell.

# Returns
- `Integer`: Amount of shared memory in bytes.
"""
function _calculate_shared_memory(threads::Ti, n_basefuncs::Ti) where {Ti <: Integer}
    return convert(Ti, sizeof(Float32) * (threads) * (n_basefuncs) * n_basefuncs + sizeof(Float32) * (threads) * n_basefuncs)
end

"""
    _can_use_dynshmem(required_shmem::Integer)

Check if the GPU supports the required amount of dynamic shared memory.

# Arguments
- `required_shmem::Integer`: Required shared memory size in bytes.

# Returns
- `Bool`: `true` if the GPU can provide the required shared memory; `false` otherwise.
"""
function _can_use_dynshmem(required_shmem::Integer)
    dev = device()
    MAX_DYN_SHMEM = CUDA.attribute(dev, CUDA.CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK)
    return required_shmem < MAX_DYN_SHMEM
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
