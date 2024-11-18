## This file manifsts the launch of GPU kernel on CUDA backend ##

"""
    Ferrite.init_kernel(::Type{BackendCUDA}, n_cells::Ti, n_basefuncs::Ti, kernel::Function, args::Tuple) where {Ti <: Integer}

Initialize a CUDA kernel for the Ferrite framework.

# Arguments
- `::Type{BackendCUDA}`: Specifies the CUDA backend.
- `n_cells::Ti`: Number of cells in the problem.
- `n_basefuncs::Ti`: Number of shape functions per cell.
- `kernel::Function`: The CUDA kernel function to execute.
- `args::Tuple`: Tuple of arguments for the kernel.

# Returns
- A `LazyKernel` object encapsulating the kernel and its execution configuration.

# Errors
Throws an `ArgumentError` if CUDA is not functional (e.g., due to missing drivers or improper installation).
"""
function Ferrite.init_kernel(::Type{BackendCUDA}, n_cells::Ti, n_basefuncs::Ti, kernel::Function, args::Tuple) where {Ti <: Integer}
    if CUDA.functional()
        return LazyKernel(n_cells, n_basefuncs, kernel, args, BackendCUDA)
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
    _can_use_dynshmem(shared_mem) && return CUDA.@sync kernel(args...; threads, blocks, shmem = shared_mem)

    ## otherwise use global memory
    nes = blocks * threads
    kes = CUDA.zeros(Float32, nes, n_basefuncs, n_basefuncs)
    fes = CUDA.zeros(Float32, nes, n_basefuncs)
    args = _to_localdh(args, kes, fes)
    CUDA.@sync @cuda blocks = blocks threads = threads ker(args...)
    return nothing
end

"""
    _to_localdh(args::Tuple, kes::AbstractArray, fes::AbstractArray)

Convert a global degree-of-freedom handler to a local handler for use on the GPU.

# Arguments
- `args::Tuple`: Kernel arguments.
- `kes::AbstractArray`: GPU storage for element stiffness matrices.
- `fes::AbstractArray`: GPU storage for element force vectors.

# Returns
- `Tuple`: Updated arguments tuple with the degree-of-freedom handler replaced by a local GPU handler.

# Errors
Throws an `ErrorException` if no `AbstractDofHandler` is found in `args`.
"""
function _to_localdh(args::Tuple, kes::AbstractArray, fes::AbstractArray)
    dh_index = findfirst(x -> x isa Ferrite.AbstractDofHandler, args)
    dh_index !== nothing || throw(ErrorException("No subtype of AbstractDofHandler found in the arguments"))
    arr = args |> collect
    local_dh = LocalsGPUDofHandler(arr[dh_index], kes, fes)
    arr[dh_index] = local_dh
    return Tuple(arr)
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
function _calculate_shared_memory(threads::Integer, n_basefuncs::Integer)
    return sizeof(Float32) * (threads) * (n_basefuncs) * n_basefuncs + sizeof(Float32) * (threads) * n_basefuncs
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
function _calculate_nblocks(threads::Integer, n_cells::Integer)
    dev = device()
    no_sms = CUDA.attribute(dev, CUDA.CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT)
    required_blocks = cld(n_cells, threads)
    required_blocks < 2 * no_sms || return 2 * no_sms
    return required_blocks
end
