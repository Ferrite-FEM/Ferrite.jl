#=
This file defines the interface between the GPU backend (extension) and the Ferrite package.
It provides abstract types, function signatures, and concrete types for managing GPU kernels
and backends, serving as a foundation for GPU-accelerated computations.
=#

### Abstract Types ###
abstract type AbstractBackend end
abstract type AbstractKernel{BKD <: AbstractBackend} end


### Functions ###

"""
    init_gpu_kernel(backend::AbstractGPUBackend, n_cells::Ti, n_basefuncs::Ti, kernel::Function, args::Tuple) where {Ti <: Integer}

Initializes a GPU kernel with the specified backend, number of cells, base functions,
kernel function, and additional arguments.

# Arguments
- `backend::AbstractGPUBackend`: The GPU backend to use for kernel execution.
- `n_cells::Ti`: Number of cells to be processed by the kernel.
- `n_basefuncs::Ti`: Number of base functions for each cell.
- `kernel::Function`: The kernel function to execute on the GPU.
- `args::Tuple`: Additional arguments required by the kernel.

# Notes
This function needs to be implemented for each specific backend. Calling this function
without a concrete implementation will raise an error.
"""
function init_kernel(backend::AbstractBackend, n_cells::Ti, n_basefuncs::Ti, kernel::Function, args::Tuple) where {Ti <: Integer}
    throw(ErrorException("A concrete implementation of init_gpu_kernel is required"))
end

"""
    launch!(kernel::AbstractGPUKernel)

Launches a GPU kernel using the specified backend. This interface provides a generic
mechanism for running GPU-accelerated computations across different GPU backends.

# Arguments
- `::AbstractGPUKernel`: The GPU kernel to be launched.

# Notes
This function must be implemented for specific GPU kernels. If not implemented,
an error will be thrown.
"""
function launch!(::AbstractKernel)
    throw(ErrorException("A concrete implementation of launch! is required"))
end


### GPU Backend ###
struct BackendCUDA <: AbstractBackend end
struct BackendCPU <: AbstractBackend end
