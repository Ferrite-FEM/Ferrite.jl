#=
This file defines the interface between the GPU backend (extension) and the Ferrite package.
It provides abstract types, function signatures, and concrete types for managing GPU kernels
and backends, serving as a foundation for GPU-accelerated computations.
=#

### Abstract Types ###
abstract type AbstractGPUKernel end
abstract type AbstractGPUBackend end


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
function init_gpu_kernel(backend::AbstractGPUBackend, n_cells::Ti, n_basefuncs::Ti, kernel::Function, args::Tuple) where {Ti <: Integer}
    throw(ErrorException("A concrete implementation of init_gpu_kernel is required"))
end

"""
    launch!(kernel::AbstractGPUKernel)

Launches a GPU kernel using the specified backend. This interface provides a generic
mechanism for running GPU-accelerated computations across different GPU backends.

# Arguments
- `kernel::AbstractGPUKernel`: The GPU kernel to be launched.

# Notes
This function must be implemented for specific GPU kernels. If not implemented,
an error will be thrown.
"""
function launch!(kernel::AbstractGPUKernel)
    throw(ErrorException("A concrete implementation of launch! is required"))
end


### Concrete Types ###

"""
    GPUKernel{Ti}(n_cells::Ti, n_basefuncs::Ti, kernel::Function, args::Tuple, backend::Type{<:AbstractGPUBackend})

Represents a high-level interface to a GPU backend for configuring and launching GPU kernels.
It stores the necessary parameters for kernel execution, such as the number of cells,
number of base functions, the kernel function, and any additional arguments.

# Fields
- `n_cells::Ti`: Number of cells to be processed.
- `n_basefuncs::Ti`: Number of base functions for each cell.
- `kernel::Function`: The GPU kernel function.
- `args::Tuple`: Additional arguments to be passed to the kernel function.
- `backend::Type{<:AbstractGPUBackend}`: The GPU backend used for execution.

# Type Parameters
- `Ti`: An integer type representing the number type used for `n_cells` and `n_basefuncs`.
"""
struct GPUKernel{Ti} <: AbstractGPUKernel
    n_cells::Ti               # Number of cells
    n_basefuncs::Ti           # Number of base functions
    kernel::Function          # Kernel function to execute
    args::Tuple               # Arguments for the kernel function
    backend::Type{<:AbstractGPUBackend} # GPU backend
end

"""
    getbackend(kernel::GPUKernel) -> Type{<:AbstractGPUBackend}

Returns the backend associated with the given `GPUKernel`.

# Arguments
- `kernel::GPUKernel`: The GPU kernel from which to retrieve the backend.

# Returns
The backend type associated with the kernel.
"""
getbackend(kernel::GPUKernel) = kernel.backend


### GPU Backend ###

"""
    BackendCUDA <: AbstractGPUBackend

Represents the CUDA backend for GPU acceleration. This type serves as a concrete
implementation of `AbstractGPUBackend` for executing GPU computations using CUDA.
"""
struct BackendCUDA <: AbstractGPUBackend end
