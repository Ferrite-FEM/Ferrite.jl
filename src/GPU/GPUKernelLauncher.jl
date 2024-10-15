#= This file represents the interface between the GPU backend (extension) and the Ferrite package. =#

### Abstract types ###
abstract type AbstractGPUKernel end
abstract type AbstractGPUBackend end


function init_gpu_kernel(backend::AbstractGPUBackend, n_cells::Ti, n_basefuncs::Ti, kernel::Function, args::Tuple) where {Ti<: Integer}
    throw(ErrorException("A concrete implementation of init_gpu_kernel is required"))
end


"""
    launch!(::AbstractGPUKernelLauncher)
Interface for launching a kernel on the GPU backend.
"""
function launch!(::AbstractGPUKernel)
    throw(ErrorException("A concrete implementation of launch! is required"))
end


### Concrete types ###

### Kernels ###
"""
    CUDAKernel{Ti}(n_cells::Int, n_basefuncs::Int)
`CUDAKernel` represents a high-level interface to the CUDA backend for launching and configuring kernels.

# Fields
- `n_cells::Ti`: number of cells
- `n_basefuncs::Ti`: number of base functions
- `kernel::Function`: kernel function
- `args::Tuple`: arguments to the kernel
"""
struct GPUKernel{Ti} <: AbstractGPUKernel
    n_cells::Ti # number of cells
    n_basefuncs::Ti # number of base functions
    kernel::Function # kernel function
    args::Tuple # arguments to the kernel
    backend::Type{<:AbstractGPUBackend} # backend
end

getbackend(kernel::GPUKernel) = kernel.backend

### Backend ###
struct BackendCUDA <: AbstractGPUBackend end
