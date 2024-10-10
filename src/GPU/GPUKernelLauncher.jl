#= This file represents the interface between the GPU backend (extension) and the Ferrite package. =#

### Abstract types ###
abstract type AbstractGPUKernelLauncher end

"""
    launch!(::AbstractGPUKernelLauncher)
Interface for launching a kernel on the GPU backend.
"""
function launch_kernel!(::AbstractGPUKernelLauncher)
    throw(ErrorException("A concrete implementation of launch_kernel! is required"))
end


### Concrete types ###

"""
    CUDAKernelLauncher{Ti}(n_cells::Int, n_basefuncs::Int)
`CUDAKernelLauncher` represents a high-level interface to the CUDA backend for launching and configuring kernels.

# Fields
- `n_cells::Ti`: number of cells
- `n_basefuncs::Ti`: number of base functions
- `kernel::Function`: kernel function
- `args::Tuple`: arguments to the kernel
"""
struct CUDAKernelLauncher{Ti} <: AbstractGPUKernelLauncher
    n_cells::Ti # number of cells
    n_basefuncs::Ti # number of base functions
    kernel::Function # kernel function
    args::Tuple # arguments to the kernel
end
