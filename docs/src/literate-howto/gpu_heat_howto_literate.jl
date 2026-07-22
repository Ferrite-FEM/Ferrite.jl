# !!! warning "Experimental GPU Support"
#     GPU support is considered experimental at this point and the interfaces shown
#     in this how-to might break between minor releases.

#=
# Heat equation on GPU
In this how-to, we demonstrate the experimental API developed to use
graphics processing units (GPUs) with Ferrite.
The how-to is split into 3 parts, first we demonstrate a standard assembly
with a portable implementation using `KernelAbstractions.jl`. Next, we show
how a specific kernel for `CUDA` can be written, which can be beneficial when special CUDA features are used in the element assembly function.
While it is possible to assemble into a global sparse matrix with GPUs, another common strategy
is to calculate and store each element matrix separately and then use this to build an efficient
matrix-vector product, how to modify the code to assemble into such a structure is demonstrated
at the end.

## Global matrix assembly with `KernelAbstractions.jl`
We start by using the required packages,
=#
using Ferrite
using CUDA
import CUDA: CUDA.CUSPARSE.CuSparseMatrixCSC
import Adapt: adapt
import KernelAbstractions: @kernel, @index
import KernelAbstractions as KA
using SparseArrays

# and define some constants to be used in the following for convenience.
# Note that for a real problem we want to increase these numbers suitably.
const NUM_THREADS = 8
const NUM_TASKS_PER_THREAD = 2

function compute_threads_and_blocks(n)
    ## Let's assign, arbitrarily, two element assembly tasks per GPU thread.
    tasks_per_thread = min(NUM_TASKS_PER_THREAD, n)
    ## To do so, let us first compute how many element groups we have to assemble.
    n_effective = cld(n, tasks_per_thread)
    ## This potentially limits the number of usable threads, e.g. when a color just has a small
    ## number of elements.
    threads = min(NUM_THREADS, n_effective)
    ## Furthermore, for CPU computing we typically group the tasks into blocks of worker threads.
    blocks = cld(n, tasks_per_thread * threads)

    return threads, blocks
end

# In this how-to we want to use an existing element routine on the GPU with Ferrite,
# and we use the `assemble_element!` from the [heat equation tutorial](@ref tutorial-heat-equation).
# To be compatible with the GPU, the element routine must be allocation free
# (this requires type stable code).
function assemble_element!(Ke::AbstractMatrix, fe::AbstractVector, cv::CellValues)
    n_basefuncs = getnbasefunctions(cv)
    for q_point in 1:getnquadpoints(cv)
        dΩ = getdetJdV(cv, q_point)
        for i in 1:n_basefuncs
            ∇δuᵢ = shape_gradient(cv, q_point, i)
            δuᵢ = shape_value(cv, q_point, i)
            fe[i] += δuᵢ * dΩ
            for j in 1:n_basefuncs
                ∇δuⱼ = shape_gradient(cv, q_point, j)
                Ke[i, j] += (∇δuᵢ ⋅ ∇δuⱼ) * dΩ
            end
        end
    end
    return Ke, fe
end

# We further define a simple cell assembly wrapping to simplify
# the matrix-free demonstration later.
function assemble_cell!(Ke, fe, cell, cv, assembler)
    fill!(Ke, 0)
    fill!(fe, 0)
    reinit!(cv, nothing, cell.coords)
    assemble_element!(Ke, fe, cv)
    assemble!(assembler, celldofs(cell), Ke, fe)
    return nothing
end

# Now to the actual assembly kernel using KernelAbstractions.jl.
# We use a grid-stride loop, which has several benefits in terms of performance and debuggability.
# For more details please consult [this blog post](https://developer.nvidia.com/blog/cuda-pro-tip-write-flexible-kernels-grid-stride-loops/) .
@kernel function ka_assembly_kernel(assemblers, @Const(color), cc, cv, Kes, fes)
    ## This is the classical grid-stride-loop
    task_index = @index(Global, Linear)
    stride = prod(KA.@ndrange())

    ## Get the local evaluation buffers for the GPU worker.
    assembler = assemblers[task_index]
    cv_i = cv[task_index]
    cc_i = cc[task_index]
    Ke = view(Kes, task_index, :, :) # Note row-major indexing, this
    fe = view(fes, task_index, :)    # is further motivated below.

    for i in task_index:stride:length(color)
        ## Work item index
        cellid = color[i]

        ## Query work item cell cache
        reinit!(cc_i, cellid)

        ## Actual assembly routine.
        assemble_cell!(Ke, fe, cc_i, cv_i, assembler)
    end
end
function assemble_global_ka!(backend, cellvalues::Ferrite.SoAContainer, K, f, cc, colors::Vector, Ke, fe, n_workers; fillzero = true)
    assemblers = Ferrite.distribute_to_tasks(backend, start_assemble(K, f; fillzero), n_workers)
    for color in colors
        ## We divide the work into blocks and fire up the kernel.
        n = length(color)
        threads, blocks = compute_threads_and_blocks(n)
        ## Now, we can build and execute the Kernel.
        ka_kernel = ka_assembly_kernel(backend, threads)
        ka_kernel(assemblers, color, cc, cellvalues, Ke, fe, ndrange = threads * blocks)
        ## Since the kernel launches asynchronously we need to add a synchronization
        ## point before proceeding here. Otherwise we will start assembling the next color,
        ## while there are still threads working on the current color, therefore potentially
        ## causing race conditions.
        KA.synchronize(backend)
    end
    return nothing
end

# Now we first setup the problem almost as usual on the host (CPU).
# The only major difference here is that we instantiate everything
# using Float32 and Int32 whenever it makes sense to lower memory
# pressure on the GPU, and because Float32 is on most GPUs quite
# a bit faster than using Float64 -- outside of high-end server GPUs.
# Please note that GPU kernels have a launch overhead. Therefore out problem
# must be sufficiently large to see any benefits of utilizing the GPU.
# The small number of elements here is just for demonstration purposes.
num_elements = 20
# We generate a Float32 coordinate grid by passing in Float32 corner coordinates.
grid = generate_grid(Hexahedron, (num_elements, num_elements, num_elements), Vec{3}((-1.0f0, -1.0f0, -1.0f0)), Vec{3}((1.0f0, 1.0f0, 1.0f0)))
ip = Lagrange{RefHexahedron, 1}()
qr = QuadratureRule{RefHexahedron}(Float32, 2)
cv = CellValues(Float32, qr, ip)
dh = DofHandler(grid)
add!(dh, :u, ip)
close!(dh)

# If we assemble into a matrix, then we still need to color the grid as usual.
# See also the threading how-to. Note that we still leave some of the integers
# 64 bit to still enable the indexing of large problems.
colors = create_coloring(grid)

# Now to the GPU side. Here we use Adapt.jl to generate GPU counterparts of
# all relevant objects.
backend = CUDABackend()
colors_gpu = [adapt(backend, c) for c in colors]
dh_gpu = adapt(backend, dh)
K_gpu = allocate_matrix(CuSparseMatrixCSC{Float32, Int32}, dh)
f_gpu = KA.zeros(backend, Float32, (ndofs(dh),))

# Furthermore, the individual GPU workers need local buffers.
# Ferrite comes with a helper, `Ferrite.distribute_to_tasks`, which transforms common buffers
# into a suitable GPU format. Since we parallelize over the colors, we need to allocate
# buffers large enough. Since we use a grid-stride loop.
max_color_size = maximum(length.(colors))
n_workers = prod(compute_threads_and_blocks(max_color_size)) # Remember, we have `threads × blocks` workers.
cv_gpu = Ferrite.distribute_to_tasks(backend, cv, n_workers)
cc = CellCache(dh_gpu)
cc_gpu = Ferrite.distribute_to_tasks(backend, cc, n_workers)
# We also need a local buffer for the element vectors and matrices,
# and these are created as a global array which we use views into in the assembly kernel.
# Since GPU thread groups favor coalesced memory access we allocate the buffers such
# that we can access the data using row-major indexing. See, e.g.
# [this blog](https://developer.nvidia.com/blog/unlock-gpu-performance-global-memory-access-in-cuda/)
# as starting point for further information.
Kes = KA.zeros(backend, Float32, n_workers, getnbasefunctions(cv), getnbasefunctions(cv))
fes = KA.zeros(backend, Float32, n_workers, getnbasefunctions(cv))

# Now everything is set to launch the assembly via KernelAbstractions.
assemble_global_ka!(backend, cv_gpu, K_gpu, f_gpu, cc_gpu, colors_gpu, Kes, fes, n_workers)

# Finally, we can apply the Dirichlet constraints and solve our linear system.
ch = ConstraintHandler(Float32, Int32, dh)
∂Ω = union(
    getfacetset(grid, "left"), getfacetset(grid, "right"),
    getfacetset(grid, "top"), getfacetset(grid, "bottom")
)
add!(ch, Dirichlet(:u, ∂Ω, (x, t) -> 1.0))
close!(ch)

ch_gpu = adapt(backend, ch)
apply!(K_gpu, f_gpu, ch_gpu)
# To not complicate the description further, we simply solve the system on the CPU with a direct solver.
u_ka = SparseMatrixCSC(K_gpu) \ Vector(f_gpu)


#=
## Global matrix assembly with `CUDA.jl`
Only minor differences from the portable `KernelAbstractions.jl` version above are required
for a specific CUDA-kernel. While this section does not use any CUDA specific features like the tensor cores
it shows how to perform the assembly using CUDA only.
=#

function cuda_assembly_kernel(assemblers, color, cc::Ferrite.SoAContainer, cv::Ferrite.SoAContainer, Kes::AbstractArray, fes::AbstractMatrix)
    task_index = (blockIdx().x - Int32(1)) * blockDim().x + threadIdx().x
    stride = gridDim().x * blockDim().x
    ## The remaining code remains the same, as we do not show any CUDA specific features here.
    assembler = assemblers[task_index]
    cv_i = cv[task_index]
    cc_i = cc[task_index]
    Ke = view(Kes, task_index, :, :)
    fe = view(fes, task_index, :)
    for i in task_index:stride:length(color)
        cellid = color[i]
        reinit!(cc_i, cellid)
        assemble_cell!(Ke, fe, cc_i, cv_i, assembler)
    end
    return nothing
end

function assemble_global_cuda!(cv::Ferrite.SoAContainer, K, f, cc, colors::Vector, Kes, fes, n_workers; fillzero = true)
    assemblers = Ferrite.distribute_to_tasks(backend, start_assemble(K, f; fillzero), n_workers)
    for color in colors
        n = length(color)
        threads, blocks = compute_threads_and_blocks(n)
        @cuda threads = threads blocks = blocks cuda_assembly_kernel(assemblers, color, cc, cv, Kes, fes)
        CUDA.synchronize()
    end
    return nothing
end

# And now we can assemble the same way as for the `KernelAbstractions.jl` version
assemble_global_cuda!(cv_gpu, K_gpu, f_gpu, cc_gpu, colors_gpu, Kes, fes, n_workers)
apply!(K_gpu, f_gpu, ch_gpu)
u_cuda = SparseMatrixCSC(K_gpu) \ Vector(f_gpu)

#=
## Matrix-free assembly with `CUDA.jl`
Using CSR and CSC matrix formats are known to be bad on the GPU (see e.g. [Settgast2023:plm; Figure 1/2](@cite)). The reason for the bad performance is that GPUs are typically used for large problems requiring an iterative solver of some form. These iterative solvers typically require sparse-matrix vector (or transpose) products as a key component. CSR and CSC have extremely low arithmetic intensity, so we cannot fully utilize the potential of the GPU, which really shines on high arithmetic intensity tasks. In finite element problems a very simple technique is called ,,Element Assembly'' (see e.g. the [MFEM docs](https://mfem.org/howto/assembly_levels/)) where we simply assemble the element matrices once, such that the sparse matrix-vector product becomes a sequence of many small dense matrix-vector products, i.e. the local products of the element matrix and element vectors. There are techniques with even higher arithmetic intensity, but they typically require modifications of the element routines. Therefore, we present the element assembly technique here as a starting point for users which want to boost their simulations further. Finally we want to highlight that a downside of matrix-free techniques is that we have the issue that most known preconditioners cannot be applied anymore. Finding efficient preconditioners for matrix-free techniques is a very active research area (see, e.g. [Schussnig2025:mhf](@cite) or [Wichrowski2026:tmp](@cite)).
=#
function assemble_cell!(Ke, fe, cell, cv)
    fill!(Ke, 0)
    fill!(fe, 0)
    reinit!(cv, nothing, cell.coords)
    assemble_element!(Ke, fe, cv)
    return nothing
end

# This is the element-assembly kernel to setup the element matrices and vectors only.
function cuda_assembly_kernel(color, cc, cv, Kes, fes)
    task_index = (blockIdx().x - Int32(1)) * blockDim().x + threadIdx().x
    stride = gridDim().x * blockDim().x
    cv_i = cv[task_index]
    cc_i = cc[task_index]
    for i in task_index:stride:length(color)
        cellid = color[i]
        reinit!(cc_i, cellid)
        Ke = view(Kes, cellid, :, :)
        fe = view(fes, cellid, :)
        assemble_cell!(Ke, fe, cc_i, cv_i)
    end
    return nothing
end
function assemble_global_cuda!(cv::Ferrite.SoAContainer, cc, colors::Vector, Ke, fe)
    for color in colors
        n = length(color)
        tasks_per_thread = min(NUM_TASKS_PER_THREAD, n)
        threads = min(NUM_THREADS, cld(n, tasks_per_thread))
        blocks = cld(n, tasks_per_thread * threads)
        @cuda threads = threads blocks = blocks cuda_assembly_kernel(color, cc, cv, Ke, fe)
        CUDA.synchronize()
    end
    return nothing
end

# In this case, we create one element stiffness and one element load vector for each cell in the grid
Kes = KA.zeros(backend, Float32, getncells(grid), getnbasefunctions(cv), getnbasefunctions(cv))
fes = KA.zeros(backend, Float32, getncells(grid), getnbasefunctions(cv))

# And assemble without using the global stiffness, `K_gpu`, and load vector, `f_gpu`.
assemble_global_cuda!(cv_gpu, cc_gpu, colors_gpu, Kes, fes)

## References

# ```@bibliography
# Pages = ["gpu_heat_howto_literate.md"]
# Canonical = false
# ```
