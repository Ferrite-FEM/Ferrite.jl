using Ferrite, CUDA
import CUDA: CUDA.CUSPARSE.CuSparseMatrixCSC
import Adapt: adapt
import KernelAbstractions: @kernel, @index
import KernelAbstractions as KA
using SparseArrays

import Ferrite: CellValuesContainer, CellCacheContainer

# We start with some to be used in the following for simple convenience.
const NUM_THREADS = 64
const NUM_TASKS_PER_THREAD = 2

# In this how-to we want to use an existing assembly routine on the GPU with Ferrite.
# We implicitly assume that nothing dynamic happens inside the routine, i.e. the routine
# is type stable, does not allocate and also does not have any dynamic dispatches.
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
    return nothing
end

# We also have a simple cell assembly wrapping the element in two variants.
# In the first variant we assemble using an assembler. In the second variant we
# only fill Ke, e.g. as part of element-assembly techniques.
function assemble_cell!(Ke, fe, cell, cv, assembler)
    reinit!(cv, nothing, cell.coords)
    fill!(Ke, 0)
    fill!(fe, 0)
    assemble_element!(Ke, fe, cv)
    assemble!(assembler, celldofs(cell), Ke, fe)
    return nothing
end
function assemble_cell!(Ke, fe, cell, cv, ::Nothing)
    reinit!(cv, nothing, cell.coords)
    fill!(Ke, 0)
    fill!(fe, 0)
    assemble_element!(Ke, fe, cv)
    return nothing
end

# Now to the actual assembly kernel. To ensure portability we show how to use KernelAbstractions.jl
# as the kernel language, although we will also show how to use CUDA directly below. In this kernel
# we use a grid-stride loop, which has several benefits in terms of performance and debuggability.
# For more details please consult https://developer.nvidia.com/blog/cuda-pro-tip-write-flexible-kernels-grid-stride-loops/ .
@kernel function ka_assembly_kernel(assembler, @Const(color), cc, cv, Kes, fes)
    ## This is the classical grid-stride-loop
    task_index = @index(Global, Linear)
    stride = prod(KA.@ndrange())
    for i in task_index:stride:length(color)
        ## Work item index
        cellid = color[i]

        ## Query the local evaluation buffer of the GPU worker.
        ## As explained later this is the secret sauce.
        cv_i = cv[task_index]

        ## Query work item cell cache. The call on the item initializes replaces the reinit! call.
        cc_i = cc[task_index](cellid)

        ## Query assembly buffer.
        Ke = view(Kes, i, :, :)
        fe = view(fes, i, :)

        ## Actual assembly routine.
        assemble_cell!(Ke, fe, cc_i, cv_i, assembler)
    end
end
function assemble_global_ka!(backend, cv::CellValuesContainer, K, f, cc, colors::Vector, Ke, fe)
    assembler = K === nothing ? nothing : start_assemble(K, f)
    for color in colors
        ## We divide the work into blocks and fire up the kernel.
        n = length(color)
        ## Let's assign, arbitrarily, two element assembly tasks per GPU thread.
        tasks_per_thread = min(NUM_TASKS_PER_THREAD, n)
        ## To do so, let us first compute how many element groups we have to assemble.
        n_effective = cld(n, tasks_per_thread)
        ## This potentially limits the number of usable threads, e.g. when a color just has a small
        ## number of elements.
        threads = min(NUM_THREADS, n_effective)
        ## Furthermore, for CPU computing we typically group the tasks into blocks of worker threads.
        blocks = cld(n, tasks_per_thread * threads)
        ## Now, we can build and execute the Kernel.
        ka_kernel = ka_assembly_kernel(backend, threads)
        ka_kernel(assembler, color, cc, cv, Ke, fe, ndrange = threads * blocks)
        ## Since the kernel launches asynchronously we need to add a synchronization
        ## point before proceeding here. Otherwise we will start assembling the next color,
        ## while there are still threads working on the current color, therefore potentially
        ## causing race conditions.
        KA.synchronize(backend)
    end
    return nothing
end

# And here now the CUDA variant. Please see above for details, as the kernels are almost the same.
function cuda_assembly_kernel(assembler, color, cc, cv, Kes, fes)
    task_index = (blockIdx().x - Int32(1)) * blockDim().x + threadIdx().x
    stride = gridDim().x * blockDim().x
    for i in task_index:stride:length(color)
        cellid = color[i]
        cv_i = cv[task_index]
        cc_i = cc[task_index](cellid)
        Ke = view(Kes, i, :, :)
        fe = view(fes, i, :)
        assemble_cell!(Ke, fe, cc_i, cv_i, assembler)
    end
    return nothing
end
function assemble_global_cuda!(cv::CellValuesContainer, K, f, cc, colors::Vector, Ke, fe)
    assembler = K === nothing ? nothing : start_assemble(K, f)
    for color in colors
        n = length(color)
        tasks_per_thread = min(NUM_TASKS_PER_THREAD, n)
        threads = min(NUM_THREADS, cld(n, tasks_per_thread))
        blocks = cld(n, tasks_per_thread * threads)
        @cuda threads = threads blocks = blocks cuda_assembly_kernel(assembler, color, cc, cv, Ke, fe)
        CUDA.synchronize()
    end
    return nothing
end

# Reference for internal testing                                                #hide
function assemble_global!(cv::CellValues, K::SparseMatrixCSC, f, dh::DofHandler) #hide
    n_basefuncs = getnbasefunctions(cv)                                         #hide
    Ke = zeros(Float32, n_basefuncs, n_basefuncs)                               #hide
    fe = zeros(Float32, n_basefuncs)                                            #hide
    assembler = start_assemble(K, f)                                            #hide
    for cell in CellIterator(dh)                                                #hide
        assemble_cell!(Ke, fe, cell, cv, assembler)                             #hide
    end                                                                         #hide
    return nothing                                                              #hide
end                                                                             #hide

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
# Ferrite comes with a little helper to transform common buffers
# into a suitable GPU format.
# n_workers = ceil(Int, length(grid.cells) / NUM_THREADS) # FIXME does not match the used 493
n_workers = getncells(grid)
cv_gpu = CellValuesContainer(backend, n_workers, cv)
cc_gpu = CellCacheContainer(backend, n_workers, dh_gpu)
# Technically we can also just get one Ke or fe per worker, but for demonstration
# purposes we allocate the full block here for element-assembly style matrix-free GPU
# usage.
Kes = KA.zeros(backend, Float32, getncells(grid), getnbasefunctions(cv), getnbasefunctions(cv))
fes = KA.zeros(backend, Float32, getncells(grid), getnbasefunctions(cv))

# Now everything is set to launch the assembly via KernelAbstractions.
# assemble_global_ka!(backend, cv_gpu, K_gpu, f_gpu, cc_gpu, colors_gpu, Kes, fes)
# Or alternatively the cuda variant.
assemble_global_cuda!(cv_gpu, K_gpu, f_gpu, cc_gpu, colors_gpu, Kes, fes)

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
u_gpu = SparseMatrixCSC(K_gpu) \ Vector(f_gpu)
