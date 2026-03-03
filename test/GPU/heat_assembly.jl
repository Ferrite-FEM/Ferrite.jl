using Ferrite, CUDA
import CUDA: CUDA.CUSPARSE.CuSparseMatrixCSC, @allowscalar
import Ferrite: get_grid, AbstractGrid, AbstractDofHandler, get_coordinate_eltype
import Adapt: Adapt, adapt, adapt_structure
import KernelAbstractions: get_backend, @kernel, @index
import KernelAbstractions as KA
using SparseArrays

import Ferrite: get_substruct, as_structure_of_arrays, materialize

function assemble_element!(Ke::AbstractMatrix, cv::CellValues)
    n_basefuncs = getnbasefunctions(cv)
    fill!(Ke, 0)
    for q_point in 1:getnquadpoints(cv)
        dΩ = getdetJdV(cv, q_point)
        for i in 1:n_basefuncs
            δu = shape_gradient(cv, q_point, i)
            for j in 1:n_basefuncs
                Ke[i, j] += (δu ⋅ shape_gradient(cv, q_point, j)) * dΩ
            end
        end
    end
    return Ke
end

function assemble_cell!(Ke, cell, cv, assembler)
    reinit!(cv, nothing, cell.coords)
    assemble_element!(Ke, cv)
    assemble!(assembler, celldofs(cell), Ke)
    return nothing
end

@kernel function assembly_kernel(assembler, @Const(color), cell_cache, cv, Kes)
    stride = KA.@groupsize()[1]
    li = @index(Local, Linear)

    # Query the local evaluation buffer of the thread
    cv_i = get_substruct(li, cv)

    for i in li:stride:length(color)
        KA.@print("i=%d li=%d \n\n", i, li)
        # Work item index
        cellid = color[i]

        # Query work item cell cache
        cc_i = get_substruct(li, cell_cache, cellid)

        # Fill buffer
        reinit!(cc_i, cellid)

        # Query assembly buffer 
        # TODO should go into shared memory and query via li
        Ke = view(Kes, i, :, :)

        # Actual assembly routine
        assemble_cell!(Ke, cc_i, cv_i, assembler)
    end

    # return nothing
end


function cuda_assembly_kernel(assembler, color, cc_prototype, cv_prototype, Kes, touched)
    index = (blockIdx().x - Int32(1)) * blockDim().x + threadIdx().x
    stride = gridDim().x * blockDim().x
    li = threadIdx().x

    # Setup the potentially shared memory of the block
    cv   = materialize(cv_prototype)
    cc   = materialize(cc_prototype)

    # # Query the local evaluation buffer of the thread
    # cv_i = get_substruct(li, cv)

    # @cushow index, stride, li, length(color)
    for i in index:stride:length(color)
        # Work item index
        cellid = color[i]

        # FIXME debug
        CUDA.@atomic touched[cellid] += 1

        # FIXME no shared memory right now :(
        li = i

        # Query the local evaluation buffer of the thread
        cv_i = get_substruct(li, cv)
    
        # Query work item cell cache
        cc_i = get_substruct(li, cc, cellid)

        # Fill buffer
        reinit!(cc_i, cellid)

        # Query assembly buffer 
        Ke = view(Kes, li, :, :)

        # # Actual assembly routine
        assemble_cell!(Ke, cc_i, cv_i, assembler)
    end

    return nothing
end

function assemble_global!(backend, cv::CellValues, K, cell_cache, colors::Vector, Ke)
    touched = CUDA.zeros(Int, sum(length.(colors)))
    assembler = start_assemble(K)
    for color in colors
        n = length(color)
        threads = min(32, n)
        blocks  = cld(length(color), threads)
        @cuda threads = threads blocks = blocks cuda_assembly_kernel(assembler, color, cell_cache, cv, Ke, touched)
        # ka_kernel = assembly_kernel(backend, threads)
        # ka_kernel(assembler, color, cell_cache, cv, Ke, ndrange=(threads,length(color)))
        KA.synchronize(backend)
    end
    return touched
end

function assemble_global!(cv::CellValues, K::SparseMatrixCSC, dh::DofHandler)
    n_basefuncs = getnbasefunctions(cv)
    Ke = zeros(Float32, n_basefuncs, n_basefuncs)
    assembler = start_assemble(K)
    for cell in CellIterator(dh)
        assemble_cell!(Ke, cell, cv, assembler)
    end
    return nothing
end

### --- Example / Tests ---

# Regular CPU
num_elements = 50
grid = generate_grid(Hexahedron, (num_elements, num_elements, num_elements), Vec{3}((-1.0f0, -1.0f0, -1.0f0)), Vec{3}((1.0f0, 1.0f0, 1.0f0)))
ip = Lagrange{RefHexahedron, 1}()
qr = QuadratureRule{RefHexahedron}(Float32, 2)
cv = CellValues(qr, ip)
dh = DofHandler(grid)
add!(dh, :u, ip)
close!(dh)
K = allocate_matrix(SparseMatrixCSC{Float32, Int32}, dh)
assemble_global!(cv, K, dh)

colors = create_coloring(grid)

# KA CPU
# begin
#     backend = KA.CPU()
#     colors_cpu = [adapt(backend, c) for c in colors]
#     n_workers = maximum(length.(colors_cpu))
#     dh_cpu = adapt(backend, dh)
#     K_cpu = allocate_matrix(SparseMatrixCSC{Float32, Int32}, dh)
#     cv_cpu = as_structure_of_arrays(backend, n_workers, cv)
#     cell_cache = as_structure_of_arrays(backend, n_workers, CellCache, dh_cpu)
#     Ke = KA.zeros(backend, Float32, n_workers, getnbasefunctions(cv), getnbasefunctions(cv))
#     assemble_global!(backend, cv_cpu, K_cpu, cell_cache, colors_cpu, Ke)
#     K_cpu ≈ K
# end

# KA CUDA
backend = CUDABackend()
colors_gpu = [adapt(backend, c) for c in colors]
n_workers = maximum(length.(colors))
dh_gpu = adapt(backend, dh)
K_gpu = allocate_matrix(CuSparseMatrixCSC{Float32, Int32}, dh)

# Variant A - Uses shared memory. Crashed with "too much shared memory used" error
# cv_gpu = adapt(backend, cv)
# cc_gpu = CellCache(backend, dh_gpu)

# Variant B - Uses global memory. Functional.
cc_gpu = as_structure_of_arrays(backend, n_workers, CellCache, dh_gpu)
cv_gpu = as_structure_of_arrays(backend, n_workers, cv)

Ke = KA.zeros(backend, Float32, n_workers, getnbasefunctions(cv), getnbasefunctions(cv))
visited = assemble_global!(backend, cv_gpu, K_gpu, cc_gpu, colors_gpu, Ke)

using Test
@test K ≈ SparseMatrixCSC(K_gpu)

# Dirichlet BCs — homogeneous (left + right)
ch = ConstraintHandler(Float32, Int32, dh)
add!(ch, Dirichlet(:u, union(getfacetset(grid, "left"), getfacetset(grid, "right")), (x, t) -> 0.0))
close!(ch)

ch_gpu = adapt(backend, ch)
f = zeros(Float32, ndofs(dh))
f_gpu = KA.zeros(backend, Float32, ndofs(dh))

Kcopy = copy(K)
K_gpucopy = copy(K_gpu)
apply!(Kcopy, f, ch)
apply!(K_gpucopy, f_gpu, ch_gpu)
@test Kcopy ≉ K
@test K_gpucopy ≉ K_gpu

x = rand(ndofs(dh))
@test Kcopy ≈ SparseMatrixCSC(K_gpucopy)
@test f ≈ Vector(f_gpu)

# Dirichlet BCs — inhomogeneous (left + right + top + bottom)
ch2 = ConstraintHandler(Float32, Int32, dh)
add!(
    ch2, Dirichlet(
        :u, union(
            getfacetset(grid, "left"), getfacetset(grid, "right"),
            getfacetset(grid, "top"), getfacetset(grid, "bottom")
        ), (x, t) -> 1.0
    )
)
close!(ch2)

ch_gpu2 = adapt(backend, ch2)
f2 = zeros(Float32, ndofs(dh))
f_gpu2 = KA.zeros(backend, Float32, ndofs(dh))

apply!(K, f2, ch2)
apply!(K_gpu, f_gpu2, ch_gpu2)

# NOTE this does not hold because the meandiag differs due to cancellation. However, the solutions should still be close
# @test K ≈ SparseMatrixCSC(K_gpu)
# @test f2 ≈ Vector(f_gpu2)

ucpu = K \ f2
ugpu = SparseMatrixCSC(K_gpu) \ Vector(f_gpu2)
@test ucpu ≈ ugpu
