using Ferrite, CUDA
import CUDA: CUDA.CUSPARSE.CuSparseMatrixCSC, @allowscalar
import Ferrite: get_grid, AbstractGrid, AbstractDofHandler, get_coordinate_eltype
import Adapt: Adapt, adapt, adapt_structure
import KernelAbstractions: get_backend, @kernel, @index
import KernelAbstractions as KA
using SparseArrays

import Ferrite: get_substruct, as_structure_of_arrays

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

@kernel function assembly_kernel(assembler, color, cell_cache, cv, Ke)
    i = @index(Global)
    # i > length(color) && return

    cellid = color[i]
    cv_i = get_substruct(i, cv)
    cc_i = get_substruct(i, cell_cache, cellid)

    reinit!(cc_i, cellid)

    assemble_cell!(view(Ke, i, :, :), cc_i, cv_i, assembler)

    # return nothing
end

function assemble_global!(backend, cv::CellValues, K, cell_cache, colors::Vector, Ke)
    assembler = start_assemble(K)
    for color in colors
        # @cuda threads = length(color) assembly_kernel!(assembler, color, cell_cache, cv, Ke)
        assembly_kernel(backend, length(color))(assembler, color, cell_cache, cv, Ke, ndrange = length(color))
        KA.synchronize(backend)
    end
    return nothing
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
grid = generate_grid(Hexahedron, (10, 10, 10), Vec{3}((-1.0f0, -1.0f0, -1.0f0)), Vec{3}((1.0f0, 1.0f0, 1.0f0)))
ip = Lagrange{RefHexahedron, 1}()
qr = QuadratureRule{RefHexahedron}(Float32, 2)
cv = CellValues(qr, ip)
dh = DofHandler(grid)
add!(dh, :u, ip)
close!(dh)
K = allocate_matrix(SparseMatrixCSC{Float32, Int32}, dh)
assemble_global!(cv, K, dh)

# KA CUDA
colors = create_coloring(grid)
backend = CUDABackend()
colors_gpu = [adapt(backend, c) for c in colors]
n_workers = maximum(length.(colors_gpu))
dh_gpu = adapt(backend, dh)
K_gpu = allocate_matrix(CuSparseMatrixCSC{Float32, Int32}, dh)
cv_gpu = as_structure_of_arrays(backend, n_workers, cv)
cell_cache = as_structure_of_arrays(backend, n_workers, CellCache, dh_gpu)
Ke = KA.zeros(backend, Float32, n_workers, getnbasefunctions(cv), getnbasefunctions(cv))
assemble_global!(backend, cv_gpu, K_gpu, cell_cache, colors_gpu, Ke)

using Test
@test CuSparseMatrixCSC(K) ≈ K_gpu

# Dirichlet BCs — homogeneous (left + right)
ch = ConstraintHandler(Float32, Int32, dh)
add!(ch, Dirichlet(:u, union(getfacetset(grid, "left"), getfacetset(grid, "right")), (x, t) -> 0.0))
close!(ch)

ch_gpu = adapt(backend, ch)
f = zeros(Float32, ndofs(dh))
f_gpu = KA.zeros(backend, Float32, ndofs(dh))

apply!(K, f, ch)
apply!(K_gpu, f_gpu, ch_gpu)

@test CuSparseMatrixCSC(K) ≈ K_gpu
@test CuVector(f) ≈ f_gpu

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

@test CuSparseMatrixCSC(K) ≈ K_gpu
@test CuVector(f2) ≈ f_gpu2

# KA CPU
begin
    backend = KA.CPU()
    colors_cpu = [adapt(backend, c) for c in colors]
    n_workers = maximum(length.(colors_cpu))
    dh_cpu = adapt(backend, dh)
    K_cpu = allocate_matrix(SparseMatrixCSC{Float32, Int32}, dh)
    cv_cpu = as_structure_of_arrays(backend, n_workers, cv)
    cell_cache = as_structure_of_arrays(backend, n_workers, CellCache, dh_cpu)
    Ke = KA.zeros(backend, Float32, n_workers, getnbasefunctions(cv), getnbasefunctions(cv))
    assemble_global!(backend, cv_cpu, K_cpu, cell_cache, colors_cpu, Ke)
end
