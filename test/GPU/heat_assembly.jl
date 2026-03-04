
include("../../docs/src/howto/gpu_heat_howto_literate.jl")

# ----------------------------- Tests --------------------------

using Test
K = allocate_matrix(SparseMatrixCSC{Float32, Int32}, dh)
f = zeros(Float32, ndofs(dh))
assemble_global!(cv, K, f, dh)
apply!(K, f, ch)
u_cpu = K \ f
# NOTE this might fail because the meandiag differs due to cancellation. However,
# the solutions are usually still very close.
@test u_cpu ≈ u_gpu

# Test KA
@testset "KernelAbstractions paths for $backend" for backend in [KA.CPU(), CUDABackend()]
    colors_device = [adapt(backend, c) for c in colors]
    n_workers = maximum(length.(colors_device))
    dh_device = adapt(backend, dh)
    K_device = if backend isa KA.CPU
        allocate_matrix(SparseMatrixCSC{Float32, Int32}, dh)
    else
        allocate_matrix(CuSparseMatrixCSC{Float32, Int32}, dh)
    end
    f_device = KA.zeros(backend, Float32, ndofs(dh))
    cv_device = CellValuesContainer(backend, n_workers, cv)
    cell_cache = CellCacheContainer(backend, n_workers, dh_device)
    Kes_device = KA.zeros(backend, Float32, getncells(grid), getnbasefunctions(cv), getnbasefunctions(cv))
    fes_device = KA.zeros(backend, Float32, getncells(grid), getnbasefunctions(cv))
    # Assembly here does not work because we are missing a SOA transformation of the assembler.
    assemble_global_ka!(backend, cv_device, nothing, nothing, cell_cache, colors_device, Kes_device, fes_device)
    @test Array(Kes_device) ≈ Array(Kes)
    @test Array(fes_device) ≈ Array(fes)
end
