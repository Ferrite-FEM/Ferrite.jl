# Runs the GPU how-to (CUDA only) and checks its results against the CPU references from
# `ka_common.jl`. The how-to redefines `assemble_element!`, `ka_assembly_kernel` and
# `assemble_global_ka!` with identical bodies.
include(joinpath(@__DIR__, "..", "..", "docs", "src", "literate-howto", "gpu_assembly.jl"))

@testset "How-To correctness" begin
    @test u_ka ≈ u_cuda

    K = allocate_matrix(SparseMatrixCSC{Float32, Int32}, dh)
    f = zeros(Float32, ndofs(dh))
    assemble_global!(cv, K, f, dh)
    apply!(K, f, ch)
    u_cpu = solve_cpu(K, f)
    # NOTE this might fail because the meandiag differs due to cancellation. However,
    # the solutions are usually still very close.
    @test SparseMatrixCSC(K_gpu) ≈ K
    @test Vector(f_gpu) ≈ f
    @test u_cpu ≈ u_ka
end
