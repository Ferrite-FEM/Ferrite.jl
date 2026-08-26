# Backend-parametrized assembly tests. Requires `ka_common.jl` to be included and the
# globals `backend::KA.Backend`, `sparse_type(Tv, Ti)` and `sparse_type_csr(Tv, Ti)` to be
# defined.

@testset "KernelAbstractions element assembly ($backend)" begin
    grid, dh, cv, ch = setup_heat_problem(Float32)
    colors = create_coloring(grid)
    n_basefuncs = getnbasefunctions(cv)

    Kes_ref = zeros(Float32, getncells(grid), n_basefuncs, n_basefuncs)
    fes_ref = zeros(Float32, getncells(grid), n_basefuncs)
    assemble_elements!(cv, Kes_ref, fes_ref, dh)

    colors_device = [adapt(backend, c) for c in colors]
    n_workers = prod(compute_threads_and_blocks(maximum(length.(colors))))
    dh_device = adapt(backend, dh)
    cv_device = Ferrite.distribute_to_workers(backend, cv, n_workers)
    cc_device = Ferrite.distribute_to_workers(backend, CellCache(dh_device), n_workers)

    @test @inferred(dof_range(dh_device.subdofhandlers[1], 1)) == @inferred(dof_range(dh_device.subdofhandlers[1], :u))

    # One element matrix and vector per cell, no global matrix involved
    Kes_cells = KA.zeros(backend, Float32, getncells(grid), n_basefuncs, n_basefuncs)
    fes_cells = KA.zeros(backend, Float32, getncells(grid), n_basefuncs)
    assemble_elements_ka!(backend, cv_device, cc_device, colors_device, Kes_cells, fes_cells)
    @test Array(Kes_cells) ≈ Kes_ref
    @test Array(fes_cells) ≈ fes_ref
end

@testset "KernelAbstractions error paths ($backend)" begin
    grid, dh, cv, ch = setup_heat_problem(Float32, 2)
    @test_throws ArgumentError Ferrite.distribute_to_workers(backend, cv, 0)

    ch_affine = ConstraintHandler(Float32, Int32, dh)
    add!(ch_affine, AffineConstraint(1, [2 => 1.0f0], 0.0f0))
    close!(ch_affine)
    @test_throws AssertionError adapt(backend, ch_affine)
end

@testset "KernelAbstractions heat problem on simple grid ($backend, $(nameof(sparse_type(Float32, Int32))))" begin
    grid, dh, cv, ch = setup_heat_problem(Float32)
    colors = create_coloring(grid)
    n_basefuncs = getnbasefunctions(cv)

    # References
    K_ref = allocate_matrix(SparseMatrixCSC{Float32, Int32}, dh)
    f_ref = zeros(Float32, ndofs(dh))
    assemble_global!(cv, K_ref, f_ref, dh)
    K_unconstrained = copy(K_ref)
    f_unconstrained = copy(f_ref)
    apply!(K_ref, f_ref, ch)
    u_ref = solve_cpu(K_ref, f_ref)
    K_zero = copy(K_unconstrained)
    f_zero = copy(f_unconstrained)
    apply_zero!(K_zero, f_zero, ch)

    # Device setup
    colors_device = [adapt(backend, c) for c in colors]
    n_workers = prod(compute_threads_and_blocks(maximum(length.(colors))))
    dh_device = adapt(backend, dh)
    cv_device = Ferrite.distribute_to_workers(backend, cv, n_workers)
    cc_device = Ferrite.distribute_to_workers(backend, CellCache(dh_device), n_workers)
    K_device = allocate_device_matrix(backend, sparse_type(Float32, Int32), dh)
    f_device = KA.zeros(backend, Float32, ndofs(dh))
    Kes_device = KA.zeros(backend, Float32, n_workers, n_basefuncs, n_basefuncs)
    fes_device = KA.zeros(backend, Float32, n_workers, n_basefuncs)

    assemble_global_ka!(backend, cv_device, K_device, f_device, cc_device, colors_device, Kes_device, fes_device, n_workers)
    @test SparseMatrixCSC(K_device) ≈ K_unconstrained
    @test Vector(f_device) ≈ f_unconstrained

    ch_device = adapt(backend, ch)
    apply!(K_device, f_device, ch_device)
    @test SparseMatrixCSC(K_device) ≈ K_ref
    @test Vector(f_device) ≈ f_ref
    @test solve_cpu(K_device, f_device) ≈ u_ref

    assemble_global_ka!(backend, cv_device, K_device, f_device, cc_device, colors_device, Kes_device, fes_device, n_workers)
    apply!(K_device, f_device, ch_device, true)
    @test SparseMatrixCSC(K_device) ≈ K_zero
    @test Vector(f_device) ≈ f_zero

    # Atomic assembly without coloring: one launch over all cells
    cells_device = adapt(backend, collect(1:getncells(grid)))
    n_workers_all = prod(compute_threads_and_blocks(getncells(grid)))
    cv_all = Ferrite.distribute_to_workers(backend, cv, n_workers_all)
    cc_all = Ferrite.distribute_to_workers(backend, CellCache(dh_device), n_workers_all)
    Kes_all = KA.zeros(backend, Float32, n_workers_all, n_basefuncs, n_basefuncs)
    fes_all = KA.zeros(backend, Float32, n_workers_all, n_basefuncs)
    K_atomic = allocate_device_matrix(backend, sparse_type(Float32, Int32), dh)
    f_atomic = KA.zeros(backend, Float32, ndofs(dh))
    assemble_global_ka_atomic!(backend, cv_all, K_atomic, f_atomic, cc_all, cells_device, Kes_all, fes_all, n_workers_all)
    @test SparseMatrixCSC(K_atomic) ≈ K_unconstrained
    @test Vector(f_atomic) ≈ f_unconstrained
end

@testset "KernelAbstractions CSR assembly on simple grid ($backend, $(nameof(sparse_type_csr(Float32, Int32))))" begin
    grid, dh, cv, ch = setup_heat_problem(Float32)
    colors = create_coloring(grid)
    n_basefuncs = getnbasefunctions(cv)

    K_ref = allocate_matrix(SparseMatrixCSC{Float32, Int32}, dh)
    f_ref = zeros(Float32, ndofs(dh))
    assemble_global!(cv, K_ref, f_ref, dh)

    colors_device = [adapt(backend, c) for c in colors]
    n_workers = prod(compute_threads_and_blocks(maximum(length.(colors))))
    dh_device = adapt(backend, dh)
    cv_device = Ferrite.distribute_to_workers(backend, cv, n_workers)
    cc_device = Ferrite.distribute_to_workers(backend, CellCache(dh_device), n_workers)
    K_device = allocate_device_matrix(backend, sparse_type_csr(Float32, Int32), dh)
    f_device = KA.zeros(backend, Float32, ndofs(dh))
    Kes_device = KA.zeros(backend, Float32, n_workers, n_basefuncs, n_basefuncs)
    fes_device = KA.zeros(backend, Float32, n_workers, n_basefuncs)

    # `apply!` is not implemented for CSR device matrices yet, so only the assembled system
    # is compared against the CSC reference.
    assemble_global_ka!(backend, cv_device, K_device, f_device, cc_device, colors_device, Kes_device, fes_device, n_workers)
    @test SparseMatrixCSC(K_device) ≈ K_ref
    @test Vector(f_device) ≈ f_ref
end

@testset "KernelAbstractions heat problem on mixed grid ($backend, $(nameof(sparse_type(Float32, Int32))))" begin
    grid = generate_mixed_grid()

    dh = DofHandler(grid)
    sdh1 = SubDofHandler(dh, getcellset(grid, "triangle"))
    ip1 = Lagrange{RefTriangle, 2}()
    add!(sdh1, :u, ip1)
    cv1 = CellValues(Float32, QuadratureRule{RefTriangle}(Float32, 3), ip1)
    sdh2 = SubDofHandler(dh, getcellset(grid, "quad"))
    ip2 = Lagrange{RefQuadrilateral, 2}()
    add!(sdh2, :u, ip2)
    cv2 = CellValues(Float32, QuadratureRule{RefQuadrilateral}(Float32, 3), ip2)
    close!(dh)

    ch = ConstraintHandler(Float32, Int32, dh)
    add!(ch, Dirichlet(:u, union(getfacetset(grid, "top"), getfacetset(grid, "bottom")), (x, t) -> 1.0f0))
    close!(ch)

    K_ref = allocate_matrix(SparseMatrixCSC{Float32, Int32}, dh)
    f_ref = zeros(Float32, ndofs(dh))
    assemble_global!(cv1, K_ref, f_ref, sdh1)
    assemble_global!(cv2, K_ref, f_ref, sdh2; fillzero = false)
    apply!(K_ref, f_ref, ch)
    u_ref = solve_cpu(K_ref, f_ref)

    dh_device = adapt(backend, dh)
    K_device = allocate_device_matrix(backend, sparse_type(Float32, Int32), dh)
    f_device = KA.zeros(backend, Float32, ndofs(dh))
    for (i, (sdh, cv)) in enumerate(((sdh1, cv1), (sdh2, cv2)))
        colors = create_coloring(grid, sdh.cellset)
        colors_device = [adapt(backend, c) for c in colors]
        n_workers = maximum(length.(colors))
        n_basefuncs = getnbasefunctions(cv)
        cv_device = Ferrite.distribute_to_workers(backend, cv, n_workers)
        cc_device = Ferrite.distribute_to_workers(backend, CellCache(dh_device.subdofhandlers[i]), n_workers)
        Kes_device = KA.zeros(backend, Float32, n_workers, n_basefuncs, n_basefuncs)
        fes_device = KA.zeros(backend, Float32, n_workers, n_basefuncs)
        assemble_global_ka!(backend, cv_device, K_device, f_device, cc_device, colors_device, Kes_device, fes_device, n_workers; fillzero = i == 1)
    end
    apply!(K_device, f_device, adapt(backend, ch))
    @test solve_cpu(K_device, f_device) ≈ u_ref
end
