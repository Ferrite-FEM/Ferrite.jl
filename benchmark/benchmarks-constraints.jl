#----------------------------------------------------------------------#
# Constraints: ConstraintHandler construction, update! and application
#----------------------------------------------------------------------#
# All benchmarks share one representative problem: a quadratic scalar field on a 50×50
# quadrilateral grid (10201 dofs) with a stiffness-like sparse matrix. Dirichlet exercises
# the prescribed-dof path, PeriodicDirichlet the affine-constraint (condensation) path.

SUITE["constraints"] = BenchmarkGroup()

let g = SUITE["constraints"]
    grid = generate_grid(Quadrilateral, (50, 50))
    dh = DofHandler(grid)
    add!(dh, :u, Lagrange{RefQuadrilateral, 2}())
    close!(dh)

    dirichlet_set = union(getfacetset.((grid,), ("left", "bottom", "right", "top"))...)
    periodic_pairs = collect_periodic_facets(grid, "left", "right")

    close_dirichlet = function (dh, set)
        ch = ConstraintHandler(dh)
        add!(ch, Dirichlet(:u, set, (x, t) -> t * x[1]))
        return close!(ch)
    end
    close_periodic = function (dh, pairs)
        ch = ConstraintHandler(dh)
        add!(ch, PeriodicDirichlet(:u, pairs))
        return close!(ch)
    end
    ch_dirichlet = close_dirichlet(dh, dirichlet_set)
    ch_periodic = close_periodic(dh, periodic_pairs)

    # Construction
    g["close! Dirichlet (4 sides)"] = @benchmarkable $close_dirichlet($dh, $dirichlet_set) evals = 1
    g["close! PeriodicDirichlet"] = @benchmarkable $close_periodic($dh, $periodic_pairs) evals = 1
    g["collect_periodic_facets"] = @benchmarkable collect_periodic_facets($grid, "left", "right") evals = 1

    # Re-evaluating the inhomogeneities for new times (once per "timestep")
    g["update! (8 timesteps)"] = @benchmarkable FerriteBenchmarkHelpers.update_sweep!($ch_dirichlet, 8) evals = 1

    # Stiffness-like matrices to apply the constraints to. The periodic one needs the
    # extra sparsity entries for the affine condensation.
    cellvalues = CellValues(QuadratureRule{RefQuadrilateral}(3), Lagrange{RefQuadrilateral, 2}())
    K_dirichlet = FerriteBenchmarkHelpers.assemble_global!(
        allocate_matrix(dh), zeros(ndofs(dh)), dh, cellvalues, FerriteBenchmarkHelpers.laplace_kernel!
    )
    K_periodic = FerriteBenchmarkHelpers.assemble_global!(
        allocate_matrix(dh, ch_periodic), zeros(ndofs(dh)), dh, cellvalues, FerriteBenchmarkHelpers.laplace_kernel!
    )

    # Application to the global system. apply! zeroes constrained rows/columns in place, so
    # each sample needs a fresh copy of the matrix.
    g["apply! Dirichlet"] = @benchmarkable(
        apply!(K, f, $ch_dirichlet),
        setup = (K = copy($K_dirichlet); f = zeros(size(K, 1))),
        evals = 1, seconds = 1.0,
    )
    g["apply! PeriodicDirichlet (condensation)"] = @benchmarkable(
        apply!(K, f, $ch_periodic),
        setup = (K = copy($K_periodic); f = zeros(size(K, 1))),
        evals = 1, seconds = 1.0,
    )
    # apply_zero! writes zeros to constrained entries whatever their value, so repeated
    # application does identical work; the sweep needs no per-sample setup.
    f = rand(ndofs(dh))
    g["apply_zero! vector (64 times)"] = @benchmarkable FerriteBenchmarkHelpers.apply_zero_sweep!($f, $ch_dirichlet, 64) evals = 1

    # The rhs-only path for repeated solves with the same matrix (apply! once, then
    # apply_rhs! per timestep). get_rhs_data must see the matrix before apply! zeroes the
    # constrained columns.
    rhs_data = get_rhs_data(ch_dirichlet, K_dirichlet)
    g["apply_rhs! (64 times)"] = @benchmarkable FerriteBenchmarkHelpers.apply_rhs_sweep!($rhs_data, $f, $ch_dirichlet, 64) evals = 1

    # Local condensation before assembly (the recommended pattern for e.g. nonlinear
    # problems). The bottom 10 rows of cells give a realistic mix of constrained cells and
    # unconstrained ones (which take the early-out path). apply_local! modifies Ke/fe in
    # place, hence the fresh copies in setup.
    nloc = 500
    local_dofs = [celldofs(dh, i) for i in 1:nloc]
    apply_local_sweep! = function (Kes, fes, local_dofs, ch)
        for i in eachindex(Kes)
            apply_local!(Kes[i], fes[i], local_dofs[i], ch)
        end
        return Kes
    end
    n = ndofs_per_cell(dh)
    g["apply_local! ($nloc cells)"] = @benchmarkable(
        $apply_local_sweep!(Kes, fes, $local_dofs, $ch_dirichlet),
        setup = (Kes = [rand($n, $n) for _ in 1:($nloc)]; fes = [rand($n) for _ in 1:($nloc)]),
        evals = 1,
    )
end

# Dirichlet on a tensor-valued field: update! goes through the tensor-return
# normalization wrapper set up in add!.
let g = SUITE["constraints"]
    grid = generate_grid(Quadrilateral, (50, 50))
    dh = DofHandler(grid)
    add!(dh, :s, TensorizedInterpolation{SymmetricTensor{2, 2}}(Lagrange{RefQuadrilateral, 2}()))
    close!(dh)
    dirichlet_set = union(getfacetset.((grid,), ("left", "bottom", "right", "top"))...)
    ch = ConstraintHandler(dh)
    add!(ch, Dirichlet(:s, dirichlet_set, (x, t) -> SymmetricTensor{2, 2}((t * x[1], x[2], 0.0))))
    close!(ch)
    g["update! Dirichlet tensor field (8 timesteps)"] = @benchmarkable FerriteBenchmarkHelpers.update_sweep!($ch, 8) evals = 1
end

# ProjectedDirichlet: boundary conditions for non-nodal (here H(div)) interpolations,
# enforced by an L2 projection on each facet. close! includes the projection solves.
let g = SUITE["constraints"]
    grid = generate_grid(Triangle, (20, 20))
    dh = DofHandler(grid)
    add!(dh, :q, RaviartThomas{RefTriangle, 1}())
    close!(dh)
    set = union(getfacetset(grid, "left"), getfacetset(grid, "bottom"))

    close_projected = function (dh, set)
        ch = ConstraintHandler(dh)
        add!(ch, ProjectedDirichlet(:q, set, (x, t, n) -> x ⋅ n))
        return close!(ch)
    end
    g["close! ProjectedDirichlet H(div)"] = @benchmarkable $close_projected($dh, $set) evals = 1
end
