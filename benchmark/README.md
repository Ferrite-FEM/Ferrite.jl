# Ferrite.jl benchmark suite

The suite is run on every PR by [Tachometer.jl](https://github.com/KristofferC/Tachometer.jl)
(see `.github/workflows/Benchmarks.yml`), which is also the tool for running it locally --
see the [performance devdocs](https://ferrite-fem.github.io/Ferrite.jl/stable/devdocs/performance/).

## Design

**Representative configurations, not sweeps.** The suite does not permute all combinations
of element type × interpolation × order × field dimension — that grows multiplicatively and
mostly re-measures the same code. Instead, each group picks the few configurations that
exercise *structurally different code paths* for the functionality it covers, e.g.:
linear Lagrange on quadrilaterals (the plain 2D tensor-product path), quadratic Lagrange on
tetrahedra (3D simplex with edge/face dofs), a vectorized interpolation on hexahedra (3D +
vector field), Raviart–Thomas (non-identity mapping) and DiscontinuousLagrange (DG /
interfaces). When adding a benchmark, ask which code path it covers that the existing ones
do not.

**Every benchmark must clear the noise floor.** The CI comparison discards differences below
~1 µs (`time-floor` in the workflow): a benchmark faster than that can never report a
regression and only wastes CI time. Per-cell operations (`reinit!`, `celldofs!`, `assemble!`
scatter, ...) are therefore measured over a batch of cells (see `helper.jl`). Keep every
benchmark above ~5 µs, and below ~10 ms where possible so that it gets enough samples within
its time budget.

**Explicit parameters instead of tuning.** Every `@benchmarkable` declares `evals = 1`
(anything above the noise floor needs no more), which makes Tachometer skip tuning
entirely; `benchmarks.jl` errors on any benchmark that forgets it. `samples` and `seconds` are
defaulted globally at the bottom of `benchmarks.jl`; benchmarks with millisecond runtimes
or expensive `setup` declare `seconds = 1.0` themselves. This bounds a full pass at roughly
0.5 s per benchmark.

**Runtime budget.** The PR workflow runs the suite four times (two interleaved passes per
revision), so every second added to a pass costs four in CI. A full pass should stay under
~2 minutes; the whole workflow under ~10, about the time of a regular test CI run.

## Groups

| Group              | Covers                                                                 |
|--------------------|------------------------------------------------------------------------|
| `mesh`             | `generate_grid`, `ExclusiveTopology` + queries (`getneighborhood`, `facetskeleton`, `vertex_star_stencils`), facet sets, coloring |
| `dofs`             | `close!` (single/multi-field, subdomains), `renumber!`, `celldofs!`     |
| `fevalues`         | `CellValues`/`FacetValues`/`MultiFieldCellValues` construction; `reinit!` for identity, Piola (Raviart–Thomas, Nedelec) and hessian-enabled mappings; `function_value/gradient`, `spatial_coordinate` |
| `assembly`         | element kernels (incl. `shape_symmetric_gradient` and a mixed `shape_divergence` block), full global assembly loops, sparse scatter (CSC/CSR/Symmetric), facet and DG interface loops |
| `constraints`      | `ConstraintHandler` close/update (`Dirichlet`, `PeriodicDirichlet`, `ProjectedDirichlet`), `apply!`/`apply_zero!`/`apply_rhs!`/`apply_local!`, periodic (affine) condensation |
| `sparsity-pattern` | pattern construction, `add_entry!`, matrix instantiation                |
| `postprocessing`   | `L2Projector`, `PointEvalHandler`, `evaluate_at_grid_nodes`, `apply_analytical!` |
| `amr`              | `refine!`, `balanceforest!` (2D/3D), `creategrid` (2D/3D), `ConformityConstraint` |

Run a single group locally by setting the environment variable
`FERRITE_SELECTED_BENCHMARKS=assembly`, both under Tachometer and when including
`benchmarks.jl` directly.

## Deliberately not covered

- **VTK export** (`VTKGridFile`, `write_solution`, ...): dominated by file IO, which is too
  noisy on CI runners to compare; the compute part (`evaluate_at_grid_nodes`) is covered.
- **Interpolations that share the identity mapping with Lagrange** (`Serendipity`,
  `CrouzeixRaviart`, `RannacherTurek`, `BubbleEnrichedLagrange`, ...): only the (covered)
  reference-value tables differ.
- **Wedge/Pyramid and embedded cells, 1D**: no code path of their own that the covered
  element types do not exercise more cheaply.
- **`BlockSparsityPattern`**: niche, and shares the `add_entry!` machinery with the covered
  `SparsityPattern`.
- **`ForestBWG` construction**: dominated by `ExclusiveTopology`, covered in `mesh`.
  **`coarsen!`/`refine_and_coarsen!`**: share the marked-leaf-list rebuild machinery with
  the covered `refine!`.
