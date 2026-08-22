# Plan: DG- and constraint-aware grid coloring

**Status: IMPLEMENTED 2026-08-15** (uncommitted, on `master`). One refinement was made
during implementation: the "sharp" tier as planned (`A_n ∪ A_f ∪ A_f²`) was internally
inconsistent — `A_n` *is* what contains the 3D vertex diagonals, and it is only needed
when a continuous field exists. The implementation therefore has three DG modes, each
exact for its field configuration: `:facet` (`A_f ∪ A_f²`, purely discontinuous
discretization — vertex diagonals excluded), `:sharp` (`A_n ∪ A_f²`, continuous fields
exist but only discontinuous ones cross interfaces), `:product` (a continuous field
crosses, or grid-only API). Mode selection is automatic in `create_coloring(dh, ...)`.

*Draft for iteration — now implemented. Companion to the WorkStream exploration
(see `WORKSTREAM.md`); coloring is the only threading strategy that needs conflict
analysis at all (atomic and the pipeline are immune).*

## Context

`create_coloring` currently defines "conflict" as "shares a node", which is only valid
for continuous fields assembled cell-locally. Two cases need a larger conflict graph:

1. **DG/interface assembly**: an item also writes dofs of its *facet* neighbors. The
   exact conflict graph depends on which fields appear in the interface term:
   - **Only discontinuous (L2) fields cross interfaces** (standard DG): dofs are
     cell-interior, so the graph is `A_n ∪ A_f ∪ A_f²` (node graph for cell terms of
     any CG fields + facet neighbors + cells sharing a facet neighbor). 2D corner
     neighbors are included implicitly via the shared facet neighbor's diagonal block;
     3D vertex-only diagonals share no facet neighbor and provably no written entry —
     correctly excluded.
   - **A continuous field crosses interfaces** (combined case): items write CG dofs of
     facet neighbors, and those are shared onward — the exact graph is the closure
     product `(I ∪ A_f)·(I ∪ A_n)·(I ∪ A_f)`. This includes 3D vertex diagonals *and*
     node-distance-3 chains (A—X—Y—D with X ∈ Nf(A), Y ∈ Nf(D), X/Y node-adjacent:
     e.g. quads (0,0) and (3,0) in a row conflict through the shared edge dofs of
     (1,0)/(2,0)). **Node-distance-2 is insufficient here.**
   The grid-based API has no field information, so `interface_coupling = true` there
   must always use the conservative product graph. The dof-based API inspects the
   interpolations and the `interface_coupling` matrix (which fields cross interfaces)
   and uses the sharp graph when all crossing fields are discontinuous — this is what
   makes mirroring the sparsity kwargs load-bearing, not just convenient.
2. **Constraint condensation** (`apply_assemble!`/`apply_local!`, e.g.
   `PeriodicDirichlet` or `AffineConstraint`): verified in `_condense_local!`
   (`src/Dofs/ConstraintHandler.jl:1835-1921`) — an item with dofs `S` writes global
   entries only within `S ∪ M(S)`, where `M(S)` = masters of nontrivially-constrained
   dofs in `S` (**plain Dirichlet causes no global writes**, hence no conflicts). Cells
   conflict when their expanded sets intersect — possibly mesh-distant cells (opposite
   periodic faces). Only master-mediated intersections are new edges; the node graph
   already covers `S_A ∩ S_B` on conforming grids.

Design principles (settled):

- **CG and DG use node/vertex-level structures only** (node-sharing graph; facet
  neighbors classified by shared-vertex counting — no `ExclusiveTopology`, no dof data).
- **Only the constraint source needs dof↔cell mapping**, and only *affine* constraints.
  Even there the map is restricted: per-cell lists only for the small dof set
  {constrained dofs ∪ masters}, built in one pass over `celldofs` (an `ndofs`-sized Int
  scratch array for O(1) lookup; no full dof→cells structure).
- The coloring algorithms (greedy/workstream) stay untouched — they only consume the
  incidence matrix; all new edges are unioned into it before partitioning/coloring.
- Explicitly rejected: running the machinery "with dofs instead of nodes" as the index
  space. It would be marginally sharper (subdomain fields; pure-L2 fields share
  nothing), but costs `ndofs`-sized structures and still would not cover DG or
  constraints by itself (interface conflicts involve *unshared* neighbor dofs, constraint
  conflicts involve masters) — both expansions are needed regardless.

**Correctness note (validated)**: workstream's odd/even zone merge requires edges to
span at most one BFS zone — automatically true for *any* edge present in the matrix
during BFS partitioning (a BFS edge never skips a level, even a periodic long-range
one). Invariant to document in a comment: never add conflict edges after partitioning;
matrix build and coloring must use identical options (single call site does this).

## Changes

### 1. `src/Grid/coloring.jl` — refactor (no behavior change)

Extract the node→cells CSR build (current lines 54-73 of `create_incidence_matrix`) into
`_build_node_to_cell_map(grid, cellvec) -> (nodeptr, nodecells)`. Keep
`_gather_neighbor_chunk!` untouched so the default path stays byte-identical (CHANGELOG
documents determinism independent of thread count).

### 2. `src/Grid/coloring.jl` — facet adjacency `A_f`

- `_facet_conflict_threshold(a, b) = min(getrefdim(a), getrefdim(b))` — facet iff
  shared-vertex count ≥ threshold (rdim 1: vertex = 1, rdim 2: edge = 2, rdim 3:
  tri/quad face ≥ 3). Conservative (over-inclusive) for mixed-rdim grids, which is safe
  (only adds colors, never misses a conflict). Count via existing
  `Ferrite._num_shared_vertices` (`src/Grid/topology.jl:175-184`) — corner **vertices**,
  not node run-lengths (a `QuadraticTetrahedron` edge neighbor shares 3 *nodes* but only
  2 vertices — run-length counting would misclassify it as a facet neighbor).
- `_build_facet_adjacency(grid, cellvec, nodeptr, nodecells) -> (facetptr, facetadj)`:
  same chunked gather/sort/dedup + count/prefix-sum/copyto pattern as the existing
  incidence build (contiguous chunks → deterministic, thread-count independent). For
  each unique node-sharing candidate, keep it iff `_num_shared_vertices ≥ threshold`.
  Must be fully built (own `@sync`) before step 3, which reads other cells' lists.

### 3. `src/Grid/coloring.jl` — union gather

`_gather_conflict_chunk!(colcount, grid, cellvec, chunk, nodeptr, nodecells, facetptr,
facetadj, extras, mode)` — per cell, concatenate candidates from:
(a) node sharing (as today);
(b) DG **sharp mode** (`A_f²`): for each facet neighbor `j`, append `j`'s facet list
    (`A_f ⊆ A_n` already);
(b′) DG **product mode** (`(I ∪ A_f)(I ∪ A_n)(I ∪ A_f)`, combined CG+DG case): for each
    `X ∈ {i} ∪ Nf(i)`, for each `Y ∈ {X} ∪ Nn(X)`, append `Y` and `Nf(Y)`. Requires the
    node adjacency `A_n` materialized as CSR first — build it with the existing
    incidence machinery, then run this second composition pass over `A_n` + `facetadj`
    (both read-only, per-cell independent → same chunked/deterministic pattern);
(c) `get(extras, cellid, nothing)` constraint edges.
One sort/dedup as today, plus skipping `candidate == cellid` in the dedup loop (the
closure lists contain self). Output keeps all invariants: sorted contiguous per-column
buffers, symmetric matrix (each edge source is symmetric — the product graph because
`A_f`, `A_n` are and the middle factor is sandwiched symmetrically), no diagonal,
deterministic in thread count.

Field-continuity detection for mode selection (dof-based method): a field crosses
interfaces if `interface_coupling` is `true` or its matrix row/col has any `true`;
a crossing field forces product mode unless its interpolation is discontinuous
(no vertex/edge/face dofs — check for an existing predicate à la
`is_discontinuous`/conformity traits on `DiscontinuousLagrange`; add a small helper
otherwise).

### 4. `src/Dofs/ConstraintHandler.jl` — constraint extras

Include-order constraint: `coloring.jl` (`src/Ferrite.jl` L129) is included before
`src/Dofs`, so `coloring.jl` holds an untyped `ch` kwarg and a stub
`_incidence_constraint_extras(::Nothing, grid, cellvec, facetptr, facetadj) = nothing`;
the `::ConstraintHandler` method lives at the bottom of `ConstraintHandler.jl`.

That method:
1. Require `isclosed(ch)` and `get_grid(ch.dh) === grid`, error otherwise.
2. Collect nontrivial affine constraints (skip `dofcoefficients === nothing`/empty);
   none → return `nothing` (Dirichlet-only `ch` reproduces the default coloring
   exactly).
3. `master_id = zeros(Int, ndofs(dh))` scratch, ids assigned in deterministic iteration
   order.
4. Build per-master cell lists in one pass over `sdh.cellset ∩ cellset` and `celldofs`:
   cell contains master `m` directly, or contains a constrained dof with master `m`;
   `sort!`/`unique!` each list.
5. If DG mode is also active, expand each list with facet neighbors of its members
   (interface items condense the neighbor's constrained dofs too — write closure).
6. Emit `Dict{Int, Vector{Int}}` with all ordered pairs of each list (dedup happens in
   the final gather).

Cost: negligible (lists are a handful of boundary cells for periodic BCs).

### 5a. Public API — option space (**decided**: Option 1 with grid sub-choice (c) —
one-shot methods; grid method conservative-only, sharpness only via the dof method
which detects it. Two-layer graph API shelved until custom conflict sources
materialize. Analysis kept for the record.)

**What each input can provide:**

| input | provides | cannot provide |
|---|---|---|
| grid | `A_n`; `A_f` via shared-**vertex** counting (threshold = refdim: 1D vertex, 2D edge ≥2, 3D face ≥3 — ">1 node" suffices only in 2D, and counting *nodes* instead of vertices misclassifies quadratic cells); hence `A_f²` and the product graph | what assembly *writes* — field continuity is not in the mesh, so "sharp" is only available as a **user assertion** |
| dh | interpolations → per-field discontinuity detection; with the `interface_coupling` matrix → *automatic* sharp/product mode selection | — |
| ch | affine-constraint edges | — |
| topology (optional) | exact facet adjacency via `create_cell_to_neighbors`, provably consistent with the sparsity pattern | — |

**Option 1 — one-shot methods with kwargs** (current plan). Sub-choice for the grid
method's DG knob:
  - (a) tri-state kwarg, e.g. enum `InterfaceConflicts.None / .Discontinuous / .Continuous`
    (with `false`/`true` as aliases for None/Continuous). One knob, self-documenting,
    `.Discontinuous` = the sharp assertion. EnumX style matches `ColoringAlgorithm`.
  - (b) two Bools: `interface_coupling::Bool` + `discontinuous::Bool = false` assertion.
    Simple but kwarg-soup; the assertion silently means nothing without the first flag.
  - (c) no sharp option on the grid method at all: grid = always conservative, sharp
    requires the dof method (which detects it). Cleanest docs story ("pass your dh to
    get a better coloring"), zero assertion footguns; pure-DG users always have a dh
    in practice.

**Option 2 — two-layer API**: make the conflict graph a public object:
`g = create_conflict_graph(grid|dh, ch; kwargs...)` → `create_coloring(g; alg)`, with
one-shot wrappers kept for the common cases. Pros: inspectable/visualizable, testable,
composable, and users can add custom edges (the deal.II `get_conflict_indices` analog)
for couplings we did not predict — the escape hatch the WorkStream API discussion
wanted. Cons: new public type and vocabulary to maintain; two calls for the common case.

**Option 3 — conflict callback** (deal.II style): `create_coloring(grid; conflicts =
cell -> indices)`. Maximum flexibility but poor performance shape (closure per cell,
no batch structure) and awkward ergonomics; not recommended as the primary API, and
Option 2 subsumes its use case.

Decision rationale: the assertion knob on the grid method buys little (pure-DG users
have a dh) and invites "asserted sharp but actually mixed" bugs. Option 2's
custom-conflict use case is instead covered by the `extra_conflicts` kwarg (section 5b),
so the two-layer API stays shelved.

### 5. Public API — grid-based and dof-based methods (current draft, Option 1)

Two entry points (user decision): a minimal grid-based one, and a dof-based one that
**mirrors the `add_sparsity_entries!`/`allocate_matrix` signature** so users can pass
the exact same arguments they used to allocate the matrix. Headline UX: *color with the
same `(dh, ch; kwargs...)` you allocated with, and the coloring is safe for the pattern
you allocated*.

```julia
# Grid-based (src/Grid/coloring.jl). Unchanged default for CG. With
# interface_coupling = true it has no field information and therefore always uses the
# conservative product graph (safe for combined CG+DG; more colors than the dof-based
# method gives for pure DG — document this as the reason to prefer the dof method):
create_coloring(
    g, cellset = 1:getncells(g);
    alg = ColoringAlgorithm.WorkStream,
    interface_coupling::Bool = false,
    topology = nothing,   # optional: exact facet adjacency from it
    extra_conflicts = nothing
)  # see 5b

# Dof-based, mirroring add_sparsity_entries!(sp, dh, ch; ...) minus `sp` (defined in
# src/Dofs/ConstraintHandler.jl due to include order):
create_coloring(
    dh::DofHandler, ch::Union{ConstraintHandler, Nothing} = nothing;
    cellset = 1:getncells(get_grid(dh)),
    alg = ColoringAlgorithm.WorkStream,
    keep_constrained::Bool = true,                                # no effect
    coupling = nothing,                                           # no effect
    interface_coupling::Union{Nothing, Bool, AbstractMatrix{Bool}} = nothing,
    topology = nothing,
    algebraic_couplings = (),                                     # must be ()
    extra_conflicts = nothing
)                                    # see 5b

# Internal:
create_incidence_matrix(
    grid, cellset = 1:getncells(grid);
    interface_coupling::Bool = false, topology = nothing,
    ch = nothing
)
```

Per-kwarg semantics of the dof-based method:

- `ch`: conflict edges from affine-constraint condensation (step 4).
- `coupling`: accepted and shape-validated (`_check_coupling_kwarg`) but **no effect**,
  documented — any shared dof conflicts regardless of which field blocks are written
  (even `coupling[λ, λ] = false` Lagrange setups conflict through `f` and mixed
  blocks), so ignoring is conservative and correct.
- `interface_coupling`: `nothing`/`false` → off; `true` or any-`true` matrix → DG
  conflicts on. The matrix content is *not* just reduced via `any`: it determines which
  fields cross interfaces and thereby the conflict-graph mode — sharp
  (`A_n ∪ A_f ∪ A_f²`) when all crossing fields are discontinuous, product graph
  otherwise (see Context). Superset of the sparsity signature, so the same arguments
  splat through.
- `topology`: when given, derive facet adjacency with the existing
  `create_cell_to_neighbors(grid, topology)` (`src/Dofs/sparsity_pattern.jl:1040`) —
  the *same helper* `add_interface_entries!` uses, so coloring conflicts are by
  construction aligned with the pattern's interface entries. When `nothing`, use the
  vertex-counting classification from step 2 (unlike the sparsity code, do *not*
  auto-build an `ExclusiveTopology` — not needed).
- `keep_constrained`: accepted, **no effect** — `_condense_local!` shows the
  write-conflict structure is identical either way (it only changes which entries
  exist); accepting it means the `keep_constrained = false` + `apply_assemble!`
  workflow (exactly where `ch`-aware coloring matters) can splat its kwargs.
- `algebraic_couplings`: accepted with default `()`, **error if nonempty** — an
  `AlgebraicCoupling` to a global variable assembled inside the cell loop makes every
  cell pair conflict (coloring degenerates to serial); refuse with a pointer to
  atomic/pipeline assembly rather than silently under- or over-color. (A lone
  `FacetCoupling` is in principle just the interface-conflict case — later refinement.)

Shared notes:

- The `dh` method forwards to the grid machinery on `get_grid(dh)`; mesh conflicts stay
  node/vertex-based even in the dof-based method. `create_coloring(dh)` with no `ch` and
  no interface coupling equals the grid method's coloring.
- Fast path: no DG, no extras → the existing gather loop runs verbatim (default
  benchmarks must not move).
- No new exports (`create_coloring` already exported). Docstrings document the kwargs,
  the closed-`ch`/`isclosed(dh)`/matching-grid requirements, that only *affine*
  constraints add conflicts (plain Dirichlet is free), and that long-range edges may
  reduce workstream zone quality (more colors), not correctness.
- Extend the comment at `coloring.jl` L184: incidence matrix must be built with the same
  cellset *and the same conflict options*.

### 5b. Custom conflict entries — `extra_conflicts` kwarg (both methods)

User-provided conflicts for couplings the built-ins don't model (element routines with
cross-cell dependencies, algebraic couplings the user understands better than we can
infer, etc.) — the escape hatch that motivated the (shelved) two-layer API, as a kwarg:

```julia
create_coloring(grid; extra_conflicts = Dict(1 => [17, 203]))         # adjacency form
create_coloring(dh, ch; extra_conflicts = [[1, 17, 203], [4, 99]])    # cliques form
```

- Accepted forms: `AbstractDict{Int, <:AbstractVector{Int}}` (cell → conflicting
  cells) or an iterable of cell groups, each treated as a clique (all pairs conflict —
  the natural way to express "all cells coupled through this global variable").
- Normalization in `create_incidence_matrix`: expand cliques to pairs, **symmetrize**
  (an asymmetric `Dict` entry `a => [b]` still yields both directions — the incidence
  matrix must stay symmetric), drop self-edges, **filter edges touching cells outside
  the cellset** (consistent with how node conflicts are restricted; also lets one
  conflict spec be reused across sub-colorings), validate ids ∈ `1:getncells` (error).
- Merged into the same `extras::Dict{Int, Vector{Int}}` as the constraint edges before
  the gather — dedup happens there, and the workstream partitioning automatically
  respects the edges (they are in the matrix before BFS zoning, per the invariant).
- Composes with everything: DG modes, `ch`, cellsets.

### 6. Tests — `test/test_grid_dofhandler_vtk.jl` (extend `@testset "grid coloring"` ~L719)

Independent references using *different* code paths: DG conflicts via
`ExclusiveTopology`/`get_facet_facet_neighborhood` write-closures; constraint conflicts
via dof-level sets `S ∪ M(S)` from `ch`. For both algorithms:

1. DG safety on Line/Triangle/Quadrilateral/QuadraticTriangle/Tetrahedron/Hexahedron
   grids + subset and unconnected-subset variants.
2. Exactness (sharp mode, via a pure-DG dh): small QuadraticTetrahedron grid — the
   incidence matrix equals brute-force `A_n ∪ A_f ∪ A_f²` (catches the node-run-length
   trap); Hexahedron — 3D vertex-only-diagonal cells not added beyond `A_n`.
2b. Combined-case necessity (product mode): mixed dh (`DiscontinuousLagrange` +
   `Lagrange` fields, CG field crossing interfaces) on a quad strip — cells (0,0) and
   (3,0) (three apart in a row) must get different colors; same via the grid API with
   `interface_coupling = true`. Safety vs a dof-level write-closure reference
   (`S(c) = dofs({c} ∪ Nf(c))`, conflict iff intersect). Also: pure-DG dh gives the
   sharp graph while the grid API gives the product graph (sharp ⊆ product, and they
   differ exactly on e.g. 3D vertex diagonals).
3. Periodic (2D quad + small 3D hex): `create_coloring(dh, ch)` — necessity (cells of
   constrained dof vs cells of master differ in color) and safety vs the dof-level
   reference. Also `create_coloring(dh)` == grid-method coloring.
4. Long-range `AffineConstraint` corner-to-corner: colors differ (stresses workstream
   zoning across a component-merging edge).
5. Combined DG + periodic: facet neighbor of a constrained cell vs master cells differ;
   safety vs closure-expanded reference.
6. Degenerate: Dirichlet-only `ch` == default coloring; wrong-grid `ch` throws; unclosed
   `ch` throws; `Set` vs sorted-vector cellset equal; empty/single-cell sets with kwargs.
7. Kwarg-mirror contract: `create_coloring(dh, ch; kwargs...)` with the *exact* kwargs
   passed to `allocate_matrix` (incl. `coupling`, `keep_constrained`, matrix-valued
   `interface_coupling`, `topology`) runs and is safe; `topology`-given vs
   vertex-counting paths produce the same incidence matrix; nonempty
   `algebraic_couplings` throws.
8. `extra_conflicts`: two mesh-distant cells given as a conflict get different colors
   (both algorithms — stresses workstream zoning like the long-range constraint case);
   asymmetric `Dict` input (`a => [b]` only) still separates both directions; cliques
   form ≡ equivalent pairwise `Dict`; edges to cells outside the cellset are ignored;
   out-of-range cell ids throw; composes with `interface_coupling` and `ch`.

### 7. Benchmarks — `benchmark/benchmarks-mesh.jl` (~L66-81)

Add to the coloring let-block: workstream DG (Hexahedron 15³), greedy DG (Tetrahedron
8³), workstream periodic (Quadrilateral 50×50 with a `PeriodicDirichlet` ch). Existing
default-path cases must not regress (fast path).

### 8. Docs + CHANGELOG

- `docs/src/literate-howto/threaded_assembly.jl` (~L58-60): note the default conflict
  definition is only valid for continuous fields assembled cell-locally; point to
  `interface_coupling` and the `(dh, ch)` method.
- CHANGELOG.md `### Added` entry; default behavior and determinism guarantee unchanged.
- `reference/grid.md` needs nothing (docstring flows through existing `@docs`).

## Verification

1. Full test suite, or at minimum the coloring testsets:
   `test/test_grid_dofhandler_vtk.jl`, `test/test_abstractgrid.jl`.
2. New tests above (safety against independent references is the load-bearing check).
3. Benchmark sanity: run the coloring cases before/after — default path unchanged,
   DG/periodic cases with sane cost.
4. End-to-end smoke test (ad-hoc script, not committed): thread the DG heat equation
   tutorial's cell+interface loops over `create_coloring(grid; interface_coupling =
   true)` colors (interfaces owned by the lower cell id, processed within the owner's
   item) and check the assembled `K` matches the serial result exactly; same for a CG
   problem with `PeriodicDirichlet` + `apply_assemble!` and `create_coloring(dh, ch)`.

## Open points for iteration

- ~~Naming/shape of `interface_coupling`~~ → settled: dof-based method mirrors the
  `add_sparsity_entries!` kwargs verbatim (accepting `Bool` as a convenience superset);
  irrelevant kwargs are accepted-and-documented as no-ops since ignoring them is
  provably conservative.
- `algebraic_couplings`: currently refused when nonempty. Possible refinement: accept
  descriptors we can reason about (`FacetCoupling` ⇒ interface conflicts;
  `CellCoupling` ⇒ nothing beyond the node graph?) and only error on
  `AlgebraicCoupling`. Needs reading the descriptor types first.
- Whether the `dh` method should restrict conflicts to cells covered by the
  `SubDofHandler`s (cells outside any sdh have no dofs — currently they'd still get
  node-based edges from the grid machinery, which is conservative but never wrong).
- Whether `_build_facet_adjacency` should be skipped in favor of classifying during the
  union gather itself (one pass fewer, but `A_f²` needs completed neighbor lists —
  probably keep two passes).
- Combined DG + constraints write-closure expansion (step 4.5): included for
  correctness; confirm it's not over-engineering for the intended use cases.
