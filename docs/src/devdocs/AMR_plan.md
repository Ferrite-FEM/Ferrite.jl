# AMR (p4est) Work Plan — PR #780

Living tracker for finishing/optimizing the `ForestBWG` adaptive mesh support.
Check items off as they land. Companion docs:
- `docs/src/devdocs/AMR_iterator.md` — source-grounded design note for the iterator.
- `benchmark/amr-perf-log.md` — performance baseline + run log.

**Branches:** work on `mk/p4est`; backup at `kc/p4est-backup`. Commits squashed
+ rebased on master.

**Two-track strategy:** Track A (Tiers 0–2) is safe, mechanical, AI-friendly —
every change must reproduce the current output byte-for-byte, verified against the
golden tests + benchmark. Track B (Tier 3, the iterator) is the architectural
rewrite and needs human design review; build it incrementally with the
(optimized) `creategrid` as the correctness oracle.

---

## Tier 0 — Groundwork ✅ (done)

- [x] Squash + rebase on master; push backup branch.
- [x] Verify the IBWG2015 iterator model against both papers (adversarial pass).
      Corrected two errors: hanging nodes are **not** detected "for free", and the
      iterator is **point-centric**, not volume/face/edge/corner. → `AMR_iterator.md`.
- [x] Capture a performance baseline (i9-12900K, Julia 1.12.6, commit `c82281999`).
      → `benchmark/amr-perf-log.md`.
- [x] Fix the latent bugs found during verification (see "Bug fixes" below).

---

## Tier 1 — Safety net + quick wins

Byte-identical against the golden oracle throughout. See `benchmark/amr-perf-log.md`.

**1a. Golden-output + invariant tests** ✅
- [x] Renumbering-invariant fingerprint of forest + grid frozen in
      `test/amr_golden/` (16 cases: 2D/3D, multi-tree, rotated, balanced, disc).
- [x] Independent correctness invariants (unique nodes, positive orientation,
      box shape, hanging = mean-of-constrainers) — catch bugs the golden would enshrine.

**1b. Quick-fixes** ✅ (the original plan mis-diagnosed the bottleneck; corrected below)
- [x] `hangingnodes` was GC-bound (1.2 GB / 800 constraints), not lookup-bound:
  - [x] `edges()` used `ntuple(f, 12)` → `n>10` hits Base's type-unstable generic
        path (verified via `return_types`/`@allocated`; `n≤10` const lengths were
        already stable). Now `ntuple(f, Val(12))`.
  - [x] hoisted `parent_`/`parentfaces`/`parentedges` out of the per-vertex loop (4–8×/leaf).
  - Result (3D, 11264 cells): **1194 → 15 MB, ~541 → 429 ms**.
- [x] `balancetree`: reuse Q/T/Tparents buffers; Set parent-check replaces O(n²)
      `p ∉ parent.(T,b)` (per Max's allocation note).
- [x] `creategrid` Phase 4: `transform_pointBWG` only the unique owner nodes.
- [x] `Base.hash(OctantBWG)` + `Set` membership in `hangingnodes` — O(1) leaf
      lookup that scales to large trees.
- [DROPPED] binary-search leaf membership: `searchsortedfirst` rides on
  `isless→morton` (~120 bit-ops/compare), so it lost to cheap-`==` scans at
  realistic tree sizes and regressed `balance_*`. A hash `Set` is the right
  scalable membership; `balance_*`/`refine!`/`coarsen!` left as original scans.
- [ ] `balanceforest!`: dirty-tree tracking (deferred — converges in 1 iteration
      on tested cases; low priority).

After 1b, `hangingnodes` is **compute-bound** (Dict lookups + the 90k-iteration
vertex loop). Further gains there need the iterator (Tier 3).

---

## Tier 2 — Abstraction layer + docs

Make the raw bit-ops reviewable (Dennis's ask) and prepare for the iterator.

- [ ] Wrap repeated leaf queries (`leaf_exists`, neighbor predicates) behind named helpers.
- [ ] Consider the index/cell types flagged at `BWG.jl:1` (`FacetIndexBWG`,
      `QuadrilateralBWG`, `HexahedronBWG`) to remove the index-convention mixing.
- [ ] Document the p4est types and how `OctantBWG`/`OctreeBWG`/`ForestBWG` interplay
      (extend `AMR.md`), and the (current) `creategrid` pipeline.

---

## Tier 3 — IBWG2015 LNodes iterator (architectural rewrite)

Replaces the multi-pass `creategrid`+`hangingnodes` (BWG2011 Alg 20) with one
O(n) traversal. Full build order + the corrected mental model are in
`AMR_iterator.md`. Serial first; distributed pieces collapse to no-ops.

- [x] Verify the Morton-sort prerequisite (golden + the base-descent test confirm
      `split_array` reconstructs the contiguous ordered leaves, 2D/3D).
- [x] **Base recursive descent** `iterate_leaves` (volume specialization of Alg 5.2):
      `split_array` partition + stop-at-singleton-leaf; visits exactly the leaves in
      Morton order (tested on all 16 golden cases). This is the foundation + the
      conceptual "base iterator" the team was blocked on.
- [x] **`split_bounds`** (Alg 3.3): non-allocating index-based partition (stack
      `NTuple` of child boundaries; no `𝐤` vector, no `SubArray` views). The descent
      carries `(leaves, lo, hi)` ranges → `iterate_leaves` over 256 leaves: 0 bytes.
- [x] **2D intra-tree hanging** (`iterate_hanging_2d`): face descent emits hanging
      edge midpoints; matches `creategrid` `conformity_info` exactly (single-tree 2D).
- [x] **3D intra-tree hanging** (`iterate_hanging_3d`): face descent emits face
      centre (4 constrainers) + 4 face-edge midpoints (2 constrainers). Captures
      face- AND edge-centre hanging for 2:1-balanced meshes (no 4-way edge descent
      needed — proven via the edge-cycle argument). Matches `creategrid` (single-tree 3D).
- [x] **Inter-tree hanging + unified `iterate_hanging(forest::ForestBWG{dim})`**
      (2D+3D): intra descent + inter-tree face neighbours via `transform_facet`
      (BWG2011 Alg 8, handles rotations). Only FACE neighbours are needed even at
      tree boundaries (every hanging node is face-interior; the 4 cells around an
      edge cycle through faces → a hanging edge always borders a refined–coarse face
      pair). **Matches `creategrid`'s `conformity_info` on all 16 golden cases**
      (2D/3D, multi-tree, rotated, balanced, disc). Inter-tree part is per-boundary-leaf
      transform for now (not a coordinated descent) — correctness first.
- [x] **`creategrid_iterator`** — full iterator materialization (2D+3D): one
      `iterate_leaves` descent per tree, node identity = physical coordinate (shared
      corners merge across trees automatically, no inter-octree merge), `iterate_hanging`
      for constraints, `reconstruct_facetsets` for facetsets. Same mesh as `creategrid`
      on all 16 golden cases; ~2× faster (~17× vs baseline). Kept beside `creategrid`.
- [x] **Min-Morton ownership numbering** (IBWG2015 §6): the descent visits leaves in
      Morton order, so "first encounter assigns the id" *is* min-Morton ownership. Dedup
      keys on the octant's exact **integer** corner coordinate (`local2id`, reused per
      tree) → each unique node transformed once (~7× fewer `_transform_point`). The
      physical-coord Dict (`coord2id`) shrinks to only cross-tree-shareable nodes (tree
      boundary + hanging endpoints); interior nodes get ids directly. 3D 90112: 125 ms /
      62.6 MiB → **90 ms / 47.6 MiB**, byte-identical. NOTE: not literally Dict-free — a
      small `coord2id` (boundary+hanging) + a reused integer `local2id` remain; a fully
      Dict-free form needs topology-based inter-tree corner coordination (deferred — low
      payoff). `iterate_hanging` (24 MiB) is now the dominant iterator alloc, not numbering.
- [x] **End-to-end verification** (the "example working" goal): a 3D adaptive Poisson
      solve materializing the *same forest* with both `creategrid` and `creategrid_iterator`
      yields bit-identical FE systems (same `ndofs`, same `#constraints`, `max|Δu|=0` at every
      node coordinate) over 4 adaptive steps, with monotone L2-error reduction. The published
      `heat_adaptivity.jl` tutorial also runs to completion with `creategrid_iterator` swapped
      in (ZZ + Dörfler loop, VTK output) — error converges as with the legacy path.
- [x] Make the iterator the default `creategrid` and **delete** the legacy multi-pass
      `creategrid` + `hangingnodes`. The physical-coordinate identity (coord→id bridge,
      `round(…;digits=10)`) is gone everywhere — node identity is now purely integer
      `(tree,coord)` + topological cross-tree merge (`_merge_cross_tree_nodes!`). This also
      removed a latent round-off bug present in *both* old paths (false-merge on graded
      meshes / false-split on rotated non-affine ones). Validated renumbering-invariant vs
      the frozen golden refs (16 cases) + 1643 invariants + `test_p4est.jl`. Perf on the
      90k-leaf 3D case: ~83 ms / 56 MiB (function-barrier `_build_cells` removed ~720k boxed
      cell allocs: 141 ms / 94.7 MiB / 1.2M → 83 ms / 56 MiB / 33k).
- [x] `heat_adaptivity.jl` tutorial uses the iterator path — it calls `creategrid`, which
      *is* the iterator now (no tutorial change needed).
- [ ] (Optional) expose `dim(c)`-filtered volume/face/edge/corner callbacks (the literal
      point-centric `Iterate_interior`/`Iterate`, Alg 5.2/5.3) as a thin API for downstream
      features. Not required for the serial node-numbering use case the iterator now covers.

---

## Follow-ups (post-iterator, from the PR description)

- [ ] DofHandler integration; solution transfer between dof handlers.
- [ ] Arbitrary polynomial order.
- [ ] Error estimators (faster local ZZ, Kelly), Dörfler marking.
- [ ] Conformization.
- [ ] `balanceforest!`: paper-faithful single schedule/response round; complete
      3D corner balancing (TODO `BWG.jl:1033`).
- [ ] Revisit `transform_facet/edge/corner` empirical orientation logic (the
      "TODO understand this" spots) — derive vs the BWG2011 transforms.

---

## Bug fixes (done 2026-06-15)

- [x] `search` (`BWG.jl:399/403`): recurse guard was inverted vs IBWG2015 Alg 3.1
      (recursed on empty match set) and dropped the `Match` callback. Fixed +
      verified the descent reaches every leaf. (Dead code, but a trap for the iterator work.)
- [x] `match` (`BWG.jl:420`): removed leftover `println` debug output.
- [x] `boundaryset` docstrings (`BWG.jl:211/230`): cited "Fig.4.1"; it is **Figure 3** (IBWG2015 C509).
- [x] `isless` (`BWG.jl:113`): the "potential bug" TODO was a **false alarm** —
      `morton(o,o.l,o.l)` is the level-independent anchor interleave; verified the
      Morton sort + `searchsortedfirst` are correct in 2D/3D. Comment clarified, logic unchanged.

### Known issues not yet fixed
- `isancestor` (`BWG.jl:1186`): loop bound `while l > 0` never tests the level-0
  root as an ancestor — incomplete, though it doesn't bite current callers
  (`linearise!` never sees the root as a non-sole leaf). Verify before relying on it.
- `transform_facet/edge/corner` orientation is empirical; `balanceforest!` 3D
  corner balancing incomplete (see Follow-ups).
