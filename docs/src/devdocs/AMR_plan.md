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

- [ ] Verify the Morton-sort prerequisite holds for all leaf arrays (done for the
      sampled cases; make it an invariant check).
- [ ] `Iterate_interior` (Alg 5.2) — point-centric recursion over `e ∈ part(c)`,
      reusing `split_array`/`ancestor_id`/`children`/`descendants`/`boundaryset`.
      **Do not** start from `search` (different algorithm).
- [ ] `Iterate` (Alg 5.3) driver — seed from tree-root closures. Validate the
      visited point set + supports against `creategrid`.
- [ ] `Lnodes_callback` (no paper pseudocode — design work): owner = min leaf in
      `leaf_supp(c)`, node ids + connectivity. Validate vs `creategrid` (linear elems).
- [ ] Hanging constraints via the remote-reference test (eq 6.1, `dim(c)<d-1`).
      Validate vs `hangingnodes`.
- [ ] Retire the old multi-pass cost centers; make iterator the default.
- [ ] (Optional) expose `dim(c)`-filtered volume/face/edge/corner callbacks as a
      thin API for downstream features.

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
