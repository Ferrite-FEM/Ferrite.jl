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

## Tier 1 — Safety net + quick wins (NEXT)

Goal: 20–100× on the dominant 3D path with zero behaviour change.

**1a. Golden-output regression tests** (prerequisite — current `test_p4est.jl`
only asserts *counts*):
- [ ] Pin full structure (cell connectivity, node coords, complete hanging-node
      map, facetsets) for a 2D/3D × multi-tree × refinement-pattern matrix.
- [ ] Make it a reusable oracle (`@test grid_new == grid_ref` style) every
      optimization runs against.

**1b. Quick-fixes** (each verified byte-identical vs 1a, then re-benchmark):
- [ ] `hangingnodes`: `findfirst(==(x), tree.leaves)` → `searchsortedfirst`
      (leaves are Morton-sorted — verified). Biggest 3D win (Phase 5 = 78% of 3D creategrid).
- [ ] `balance_face/edge/corner`: replace `∉ neighbor_tree.leaves` linear scans
      with binary search.
- [ ] `refine!`/`coarsen!`: replace the `findfirst` leaf lookup with binary search.
- [ ] `creategrid` Phase 4: run `transform_pointBWG` on **unique** nodes only
      (3D: 103 ms wasted on pre-dedup duplicates).
- [ ] `balancetree`: Set-based parent check instead of `p ∉ parent.(T,(b,))` (~10%).
- [ ] `balanceforest!`: dirty-tree tracking — only re-balance changed trees.
- [ ] Re-run benchmark; record before/after in the perf log.

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
