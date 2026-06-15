# AMR Performance Log

Living record of AMR (`ForestBWG`) performance as we optimize. Append a new
"Run" section after each meaningful change; keep the **Baseline** untouched.

**Reproduce:**
```bash
julia --project --startup-file=no benchmark/benchmarks-amr.jl
```
Component profiling: `profile_creategrid(forest)`, `profile_balancetree(forest)`,
`profile_balanceforest(forest)`, `identify_hotspots(Val(3); n=4, levels=2)`.

---

## Baseline

- **Machine:** 12th Gen Intel Core i9-12900K
- **Julia:** 1.12.6
- **Commit:** `c82281999` (branch `mk/p4est`)
- **Date:** 2026-06-15
- Times are `minimum` over 5 runs. Full pipeline = create → uniform refine →
  adaptive refine (25%) → balance(per-tree) → balance(forest) → creategrid.

### 2D full pipeline (`max_level=8`)

| n | lvls | #balanced | #hang | bal_tree | bal_forest | creategrid | TOTAL |
|---|------|-----------|-------|----------|------------|------------|-------|
| 8 | 3 | 7168 | 64 | 16.19 ms | 52.58 ms | 16.00 ms | ~86 ms |
| 16 | 2 | 7168 | 64 | 11.19 ms | 26.00 ms | 16.89 ms | ~56 ms |
| 16 | 3 | 28672 | 128 | 61.90 ms | 106.83 ms | 212.98 ms | **409.25 ms** |

Largest 2D case (n=16, lvl3, 28672 cells): creategrid **52.0%**, bal_forest
26.1%, bal_tree 15.1%, adaptive 5.4%.

### 3D full pipeline (`max_level=8`)

| n | lvls | #balanced | #hang | bal_tree | bal_forest | creategrid | TOTAL |
|---|------|-----------|-------|----------|------------|------------|-------|
| 2 | 3 | 11264 | 800 | 62.33 ms | 135.79 ms | 575.33 ms | ~777 ms |
| 4 | 3 | 90112 | 3136 | 588.03 ms | 1.136 s | 5.066 s | ~6.83 s |
| 8 | 2 | 90112 | 3136 | 263.39 ms | 1.189 s | 4.613 s | **6.090 s** |

Largest 3D case (n=8, lvl2, 90112 cells): creategrid **75.7%**, bal_forest
19.5%, bal_tree 4.3%.

### `creategrid` phase breakdown

2D (n=8, lvl3, 7168 cells), TOTAL 26.34 ms:
| Phase | Time | Note |
|-------|------|------|
| 1 intra-octree nodes | 3.22 ms | 28672 entries → 8512 keys |
| 2 inter-octree merge | **11.96 ms** | dominant in 2D |
| 3 deduplication | 6.05 ms | |
| 4 coord transform | 1.47 ms | |
| 5 hanging nodes | 3.50 ms | 64 constraints |
| 6 facetset reconstr | 0.14 ms | |

3D (n=4, lvl2, 11264 cells), TOTAL 694.89 ms:
| Phase | Time | Note |
|-------|------|------|
| 1 intra-octree nodes | 10.98 ms | 90112 entries → 17664 keys |
| 2 inter-octree merge | 29.12 ms | |
| 3 deduplication | 7.77 ms | |
| 4 coord transform | 102.64 ms | `transform_pointBWG` on pre-dedup nodes |
| 5 hanging nodes | **541.78 ms** | **78% of creategrid**, only 800 constraints |
| 6 facetset reconstr | 2.61 ms | |

### `balancetree` profile (single tree)

| Op | 2D (256 leaves) | 3D (512 leaves) |
|----|-----------------|-----------------|
| Sorting Q | 8.9% | 5.6% |
| Parent checks `p ∉ parent.(T,b)` | 10.9% (8823 cmp) | 9.6% (16863 cmp) |
| Neighbors+siblings | 37.9% | 47.7% |
| Final sort | 38.0% | 34.6% |
| Linearise | 1.2% | 0.9% |

`balanceforest!` converges in 1 outer iteration for these cases. In 3D, the
per-leaf `possibleneighbors`+`inside` scan (12.67 ms + 35.67 ms over 11264
leaves) is non-trivial.

**Takeaways guiding the quick-fixes:**
- **3D is creategrid-bound, and creategrid is `hangingnodes`-bound** (78%) — the
  `findfirst` linear scans on `tree.leaves` are the target (→ `searchsortedfirst`).
- **2D creategrid is spread**: Phase 2 (inter-octree merge) + Phase 3 (dedup)
  dominate; less of a single hotspot.
- **`transform_pointBWG` (Phase 4, 3D 103 ms)** runs on all pre-dedup node
  entries — should run on unique nodes only.
- **balancetree** cost is mostly the two `sort!`s and neighbor allocation; the
  O(n²) parent check is ~10% (Set fix helps but isn't the main cost).

---

## Run log

### 2026-06-15 — balanceforest! allocation 2.0 GiB → 450 MiB (type-stable possibleneighbors)
`balanceforest!` allocated ~2 GiB (on an already-balanced forest that adds 0 cells — pure
overhead). The allocation profiler (`Profile.Allocs`) pinned it, after a couple of false
starts, to one root cause: **`possibleneighbors` returned `Union{NTuple{26}, Vector}`** —
the `insidetree` kwarg returned the raw tuple when `false` but a `filter`ed `Vector` when
`true`. That Union poisoned inference of the per-neighbour balance loop, boxing every
element AND heap-allocating the 26-tuple per leaf (~1.5 GiB across the loop). Fix: drop the
kwarg, always return the `NTuple` (`possibleneighbors` now 0 B/call); `balancetree` filters
inline with `inside`. Supporting fixes: extracted the per-leaf inter-tree work into a typed
`_balance_leaf!` (concrete args); `rootedges = dim==3 ? edges : nothing` → `_rootedges(Val)`
(killed a `Union{Nothing,NTuple}`); removed the dead `o′ = transform_*(…)` in
`balance_corner/face/edge`; `transform_edge`'s `zeros(T2,3)` → non-allocating ntuple scatter.

3D 90112 (n8 l2): **balanceforest! 2017 → 450 MiB, 534 → 284 ms**; balancetree 165 → 69 MiB.
Byte-identical (golden 16 + invariants 1643 + e2e solve); `test_p4est.jl` Balancing green.
Remaining 450 MiB is distributed (balancetree rebuild ~69, inter-loop + `_sort_by_morton!`
+ `refine!` the rest) — no single hotspot.

### 2026-06-15 — iterate_hanging allocation (per-call hot-path fixes)
`iterate_hanging` was the dominant `creategrid_iterator` alloc (24 MiB). Profiling the
sub-parts: intra descent 0.04 MiB, `leafsets` 5.77 MiB, and ~18 MiB in the inter-tree
per-leaf scan. The scan itself (`face`/`contains_facet`) is 0-alloc in a typed function;
the cost was per-call heap allocs in the functions it calls for every boundary leaf:
- `compute_face_orientation` built **two Vector comprehensions** (`nodes_f`, `nodes_f′`)
  per call → `ntuple`. It depends only on `(k,f)` but runs per-leaf; this was ~11 MiB.
- `transform_facet` (3D) allocated `xyz = zeros(T2, 3)` per call → non-allocating
  `ntuple` scatter (`a` is a permutation of (0,1,2); the old commented tuple form
  *gathered* instead of scattering — wrong unless `a` is an involution).
- `face`/`edge` built a `SubArray` via `view(𝒱₃, f, :)` → direct `𝒱₃[f, i]` indexing.

3D 90112: `iterate_hanging` **24.1 → 7.25 MiB**, `creategrid_iterator` **47.6 → 30.7 MiB**
(time ~89 ms). Byte-identical (golden 16 + invariants 1643 + e2e solve); `test_p4est.jl`
green (these functions are also used by `balanceforest!`/legacy `creategrid`). Residual
`iterate_hanging` alloc is now mostly `leafsets` (5.77 MiB, `Set` of all leaves for O(1)
membership) + a small `transform_facet` remainder.

### 2026-06-15 — min-Morton ownership numbering (Dict-free-ish creategrid_iterator)
`creategrid_iterator` node numbering rewritten to **min-Morton ownership** (IBWG2015 §6):
the descent visits leaves in Morton order, so "first encounter assigns the id" *is*
min-Morton ownership. Dedup now keys on the octant's exact **integer** corner coordinate
(`local2id`, one Dict reused across trees via `empty!`) → each unique node is transformed
to physical space **once** instead of once per touching leaf (~7× fewer `_transform_point`
calls). The physical-coord Dict (`coord2id`) shrinks to only the nodes that can be shared
across trees — tree-boundary corners + hanging-constraint endpoints; interior nodes get
ids directly. Per-tree work moved into `_materialize_tree!`/`_owner_id!` so the `do`-block
closures capture stable function args, not boxed `for`-loop vars; `sizehint!` on
`conns`/`nodecoords`. Byte-identical (all 16 golden cases; end-to-end solve identical).

3D 90112 (n8 l2), best of 7 + `@allocated`:

| | time | alloc |
|--|------|-------|
| `creategrid_iterator` (coord-keyed, prev) | 125 ms | 62.6 MiB |
| `creategrid_iterator` (ownership, now)    | **90 ms** | **47.6 MiB** |

Remaining alloc breakdown: **`iterate_hanging` 24.1 MiB (now the dominant 44%)**,
inherent output (nodes 2.2 + cells 5.5 MiB + Vector growth ~15 MiB), descent bookkeeping
(coord2id/local2id/closures) the rest. Next lever is `iterate_hanging` (per-tree leaf
`Set`s + per-constraint `Vector`s + per-node `transform_pointBWG`), not the numbering.

### 2026-06-15 — creategrid_iterator (full iterator materialization)
`creategrid_iterator` (kept beside `creategrid`): one `iterate_leaves` descent per
tree, node identity = physical coordinate (shared corners across trees merge
automatically → no topology/transform inter-octree merge), `iterate_hanging` for
constraints, `reconstruct_facetsets` for facetsets. Same mesh as `creategrid`
(renumbering-invariant, all 16 golden cases). best of 7:

| cells | `creategrid` | `creategrid_iterator` |
|------|--------------|-----------------------|
| 2D 28672 | 53 ms / 87 MB | **19 ms / 57 MB** |
| 3D 11264 | 53 ms / 96 MB | **29 ms / 78 MB** |
| 3D 90112 | 570 ms / 709 MB | **278 ms / 585 MB** (~17× vs 4.6 s baseline) |

The one-pass descent + coord dedup beats the 5-phase Dict pipeline despite redundant
per-corner `transform_pointBWG` calls. Remaining iterator alloc/time: the redundant
transforms (shared corners transformed per touching leaf), the coord→id Dict, and the
descent closures — the Dict-free LNodes *ownership* numbering is the next step.



### 2026-06-15 — balanceforest! optimization (Tier 3 follow-on)
`balanceforest!` became the dominant 3D cost after the creategrid wiring. Profiling
the *real* function (the benchmark's `profile_balanceforest` reimplements the loop
+ uses the old `balancetree`, so it's misleading) found: sorting (isless→morton ×2
per compare) in `balancetree`, then a per-leaf `possibleneighbors`+`inside.(26)`+
`findall(.!)` over *every* leaf with runtime 26-tuple indexing. Fixes (all byte-identical):
- `balancetree`: sort Q/R by precomputed `(morton-anchor, level)` key (morton once
  per octant, not O(n log n) times) → `bal_tree` ~2.2–2.6×.
- `possibleneighbors` 3D: `ntuple(26)` → `ntuple(Val(26))` (was n>10 type-unstable+alloc).
- inter-tree loop: skip interior leaves (`_touches_tree_boundary`), iterate
  `enumerate(ss)` (no runtime tuple index), skip in-tree neighbours inline (no
  `findall` Vector), and drop the edge-branch `findall`.

| case | balanceforest! (baseline → now) | total (baseline → now) |
|------|---------------------------------|------------------------|
| 3D 90112 (n8 l2) | 1.19 s → **0.54 s** (~2.2×) | 6.09 s → **1.38 s** (~4.4×) |
| 3D 90112 (n4 l3) | 1.14 s → **0.43 s** (~2.6×) | 6.83 s → **1.26 s** (~5.4×) |
| 2D 28672 | 107 ms → 38 ms | 409 ms → **123 ms** |

`balanceforest!` and `creategrid` are now balanced (~0.5–0.6 s each on 90 k). Remaining
`balanceforest!` cost is distributed: `possibleneighbors` (builds 26 neighbours/boundary
leaf), `ExclusiveTopology` corner/edge `getindex` (Ferrite core), and the outer
fix-point re-running `balancetree` on all trees each iteration.



### 2026-06-15 — Iterator wired into creategrid (Tier 3)
`creategrid` Phase 5 now detects hanging nodes via the O(n) iterator
(`iterate_hanging`) instead of the old `hangingnodes`. The old function stayed
~422 ms even after Tier 1b (compute-bound: the 90k-iteration per-vertex×per-face
`iscenter` scan + Dict lookups); the descent does it in a few ms. Clean
full-pipeline, best of 3:

| case | creategrid (baseline → Tier1b → iterator) | total (baseline → iterator) |
|------|-------------------------------------------|-----------------------------|
| 2D 28672 | 213 ms → 92 ms → **55 ms** | 409 ms → **175 ms** |
| 3D 11264 | 745 ms → 571 ms → **61 ms** | — |
| 3D 90112 (n8 l2) | 4.61 s → ~4.7 s → **0.59 s** (~8×) | 6.09 s → **1.88 s** (~3.2×) |
| 3D 90112 (n4 l3) | 5.07 s → ~4.9 s → **0.55 s** (~9×) | 6.83 s → **1.82 s** |

`balanceforest!` (~1.1 s on 90112) is now the dominant 3D cost. `iterate_hanging`
itself is allocation-free (output Dict only); the remaining creategrid alloc/time
is the Dict-based node numbering (Phases 1–4), the target of the LNodes rewrite.
Output byte-identical (golden + invariants); conformity_info constraint sets
unchanged (constrainer order now sorted).



<!-- Append: ### <date> — <change> (commit), with the same tables or a delta vs baseline. -->

### 2026-06-15 — Bug fixes only (no perf-relevant change expected)
Fixed `search` recurse-guard inversion + dropped callback, `boundaryset`
docstring (Fig 3), removed `match` debug prints, clarified `isless` comment.
None of these are on a hot path — baseline numbers above remain the reference.

### 2026-06-15 — Tier 1b
Root cause of the dominant 3D cost was **allocation**, not the `findfirst` leaf
lookups the plan blamed. `hangingnodes` allocated 1.2 GB for 800 constraints:
`edges()` used `ntuple(f, 12)` (n>10 → Base's type-unstable generic path,
confirmed via `return_types`/`@allocated`; n≤10 const lengths were already
stable), and `parent_`/`parentfaces`/`parentedges` were rebuilt 4–8× per leaf.
Fix = `ntuple(f, Val(12))` in `edges` + hoist out of the vertex loop. Plus
`balancetree` buffer reuse + Set parent-check, `creategrid` Phase-4 transform on
unique nodes, a hash `Set` for O(1) leaf membership, and `children` rewritten to a
direct z-order computation using the concrete `OctantBWG{dim,N,T}` constructor (it
returned `NTuple{Union{OctantBWG{2,4},OctantBWG{3,8}}}` — type-unstable via the
runtime-`dim` morton-decode constructor; now concrete `NTuple{N,OctantBWG{dim,N,T}}`).

| case | metric | baseline | Tier 1b |
|------|--------|----------|---------|
| 3D 11264 | hangingnodes alloc | 1194 MB | **15 MB** |
| 3D 11264 | hangingnodes time (min/25) | ~541 ms | ~429 ms |
| 3D 11264 | creategrid alloc | 1274 MB | 94 MB |
| 2D 28672 | creategrid time | 213 ms | ~50 ms |
| 2D 28672 | balancetree (bal_tree) | 62 ms | ~50 ms |
| 3D 90112 | balancetree (bal_tree) | 263 ms | ~200 ms |
| 3D | children alloc / 100k calls | 205 MB | 38 MB (Union→concrete) |
| 3D 11264 | balanceforest! | 125 ms | 119 ms |

Allocation counts are deterministic and the reliable signal; 3D full-pipeline
wall-clock is noisy on a loaded machine. `hangingnodes` is now compute-bound
(Dict lookups + 90k-iteration vertex loop) — the iterator (Tier 3) is the next
lever there. Dropped idea: binary-search leaf membership (`searchsortedfirst`
rides on `isless→morton`, loses to cheap-`==` scans at realistic tree sizes).
