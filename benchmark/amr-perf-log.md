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
