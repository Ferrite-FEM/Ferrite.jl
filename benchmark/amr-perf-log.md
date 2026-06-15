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

<!-- Append: ### <date> — <change> (commit), with the same tables or a delta vs baseline. -->

### 2026-06-15 — Bug fixes only (no perf-relevant change expected)
Fixed `search` recurse-guard inversion + dropped callback, `boundaryset`
docstring (Fig 3), removed `match` debug prints, clarified `isless` comment.
None of these are on a hot path — baseline numbers above remain the reference.
