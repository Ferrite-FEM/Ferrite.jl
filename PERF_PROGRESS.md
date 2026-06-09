# Ferrite.jl performance work — progress log

Goal: find & implement performance improvements, guided by benchmarks, with
before/after numbers. Focus areas (from user): dof distribution (`close!`),
assembly into sparse matrix (`_assemble_inner!`), applying BCs (`apply!`),
and the hot loops (`reinit!` → `calculate_mapping` + `apply_mapping!`).

## Status: DONE — 2 verified wins landed (uncommitted, on master)

### Headline results (clean same-session A/B, julia 1.12.6 -O3)
| Operation | case | before | after | speedup |
|-----------|------|--------|-------|---------|
| scatter `assemble!` | 3D Hex Q2 (9261) | 1.514 ms | 1.357 ms | 1.10x |
| scatter `assemble!` | 3D Tet P1 vec (12288) | 7.878 ms | 6.947 ms | 1.13x |
| scatter `assemble!` | 2D Quad Q1 (10201) | 590 µs | 493 µs | 1.20x |
| `allocate_matrix(dh,ch)` | 3D Hex Q2 | 6.503 ms | 0.997 ms | 6.5x |
| `allocate_matrix(dh,ch)` | 3D Tet P1 vec | 16.30 ms | 2.70 ms | 6.0x |
| `allocate_matrix(dh,ch)` | 2D Quad Q1 | 1.870 ms | 0.327 ms | 5.7x |

### Correctness (all PASS, against modified code)
- `test_assemble.jl` + `test_sparsity_patterns.jl`: 6,352,316 assertions PASS
- `test_constraints.jl`: 14,477 PASS
- CSR vs CSC assembly match: PASS (3 element types)
- Full heat solve via `allocate_matrix(dh,ch)`: norm == 3.307743912641305 exactly
- fast(dh,ch) pattern == slow(dh,ch) pattern for pure Dirichlet: PASS (5 elt types)
- affine/periodic correctly routed to slow path: PASS

Benchmark kept at `benchmark/bench_perf_changes.jl` (run, then `git stash` src for before).

---
## Status history: BASELINE in progress

## Benchmark harness
- `bench_baseline.jl` — realistic assembly benchmarks (4 cases) + sub-component
  timings (reinit, full assembly, sparsity, dof close, apply!).

## Cases
1. 3D Hex Q2 scalar heat (1000 cells)
2. 3D Tet P1 vector elasticity (~3375*6 cells)
3. 2D Quad Q1 scalar heat (10000 cells)
4. 3D Hex Q2 vector elasticity (512 cells)

## Baseline numbers (julia 1.12.6, -O3, this machine)

| Case | reinit | full asm | sparsity | dof close | apply! |
|------|--------|----------|----------|-----------|--------|
| 1: 3D Hex Q2 scalar (1000 cells, 9261 dof) | 2.02 ms | 29.7 ms | 1.02 ms (13.3 MiB) | 228 µs (1.70 MiB) | 421 µs |
| 2: 3D Tet P1 vector (20250 cells, 12288 dof) | 2.26 ms | 26.5 ms | 2.80 ms (16.1 MiB) | — | — |
| 3: 2D Quad Q1 scalar (10000 cells, 10201 dof) | 272 µs | 1.15 ms | 334 µs (3.44 MiB) | — | 274 µs dofclose |
| 4: 3D Hex Q2 vector (512 cells, 14739 dof) | 4.15 ms | 299.7 ms | (running) | — | — |

Notes:
- reinit-only is allocation-free (good).
- full asm dominated by user element arithmetic (esp. Case 4: 81x81 sym-grad).
- sparsity allocates a lot (13-16 MiB).
- apply! 421µs for 9261 dofs is notable for transient/nonlinear (called every step).

## Granular numbers (Case 1, 3D Hex Q2 scalar, 9261 dof, nnz=531441)
- scatter assemble!: 1.50 ms / 1000 cells (~5% of full asm; rest is user element arith)
- apply! 452µs = meandiag 52 + zero_cols 10 + **zero_rows 278** + add_inhomog 0.3 + final-loop ~110
- zero_out_rows! dominates apply! but is O(nnz) streaming rowval (~4MB) -> memory-bound.

## Findings / ruled-out
- [RULED OUT] reinit double det(J): MEASURED — LLVM already CSEs it.
  det+inv separate = 2.42ns ≈ inv alone 2.38ns. reinit_B (det-from-inv) was SLOWER. No win.
- [profile] reinit! hot spot = apply_mapping! store into dNdx (_setindex! 169 vs dot 51).
  Store-bound, near-optimal. No easy win.
- [profile] scatter hot spot = merge loop in _assemble_inner! (sort negligible). Tight.
- allocate_matrix(dh) uses FastSparsityPattern fast-path already; 13MiB is mostly required
  rowval+nzval (~8.5MB) + one-time work buffers.

## Workflow results: 42 raw -> 16 confirmed candidates (with my skepticism)
HIGH/real:
- [DONE-ish] assembler merge loop: use sequential sortedrowdofs[ri] instead of gather
  rowdofs[rowpermutation[ri]] + defer Ke load. (MY find, not workflow). Measured win.
- [TODO] sparsity SLOW path (allocate_matrix(dh,ch), Symmetric): O(m^2) insert_sorted +
  redundant sp.rows[row] extract/store in _add_cell_entries! inner loop (sparsity_pattern.jl
  :500-513). Maintainer TODO confirms. Only affects SparsityPattern path (not Fast).
- [SKEPTICAL] maxcelldofs_hint=0 -> resize! allocs. Workflow claimed "26.7x"; my baseline
  shows only 13 allocs TOTAL across 1000 cells -> claim is bogus. Verify, likely tiny.
MEDIUM:
- _add_constraint_entries! redundant row extract (constraint path).
- DofHandler close!: vertexdicts allocates zeros(nnodes) per field even w/o vertex dofs.
- Lagrange Hex2 gradient via ForwardDiff in precompute (one-time, low real impact).
LOW/negligible (deprioritized): findfirst in _condense_local, ndofs_per_cell lookup,
  sym branch in _allocate_matrix, edge/facedict sizehint, _apply_local dict lookups,
  @inbounds on cleanup loop, CellCache dof sizing, MultiField redundant J (high risk).

## Changes made (all verified correct + benchmarked)

### 1. Assembler merge loop (src/assembler.jl `_assemble_inner!`, CSC)
Use sequential `sortedrowdofs[ri]` (cache-friendly) instead of the gather
`rowdofs[rowpermutation[ri]]`, and defer the `Ke[...]` load to only the branches
that use it. Provably equivalent (sortedrowdofs[ri] == rowdofs[rowpermutation[ri]]).
- Scatter 3D Hex Q2: BEFORE 1.561 ms -> AFTER 1.444 ms (clean A/B, ~7.5%).
  (earlier runs showed up to ~13%; ~8-10% typical.)
- Tet P1: 752 -> 698 µs. Quad Q1: ~3%.
- Correctness PASS (CSC + Symmetric assemblers, 3 element types).
Also applied the same transform to the CSR assembler (ext/FerriteSparseMatrixCSR.jl)
using sequential `sortedcoldofs[ci]`.

### 2. Sparsity fast-path for constrained problems (src/Dofs/sparsity_pattern.jl)
`_can_use_fastsp` now accepts a ConstraintHandler that has NO non-trivial affine
constraints (pure Dirichlet) when keep_constrained=true & no coupling/topology.
Such a ch adds no entries vs the fast path, so the pattern is identical. This routes
the very common `allocate_matrix(dh, ch)` (pure Dirichlet) to the 1ms fast path
instead of the 7ms slow O(m^2)-insert SparsityPattern path.
- allocate_matrix(dh,ch): 3D Hex Q2 7.20->1.11 ms (6.5x), 3D Tet P1 vec 18.15->2.66 ms
  (6.8x), 2D Quad Q1 2.03->0.33 ms (6.2x). Allocs 20.6->13.3 MiB.
- Full heat solve through allocate_matrix(dh,ch): norm matches exactly (PASS).
- Pattern fast(dh,ch) == slow(dh,ch) for pure Dirichlet: PASS (5 element types).
- Affine/periodic correctly still routed to slow path (has_affine guard): PASS.

## Ruled out / not pursued
- maxcelldofs_hint default: workflow claimed 26.7x; measured only 13 allocs TOTAL/1000
  cells -> resize! is once-per-assembler, negligible. Skipped.
- reinit double det: LLVM CSEs it (measured). reinit/scatter hot loops already optimal.
- apply! zero_out_rows: dominant but O(nnz) memory-bound; no clean win.
