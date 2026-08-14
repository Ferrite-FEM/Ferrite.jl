# WorkStream / `mesh_loop` exploration — findings summary

*Status as of 2026-08-14, branch `fe/workstream`. Exploration of a deal.II-style
WorkStream API for threaded assembly (`src/WorkStream.jl`), compared against the
approaches in `docs/src/literate-howto/threaded_assembly.jl` (OhMyThreads colored loop
and atomic assemblers).*

## Code state

- `src/WorkStream.jl`: rewritten (see "Improvements" below). Two methods of `mesh_loop`:
  colored (worker+copier inline per task) and uncolored pipeline (parallel workers, one
  ordered copier task).
- `workstream_bench.jl` (repo root): benchmark driver. Modes via env vars:
  plain run (variant comparison), `WS_SWEEP` (ntasks sweep for the pipeline),
  `ATOMIC_CMP` (atomic vs non-atomic serial), `COLOR_ORDER` (traversal-order locality).
  Run with `julia --project --threads=N workstream_bench.jl [n]`.
- `threaded_assembly.jl` (repo root): old experiment driver; calls `doassemble` which is
  commented out, so it errors as-is.
- `Project.toml` gained ChunkSplitters/OhMyThreads/TaskLocalValues/StaticArrays deps
  (StaticArrays is now unused; no compat bounds yet). Manifest resolved.

## Bugs found in the original sketch

1. **ChunkSplitters v2 → v3 API break**: `chunks(x; size)` used to yield *index ranges*,
   v3 yields *views of the collection*. The original `cid = color[idx]` double-indexed:
   out-of-bounds crash for small colors, **silently wrong cells assembled** for large
   ones. The uncolored variant only worked by accident (`color = 1:ncells` ⇒
   `color[idx] == idx`). Fixed; add compat bounds to prevent recurrence.
2. Original uncolored pipeline did per-cell `take!`/`put!` on two shared channels with
   `ntasks = nthreads()` workers + copier + producer: catastrophic at 8 threads
   (775 ms vs 117 ms serial), and non-deterministic (unordered copier).
3. Task-local values + fresh tasks per color ⇒ `ntasks × ncolors` scratch copies
   (128 CellValues copies at 8 tasks / 16 colors). The howto has the same flaw.
4. Error paths leaked tasks (producer blocked on `put!` forever; copier task leaked on
   worker failure).

## Improvements made

**Colored variant**: static `ChunkSplitters` split (default one chunk per task; passing
`chunk_size` gives load balancing with one short-lived task per chunk); bounded resource
pool (`Channel` of `(scratch, copy_data, cell_cache)`) caps copies at `ntasks` regardless
of color count; plain `@sync`, `try/finally` returns resources on error.

**Uncolored pipeline** (the deal.II design done properly):
- Batched handoff: copy data circulates in per-chunk batches (default `chunk_size = 64`),
  amortizing channel operations over the chunk.
- `ntasks = nthreads() - 1` default leaves a thread for the copier.
- **Ordered copier**: chunks carry a sequence number; the copier processes them in order
  via a reorder buffer (deal.II's `serial_in_order` filter). Result is **bitwise
  identical to a serial loop**, any thread count.
- Deadlock subtlety: workers must check out a batch *before* taking a chunk, else the
  worker holding the next-in-sequence chunk can starve (all batches parked in the
  reorder buffer waiting for exactly that chunk). Comment in the code.
- Channels are `bind`-ed/closed on all exit paths so failures propagate instead of hang.

## Benchmark results

Linear elasticity, trilinear hexes, 17,280 cells / 61,347 dofs, 16 colors, best of 3,
aarch64 (linuxkit), Julia 1.12.6. "omt" = howto approach.

| variant                  | 1 thread | 4 threads | 8 threads |
|--------------------------|---------:|----------:|----------:|
| serial                   |   116 ms |         — |         — |
| omt colored (howto)      |   158 ms |     55 ms |     53 ms |
| omt atomic (howto)       |   125 ms |     33 ms | **32 ms** |
| ws colored               |   154 ms |     46 ms |     52 ms |
| ws colored, chunk=8      |   166 ms |     47 ms | **41 ms** |
| ws uncolored (pipeline)  |   118 ms | **35 ms** |     54 ms |

- Pipeline: 775 → 54 ms after the rewrite; at 4 threads it beats every colored variant
  while being exactly deterministic with no coloring. Plateaus at high thread counts
  (Amdahl on the serial copier ≈ total `assemble!` scatter time). Only ~2% overhead at
  1 thread.
- Coloring setup itself costs ~0.1 s (≈ one assembly).

### Atomic vs non-atomic (serial, isolating scatter)

| | non-atomic | atomic | overhead |
|---|---:|---:|---:|
| full assembly | 113.8 ms | 124.3 ms | +9% |
| scatter only (`assemble!`) | 21.9 ms | 31.6 ms | +44% |

Atomic float add is a CAS/LL-SC loop: +44% on the scatter, +9% overall for this cheap
kernel (near worst case; expensive kernels → noise).

### Why atomic beats coloring (the key measurement)

Identical serial work, single thread, non-atomic — only the traversal order differs:

| traversal | time |
|---|---:|
| natural order | 115.2 ms |
| color order   | 150.6 ms (**+31%**) |

Coloring's defining property (same-color cells never adjacent) destroys cache locality:
consecutive cells share no nodes/dofs/K-columns, the working set (K ≈ 40 MB) is far
beyond cache, and each assembly sweeps the mesh once per color (16×) instead of once.
Coloring therefore parallelizes ~151 ms of serial-equivalent work across 16 barrier-
separated regions; atomic parallelizes 124 ms in one region. Locality tax (+31%) >
atomic tax (+9%). This is intrinsic — the Turcksin et al. partition-then-color algorithm
mitigates it, it cannot eliminate it. The pipeline pays *neither* tax (natural order,
plain adds); its only tax is the serialized copier.

## Conclusions

1. **For colored/atomic strategies, `mesh_loop` and a tuned howto loop converge to the
   same execution** (confirmed: timings within noise). The worker/copier split collapses
   to an inline call there; differences (scratch pooling, load balancing) are adoptable
   bookkeeping. The pipeline is the only thing the howto pattern *cannot* express.
2. **The case for `mesh_loop` is the API, not performance**: the howto is a copy-paste
   pattern whose footguns (`fillzero = false`, per-task assemblers, scratch sharing) are
   exactly what users get wrong. Contract: implement `cell_worker(cc, scratch, copy_data)`
   + `copier(copy_data)` + the two data types; the library owns tasking, duplication,
   chunking, ordering, and the safety reasoning. Strategy becomes a parameter
   (colored / pipeline / atomic) with unchanged user code — same shape as deal.II's
   `WorkStream::run`, which is *the* standard way deal.II assembly is written.
3. **Pipeline niche**: deterministic (bitwise = serial), no coloring prerequisite, serial
   commit hook (safe for non-thread-safe reductions: output, global quantities, later
   MPI), works for any eltype. Atomic is the raw-speed winner but non-deterministic and
   restricted to Float16/32/64 (+ complex). Suggested default: pipeline; `atomic` as
   documented speed opt-in; colored mostly legacy.
4. **Atomic should not be the `start_assemble` default**: common (serial) case pays +9%
   for nothing; generic eltypes (Dual, BigFloat) unsupported ⇒ can't be the generic
   default; and it wouldn't deliver safe-by-default anyway (sharing one assembler across
   tasks is still a race on its permutation buffers). Auto-picking atomic *inside*
   `mesh_loop` when the eltype allows is reasonable.
5. **Boundary/interface workers** (deferred): not supported — `mesh_loop` is cell-only.
   Boundary workers are easy (own-cell dofs only, existing coloring suffices). Interface
   workers break colored mode (writes to both neighbors' dofs need distance-2 coloring in
   the face-adjacency graph; `create_coloring` would need an interface-aware mode + a
   visit-once ownership rule, cf. `InterfaceIterator`). The pipeline gets interfaces
   almost for free (all shared-state writes in the copier) — the strongest future
   argument for it, since hand-rolled safe parallel DG assembly is very hard.

## Suggested next steps

- Decide on pursuing a unified `mesh_loop(dh, worker, copier, ...; strategy)` API;
  implement colored/atomic as thin OhMyThreads wrappers (copier inlined), keep the
  hand-rolled channel machinery only for the pipeline.
- API refinement: factory closures (`() -> Scratch(...)`) instead of sample +
  `Base.copy`.
- Scope: cellsets/`SubDofHandler` support; later facet/interface workers (see above).
- Housekeeping: compat bounds for new deps; drop unused StaticArrays dep; tests for
  `mesh_loop` (incl. determinism `==` serial, error propagation, chunk edge cases);
  delete or fix stale root `threaded_assembly.jl`.
- Re-benchmark on real hardware (this was 8 cores under docker/linuxkit) and with an
  expensive kernel to see the strategies converge.
