# Design Note: Implementing the IBWG2015 Node Iterator (LNodes)

> Working design note (not yet wired into the rendered docs). Replaces the
> current `creategrid`+`hangingnodes` (BWG2011 Algorithm 20 "Nodes" style) with
> the IBWG2015 node iterator. Every structural claim is grounded in a paper
> algorithm/page (IBWG2015 cited as **C-pages**). Code refs are to
> `src/Adaptivity/BWG.jl`.

**Sources:** Isaac, Burstedde, Wilcox, Ghattas, *Recursive Algorithms for
Distributed Forests of Octrees*, SIAM J. Sci. Comput. 37(5), 2015 (**IBWG2015**);
Burstedde, Wilcox, Ghattas, *p4est*, SIAM J. Sci. Comput. 33(3), 2011 (**BWG2011**).

> **Scope.** None of the `Iterate`/`Lnodes` machinery exists in `BWG.jl` yet
> (TODO at BWG.jl:593). Only `split_array` (Alg 3.3, BWG.jl:367) and `ancestor_id`
> (Alg 3.2, BWG.jl:1289) are present and faithful — those are the building blocks.

---

## 1. The one core property, and the exact stop condition

**The property** that makes the recursion fast:

> *"If we applied Ancestor_id to each octant in A, we would get a monotonic
> sequence of integers, so if we search A with the key i and use Ancestor_id to
> test equality, the lowest matching index will give the first descendant of
> child(a)[i] in A."* — IBWG2015 C507

Because the leaf array `A` is **Morton-sorted** (total order, Alg 2.1, C504:
*"ancestors precede descendants"*), the descendants of any fixed octant occupy a
**contiguous block** of `A`, and `Ancestor_id(A[j], a.l+1)` is **nondecreasing in
`j`**. That single fact turns "partition the leaves under `a` into its `2^d`
children" from a scan into one combined binary search (`split_array`, Alg 3.3).
The engine never sorts and never hashes during descent — it only narrows
contiguous ranges of an already-sorted array.

**The exact stop condition** of `Iterate_interior(c, S, …)` (Alg 5.2, C515):

1. **Hard base case (line 1):** `if ∪S ∩ O_p = ∅ then return` — no carried array
   holds a locally-owned leaf. (Serial: every leaf is owned, so this only prunes
   empty branches.)
2. **Point finalized (`stop ← true`), fire callback *instead of* recursing, when:**
   - `dim(c) > 0` branch: some support array is the singleton `S[i] == {s}` with
     `s = supp(c)[i]` (line 7 — purely structural, `s` is a leaf); **or**
   - `dim(c) == 0` branch: **always** (line 16).

Termination is guaranteed by the **level cap**: each recursive call descends one
octree level (`child` level `a.l+1`), bounded by `b` (`_maxlevel=[30,19]`,
BWG.jl:13). The `2^d` child subarrays from `split_array` **partition** the parent
array, bottoming out at empty (return) or singleton (leaf-stop). Leaf equality is
**total-order identity** (level+coords, `isequal`), not pointer identity.

---

## 2. The true structure: point-centric, NOT volume/face/edge/corner

The paper's abstraction is a single recursion keyed on a **point `c`** —
*"iterating in the interior of a point"* (§5.2). It is **not** the p4est C-API
mental model of four separate Volume/Face/Edge/Corner callbacks, and it does
**not** recurse into "adjacent child-pairs."

`Iterate_interior(point c, arrays S)`:
- `S[i]` = sorted array of all leaves descending from `supp(c)[i]` (the `i`-th
  support octant of `c`). **`S` is the load-bearing recursion state.**
- One callback per point `c`, **dispatched internally by `dim(c)`**: `dim=d` →
  volume, `0<dim<d` → face/edge, `dim=0` → corner.
- Descent (`stop=false`): recurse over **`e ∈ part(c)`** — the *child partition of
  the point* `c` (eq 2.7) — building each `S_e[i]` from the `split_array`
  subarrays (`S_e[i] = H_j[k]` such that `h_j[k] = supp(e)[i]`, line 24). So
  `split_array` is called once per support octant and its subarrays are reused.
- The only table needed in descent is the **child-boundary-intersection set
  `B_∩^i`** (Fig 3 / eq 4.5; code `boundaryset`, BWG.jl:214/233), used at line 14.
  There is **no** `CHILD_FACE_PAIRS`/edge/corner adjacency table in the algorithm.

**Reconcile with a p4est_iterate-style API:** the per-dimension
Volume/Face/Edge/Corner callbacks are a **supported specialization** (§5.4) — a
face-only callback recurses into `e ∈ part(c)` only when `dim(e) ≥ d-1`. **Build
the point-keyed engine first; expose `dim(c)`-filtered callbacks as a thin layer
on top.** Do not invert this.

`Iterate` (Alg 5.3, the driver) is thin: for each tree form `S^t = O_p^t ∪ ghosts`,
then for every point `c` in the closure of every tree root, seed
`U[i] = S^{supp(c)[i].t}` and call `Iterate_interior`. Seeding from root closures
is what makes inter-tree boundary points get visited exactly once.

---

## 3. How hanging nodes are detected — NOT in the descent

**Correction to the obvious-but-wrong model:** there is **no** "one side is a
leaf, the other internal → hanging for free from a level mismatch" mechanism. The
iterator only ever visits **non-hanging** points:

> *"P_Ω is the set of all points shown: note that because some points in clos(o)
> are hanging, they are not included."* — IBWG2015 C512

The leaf-detection test (`S[i]=={s}`) is purely structural — no level test, no
hanging test inside `Iterate_interior`.

Hanging/master–slave coupling is a **downstream LNodes concept**, the
*remote-reference* test (eq 6.1, C517), evaluated inside the **LNodes node-callback**
(Alg 6.2 line 3), keyed on points `c`, using `leaf_supp_p(c)`:

> *"a leaf o remotely references a point c … if c ∉ leaf supp(c) and there exists e
> such that o ∈ leaf supp(e) and c ∈ bound(e)"* — IBWG2015 C517, **only when `dim(c) < d-1`**.

Concretely: at a non-conformal interface the **larger (coarse) leaf owns the
global node at `c`**; the smaller leaf's element node is interpolated from the
coarse basis and introduces **no new DOF**. This is the constrained→constrainer
map that `hangingnodes` (BWG.jl:803) builds in a separate pass today; in the LNodes
design it is produced in the **same single traversal** as the node ids — that
fusion is the point of the rewrite. The paper gives **no pseudocode** for the
LNodes callback (C519) — this is design work, not transcription.

---

## 4. Symbol → algorithm → status

| `BWG.jl` symbol (line) | Paper algorithm | Status |
|---|---|---|
| `morton` (76) | BWG2011 Alg 3 / eq 2.3 | done |
| `Base.isless` (111) | IBWG2015 Alg 2.1 (single-tree) | done — Morton order verified (morton-arg is level-independent), see §6 |
| `child_id` (1272) | BWG2011 Alg 1 | done (1-based) |
| `ancestor_id` (1289) | IBWG2015 Alg 3.2 | done — monotone |
| `parent` (1301), `descendants` (1317) | BWG2011 Alg 2, 4 | done |
| `facet/edge/corner_neighbor` (1343/1735/1760) | BWG2011 Alg 5/6/7 | done |
| `split_array` (367) | IBWG2015 Alg 3.3 | done — `2^d` contiguous views; reusable |
| `boundaryset` (214/233) | Fig 3 / eq 4.5 (`B_∩^i`) | done — docstring correctly cites Figure 3 |
| `find_range_boundaries` (261) | IBWG2015 Alg 4.2 | done — ghost machinery, not the iterator |
| `transform_facet/edge/corner` (1508/1698/1612) | BWG2011 Alg 8/10/12 | partial — empirical orientation, see §6 |
| `balanceforest!` (967) | BWG2011 Alg 17 | partial — serial fixpoint, 3D corner balancing incomplete |
| `search` (390) | IBWG2015 Alg 3.1 | fixed — guard inversion + dropped callback corrected (§6); separate algo from the iterator |
| `isrelevant` (296) | IBWG2015 Alg 5.1 | stub `return true` (correct for serial) |
| `iterate_leaves` | IBWG2015 Alg 5.2 (volume specialization) | done — integer `split_bounds` descent, Morton order |
| `iterate_hanging` | eq 6.1 remote-reference (face-descent realization) | done — **integer** `(tree,coord)` keys, intra + inter-tree |
| `_merge_cross_tree_nodes!` | inter-tree point identity (φ^t maps + Alg 8/10/12 transforms) | done — integer/topological, no physical coords |
| `creategrid` / `_number_tree!` | IBWG2015 §6 LNodes; owner = min leaf supp (eq 6.2) | done — single-pass min-Morton numbering, the default materializer |
| `hangingnodes` | (was BWG2011 Alg 20 "Nodes" multi-pass) | **deleted** — replaced by `iterate_hanging` |
| legacy multi-pass `creategrid` | (was BWG2011 Alg 20) | **deleted** — replaced by the iterator above |
| `Iterate_interior` / `Iterate` (literal point-centric) | IBWG2015 Alg 5.2 / 5.3 | not implemented as-such — the volume + face-descent specialization above covers the serial node-numbering use case (see §3) |
| `Global_numbering` (6.1), `Determine_owner_process` (6.3), `Reconstruct_remote` (6.4) | Alg 6.1/6.3/6.4 | distributed-only; serial collapses to a contiguous counter + topological cross-tree merge |

---

## 5. Minimal build order (serial; current `creategrid` as oracle)

Single process collapses the distributed pieces: `is_relevant ≡ true`, owner ≡
`min`-leaf in `leaf_supp(c)` (the code's "lowest tree index `k`" rule,
BWG.jl:632/656/706), `Global_numbering ≡` contiguous counter, `Reconstruct_remote`
≡ skip.

1. **Verify the ordering prerequisite first** (everything depends on it): confirm
   each tree's `leaves` is Morton-sorted by Alg 2.1 and `ancestor_id` is monotone
   along it. See the `isless` concern, §6.
2. **Write `Iterate_interior` fresh against Alg 5.2.** Reuse only `split_array`,
   `ancestor_id`, `children`, `descendants`, `boundaryset`. **Do not** start from
   `search` (390). State = the `S` arrays; descend over `e ∈ part(c)`; leaf-stop =
   `S[i]=={supp(c)[i]}` / `dim(c)==0`.
3. **Write `Iterate` (Alg 5.3)** driver: per tree `S^t = leaves` (serial, no
   ghosts), seed from each tree-root closure. First with a trivial callback that
   collects `(c, leaf_supp(c))`; assert visited points + supports match what
   `creategrid` enumerates.
4. **Write `Lnodes_callback`** (design work, no paper pseudocode): per point `c`
   assign owner (`min` over `leaf_supp(c)`), emit one global id for non-remote
   points, fill connectivity. Validate node ids/connectivity against `creategrid`
   for linear elements — §7.3 (C523) proves Nodes and LNodes are essentially
   equivalent at `n=1`, so the resulting `NonConformingGrid` must be identical.
5. **Fold in hanging detection** via the remote-reference test (eq 6.1,
   `dim(c)<d-1`). Validate the constrained→constrainer map against `hangingnodes`.
6. **Delete the multi-pass cost centers** (Dict-of-tuples hashing BWG.jl:605-606,
   pairwise neighbor re-traversal Phase 2, `transform_pointBWG` on pre-dedup
   duplicates) once 4–5 match the oracle.

---

## 6. Code divergences that are latent bugs

1. **`search` (BWG.jl:399/403) — FIXED 2026-06-15.** The recurse guard was inverted
   (`isempty(idxset_match)` — recursing on the *empty* match set, vs Alg 3.1 line 6
   which recurses when non-empty) and the recursion dropped the `Match` callback.
   Both corrected; verified the descent reaches every leaf. `match` (420) is still a
   stub. This is the generic Search (Alg 3.1), a different algorithm from the
   LNodes iterator — write `Iterate_interior` fresh, don't extend `search`.
2. **`isless` (BWG.jl:113) — VERIFIED CORRECT (false alarm).**
   `morton(o, o.l, o.l)` shifts by `(b-l)*dim = 0`, so it returns the full,
   level-independent anchor interleave — the correct Alg 2.1 Z-order. Confirmed
   empirically: `issorted` + `searchsortedfirst` locate every leaf and reject
   refined parents in 2D/3D on balanced non-uniform forests, and `split_array`
   reconstructs the contiguous ordered leaves. The `searchsortedfirst` quick-fixes
   are safe. (Misleading TODO removed.)
3. **`transform_facet/edge/corner` orientation logic is empirical** (BWG.jl:1533
   "arithmetic switch: TODO understand this", :1567 "What is this condition
   exactly?"). The inside-vs-remote normal-axis sign differs from the paper /
   `_remote` variants; reproduced empirically. A soft spot if inter-tree node
   coordinates come out wrong.
4. **`boundaryset` docstrings mis-cite "Fig 4.1"** (BWG.jl:211/230) — it is
   **Figure 3**. Cosmetic, but it backs `B_∩^i`.
5. **`balanceforest!` uses an outer fixpoint** (BWG.jl:975) instead of the paper's
   single schedule/response round; 3D corner balancing incomplete (TODO :1033).
   2:1 balance is the load-bearing precondition for the iterator's `B_∩^i`
   selection — broken balance silently breaks descent assumptions.

---

## Implementation progress

- **Step 1 (done):** `iterate_leaves` (BWG.jl) — base recursive `split_array`
  descent, stop-at-singleton-leaf; visits exactly the leaves in Morton order.
  Tested on all 16 golden cases.
- **Step 2 (done — in `src`, tested):** a p4est_iterate-style **face descent**
  (`iterate_hanging_2d`) that detects hanging nodes during traversal, reproducing
  `creategrid`'s `conformity_info` exactly for single-tree 2D. The child-pair
  adjacency is *derived* by matching face coordinates — no hand-built
  CHILD_FACE_PAIRS table. Both descents are **allocation-free**: `split_array`
  (which built a `𝐤` vector + `2^dim` `SubArray` views per call) was replaced by
  `split_bounds`, returning the `2^dim+1` child boundary *indices* as a stack
  `NTuple`; the descent carries `(leaves, lo, hi)` ranges, never materializing
  views (`iterate_leaves` over 256 leaves: 0 bytes). Sketch (carries ranges):

  ```julia
  # during volume descent of an internal octant, visit its 4 internal child-child
  # face interfaces (z-order children 1=SW,2=SE,3=NW,4=NE):
  #   iter_face(c1,c2,+x); iter_face(c3,c4,+x); iter_face(c1,c3,+y); iter_face(c2,c4,+y)
  function iter_face!(hang, octL, leavesL, octR, leavesR, f, b) # f = octL's face toward octR
      lL = isleaf(leavesL, octL); lR = isleaf(leavesR, octR)
      lL && lR && return                                  # conforming face
      if lL && !lR                                        # coarse leaf vs refined → hanging
          fc = face(octL, f, b); push!(hang, center(fc) => collect(fc)); return
      elseif !lL && lR
          fc = face(octR, opp(f), b); push!(hang, center(fc) => collect(fc)); return
      end
      subL, cL = split_array(leavesL, octL, b), children(octL, b)   # both refined → recurse
      subR, cR = split_array(leavesR, octR, b), children(octR, b)
      for i in 1:4, j in 1:4
          contains_facet(face(octL, f, b), face(cL[i], f, b)) || continue
          face(cR[j], opp(f), b) == face(cL[i], f, b) &&
              iter_face!(hang, cL[i], subL[i], cR[j], subR[j], f, b)
      end
  end
  ```
  Hanging node = midpoint of the coarse face; constrainers = the coarse face's two
  endpoints (matches the 0.5/0.5 affine constraint).
- **Step 3 (done — in `src`, tested):** `iterate_hanging_3d`. Same face descent,
  faces have 4 corners; child matching uses the face *centre* (3D face-corner order
  differs between adjacent octants). Each coarse face bordering a refined neighbour
  emits its centre (4 constrainers) **and** its 4 edge midpoints (2 constrainers).
  This captures both face-centre and edge-centre hanging **without a 4-way edge
  descent**: the 4 cells around any edge form a cycle, so a hanging edge always
  borders a refined–coarse face pair (so it is an edge of some coarse face processed
  here). Matches `creategrid`'s `conformity_info` exactly on diverse single-tree 3D
  cases. Allocation-free descent (output Dict only): 2D 0.8 µs, 3D 18 µs on small trees.
- **Step 4 (done — in `src`, tested):** unified `iterate_hanging(forest::ForestBWG{dim})`
  (2D+3D). Intra-tree via the face descent above, plus **inter-tree** face neighbours
  matched with `transform_facet` (BWG2011 Alg 8, handles rotations) +
  `_emit_coarse_face_phys!`. Key simplification: **only FACE neighbours are needed
  even at tree boundaries** — every hanging node is face-interior, and the 4 cells
  around any edge cycle through faces, so a hanging edge always borders a
  refined–coarse face pair (no tree-edge/corner-neighbour handling). Matches
  `creategrid`'s `conformity_info` on **all 16 golden cases** (2D/3D, multi-tree,
  rotated trees, balanced, disc). The inter-tree part is a per-boundary-leaf
  transform (not yet a coordinated descent) — correctness first.
- **Step 5 (done — node numbering, IBWG2015 §6 LNodes):** `creategrid` (the default
  materializer) assigns each node an id in a single `iterate_leaves` pass — owner =
  min-Morton leaf touching it (eq 6.2), realized as first-encounter in Morton order.
  Identity is the exact integer `(tree, octree-coord)`; cross-tree shared nodes are
  merged by `_merge_cross_tree_nodes!` through the macro-mesh maps + orientation
  transforms (`transform_corner/facet/edge`, `rotation_permutation`) — **no physical
  coordinates, no rounding** (physical positions are emitted once per owner at the very
  end). `iterate_hanging` supplies the constraints on the same integer points. Validated
  renumbering-invariant against the frozen golden references on all 16 cases (2D/3D,
  multi-tree, rotated, balanced, disc) + 1643 correctness invariants.
- **Step 6 (done — cutover):** the legacy multi-pass `creategrid` (BWG2011 Alg 20 "Nodes")
  and `hangingnodes` are **deleted**; the iterator is the sole `creategrid`. This also
  removed a latent round-off bug shared by both old paths: rounded-physical-coordinate
  node identity can false-merge graded meshes / false-split rotated non-affine ones
  (silently breaking continuity); the integer/topological identity is immune.
- **Performance (90,112-leaf 3D case):** ~83 ms / 56 MiB — ~4× faster and ~6× lighter than
  the optimized legacy multi-pass, and faster than the (unsafe) physical-coordinate
  prototype. A function barrier on the cell build (`_build_cells`, concrete-type parameter +
  `map` over the connectivity tuple) removed ~720k boxed-cell allocations (was 141 ms /
  94.7 MiB / 1.2M allocs before the barrier).

> **Remaining (optional, not required for the serial use case):** the *literal*
> point-centric `Iterate_interior`/`Iterate` (Alg 5.2/5.3) with `S` support arrays,
> `part(c)`, and dim-dispatched callbacks (§2). The current volume + face-descent is a
> proven, integer-faithful specialization for node numbering + hanging detection; the
> point-centric engine would be the generalization for arbitrary per-dimension callbacks.
