# Adaptive Mesh Refinement (AMR)

## P4est

Ferrite's P4est implementation is based on these papers:

- [BWG2011](@citet)
- [IBWG2015](@citet)

where almost everything is implemented in a serial way from the first paper.
Only certain specific algorithms of the second paper are implemented and there is a lot of open work to include the iterators of the second paper.
Look into the issues of Ferrite.jl and search for the AMR tag.

### Important Concepts

One of the most important concepts, which everything is based on, are space filling curves (SFC).
In particular, [Z-order (also named Morton order, Morton space-filling curves)](https://en.wikipedia.org/wiki/Z-order_curve) are used in p4est.
The basic idea is that each Octant (in 3D) or quadrant (in 2D) can be encoded by 2 quantities

- the level `l`
- the lower left (front) coordinates `xyz`

Based on them a unique identifier, the morton index, can be computed.
The mapping from (`l`, `xyz`) -> `mortonidx(l,xyz)` is bijective, meaning we can flip the approach
and can construct each octant/quadrant solely by the `mortonidx` and a given level `l`.

The current implementation of an octant looks currently like this:
```julia
struct OctantBWG{dim, N, T <: Integer} <: AbstractCell{RefHypercube{dim}}
    #Refinement level
    l::T
    #x,y,z \in {0,...,2^b} where (0 ≤ l ≤ b)}
    xyz::NTuple{dim, T}
end
```
whenever coordinates are considered we follow the z order logic, meaning x before y before z.
Note that the acronym BWG stands for the initials of the surname of the authors of the p4est paper.
The coordinates of an octant are described in the *octree coordinate system* which goes from $[0,2^b]^{dim}$.
The parameter $b$ describes the maximum level of refinement and is set a priori.
Another important aspect of the octree coordinate system is, that it is a discrete integer coordinate system.
The size of an octant at the lowest possible level `b` is always 1, sometimes these octants are called atoms.

The octree is implemented as:
```julia
struct OctreeBWG{dim, N, T <: Integer} <: AbstractAdaptiveCell{RefHypercube{dim}}
    leaves::Vector{OctantBWG{dim, N, T}}
    #maximum refinement level
    b::T
    nodes::NTuple{N, Int}
end
```

So, only the leaves of the tree are stored and not any intermediate refinement level.
The field `b` is the maximum refinement level and is crucial. This parameter determines the size of the octree coordinate system.
The octree coordinate system is the coordinate system in which the coordinates `xyz` of any `octant::OctantBWG` are described.

### Examples

Let's say the maximum octree level is $b=3$, then the coordinate system is in 2D $[0,2^3]^2 = [0, 8]^2$.
So, our root is on level 0 of size 8 and has the lower left coordinates `(0,0)`

```julia
# different constructors available, first one OctantBWG(dim,level,mortonid,maximumlevel)
# other possibility by giving directly level and a tuple of coordinates OctantBWG(level,(x,y))
julia > dim = 2; level = 0; maximumlevel = 3
julia > oct = OctantBWG(dim, level, 1, maximumlevel)
OctantBWG{2, 4, 4}
l = 0
xy = 0, 0
```
The size of octants at a specific level can be computed by a simple operation
```julia
julia > Ferrite.AMR._compute_size(#=b=# 3, #=l=# 0)
8
```
This computation is based on the relation $\text{size}=2^{b-l}$.
Now, to fully understand the octree coordinate system we go a level down, i.e. we cut the space in $x$ and $y$ in half.
This means, that the octants are now of size $2^{3-1}=4$.
Construct all level 1 octants based on mortonid:
```julia
# note the arguments are dim,level,mortonid,maximumlevel
julia > dim = 2; level = 1; maximumlevel = 3
julia > oct = Ferrite.AMR.OctantBWG(dim, level, 1, maximumlevel)
OctantBWG{2, 4, 4}
l = 1
xy = 0, 0

julia > oct = Ferrite.AMR.OctantBWG(dim, level, 2, maximumlevel)
OctantBWG{2, 4, 4}
l = 1
xy = 4, 0

julia > oct = Ferrite.AMR.OctantBWG(dim, level, 3, maximumlevel)
OctantBWG{2, 4, 4}
l = 1
xy = 0, 4

julia > oct = Ferrite.AMR.OctantBWG(dim, level, 4, maximumlevel)
OctantBWG{2, 4, 4}
l = 1
xy = 4, 4
```

So, the morton index is on **one** specific level just a x before y before z "cell" or "element" identifier
```
x-----------x-----------x
|           |           |
|           |           |
|     3     |     4     |
|           |           |
|           |           |
x-----------x-----------x
|           |           |
|           |           |
|     1     |     2     |
|           |           |
|           |           |
x-----------x-----------x
```

The operation to compute octants/quadrants is cheap, since it is just bitshifting.
An important aspect of the morton index is that it's only consecutive on **one** level in this specific implementation.
Note that other implementation exists that incorporate the level integer within the morton identifier and by that have a unique identifier across levels.
If you have a tree like this below:

```
x-----------x-----------x
|           |           |
|           |           |
|     9     |    10     |
|           |           |
|           |           |
x-----x--x--x-----------x
|     |6 |7 |           |
|  3  x--x--x           |
|     |4 |5 |           |
x-----x--x--x     8     |
|     |     |           |
|  1  |  2  |           |
x-----x-----x-----------x
```

you would maybe think this is the morton index, but strictly speaking it is not.
What we see above is just the `leafindex`, i.e. the index where you find this leaf in the `leaves` array of `OctreeBWG`.
Let's try to construct the lower right based on the morton index on level 1

```julia
julia> o = Ferrite.OctantBWG(2,1,8,3)
ERROR: AssertionError: m ≤ (one(T) + one(T)) ^ (dim * l) # 8 > 4
Stacktrace:
 [1] OctantBWG(dim::Int64, l::Int32, m::Int32, b::Int32)
   @ Ferrite ~/repos/Ferrite.jl/src/Adaptivity/AdaptiveCells.jl:23
 [2] OctantBWG(dim::Int64, l::Int64, m::Int64, b::Int64)
   @ Ferrite ~/repos/Ferrite.jl/src/Adaptivity/AdaptiveCells.jl:43
 [3] top-level scope
   @ REPL[93]:1
```

The assertion expresses that it is not possible to construct a morton index 8 octant, since the upper bound of the morton index is 4 on level 1.
The morton index of the lower right cell is 2 on level 1.

```julia
julia > o = Ferrite.AMR.OctantBWG(2, 1, 2, 3)
OctantBWG{2, 4, 4}
l = 1
xy = 4, 0
```

### Octant operation

There are multiple useful functions to compute information about an octant e.g. parent, children, etc.

```@docs
Ferrite.AMR.isancestor
Ferrite.AMR.morton
Ferrite.AMR.children
Ferrite.AMR.vertices
Ferrite.AMR.edges
Ferrite.AMR.faces
```

### Intraoctree operation

Intraoctree operation stay within one octree and compute octants that are attached in some way to a pivot octant `o`.
These operations are useful to collect unique entities within a single octree or to compute possible neighbors of `o`.
[BWG2011](@citet) Algorithm 5, 6, and 7 describe the following intraoctree operations:

```@docs
Ferrite.AMR.corner_neighbor
Ferrite.AMR.edge_neighbor
Ferrite.AMR.facet_neighbor
Ferrite.AMR.possibleneighbors
```

### Interoctree operation

Interoctree operation are in contrast to intraoctree operation by computing octant transformations across different octrees.
Thereby, one needs to account for topological connections between the octrees as well as possible rotations of the octrees.
[BWG2011](@citet) Algorithm 8, 10, and 12 explain the algorithms that are implemented in the following functions:

```@docs
Ferrite.AMR.transform_corner
Ferrite.AMR.transform_edge
Ferrite.AMR.transform_facet
```

Note that we flipped the input and to expected output logic a bit to the proposed algorithms of the paper.
However, the original proposed versions are implemented as well in:

```@docs
Ferrite.AMR.transform_corner_remote
Ferrite.AMR.transform_edge_remote
Ferrite.AMR.transform_facet_remote
```

despite being never used in the code base so far.

### Refinement and coarsening

Refinement replaces a leaf by its `2^dim` children; coarsening replaces a `2^dim`-sibling
family by its parent. Both operate on each tree's Morton-sorted `leaves` vector and preserve
that order, which the rest of the pipeline (balancing, the point iterator) relies on.

The production entry point is `refine!(forest, cellids)`: an adaptive FE step marks cells
with an error estimator and passes their global ids here. It is implemented to scale
linearly in the number of leaves — the marked ids are mapped to per-tree local indices and
each tree's leaf list is rebuilt in a single pass, rather than refining cells one at a time
(every in-place `insert!` would memmove the array tail, giving `O(n^2)`). `refine_all!` is
the uniform-refinement convenience wrapper and is linear for the same reason.

```@docs
Ferrite.AMR.refine!
Ferrite.AMR.refine_all!
Ferrite.AMR.coarsen!
Ferrite.AMR.coarsen_all!
```

### Balancing

Before a forest can be materialised into a grid it must satisfy the **2:1 balance** condition:
no two leaves sharing a face, edge or corner may differ by more than one refinement level. This
is what guarantees that hanging nodes only ever appear at edge midpoints / face centers (see
[Hanging nodes](@ref) below). [`balanceforest!`](@ref Ferrite.AMR.balanceforest!) enforces it,
balancing each tree internally and propagating across tree boundaries for the leaves that touch
them.

```@docs
Ferrite.AMR.balanceforest!
Ferrite.AMR._balance_leaf!
Ferrite.AMR._touches_tree_boundary
Ferrite.AMR.inside
Ferrite.AMR._maximum_size
```

## From a forest to a `NonConformingGrid`

The operations above manipulate the forest of octrees (refine, coarsen, balance, neighbour
lookups). To actually solve a finite element problem we must turn that forest into a concrete
grid — this is [`creategrid`](@ref Ferrite.AMR.creategrid), which produces a
`NonConformingGrid`: an ordinary grid plus the *hanging-node constraints* (`conformity_info`)
that make a conforming finite element field possible.

!!! warning "`conformity_info` is subject to change"
    `conformity_info` currently stores hanging *vertices* and their master vertices — exactly
    the information a linear (Q1) discretization needs, and no more. For general
    discretizations the non-conforming interface itself must be exposed: hanging *edges and
    faces* need to be detected and stored as entities, so that a field of any order can
    constrain all of its dofs on such an entity (with weights obtained by evaluating the
    coarse side's basis). Expect the layout of this field to change when support for
    higher-order discretizations lands.

Two ideas carry the whole construction:

- **Integer / topological identity, no global node map.** Every node is identified integer /
  topologically — a corner of the integer octree lattice of one tree — never by a floating-point
  physical position, so shared nodes are recognised exactly, with no tolerances. Physical
  coordinates are interpolated only once per node. There is **no coordinate→id map of any
  kind**, because the traversal below discovers every mesh entity *exactly once*: identity is
  established by construction, so "assign an id" degenerates to a counter increment. Node ids
  are assigned by the iterator callbacks and scattered into the element-node matrix `E`
  ([IBWG2015](@citet) §6, `Lnodes`), and only *tree-boundary* nodes additionally enter small
  per-tree sorted tables used to reconcile identity across tree boundaries (see phase 3 of
  the pipeline below). This is the data layout that generalizes to
  distributed forests: each process numbers the nodes it owns, and only interface node ids are
  exchanged.
- **2:1 balance is what keeps non-conformity tractable.** On a balanced forest the two sides of
  a non-conforming interface differ by exactly one level, which buys the construction two
  things. First, hanging nodes appear only at *predictable* integer coordinates — the midpoint
  of a coarse edge or the center of a coarse face — so the traversal detects and numbers them
  without any search (with larger level jumps they could sit at quarter points and deeper).
  Second, the *masters* of a hanging node are nodes of the coarse entity and therefore regular
  themselves: every conformity constraint is resolved within one level and never chains through
  other hanging nodes. The algorithms of [IBWG2015](@citet) assume a balanced forest throughout.

### Vocabulary: points, closure, support, part

The traversal machinery speaks the vocabulary of [IBWG2015](@citet) §2. Four terms carry
everything, and all of them are purely integer/topological — no physical coordinate and no
floating-point comparison appears anywhere:

| Term | Meaning | Paper | Code |
|:-----|:--------|:------|:-----|
| **point** | *One topological entity* of the mesh — vertex, edge, face or volume — encoded as a (possibly degenerate) axis-aligned box: an anchor corner, a level, and per axis a flag whether the box extends along it. Two points are equal iff their encodings are equal. | §2.1 | [`IteratePoint`](@ref Ferrite.AMR.IteratePoint) |
| **closure** | A box *including* its boundary faces, edges and corners. "Octant `o` touches point `c`" always means `c ⊂ closure(o)`. | §2.1 | [`_child_touches_point`](@ref Ferrite.AMR._child_touches_point) |
| **support** of `c` | The octants **at `c`'s level** whose closure contains `c` — up to `2^(dim - dim(c))` boxes around it (fewer on the domain boundary): 1 for a volume, 2 across a face, 4 around a 3D edge, `2^dim` around a corner. These are the only octants that can decide what happens at `c`. | eq 2.11 | `sc.supp` in [`_iterate_interior!`](@ref Ferrite.AMR._iterate_interior!) |
| **part** of `c` | What splitting the point once decomposes its **interior** into: the `3^dim(c)` points one level finer. | eq 2.7 | [`_foreach_partc`](@ref Ferrite.AMR._foreach_partc) |

```julia
struct IteratePoint{dim}
    anchor::NTuple{dim, Int}   # minimum integer (octree) corner of the box
    level::Int                 # so the box has edge length _compute_size(b, level)
    axes::NTuple{dim, Bool}    # the directions the box extends along
end
```

The number of extending axes is the *dimension of the point*, `point_dim(c) = count(c.axes)`
(the paper's `dim(c)`). An octant is simply its own volume point (Remark 2.2 of
[IBWG2015](@citet)).

One drawing per point dimension, each with its support (cf. Table 2.2 of
[IBWG2015](@citet)); all boxes are at the same level `ℓ`, i.e. of size `h = 2^(b-ℓ)`:

```
2D  ────────────────────────────────────────────────────────────────────────────

 volume point, dim(c) = 2   face point, dim(c) = 1    corner point, dim(c) = 0
 axes = (true, true)        axes = (false, true)      axes = (false, false)

  ┏━━━━━━━┓                          ┃                          │
  ┃       ┃                    s1    ┃    s2              s3    │    s4
  ┃   c   ┃                          ┃ c                  ──────●──────
  ┃       ┃                          ┃                    s1    │ c  s2
  ┗━━━━━━━┛                          ┃                          │

 supp(c) = {the octant     supp(c) = {s1, s2},        supp(c) = {s1, s2, s3, s4},
 itself}: c IS the         the 2 octants whose        the 2^dim octants whose
 octant's box              closure contains the       closure contains the
                           face                       corner

3D  ────────────────────────────────────────────────────────────────────────────

 volume: as in 2D, supp = {itself}          face (dim 2): 2 supports, as in 2D
 corner (dim 0): 2³ = 8 supports            edge (dim 1): 4 supports — looking
                                            down the edge axis it is exactly
                                            the 2D corner picture
```

Splitting a 2D volume point `c` once illustrates `part(c)` — the `3² = 9` interior
sub-points, each one level finer:

```
  ┌─────────┬─────────┐
  │         │         │     4 volume points  ▢
  │    ▢    f    ▢    │     4 face points    f
  │         │         │     1 corner point   ●   (the center of c's box)
  ├────f────●────f────┤
  │         │         │     the boundary of c's box is NOT in part(c)!
  │    ▢    f    ▢    │
  │         │         │
  └─────────┴─────────┘
```

The center point `●` ties the two figures together: it is a corner point (`dim(c) = 0`) one
level finer than `c`, and its support (eq 2.11) are exactly the four volume sub-points `▢`
around it — the corner-point column of the table above, one level down. Likewise each face
point `f` has the two adjacent volume sub-points as its support.

The boundary remark is the fact to internalize: the boundary features of `c`'s box are **not**
in `part(c)` — they were already produced when `c`'s own *parent* point was split. Every
entity of the leaf mesh therefore lies in the interior of exactly one ancestor box, at
exactly one level, so a recursion that descends through `part` reaches every mesh entity
**exactly once** — no deduplication, no lookup, identity by construction. The only
exception is the root's own boundary, which has no parent split to produce it; its `3^dim`
closure points are seeded explicitly
([`_foreach_root_closure`](@ref Ferrite.AMR._foreach_root_closure), Alg 5.3 line 4).

### The descent and the stop rule

The heart of the materializer is the recursive traversal [`iterate_points`](@ref
Ferrite.AMR.iterate_points) (`Iterate`, Alg 5.3, serial), which drives
[`_iterate_interior!`](@ref Ferrite.AMR._iterate_interior!) (`Iterate_interior`, Alg 5.2)
from every root-closure seed. Each recursion step carries a point `c` together with its
support octants and, per support octant, the index range of the actual leaves below it in
the tree's Morton-sorted `leaves` vector (the paper's `S` arrays). At every step the
recursion asks one question — **is some support octant itself a leaf?** (Alg 5.2 line 7):

- **No** — everything around `c` is refined further, so `c`'s current description is too
  coarse to be a mesh entity. *Descend*: split `c` into `part(c)`
  ([`_foreach_partc`](@ref Ferrite.AMR._foreach_partc)), give each sub-point its support
  from the children of `c`'s supports (a combinatorial constant served by precomputed mask
  tables, `_part_mask`), and slice the leaf ranges with
  [`split_bounds`](@ref Ferrite.AMR.split_bounds) — descendants of an octant are contiguous
  in Morton order, so this is index arithmetic, not search.
- **Yes** — a leaf has no children, so the mesh has no finer structure touching `c` from
  that side: `c`, as described, *is* an entity of the final mesh. The point is
  **finalized**: the recursion stops, builds the *leaf support* `leaf_supp(c)` — each
  support octant that is a leaf enters as-is; for the refined ones, their children adjacent
  to `c` enter (one level down suffices under 2:1 balance; `B_∩^j`, Alg 5.2 line 14,
  [`_child_touches_point`](@ref Ferrite.AMR._child_touches_point)) — and fires the callback
  `visit(c, leaf_supp)` for it, exactly once, ever. Corner points cannot be split and
  always finalize (Alg 5.2 lines 15–18; the leaf per support subtree is found by
  [`_descend_to_corner`](@ref Ferrite.AMR._descend_to_corner)).

The visited set is exactly `PΩ` of §5.1: every leaf volume and every face/edge/corner
*between* leaves. Two consequences deserve emphasis:

1. **Exactly-once with complete support.** Each visited entity fires one callback, and that
   callback sees *all* leaves touching the entity — the traversal never delivers a partial
   neighbourhood.
2. **Hanging points are never visited** (Fig 5). At a face between a coarse leaf and finer
   neighbours the recursion stops at the *coarse* level — it cannot descend past a leaf —
   so the finer vertices in the face's interior lie beyond the recursion frontier and never
   become points. Their information is not lost: it arrives at the finalized coarse face,
   whose leaf support then mixes two levels, and that mixed support *is* the complete
   hanging-node configuration (see [Hanging nodes](@ref) below).

```
the stop rule at a non-conforming face (2D) — the face point c is the interface
between the coarse leaf and its refined neighbour, i.e. the segment from ● to ●:

   ┌─────────────────●────────┐
   │                 │        │     the recursion stops at c: its LEFT support
   │                 │  fine  │     (the coarse leaf, level ℓ) is itself a leaf,
   │                 │ (ℓ+1)  │     so c is finalized with
   │     coarse      │        │        leaf_supp(c) = {coarse, fine, fine}
   │      leaf       ○────────┤
   │   (level ℓ)     │        │     ○, the midpoint of c: a corner of the two
   │                 │  fine  │     fine leaves but not a vertex of the coarse
   │                 │ (ℓ+1)  │     leaf, and never visited as a point of its
   │                 │        │     own — the face callback creates it as the
   └─────────────────●────────┘     hanging node, with the two ● as masters
```

Three keywords of `iterate_points` give the §5.4 callback specialisations, the analogue of
passing `NULL` callbacks to `p4est_iterate`: `mindim` (don't recurse into / fire for points
below this dimension), `maxdim` (don't fire the callback above this dimension; the volume
recursion still runs, being the spine of the traversal), and `skip_conforming` (skip the
callback for faces/edges whose supports are all equal-level leaves — conforming interfaces
— for callbacks that only act on non-conforming ones; corners always fire). The whole
descent is allocation-free: all per-depth state lives in one preallocated
[`IterScratch`](@ref Ferrite.AMR.IterScratch), reused across the trees of the forest.

```@docs
Ferrite.AMR.IteratePoint
Ferrite.AMR.iterate_points
Ferrite.AMR._iterate_interior!
Ferrite.AMR.IterScratch
Ferrite.AMR._foreach_partc
Ferrite.AMR._foreach_root_closure
Ferrite.AMR._child_touches_point
Ferrite.AMR._descend_to_corner
Ferrite.AMR.split_bounds
```

### The traversal types at a glance

Four types cooperate, with sharply separated roles — one is a message, one is memory, one
is a window, one is the consumer:

| Type | Role | Lifetime |
|:-----|:-----|:---------|
| [`IteratePoint`](@ref Ferrite.AMR.IteratePoint) | The *message*: describes the visited entity. | Created and discarded during the descent; never stored. |
| [`IterScratch`](@ref Ferrite.AMR.IterScratch) | The *memory*: per-depth buffers of the recursion, so the traversal allocates nothing. Carries no meaning between traversals. | One per forest, reused for every tree. |
| [`LeafSupport`](@ref Ferrite.AMR.LeafSupport) | The *window*: the leaves touching the visited entity **plus each leaf's index** in `tree.leaves` — the element index `j` of §6.4 ("`Iterate` provides the index"), which lets a callback address per-element data directly. Wraps the scratch's buffers. | Valid only during the callback call — copy what you retain. |
| [`LnodesVisitor`](@ref Ferrite.AMR.LnodesVisitor) | The *consumer*: `creategrid`'s callback (`Lnodes_callback`, Alg 6.2) as a callable struct, so it can carry the output arrays it fills. | One per tree; its fields reference forest-wide outputs. |

All four meet at the callback boundary of one per-tree traversal:

```
 iterate_points(visitor::LnodesVisitor, tree, sc::IterScratch; …)
       │
       ▼  _iterate_interior! descends; for every finalized point c:
 visitor(c::IteratePoint, ls::LeafSupport)    # ls: a window into sc's reused buffers
```

What the visitor *does* with each visit — and which forest-wide output arrays its fields
reference — is the subject of the pipeline below.

### The `creategrid` pipeline

`creategrid` drives the iterator and assembles the grid in a few phases. The call graph:

```
creategrid(forest)
│
├─ for each tree:  iterate_points(visitor, tree, sc;                  # IBWG2015 Alg 5.2/5.3
│                                 mindim = 0, maxdim = dim-1,         #  = Alg 6.2 Lnodes
│                                 skip_conforming = true)
│      ├─ corner callback → _visit_corner!    # create node id, scatter into E[slot, element]
│      ├─ face   callback → _visit_face!      # hanging face midpoint (2D) / center (3D)
│      └─ edge   callback → _visit_edge3d!    # hanging edge midpoints (3D)
│
├─ _iterate_interface_hanging!                # inter-tree hanging constraints (reads E)
│      └─ _iter_interface!  (per shared tree face) → _emit_interface_face!
│
├─ _merge_intertree_nodes!    # alias ids shared across tree boundaries (boundary tables)
├─ _global_numbering          # Alg 6.1: final dense ids in one sweep over E
├─ _build_cells               # E columns → Quadrilateral / Hexahedron cells
└─ reconstruct_facetsets      # carry named boundaries onto the refined grid
```

1. **Numbering and hanging detection** happen inside the single per-tree `iterate_points` pass
   (the `Lnodes_callback` of IBWG2015 Alg 6.2, [`LnodesVisitor`](@ref Ferrite.AMR.LnodesVisitor)).
   The *corner* callback creates the node at the visited point — a running provisional id, its
   physical coordinate, and a boundary-table entry if it lies on the tree boundary — and scatters
   the id into the element-node matrix `E` of every supporting leaf ("complete the entries in
   `Ep` that refer to `g`", §6.4). `E` is a `2^dim × ncells` integer matrix holding the node id
   of every element corner in z-order — connectivity and node numbering in one array, and the
   single structure every later phase reads from. The *face*/*edge* callbacks detect non-conformity from the
   level mismatch in their support, create the hanging vertex the same way (hanging vertices are
   genuine fine-leaf corners) and record its constraint as `(element, slot)` references into `E`,
   resolved after the traversal.
2. **Inter-tree hanging** is collected by a cross-tree two-sided face descent
   ([`_iter_interface!`](@ref Ferrite.AMR._iter_interface!)) seeded at every shared tree face —
   the same idea as the intra-tree callbacks, but matching the two sides across a tree
   boundary via [`transform_facet`](@ref Ferrite.AMR.transform_facet) (handling rotations).
3. **Cross-tree identity.** The traversal is strictly per-tree, so a node on a shared tree
   boundary is visited once per incident tree and briefly holds one provisional id per tree.
   To merge the duplicates one lookup structure is unavoidable — "tree `k`, which id did you
   give the node at coordinate `x`?" — and it is deliberately confined to the tree *surface*:
   whenever the corner callback creates a node whose coordinate lies on the root boundary
   (a component is `0` or `2^b`), it also appends `(key, id)` to that tree's boundary table,
   with the coordinate bit-packed into a single `UInt64` key (`_packcoord`) so comparisons
   are one machine word. Each table is sorted once after its tree's traversal;
   [`_bnd_lookup`](@ref Ferrite.AMR._bnd_lookup) then answers queries by binary search.
   [`_merge_intertree_nodes!`](@ref Ferrite.AMR._merge_intertree_nodes!) walks the shared
   root vertices/faces/edges, maps coordinates into the neighbour tree's frame via
   [`transform_facet`](@ref Ferrite.AMR.transform_facet)/`transform_corner`/`transform_edge`
   (handling tree rotations), and records `alias[duplicate] = owner` (the lower tree index
   owns). A lookup miss is routine and meaningful: a hanging node exists as a vertex only on
   the refined side of an interface, so the coarse neighbour has no entry for it. Interior
   nodes — the overwhelming majority — never enter any table: `O(surface)` data, the only
   node-lookup structure of the materializer.
4. **Global numbering, cells and constraints.** [`_global_numbering`](@ref
   Ferrite.AMR._global_numbering) is the serial `Global_numbering` (Alg 6.1): one linear sweep
   over `E` in (element, element-node) order assigns final dense ids by first encounter — with
   the ownership rule `owner(c) = min leaf supp(c)` (eq 6.2) this is the paper's
   partition-independent numbering. [`_build_cells`](@ref Ferrite.AMR._build_cells) then reads
   the cells straight off `E`, the constraint records are resolved against `E`, and
   [`reconstruct_facetsets`](@ref Ferrite.AMR.reconstruct_facetsets) transfers the boundary sets.

```@docs
Ferrite.AMR.creategrid
Ferrite.AMR.LeafSupport
Ferrite.AMR.LnodesVisitor
Ferrite.AMR._visit_corner!
Ferrite.AMR._mixed_support
Ferrite.AMR._global_numbering
Ferrite.AMR._merge_intertree_nodes!
Ferrite.AMR._bnd_lookup
Ferrite.AMR._build_cells
Ferrite.AMR.reconstruct_facetsets
```

### Physical coordinates

Node identity is purely integer; physical positions enter only here. Each macro element (tree) is
an isoparametric ``Q_1`` cell, so an octree coordinate is mapped to physical space by interpolating
the tree's corner nodes with the bi-/trilinear Lagrange shape functions.

```@docs
Ferrite.AMR._treecorners
Ferrite.AMR._interp_treepoint
```

### Hanging nodes

A hanging node is a node that exists on the fine side of a non-conforming interface but is not a
vertex on the coarse side. On a 2:1-balanced forest these are exactly the **center of a coarse
face** (bordering a refined neighbour) and the **midpoint of a coarse edge** (bordering a finer
leaf) — balance caps the level jump at one, so no ¼-points exist:

```
3D face fc = (c1,c2,c3,c4) in z-order — ● corner (master), ◆ face center, ○ edge midpoint:

    c3 ●━━━━━━━○━━━━━━━● c4      constraints:
       ┃      m34      ┃          ◆  hnodes[c  ] = {c1,c2,c3,c4}   (face callback)
       ┃               ┃          ○  hnodes[m12] = {c1,c2}         (edge callbacks)
    m13○       ◆c      ○m24       ○  hnodes[m34] = {c3,c4}
       ┃   (center)    ┃          ○  hnodes[m13] = {c1,c3}
       ┃      m12      ┃          ○  hnodes[m24] = {c2,c4}
    c1 ●━━━━━━━○━━━━━━━● c2
```

In 2D a face *is* an edge, so there is just the midpoint with its two masters. Because hanging
points are exactly the points the iterator never visits (the stop rule halts at the coarse
leaf; see [The descent and the stop rule](@ref) above), they are created by the *feature that
owns them* — detected via the mixed-level leaf support
([`_mixed_support`](@ref Ferrite.AMR._mixed_support)): a non-conforming **face point** creates
its center, and a non-conforming **edge point** creates its midpoint ([`_visit_face!`](@ref Ferrite.AMR._visit_face!),
[`_visit_edge3d!`](@ref Ferrite.AMR._visit_edge3d!)). Each hanging vertex belongs to exactly one
such coarse feature (two octant edges cannot share their midpoints), and each feature is visited
exactly once — even an edge shared by several non-conforming faces — so every hanging vertex is
created exactly once, with its constraint recorded as `(element, slot)` references into `E` (a
master corner may not be numbered yet when the feature is visited; the references are resolved
after the traversal).

```@docs
Ferrite.AMR._visit_face!
Ferrite.AMR._visit_edge3d!
Ferrite.AMR._iter_interface!
Ferrite.AMR._emit_interface_face!
Ferrite.AMR._iterate_interface_hanging!
Ferrite.AMR.center
Ferrite.AMR.contains_facet
```

### Conformity constraints

The hanging-node map produced by `creategrid` is turned into affine constraints by adding a
`ConformityConstraint` to a `ConstraintHandler`. For linear (``Q_1``) interpolations each hanging
node is constrained to the **average** of its masters — weight `1/length(masters)`, i.e. `1/2` for
an edge midpoint and `1/4` for a 3D face center — which is exactly the value that makes the field
continuous across the non-conforming interface.

```@docs
Ferrite.AMR.ConformityConstraint
```
