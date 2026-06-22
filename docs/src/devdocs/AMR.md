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
struct OctantBWG{dim, N, T} <: AbstractCell{RefHypercube{dim}}
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
struct OctreeBWG{dim, N, T} <: AbstractAdaptiveCell{RefHypercube{dim}}
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

There are multiple useful functions to compute information about an octant e.g. parent, childs, etc.

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

### Balancing

Before a forest can be materialised into a grid it must satisfy the **2:1 balance** condition:
no two leaves sharing a face, edge or corner may differ by more than one refinement level. This
is what guarantees that hanging nodes only ever appear at edge midpoints / face centres (see
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

Two ideas carry the whole construction:

- **Integer / topological identity.** Every node is identified by an integer key
  `(tree, octree-coordinate)`, never by a floating-point physical position. Two leaves that meet
  at a vertex produce the *same* integer key, so shared nodes are recognised exactly, with no
  tolerances. Physical coordinates are interpolated only at the very end.
- **On a 2:1-balanced forest, hanging nodes are midpoints.** A non-conforming interface always
  places a node at the midpoint of a coarse edge or the centre of a coarse face. Each such node
  is recorded as a linear constraint: the hanging node equals the average of its *master* corners.

### The point iterator (IBWG2015, Algorithm 5)

The heart of the materialiser is a single recursive traversal, [`iterate_points`](@ref
Ferrite.AMR.iterate_points), the literal realisation of Algorithm 5.2/5.3 of [IBWG2015](@citet).
It visits every **non-hanging** topological entity of a tree — each leaf volume, and each
face/edge/corner *between* leaves — exactly once, calling a user callback `visit(c, leaf_supp)`
where `leaf_supp` are the leaves surrounding the entity `c`.

An entity is encoded integer/topologically as an axis-aligned box, the `IteratePoint`:

```julia
struct IteratePoint{dim}
    anchor::NTuple{dim, Int}   # minimum integer (octree) corner of the box
    level::Int                 # so the box has edge length _compute_size(b, level)
    axes::NTuple{dim, Bool}    # the directions the box extends along
end
```

The number of extending axes is the *dimension of the entity*,
`point_dim(c) = count(c.axes)`: `dim` for a volume, `dim-1` for a face, `1` for a 3D edge, `0`
for a corner. The callback dispatches on it. The keyword `mindim` is the §5.4 specialisation:
passing `mindim = dim-1` recurses into and fires the callback only for volumes and faces, which
is all that hanging-node detection on a balanced mesh needs (faces capture the edge hangers too,
see below).

Because the leaves of an octant form a *contiguous* Morton-sorted range, the descent slices that
range with [`split_bounds`](@ref Ferrite.AMR.split_bounds) instead of searching — it is
allocation-free, carrying index ranges rather than views.

### The `creategrid` pipeline

`creategrid` drives the iterator and assembles the grid in a few phases. The call graph:

```
creategrid(forest)
│
├─ for each tree:  iterate_points(tree; mindim = dim-1)        # IBWG2015 Alg 5.2/5.3
│      ├─ volume callback → _lnodes_number_leaf!               # number vertices + connectivity
│      └─ face   callback → _emit_coarse_face_int!             # intra-tree hanging nodes
│
├─ _iterate_interface_hanging!(forest)                         # inter-tree hanging nodes
│      └─ _iter_interface!  (per shared tree face)
│             └─ _emit_coarse_face_int!
│
├─ _merge_intertree_nodes!         # unify node ids shared across tree boundaries
├─ _treecorners / _interp_treepoint   # integer coords → physical coords (Q1 geometry map)
├─ _build_cells                    # connectivity tuples → Quadrilateral / Hexahedron cells
└─ reconstruct_facetsets           # carry named boundaries onto the refined grid
```

1. **Numbering, connectivity and intra-tree hanging** are *fused* into the single per-tree
   `iterate_points` pass. The volume callback assigns each leaf vertex a provisional id keyed on
   `(tree, coord)` and pushes the cell connectivity (Morton order); the face callback, whenever a
   coarse face borders a finer leaf, emits that face's interior hanging nodes.
2. **Inter-tree hanging** is collected by a cross-tree two-sided face descent
   ([`_iter_interface!`](@ref Ferrite.AMR._iter_interface!)) seeded at every shared tree face —
   the same idea as the intra-tree face callback, but matching the two sides across a tree
   boundary via [`transform_facet`](@ref Ferrite.AMR.transform_facet) (handling rotations).
3. **Cross-tree identity, compaction and coordinates.** Per-tree numbering yields one
   `(tree,coord)` key per incident tree; [`_merge_intertree_nodes!`](@ref
   Ferrite.AMR._merge_intertree_nodes!) canonicalises shared-boundary keys onto a single owner.
   The provisional ids are then compacted to a dense range and each owner's physical coordinate
   is computed once with the tree's ``Q_1`` geometry map.
4. **Cells and constraints.** [`_build_cells`](@ref Ferrite.AMR._build_cells) maps connectivity
   to final ids, the hanging map is translated to final ids, and
   [`reconstruct_facetsets`](@ref Ferrite.AMR.reconstruct_facetsets) transfers the boundary sets.

```@docs
Ferrite.AMR.creategrid
Ferrite.AMR.iterate_points
Ferrite.AMR.split_bounds
Ferrite.AMR._lnodes_number_leaf!
Ferrite.AMR._merge_intertree_nodes!
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
vertex on the coarse side. On a 2:1-balanced forest these are exactly the **centre of a coarse
face** (bordering a refined neighbour) and the **midpoints of that face's edges**.
[`_emit_coarse_face_int!`](@ref Ferrite.AMR._emit_coarse_face_int!) records them, keyed on the
integer `(tree, coord)` of the hanging node, with the value listing its master corners:

```
3D face fc = (c1,c2,c3,c4) in z-order — ● corner (master), ◆ face centre, ○ edge midpoint:

    c3 ●━━━━━━━○━━━━━━━● c4      emitted:
       ┃      m34      ┃          ◆  hang[c  ] = {c1,c2,c3,c4}
       ┃               ┃          ○  hang[m12] = {c1,c2}
    m13○       ◆c      ○m24       ○  hang[m34] = {c3,c4}
       ┃   (centre)    ┃          ○  hang[m13] = {c1,c3}
       ┃      m12      ┃          ○  hang[m24] = {c2,c4}
    c1 ●━━━━━━━○━━━━━━━● c2
```

The diagonals `c1–c4` and `c2–c3` differ in *two* coordinates and are skipped — they are not
octant edges. In 2D a face *is* an edge, so the face centre and the single edge midpoint coincide
into one constraint with two masters.

#### Why a face descent captures hanging *edges*

It is not obvious that visiting only **faces** finds every hanging **edge** node. The argument:
the descent runs over *every* coarse face bordering a refined neighbour and emits that face's four
edge-midpoints, so a hanging edge-midpoint is captured as long as its edge is an edge of *some*
emitted face — and it always is.

Look down a coarse edge `E` (a point `⊙` in the cross-section); four cells surround it, and `E` is
an edge of all four faces meeting at `E`. The midpoint `m = center(E)` becomes a node only if some
surrounding cell is refined. The coarse owner `C` (level ℓ) and at least one refined cell
(level ℓ+1) both sit around `E`, so going around the four-cell cycle a coarse cell must be
face-adjacent to a refined one — that shared face is coarse-bordering-refined and has `E` as an
edge, so the descent emits `m`. The subtle case is a refined cell *diagonal* to `C` (sharing only
`E`, not a face):

```
    Q4 │ Q3      C|Q2 and C|Q4 are coarse–coarse (conforming), so C's own faces miss E.
    ℓ  │ ℓ+1     But Q2|Q3 and Q4|Q3 are coarse(ℓ)–refined(ℓ+1): those faces are emitted,
    ───⊙───      and E is one of their edges → m is still emitted. ✓
    C  │ Q2
   (ℓ) │ ℓ
```

2:1 balance is what makes this exhaustive: it caps the level jump at one, so hanging nodes are
*always* midpoints (never ¼/¾ points from a two-level jump) and the finest cell around any edge is
at most one level finer. Hence emitting face-centres + edge-midpoints over all coarse faces
bordering refined neighbours captures every hanging node — no separate edge descent is needed.

```@docs
Ferrite.AMR._emit_coarse_face_int!
Ferrite.AMR._iter_interface!
Ferrite.AMR._iterate_interface_hanging!
Ferrite.AMR.center
Ferrite.AMR.contains_facet
```

### Conformity constraints

The hanging-node map produced by `creategrid` is turned into affine constraints by adding a
`ConformityConstraint` to a `ConstraintHandler`. For linear (``Q_1``) interpolations each hanging
node is constrained to the **average** of its masters — weight `1/length(masters)`, i.e. `1/2` for
an edge midpoint and `1/4` for a 3D face centre — which is exactly the value that makes the field
continuous across the non-conforming interface.

```@docs
Ferrite.AMR.ConformityConstraint
```
