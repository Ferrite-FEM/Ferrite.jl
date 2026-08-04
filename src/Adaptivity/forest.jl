# The forest of octrees: everything that spans more than one tree, plus the grid
# materialization pipeline. Builds on the single-tree primitives in `octree.jl`.
#
#   `ForestBWG`                    the forest, its Ferrite `AbstractGrid` interface, and the
#                                  user-facing `refine!`/`coarsen!`/`refine_and_coarsen!`
#   inter-tree node merging        `_merge_intertree_nodes!` and the set reconstruction
#   balancing                      `balanceforest!` / `balancetree` ([BWG2011](@cite) Alg. 17)
#   inter-tree transforms          `transform_facet`/`_corner`/`_edge` and the orientations
#   the point iterator             `iterate_points` ([IBWG2015](@cite) Alg. 5.2)
#   materialization                `creategrid` (Lnodes numbering) and `facetskeleton`

"""
    ForestBWG{dim, C <: OctreeBWG, T <: Real} <: Ferrite.AbstractGrid{dim}
`p4est` adaptive grid implementation based on [BWG2011](@citet)
and [IBWG2015](@citet).

## Constructor
    ForestBWG(grid::AbstractGrid{dim}, b) where dim
Builds an adaptive grid based on a non-adaptive one `grid` and a given max refinement level `b`,
i.e. no leaf may be refined beyond level `b`.

`b` must satisfy `0 ≤ b ≤ 30` in 2D and `0 ≤ b ≤ 19` in 3D, and defaults to those upper bounds
(p4est's `P4EST_MAXLEVEL`/`P8EST_MAXLEVEL`). They are hard limits, not just defaults: a larger
`b` makes octree coordinates exceed the per-axis bit budget of the `UInt64` boundary-table keys
that [`creategrid`](@ref Ferrite.AMR.creategrid) uses to identify nodes across tree boundaries.
An out-of-range `b` therefore throws a `DomainError` rather than silently producing a grid with
wrongly merged nodes.
"""
struct ForestBWG{dim, C <: OctreeBWG, T <: Real} <: Ferrite.AbstractGrid{dim}
    cells::Vector{C}
    nodes::Vector{Node{dim, T}}
    # Sets
    cellsets::Dict{String, OrderedSet{Int}}
    nodesets::Dict{String, OrderedSet{Int}}
    facetsets::Dict{String, OrderedSet{Ferrite.FacetIndex}}
    vertexsets::Dict{String, OrderedSet{Ferrite.VertexIndex}}
    #Topology
    topology::ExclusiveTopology
end

function ForestBWG(grid::Ferrite.AbstractGrid{dim}, b = DEFAULT_MAXLEVEL[dim]) where {dim}
    cells = getcells(grid)
    C = eltype(cells)
    @assert isconcretetype(C)
    @assert (C == Quadrilateral && dim == 2) || (C == Hexahedron && dim == 3)
    topology = ExclusiveTopology(grid)
    cells = OctreeBWG.(grid.cells, b)
    nodes = getnodes(grid)
    cellsets = Ferrite.getcellsets(grid)
    nodesets = Ferrite.getnodesets(grid)
    facetsets = Ferrite.getfacetsets(grid)
    vertexsets = Ferrite.getvertexsets(grid)
    return ForestBWG(cells, nodes, cellsets, nodesets, facetsets, vertexsets, topology)
end

function Ferrite.get_facet_facet_neighborhood(g::ForestBWG{dim}) where {dim}
    return Ferrite._get_facet_facet_neighborhood(g.topology, Val(dim))
end

# Directly working with mixed-dimensional forests is not supported for now.
get_reference_dimension(g::ForestBWG, cellid::Int) = get_reference_dimension(g)

# Pack an integer octree coordinate into a single `UInt64` — the key of the per-tree
# *boundary node tables* that resolve node identity across tree boundaries (see
# `_merge_intertree_nodes!`). Node identity is integer/topological throughout (IBWG2015 §2:
# a point is an octant + boundary index; physical positions are emitted only at the very
# end), so an in-range coordinate packs losslessly:
#
# Octree coords lie in `[0, 2^b]` with `b ≤ DEFAULT_MAXLEVEL[dim]` (`30`/`19` for 2D/3D):
# 31 bits/axis × 2 = 62 bits (2D) and 21 bits/axis × 3 = 63 bits (3D) both fit a `UInt64` with
# no overlap. Works for `Int` and `Int32` coords alike (`UInt64` of equal values agree). Only
# ever called on in-range, non-negative coords (`_bnd_lookup` bounds-checks first).
@inline _packcoord(c::NTuple{2, <:Integer}) = UInt64(c[1]) | (UInt64(c[2]) << 31)
@inline _packcoord(c::NTuple{3, <:Integer}) = UInt64(c[1]) | (UInt64(c[2]) << 21) | (UInt64(c[3]) << 42)

"""
    refine_all!(forest::ForestBWG, l)

Uniformly refine every leaf currently at level `l - 1` across all trees of `forest`,
i.e. take a forest refined to level `l - 1` to level `l`. A convenience wrapper for a
uniform refinement; adaptive refinement of marked cells goes through the
`refine!(forest, cellids)` vector method.

Runs in `O(n)`: each tree's leaf list is rebuilt in a single pass (children spliced in
z-order in place of their parent, preserving Morton order) rather than via `n`
in-place `insert!`s, which would be `O(n^2)`.
"""
function refine_all!(forest::ForestBWG, l)
    for tree in forest.cells
        leaves = tree.leaves
        b = tree.b
        # Refine every level-(l-1) leaf in a single linear pass. Doing this with a
        # per-leaf `refine!` is O(n^2): each in-place `insert!` memmoves the array
        # tail. Instead, rebuild the leaf list once. Children are emitted in z-order
        # in the parent's slot, so the result stays Morton-sorted.
        any(leaf -> leaf.l == l - 1, leaves) || continue
        refined = similar(leaves, 0)
        sizehint!(refined, length(leaves))
        for leaf in leaves
            if leaf.l == l - 1 && leaf.l + 1 <= b
                for child in children(leaf, b)
                    push!(refined, child)
                end
            else
                push!(refined, leaf)
            end
        end
        resize!(leaves, length(refined))
        copyto!(leaves, refined)
    end
    return
end

"""
    refine!(forest::ForestBWG, cellid::Integer)

Refine the single leaf addressed by the global `cellid` (the same flat,
tree-major / Morton-within-tree numbering used by the grid from
[`creategrid`](@ref Ferrite.AMR.creategrid)). The owning tree is found from the per-tree
leaf counts; the leaf is then refined via the `refine_octant!(octree, octant)` method. To refine
several cells, use the vector method below — it is linear, whereas looping this one is
`O(n^2)`.
"""
function refine!(forest::ForestBWG, cellid::Integer)
    nleaves_k = length(forest.cells[1].leaves)
    prev_nleaves_k = 0
    k = 1
    while nleaves_k < cellid
        k += 1
        prev_nleaves_k = nleaves_k
        nleaves_k += length(forest.cells[k].leaves)
    end
    return refine_octant!(forest.cells[k], forest.cells[k].leaves[cellid - prev_nleaves_k])
end

"""
    refine!(forest::ForestBWG, cellids::AbstractVector{<:Integer})

Refine all leaves addressed by the global `cellids` — the production refinement entry
point, e.g. for the cells flagged by an error estimator in an adaptive FE loop. `cellids`
are global cell ids in the grid's flat numbering (tree-major, Morton-within-tree);
duplicates are ignored and ids at the maximum level `tree.b` are skipped.

Runs in `O(n + k)` for `n` leaves and `k = length(cellids)`: the sorted ids are mapped to
per-tree local indices in one merge pass and each tree's leaf list is rebuilt once
(children spliced in z-order in place of their parent, preserving Morton order). This
avoids the `O(n^2)` of refining cells one at a time, where every in-place `insert!`
memmoves the leaf-array tail. The caller's `cellids` vector is not modified (a sorted
copy is taken when needed).

Combine with [`balanceforest!`](@ref Ferrite.AMR.balanceforest!) to restore 2:1 balance and
[`coarsen!`](@ref Ferrite.AMR.coarsen!) for derefinement; both preserve the Morton-sorted
leaf invariant this method relies on.
"""
function refine!(forest::ForestBWG, cellids::AbstractVector{<:Integer})
    isempty(cellids) && return
    # Refine all marked cells in a single linear pass per tree. Refining them one at
    # a time (the cellid+shift loop) is O(n^2): every in-place `insert!` memmoves the
    # leaf-array tail. Instead, map the (sorted) global ids to per-tree local leaf
    # indices and rebuild each tree's leaf list once, splicing children in z-order
    # in place of their parent so the result stays Morton-sorted. `sort` (not `sort!`)
    # leaves the caller's marking vector untouched.
    marked = issorted(cellids) ? cellids : sort(cellids)
    cursor = 1                # cursor into `marked`
    offset = 0                # number of leaves in already-processed trees
    for tree in forest.cells
        leaves = tree.leaves
        n = length(leaves)
        # marked global ids in (offset, offset + n] belong to this tree
        first_marked = cursor
        while cursor <= length(marked) && marked[cursor] <= offset + n
            cursor += 1
        end
        if cursor > first_marked
            b = tree.b
            # NOT `length(children(leaves[1], b))`: that throws if the first leaf happens to
            # sit at the maximum level, even when every marked id is legitimately skippable.
            nchild = _nchildren(tree)
            refined = similar(leaves, 0)
            sizehint!(refined, n + (cursor - first_marked) * nchild)
            m = first_marked
            for localidx in 1:n
                leaf = leaves[localidx]
                if m < cursor && marked[m] - offset == localidx
                    while m < cursor && marked[m] - offset == localidx # skip duplicate ids
                        m += 1
                    end
                    if leaf.l + 1 <= b
                        for child in children(leaf, b)
                            push!(refined, child)
                        end
                        continue
                    end
                end
                push!(refined, leaf)
            end
            resize!(leaves, length(refined))
            copyto!(leaves, refined)
        end
        offset += n
    end
    return
end

"""
    coarsen!(forest::ForestBWG, cellids::AbstractVector{<:Integer}; require_all_siblings::Bool = true)

Coarsen the `2^dim`-sibling families addressed by the global `cellids` — the batch, derefinement
counterpart of [`refine!`](@ref Ferrite.AMR.refine!). `cellids` are global cell ids in the grid's
flat numbering (tree-major, Morton-within-tree); each replaces the family it belongs to with the
common parent (one level up).

`require_all_siblings` selects the trigger policy:
- `true` (default): a family is coarsened only if **all** `2^dim` of its children are in `cellids`
  (standard p4est semantics; never collapses un-flagged cells).
- `false`: a single marked sibling collapses its whole family (like `coarsen_octant!(octree, octant)`).

Only complete families count: a family is coarsened solely when its full `2^dim` same-level sibling
set is physically present and contiguous in the leaves. Ids addressing an incomplete family (e.g. a
sibling that was refined further) are silently skipped, mirroring how [`refine!`](@ref
Ferrite.AMR.refine!) skips ids already at the maximum level. Coarsening is one level per call and
duplicates are ignored.

Runs in `O(n + k)` like the vector [`refine!`](@ref Ferrite.AMR.refine!): the sorted ids are mapped
to per-tree local indices in one merge pass and each tree's leaf list is rebuilt once (families
collapsed to their parent in place, preserving Morton order). The caller's `cellids` vector is not
modified (a sorted copy is taken when needed). Combine with [`balanceforest!`](@ref
Ferrite.AMR.balanceforest!) to restore 2:1 balance before [`creategrid`](@ref
Ferrite.AMR.creategrid).
"""
function coarsen!(forest::ForestBWG, cellids::AbstractVector{<:Integer}; require_all_siblings::Bool = true)
    isempty(cellids) && return
    cmarked = issorted(cellids) ? cellids : sort(cellids)
    _apply_refine_coarsen!(forest, cmarked, eltype(cmarked)[], require_all_siblings)
    return
end

"""
    refine_and_coarsen!(forest::ForestBWG, coarsen_ids, refine_ids; balance = true, require_all_siblings = true)

Coarsen the families addressed by `coarsen_ids` and refine the leaves addressed by `refine_ids` in a
single pass, then (if `balance`, the default) restore 2:1 balance via
[`balanceforest!`](@ref Ferrite.AMR.balanceforest!). Both id vectors use the same global, flat cell
numbering (tree-major, Morton-within-tree) as [`refine!`](@ref Ferrite.AMR.refine!) and
[`coarsen!`](@ref Ferrite.AMR.coarsen!).

Doing both at once is the point: a global cell id encodes a leaf's position in the current leaf
lists, so as soon as one operation runs, every id at or after the touched leaf is stale. Calling
`coarsen!` and `refine!` back-to-back with ids gathered against one numbering would therefore
misfire. Instead this resolves **both** id sets against the original numbering and applies them in
one rebuild pass per tree, so no intermediate renumbering ever exists.

`require_all_siblings` has the same meaning as in [`coarsen!`](@ref Ferrite.AMR.coarsen!). The two id
sets must be disjoint, and no `refine_ids` entry may fall inside a family selected for coarsening;
either conflict throws an `ArgumentError`. Refinement and coarsening are one level per call.
"""
function refine_and_coarsen!(
        forest::ForestBWG, coarsen_ids::AbstractVector{<:Integer}, refine_ids::AbstractVector{<:Integer};
        balance::Bool = true, require_all_siblings::Bool = true
    )
    overlap = intersect(coarsen_ids, refine_ids)
    isempty(overlap) || throw(ArgumentError("`coarsen_ids` and `refine_ids` must be disjoint, but share $(sort!(collect(overlap)))."))
    if !(isempty(coarsen_ids) && isempty(refine_ids))
        cmarked = issorted(coarsen_ids) ? coarsen_ids : sort(coarsen_ids)
        rmarked = issorted(refine_ids) ? refine_ids : sort(refine_ids)
        _apply_refine_coarsen!(forest, cmarked, rmarked, require_all_siblings)
    end
    balance && balanceforest!(forest)
    return
end

# Shared driver for `coarsen!(forest, ids)` and `refine_and_coarsen!`: `cmarked`/`rmarked` are the
# (sorted) global coarsen/refine ids. Locate each tree's id range in one merge pass and rebuild the
# tree's leaves once. Either id vector may be empty.
function _apply_refine_coarsen!(forest::ForestBWG, cmarked::AbstractVector{<:Integer}, rmarked::AbstractVector{<:Integer}, require_all_siblings::Bool)
    ccursor = 1               # cursor into `cmarked`
    rcursor = 1               # cursor into `rmarked`
    offset = 0                # number of leaves in already-processed trees
    for tree in forest.cells
        n = length(tree.leaves)
        # global ids in (offset, offset + n] belong to this tree
        cfirst = ccursor
        while ccursor <= length(cmarked) && cmarked[ccursor] <= offset + n
            ccursor += 1
        end
        rfirst = rcursor
        while rcursor <= length(rmarked) && rmarked[rcursor] <= offset + n
            rcursor += 1
        end
        if ccursor > cfirst || rcursor > rfirst
            _refine_coarsen_tree!(tree, cmarked, cfirst, ccursor, rmarked, rfirst, rcursor, offset, require_all_siblings)
        end
        offset += n
    end
    return
end

# Single linear pass over one tree's Morton-sorted leaves, splicing children for refine marks and
# collapsing complete families for coarsen marks. Both operations act on contiguous windows and
# preserve Morton order, so the rebuilt leaf list stays sorted. Coarsening is only ever triggered at
# a family's first sibling, so the whole family is consumed contiguously.
function _refine_coarsen_tree!(
        tree::OctreeBWG, cmarked, cfirst, clast, rmarked, rfirst, rlast, offset, require_all_siblings::Bool
    )
    leaves = tree.leaves
    n = length(leaves)
    b = tree.b
    nchild = _nchildren(tree) # NOT via `children(leaves[1], b)`, which throws at max level
    buf = similar(leaves, 0)
    sizehint!(buf, n + (rlast - rfirst) * nchild)
    cc = cfirst               # cursor into `cmarked`; local index = cmarked[cc] - offset
    rc = rfirst               # cursor into `rmarked`; local index = rmarked[rc] - offset
    localidx = 1
    while localidx <= n
        leaf = leaves[localidx]
        # --- refine branch -------------------------------------------------------------
        if rc < rlast && rmarked[rc] - offset == localidx
            while rc < rlast && rmarked[rc] - offset == localidx   # skip duplicate ids
                rc += 1
            end
            if leaf.l + 1 <= b
                for child in children(leaf, b)
                    push!(buf, child)
                end
            else
                push!(buf, leaf)   # already at max level: keep verbatim
            end
            localidx += 1
            continue
        end
        # --- coarsen branch (only at a complete family's first sibling) -----------------
        winend = localidx + nchild - 1
        if leaf.l > 0 && child_id(leaf, b) == 1 && winend <= n && _is_complete_family(leaves, localidx, leaf, b, nchild)
            # distinct coarsen marks landing in the family window [localidx, winend]
            ncmarks = 0
            prev = -1
            tc = cc
            while tc < clast && cmarked[tc] - offset <= winend
                li = cmarked[tc] - offset
                li != prev && (ncmarks += 1; prev = li)
                tc += 1
            end
            do_coarsen = require_all_siblings ? (ncmarks == nchild) : (ncmarks >= 1)
            if do_coarsen
                # a refine id inside a coarsening family is a conflict (exact-id overlap is
                # already rejected up front; this catches refine marks on sibling leaves)
                if rc < rlast && rmarked[rc] - offset <= winend
                    throw(ArgumentError("cell $(rmarked[rc]) is marked for refinement but lies in a family marked for coarsening."))
                end
                push!(buf, parent(leaf, b))
                cc = tc            # consume all coarsen marks in the window
                localidx = winend + 1
                continue
            end
        end
        # --- keep branch ---------------------------------------------------------------
        while cc < clast && cmarked[cc] - offset == localidx   # drop marks that cannot coarsen
            cc += 1
        end
        push!(buf, leaf)
        localidx += 1
    end
    resize!(leaves, length(buf))
    copyto!(leaves, buf)
    return
end

# `leaves[i:i+nchild-1]` are exactly the `2^dim` children of `firstchild`'s parent, in z-order.
# `firstchild` must be a first sibling (`child_id == 1`), so `children(parent(firstchild))` starts
# with it and equals the leaf slice (both Morton/z-order) exactly when the family is intact.
function _is_complete_family(leaves, i, firstchild::OctantBWG, b, nchild)
    fam = children(parent(firstchild, b), b)
    @inbounds for j in 1:nchild
        leaves[i + j - 1] == fam[j] || return false
    end
    return true
end

"""
    _coarsen_all!(forest::ForestBWG)

Internal convenience for tests and development — not part of the public API.

Coarsen every `2^dim`-sibling family in `forest` by one level — the inverse of
[`refine_all!`](@ref Ferrite.AMR.refine_all!). Each leaf that is a first sibling
(`child_id == 1`) is replaced by its parent via [`coarsen!`](@ref Ferrite.AMR.coarsen!).

!!! warning
    This assumes every first sibling has its complete same-level family present, which
    holds for a uniformly refined forest but **not** for an arbitrary adaptively refined
    one. Calling it on a forest with incomplete families violates [`coarsen!`](@ref
    Ferrite.AMR.coarsen!)'s precondition and corrupts the leaf list. For selective
    derefinement, coarsen individual complete families with [`coarsen!`](@ref
    Ferrite.AMR.coarsen!) instead.
"""
function _coarsen_all!(forest::ForestBWG)
    for tree in forest.cells
        for leaf in tree.leaves
            if child_id(leaf, tree.b) == 1
                coarsen_octant!(tree, leaf)
            end
        end
    end
    return
end

Ferrite.getneighborhood(forest::ForestBWG, idx) = getneighborhood(forest.topology, forest, idx)

function Ferrite.getncells(grid::ForestBWG)
    numcells = 0
    for tree in grid.cells
        numcells += length(tree)
    end
    return numcells
end

"""
    getcells(forest::ForestBWG) -> Vector{OctantBWG}

Collect the leaf octants of all trees of `forest` into a single vector, in ascending cell id
order (tree by tree, Morton order within each tree) — i.e. the octant `getcells(forest)[i]`
materializes into cell `i` of [`creategrid`](@ref Ferrite.AMR.creategrid)`(forest)`.

!!! warning "Allocates on every call"
    This materializes a fresh vector of all leaves each time it is called — `O(n)` in the
    number of cells. Call it once and reuse the result instead of calling it inside loops.

The returned octants live in the coordinate system of their respective tree, so the scalar
`getcells(forest, cellid)` from the `AbstractGrid` interface is deliberately not supported
(an octant is not interpretable without its tree) and throws instead of falling back to
`forest.cells[cellid]`, which would inconsistently return a whole tree. Take
`forest.cells[k].leaves` for tree-local work.
"""
function Ferrite.getcells(forest::ForestBWG{dim, C}) where {dim, C}
    ncells = getncells(forest)
    nnodes = 2^dim
    cellvector = Vector{OctantBWG{dim, nnodes, eltype(C)}}(undef, ncells)
    o = one(eltype(C))
    cellid = o
    for tree in forest.cells
        for leaf in tree.leaves
            cellvector[cellid] = leaf
            cellid += o
        end
    end
    return cellvector
end

# Block the generic `getcells(grid, i) = grid.cells[i]` fallback: `forest.cells` are trees,
# not cells, so it would silently return a whole tree while `getcells(forest)` returns leaves.
function Ferrite.getcells(forest::ForestBWG, cellid::Union{Int, AbstractVector{Int}})
    throw(ArgumentError("getcells(forest, cellid) is not supported: a leaf octant is not interpretable without its tree. Use getcells(forest) for all leaves, or forest.cells[k].leaves for tree-local work."))
end
# All trees share one octree type (the `ForestBWG` constructor requires a concrete,
# uniform cell type), so the celltype does not depend on `i`.
Ferrite.getcelltype(grid::ForestBWG, i::Int) = eltype(grid.cells)

"""
    _treecorners(forest::ForestBWG{dim}, k::Integer) -> NTuple{2^dim, Vec{dim}}

Physical coordinates of macro-tree `k`'s `2^dim` corner nodes, in Ferrite's vertex order for the
tree's cell. These are the interpolation support points for [`_interp_treepoint`](@ref); indexing
`forest.nodes` through `forest.cells[k].nodes` directly keeps the result concrete and
allocation-free.
"""
@inline function _treecorners(forest::ForestBWG{dim}, k::Integer) where {dim}
    nodes = forest.nodes
    return ntuple(j -> get_node_coordinate(nodes[forest.cells[k].nodes[j]]), Val(2^dim))
end

"""
    _interp_treepoint(corners::NTuple{N, Vec{dim}}, b, vertex::NTuple{dim}) -> Vec{dim}

Map an integer octree coordinate `vertex` of a tree to physical space — the isoparametric ``Q_1``
geometry map of the macro element (tree). Two steps:

1. affine-scale the octree coordinate (in `[0, 2^b]^dim`, see [`_maximum_size`](@ref)) to the
   reference cube ``\\xi \\in [-1,1]^{dim}`` via ``\\xi = \\texttt{vertex} \\cdot 2/2^b - 1``;
2. interpolate the tree's physical `corners` with the bi-/trilinear Lagrange shape functions,
   ``x = \\sum_{j=1}^{N} N_j(\\xi)\\, \\texttt{corners}[j]``.

`corners` are the tree's `2^dim` physical corner nodes (see [`_treecorners`](@ref)), passed in
explicitly so the per-tree corners are computed once and reused for every node of the tree. This
is the single bridge from the integer/topological octree world into physical coordinates.
"""
@inline function _interp_treepoint(corners::NTuple{N, Vec{dim, V}}, b, vertex::NTuple{dim, <:Integer}) where {N, dim, V}
    ξ = Vec(vertex .* (convert(V, 2) / (2^b)) .- 1)
    return sum(j -> corners[j] * Ferrite.reference_shape_value(Lagrange{Ferrite.RefHypercube{dim}, 1}(), ξ, j), 1:N)
end

"""
    _bnd_lookup(bnd, coord::NTuple{dim, <:Integer}, b) -> Int

O(log) lookup in a tree's sorted boundary node table `bnd` (`(packed coord, provisional id)`
pairs, see [`_merge_intertree_nodes!`](@ref)): the provisional id of the node at integer
octree coord `coord`, or `0` if `coord` is out of the tree's `[0, 2^b]` range or is not a
node of that tree. Both happen routinely during the cross-tree merge: the
[`transform_facet`](@ref)/[`transform_edge`](@ref) images can land outside the neighbour's
domain, and a hanging node exists as a leaf vertex on the refined side of an interface only.
"""
@inline function _bnd_lookup(bnd::Vector{Tuple{UInt64, Int}}, coord::NTuple{dim, <:Integer}, b::Integer) where {dim}
    hilim = 1 << b
    for d in 1:dim
        (0 <= coord[d] <= hilim) || return 0
    end
    key = _packcoord(coord)
    lo = 1
    hi = length(bnd)
    @inbounds while lo <= hi
        mid = (lo + hi) >>> 1
        mkey = bnd[mid][1]
        if mkey < key
            lo = mid + 1
        elseif mkey > key
            hi = mid - 1
        else
            return bnd[mid][2]
        end
    end
    return 0
end

"""
    _merge_intertree_nodes!(forest::ForestBWG{dim}, bnd, alias)

Identify nodes shared across tree boundaries (`creategrid`'s cross-tree pass). For each tree `k`,
walk its root-vertex, root-face and (3D) root-edge neighbours; a node on a shared boundary is
matched to its image in the lower-index neighbour `k′` via [`transform_facet`](@ref)/
[`transform_corner`](@ref)/[`transform_edge`](@ref) (handling tree rotations), and aliased onto
that owner. Only the lower-index tree owns a shared node (`k > k′`), giving a single canonical id
per geometric node across all incident trees.

Node ids are looked up in `bnd`, the per-tree *boundary node tables*: `bnd[k]` holds
`(packed coord, provisional id)` for every node of tree `k` lying on its root boundary, sorted by
key (filled by the numbering traversal, see [`creategrid`](@ref)). This is the only node-lookup
structure of the whole materializer — `O(surface)` per tree instead of a global coordinate hash
map — and the walk itself visits only boundary leaves. The canonicalization is recorded in
`alias::Vector{Int}` (indexed by provisional id, identity-initialized): `alias[p]` is the
provisional id that `p` is merged onto, so the per-node canonical lookup in `creategrid` is an
array index.
"""
function _merge_intertree_nodes!(forest::ForestBWG{dim}, bnd::Vector{Vector{Tuple{UInt64, Int}}}, alias::Vector{Int}) where {dim}
    _perm = dim == 2 ? 𝒱₂_perm : 𝒱₃_perm
    _perminv = dim == 2 ? 𝒱₂_perm_inv : 𝒱₃_perm_inv
    node_map = dim < 3 ? node_map₂ : node_map₃
    facet_neighborhood = Ferrite.get_facet_facet_neighborhood(forest)
    # Only leaves touching the tree boundary can carry shared nodes; collect them once per
    # tree (Morton order preserved) so the per-face/per-edge walks below skip the interior.
    bleaves = [filter(o -> _touches_tree_boundary(o, tree.b), tree.leaves) for tree in forest.cells]
    for (k, tree) in enumerate(forest.cells)
        _vertices = vertices(root(dim), tree.b)
        # Vertex neighbors
        @debug println("Setting vertex neighbors for octree $k")
        for (v, vc) in enumerate(_vertices)
            vertex_neighbor = forest.topology.vertex_vertex_neighbor[k, node_map[v]]
            for (k′, v′) in vertex_neighbor
                @debug println("  pair $v $v′")
                if k > k′
                    new_v = vertex(root(dim), node_map[v′], tree.b)
                    # Root corners are vertices of the corner-touching leaf in both trees, so
                    # both lookups hit (0 would throw on the `alias` access).
                    alias[_bnd_lookup(bnd[k], vc, tree.b)] = alias[_bnd_lookup(bnd[k′], new_v, forest.cells[k′].b)]
                    @debug println("    Matching $vc (local) to $new_v (neighbor)")
                end
            end
        end
        if dim > 1
            _faces = faces(root(dim), tree.b)
            # Face neighbors
            @debug println("Updating face neighbors for octree $k")
            for (f, fc) in enumerate(_faces) # f in p4est notation
                # Skip boundary edges
                facet_neighbor_ = facet_neighborhood[k, _perm[f]]
                if length(facet_neighbor_) == 0
                    continue
                end
                @debug @assert length(facet_neighbor_) == 1
                k′, f′_ferrite = facet_neighbor_[1]
                f′ = _perminv[f′_ferrite]
                @debug println("  Neighboring tree: $k′, face $f′_ferrite (Ferrite)/$f′ (p4est)")
                if k > k′ # Owner
                    tree′ = forest.cells[k′]
                    for leaf in bleaves[k]
                        fnodes = face(leaf, f, tree.b)
                        if !contains_facet(fc, fnodes)
                            @debug println("  Rejecting leaf $leaf because its facet $fnodes is not on the octant boundary")
                            continue
                        end
                        neighbor_candidate = transform_facet(forest, k′, f′, leaf)
                        # Candidate must be the face opposite to f'
                        f′candidate = p4est_opposite_face_index(f′)
                        fnodes_neighbor = face(neighbor_candidate, f′candidate, tree′.b)
                        r = compute_face_orientation(forest, k, f)
                        @debug println("  Trying to match $fnodes (local) to $fnodes_neighbor (neighbor $neighbor_candidate)")
                        if dim == 2
                            for i in 1:ncorners_face2D
                                i′ = rotation_permutation(r, i)
                                p2 = _bnd_lookup(bnd[k′], fnodes_neighbor[i′], tree′.b)
                                if p2 != 0
                                    alias[_bnd_lookup(bnd[k], fnodes[i], tree.b)] = alias[p2]
                                end
                            end
                        else
                            for i in 1:ncorners_face3D
                                rotated_ξ = rotation_permutation(f′, f, r, i)
                                p2 = _bnd_lookup(bnd[k′], fnodes_neighbor[i], tree′.b)
                                if p2 != 0
                                    alias[_bnd_lookup(bnd[k], fnodes[rotated_ξ], tree.b)] = alias[p2]
                                end
                            end
                        end
                    end
                end
            end
        end
        if dim > 2
            # edge neighbors
            @debug println("Updating edge neighbors for octree $k")
            for (e, ec) in enumerate(edges(root(dim), tree.b)) # e in p4est notation
                # Skip boundary edges
                edge_neighbor_ = forest.topology.edge_edge_neighbor[k, edge_perm[e]]
                if length(edge_neighbor_) == 0
                    continue
                end
                @debug @assert length(edge_neighbor_) == 1
                k′, e′_ferrite = edge_neighbor_[1]
                e′ = edge_perm_inv[e′_ferrite]
                @debug println("  Neighboring tree: $k′, edge $e′_ferrite (Ferrite)/$e′ (p4est)")
                if k > k′ # Owner
                    tree′ = forest.cells[k′]
                    for leaf in bleaves[k]
                        # First we skip edges which are not on the current edge of the root element
                        enodes = edge(leaf, e, tree.b)
                        if !contains_edge(ec, enodes)
                            @debug println("  Rejecting leaf $leaf because its edge $enodes is not on the octant boundary")
                            continue
                        end
                        neighbor_candidate = transform_edge(forest, k, e, k′, e′, leaf, false)
                        # Candidate must be the edge opposite to e'
                        e′candidate = p4est_opposite_edge_index(e′)

                        enodes_neighbor = edge(neighbor_candidate, e′candidate, tree′.b)
                        r = compute_edge_orientation(forest, k, e, k′, e′)
                        @debug println("  Trying to match $enodes (local) to $enodes_neighbor (neighbor $neighbor_candidate)")
                        for i in 1:ncorners_edge
                            i′ = rotation_permutation(r, i)
                            p2 = _bnd_lookup(bnd[k′], enodes_neighbor[i′], tree′.b)
                            if p2 != 0
                                alias[_bnd_lookup(bnd[k], enodes[i], tree.b)] = alias[p2]
                            end
                        end
                    end
                end
            end
        end
    end
    return
end

"""
    _build_cells(::Type{CT}, E, node_map, final_of_prov, ::Val{NV}) -> Vector{CT}

Materialize the cell vector from the element-node matrix `E` (provisional ids, z-order slots):
column `gid` is cell `gid`'s connectivity, remapped through `final_of_prov` to the final node
ids and reordered to Ferrite's vertex order via `node_map`, wrapped in cell type `CT`
(`Quadrilateral`/`Hexahedron`). A top-level function barrier so the cell construction compiles
concretely (building cells in the type-unstable `creategrid` body boxes every cell).
"""
function _build_cells(::Type{CT}, E::Matrix{Int}, node_map, final_of_prov::Vector{Int}, ::Val{NV}) where {CT, NV}
    ncells = size(E, 2)
    cells = Vector{CT}(undef, ncells)
    @inbounds for gid in 1:ncells
        cells[gid] = CT(ntuple(i -> final_of_prov[E[node_map[i], gid]], Val(NV)))
    end
    return cells
end


"""
    reconstruct_facetsets(forest::ForestBWG{dim}) -> Dict{String, OrderedSet{FacetIndex}}

Transfer the macro-mesh facet sets onto the materialized (refined) grid. For each original
`FacetIndex` (tree, face), emit a `FacetIndex` for every leaf of that tree lying on the root
face, converting between p4est and Ferrite face ordering (`𝒱₂_perm`/`𝒱₃_perm`). This keeps named
boundaries (e.g. Dirichlet/Neumann sets) valid after refinement.

Staying inside one tree there is no rotation, so a leaf is on the root face iff its anchor lies on
that face's axis-aligned plane (`leaf.xyz[axis] == 0` for a low face, `== 2^b - leafsize` for a
high face), and the contributing local face index is exactly the root face index. This is an
`O(#leaves)` plane test, replacing a former `O(#leaves · 2dim)` [`contains_facet`](@ref) scan over
each leaf's `faces`.
"""
function reconstruct_facetsets(forest::ForestBWG{dim}) where {dim}
    _perm = dim == 2 ? 𝒱₂_perm : 𝒱₃_perm
    _perm_inv = dim == 2 ? 𝒱₂_perm_inv : 𝒱₃_perm_inv
    new_facesets = typeof(forest.facetsets)()
    for (facetsetname, facetset) in forest.facetsets
        new_facetset = typeof(facetset)()
        for facetidx in facetset
            pivot_tree = forest.cells[facetidx[1]]
            b = pivot_tree.b
            rootlen = _compute_size(b, 0)                       # 2^b, the root extent
            last_cellid = facetidx[1] != 1 ? sum(length, @view(forest.cells[1:(facetidx[1] - 1)])) : 0
            pivot_faceid = facetidx[2]
            # The root face in p4est ordering, and the axis-aligned plane it pins. p4est faces
            # pair up as (1,2)=(x-,x+), (3,4)=(y-,y+), (5,6)=(z-,z+): axis = (f-1)÷2+1, the odd
            # index is the low face (coord 0), the even index the high face (coord 2^b). A leaf
            # contributes a facet to this set iff it lies on that plane; staying inside one tree
            # there is no rotation, so the contributing local face index equals the root's (`f`)
            # — replacing the former O(#leaves · 2dim) `faces`/`contains_facet` scan (the loop
            # below is O(#leaves) with an O(1) plane test and no per-leaf allocation).
            f = _perm_inv[pivot_faceid]                         # p4est face index of the root face
            axis = (f - 1) ÷ 2 + 1
            is_low = isodd(f)
            ferrite_leaf_face_idx = _perm[f]                    # == pivot_faceid
            for (leaf_idx, leaf) in enumerate(pivot_tree.leaves)
                onface = is_low ? (leaf.xyz[axis] == 0) : (leaf.xyz[axis] + _compute_size(b, leaf.l) == rootlen)
                if onface
                    push!(new_facetset, FacetIndex(last_cellid + leaf_idx, ferrite_leaf_face_idx))
                end
            end
        end
        new_facesets[facetsetname] = new_facetset
    end
    return new_facesets
end

"""
    reconstruct_cellsets(forest::ForestBWG) -> Dict{String, OrderedSet{Int}}

Transfer the macro-mesh cell sets onto the materialized (refined) grid: every leaf inherits
the set membership of its tree (macro cell), so each macro cell id in a set is replaced by
the cell ids of that tree's leaves (contiguous by [`_element_offsets`](@ref)). This keeps
named subdomains (e.g. material regions) valid after refinement.
"""
function reconstruct_cellsets(forest::ForestBWG)
    offsets = _element_offsets(forest)
    new_cellsets = typeof(forest.cellsets)()
    for (cellsetname, cellset) in forest.cellsets
        new_cellset = typeof(cellset)()
        for k in cellset
            for leaf_idx in 1:length(forest.cells[k].leaves)
                push!(new_cellset, offsets[k] + leaf_idx)
            end
        end
        new_cellsets[cellsetname] = new_cellset
    end
    return new_cellsets
end

"""
    balance_corner(forest, k′, c′, o, s)   # and balance_face / balance_edge(forest, k, e, k′, e′, o, s)

Restore 2:1 balance across a single tree interface (corner, face or edge respectively). `s` is the
neighbour octant at pivot octant `o`'s level (`balance_edge` additionally takes the pivot tree `k`
and its edge `e`, needed to orient the along-edge coordinate); transformed into the neighbour tree `k′` (via
[`transform_corner`](@ref)/[`transform_facet`](@ref)/[`transform_edge`](@ref)) it is `s′`. If the
neighbour there is more than one level coarser than `o` — neither `s′` nor `parent(s′)` is a leaf
but `parent(parent(s′))` is — that grandparent leaf is [`refine_octant!`](@ref Ferrite.AMR.refine_octant!)ed, leaving the neighbour
exactly one level coarser. Level-1 pivots need no balancing.
"""
function balance_corner(forest, k′, c′, o, s)
    o.l == 1 && return # no balancing needed for pivot octant level == 1
    s′ = transform_corner(forest, k′, c′, s, true)
    neighbor_tree = forest.cells[k′]
    leaves = neighbor_tree.leaves
    return if !_in_leaves(leaves, s′) && !_in_leaves(leaves, parent(s′, neighbor_tree.b))
        gp = parent(parent(s′, neighbor_tree.b), neighbor_tree.b)
        if _in_leaves(leaves, gp)
            refine_octant!(neighbor_tree, gp)
        end
    end
end

function balance_face(forest, k′, f′, o, s)
    o.l == 1 && return # no balancing needed for pivot octant level == 1
    s′ = transform_facet(forest, k′, f′, s)
    neighbor_tree = forest.cells[k′]
    leaves = neighbor_tree.leaves
    return if !_in_leaves(leaves, s′) && !_in_leaves(leaves, parent(s′, neighbor_tree.b))
        gp = parent(parent(s′, neighbor_tree.b), neighbor_tree.b)
        if _in_leaves(leaves, gp)
            refine_octant!(neighbor_tree, gp)
        end
    end
end

function balance_edge(forest, k, e, k′, e′, o, s)
    o.l == 1 && return # no balancing needed for pivot octant level == 1
    s′ = transform_edge(forest, k, e, k′, e′, s, true)
    neighbor_tree = forest.cells[k′]
    leaves = neighbor_tree.leaves
    return if !_in_leaves(leaves, s′) && !_in_leaves(leaves, parent(s′, neighbor_tree.b))
        gp = parent(parent(s′, neighbor_tree.b), neighbor_tree.b)
        if _in_leaves(leaves, gp)
            refine_octant!(neighbor_tree, gp)
        end
    end
end

"""
    _touches_tree_boundary(o::OctantBWG{dim}, b) -> Bool

`true` iff octant `o` has a face on its tree's boundary (some axis anchor at `0` or at the root
extent `2^b`). Only such leaves can have out-of-tree neighbours, so [`balanceforest!`](@ref) uses
this to skip the interior leaves (the majority) when balancing across tree interfaces.
"""
function _touches_tree_boundary(o::OctantBWG{dim}, b) where {dim}
    h = _compute_size(b, o.l); m = _maximum_size(b)
    return any(d -> o.xyz[d] == 0 || o.xyz[d] + h == m, 1:dim)
end

"""
    _balance_leaf!(forest, k, tree, o, perm_face, perm_face_inv, perm_corner, perm_corner_inv, rootfaces, rootedges, rootvertices, facet_neighborhood)

Per-leaf kernel of [`balanceforest!`](@ref) handling the *inter-tree* part of the 2:1 balance.

Operates on a single "pivot" leaf `o` of tree `k`. In-tree balancing is already taken care
of by `balancetree`; this function only propagates balance across tree boundaries. It walks
the `possibleneighbors` of `o`, keeps those lying outside the current tree (reachable only
through a corner/face/edge connection to another tree), decodes the neighbour type from the
`possibleneighbors` index `s_i`, maps the pivot's local index into the neighbour tree via the
permutation tables, and calls `balance_face`/`balance_corner`/`balance_edge` to refine the
neighbour tree where the balance condition requires it.
"""
function _balance_leaf!(forest::ForestBWG{dim}, k, tree, o, perm_face, perm_face_inv, perm_corner, perm_corner_inv, rootfaces, rootedges, rootvertices, facet_neighborhood) where {dim}
    ss = possibleneighbors(o, o.l, tree.b)
    # s_i indexes possibleneighbors (encodes the neighbourhood type); skip in-tree neighbours inline.
    for (s_i, s) in enumerate(ss)
        inside(s, tree.b) && continue
        if dim == 2
            if s_i <= 4 #corner neighbor, only true for 2D see possibleneighbors
                if vertex(o, s_i, tree.b) == rootvertices[s_i]
                    # pivot corner at the tree's corner: balance across the macro (pure vertex)
                    # connections of the root topology
                    for corner_connection in forest.topology.vertex_vertex_neighbor[k, perm_corner[s_i]]
                        k′, c′ = corner_connection[1], perm_corner_inv[corner_connection[2]]
                        balance_corner(forest, k′, c′, o, s)
                    end
                else
                    # corner connection newly introduced by refinement (the pivot corner lies
                    # interior to a root face, not on a macro vertex): the diagonal octant
                    # leaves the tree through a face touching the corner, so route it through
                    # the face transform of that face's neighbour tree
                    pivot_faces = faces(o, tree.b)
                    for j in 1:2
                        face_idx = 𝒱₂_inv[s_i, j] # the two faces touching corner s_i
                        contains_facet(rootfaces[face_idx], pivot_faces[face_idx]) || continue
                        fc = facet_neighborhood[k, perm_face[face_idx]]
                        isempty(fc) && continue
                        @assert length(fc) == 1
                        fc = fc[1]
                        k′, f′ = fc[1], perm_face_inv[fc[2]]
                        balance_face(forest, k′, f′, o, s)
                    end
                end
            else # face neighbor, only true for 2D
                s_i -= 4
                fc = facet_neighborhood[k, perm_face[s_i]]
                isempty(fc) && continue
                @assert length(fc) == 1
                fc = fc[1]
                k′, f′ = fc[1], perm_face_inv[fc[2]]
                balance_face(forest, k′, f′, o, s)
            end
        else # the 3D branch mirrors the 2D one above; unifying them is tracked in #1408
            if s_i <= 8 #corner neighbor
                if vertex(o, s_i, tree.b) == rootvertices[s_i]
                    # pivot corner at the tree's corner: balance across the macro (pure vertex)
                    # connections of the root topology
                    for corner_connection in forest.topology.vertex_vertex_neighbor[k, perm_corner[s_i]]
                        k′, c′ = corner_connection[1], perm_corner_inv[corner_connection[2]]
                        balance_corner(forest, k′, c′, o, s)
                    end
                else
                    # corner connection newly introduced by refinement (the pivot corner lies
                    # interior to a root face or root edge, not on a macro vertex): the diagonal
                    # octant leaves the tree through a face or edge touching the corner, so
                    # route it through the respective face/edge transform of that neighbour
                    # tree (the transform of a route the octant does not leave through lands
                    # outside the neighbour's root and is a no-op in balance_face/balance_edge)
                    pivot_faces = faces(o, tree.b)
                    for j in 1:3
                        face_idx = 𝒱₃_inv[s_i, j] # the three faces touching corner s_i
                        contains_facet(rootfaces[face_idx], pivot_faces[face_idx]) || continue
                        fc = facet_neighborhood[k, perm_face[face_idx]]
                        isempty(fc) && continue
                        @assert length(fc) == 1
                        fc = fc[1]
                        k′, f′ = fc[1], perm_face_inv[fc[2]]
                        balance_face(forest, k′, f′, o, s)
                    end
                    for j in 1:3
                        edge_idx = 𝒰_inv[s_i, j] # the three edges touching corner s_i
                        contains_edge(rootedges[edge_idx], edge(o, edge_idx, tree.b)) || continue
                        for edge_connection in forest.topology.edge_edge_neighbor[k, edge_perm[edge_idx]]
                            k′, e′ = edge_connection[1], edge_perm_inv[edge_connection[2]]
                            balance_edge(forest, k, edge_idx, k′, e′, o, s)
                        end
                    end
                end
            elseif 8 < s_i <= 14
                s_i -= 8
                fc = facet_neighborhood[k, perm_face[s_i]]
                isempty(fc) && continue
                @assert length(fc) == 1
                fc = fc[1]
                k′, f′ = fc[1], perm_face_inv[fc[2]]
                balance_face(forest, k′, f′, o, s)
            else
                s_i -= 14
                ec = forest.topology.edge_edge_neighbor[k, edge_perm[s_i]]
                pivot_edge = edge(o, s_i, tree.b)
                if !contains_edge(rootedges[s_i], pivot_edge) # pivot edge interior to a root face, not an octree edge
                    handled = false
                    for (face_idx, rf) in enumerate(rootfaces)
                        face_contains_edge(rf, pivot_edge) || continue
                        handled = true
                        fc = facet_neighborhood[k, perm_face[face_idx]]
                        isempty(fc) && continue
                        @assert length(fc) == 1
                        fc = fc[1]
                        k′, f′ = fc[1], perm_face_inv[fc[2]]
                        balance_face(forest, k′, f′, o, s)
                    end
                    handled && continue
                end
                isempty(ec) && continue
                for edge_connection in ec
                    !contains_edge(rootedges[s_i], pivot_edge) && continue
                    k′, e′ = edge_connection[1], edge_perm_inv[edge_connection[2]]
                    balance_edge(forest, k, s_i, k′, e′, o, s)
                end
            end
        end
    end
    return
end

# Reusable scratch for `balancetree`, allocated once in `balanceforest!` and reused across
# every tree and pass. Without reuse, the per-level `push!`/`append!` (and `unique!`'s hash
# table) reallocate on every one of the hundreds of `balancetree` calls.
struct BalanceBuffers{OT, K}
    keybuf::Vector{K}     # (Morton-anchor, level) sort keys
    permbuf::Vector{Int}
    scratch::Vector{OT}
    W::Vector{OT}
    P::Vector{OT}
    R::Vector{OT}
    Q::Vector{OT}
    T::Vector{OT}
    Tparents::Set{OT}
    seen::Set{OT}
    inds::Vector{Int}
end

function BalanceBuffers(s0::OT) where {OT <: OctantBWG}
    K = Tuple{typeof(morton(s0, s0.l, s0.l)), typeof(s0.l)}
    return BalanceBuffers{OT, K}(K[], Int[], OT[], OT[], OT[], OT[], OT[], OT[], Set{OT}(), Set{OT}(), Int[])
end

"""
    balanceforest!(forest::ForestBWG)

Enforce the 2:1 balance condition across the whole forest: no two leaves sharing a face, edge or
corner may differ by more than one refinement level. Each tree is balanced internally
(`balancetree`); boundary leaves ([`_touches_tree_boundary`](@ref)) are additionally balanced
against their out-of-tree neighbours via [`_balance_leaf!`](@ref). Iterated to a fixed point, then
duplicate/over-refined leaves are pruned and re-sorted into Morton order.

A balance refinement can itself create new 2:1 violations (also in trees processed earlier in the
same pass), which is why the outer loop reruns until a whole pass adds no cells. This ripple can
cascade across the domain, but it always terminates: balancing only ever *refines*, the cell
count is strictly increasing across repeated passes, and it is bounded above by the uniformly
max-refined forest. On return the invariant holds globally — the non-conformity level after
`refine!` + `balanceforest!` is always exactly one, independent of the refinement history.

Algorithm 17 of [BWG2011](@citet).
"""
function balanceforest!(forest::ForestBWG{dim}) where {dim}
    perm_face = dim == 2 ? 𝒱₂_perm : 𝒱₃_perm
    perm_face_inv = dim == 2 ? 𝒱₂_perm_inv : 𝒱₃_perm_inv
    perm_corner = dim == 2 ? node_map₂ : node_map₃
    perm_corner_inv = dim == 2 ? node_map₂_inv : node_map₃_inv
    root_ = root(dim)
    nrefcells = 0
    facet_neighborhood = Ferrite.get_facet_facet_neighborhood(forest)
    # `balancetree` scratch, allocated once and reused across every tree and pass.
    bb = BalanceBuffers(forest.cells[1].leaves[1])
    while nrefcells - getncells(forest) != 0
        nrefcells = getncells(forest)
        for k in 1:length(forest.cells)
            tree = forest.cells[k]
            rootfaces = faces(root_, tree.b)
            rootedges = dim == 3 ? edges(root_, tree.b) : nothing
            rootvertices = vertices(root_, tree.b)
            balanced = balancetree(tree, bb)
            forest.cells[k] = balanced
            for o in forest.cells[k].leaves
                # Only leaves touching the tree boundary can have out-of-tree neighbours;
                # skip the interior (the majority) → no possibleneighbors/findall there.
                _touches_tree_boundary(o, tree.b) || continue
                _balance_leaf!(forest, k, tree, o, perm_face, perm_face_inv, perm_corner, perm_corner_inv, rootfaces, rootedges, rootvertices, facet_neighborhood)
            end
        end
    end
    return
end

# Sort octants in place by (Morton-anchor key, level) — the same total order as
# `isless`, but computing each Morton key once instead of letting `sort!` call
# `morton` twice per comparison (`morton` is a ~b·dim-bit interleave).
function _sort_by_morton!(v::Vector{OT}, keybuf::Vector, permbuf::Vector{Int}, scratch::Vector{OT}) where {OT <: OctantBWG}
    n = length(v)
    n < 2 && return v
    resize!(keybuf, n)
    @inbounds for i in 1:n
        o = v[i]
        keybuf[i] = (morton(o, o.l, o.l), o.l)
    end
    resize!(permbuf, n)
    sortperm!(permbuf, keybuf; alg = QuickSort)
    resize!(scratch, n)
    @inbounds for i in 1:n
        scratch[i] = v[permbuf[i]]
    end
    copyto!(v, scratch)
    return v
end

# Drop all octants at level `lm1` from `W`, compacting in place. Unlike `filter!`, `resize!`
# down keeps the buffer capacity, so the reused `W` is not reallocated by the next `append!`.
function _drop_level!(W, lm1)
    j = 0
    @inbounds for x in W
        if x.l != lm1
            j += 1
            W[j] = x
        end
    end
    return resize!(W, j)
end

# Order-preserving in-place dedup with a reused `Set`, replacing `unique!` (which allocates
# a fresh hash table on every call).
function _unique!(P, seen)
    empty!(seen)
    j = 0
    @inbounds for x in P
        if x ∉ seen
            push!(seen, x)
            j += 1
            P[j] = x
        end
    end
    return resize!(P, j)
end

"""
Algorithm 7 of [SSB2008](@citet)
"""
function balancetree(tree::OctreeBWG)
    length(tree.leaves) == 1 && return tree
    return balancetree(tree, BalanceBuffers(tree.leaves[1]))
end

function balancetree(tree::OctreeBWG, bb::BalanceBuffers)
    length(tree.leaves) == 1 && return tree
    W, P, R, Q, T = bb.W, bb.P, bb.R, bb.Q, bb.T
    empty!(W)
    append!(W, tree.leaves)
    empty!(P)
    empty!(R)
    for l in tree.b:-1:1 # levels b..1 as in Alg. 3 of [BWG2011]; a level-0 leaf is the root
        # itself, so there is nothing coarser left to balance against
        empty!(Q)
        for o in W
            o.l == l && push!(Q, o)
        end
        _sort_by_morton!(Q, bb.keybuf, bb.permbuf, bb.scratch)
        # T: one representative per distinct parent (first in Q order)
        empty!(T); empty!(bb.Tparents)
        for x in Q
            p = parent(x, tree.b)
            if p ∉ bb.Tparents
                push!(T, x)
                push!(bb.Tparents, p)
            end
        end
        for t in T
            append!(R, children(parent(t, tree.b), tree.b)) # == t and its siblings
            for nb in possibleneighbors(parent(t, tree.b), l - 1, tree.b)
                inside(nb, tree.b) && push!(P, nb)
            end
        end
        append!(P, x for x in W if x.l == l - 1)
        _drop_level!(W, l - 1) # capacity-preserving in-place filter (see note above)
        _unique!(P, bb.seen)
        append!(W, P)
        empty!(P)
    end
    _sort_by_morton!(R, bb.keybuf, bb.permbuf, bb.scratch) # (morton-anchor, level) key disambiguates at max depth
    linearise!(R, tree.b, bb.inds)
    return OctreeBWG(copy(R), tree.b, tree.nodes) # copy: R is the reused buffer; the tree owns its leaves
end

"""
Algorithm 8 of [SSB2008](@citet)

Inverted the algorithm to delete! instead of add incrementally to a new array
"""
function linearise!(leaves::Vector{T}, b, inds) where {T <: OctantBWG}
    empty!(inds)
    @inbounds for i in 1:(length(leaves) - 1)
        isancestor(leaves[i], leaves[i + 1], b) && push!(inds, i)
    end
    return deleteat!(leaves, inds)
end

function Base.show(io::IO, ::MIME"text/plain", agrid::ForestBWG)
    println(io, "ForestBWG with ")
    println(io, "   $(getncells(agrid)) cells")
    return println(io, "   $(length(agrid.cells)) trees")
end

"""
    compute_face_orientation(forest::ForestBWG, k::Integer, f::Integer)
Determine the face orientation of face `f` from octree `k` following definition 2.1 from
[BWG2011](@citet), by comparing the macro node numbers of the two incident trees. Deliberately
simple; a faster table-driven version (Table 3 of the paper) is possible if this ever shows up
in profiles.
"""
function compute_face_orientation(forest::ForestBWG{<:Any, <:OctreeBWG{dim, <:Any, T2}}, k::T1, f::T1) where {dim, T1, T2}
    f_perm = (dim == 2 ? 𝒱₂_perm : 𝒱₃_perm)
    f_perminv = (dim == 2 ? 𝒱₂_perm_inv : 𝒱₃_perm_inv)
    n_perm = (dim == 2 ? node_map₂ : node_map₃)
    n_perminv = (dim == 2 ? node_map₂_inv : node_map₃_inv)

    f_ferrite = f_perm[f]
    facet_neighbor_table = Ferrite.get_facet_facet_neighborhood(forest)
    k′, f′_ferrite = facet_neighbor_table[k, f_ferrite][1]
    f′ = f_perminv[f′_ferrite]
    reffacenodes = reference_faces_bwg(Ferrite.RefHypercube{dim})
    nodes_f = ntuple(i -> forest.cells[k].nodes[n_perm[reffacenodes[f][i]]], length(reffacenodes[f]))
    nodes_f′ = ntuple(i -> forest.cells[k′].nodes[n_perm[reffacenodes[f′][i]]], length(reffacenodes[f′]))
    if f > f′
        return T2(findfirst(isequal(nodes_f′[1]), nodes_f) - 1)
    else
        return T2(findfirst(isequal(nodes_f[1]), nodes_f′) - 1)
    end
end

"""
    compute_edge_orientation(forest::ForestBWG, k::Integer, e::Integer, k′::Integer, e′::Integer)
    compute_edge_orientation(forest::ForestBWG, k::Integer, e::Integer)
Determine the edge orientation of edge `e` from octree `k` following the definition below Table 3
[BWG2011](@citet), by comparing the macro node numbers of the two incident trees (deliberately
simple, like [`compute_face_orientation`](@ref)).

`0` if trees `k` and `k′` traverse the shared macro edge (edge `e` of `k`, edge `e′` of `k′`, both
in BWG numbering) in the same direction, `1` if in opposite directions. The two-argument form
orients against `edge_edge_neighbor[k, e][1]`, which is only well defined when exactly two trees
share the macro edge.
"""
function compute_edge_orientation(forest::ForestBWG{<:Any, <:OctreeBWG{3, <:Any, T2}}, k::T1, e::T1, k′::T1, e′::T1) where {T1, T2}
    nodes_e = ntuple(i -> forest.cells[k].nodes[node_map₃[𝒰[e, i]]], ncorners_edge)
    nodes_e′ = ntuple(i -> forest.cells[k′].nodes[node_map₃[𝒰[e′, i]]], ncorners_edge)
    if nodes_e == nodes_e′
        s = T2(0)
    else
        s = T2(1)
    end
    return s
end

function compute_edge_orientation(forest::ForestBWG{<:Any, <:OctreeBWG{3, <:Any, T2}}, k::T1, e::T1) where {T1, T2}
    k′, e′_ferrite = forest.topology.edge_edge_neighbor[k, edge_perm[e]][1]
    return compute_edge_orientation(forest, k, e, T1(k′), T1(edge_perm_inv[e′_ferrite]))
end

"""
    transform_facet_remote(forest::ForestBWG, k::T1, f::T1, o::OctantBWG{dim, N, T2}) -> OctantBWG{dim, N, T2}
    transform_facet_remote(forest::ForestBWG, f::FacetIndex, o::OctantBWG{dim, N, T2}) -> OctantBWG{dim, N, T2}
Interoctree coordinate transformation of an given octant `o` to the face-neighboring of octree `k` by virtually pushing `o`s coordinate system through `k`s face `f`.
Implements Algorithm 8 of [BWG2011](@citet).

    x-------x-------x
    |       |       |
    |   3   |   4   |
    |       |       |
    x-------x-------x
    |       |       |
    |   1   *   2   |
    |       |       |
    x-------x-------x

Consider 4 octrees with a single leaf each and a maximum refinement level of 1
This function transforms octant 1 into the coordinate system of octant 2 by specifying `k=2` and `f=1`.
While in the own octree coordinate system octant 1 is at `xyz=(0,0)`, the returned and transformed octant is located at `xyz=(-2,0)`
"""
function transform_facet_remote(forest::ForestBWG, k::T1, f::T1, o::OctantBWG{dim, N, T2}) where {dim, N, T1 <: Integer, T2 <: Integer}
    _one = one(T2)
    _two = T2(2)
    _perm = (dim == 2 ? 𝒱₂_perm : 𝒱₃_perm)
    _perminv = (dim == 2 ? 𝒱₂_perm_inv : 𝒱₃_perm_inv)
    facet_neighbor_table = Ferrite.get_facet_facet_neighborhood(forest)
    k′, f′ = facet_neighbor_table[k, _perm[f]][1]
    f′ = _perminv[f′]
    s′ = _one - (_face_side(f) ⊻ _face_side(f′))
    s = zeros(T2, dim - 1)
    a = zeros(T2, 3) # Coordinate axes of f
    b = zeros(T2, 3) # Coordinate axes of f'
    r = compute_face_orientation(forest, k, f)
    a[3] = _face_axis(f); b[3] = _face_axis(f′) # origin and target normal axis
    if dim == 2
        a[1] = 1 - a[3]; b[1] = 1 - b[3]; s[1] = r
    else
        a[1] = (f < 3) ? 1 : 0; a[2] = (f < 5) ? 2 : 1
        u = (ℛ[1, f] - _one) ⊻ (ℛ[1, f′] - _one) ⊻ (((r == 0) | (r == 3)))
        b[u + 1] = (f′ < 3) ? 1 : 0; b[1 - u + 1] = (f′ < 5) ? 2 : 1 # r = 0 -> index 1
        if ℛ[f, f′] == 1 + 1 # R is one-based
            s[2] = r & 1; s[1] = r & 2
        else
            s[1] = r & 1; s[2] = r & 2
        end
    end
    maxlevel = forest.cells[1].b
    l = o.l; g = 2^maxlevel - 2^(maxlevel - l)
    xyz = zeros(T2, dim)
    xyz[b[1] + _one] = T2((s[1] == 0) ? o.xyz[a[1] + _one] : g - o.xyz[a[1] + _one])
    xyz[b[3] + _one] = T2((_two * _face_side(f′) - _one) * 2^maxlevel + s′ * g + (1 - 2 * s′) * o.xyz[a[3] + _one])
    if dim == 2
        return OctantBWG(l, (xyz[1], xyz[2]))
    else
        xyz[b[2] + _one] = T2((s[2] == 0) ? o.xyz[a[2] + _one] : g - o.xyz[a[2] + _one])
        return OctantBWG(l, (xyz[1], xyz[2], xyz[3]))
    end
end

transform_facet_remote(forest::ForestBWG, f::FacetIndex, oct::OctantBWG) = transform_facet_remote(forest, f[1], f[2], oct)

"""
    transform_facet(forest::ForestBWG, k', f', o::OctantBWG) -> OctantBWG
    transform_facet(forest::ForestBWG, f'::FacetIndex, o::OctantBWG) -> OctantBWG
Interoctree coordinate transformation of an given octant `o` that lies outside of the pivot octree `k`, namely in neighbor octree `k'`.
However, the coordinate of `o` is given in octree coordinates of `k`.
Thus, this algorithm implements the transformation of the octree coordinates of `o` into the octree coordinates of `k'`.
Useful in order to check whether or not a possible neighbor exists in a neighboring octree.
Implements Algorithm 8 of [BWG2011](@citet).

    x-------x-------x
    |       |       |
    |   3   |   4   |
    |       |       |
    x-------x-------x
    |       |       |
    |   1   *   2   |
    |       |       |
    x-------x-------x

Consider 4 octrees with a single leaf each and a maximum refinement level of 1
This function transforms octant 1 into the coordinate system of octant 2 by specifying `k=1` and `f=2`.
While from the perspective of octree coordinates `k=2` octant 1 is at `xyz=(-2,0)`, the returned and transformed octant is located at `xyz=(0,0)`
"""
function transform_facet(forest::ForestBWG, k::T1, f::T1, o::OctantBWG{2, <:Any, T2}) where {T1 <: Integer, T2 <: Integer}
    _one = one(T2)
    _two = T2(2)
    _perm = 𝒱₂_perm
    _perminv = 𝒱₂_perm_inv
    k′, f′ = forest.topology.edge_edge_neighbor[k, _perm[f]][1]
    f′ = _perminv[f′]

    r = compute_face_orientation(forest, k, f)
    # Coordinate axes of f
    a = (
        f ≤ 2, # tangent
        f > 2,  # normal
    )
    a_sign = _two * _face_side(f) - _one # -1 for a lower face, +1 for an upper face
    # Coordinate axes of f'
    b = (
        f′ ≤ 2, # tangent
        f′ > 2,  # normal
    )

    maxlevel = forest.cells[1].b
    depth_offset = 2^maxlevel - 2^(maxlevel - o.l)

    s′ = _one - (_face_side(f) ⊻ _face_side(f′)) # arithmetic switch: 1 iff the two faces sit on the same side of their axes

    # Scattering the values into positions `a` (Algorithm 8 writes result[a[i]] = value[i])
    # is here the same as gathering them from positions `a` below: `a` permutes the two
    # axes {0, 1}, and every permutation of two elements is its own inverse.
    xyz = (
        T2((r == 0) ? o.xyz[b[1] + _one] : depth_offset - o.xyz[b[1] + _one]),
        T2(a_sign * 2^maxlevel + s′ * depth_offset + (1 - 2 * s′) * o.xyz[b[2] + _one]),
    )
    return OctantBWG(o.l, (xyz[a[1] + _one], xyz[a[2] + _one]))
end

function transform_facet(forest::ForestBWG, k::T1, f::T1, o::OctantBWG{3, <:Any, T2}) where {T1 <: Integer, T2 <: Integer}
    _one = one(T2)
    _two = T2(2)
    _perm = 𝒱₃_perm
    _perminv = 𝒱₃_perm_inv
    k′, f′ = forest.topology.face_face_neighbor[k, _perm[f]][1]
    f′ = _perminv[f′]
    s′ = _one - (_face_side(f) ⊻ _face_side(f′))
    r = compute_face_orientation(forest, k, f)

    # Coordinate axes of f
    a = (
        (f ≤ 2) ? 1 : 0,
        (f ≤ 4) ? 2 : 1,
        _face_axis(f),
    )
    a_sign = _two * _face_side(f) - _one # -1 for a lower face, +1 for an upper face

    # Coordinate axes of f'
    # Whether the two tangent axes of f swap when mapped onto f′, decided by the alignment
    # cases ℛ of the two faces together with the orientation r (Table 3 of [BWG2011]).
    b = if Bool(ℛ[1, f] - _one) ⊻ Bool(ℛ[1, f′] - _one) ⊻ (((r == 0) || (r == 3)))
        (
            (f′ < 5) ? 2 : 1,
            (f′ < 3) ? 1 : 0,
            _face_axis(f′),
        )
    else
        (
            (f′ < 3) ? 1 : 0,
            (f′ < 5) ? 2 : 1,
            _face_axis(f′),
        )
    end

    s = if ℛ[f, f′] == 1 + 1 # R is one-based
        (r & 2, r & 1)
    else
        (r & 1, r & 2)
    end
    maxlevel = forest.cells[1].b
    depth_offset = 2^maxlevel - 2^(maxlevel - o.l)
    v1 = T2((s[1] == 0) ? o.xyz[b[1] + _one] : depth_offset - o.xyz[b[1] + _one])
    v2 = T2((s[2] == 0) ? o.xyz[b[2] + _one] : depth_offset - o.xyz[b[2] + _one])
    v3 = T2(a_sign * 2^maxlevel + s′ * depth_offset + (1 - 2 * s′) * o.xyz[b[3] + _one])
    xyz = ntuple(p -> (p == a[1] + _one ? v1 : (p == a[2] + _one ? v2 : v3)), Val(3))
    return OctantBWG(o.l, xyz)
end

transform_facet(forest::ForestBWG, f::FacetIndex, oct::OctantBWG) = transform_facet(forest, f[1], f[2], oct)

"""
    transform_corner(forest, k, c, oct, inside::Bool)
    transform_corner(forest, v::VertexIndex, oct, inside::Bool)

Algorithm 12 but with flipped logic in [BWG2011](@citet) to transform corner into different octree coordinate system
Implements flipped logic in the sense of pushing the Octant `oct` through vertex v and stays within octree coordinate system `k`.

`c` is the corner of tree `k` (in BWG corner numbering) at the shared vertex; `oct` is placed at
that corner of `k`, inside the root (`inside = true`) or diagonally outside of it (`inside = false`).
A corner octant is fully determined by the corner index and the level, so no connectivity lookup is
needed — in particular the corner must not be re-derived from `vertex_vertex_neighbor[k, ...][1]`,
which is ambiguous (and wrong) as soon as more than two trees meet at the vertex.
"""
function transform_corner(forest::ForestBWG, k::T1, c::T1, oct::OctantBWG{dim, N, T2}, inside::Bool) where {dim, N, T1 <: Integer, T2 <: Integer}
    b = forest.cells[k].b
    l = oct.l; g = 2^b - 2^(b - l)
    h⁻ = inside ? 0 : -2^(b - l); h⁺ = inside ? g : 2^b
    xyz = ntuple(i -> (_corner_side(c, i) == 0) ? h⁻ : h⁺, dim)
    return OctantBWG(l, xyz)
end

transform_corner(forest::ForestBWG, v::VertexIndex, oct::OctantBWG, inside) = transform_corner(forest, v[1], v[2], oct, inside)

"""
    transform_corner_remote(forest, k, c, oct, inside::Bool)
    transform_corner_remote(forest, v::VertexIndex, oct, inside::Bool)

Algorithm 12 in [BWG2011](@citet) to transform corner into different octree coordinate system.
Follows exactly the version of the paper by taking `oct` and looking from the neighbor octree coordinate system (neighboring to `k`,`v`) at `oct`.
"""
function transform_corner_remote(forest::ForestBWG, k::T1, c::T1, oct::OctantBWG{dim, N, T2}, inside::Bool) where {dim, N, T1 <: Integer, T2 <: Integer}
    _perm = dim == 2 ? node_map₂ : node_map₃
    _perminv = dim == 2 ? node_map₂_inv : node_map₃_inv
    k′, c′′ = forest.topology.vertex_vertex_neighbor[k, _perm[c]][1]
    c′ = _perminv[c′′] # assign c′ once so the ntuple closure below doesn't box it
    b = forest.cells[k].b
    l = oct.l; g = 2^b - 2^(b - l)
    h⁻ = inside ? 0 : -2^(b - l); h⁺ = inside ? g : 2^b
    xyz = ntuple(i -> (_corner_side(c′, i) == 0) ? h⁻ : h⁺, dim)
    return OctantBWG(l, xyz)
end

transform_corner_remote(forest::ForestBWG, v::VertexIndex, oct::OctantBWG, inside) = transform_corner_remote(forest, v[1], v[2], oct, inside)


"""
    transform_edge_remote(forest, k, e, oct, inside::Bool)
    transform_edge_remote(forest, e::EdgeIndex, oct, inside::Bool)

Algorithm 10 in [BWG2011](@citet) to transform edge into different octree coordinate system.
This function looks at the octant from the octree coordinate system of the neighbor that can be found at (k,e)
"""
function transform_edge_remote(forest::ForestBWG, k::T1, e::T1, oct::OctantBWG{3, N, T2}, inside::Bool) where {N, T1 <: Integer, T2 <: Integer}
    _four = T2(4)
    _one = T2(1)
    _two = T2(2)
    z = zero(T2)
    e_perm = edge_perm
    e_perminv = edge_perm_inv

    e_ferrite = e_perm[e]
    k′, e′_ferrite = forest.topology.edge_edge_neighbor[k, e_ferrite][1]
    e′ = e_perminv[e′_ferrite]
    #see Algorithm 9, line 18
    𝐛 = (
        _edge_axis(e′),
        e′ - _one < 4 ? 1 : 0,
        e′ - _one < 8 ? 2 : 1,
    )
    a₀ = _edge_axis(e) + _one # 1-based along-edge axis
    b = forest.cells[k].b
    l = oct.l; g = _two^b - _two^(b - l)
    h⁻ = inside ? z : -_two^(b - l); h⁺ = inside ? g : _two^b
    s = compute_edge_orientation(forest, k, e)
    xyz = zeros(T2, 3)
    xyz[𝐛[1] + _one] = s * g + (_one - (_two * s)) * oct.xyz[a₀]
    xyz[𝐛[2] + _one] = _edge_side(e′, 1) == 0 ? h⁻ : h⁺
    xyz[𝐛[3] + _one] = _edge_side(e′, 2) == 0 ? h⁻ : h⁺
    return OctantBWG(l, (xyz[1], xyz[2], xyz[3]))
end

transform_edge_remote(forest::ForestBWG, e::EdgeIndex, oct::OctantBWG, inside) = transform_edge_remote(forest, e[1], e[2], oct, inside)

"""
    transform_edge(forest, k, e, k′, e′, oct, inside::Bool)
    transform_edge(forest, k′, e′, oct, inside::Bool)
    transform_edge(forest, e′::EdgeIndex, oct, inside::Bool)

Algorithm 10 in [BWG2011](@citet) to transform an edge into a different octree coordinate system, but with reversed logic.
See `transform_edge_remote` with logic from paper.
Transform the octant `oct`, which sits at edge `e` of pivot tree `k` (in `k`'s coordinates),
into the coordinate system of the neighbouring tree `k′` at its edge `e′`. The along-edge
coordinate is taken from the pivot's edge axis and mirrored iff trees `k` and `k′` traverse the
shared macro edge in opposite directions.

Both the pivot pair `(k, e)` and the target pair `(k′, e′)` must be passed explicitly: a macro
edge can be shared by more than two trees, so neither the pivot nor the relative orientation can
be re-derived from `edge_edge_neighbor[..][1]` lookups (those pick an arbitrary incident tree).
The two shorter forms assume exactly two trees at the macro edge and take
`edge_edge_neighbor[k′, e′][1]` as the pivot.
"""
function transform_edge(forest::ForestBWG, k::T1, e::T1, k′::T1, e′::T1, oct::OctantBWG{3, N, T2}, inside::Bool) where {N, T1 <: Integer, T2 <: Integer}
    _four = T2(4)
    _one = T2(1)
    _two = T2(2)
    z = zero(T2)
    #see Algorithm 9, line 18: axes of the target edge e′ in tree k′
    𝐛 = (
        _edge_axis(e′),
        e′ - _one < 4 ? 1 : 0,
        e′ - _one < 8 ? 2 : 1,
    )
    a₀ = _edge_axis(e) + _one # 1-based; `oct` is in pivot coordinates, so the along-edge axis is the pivot edge's
    b = forest.cells[k′].b
    l = oct.l; g = _two^b - _two^(b - l)
    h⁻ = inside ? z : -_two^(b - l); h⁺ = inside ? g : _two^b
    s = compute_edge_orientation(forest, k, e, k′, e′)
    v1 = T2(s * g + (_one - (_two * s)) * oct.xyz[a₀])
    v2 = _edge_side(e′, 1) == 0 ? h⁻ : h⁺
    v3 = _edge_side(e′, 2) == 0 ? h⁻ : h⁺
    xyz = ntuple(p -> (p == 𝐛[1] + _one ? v1 : (p == 𝐛[2] + _one ? v2 : v3)), Val(3))
    return OctantBWG(l, xyz)
end

function transform_edge(forest::ForestBWG, k′::T1, e′::T1, oct::OctantBWG{3, <:Any, <:Integer}, inside::Bool) where {T1 <: Integer}
    k, e_ferrite = forest.topology.edge_edge_neighbor[k′, edge_perm[e′]][1]
    return transform_edge(forest, T1(k), T1(edge_perm_inv[e_ferrite]), k′, e′, oct, inside)
end

transform_edge(forest::ForestBWG, e::EdgeIndex, oct::OctantBWG, inside) = transform_edge(forest, e[1], e[2], oct, inside)

"""
    IteratePoint{dim}

A *point* in the sense of [IBWG2015](@citet) §2.1: an octant volume or one of its
lower-dimensional features (face, 3D edge, corner), encoded topologically by the box
`anchor` (minimum integer corner), `level` (box extent `h = _compute_size(b, level)`) and
`axes` (the directions the box extends along). `point_dim(c) = count(axes)` is the paper's
`dim(c)`; equality is field-wise — no physical coordinates, no rounding.
"""
struct IteratePoint{dim}
    anchor::NTuple{dim, Int}
    level::Int
    axes::NTuple{dim, Bool}
end

point_dim(c::IteratePoint) = count(c.axes)

"""
    _child_touches_point(ch::OctantBWG, c::IteratePoint, b) -> Bool

Whether point `c` (a feature of `ch`'s parent) lies in the closure of child octant `ch` —
the child-boundary-intersection set `B_∩^j` (eq 4.5 / Alg 5.2 line 14 of [IBWG2015](@citet)):
true iff `ch`'s box straddles `c`'s coordinate along every degenerate axis of `c`.
"""
function _child_touches_point(ch::OctantBWG{dim}, c::IteratePoint{dim}, b::Integer) where {dim}
    hc = _compute_size(b, ch.l)
    for d in 1:dim
        if !c.axes[d]
            p = c.anchor[d]
            (Int(ch.xyz[d]) <= p <= Int(ch.xyz[d]) + hc) || return false
        end
    end
    return true
end

"""
    _foreach_partc(f, c::IteratePoint, b)

Call `f(e, combo)` for each `e ∈ part(c)`, the *partition* of `c` (eq 2.7 of
[IBWG2015](@citet)): the `3^dim(c)` one-level-finer points strictly interior to `dom(c)` —
per extending axis the lower half, the degenerate mid plane, or the upper half (boundary
features belong to the parent point's partition, which is what makes the descent visit
every entity exactly once). `combo` is the base-3 slot encoding (one digit per extending
axis, ascending), the key into the precomputed `_part_mask` tables.
"""
function _foreach_partc(f::F, c::IteratePoint{dim}, b::Integer) where {F, dim}
    h = _compute_size(b, c.level); hh = h ÷ 2
    nd = point_dim(c)
    for combo in 0:(3^nd - 1)               # base-3 digit per extending axis
        anchor = c.anchor; axes = ntuple(_ -> false, dim); rem = combo
        for d in 1:dim
            c.axes[d] || continue
            s = rem % 3; rem ÷= 3
            if s == 0                        # lower half
                axes = Base.setindex(axes, true, d)
            elseif s == 1                    # mid (strictly interior, degenerate)
                anchor = Base.setindex(anchor, anchor[d] + hh, d)
            else                             # upper half
                anchor = Base.setindex(anchor, anchor[d] + hh, d)
                axes = Base.setindex(axes, true, d)
            end
        end
        f(IteratePoint{dim}(anchor, c.level + 1, axes), combo)
    end
    return
end

# The support of a part point `e ∈ part(c)` in terms of the children of `c`'s supports is a
# *combinatorial constant*: which children of a support octant touch `e` depends only on
# (a) the extension axes of `c` (`A`, a bitmask), (b) the support's side relative to `c`
# along the degenerate axes (`σ`, bit set = support anchored below `c`), and (c) the part
# slot `combo` — not on levels or coordinates. Per axis, a child's bit must be: `0`/`1` for
# a lower/upper-half slot, either for a mid slot (extending axes); opposite the support's
# side (degenerate axes: the children touching the pinned plane). These tables give the
# recursion of `_iterate_interior!` the support children of each part point with one
# mask lookup — the realization of the boundary-set intersections `B_∩` of IBWG2015 §4
# as compile-time data.
function _build_part_table(dim::Int)
    n = 2^dim
    # UInt8 (not Int8): entries are 2^dim-bit child masks, up to 0xff in 3D, which does not
    # fit Int8. Signedness carries no performance implication in Julia (integer overflow is
    # defined two's-complement wrap-around, unlike C/C++), and the use site widens the mask
    # to `Int` before working with it.
    tab = zeros(UInt8, n * n * 3^dim)
    for A in 0:(n - 1), σ in 0:(n - 1)
        (σ & A) == 0 || continue                     # σ ranges over degenerate axes only
        for combo in 0:(3^count_ones(A) - 1)
            mask = 0
            for j in 0:(n - 1)                       # candidate child bit pattern
                ok = true
                rem = combo
                for d in 0:(dim - 1)
                    bit = (j >> d) & 1
                    if (A >> d) & 1 == 1             # extending axis: slot digit decides
                        s = rem % 3; rem ÷= 3
                        ((s == 0 && bit == 1) || (s == 2 && bit == 0)) && (ok = false)
                    else
                        # degenerate axis: a support anchored below the pinned plane
                        # (σ bit set) touches it with its high children, and vice versa
                        bit == (σ >> d) & 1 || (ok = false)
                    end
                end
                ok && (mask |= 1 << j)
            end
            tab[(A * n + σ) * 3^dim + combo + 1] = mask
        end
    end
    return tab
end
const _PARTSUPP2 = _build_part_table(2)
const _PARTSUPP3 = _build_part_table(3)
@inline _part_mask(::Val{2}, A::Int, σ::Int, combo::Int) = @inbounds _PARTSUPP2[(A * 4 + σ) * 9 + combo + 1]
@inline _part_mask(::Val{3}, A::Int, σ::Int, combo::Int) = @inbounds _PARTSUPP3[(A * 8 + σ) * 27 + combo + 1]

# Bitmask of `c`'s extension axes / a support's side relative to `c` (bit `d-1` set = the
# support is anchored below `c.anchor` along degenerate axis `d`) — the table keys above.
@inline function _axes_mask(c::IteratePoint{dim}) where {dim}
    A = 0
    for d in 1:dim
        A |= Int(c.axes[d]) << (d - 1)
    end
    return A
end
@inline function _side_mask(c::IteratePoint{dim}, o::OctantBWG{dim}) where {dim}
    σ = 0
    for d in 1:dim
        σ |= Int(!c.axes[d] && Int(o.xyz[d]) != c.anchor[d]) << (d - 1)
    end
    return σ
end

"""
    _foreach_root_closure(f, ::Val{dim}, b)

Call `f(c)` for each point in the *closure* of the tree root (Alg 5.3 line 4 of
[IBWG2015](@citet), single tree): the root volume and all its boundary faces/edges/corners
— the `3^dim` seeds of the recursive descent. The root's boundary features have no parent
split to produce them (unlike interior features, which arise as `part` of an ancestor), so
they get their own descent seeds here. Along each axis the feature is pinned to the low
face (coord 0), spans the full root, or is pinned to the high face.
"""
function _foreach_root_closure(f::F, ::Val{dim}, b::Integer) where {F, dim}
    h = _compute_size(b, 0)
    for combo in 0:(3^dim - 1)
        anchor = ntuple(_ -> 0, dim); axes = ntuple(_ -> false, dim); rem = combo
        for d in 1:dim
            s = rem % 3; rem ÷= 3
            if s == 1
                axes = Base.setindex(axes, true, d)      # spans the full root along d
            elseif s == 2
                anchor = Base.setindex(anchor, h, d)     # high face along d
            end
        end
        f(IteratePoint{dim}(anchor, 0, axes))
    end
    return
end

# The child of `o` at z-order corner slot `ci` (anchor offset by the child size along the
# slot's high axes). A top-level helper so callers' loop variables are not captured by the
# `ntuple` closure (a reassigned capture boxes).
@inline function _corner_child(o::OctantBWG{dim, N, T}, ci::Int, b::Integer) where {dim, N, T}
    hh = T(_compute_size(b, o.l + one(T)))
    return OctantBWG{dim, N, T}(o.l + one(T), ntuple(d -> o.xyz[d] + (((ci - 1) >> (d - 1)) & 1) * hh, dim))
end

"""
    _descend_to_corner(c::IteratePoint, s::OctantBWG, lo, hi, leaves, b) -> (leaf, index)

Find the leaf under support octant `s` (`leaves[lo:hi]`) whose closure contains the
0-point `c` — the `atom supp(c)` search (Alg 5.2 line 18 / Prop 2.8 of [IBWG2015](@citet))
— returning it with its index into `leaves` (the paper's element index `j`, §6.4). `c` is
a corner of `s`, so the descent follows the fixed `ci`-most path, narrowed with
[`split_bounds`](@ref); the Morton-first/-last slots resolve in O(1).
"""
function _descend_to_corner(c::IteratePoint{dim}, s::OctantBWG{dim, N, T}, lo::Int, hi::Int, leaves, b::Integer) where {dim, N, T}
    ci = _corner_slot(s, c.anchor)
    ci == 1 && return (leaves[lo], lo)
    ci == N && return (leaves[hi], hi)
    o = s
    while !(lo == hi && leaves[lo] == o)
        k = split_bounds(leaves, lo, hi, o, b)
        o = _corner_child(o, ci, b)
        lo = k[ci]; hi = k[ci + 1] - 1
    end
    return (o, lo)
end

"""
    IterScratch{N, M, OT}
    IterScratch(tree::OctreeBWG)

Preallocated per-depth working memory of the recursive descent — the `sc` argument of
[`iterate_points`](@ref) — so the traversal allocates nothing ([IBWG2015](@citet) §5.4).
In a DFS only one root-to-node path is live, so buffers are indexed by recursion depth and
reused across siblings; see the field comments for what each buffer holds. `M = N + 1` is
the `split_bounds` tuple length. Sizes depend only on the maximum level `b`, so one
scratch is shared across all trees of a forest (`creategrid` does exactly that).
"""
struct IterScratch{N, M, OT}
    supp::Vector{Vector{OT}}                  # [depth] -> support octants of the point at this depth
    S::Vector{Vector{NTuple{2, Int}}}         # [depth] -> leaf index ranges, one per support octant
    prov::Vector{Vector{Int}}                 # [depth] -> parent-frame slot (i-1)*N + j of each support
    child_octants::Vector{Vector{NTuple{N, OT}}}     # [depth] -> children of each support octant (Split_array)
    splits::Vector{Vector{NTuple{M, Int}}}    # [depth] -> split_bounds of each support octant
    msplit::Vector{Vector{NTuple{M, Int}}}    # [depth] -> per-frame split memo, slot-indexed
    mepoch::Vector{Vector{Int}}               # [depth] -> per-slot epoch stamp: valid iff == epoch[depth]
    epoch::Vector{Int}                        # [depth] -> current frame epoch (O(1) memo invalidation)
    L::Vector{OT}                             # reused leaf_supp buffer passed to the callback
    Lidx::Vector{Int}                         # reused leaf-index buffer parallel to L
end

function IterScratch(tree::OctreeBWG{dim, N, T}) where {dim, N, T}
    OT = OctantBWG{dim, N, T}
    nd = Int(tree.b) + 2                       # max recursion depth is the octree level + 1
    return IterScratch{N, N + 1, OT}(
        [OT[] for _ in 1:nd], [NTuple{2, Int}[] for _ in 1:nd], [Int[] for _ in 1:nd],
        [NTuple{N, OT}[] for _ in 1:nd], [NTuple{N + 1, Int}[] for _ in 1:nd],
        [Vector{NTuple{N + 1, Int}}(undef, N * N) for _ in 1:nd], [zeros(Int, N * N) for _ in 1:nd],
        zeros(Int, nd), OT[], Int[]
    )
end

"""
    LeafSupport{OT}

The local leaf support set `leaf_supp(c)` handed to the [`iterate_points`](@ref) callback:
the leaves whose closure touches the visited point `c`, each paired with its index into the
tree's Morton-sorted `leaves` (the element index `j` of [IBWG2015](@citet) §6.4). Iterating
yields the octants; `ls.idxs[i]` belongs to `ls.octs[i]`; an index of `0` marks a non-leaf
entry — impossible on a 2:1-balanced forest, and turned into an error by `creategrid`. Both
vectors are reused buffers of the traversal — **copy them if you retain them past the
callback.**
"""
struct LeafSupport{OT}
    octs::Vector{OT}
    idxs::Vector{Int}
end
Base.iterate(ls::LeafSupport, state...) = iterate(ls.octs, state...)
Base.length(ls::LeafSupport) = length(ls.octs)
Base.getindex(ls::LeafSupport, i::Int) = ls.octs[i]
Base.eltype(::Type{LeafSupport{OT}}) where {OT} = OT

"""
    _iterate_interior!(visit, c::IteratePoint, depth, sc::IterScratch, leaves, b, mindim, maxdim, skipconf)

[IBWG2015](@citet) Algorithm 5.2 (`Iterate_interior`), serial and allocation-free. On
entry `sc.supp[depth]` holds `c`'s support octants (eq 2.11) and `sc.S[depth][i]` the
`leaves` index range under each. Exactly as in Alg 5.2, the recursion stops when some
support octant is itself a leaf (line 7; for a corner point always, lines 16-18) — so
hanging points are never visited (`PΩ`, eq 5.1) — and calls `visit(c, leaf_supp)` with the
[`LeafSupport`](@ref) built via `_child_touches_point` (line 14) / `_descend_to_corner`
(line 18); otherwise it descends `part(c)`, slicing each range with
[`split_bounds`](@ref). `leaf_supp` wraps reused buffers — **copy if retained.**
`mindim`/`maxdim` specialize the callback (§5.4): dims below `mindim` are not recursed
into, and `visit` fires only for `point_dim(c) ∈ mindim:maxdim`. `skipconf` additionally
suppresses the visit at conforming (all-supports-leaf) interfaces, see the comment inline.
"""
function _iterate_interior!(visit::F, c::IteratePoint{dim}, depth::Int, sc::IterScratch{N, M, OT}, leaves, b::Integer, mindim::Int, maxdim::Int, skipconf::Bool) where {F, dim, N, M, OT}
    supp = sc.supp[depth]; S = sc.S[depth]
    m = length(supp)
    m == 0 && return
    anylocal = false
    stop = false
    refined = false
    for i in 1:m
        lo, hi = S[i]
        if lo <= hi
            anylocal = true
            (lo == hi && leaves[lo] == supp[i]) ? (stop = true) : (refined = true)
        end
    end
    anylocal || return                                 # Alg 5.2 line 1 (serial: empty support)
    dimc = point_dim(c)

    if dimc == 0                                       # 0-point: always stop (lines 15-18)
        if dimc >= mindim
            L = sc.L; Lidx = sc.Lidx
            empty!(L); empty!(Lidx)
            for i in 1:m
                # Disjoint support subtrees -> one distinct leaf each, no dedup needed.
                o, oi = _descend_to_corner(c, supp[i], S[i][1], S[i][2], leaves, b)
                push!(L, o); push!(Lidx, oi)
            end
            visit(c, LeafSupport(L, Lidx))
        end
        return
    end

    if stop                                            # finalize: build leaf_supp (lines 5-14)
        # `skipconf`: a stop where every support is a leaf (`!refined`) is a conforming
        # interface; a callback that only acts on non-conforming ones (hanging detection)
        # opts out of the leaf_supp build + visit there. Corner points are unaffected
        # (handled above) — they must always fire.
        if mindim <= dimc <= maxdim && (refined || !skipconf)
            L = sc.L; Lidx = sc.Lidx
            empty!(L); empty!(Lidx)
            # Supports are pairwise disjoint octants, so leaves/children collected from
            # different supports cannot collide -> no dedup needed.
            for i in 1:m
                if S[i][1] == S[i][2] && leaves[S[i][1]] == supp[i]
                    push!(L, supp[i]); push!(Lidx, S[i][1])
                else
                    # Refined neighbour: its children adjacent to c (line 14, `B_∩^j` via
                    # `_child_touches_point`). Under the 2:1 balance these children are
                    # themselves leaves, sitting at the start of their `split_bounds`
                    # sub-range; a non-leaf child (unbalanced forest) gets index 0.
                    kb = split_bounds(leaves, S[i][1], S[i][2], supp[i], b)
                    ch = children(supp[i], b)
                    for j in 1:N
                        if _child_touches_point(ch[j], c, b)
                            push!(L, ch[j])
                            push!(Lidx, (kb[j] == kb[j + 1] - 1 && leaves[kb[j]] == ch[j]) ? kb[j] : 0)
                        end
                    end
                end
            end
            visit(c, LeafSupport(L, Lidx))
        end
        return
    end

    # No support octant is a leaf -> recurse over part(c) (lines 21-25). Every support
    # octant is internal; cache its children + leaf sub-ranges (H_i) in the depth buffers.
    # A support octant is shared by several sibling points of this frame's parent (it is in
    # the support of every part point of its closure), so its split is served by the
    # parent-frame memo (`msplit`, keyed by the provenance slot the parent recorded) and
    # computed only once per frame instead of once per sibling point.
    child_octants = sc.child_octants[depth]; splits = sc.splits[depth]
    prov = sc.prov[depth]; msplit = sc.msplit[depth]; mepoch = sc.mepoch[depth]
    ep = sc.epoch[depth]
    empty!(child_octants); empty!(splits)
    for i in 1:m
        push!(child_octants, children(supp[i], b))
        slot = prov[i]
        if slot != 0 && mepoch[slot] == ep
            push!(splits, msplit[slot])
        else
            ksp = split_bounds(leaves, S[i][1], S[i][2], supp[i], b)
            if slot != 0
                msplit[slot] = ksp
                mepoch[slot] = ep
            end
            push!(splits, ksp)
        end
    end
    # Which children of support i touch a part point is a combinatorial constant (`B_∩`,
    # eq 4.5) keyed on (extension axes of c, support side, part slot) — precomputed in the
    # `_part_mask` tables, replacing a per-child geometric test. Supports are pairwise
    # disjoint, so their children never collide -> no membership test.
    A = _axes_mask(c)
    σs = ntuple(i -> i <= m ? _side_mask(c, supp[i]) : 0, Val(N))
    esupp = sc.supp[depth + 1]; eS = sc.S[depth + 1]   # the next depth's (reused) support buffers
    eprov = sc.prov[depth + 1]
    sc.epoch[depth + 1] += 1                           # fresh memo for the sub-frame, O(1)
    _foreach_partc(c, b) do e, combo
        point_dim(e) >= mindim || return
        empty!(esupp); empty!(eS); empty!(eprov)
        for i in 1:m
            ch = child_octants[i]
            ki = splits[i]
            mask = Int(_part_mask(Val(dim), A, σs[i], combo))
            while mask != 0
                j = trailing_zeros(mask) + 1
                mask &= mask - 1
                push!(esupp, ch[j]); push!(eS, (ki[j], ki[j + 1] - 1)); push!(eprov, (i - 1) * N + j)
            end
        end
        _iterate_interior!(visit, e, depth + 1, sc, leaves, b, mindim, maxdim, skipconf)
        return
    end
    return
end

"""
    iterate_points(visit, tree::OctreeBWG, sc::IterScratch; mindim = 0, maxdim = dim, skip_conforming = false)

[IBWG2015](@citet) Algorithm 5.3 (`Iterate`), serial: drive `_iterate_interior!` from the
closure of each tree root. `visit(c::IteratePoint, leaf_supp::LeafSupport)` is called once
for every point `c ∈ PΩ` (5.1) — every non-hanging volume / face / edge / corner — with
`leaf_supp` the leaves surrounding it plus their leaf indices (§6.4: "`Iterate` provides
the index"). Use `point_dim(c)` to dispatch per dimension (volume `= dim`, face `= dim-1`,
edge `= 1`, corner `= 0`), or pass `mindim`/`maxdim` for the §5.4 specialization
(e.g. `mindim = dim - 1` to visit only volumes + faces, or `maxdim = dim - 1` to skip the
volume callback like a `NULL` volume callback in `p4est_iterate`). With
`skip_conforming = true`, face/edge points whose supports are all leaves of equal level
(conforming interfaces) are skipped as well — the specialization for callbacks that only
act on non-conforming interfaces, like the hanging-node detection; corner points always
fire. The descent is allocation-free: callers traversing many trees (`creategrid`,
`facetskeleton`) allocate one `IterScratch` and pass it to every tree of equal maximum
level `b` (the buffers are depth-indexed, so they only depend on `b`);
**`leaf_supp` wraps reused buffers — copy what you retain.**

Looping the trees of a forest is the serial Alg 5.3. Within a tree this visits `PΩ`
exactly; at *shared tree boundaries* a feature is currently visited once per incident
tree (its per-tree `leaf_supp` covers only that tree's leaves) — `creategrid` reconciles
the per-tree visits through its boundary tables. Cross-tree coordinated descent
(single-visit boundary `leaf_supp` via the orientation transforms, fully mirroring
`p4est_iterate`) is the documented next step, consistent with
[`_iterate_interface_hanging!`](@ref)'s inter-tree face descent.
"""
function iterate_points(visit::F, tree::OctreeBWG{dim}, sc::IterScratch; mindim::Int = 0, maxdim::Int = dim, skip_conforming::Bool = false) where {F, dim}
    leaves = tree.leaves
    isempty(leaves) && return
    b = tree.b
    r = root(dim)                            # root octant (zero octant; eltype matches leaves)
    full = (1, length(leaves))
    _foreach_root_closure(Val(dim), b) do c
        # A root feature of dim < mindim leads only to lower-dim features -> skip; interior
        # features descend from the root volume, boundary ones from their own seed here.
        point_dim(c) >= mindim || return
        empty!(sc.supp[1]); push!(sc.supp[1], r)         # seed depth 1 with the single root support
        empty!(sc.S[1]); push!(sc.S[1], full)
        empty!(sc.prov[1]); push!(sc.prov[1], 0)         # no parent frame -> no memo slot
        _iterate_interior!(visit, c, 1, sc, leaves, b, mindim, maxdim, skip_conforming)
        return
    end
    return
end

# An `(element, z-order corner slot)` reference into the element-node matrix `E`. Hanging
# constraints are recorded as E-references and resolved after the numbering traversal.
const ERef = Tuple{Int, Int}

"""
    _iter_interface!(
        cons2, cons4, E, offR, forest, kL, lvsL, octL, loL, hiL, fL,
        kR, lvsR, octR, loR, hiR, fR, bL, bR
    )

Synchronized two-sided descent of a shared *tree* face, emitting inter-tree hanging-node
constraints. `octL ∈ tree kL` (leaves `lvsL[loL:hiL]`, native frame) and `octR ∈ tree kR` are
images of each other across the shared face — `fL`/`fR` are the local face indices toward it —
and descend in lock-step at equal levels.

- both sides leaves of equal size → conforming, nothing emitted;
- one side a leaf, the other refined → the leaf is the **coarse** side and the hanging nodes lie
  on the refined side's face in the *fine* tree's frame (genuine fine-leaf vertices), emitted via
  [`_emit_interface_face!`](@ref) as `(element, slot)` references into `E` (`offR` is tree `kR`'s
  element offset). Only the `kL`-coarse case emits here; the `kR`-coarse case is emitted when the
  descent is run from `(kR, fR)`, so each interface is handled once per direction;
- both refined → recurse, matching child `i` on the `kL` side to its image on the `kR` side via
  [`transform_facet`](@ref) (the validated cross-tree orientation pattern — no new logic).

The integer/topological analogue, across trees, of the intra-tree face/edge callbacks of the
numbering traversal (see [`creategrid`](@ref)).
"""
function _iter_interface!(
        cons2, cons4, E::Matrix{Int}, offR::Int, forest::ForestBWG, kL::Int, lvsL, octL::OctantBWG{dim, N}, loL::Int, hiL::Int, fL::Int,
        kR::Int, lvsR, octR::OctantBWG{dim, N}, loR::Int, hiR::Int, fR::Int, bL::Integer, bR::Integer
    ) where {dim, N}
    lL = _isleaf(lvsL, loL, hiL, octL)
    lR = _isleaf(lvsR, loR, hiR, octR)
    lL && lR && return                                           # same-size leaves both sides -> conforming
    if lL && !lR                                                 # kL coarse, kR refined -> hanging (fine = kR)
        _emit_interface_face!(cons2, cons4, E, offR, lvsR, loR, hiR, octR, fR, bR)
        return
    elseif !lL && lR                                             # kL refined, kR coarse -> caught from (kR,fR)
        return
    end
    kb = split_bounds(lvsL, loL, hiL, octL, bL); cL = children(octL, bL)
    kbR = split_bounds(lvsR, loR, hiR, octR, bR); cR = children(octR, bR)
    for i in 1:N
        contains_facet(face(octL, fL, bL), face(cL[i], fL, bL)) || continue   # child i on the shared face
        nbR = transform_facet(forest, kR, fR, facet_neighbor(cL[i], fL, bL))  # its image in kR
        for j in 1:N
            cR[j] == nbR || continue
            _iter_interface!(
                cons2, cons4, E, offR, forest, kL, lvsL, cL[i], kb[i], kb[i + 1] - 1, fL,
                kR, lvsR, cR[j], kbR[j], kbR[j + 1] - 1, fR, bL, bR
            )
            break
        end
    end
    return
end

# `(leaf index, slot)` reference of the vertex at coord `x` within child `j`'s subtree
# (`kb`/`ch` from the parent's `split_bounds`/`children`): the child itself when it is a leaf
# — the only case on a 2:1-balanced forest — else the descent to the child's `x`-corner leaf.
# The descent covers interfaces with a >1 level jump, which the materializer tolerates
# (matching its legacy behaviour) by constraining only the top-level face-interior points.
function _subtree_corner_ref(lvs, kb::NTuple, ch::NTuple, j::Int, x::NTuple{dim, Int}, b) where {dim}
    lo = kb[j]; hi = kb[j + 1] - 1
    o = ch[j]
    if lo == hi && lvs[lo] == o
        return (lo, _corner_slot(o, x))
    end
    leaf, li = _descend_to_corner(IteratePoint{dim}(x, Int(o.l), ntuple(_ -> false, dim)), o, lo, hi, lvs, b)
    return (li, _corner_slot(leaf, x))
end

"""
    _emit_interface_face!(cons2, cons4, E, off, lvs, lo, hi, octR, fR, b)

The `kL` side of an interface octant pair is a leaf while the `kR` side `octR` (leaves
`lvs[lo:hi]`, element offset `off`) is refined: the interior points of the shared face hang.
Every one of them is a vertex of `octR`'s face children — leaves under the 2:1 balance, already
numbered by tree `kR`'s own traversal — so the constrained ids are read straight off `E` and the
constraints recorded exactly like the intra-tree emitters: the face midpoint (2D) / face center
(3D) constrained by the face corners, and in 3D each face-edge midpoint constrained by its edge's
endpoints, masters in ascending coordinate order. (On a >1 level jump the ids are read from the
corner leaves via `_subtree_corner_ref`'s descent.)
"""
function _emit_interface_face!(cons2, cons4, E::Matrix{Int}, off::Int, lvs, lo::Int, hi::Int, octR::OctantBWG{dim, N}, fR::Int, b) where {dim, N}
    kb = split_bounds(lvs, lo, hi, octR, b)
    ch = children(octR, b)
    if dim == 2
        j1 = 𝒱₂[fR, 1]; j2 = 𝒱₂[fR, 2]
        c1 = map(Int, vertex(octR, j1, b)); c2 = map(Int, vertex(octR, j2, b))
        m = (c1 .+ c2) .÷ 2
        lm, sm = _subtree_corner_ref(lvs, kb, ch, j1, m, b)
        r1 = _subtree_corner_ref(lvs, kb, ch, j1, c1, b)
        r2 = _subtree_corner_ref(lvs, kb, ch, j2, c2, b)
        push!(cons2, (E[sm, off + lm], (off + r1[1], r1[2]), (off + r2[1], r2[2])))
    else
        j = (𝒱₃[fR, 1], 𝒱₃[fR, 2], 𝒱₃[fR, 3], 𝒱₃[fR, 4])
        cc = ntuple(i -> map(Int, vertex(octR, j[i], b)), Val(4))    # face corners, z-order
        r = ntuple(Val(4)) do i
            li, sl = _subtree_corner_ref(lvs, kb, ch, j[i], cc[i], b)
            (off + li, sl)
        end
        # face center, masters = the 4 corners in lexicographic coordinate order (c1, c3, c2, c4)
        m = (cc[1] .+ cc[4]) .÷ 2
        lm, sm = _subtree_corner_ref(lvs, kb, ch, j[1], m, b)
        push!(cons4, (E[sm, off + lm], r[1], r[3], r[2], r[4]))
        # the 4 face-edge midpoints, each constrained by its (ascending) endpoint pair
        for (i1, i2) in ((1, 2), (3, 4), (1, 3), (2, 4))
            me = (cc[i1] .+ cc[i2]) .÷ 2
            le, se = _subtree_corner_ref(lvs, kb, ch, j[i1], me, b)
            push!(cons2, (E[se, off + le], r[i1], r[i2]))
        end
    end
    return
end

"""
    _iterate_interface_hanging!(cons2, cons4, E, offsets, forest::ForestBWG)

Collect all *inter-tree* hanging-node constraints of `forest` into `cons2`/`cons4`. For each
shared tree face (found via the facet–facet neighbourhood), seed [`_iter_interface!`](@ref) with
the two tree roots and let it descend both sides in lock-step. The forest-level counterpart of
the intra-tree face/edge callbacks; together they capture every hanging node of the materialized
grid. Single-tree forests have no shared faces, so this is a no-op there.
"""
function _iterate_interface_hanging!(cons2, cons4, E::Matrix{Int}, offsets::Vector{Int}, forest::ForestBWG{dim}) where {dim}
    perm = dim == 2 ? 𝒱₂_perm : 𝒱₃_perm
    perminv = dim == 2 ? 𝒱₂_perm_inv : 𝒱₃_perm_inv
    fn = Ferrite.get_facet_facet_neighborhood(forest)
    r = root(dim)
    for (k, tree) in enumerate(forest.cells)
        bL = tree.b
        for f in 1:(2 * dim)
            nb = fn[k, perm[f]]
            isempty(nb) && continue
            k′ = nb[1][1]; f′ = perminv[nb[1][2]]
            treeR = forest.cells[k′]
            _iter_interface!(
                cons2, cons4, E, offsets[k′], forest, k, tree.leaves, r, 1, length(tree.leaves), f,
                k′, treeR.leaves, r, 1, length(treeR.leaves), f′, bL, treeR.b
            )
        end
    end
    return
end

@noinline _unbalanced_error() = throw(ArgumentError("creategrid requires a 2:1-balanced forest (an element vertex has no node id) — call balanceforest! first"))

# z-order corner slot of integer coord `xyz` in octant `o` — the inverse of [`vertex`](@ref).
# Along axis `d`, `xyz` sits either at the anchor (low, bit `d-1` clear) or at anchor + size
# (high, bit set); only ever called when `xyz` is a corner of `o`.
@inline function _corner_slot(o::OctantBWG{dim}, xyz::NTuple{dim, <:Integer}) where {dim}
    s = 1
    for d in 1:dim
        xyz[d] == o.xyz[d] || (s += 1 << (d - 1))
    end
    return s
end

"""
    LnodesVisitor{dim, N, T, TI}

Per-tree numbering context of the [`creategrid`](@ref) traversal — the `Lnodes_callback` of
[IBWG2015](@citet) Alg 6.2, as a callable struct (a concrete top-level type instead of a
closure, so the captures don't box). One instance per tree: `E`, the provisional node data
(`nodecoords_prov`, `cnt`) and the constraint records (`cons2`/`cons4`) are shared across
trees; `bnd`, the geometry and the element offset are the tree's own.

Callback dispatch by `point_dim(c)`: corners create + scatter node ids
([`_visit_corner!`](@ref)); non-conforming faces and (3D) edges create the hanging vertex and
record its constraint ([`_visit_face!`](@ref), [`_visit_edge3d!`](@ref)); volumes need no work
(cell connectivity IS the filled `E`).
"""
struct LnodesVisitor{dim, N, T <: Real, TI <: Integer}
    E::Matrix{Int}                                     # element-node matrix, NV × ncells, z-order slots
    nodecoords_prov::Vector{Vec{dim, T}}               # physical coordinate per provisional id
    bnd::Vector{Tuple{UInt64, Int}}                    # this tree's boundary node table (sorted later)
    cons2::Vector{Tuple{Int, ERef, ERef}}              # hanging midpoints: (id, 2 master refs)
    cons4::Vector{Tuple{Int, ERef, ERef, ERef, ERef}}  # 3D hanging face centers: (id, 4 master refs)
    treecorners::NTuple{N, Vec{dim, T}}                # tree geometry for _interp_treepoint
    b::TI                                              # tree's max refinement level
    hilim::Int                                         # 2^b, the root extent
    off::Int                                           # tree's global element offset
    cnt::Base.RefValue{Int}                            # forest-wide provisional id counter
end

# Create the node at integer coord `xyz` of the visitor's tree: draw the next provisional id,
# interpolate its physical coordinate once, and — for coords on the root boundary — record a
# boundary-table entry for the cross-tree merge.
@inline function _create_node!(v::LnodesVisitor{dim}, xyz::NTuple{dim, Int}) where {dim}
    id = (v.cnt[] += 1)
    push!(v.nodecoords_prov, _interp_treepoint(v.treecorners, v.b, xyz))
    onb = false
    for d in 1:dim
        if xyz[d] == 0 || xyz[d] == v.hilim
            onb = true
            break
        end
    end
    onb && push!(v.bnd, (_packcoord(xyz), id))
    return id
end

function (v::LnodesVisitor{dim})(c::IteratePoint{dim}, ls::LeafSupport) where {dim}
    d = point_dim(c)
    if d == 0
        _visit_corner!(v, c, ls)
    elseif d == dim - 1
        _visit_face!(v, c, ls)
    elseif dim == 3 && d == 1
        _visit_edge3d!(v, c, ls)
    end
    return
end

"""
    _visit_corner!(v::LnodesVisitor, c, ls)

Corner callback (`dim(c) == 0`): `c` is a non-hanging mesh vertex — hanging points are never
visited by the iterator — so every supporting leaf has `c` as a corner. Create its node and
scatter the id into each leaf's `E` slot (the leaf indices come with the support set).
"""
function _visit_corner!(v::LnodesVisitor{dim}, c::IteratePoint{dim}, ls::LeafSupport) where {dim}
    xyz = c.anchor
    id = _create_node!(v, xyz)
    E = v.E
    @inbounds for i in 1:length(ls.octs)
        E[_corner_slot(ls.octs[i], xyz), v.off + ls.idxs[i]] = id
    end
    return
end

"""
    _mixed_support(c::IteratePoint, ls::LeafSupport) -> (coarse, fine)

Non-conformity test of a finalized face/edge point ([IBWG2015](@citet) Fig 5): a leaf
support at `level(c)` (the coarse side) coexisting with finer children. Returns the index
of the first coarse leaf in `ls` (its corners are the constraint masters) and whether finer
leaves exist — the interface hangs iff finer leaves coexist with the coarse one. A
finalized face/edge always has at least one support that is itself a leaf at `level(c)`
(that is what stopped the recursion in `_iterate_interior!`), so with all supports
equal-level leaves the interface is conforming and there is nothing to do.
"""
@inline function _mixed_support(c::IteratePoint, ls::LeafSupport)
    lvl = c.level
    coarse = 0
    fine = false
    @inbounds for i in 1:length(ls.octs)
        l = Int(ls.octs[i].l)
        if l == lvl
            coarse == 0 && (coarse = i)
        elseif l > lvl
            fine = true
        end
    end
    return coarse, fine
end

# Scatter the id of the hanging vertex at coord `m` into every finer support leaf — exactly
# the leaves that have `m` as a corner (for the coarse side `m` is interior to the feature,
# not a vertex). A leaf index of 0 means a `B_∩` child was not a leaf: unbalanced forest.
function _scatter_hanging!(v::LnodesVisitor{dim}, ls::LeafSupport, lvl::Int, m::NTuple{dim, Int}, id::Int) where {dim}
    E = v.E
    for i in 1:length(ls.octs)
        o = ls.octs[i]
        Int(o.l) > lvl || continue
        ls.idxs[i] == 0 && _unbalanced_error()
        E[_corner_slot(o, m), v.off + ls.idxs[i]] = id
    end
    return
end

# Shared 1-dimensional hanging emitter (a 2D non-conforming face or a 3D non-conforming
# edge): the midpoint of the coarse feature hangs, constrained by the feature's two endpoints
# — corners of the coarse leaf `coarse` (element `cgid`) — in ascending coordinate order.
function _emit_hanging_mid!(v::LnodesVisitor{dim}, c::IteratePoint{dim}, ls::LeafSupport, coarse::OctantBWG{dim}, cgid::Int, h::Int) where {dim}
    ax = 1
    for d in 1:dim
        if c.axes[d]
            ax = d
            break
        end
    end
    a = c.anchor
    m = Base.setindex(a, a[ax] + (h >> 1), ax)
    id = _create_node!(v, m)
    _scatter_hanging!(v, ls, c.level, m, id)
    a2 = Base.setindex(a, a[ax] + h, ax)
    push!(v.cons2, (id, (cgid, _corner_slot(coarse, a)), (cgid, _corner_slot(coarse, a2))))
    return
end

"""
    _visit_face!(v::LnodesVisitor, c, ls)

Face callback (`dim(c) == dim-1`). Conforming faces need no work. On a non-conforming face —
a coarse leaf on one side, its refined neighbour's children on the other — the face's interior
vertices hang: in 2D the face midpoint (constrained by the 2 endpoints), in 3D the face
*center* (constrained by the 4 face corners). The 3D face's edge midpoints are **not** emitted
here: each hanging edge is its own iterator point, visited exactly once even when shared by
several non-conforming faces, and handled by [`_visit_edge3d!`](@ref).
"""
function _visit_face!(v::LnodesVisitor{dim}, c::IteratePoint{dim}, ls::LeafSupport) where {dim}
    ci, fine = _mixed_support(c, ls)
    (ci != 0 && fine) || return
    h = Int(_compute_size(v.b, c.level))
    coarse = ls.octs[ci]
    cgid = v.off + ls.idxs[ci]
    if dim == 2
        _emit_hanging_mid!(v, c, ls, coarse, cgid, h)
    else
        hh = h >> 1
        e1 = 0; e2 = 0                       # the two extending axes, e1 < e2
        for d in 1:3
            if c.axes[d]
                e1 == 0 ? (e1 = d) : (e2 = d)
            end
        end
        a = c.anchor
        m = Base.setindex(Base.setindex(a, a[e1] + hh, e1), a[e2] + hh, e2)
        id = _create_node!(v, m)
        _scatter_hanging!(v, ls, c.level, m, id)
        # face corners in z-order: a, c2 = a + h·e1, c3 = a + h·e2, c4 = a + h·(e1+e2);
        # lexicographic coordinate order is (a, c3, c2, c4) — the +h on the lower axis
        # sorts *later* — which is the master order of the constraint.
        c2 = Base.setindex(a, a[e1] + h, e1)
        c3 = Base.setindex(a, a[e2] + h, e2)
        c4 = Base.setindex(c2, c2[e2] + h, e2)
        push!(
            v.cons4, (
                id,
                (cgid, _corner_slot(coarse, a)), (cgid, _corner_slot(coarse, c3)),
                (cgid, _corner_slot(coarse, c2)), (cgid, _corner_slot(coarse, c4)),
            )
        )
    end
    return
end

"""
    _visit_edge3d!(v::LnodesVisitor, c, ls)

Edge callback (3D, `dim(c) == 1`): the midpoint of a non-conforming coarse edge hangs,
constrained by the edge's endpoints. On a 2:1-balanced forest every hanging vertex is either
such an edge midpoint or a face center ([`_visit_face!`](@ref)) — level jumps are capped at
one, so no ¼-points exist — and each is created by exactly one callback: the edge midpoint
belongs to exactly one coarse edge (two distinct octant edges cannot share their midpoints),
even when that edge borders several non-conforming faces.
"""
function _visit_edge3d!(v::LnodesVisitor{3}, c::IteratePoint{3}, ls::LeafSupport)
    ci, fine = _mixed_support(c, ls)
    (ci != 0 && fine) || return
    _emit_hanging_mid!(v, c, ls, ls.octs[ci], v.off + ls.idxs[ci], Int(_compute_size(v.b, c.level)))
    return
end

"""
    _global_numbering(E, alias, nodecoords_prov) -> (final_of_prov, nodecoords)

Serial `Global_numbering` ([IBWG2015](@citet) Alg 6.1): sweep the element-node matrix in
(element, element-node) lexicographic order — the linear (column-major) order of `E` — and
assign each canonical node its final dense id at first encounter. With the ownership rule
`owner(c) = min leaf supp(c)` (eq 6.2), the first element referencing a node *is* its owner, so
this reproduces the paper's partition-independent numbering without any per-node search. Also
gathers the final node coordinates (the canonical provisional node's) in final-id order.

A zero entry in `E` means some element vertex was never assigned a node — impossible on a
2:1-balanced forest — and raises an error.
"""
function _global_numbering(E::Matrix{Int}, alias::Vector{Int}, nodecoords_prov::Vector{Vec{dim, T}}) where {dim, T}
    nprov = length(alias)
    canon_to_dense = zeros(Int, nprov)
    nodecoords = Vec{dim, T}[]
    sizehint!(nodecoords, nprov)
    ndense = 0
    @inbounds for j in eachindex(E)
        p = E[j]
        p == 0 && _unbalanced_error()
        cid = alias[p]
        if canon_to_dense[cid] == 0
            ndense += 1
            canon_to_dense[cid] = ndense
            push!(nodecoords, nodecoords_prov[cid])
        end
    end
    final_of_prov = Vector{Int}(undef, nprov)
    @inbounds for p in 1:nprov
        final_of_prov[p] = canon_to_dense[alias[p]]
    end
    return final_of_prov, nodecoords
end

"""
    _element_offsets(forest::ForestBWG) -> Vector{Int}

Element offset of each tree into the materialized cell vector: tree `k`'s leaf `j` (Morton
order) is cell `offsets[k] + j` of the grid returned by [`creategrid`](@ref).
"""
function _element_offsets(forest::ForestBWG)
    ntrees = length(forest.cells)
    offsets = Vector{Int}(undef, ntrees)
    acc = 0
    for k in 1:ntrees
        offsets[k] = acc
        acc += length(forest.cells[k].leaves)
    end
    return offsets
end

"""
    creategrid(forest::ForestBWG) -> NonConformingGrid

Materialize a `ForestBWG` (a forest of adaptively refined octrees) into a `NonConformingGrid`.
The returned grid can be used like any Ferrite grid, complete with the hanging-node constraints
(`conformity_info`) and the transferred boundary and subdomain sets (`facetsets`, `cellsets`).

!!! warning "Only `facetsets` and `cellsets` are transferred"
    The `vertexsets` and `nodesets` of the base grid are **not** carried onto the materialized
    grid — they are kept on the `ForestBWG` but the returned `NonConformingGrid` has both
    empty. Re-attach them on the refined grid if you need them (e.g. with `addvertexset!`).

This is the Lnodes construction of [IBWG2015](@citet) §6 on the point iterator
[`iterate_points`](@ref) (Alg 5.2/5.3): node ids are assigned inside the iterator callbacks and
scattered into the element-node matrix `E` — there is **no global node map**; identity is carried
by `E` plus `O(surface)` per-tree boundary tables, the layout that generalizes to distributed
forests (each process numbers the nodes it owns; only interface ids are reconciled). The whole
construction is **integer/topological** — node identity is decided on integer octree coordinates,
never physical positions, which are interpolated once per node ([`_interp_treepoint`](@ref)).

Pipeline:

1. **Numbering + hanging detection.** One `iterate_points` traversal per tree (`mindim = 0`)
   fires the [`LnodesVisitor`](@ref) callbacks: corners create + scatter node ids, non-conforming
   faces and (3D) edges create the hanging vertices and record their constraints as `(element,
   slot)` references into `E`.
2. **Inter-tree hanging.** [`_iterate_interface_hanging!`](@ref) descends each shared tree face
   from both sides and records the cross-tree hanging constraints (no-op for a single tree).
3. **Cross-tree identity.** A shared boundary node has one provisional id per incident tree;
   [`_merge_intertree_nodes!`](@ref) aliases them onto a single owner through the per-tree
   boundary tables.
4. **Global numbering + cells + constraints.** [`_global_numbering`](@ref) (Alg 6.1) assigns the
   final dense ids in one sweep over `E`, [`_build_cells`](@ref) materializes the cells, the
   constraint records are resolved against `E`, and [`reconstruct_facetsets`](@ref) /
   [`reconstruct_cellsets`](@ref) carry the named boundaries and subdomains onto the refined grid
   (vertex and node sets are not transferred, see the warning above).

Requires a 2:1-balanced forest (see [`balanceforest!`](@ref)) — balance is what guarantees
hanging vertices are simple feature midpoints with non-hanging masters, and it is checked (an
unbalanced forest leaves element vertices without node ids, which raises an error).
"""
function creategrid(forest::ForestBWG{dim, C, T}) where {dim, C, T}
    node_map = dim == 2 ? node_map₂ : node_map₃
    celltype = dim == 2 ? Quadrilateral : Hexahedron
    NV = 2^dim
    ncells = getncells(forest)
    ntrees = length(forest.cells)
    offsets = _element_offsets(forest)

    E = zeros(Int, NV, ncells)
    nodecoords_prov = Vec{dim, T}[]
    sizehint!(nodecoords_prov, ncells)
    bnd = [Tuple{UInt64, Int}[] for _ in 1:ntrees]
    cons2 = Tuple{Int, ERef, ERef}[]
    cons4 = Tuple{Int, ERef, ERef, ERef, ERef}[]
    cnt = Ref(0)

    # Phase 1 — numbering + scatter + hanging constraints, one traversal per tree (the
    # iterator scratch is shared: all trees of a forest have the same maximum level). Each
    # tree's boundary table is sorted right after its traversal (each node is created exactly
    # once, so the tables hold no duplicates).
    sc = IterScratch(forest.cells[1])
    for (k, tree) in enumerate(forest.cells)
        visitor = LnodesVisitor(
            E, nodecoords_prov, bnd[k], cons2, cons4,
            _treecorners(forest, k), tree.b, Int(_maximum_size(tree.b)), offsets[k], cnt
        )
        iterate_points(visitor, tree, sc; mindim = 0, maxdim = dim - 1, skip_conforming = true)
        sort!(bnd[k]; alg = QuickSort, by = first)
    end

    # Phase 2 — inter-tree hanging constraints (reads ids off `E`; no-op for a single tree).
    _iterate_interface_hanging!(cons2, cons4, E, offsets, forest)

    # Phase 3 — cross-tree identity: alias the provisional ids of shared boundary nodes onto
    # their owner (recorded in `alias`, an array — the canonical lookup is an index).
    nprov = cnt[]
    alias = collect(1:nprov)
    _merge_intertree_nodes!(forest, bnd, alias)

    # Phase 4 — final numbering (Alg 6.1 sweep), cells, constraint resolution.
    final_of_prov, nodecoords = _global_numbering(E, alias, nodecoords_prov)
    cells = _build_cells(celltype, E, node_map, final_of_prov, Val(NV))
    hnodes = Dict{Int, Vector{Int}}()
    for (p, m1, m2) in cons2
        hnodes[final_of_prov[p]] = [final_of_prov[E[m1[2], m1[1]]], final_of_prov[E[m2[2], m2[1]]]]
    end
    for (p, m1, m2, m3, m4) in cons4
        hnodes[final_of_prov[p]] = [
            final_of_prov[E[m1[2], m1[1]]], final_of_prov[E[m2[2], m2[1]]],
            final_of_prov[E[m3[2], m3[1]]], final_of_prov[E[m4[2], m4[1]]],
        ]
    end
    return NonConformingGrid(
        cells, Node.(nodecoords);
        conformity_info = hnodes,
        facetsets = reconstruct_facetsets(forest),
        cellsets = reconstruct_cellsets(forest),
    )
end

@noinline _skeleton_unbalanced_error() = throw(ArgumentError("facetskeleton requires a 2:1-balanced forest (a face child of a refined neighbour is not a leaf) — call balanceforest! first"))

"""
    FacetSkeletonVisitor{dim}

Per-tree face callback of [`facetskeleton`](@ref)'s intra-tree traversal. At each
(non-hanging) face point the `LeafSupport` holds the leaves of both sides of the face; the
visitor recovers each side's local facet index from the face normal (the point's degenerate
axis, sides told apart by the anchor comparison of `_side_mask`) and pushes the
facet pair(s): one pair for a conforming face, one pair per fine child — fine side first,
the coarse leaf second — for a non-conforming one. Face points on the tree boundary
(anchor `0`/`2^b` along the normal) are skipped: their support is one-sided within the
tree, and they belong to either the inter-tree descent or the domain boundary.
"""
struct FacetSkeletonVisitor{dim}
    skel::Vector{NTuple{2, FacetIndex}}
    perm::Vector{Int}   # p4est face index -> Ferrite facet index
    off::Int            # the tree's element offset (`_element_offsets`)
    maxsize::Int        # 2^b, octree coordinate of the far tree boundary
end

function (v::FacetSkeletonVisitor{dim})(c::IteratePoint{dim}, ls::LeafSupport) where {dim}
    d = 1                                              # the degenerate axis = face normal
    while c.axes[d]
        d += 1
    end
    p = c.anchor[d]
    (p == 0 || p == v.maxsize) && return               # tree-boundary face
    fbelow = v.perm[2 * d]                             # facet toward the face of a leaf below it (+ side)
    fabove = v.perm[2 * d - 1]                         # ... of a leaf above it (- side)
    if length(ls) == 2                                 # conforming: one equal-size leaf per side
        below1 = Int(ls[1].xyz[d]) != p
        push!(
            v.skel, (
                FacetIndex(v.off + ls.idxs[1], below1 ? fbelow : fabove),
                FacetIndex(v.off + ls.idxs[2], below1 ? fabove : fbelow),
            )
        )
    else                                               # hanging: the coarse leaf sits at the point's own level
        ci = findfirst(o -> Int(o.l) == c.level, ls.octs)::Int
        cbelow = Int(ls[ci].xyz[d]) != p
        coarse = FacetIndex(v.off + ls.idxs[ci], cbelow ? fbelow : fabove)
        ffine = cbelow ? fabove : fbelow
        for i in 1:length(ls)
            i == ci && continue
            ls.idxs[i] == 0 && _skeleton_unbalanced_error()
            push!(v.skel, (FacetIndex(v.off + ls.idxs[i], ffine), coarse))
        end
    end
    return
end

"""
    _emit_interface_facets!(skel, perm, off, lvs, lo, hi, oct, f, b, coarse)

One-sided descent of the refined side of a hanging inter-tree interface: enumerate the
leaf subfacets on `oct`'s face `f` (leaves `lvs[lo:hi]`, element offset `off`) and pair
each with the coarse side's `FacetIndex` `coarse`, fine side first. Recursing to the
leaves instead of stopping at `oct`'s face children tolerates interfaces with a >1 level
jump, like `_subtree_corner_ref`'s descent.
"""
function _emit_interface_facets!(skel::Vector{NTuple{2, FacetIndex}}, perm::Vector{Int}, off::Int, lvs, lo::Int, hi::Int, oct::OctantBWG{dim, N}, f::Int, b, coarse::FacetIndex) where {dim, N}
    if _isleaf(lvs, lo, hi, oct)
        push!(skel, (FacetIndex(off + lo, perm[f]), coarse))
        return
    end
    kb = split_bounds(lvs, lo, hi, oct, b)
    ch = children(oct, b)
    fc = face(oct, f, b)
    for j in 1:N
        contains_facet(fc, face(ch[j], f, b)) || continue
        _emit_interface_facets!(skel, perm, off, lvs, kb[j], kb[j + 1] - 1, ch[j], f, b, coarse)
    end
    return
end

"""
    _iter_interface_facets!(
        skel, perm, offL, offR, forest, kL, lvsL, octL, loL, hiL, fL,
        kR, lvsR, octR, loR, hiR, fR, bL, bR
    )

[`facetskeleton`](@ref)'s inter-tree counterpart of [`_iter_interface!`](@ref): the same
synchronized two-sided descent of a shared tree face, but emitting the facet pairs of the
interface instead of hanging-node constraints. Each shared face is descended once per
direction, so to emit exactly once a conforming leaf pair fires only from the direction
with the smaller `(tree, face)` key, and a hanging interface only when the coarse side is
`kL` — the fine subfacets then live in tree `kR`'s frame and are enumerated by
[`_emit_interface_facets!`](@ref), fine side first.
"""
function _iter_interface_facets!(
        skel::Vector{NTuple{2, FacetIndex}}, perm::Vector{Int}, offL::Int, offR::Int, forest::ForestBWG, kL::Int, lvsL, octL::OctantBWG{dim, N}, loL::Int, hiL::Int, fL::Int,
        kR::Int, lvsR, octR::OctantBWG{dim, N}, loR::Int, hiR::Int, fR::Int, bL::Integer, bR::Integer
    ) where {dim, N}
    lL = _isleaf(lvsL, loL, hiL, octL)
    lR = _isleaf(lvsR, loR, hiR, octR)
    if lL && lR                                                  # conforming leaf pair
        if (kL, fL) < (kR, fR)
            push!(skel, (FacetIndex(offL + loL, perm[fL]), FacetIndex(offR + loR, perm[fR])))
        end
        return
    end
    if lL && !lR                                                 # kL coarse, kR refined -> hanging (fine = kR)
        _emit_interface_facets!(skel, perm, offR, lvsR, loR, hiR, octR, fR, bR, FacetIndex(offL + loL, perm[fL]))
        return
    elseif !lL && lR                                             # kL refined, kR coarse -> caught from (kR,fR)
        return
    end
    kb = split_bounds(lvsL, loL, hiL, octL, bL); cL = children(octL, bL)
    kbR = split_bounds(lvsR, loR, hiR, octR, bR); cR = children(octR, bR)
    for i in 1:N
        contains_facet(face(octL, fL, bL), face(cL[i], fL, bL)) || continue   # child i on the shared face
        nbR = transform_facet(forest, kR, fR, facet_neighbor(cL[i], fL, bL))  # its image in kR
        for j in 1:N
            cR[j] == nbR || continue
            _iter_interface_facets!(
                skel, perm, offL, offR, forest, kL, lvsL, cL[i], kb[i], kb[i + 1] - 1, fL,
                kR, lvsR, cR[j], kbR[j], kbR[j + 1] - 1, fR, bL, bR
            )
            break
        end
    end
    return
end

"""
    facetskeleton(forest::ForestBWG{dim}) -> Vector{NTuple{2, FacetIndex}}

Materialize the interior facet skeleton of the *refined* forest. The skeleton holds one entry
per leaf-level facet interface, as a pair of `FacetIndex` into the grid returned by [`creategrid`](@ref)
(the cell numbering is identical — tree by tree, leaves in Morton order). For a conforming
interface the pair holds the two equal-size cells sharing the facet. For a non-conforming
(hanging) interface each **fine subfacet** gets its own pair, fine side first and the
coarse cell's facet second — the orientation facet-jump (Kelly-type) error estimators
integrate: over the fine subfacet, evaluating the neighbour on the coarse side.

Domain-boundary facets are not part of the skeleton: a `FacetIndex` of the materialized
grid that appears in no pair lies on the domain boundary, so the boundary is recovered by
a complement sweep over all cell facets.

In contrast to `facetskeleton(::ExclusiveTopology, ::AbstractGrid)`, this adjacency cannot
be reconstructed from the materialized grid's topology — `ExclusiveTopology` only knows
conforming neighbourhoods, while the true adjacency of the refined forest contains
coarse↔fine (hanging) and across-tree interfaces. It is instead read off the forest: one
face-only [`iterate_points`](@ref) traversal per tree (`mindim = maxdim = dim - 1`,
[`FacetSkeletonVisitor`](@ref)) covers the intra-tree interfaces, and a two-sided
lock-step descent of every shared tree face ([`_iter_interface_facets!`](@ref)) the
inter-tree ones — the same primitives [`creategrid`](@ref) uses for node numbering and
hanging-constraint detection.

Requires a 2:1-balanced forest (see [`balanceforest!`](@ref)), like `creategrid`.
"""
function Ferrite.facetskeleton(forest::ForestBWG{dim}) where {dim}
    perm = dim == 2 ? 𝒱₂_perm : 𝒱₃_perm
    perminv = dim == 2 ? 𝒱₂_perm_inv : 𝒱₃_perm_inv
    offsets = _element_offsets(forest)
    skel = NTuple{2, FacetIndex}[]

    # Intra-tree interfaces: one face-only traversal per tree (shared scratch, cf. creategrid).
    sc = IterScratch(forest.cells[1])
    for (k, tree) in enumerate(forest.cells)
        visitor = FacetSkeletonVisitor{dim}(skel, perm, offsets[k], Int(_maximum_size(tree.b)))
        iterate_points(visitor, tree, sc; mindim = dim - 1, maxdim = dim - 1)
    end

    # Inter-tree interfaces: descend each shared tree face from both sides in lock-step
    # (cf. `_iterate_interface_hanging!`).
    fn = Ferrite.get_facet_facet_neighborhood(forest)
    r = root(dim)
    for (k, tree) in enumerate(forest.cells)
        for f in 1:(2 * dim)
            nb = fn[k, perm[f]]
            isempty(nb) && continue
            k′ = nb[1][1]; f′ = perminv[nb[1][2]]
            treeR = forest.cells[k′]
            _iter_interface_facets!(
                skel, perm, offsets[k], offsets[k′], forest, k, tree.leaves, r, 1, length(tree.leaves), f,
                k′, treeR.leaves, r, 1, length(treeR.leaves), f′, tree.b, treeR.b
            )
        end
    end
    return skel
end
