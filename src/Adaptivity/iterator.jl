# =============================================================================
# Literal point-centric node iterator — IBWG2015 Algorithm 5.2 (Iterate_interior)
# and 5.3 (Iterate) — and a materializer (`creategrid_iterator`) built on top of it.
#
# This is the *general* engine described in §5.2: one recursion keyed on a point `c`,
# carrying the support-leaf arrays `S` as state, descending over the child partition
# `part(c)` (eq 2.7), and firing a callback dispatched by `dim(c)` (volume / face /
# edge / corner). The fast `iterate_leaves` (volume) and `iterate_hanging` (face
# descent) in `BWG.jl` are *specializations* of the same §5 recursion; this file is
# the literal Alg 5.2/5.3 the paper presents, kept beside them for arbitrary
# per-dimension callbacks (higher-order Lnodes, face/edge functionals, …). See
# `docs/src/devdocs/AMR_iterator.md` §2.
#
# Everything is integer/topological (no physical coordinates): a *point* `c = (o, b)`
# (§2.1) is the axis-aligned integer box `IteratePoint`.
# =============================================================================

"""
    IteratePoint{dim}

A *point* `c = (o, b)` in the sense of [IBWG2015](@citet) §2.1 — the data type that
encompasses both octants and their lower-dimensional interfaces (corners, edges,
faces) and the octant volume. Encoded integer/topologically by

- `anchor` : the minimum integer (octree) corner of the point's box,
- `level`  : refinement level, so the box has extent `h = _compute_size(b, level)`,
- `axes`   : the `dim` directions the box extends along; `point_dim(c) = count(axes)`.

`point_dim(c)` is `dim(c)` from the paper: `dim` for a volume, `dim-1` for a face,
`1` for a 3D edge, `0` for a corner. Two points are equal iff their `(anchor, level,
axes)` agree — no physical coordinates, no rounding (cf. `iterate_hanging`).
"""
struct IteratePoint{dim}
    anchor::NTuple{dim, Int}
    level::Int
    axes::NTuple{dim, Bool}
end

point_dim(c::IteratePoint) = count(c.axes)

# The volume point (o, v0) of an octant (Remark 2.2: an octant is the point (o, v0)).
iteratepoint(o::OctantBWG{dim}) where {dim} = IteratePoint{dim}(map(Int, o.xyz), Int(o.l), ntuple(_ -> true, dim))

# Integer corner coordinates of a point's box: 2^point_dim(c) of them.
function _pt_corners(c::IteratePoint{dim}, b::Integer) where {dim}
    h = _compute_size(b, c.level)
    extdims = ntuple(d -> c.axes[d], dim)
    nd = point_dim(c)
    out = NTuple{dim, Int}[]
    for m in 0:(2^nd - 1)
        coord = c.anchor; e = 0
        for d in 1:dim
            extdims[d] || continue
            ((m >> e) & 1) == 1 && (coord = Base.setindex(coord, coord[d] + h, d))
            e += 1
        end
        push!(out, coord)
    end
    return out
end

# Is the point-box of `c` contained in the closure of octant `o`'s box? Used to test
# octant ∈ supp(point) (an octant at level(c) whose closure includes c, eq 2.11) and to
# pick the descent child toward a corner (atom_supp, Prop 2.8).
function _pt_in_oct_closure(c::IteratePoint{dim}, o::OctantBWG{dim}, b::Integer) where {dim}
    h = _compute_size(b, c.level); ho = _compute_size(b, o.l)
    for d in 1:dim
        clo = c.anchor[d]; chi = c.anchor[d] + (c.axes[d] ? h : 0)
        olo = Int(o.xyz[d]); ohi = olo + ho
        (olo <= clo && chi <= ohi) || return false
    end
    return true
end

# Geometric realization of the child-boundary-intersection set `B_∩^j` (eq 4.5 /
# `boundaryset`, line 14 of Alg 5.2): does child octant `ch` (level c.level+1) touch the
# point `c` (a feature of its parent at c.level)? True iff, along every axis where `c` is
# degenerate, `ch` lies on `c`'s side (its box straddles that coordinate). Picks exactly
# the parent's children adjacent to `c` — the ones in `leaf_supp(c)` under the 2:1 balance.
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

# Call `f(e)` for each point `e ∈ part(c)` (eq 2.7): the `3^dim(c)` points one level
# finer whose (open) domain lies strictly inside dom(c). Along each axis `c` extends, a
# child-partition point takes one of three slots — lower half `[x,x+h/2]`, the
# strictly-interior mid plane `{x+h/2}` (degenerate), or upper half `[x+h/2,x+h]`;
# degenerate axes of `c` stay fixed. No allocation of the point set (callback form).
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
        f(IteratePoint{dim}(anchor, c.level + 1, axes))
    end
    return
end

# Call `f(c)` for each point in the closure of the tree root (Alg 5.3 line 4, single
# tree): the root volume and all its boundary faces/edges/corners. Along each axis the
# feature is pinned to the low face (coord 0), spans the full root, or the high face.
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

# Descend the subtree of support octant `s` (leaves[lo:hi]) to the leaf whose closure
# contains the 0-point `c` — the realization of the `atom supp(c)` search (Alg 5.2 line
# 18 / Prop 2.8): at each level pick the child whose box contains the corner.
function _descend_to_corner(c::IteratePoint{dim}, s::OctantBWG{dim}, lo::Int, hi::Int, leaves, b::Integer) where {dim}
    o = s
    while !(lo == hi && leaves[lo] == o)
        k = split_bounds(leaves, lo, hi, o, b)
        ch = children(o, b)
        idx = 0
        for j in 1:length(ch)
            if _pt_in_oct_closure(c, ch[j], b)
                idx = j; break
            end
        end
        idx == 0 && return o
        o = ch[idx]; lo = k[idx]; hi = k[idx + 1] - 1
    end
    return o
end

# Preallocated scratch for the allocation-free descent (IBWG2015 §5.4: the recursion is
# `O(lmax)` deep, so all the per-node state — the support arrays `supp`/`S`, the
# `Split_array` results `childs`/`splits`, and the `leaf_supp` buffer `L` — is preallocated
# once per traversal and reused). Buffers are indexed by recursion depth: in a DFS only one
# root-to-node path is live, so depth `d`'s buffers are free to refill for the next sibling
# once depth `d`'s subtree returns. `M = N + 1` is the `split_bounds` tuple length.
struct IterScratch{N, M, OT}
    supp::Vector{Vector{OT}}                  # [depth] -> support octants of the point at this depth
    S::Vector{Vector{NTuple{2, Int}}}         # [depth] -> leaf index ranges, one per support octant
    childs::Vector{Vector{NTuple{N, OT}}}     # [depth] -> children of each support octant (Split_array)
    splits::Vector{Vector{NTuple{M, Int}}}    # [depth] -> split_bounds of each support octant
    L::Vector{OT}                             # reused leaf_supp buffer passed to the callback
end

function IterScratch(tree::OctreeBWG{dim, N, T}) where {dim, N, T}
    OT = OctantBWG{dim, N, T}
    nd = Int(tree.b) + 2                       # max recursion depth is the octree level + 1
    return IterScratch{N, N + 1, OT}(
        [OT[] for _ in 1:nd], [NTuple{2, Int}[] for _ in 1:nd],
        [NTuple{N, OT}[] for _ in 1:nd], [NTuple{N + 1, Int}[] for _ in 1:nd], OT[]
    )
end

"""
    _iterate_interior!(visit, c::IteratePoint, depth, sc::IterScratch, leaves, b, mindim)

[IBWG2015](@citet) Algorithm 5.2 (`Iterate_interior`), serial, allocation-free. The point
`c` at recursion `depth` has its support set in `sc.supp[depth]` (octants at `level(c)`
whose closure contains `c`, eq 2.11) and `sc.S[depth][i] = (lo, hi)` the index range in
`leaves` of the leaves descending from `supp[i]` (the `S` arrays of the paper). When `c`
is finalized (`c ∈ PΩ`), `visit(c, leaf_supp)` is called with `leaf_supp` the local leaf
support set (5.4); otherwise the recursion descends `part(c)`, slicing each `S[i]` with
`split_bounds`. `leaf_supp` is the reused buffer `sc.L` — **copy it if you retain it.**

Termination/finalization exactly as Alg 5.2:
- `dim(c) > 0`: `stop` iff some `supp[i]` is itself a leaf (`S[i] = {supp[i]}`, line 7).
  `leaf_supp` then collects each such leaf, plus — for the refined neighbours — their
  children adjacent to `c` (line 14, `B_∩^j` via `_child_touches_point`).
- `dim(c) == 0`: always `stop` (line 16); `leaf_supp` is the leaf of each support subtree
  touching the corner (lines 17-18, `atom supp`).
Hanging points are never visited: when a coarse support octant is a leaf the recursion
stops, so the finer features interior to `c` (the hanging ones) are skipped — exactly `PΩ`
(5.1, Fig 5). `mindim` is the §5.4 callback specialization (only recurse into / fire the
callback for points of dim `≥ mindim`). `is_relevant` (Alg 5.1) is `true` in serial.
"""
function _iterate_interior!(visit::F, c::IteratePoint{dim}, depth::Int, sc::IterScratch{N, M, OT}, leaves, b::Integer, mindim::Int) where {F, dim, N, M, OT}
    supp = sc.supp[depth]; S = sc.S[depth]
    m = length(supp)
    m == 0 && return
    anylocal = false
    for i in 1:m
        (S[i][1] <= S[i][2]) && (anylocal = true; break)
    end
    anylocal || return                                 # Alg 5.2 line 1 (serial: empty support)
    dimc = point_dim(c)

    if dimc == 0                                       # 0-point: always stop (lines 15-18)
        if dimc >= mindim
            L = sc.L; empty!(L)
            for i in 1:m
                o = _descend_to_corner(c, supp[i], S[i][1], S[i][2], leaves, b)
                o ∉ L && push!(L, o)                   # disjoint subtrees -> dedup is cheap (no Set)
            end
            visit(c, L)
        end
        return
    end

    stop = false
    for i in 1:m
        if S[i][1] == S[i][2] && leaves[S[i][1]] == supp[i]
            stop = true
            break
        end
    end
    if stop                                            # finalize: build leaf_supp (lines 5-14)
        if dimc >= mindim
            L = sc.L; empty!(L)
            for i in 1:m
                if S[i][1] == S[i][2] && leaves[S[i][1]] == supp[i]
                    supp[i] ∉ L && push!(L, supp[i])
                else
                    for ch in children(supp[i], b)
                        (_child_touches_point(ch, c, b) && ch ∉ L) && push!(L, ch)
                    end
                end
            end
            visit(c, L)
        end
        return
    end

    # No support octant is a leaf -> recurse over part(c) (lines 21-25). Every support
    # octant is internal; cache its children + leaf sub-ranges (H_i) in the depth buffers.
    childs = sc.childs[depth]; splits = sc.splits[depth]
    empty!(childs); empty!(splits)
    for i in 1:m
        push!(childs, children(supp[i], b))
        push!(splits, split_bounds(leaves, S[i][1], S[i][2], supp[i], b))
    end
    esupp = sc.supp[depth + 1]; eS = sc.S[depth + 1]   # the next depth's (reused) support buffers
    _foreach_partc(c, b) do e
        point_dim(e) >= mindim || return
        empty!(esupp); empty!(eS)
        for i in 1:m
            ch = childs[i]
            for j in 1:N
                if _pt_in_oct_closure(e, ch[j], b) && ch[j] ∉ esupp
                    push!(esupp, ch[j]); push!(eS, (splits[i][j], splits[i][j + 1] - 1))
                end
            end
        end
        _iterate_interior!(visit, e, depth + 1, sc, leaves, b, mindim)
        return
    end
    return
end

"""
    iterate_points(visit, tree::OctreeBWG; mindim = 0)
    iterate_points(visit, forest::ForestBWG; mindim = 0)

[IBWG2015](@citet) Algorithm 5.3 (`Iterate`), serial: drive `_iterate_interior!` from the
closure of each tree root. `visit(c::IteratePoint, leaf_supp)` is called once for every
point `c ∈ PΩ` (5.1) — every non-hanging volume / face / edge / corner — with `leaf_supp`
the leaves surrounding it. Use `point_dim(c)` to dispatch per dimension (volume `= dim`,
face `= dim-1`, edge `= 1`, corner `= 0`), or pass `mindim` for the §5.4 specialization
(e.g. `mindim = dim - 1` to visit only volumes + faces). The descent is allocation-free
(one `IterScratch` per tree); **`leaf_supp` is a reused buffer — copy it if you retain it.**

The forest driver loops over trees (serial Alg 5.3). Within a tree this visits `PΩ`
exactly; at *shared tree boundaries* a feature is currently visited once per incident
tree (its per-tree `leaf_supp` covers only that tree's leaves) — cross-tree coordinated
descent (single-visit boundary `leaf_supp` via the orientation transforms) is the
documented next step, consistent with `iterate_hanging`'s inter-tree handling.
"""
function iterate_points(visit::F, tree::OctreeBWG{dim}; mindim::Int = 0) where {F, dim}
    leaves = tree.leaves
    isempty(leaves) && return
    b = tree.b
    sc = IterScratch(tree)
    r = root(dim)                            # root octant (zero octant; eltype matches leaves)
    full = (1, length(leaves))
    _foreach_root_closure(Val(dim), b) do c
        # A root feature of dim < mindim leads only to lower-dim features -> skip; interior
        # features descend from the root volume, boundary ones from their own seed here.
        point_dim(c) >= mindim || return
        empty!(sc.supp[1]); push!(sc.supp[1], r)         # seed depth 1 with the single root support
        empty!(sc.S[1]); push!(sc.S[1], full)
        _iterate_interior!(visit, c, 1, sc, leaves, b, mindim)
        return
    end
    return
end

function iterate_points(visit::F, forest::ForestBWG{dim}; mindim::Int = 0) where {F, dim}
    for tree in forest.cells
        iterate_points(visit, tree; mindim)
    end
    return
end

# =============================================================================
# `creategrid_iterator` — the LNodes materializer (IBWG2015 §6) built on the literal
# engine above, kept beside `creategrid` (which uses the `iterate_leaves` +
# `iterate_hanging` specializations). Same `NonConformingGrid` output.
#
# Numbering, connectivity AND intra-tree hanging are produced in ONE `iterate_points`
# traversal per tree (the single-traversal Lnodes fusion §3 calls the point of the
# rewrite), routed through the one point engine instead of the two specializations:
#   - the leaf-volume callback (`dim(c)=dim`) numbers the leaf's vertices (min-Morton owner
#     = first encounter, since volume points are visited in Morton order) and pushes the
#     cell connectivity — identical to `creategrid`'s `_number_tree!`/`iterate_leaves` pass;
#   - the face callback (`dim(c)=dim-1`) emits the hanging nodes interior to every
#     *non-conforming* coarse face (center + edge midpoints, constrained by the coarse
#     face's corners — identical to `iterate_hanging`'s `_emit_coarse_face_int!`).
# Inter-tree hanging at shared tree boundaries uses the cross-tree two-sided face descent
# `_iterate_interface_hanging!` (same recursive split-descent primitive, carried across the
# shared face). Cross-tree node *identity* (a node shared between trees gets one global id)
# is still resolved by `_merge_cross_tree_nodes!` — the per-tree numbering produces a
# `(tree,coord)` key per incident tree, which the merge canonicalizes via the macro topology
# + orientation transforms. (The fully-faithful Alg 5.3 alternative — seeding the iterator
# from the *deduped* union of root closures so each shared node is visited once — would fold
# that merge into the seeding, but needs cross-tree corner+edge descent too; deferred.)
# Compaction, physical coords, cells and facetsets reuse `creategrid`'s helpers.
# =============================================================================

# Cross-tree hanging via a two-sided face descent — the iterator's recursive split-descent
# (the same primitive `iterate_points` uses for intra-tree faces) carried across a shared
# tree face. `octL ∈ tree kL` (leaves `lvsL`, native frame) and `octR ∈ tree kR` are images
# of each other across the shared face; they descend in lock-step at equal levels. When the
# `kL` side is a leaf and the `kR` side is refined, `kL` is the coarse side and the hanging
# nodes lie on `octR`'s face `fR` in the *fine* tree `kR`'s frame (genuine fine-leaf vertices).
# Children are matched across the boundary by the *validated* `transform_facet` pattern from
# `_iterate_hanging_inter!`, so no new orientation logic is introduced. Each shared face is
# descended once per direction (only the `kL`-coarse case emits; the `kR`-coarse case is
# emitted when the descent runs from `(kR, fR)`), matching `_iterate_hanging_inter!` exactly.
function _iter_interface!(hang, forest::ForestBWG, kL::Int, lvsL, octL::OctantBWG{dim, N}, loL::Int, hiL::Int, fL::Int,
        kR::Int, lvsR, octR::OctantBWG{dim, N}, loR::Int, hiR::Int, fR::Int, bL::Integer, bR::Integer) where {dim, N}
    lL = _isleaf(lvsL, loL, hiL, octL)
    lR = _isleaf(lvsR, loR, hiR, octR)
    lL && lR && return                                           # same-size leaves both sides -> conforming
    if lL && !lR                                                 # kL coarse, kR refined -> hanging (fine = kR)
        _emit_coarse_face_int!(hang, kR, face(octR, fR, bR))
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
            _iter_interface!(hang, forest, kL, lvsL, cL[i], kb[i], kb[i + 1] - 1, fL,
                kR, lvsR, cR[j], kbR[j], kbR[j + 1] - 1, fR, bL, bR)
            break
        end
    end
    return
end

# Forest inter-tree hanging: seed the cross-tree face descent at each shared tree face with
# the two tree roots. Replaces the per-boundary-leaf scan `_iterate_hanging_inter!`.
function _iterate_interface_hanging!(hang, forest::ForestBWG{dim}) where {dim}
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
            _iter_interface!(hang, forest, k, tree.leaves, r, 1, length(tree.leaves), f,
                k′, treeR.leaves, r, 1, length(treeR.leaves), f′, bL, treeR.b)
        end
    end
    return
end

# Number one leaf's vertices + push its cell connectivity. A top-level function barrier
# (concrete args) so the `ntuple`/`get!` closures compile without boxing — exactly the role
# `_number_tree!` plays for `creategrid`; called from the iterator's leaf-volume callback.
function _lnodes_number_leaf!(conns, nodeids, prov_key, leaf::OctantBWG, k::Int, b::Integer, node_map, ::Val{NV}) where {NV}
    v = vertices(leaf, b)
    ids = ntuple(Val(NV)) do i
        key = (k, map(Int, v[i]))
        get!(nodeids, key) do
            push!(prov_key, key); length(prov_key)
        end
    end
    push!(conns, ntuple(i -> ids[node_map[i]], Val(NV)))
    return
end

"""
    creategrid_iterator(forest::ForestBWG) -> NonConformingGrid

Materialize the same `NonConformingGrid` as [`creategrid`](@ref), driven entirely by the
literal point-centric iterator (`iterate_points`, IBWG2015 Alg 5.2/5.3) instead of the
`iterate_leaves` + `iterate_hanging` specializations. ONE traversal per tree (§5.4 face
specialization, `mindim = dim-1`: volumes + faces, no corner/edge descent) does it all:
the leaf-volume callback numbers vertices + builds connectivity (Morton order ⇒ same
min-Morton owner as `creategrid`), the non-conforming face callback emits the interior
hanging nodes. Inter-tree hanging uses the cross-tree two-sided face descent
`_iterate_interface_hanging!`. Cross-tree node *identity* still goes through
`_merge_cross_tree_nodes!` (per-tree numbering ⇒ one `(tree,coord)` key per incident tree,
canonicalized by the merge); compaction and cells reuse `creategrid`'s machinery. Produces
the **same grid** as `creategrid` (byte-identical) on all golden cases, at memory parity.
"""
function creategrid_iterator(forest::ForestBWG{dim}) where {dim}
    node_map = dim == 2 ? node_map₂ : node_map₃
    celltype = dim == 2 ? Quadrilateral : Hexahedron
    NV = 2^dim
    KeyT = Tuple{Int, NTuple{dim, Int}}
    ncells = getncells(forest)

    nodeids = Dict{KeyT, Int}(); sizehint!(nodeids, ncells)
    prov_key = KeyT[]; sizehint!(prov_key, ncells)
    conns = NTuple{NV, Int}[]; sizehint!(conns, ncells)
    hang = Dict{HangingKey{dim}, Vector{HangingKey{dim}}}()

    # Phase 1 — ONE point-iterator traversal per tree fuses numbering + connectivity (leaf
    # volume callback, Morton order) and intra-tree hanging (non-conforming face callback).
    for (k, tree) in enumerate(forest.cells)
        b = tree.b
        iterate_points(tree; mindim = dim - 1) do c, leaf_supp
            d = point_dim(c)
            if d == dim                                      # leaf volume: number vertices + cell
                _lnodes_number_leaf!(conns, nodeids, prov_key, leaf_supp[1], k, b, node_map, Val(NV))
            elseif d == dim - 1                              # face: hanging iff a finer leaf borders it
                any(o -> o.l > c.level, leaf_supp) || return
                _emit_coarse_face_int!(hang, k, _pt_corners(c, b))
            end
            return
        end
    end

    # Phase 1b — inter-tree hanging via the cross-tree two-sided face descent (iterator-style;
    # replaces the per-boundary-leaf scan). Single-tree forests have no shared faces -> no-op.
    _iterate_interface_hanging!(hang, forest)

    # Phase 3 — cross-tree identity merge + compaction + owner physical coords (reuse).
    _merge_cross_tree_nodes!(forest, nodeids, Dict{KeyT, KeyT}())
    treecorners = [_treecorners(forest, k) for k in eachindex(forest.cells)]
    nprov = length(prov_key)
    final_of_prov = Vector{Int}(undef, nprov)
    compact = Dict{Int, Int}(); sizehint!(compact, nprov)
    nodecoords = Vec{dim, Float64}[]
    for p in 1:nprov
        cid = nodeids[prov_key[p]]
        final_of_prov[p] = get!(compact, cid) do
            kk = prov_key[cid][1]
            push!(nodecoords, _transform_point(treecorners[kk], forest.cells[kk].b, prov_key[cid][2]))
            length(compact) + 1
        end
    end

    # Phase 4 — cells + hanging constraints.
    cells = _build_cells(celltype, conns, final_of_prov)
    hnodes = Dict{Int, Vector{Int}}()
    for (hkey, mkeys) in hang
        hnodes[compact[nodeids[hkey]]] = [compact[nodeids[m]] for m in mkeys]
    end
    return NonConformingGrid(cells, Node.(nodecoords); conformity_info = hnodes, facetsets = reconstruct_facetsets(forest))
end
