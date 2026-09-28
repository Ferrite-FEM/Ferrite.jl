# The forest-level part of the simplex AMR ([BH2016](@cite)): inter-tree point transforms,
# the inter-tree half of the 2:1 balancing, and the materialization (`creategrid`,
# `facetskeleton`, facet-set transfer) of forests of `SimplexTreeBH` trees. The algorithms
# shared with octrees — `refine!`/`coarsen!`, `balancetree`, the set transfer of cell sets —
# live in `forest.jl`.
#
# The one structural difference to the octree code: two trees sharing a face have unrelated
# Kuhn structures, so a simplex can be carried into a neighbouring tree only as a set of
# *points* — integer points transform exactly through their root barycentric coordinates
# (`_transform_point`) — never as an element. Every inter-tree operation is therefore
# formulated on element corners and edge midpoints, and the neighbouring element across a
# shared root face is rebuilt from its transformed face (`_leaf_on_root_face`).

const SimplexForest{dim} = ForestBWG{dim, <:SimplexTreeBH}

@noinline _unbalanced_simplex_error() = throw(ArgumentError("a 2:1-balanced forest is required (two touching leaves differ by more than one level) — call balanceforest! first"))

"""
    _transform_point(forest, k, k′, p) -> NTuple{dim, T}

Express the integer point `p` of tree `k`'s root frame in tree `k′`'s. `p` must lie on the root
sub-simplex the two trees share (a face, edge or vertex), so only the barycentric weights of
shared macro nodes are nonzero: `p′ = Σ μᵢ S0[σ(i)]` with `σ(i)` the position of macro node `i`
of `k` in `k′`. Exact integer arithmetic; `nothing` if some weighted node is not shared.
"""
function _transform_point(forest::SimplexForest{dim}, k::Integer, k′::Integer, p::NTuple{dim, T}) where {dim, T}
    L = _maximum_size(forest.cells[k].b)
    μ = _barycentric(p, L)
    nodes = forest.cells[k].nodes
    nodes′ = forest.cells[k′].nodes
    S0 = _unit_simplex(Val(dim))
    p′ = ntuple(_ -> zero(T), Val(dim))
    for i in 1:(dim + 1)
        μ[i] == 0 && continue
        j = findfirst(==(nodes[i]), nodes′)
        j === nothing && return nothing
        p′ = p′ .+ T(μ[i]) .* S0[j]
    end
    return p′
end

"""
    _trees_touching!(buf, forest, k, μ) -> buf

The trees other than `k` that contain the smallest root sub-simplex of tree `k` holding a
point with root barycentric coordinates `μ`: those containing every macro node with
`μᵢ ≠ 0` (the macro mesh is conforming, so cells meet in common sub-simplices). Read off the
topology's `vertex_to_cell`, no face/edge/vertex case split needed.
"""
function _trees_touching!(buf::Vector{Int}, forest::SimplexForest{dim}, k::Integer, μ) where {dim}
    empty!(buf)
    nodes = forest.cells[k].nodes
    v2c = forest.topology.vertex_to_cell
    first = findfirst(!iszero, μ)::Int
    for k′ in v2c[nodes[first]]
        k′ == k && continue
        shared = true
        for i in (first + 1):(dim + 1)
            μ[i] == 0 && continue
            if k′ ∉ v2c[nodes[i]]
                shared = false
                break
            end
        end
        shared && push!(buf, k′)
    end
    return buf
end

##### PHYSICAL GEOMETRY OF A TREE #####

# Physical coordinates of tree `k`'s macro nodes, in the paper's (== Ferrite's) vertex order.
@inline function _macro_coordinates(forest::SimplexForest{dim}, k::Integer) where {dim}
    nodes = forest.nodes
    return ntuple(i -> get_node_coordinate(nodes[forest.cells[k].nodes[i]]), Val(dim + 1))
end

"""
    _physical_point(X, L, p) -> Vec

Map the integer point `p` of a tree with macro corners `X` to physical space: the affine
map `x = Σ (μᵢ / L) Xᵢ` through the root barycentric coordinates (exact for straight-sided
simplices). The simplex counterpart of `_interp_treepoint`.
"""
@inline function _physical_point(X::NTuple{N, Vec{dim, V}}, L, p::NTuple{dim, <:Integer}) where {N, dim, V}
    μ = _barycentric(p, L)
    x = zero(Vec{dim, V})
    for i in 1:N
        x += (V(μ[i]) / V(L)) * X[i]
    end
    return x
end

"""
    _tree_map_sign(X) -> Int

Sign of the determinant of the affine map from the tree frame to physical space (`X` the
physical macro corners): the orientation of the physical macro cell divided by that of the
root in its frame, which is `-1` in 3D (`S0`'s edge axes are an odd permutation) and `+1` in 2D.
With it a leaf's physical orientation is `_tree_map_sign * _orientation(leaf)`.
"""
function _tree_map_sign(X::NTuple{4, Vec{3, V}}) where {V}
    d = (X[2] - X[1]) ⋅ ((X[3] - X[1]) × (X[4] - X[1]))
    d == 0 && throw(ArgumentError("degenerate macro tetrahedron"))
    return d > 0 ? -1 : 1
end
function _tree_map_sign(X::NTuple{3, Vec{2, V}}) where {V}
    a = X[2] - X[1]
    c = X[3] - X[1]
    d = a[1] * c[2] - a[2] * c[1]
    d == 0 && throw(ArgumentError("degenerate macro triangle"))
    return d > 0 ? 1 : -1
end
_tree_map_signs(forest::SimplexForest) = [_tree_map_sign(_macro_coordinates(forest, k)) for k in 1:length(forest.cells)]

# Ferrite local facet index of canonical face `f` of leaf `o` in a tree with map sign `sgn`.
_ferrite_facet(o::SimplexBH{dim}, f::Integer, sgn::Integer) where {dim} = _ferrite_face(Val(dim), sgn * _orientation(o) > 0, f)

##### BALANCING ACROSS TREES #####

function _balance_context(forest::ForestBWG{dim, C}) where {dim, C <: SimplexTreeBH}
    return (buf = Int[], sbuf = _leaftype(C)[])
end

# The balance checks below visit only the corners of a level-`l` leaf that lie on the
# level-`(l - 1)` lattice, i.e. the corners it shares with its parent. That is sufficient: a
# violating coarse leaf `A` (two or more levels coarser) touching the leaf contains one of its
# corners; if that corner is a midpoint of an edge `e` of the parent, `A` contains the whole of
# `e` (the level-`(l - 1)` Kuhn simplices form a simplicial complex, so a closed simplex meeting
# the relative interior of `e` has `e` as a face) and hence the endpoints of `e`, which are
# parent corners carried by the parent's corner children — leaves or refined — whose own
# corner visit then detects `A`. Skipping the other corners avoids two thirds of the queries.

"""
    _balance_tree_boundary!(forest::SimplexForest, k, tree, ctx)

The inter-tree 2:1 balance for a simplex tree, vertex-based (a finer leaf touches a coarser
one iff one of its corners lies in the coarser one's closure): every lattice corner `v` (see
above) of a leaf of level `l ≥ 2` on the root boundary is carried into each tree sharing it,
where the level-`(l - 1)` simplices containing it must not be hidden inside a coarser leaf
([`_resolve_corner!`](@ref)). Refines one level per violation; `balanceforest!` iterates to
the fixed point.
"""
function _balance_tree_boundary!(forest::SimplexForest{dim}, k, tree, ctx) where {dim}
    b = tree.b
    L = _maximum_size(b)
    for o in tree.leaves
        o.l >= 2 || continue # level-1 pivots need no balancing
        hp = eltype(o.xyz)(_compute_size(b, o.l - 1))
        for v in vertices(o, b)
            _on_lattice(v, hp) || continue
            μ = _barycentric(v, L)
            any(iszero, μ) || continue # interior corner
            _trees_touching!(ctx.buf, forest, k, μ)
            for k′ in ctx.buf
                v′ = _transform_point(forest, k, k′, v)::typeof(v)
                _resolve_corner!(forest.cells[k′], v′, Int(o.l) - 1, ctx.sbuf, true)
            end
        end
    end
    return
end

"""
    _isbalanced(forest::SimplexForest) -> Bool

Whether the forest satisfies the vertex 2:1 balance: at every lattice corner (see above) of
every leaf of level `l ≥ 2` — in its own tree and, on the root boundary, in every tree sharing
the corner — no level-`(l - 1)` simplex containing the corner lies inside a coarser leaf.
`creategrid` and `facetskeleton` check this up front: on an unbalanced forest nodes may lie in
the interior of a coarse leaf's edge or face without being its midpoint, and would silently
stay unconstrained. Each `(tree, point, level)` query is made once (a corner is shared by many
leaves) against the trees' precomputed sort keys.
"""
function _isbalanced(forest::ForestBWG{dim, C}) where {dim, C <: SimplexTreeBH}
    buf = Int[]
    sbuf = _leaftype(C)[]
    keys = [[_sortkey(o, tree.b) for o in tree.leaves] for tree in forest.cells]
    seen = Set{Tuple{Int, NTuple{dim, Int}, Int}}()
    for (k, tree) in enumerate(forest.cells)
        b = tree.b
        L = _maximum_size(b)
        for o in tree.leaves
            o.l >= 2 || continue
            l = Int(o.l) - 1
            hp = eltype(o.xyz)(_compute_size(b, l))
            for v in vertices(o, b)
                _on_lattice(v, hp) || continue
                vi = Int.(v)
                (k, vi, l) ∈ seen && continue
                push!(seen, (k, vi, l))
                _resolve_corner!(tree, v, l, sbuf, false; keys = keys[k]) && return false
                μ = _barycentric(v, L)
                any(iszero, μ) || continue
                _trees_touching!(buf, forest, k, μ)
                for k′ in buf
                    v′ = _transform_point(forest, k, k′, v)::typeof(v)
                    (k′, Int.(v′), l) ∈ seen && continue
                    push!(seen, (k′, Int.(v′), l))
                    _resolve_corner!(forest.cells[k′], v′, l, sbuf, false; keys = keys[k′]) && return false
                end
            end
        end
    end
    return true
end

##### MATERIALIZATION #####

"""
    creategrid(forest::ForestBWG{dim, <:SimplexTreeBH}) -> NonConformingGrid

Materialize a forest of Burstedde–Holke simplex trees into a `NonConformingGrid` of
`Triangle`/`Tetrahedron` cells; the simplex counterpart of the octree `creategrid`, with the
same cell numbering (tree by tree, leaves in curve order) and the same `conformity_info`
layout. Node identity is decided on integer coordinates: within a tree a node is a distinct
leaf corner, across trees corners on the root boundary are matched through
[`_transform_point`](@ref). Bey's refinement introduces edge midpoints only, so a hanging
node is exactly a node that is the midpoint of an edge of some (coarser) leaf, constrained
by that edge's endpoints: every leaf reports the midpoints of its edges that are nodes, in
its own tree and — for edges on the root boundary — in the trees sharing the edge.

Requires a 2:1-balanced forest ([`_isbalanced`](@ref)); throws otherwise.
"""
function creategrid(forest::ForestBWG{dim, C, T}) where {dim, C <: SimplexTreeBH, T}
    _isbalanced(forest) || _unbalanced_simplex_error()
    NV = dim + 1
    celltype = dim == 2 ? Triangle : Tetrahedron
    ncells = getncells(forest)
    ntrees = length(forest.cells)
    offsets = _element_offsets(forest)

    E = zeros(Int, NV, ncells)
    nodecoords_prov = Vec{dim, T}[]
    bnd = [Tuple{UInt64, Int}[] for _ in 1:ntrees]     # (packed coord, provisional id) of the root-boundary nodes
    bndcoords = [NTuple{dim, Int}[] for _ in 1:ntrees] # their coordinates, same order
    cons = Tuple{Int, Int, Int}[]                        # (hanging node, master, master), provisional ids
    buf = Int[]
    cnt = 0

    # Phase 1 — per tree: number the distinct leaf corners, fill `E`, record the boundary
    # table and the in-tree hanging nodes.
    for (k, tree) in enumerate(forest.cells)
        b = tree.b
        L = _maximum_size(b)
        X = _macro_coordinates(forest, k)
        coords = NTuple{dim, Int}[]
        sizehint!(coords, NV * length(tree.leaves))
        for leaf in tree.leaves
            for v in vertices(leaf, b)
                push!(coords, Int.(v))
            end
        end
        sort!(coords; by = _packcoord)
        unique!(coords)
        table = Vector{Tuple{UInt64, Int}}(undef, length(coords))
        for (i, c) in enumerate(coords)
            id = cnt + i
            table[i] = (_packcoord(c), id)
            push!(nodecoords_prov, _physical_point(X, L, c))
            if any(iszero, _barycentric(c, L))
                push!(bnd[k], (_packcoord(c), id))
                push!(bndcoords[k], c)
            end
        end
        cnt += length(coords)
        for (j, leaf) in enumerate(tree.leaves)
            gid = offsets[k] + j
            vs = vertices(leaf, b)
            for s in 1:NV
                E[s, gid] = _bnd_lookup(table, vs[s], b)
            end
            leaf.l < b || continue # maximum-level leaves have no finer neighbours
            for (a, c) in _edge_pairs(Val(dim))
                m = (vs[a] .+ vs[c]) .÷ 2
                idm = _bnd_lookup(table, m, b)
                idm == 0 || push!(cons, (idm, E[a, gid], E[c, gid]))
            end
        end
    end

    # Phase 2 — hanging nodes across trees: a midpoint of a boundary edge of a coarse leaf
    # that is a node of a neighbouring tree only.
    for (k, tree) in enumerate(forest.cells)
        b = tree.b
        L = _maximum_size(b)
        for (j, leaf) in enumerate(tree.leaves)
            leaf.l < b || continue
            _touches_tree_boundary(leaf, b) || continue
            gid = offsets[k] + j
            vs = vertices(leaf, b)
            for (a, c) in _edge_pairs(Val(dim))
                μa = _barycentric(vs[a], L)
                μc = _barycentric(vs[c], L)
                any(i -> μa[i] == 0 && μc[i] == 0, 1:NV) || continue # edge not on the root boundary
                m = (vs[a] .+ vs[c]) .÷ 2
                _trees_touching!(buf, forest, k, _barycentric(m, L))
                for k′ in buf
                    m′ = _transform_point(forest, k, k′, m)::typeof(m)
                    idm = _bnd_lookup(bnd[k′], m′, forest.cells[k′].b)
                    idm == 0 || push!(cons, (idm, E[a, gid], E[c, gid]))
                end
            end
        end
    end

    # Phase 3 — cross-tree identity: alias every boundary node onto its image in the lowest
    # tree holding it (a node hanging on the coarse side of an interface has no image there).
    alias = collect(1:cnt)
    for k in 2:ntrees
        L = _maximum_size(forest.cells[k].b)
        for (n, c) in enumerate(bndcoords[k])
            id = bnd[k][n][2]
            _trees_touching!(buf, forest, k, _barycentric(c, L))
            for k′ in buf
                k′ < k || continue
                c′ = _transform_point(forest, k, k′, c)::typeof(c)
                id′ = _bnd_lookup(bnd[k′], c′, forest.cells[k′].b)
                if id′ != 0
                    alias[id] = alias[id′]
                    break
                end
            end
        end
    end

    # Phase 4 — final numbering, cells, constraints.
    final_of_prov, nodecoords = _global_numbering(E, alias, nodecoords_prov)
    cells = _build_simplex_cells(celltype, forest, E, final_of_prov, offsets, _tree_map_signs(forest))
    hnodes = Dict{Int, Vector{Int}}()
    for (m, p, q) in cons
        hnodes[final_of_prov[m]] = [final_of_prov[p], final_of_prov[q]]
    end
    return NonConformingGrid(
        cells, Node.(nodecoords);
        conformity_info = hnodes,
        facetsets = reconstruct_facetsets(forest),
        cellsets = reconstruct_cellsets(forest),
    )
end

"""
    _build_simplex_cells(::Type{CT}, forest, E, final_of_prov, offsets, signs) -> Vector{CT}

Materialize the cells from the element-node matrix `E` (provisional ids in the canonical
corner order), applying the orientation permutation of each leaf (`_node_perm`) so every
cell is positively oriented in physical space. A function barrier like `_build_cells`.
"""
function _build_simplex_cells(::Type{CT}, forest::SimplexForest{dim}, E::Matrix{Int}, final_of_prov::Vector{Int}, offsets::Vector{Int}, signs::Vector{Int}) where {CT, dim}
    cells = Vector{CT}(undef, size(E, 2))
    for (k, tree) in enumerate(forest.cells)
        sgn = signs[k]
        @inbounds for (j, leaf) in enumerate(tree.leaves)
            gid = offsets[k] + j
            perm = _node_perm(Val(dim), sgn * _orientation(leaf) > 0)
            cells[gid] = CT(ntuple(m -> final_of_prov[E[perm[m], gid]], Val(dim + 1)))
        end
    end
    return cells
end

"""
    reconstruct_facetsets(forest::SimplexForest) -> Dict{String, OrderedSet{FacetIndex}}

Transfer the macro-mesh facet sets onto the materialized grid: a root face is tiled by the
leaf faces lying on it (`_root_face_of`; leaves of several types contribute), each converted
to Ferrite's local numbering per leaf orientation.
"""
function reconstruct_facetsets(forest::SimplexForest{dim}) where {dim}
    offsets = _element_offsets(forest)
    signs = _tree_map_signs(forest)
    new_facetsets = typeof(forest.facetsets)()
    for (name, facetset) in forest.facetsets
        new_facetset = typeof(facetset)()
        for facetidx in facetset
            k = facetidx[1]
            tree = forest.cells[k]
            rootface = _paper_root_face(Val(dim), facetidx[2])
            for (j, leaf) in enumerate(tree.leaves)
                _touches_tree_boundary(leaf, tree.b) || continue
                for f in 1:(dim + 1)
                    _root_face_of(leaf, f, tree.b) == rootface || continue
                    push!(new_facetset, FacetIndex(offsets[k] + j, _ferrite_facet(leaf, f, signs[k])))
                end
            end
        end
        new_facetsets[name] = new_facetset
    end
    return new_facetsets
end

"""
    facetskeleton(forest::ForestBWG{dim, <:SimplexTreeBH}) -> Vector{NTuple{2, FacetIndex}}

The interior facet skeleton of the refined forest, with the same conventions as the octree
method: one pair per leaf-level interface, a conforming pair holding the two equal-level
cells, a hanging interface one pair per fine subfacet with the fine side first. Per leaf and
face, the same-level neighbour (Algorithm 4.6, rebuilt in the neighbouring tree across a
root face) is either a leaf (conforming, emitted from the lower cell id), the child of a
leaf (hanging, emitted here with the coarse leaf's face that contains the fine one) or
refined (emitted from the finer side). Requires a 2:1-balanced forest.
"""
function Ferrite.facetskeleton(forest::SimplexForest{dim}) where {dim}
    _isbalanced(forest) || _unbalanced_simplex_error()
    offsets = _element_offsets(forest)
    signs = _tree_map_signs(forest)
    skel = NTuple{2, FacetIndex}[]
    buf = Int[]
    for (k, tree) in enumerate(forest.cells)
        b = tree.b
        for (j, leaf) in enumerate(tree.leaves)
            this_cell = offsets[k] + j
            for f in 1:(dim + 1)
                this = FacetIndex(this_cell, _ferrite_facet(leaf, f, signs[k]))
                nb, f̃ = facet_neighbor_face(leaf, f, b)
                if inside(nb, b)
                    _push_facet_pair!(skel, forest, this, face(leaf, f, b), k, nb, f̃, offsets, signs)
                else
                    rootface = _root_face_of(leaf, f, b)::Int # the neighbour is outside iff the face is on the root boundary
                    _trees_touching!(buf, forest, k, ntuple(i -> i == rootface ? 0 : 1, Val(dim + 1)))
                    isempty(buf) && continue # domain boundary
                    k′ = buf[1]              # conforming macro mesh: one neighbour across a face
                    fv′ = map(v -> _transform_point(forest, k, k′, v)::typeof(v), face(leaf, f, b))
                    nb′ = _leaf_on_root_face(fv′, leaf.l, forest.cells[k′].b)
                    _push_facet_pair!(skel, forest, this, fv′, k′, nb′, _face_index(nb′, fv′, forest.cells[k′].b), offsets, signs)
                end
            end
        end
    end
    return skel
end

# Pair `this` (a leaf face with corners `fv` in tree `k′`'s frame) with what lies across it in
# tree `k′`: the same-level neighbour `nb` if it is a leaf, or its parent if that is a leaf.
function _push_facet_pair!(skel::Vector{NTuple{2, FacetIndex}}, forest::SimplexForest{dim}, this::FacetIndex, fv, k′::Integer, nb::SimplexBH, f̃::Integer, offsets::Vector{Int}, signs::Vector{Int}) where {dim}
    tree′ = forest.cells[k′]
    b′ = tree′.b
    leaves′ = tree′.leaves
    idx, found = _leaf_position(leaves′, nb, b′)
    if found
        other = offsets[k′] + idx
        this[1] < other && push!(skel, (this, FacetIndex(other, _ferrite_facet(nb, f̃, signs[k′]))))
        return
    end
    if idx > 1 && isancestor(leaves′[idx - 1], nb, b′)
        coarse = leaves′[idx - 1]
        coarse.l == nb.l - 1 || _skeleton_unbalanced_error()
        fc = _face_index(coarse, fv, b′) # the coarse face holding the fine one
        push!(skel, (this, FacetIndex(offsets[k′] + idx - 1, _ferrite_facet(coarse, fc, signs[k′]))))
    end
    return # otherwise the neighbour is refined and the pairs come from the finer side
end
