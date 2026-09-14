# Simplex elements and trees: the integer/topological core of the tetrahedral (and triangular)
# AMR after Burstedde & Holke ([BH2016](@cite)). A `SimplexBH` is a Kuhn simplex in a tree's
# integer coordinate system, identified by its Tet-id `(anchor, type)` plus its level; a
# `SimplexTreeBH` the leaf list of one macro simplex, sorted along the tetrahedral Morton
# (TM) curve of the paper. Everything here is local to a single tree — the forest-level parts
# (inter-tree point transforms, balancing, `creategrid`, `facetskeleton`) live in
# `simplex_forest.jl`.
#
# Conventions: a simplex *type* is a 0-based value `0:dim!-1` exactly as in the paper (it is
# data, not an index); every index — vertex, face, edge, child, axis — is 1-based. "Paper face
# `f`" (0-based, opposite vertex `x_f`) is Julia face `f + 1`, opposite vertex `f + 1`.
#
# The cube corners are numbered `c_0..c_7` with `x` varying fastest (`c1 = (1,0,0)`,
# `c2 = (0,1,0)`, `c4 = (0,0,1)`). The `dim!` types of the unit cube (eq. 3 of the paper) are
#   3D: S0 = (c0,c1,c5,c7)  S1 = (c0,c1,c3,c7)  S2 = (c0,c2,c3,c7)
#       S3 = (c0,c2,c6,c7)  S4 = (c0,c4,c6,c7)  S5 = (c0,c4,c5,c7)
#   2D: S0 = (c0,c1,c3)     S1 = (c0,c2,c3)
# and a tree's root is `2^b S0`, the simplex `0 ≤ y ≤ z ≤ x ≤ 2^b` (2D: `0 ≤ y ≤ x ≤ 2^b`).

"""
    SimplexBH{dim, T <: Integer}

A `dim`-simplex (`dim ∈ (2, 3)`) of a Burstedde–Holke tree: refinement level `l`, the integer
anchor node `xyz` (its corner with the smallest coordinates) and the Kuhn `type` in `0:dim!-1`.
Together with the tree's maximum level `b` this determines the simplex (Corollary 7 of
[BH2016](@citet)); see [`vertices`](@ref) for its corners.
"""
struct SimplexBH{dim, T <: Integer} <: AbstractElement{Ferrite.RefSimplex{dim}}
    l::T
    xyz::NTuple{dim, T}
    type::T
end

SimplexBH(l::Integer, xyz::NTuple{dim, T}, type::Integer) where {dim, T} = SimplexBH{dim, T}(T(l), xyz, T(type))

Base.hash(o::SimplexBH, h::UInt) = hash(o.type, hash(o.xyz, hash(o.l, h)))
Base.isequal(o1::SimplexBH, o2::SimplexBH) = o1 == o2

function Base.show(io::IO, ::MIME"text/plain", o::SimplexBH{dim}) where {dim}
    println(io, "SimplexBH{$dim}")
    println(io, "   l = $(o.l)")
    println(io, "   xyz = $(join(o.xyz, ','))")
    return println(io, "   type = $(o.type)")
end

_simplex_root(::Val{dim}, ::Type{T}) where {dim, T} = SimplexBH{dim, T}(zero(T), ntuple(_ -> zero(T), Val(dim)), zero(T))

##### LOOKUP TABLES (transcribed from [BH2016]; rows/columns are 0-based there, accessed `[t + 1][i + 1]` here) #####

# The two edge axes `(i, j)` of a type: `X1 = X0 + h eᵢ`, `X2 = X1 + h eⱼ`, `X3 = X0 + h(1,1,1)`
# (Algorithm 4.1). The third axis `k` completes the interior ordering `q[k] ≤ q[j] ≤ q[i]`.
const _EDGE_AXES3 = ((1, 3), (1, 2), (2, 1), (2, 3), (3, 2), (3, 1))
# 2D: `X1 = X0 + h eᵢ`, `X2 = X0 + (h, h)`; interior ordering `q[j] ≤ q[i]`.
const _EDGE_AXES2 = ((1,), (2,))

# Table 1: type of the `i`-th Bey child of a type-`t` simplex.
const _CT3 = (
    (0, 0, 0, 0, 4, 5, 2, 1),
    (1, 1, 1, 1, 3, 2, 5, 0),
    (2, 2, 2, 2, 0, 1, 4, 3),
    (3, 3, 3, 3, 5, 4, 1, 2),
    (4, 4, 4, 4, 2, 3, 0, 5),
    (5, 5, 5, 5, 1, 0, 3, 4),
)
const _CT2 = ((0, 0, 0, 1), (1, 1, 1, 0))

# Table 2: TM local index σ_t(i) of Bey child `i` of a type-`t` simplex. Siblings are ordered
# by their TM digit `(cube-id, type)` (eq. 15b), which is how these rows are verified in the
# tests against Tables 1 and 6–8.
const _SIGMA3 = (
    (0, 1, 4, 7, 2, 3, 6, 5),
    (0, 1, 5, 7, 3, 2, 6, 4),
    (0, 3, 4, 7, 1, 2, 6, 5),
    (0, 1, 6, 7, 3, 2, 4, 5),
    (0, 3, 5, 7, 1, 2, 4, 6),
    (0, 3, 6, 7, 2, 1, 4, 5),
)
const _SIGMA2 = ((0, 1, 3, 2), (0, 2, 3, 1))
# ... and its inverse: the Bey child with TM local index `k` (Algorithm 4.5).
const _SIGMA_INV3 = map(σ -> Tuple(invperm(σ .+ 1) .- 1), _SIGMA3)
const _SIGMA_INV2 = map(σ -> Tuple(invperm(σ .+ 1) .- 1), _SIGMA2)

# Bey's children (eq. 2): the anchor of child `i` is the midpoint of `X0` and `X_j`
# (Algorithm 4.4), `j` per child listed here (0-based vertex).
const _BEY_ANCHOR_VERTEX3 = (0, 1, 2, 3, 1, 1, 2, 2)
const _BEY_ANCHOR_VERTEX2 = (0, 1, 2, 1)

# Figure 8: type of the parent from the cube-id `c` and type `t` of the child, `Pt[c][t]`.
const _PT3 = (
    (0, 1, 2, 3, 4, 5),
    (0, 1, 1, 1, 0, 0),
    (2, 2, 2, 3, 3, 3),
    (1, 1, 2, 2, 2, 1),
    (5, 5, 4, 4, 4, 5),
    (0, 0, 0, 5, 5, 5),
    (4, 3, 3, 3, 4, 4),
    (0, 1, 2, 3, 4, 5),
)
const _PT2 = ((0, 1), (0, 0), (1, 1), (0, 1))

# Table 6: TM local index from the type `t` and cube-id `c` of a simplex, `Iloc[t][c]`.
const _ILOC3 = (
    (0, 1, 1, 4, 1, 4, 4, 7),
    (0, 1, 2, 5, 2, 5, 4, 7),
    (0, 2, 3, 4, 1, 6, 5, 7),
    (0, 3, 1, 5, 2, 4, 6, 7),
    (0, 2, 2, 6, 3, 5, 5, 7),
    (0, 3, 3, 6, 3, 6, 6, 7),
)
const _ILOC2 = ((0, 1, 1, 3), (0, 2, 2, 3))

# Tables 7 and 8: cube-id and type of the child with TM local index `k` of a type-`t` parent.
# Not needed by the tree operations (which go through Tables 1/2), kept for the consistency
# tests of the transcription.
const _CHILD_CUBEID3 = (
    (0, 1, 1, 1, 5, 5, 5, 7),
    (0, 1, 1, 1, 3, 3, 3, 7),
    (0, 2, 2, 2, 3, 3, 3, 7),
    (0, 2, 2, 2, 6, 6, 6, 7),
    (0, 4, 4, 4, 6, 6, 6, 7),
    (0, 4, 4, 4, 5, 5, 5, 7),
)
const _CHILD_CUBEID2 = ((0, 1, 1, 3), (0, 2, 2, 3))
const _CHILD_TYPE3 = (
    (0, 0, 4, 5, 0, 1, 2, 0),
    (1, 1, 2, 3, 0, 1, 5, 1),
    (2, 0, 1, 2, 2, 3, 4, 2),
    (3, 3, 4, 5, 1, 2, 3, 3),
    (4, 2, 3, 4, 0, 4, 5, 4),
    (5, 0, 1, 5, 3, 4, 5, 5),
)
const _CHILD_TYPE2 = ((0, 0, 1, 0), (1, 0, 1, 1))

# The unit simplex S0 of each dimension, i.e. the root scaled by 2^-b.
const _S0_3 = ((0, 0, 0), (1, 0, 0), (1, 0, 1), (1, 1, 1))
const _S0_2 = ((0, 0), (1, 0), (1, 1))

# All vertex pairs = the edges of a simplex, and the vertices of face `f` (those ≠ `f`).
const _EDGE_PAIRS3 = ((1, 2), (1, 3), (1, 4), (2, 3), (2, 4), (3, 4))
const _EDGE_PAIRS2 = ((1, 2), (1, 3), (2, 3))
const _FACE_VERTICES3 = ((2, 3, 4), (1, 3, 4), (1, 2, 4), (1, 2, 3))
const _FACE_VERTICES2 = ((2, 3), (1, 3), (1, 2))

_edge_axes(::Val{3}, t) = _EDGE_AXES3[t + 1]
_edge_axes(::Val{2}, t) = _EDGE_AXES2[t + 1]
_child_type(::Val{3}, t, i) = _CT3[t + 1][i + 1]
_child_type(::Val{2}, t, i) = _CT2[t + 1][i + 1]
_bey_child_of_tm(::Val{3}, t, k) = _SIGMA_INV3[t + 1][k + 1]
_bey_child_of_tm(::Val{2}, t, k) = _SIGMA_INV2[t + 1][k + 1]
_bey_anchor_vertex(::Val{3}, i) = _BEY_ANCHOR_VERTEX3[i + 1]
_bey_anchor_vertex(::Val{2}, i) = _BEY_ANCHOR_VERTEX2[i + 1]
_parent_type(::Val{3}, c, t) = _PT3[c + 1][t + 1]
_parent_type(::Val{2}, c, t) = _PT2[c + 1][t + 1]
_local_index(::Val{3}, t, c) = _ILOC3[t + 1][c + 1]
_local_index(::Val{2}, t, c) = _ILOC2[t + 1][c + 1]
_unit_simplex(::Val{3}) = _S0_3
_unit_simplex(::Val{2}) = _S0_2
_edge_pairs(::Val{3}) = _EDGE_PAIRS3
_edge_pairs(::Val{2}) = _EDGE_PAIRS2
_face_vertices(::Val{3}, f) = _FACE_VERTICES3[f]
_face_vertices(::Val{2}, f) = _FACE_VERTICES2[f]
_ntypes(::Val{dim}) where {dim} = factorial(dim)

##### GEOMETRY IN THE TREE FRAME #####

"""
    vertices(o::SimplexBH{dim}, b) -> NTuple{dim + 1, NTuple{dim, T}}

Integer coordinates of the `dim + 1` corners of `o` in the paper's canonical order
(Algorithm 4.1 of [BH2016](@citet)): `X1 = X0 + h eᵢ`, `X2 = X1 + h eⱼ`, `X3 = X0 + (h, h, h)`
with `h = 2^(b - l)` and the type's edge axes `(i, j)`; in 2D `X2 = X0 + (h, h)`.
"""
function vertices(o::SimplexBH{3, T}, b::Integer) where {T}
    h = T(_compute_size(b, o.l))
    i, j = _edge_axes(Val(3), o.type)
    x0 = o.xyz
    x1 = Base.setindex(x0, x0[i] + h, i)
    x2 = Base.setindex(x1, x1[j] + h, j)
    return (x0, x1, x2, x0 .+ h)
end
function vertices(o::SimplexBH{2, T}, b::Integer) where {T}
    h = T(_compute_size(b, o.l))
    i, = _edge_axes(Val(2), o.type)
    x0 = o.xyz
    return (x0, Base.setindex(x0, x0[i] + h, i), x0 .+ h)
end

vertex(o::SimplexBH, c::Integer, b::Integer) = vertices(o, b)[c]
face(o::SimplexBH{dim}, f::Integer, b::Integer) where {dim} = map(c -> vertex(o, c, b), _face_vertices(Val(dim), f))
faces(o::SimplexBH{dim}, b::Integer) where {dim} = ntuple(f -> face(o, f, b), Val(dim + 1))
edge(o::SimplexBH{dim}, e::Integer, b::Integer) where {dim} = map(c -> vertex(o, c, b), _edge_pairs(Val(dim))[e])
edges(o::SimplexBH{dim}, b::Integer) where {dim} = ntuple(e -> edge(o, e, b), Val(length(_edge_pairs(Val(dim)))))

"""
    _local_barycentric(o::SimplexBH, p, b) -> NTuple{dim + 1, T}

The barycentric coordinates of the integer point `p` with respect to `o`, scaled by `o`'s
edge length `h` so that they stay integers: `p = X0 + Σ μᵢ (Xᵢ - X0) / h`. `p` lies in the
closure of `o` iff all `μᵢ ≥ 0`, and on its face `f` iff `μ_f == 0`. Inverting the canonical
vertex structure of [`vertices`](@ref) gives `μ = (h - qᵢ, qᵢ - qⱼ, qⱼ - qₖ, qₖ)` for
`q = p - X0` and the type's axes `(i, j, k)` (2D: `(h - qᵢ, qᵢ - qⱼ, qⱼ)`).
"""
function _local_barycentric(o::SimplexBH{3, T}, p::NTuple{3, <:Integer}, b::Integer) where {T}
    h = T(_compute_size(b, o.l))
    i, j = _edge_axes(Val(3), o.type)
    k = 6 - i - j
    q = p .- o.xyz
    return (h - q[i], q[i] - q[j], q[j] - q[k], q[k])
end
function _local_barycentric(o::SimplexBH{2, T}, p::NTuple{2, <:Integer}, b::Integer) where {T}
    h = T(_compute_size(b, o.l))
    i, = _edge_axes(Val(2), o.type)
    j = 3 - i
    q = p .- o.xyz
    return (h - q[i], q[i] - q[j], q[j])
end

_contains_point(o::SimplexBH, p, b) = all(μ -> μ >= 0, _local_barycentric(o, p, b))

"""
    _barycentric(p::NTuple{dim, <:Integer}, L) -> NTuple{dim + 1, Int}

Root-frame barycentric coordinates of `p`, scaled by the root extent `L = 2^b`:
`p = Σ μᵢ Xᵢ / L` for the root corners `X = L ⋅ S0`. Integers by construction, `Σ μᵢ = L`;
`p` lies inside the root iff all `μᵢ ≥ 0` and on root face `f` iff `μ_f == 0`.
"""
_barycentric(p::NTuple{3, <:Integer}, L) = (L - p[1], p[1] - p[3], p[3] - p[2], p[2])
_barycentric(p::NTuple{2, <:Integer}, L) = (L - p[1], p[1] - p[2], p[2])

"""
    inside(o::SimplexBH, b) -> Bool

Whether `o` lies within its tree's root simplex. The root is convex, so this holds iff all
corners do (equivalently, iff `o` is a descendant of the root, Property 4 of [BH2016](@citet)).
"""
function inside(o::SimplexBH, b::Integer)
    L = _maximum_size(b)
    return all(v -> all(μ -> μ >= 0, _barycentric(v, L)), vertices(o, b))
end

# `true` iff some corner of `o` lies on the root boundary (a vanishing root barycentric).
function _touches_tree_boundary(o::SimplexBH, b::Integer)
    L = _maximum_size(b)
    return any(v -> any(iszero, _barycentric(v, L)), vertices(o, b))
end

"""
    _orientation(o::SimplexBH) -> Int

Sign of the volume of `o` in its tree frame with the canonical vertex order: `(-1)^(type + 1)`
in 3D (the edge axes `(i, j, k)` are an odd permutation for even types) and `(-1)^type` in 2D.
"""
_orientation(o::SimplexBH{3}) = isodd(o.type) ? 1 : -1
_orientation(o::SimplexBH{2}) = iseven(o.type) ? 1 : -1

##### TREE OPERATIONS (Section 4 of [BH2016]) #####

"""
    cube_id(o::SimplexBH, i, b) -> Int

Algorithm 4.2: the cube-id (z-order child slot, `0:2^dim-1`) of `o`'s level-`i` ancestor
within its parent's cube — bit `b - i` of the anchor coordinates.
"""
function cube_id(o::SimplexBH{dim, T}, i::Integer, b::Integer) where {dim, T}
    h = T(_compute_size(b, i))
    c = 0
    for d in 1:dim
        (o.xyz[d] & h) != zero(T) && (c |= 1 << (d - 1))
    end
    return c
end

"""
    parent(o::SimplexBH, b) -> SimplexBH

Algorithm 4.3: clear the level bit of the anchor and look the parent's type up from `o`'s
cube-id and type (Figure 8). The root returns itself.
"""
function parent(o::SimplexBH{dim, T}, b::Integer) where {dim, T}
    o.l > zero(T) || return o
    h = T(_compute_size(b, o.l))
    return SimplexBH{dim, T}(o.l - one(T), o.xyz .& ~h, T(_parent_type(Val(dim), cube_id(o, o.l, b), o.type)))
end

"""
    _bey_child(o::SimplexBH, i, b) -> SimplexBH

Algorithm 4.4: child `i ∈ 0:2^dim-1` of `o` in Bey's numbering (eq. 2). Its anchor is the
midpoint of `X0` and the vertex given by `_bey_anchor_vertex`, its type from Table 1.
"""
function _bey_child(o::SimplexBH{dim, T}, i::Integer, b::Integer) where {dim, T}
    X = vertices(o, b)
    j = _bey_anchor_vertex(Val(dim), i)
    anchor = (X[1] .+ X[j + 1]) .÷ T(2)
    return SimplexBH{dim, T}(o.l + one(T), anchor, T(_child_type(Val(dim), o.type, i)))
end

"""
    children(o::SimplexBH{dim}, b) -> NTuple{2^dim, SimplexBH}

The `2^dim` Bey children of `o` in tetrahedral-Morton order (Algorithm 4.5 with Table 2), so
splicing them into a sorted leaf list in place of `o` keeps the list sorted.
"""
function children(o::SimplexBH{dim}, b::Integer) where {dim}
    return ntuple(k -> _bey_child(o, _bey_child_of_tm(Val(dim), o.type, k - 1), b), Val(2^dim))
end

"""
    child_id(o::SimplexBH, b) -> Int

The 1-based position of `o` among its siblings in TM order (Table 6 of [BH2016](@citet)).
"""
child_id(o::SimplexBH{dim}, b::Integer) where {dim} = _local_index(Val(dim), o.type, cube_id(o, o.l, b)) + 1

"""
    consecutive_index(o::SimplexBH{dim}, b) -> Int

The consecutive index of eq. (55) of [BH2016](@citet): the `2^dim`-ary number whose digits
are the TM local indices of `o`'s ancestors, most significant first. Computed by walking up
from `o` (the cube-id at level `i` is a bit of the anchor, the type via Figure 8), i.e. in
`O(l)`. Orders all simplices of one level like their TM-index (eq. 53).
"""
function consecutive_index(o::SimplexBH{dim}, b::Integer) where {dim}
    I = 0
    t = Int(o.type)
    l = Int(o.l)
    for i in l:-1:1
        c = cube_id(o, i, b)
        I += _local_index(Val(dim), t, c) << (dim * (l - i))
        t = _parent_type(Val(dim), c, t)
    end
    return I
end

"""
    _sortkey(o, b)

The `(space-filling-curve position, level)` key that orders a tree's leaves: ancestors before
descendants, and every simplex's descendants contiguous. For a `SimplexBH` the position is
the consecutive index of its first maximum-level descendant, `I(o) << dim (b - l)`; for an
`OctantBWG` the level-independent Morton interleave of the anchor. Unlike the octant order,
the simplex order needs the tree's maximum level `b` (the ancestor chain is read off the
anchor bits), which is why the shared tree code compares keys instead of relying on
`Base.isless`.
"""
_sortkey(o::SimplexBH{dim}, b::Integer) where {dim} = (consecutive_index(o, b) << (dim * (b - o.l)), o.l)

"""
    facet_neighbor_face(o::SimplexBH, f, b) -> (SimplexBH, f̃)

Algorithm 4.6 of [BH2016](@citet): the same-level neighbour of `o` across its face `f` and the
face `f̃` of the neighbour across which `o` is reached back (eq. 49). The neighbour may lie
outside the tree, see [`inside`](@ref). [`facet_neighbor`](@ref) returns the neighbour only.
"""
function facet_neighbor_face(o::SimplexBH{3, T}, f::Integer, b::Integer) where {T}
    t = Int(o.type)
    fp = f - 1
    xyz = o.xyz
    if fp == 1 || fp == 2
        ft = fp
        t′ = ((iseven(t) && fp == 2) || (isodd(t) && fp == 1)) ? t + 1 : t - 1
    elseif fp == 0
        ft = 3
        h = T(_compute_size(b, o.l))
        i = t ÷ 2 + 1
        xyz = Base.setindex(xyz, xyz[i] + h, i)
        t′ = t + (isodd(t) ? 2 : 4)
    else
        ft = 0
        h = T(_compute_size(b, o.l))
        i = ((t + 3) % 6) ÷ 2 + 1
        xyz = Base.setindex(xyz, xyz[i] - h, i)
        t′ = t + (iseven(t) ? 2 : 4)
    end
    return SimplexBH{3, T}(o.l, xyz, T(mod(t′, 6))), ft + 1
end

# Table 3: the 2D neighbour has the other type and the mirrored face; only the anchor of the
# neighbours across faces 0 and 2 moves (by `h` along the type's edge axis, resp. the other one).
function facet_neighbor_face(o::SimplexBH{2, T}, f::Integer, b::Integer) where {T}
    t = Int(o.type)
    fp = f - 1
    xyz = o.xyz
    if fp == 0
        h = T(_compute_size(b, o.l))
        i = t + 1
        xyz = Base.setindex(xyz, xyz[i] + h, i)
    elseif fp == 2
        h = T(_compute_size(b, o.l))
        i = 2 - t
        xyz = Base.setindex(xyz, xyz[i] - h, i)
    end
    return SimplexBH{2, T}(o.l, xyz, T(1 - t)), 3 - fp
end

facet_neighbor(o::SimplexBH, f::Integer, b::Integer) = facet_neighbor_face(o, f, b)[1]

# Whether the integer point `v` lies on the lattice of level-`l` anchors (all coordinates
# multiples of the edge length `h = 2^(b - l)`, a power of two).
_on_lattice(v::NTuple{dim, T}, h::T) where {dim, T} = all(x -> x & (h - one(T)) == zero(T), v)

"""
    _push_simplices_at!(P, v, l, b)

Append every level-`l` simplex of the tree whose closure contains the integer point `v`. The
candidates are the `dim!` simplices of the level-`l` cubes whose closure contains `v`: along an
axis on which `v` sits on the level-`l` lattice the cube may start at `v` or one edge length
below it, otherwise only the cube containing `v` in its interior qualifies (so a corner of a
finer simplex, e.g. an edge midpoint, is handled as well as a lattice point). The closure test
`_contains_point` (all local barycentric coordinates nonnegative) keeps the touching ones and
[`inside`](@ref) drops those outside the root. This corner query is the building block of every
vertex-based neighbourhood operation on simplex trees (in-tree balancing, inter-tree balancing
and its check).
"""
function _push_simplices_at!(P, v::NTuple{dim, T}, l::Integer, b::Integer) where {dim, T}
    h = T(_compute_size(b, l))
    lo = ntuple(d -> v[d] & ~(h - one(T)), Val(dim)) # the level-`l` lattice point at or below `v`
    for δ in 0:(2^dim - 1)
        # the cube one edge length below along axis `d` touches `v` only if `v` is on the lattice there
        any(d -> (δ >> (d - 1)) & 1 == 1 && lo[d] != v[d], 1:dim) && continue
        anchor = ntuple(d -> (δ >> (d - 1)) & 1 == 1 ? lo[d] - h : lo[d], Val(dim))
        for t in 0:(_ntypes(Val(dim)) - 1)
            cand = SimplexBH{dim, T}(T(l), anchor, T(t))
            _contains_point(cand, v, b) && inside(cand, b) && push!(P, cand)
        end
    end
    return P
end

"""
    _push_same_level_neighbors!(P, o::SimplexBH, b)

Append every same-level simplex of the tree whose closure meets that of `o` (other than `o`
itself). Same-level Kuhn simplices touch iff they share a corner, so the union of the corner
queries [`_push_simplices_at!`](@ref) over `o`'s corners is exactly this set. Duplicates are
appended; callers deduplicate. The simplex counterpart of `possibleneighbors`.
"""
function _push_same_level_neighbors!(P, o::SimplexBH, b::Integer)
    n0 = length(P)
    for v in vertices(o, b)
        _push_simplices_at!(P, v, o.l, b)
    end
    j = n0
    for i in (n0 + 1):length(P)
        P[i] == o && continue
        j += 1
        P[j] = P[i]
    end
    return resize!(P, j)
end

##### TREES #####

"""
    SimplexTreeBH{dim, N, T <: Integer}

A Burstedde–Holke tree: the TM-sorted leaf simplices of one macro triangle/tetrahedron, the
maximum refinement level `b` and the `N = dim + 1` global node ids of the macro cell, in
Ferrite's vertex order — which coincides with the paper's `x_0, …, x_dim` of the root.
"""
struct SimplexTreeBH{dim, N, T <: Integer} <: AbstractTree{Ferrite.RefSimplex{dim}}
    leaves::Vector{SimplexBH{dim, T}}
    b::T
    nodes::NTuple{N, Int}
end

function SimplexTreeBH{dim}(nodes::NTuple{N, Int}, b = DEFAULT_MAXLEVEL[dim]) where {dim, N}
    N == dim + 1 || throw(ArgumentError("a $(dim)D simplex tree needs $(dim + 1) macro nodes, got $N"))
    return SimplexTreeBH{dim, N, Int64}([_simplex_root(Val(dim), Int64)], Int64(_check_maxlevel(dim, b)), nodes)
end
SimplexTreeBH(cell::Triangle, b = DEFAULT_MAXLEVEL[2]) = SimplexTreeBH{2}(cell.nodes, b)
SimplexTreeBH(cell::Tetrahedron, b = DEFAULT_MAXLEVEL[3]) = SimplexTreeBH{3}(cell.nodes, b)

_nchildren(::SimplexTreeBH{dim}) where {dim} = 2^dim
_leaftype(::Type{SimplexTreeBH{dim, N, T}}) where {dim, N, T} = SimplexBH{dim, T}
Base.length(tree::SimplexTreeBH) = length(tree.leaves)
Base.eltype(::Type{SimplexTreeBH{dim, N, T}}) where {dim, N, T} = T

"""
    coarsen_octant!(tree::SimplexTreeBH, o::SimplexBH)

Replace the sibling family of `o` by its parent, the inverse of
[`refine_octant!`](@ref Ferrite.AMR.refine_octant!). The family starts at the parent's first
TM child and occupies `2^dim` consecutive slots; the whole family must be present at the same
level (e.g. after [`balanceforest!`](@ref Ferrite.AMR.balanceforest!)).
"""
function coarsen_octant!(tree::SimplexTreeBH{dim}, o::SimplexBH) where {dim}
    o.l > 0 || throw(ArgumentError("cannot coarsen the root"))
    p = parent(o, tree.b)
    nchild = 2^dim
    idx, found = _leaf_position(tree.leaves, children(p, tree.b)[1], tree.b)
    (found && idx + nchild - 1 <= length(tree.leaves) && _is_complete_family(tree.leaves, idx, tree.leaves[idx], tree.b, nchild)) ||
        throw(ArgumentError("cannot coarsen $o: its sibling family is not completely present in the tree"))
    tree.leaves[idx] = p
    return deleteat!(tree.leaves, (idx + 1):(idx + nchild - 1))
end

"""
    _resolve_corner!(tree::SimplexTreeBH, v, l, sbuf, refine::Bool; keys = nothing) -> Bool

The 2:1 balance condition at a point `v` in the closure of a level-`(l + 1)` leaf: no level-`l`
simplex containing `v` may lie strictly inside a leaf (such a leaf would be at least two
levels coarser and touch the pivot). Returns whether the condition is violated; with
`refine = true` every violating leaf is refined by one level (and all violations are visited),
otherwise the first violation returns. A simplex `s` lies strictly inside a leaf iff it is not a
leaf and the leaf preceding its curve position is an ancestor (a leaf's descendants follow it
contiguously). For check-only calls the leaves' sort keys can be passed precomputed as `keys`
(the tree must not change meanwhile), which turns each probe into a plain binary search.
"""
function _resolve_corner!(tree::SimplexTreeBH, v, l::Integer, sbuf::Vector, refine::Bool; keys = nothing)
    b = tree.b
    leaves = tree.leaves
    empty!(sbuf)
    _push_simplices_at!(sbuf, v, l, b)
    violated = false
    for s in sbuf
        if keys === nothing
            idx, found = _leaf_position(leaves, s, b)
        else
            ks = _sortkey(s, b)
            idx = searchsortedfirst(keys, ks)
            found = idx <= length(keys) && keys[idx] == ks
        end
        found && continue
        (idx > 1 && isancestor(leaves[idx - 1], s, b)) || continue # otherwise `s` is refined
        violated = true
        refine || return true
        refine_octant!(tree, leaves[idx - 1])
    end
    return violated
end

##### MATERIALIZATION HELPERS: Ferrite vertex / face numbering of a simplex leaf #####

# Ferrite's `Tetrahedron` faces `(1,3,2), (1,2,4), (2,3,4), (1,4,3)` are opposite vertices
# `4, 3, 1, 2`; the `Triangle` edges `(1,2), (2,3), (3,1)` opposite vertices `3, 1, 2`.
_ferrite_opposite_vertex(::Val{3}) = (4, 3, 1, 2)
_ferrite_opposite_vertex(::Val{2}) = (3, 1, 2)

# A leaf's canonical vertex order has orientation `_orientation(o)` in the tree frame; with the
# tree map's sign it may be negative in physical space, where Ferrite requires `det J > 0`.
# Cells are then built with vertices 2 and 3 swapped (`_node_perm(false)`): Ferrite vertex `m`
# is canonical vertex `perm[m]`.
_node_perm(::Val{3}, positive::Bool) = positive ? (1, 2, 3, 4) : (1, 3, 2, 4)
_node_perm(::Val{2}, positive::Bool) = positive ? (1, 2, 3) : (1, 3, 2)

# Ferrite local face index of canonical (paper) face `f` of a leaf built with `_node_perm`:
# Ferrite face `j` is opposite Ferrite vertex `opp[j]` = canonical vertex `perm[opp[j]]`.
function _ferrite_face(::Val{dim}, positive::Bool, f::Integer) where {dim}
    perm = _node_perm(Val(dim), positive)
    opp = _ferrite_opposite_vertex(Val(dim))
    return findfirst(j -> perm[opp[j]] == f, 1:(dim + 1))::Int
end

# The root cell keeps Ferrite's node order, so its Ferrite face `j` is the canonical root face
# opposite vertex `opp[j]`.
_paper_root_face(::Val{dim}, ferrite_face::Integer) where {dim} = _ferrite_opposite_vertex(Val(dim))[ferrite_face]

"""
    _root_face_of(o::SimplexBH, f, b) -> Int

The root face on which face `f` of `o` lies (the root barycentric coordinate vanishing on all
its corners), or `0` for an interior face. Leaves of several types have faces on the root
boundary — away from the root's diagonal planes the faces of types 1, 2, 4 and 5 tile them
too — and the leaf's face index is unrelated to the root face index, hence this query.
"""
function _root_face_of(o::SimplexBH{dim}, f::Integer, b::Integer) where {dim}
    L = _maximum_size(b)
    fv = face(o, f, b)
    for i in 1:(dim + 1)
        all(v -> _barycentric(v, L)[i] == 0, fv) && return i
    end
    return 0
end

# The face of `o` with corners `fv` (as a set), or `0`: the corner index whose local
# barycentric coordinate vanishes on all of them.
function _face_index(o::SimplexBH{dim}, fv, b::Integer) where {dim}
    for m in 1:(dim + 1)
        all(v -> _local_barycentric(o, v, b)[m] == 0, fv) && return m
    end
    return 0
end

"""
    _leaf_on_root_face(fv, l, b) -> SimplexBH

The level-`l` simplex inside the root having the (transformed) face corners `fv` as a face —
unique, since `fv` lies on the root boundary. Found among the `dim!` simplices of the `2^dim`
level-`l` cubes whose anchor is the componentwise minimum of the corners minus a multiple of
the edge length.
"""
function _leaf_on_root_face(fv::NTuple{N, NTuple{dim, T}}, l::Integer, b::Integer) where {N, dim, T}
    h = T(_compute_size(b, l))
    lo = ntuple(d -> minimum(v -> v[d], fv), Val(dim))
    for δ in 0:(2^dim - 1)
        anchor = ntuple(d -> (δ >> (d - 1)) & 1 == 1 ? lo[d] - h : lo[d], Val(dim))
        for t in 0:(_ntypes(Val(dim)) - 1)
            cand = SimplexBH{dim, T}(T(l), anchor, T(t))
            (all(v -> _contains_point(cand, v, b), fv) && _face_index(cand, fv, b) != 0 && inside(cand, b)) && return cand
        end
    end
    throw(ArgumentError("no simplex of level $l inside the root has the face $fv"))
end
