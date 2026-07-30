# Octants and octrees: the integer/topological core of the p4est-style AMR implementation
# ([BWG2011](@cite)). An `OctantBWG` is a box in a tree's integer coordinate system, an
# `OctreeBWG` the Morton-sorted leaf list of one macro element. Everything here is local to a
# single tree — inter-tree connectivity, refinement/coarsening of a whole forest, balancing and
# grid materialization live in `forest.jl`.
#
# The lookup tables at the bottom encode the p4est reference numbering (Section 2.1 of the
# paper); the `*_perm`/`*_inv` ones translate between the p4est and Ferrite orderings.

const ncorners_face3D = 4
const ncorners_face2D = 2
const ncorners_edge = ncorners_face2D

# `DEFAULT_MAXLEVEL[dim]`: default *and* largest admissible maximum refinement level for
# `dim`. The 2D/3D values are p4est's `P4EST_MAXLEVEL`/`P8EST_MAXLEVEL`, which bound the
# octant coordinates to 32 bits. Pass `b` explicitly to `ForestBWG(grid, b)` to use a
# smaller limit; larger ones are rejected (see `_check_maxlevel`).
const DEFAULT_MAXLEVEL = (0, 30, 19)

@noinline function _throw_maxlevel(dim, b)
    msg = "maximum refinement level b = $b is out of range for a $(dim)D forest, it must satisfy 0 ≤ b ≤ $(DEFAULT_MAXLEVEL[dim]) (p4est's $(dim == 2 ? "P4EST_MAXLEVEL" : "P8EST_MAXLEVEL")). Beyond it an octree coordinate no longer fits the UInt64 keys of the cross-tree boundary node tables, whose collisions would silently merge unrelated nodes."
    return throw(DomainError(b, msg))
end

# Validate a tree's maximum refinement level: `b` must lie in `0:DEFAULT_MAXLEVEL[dim]`, since a
# larger `b` pushes octree coordinates past the per-axis bit budget of `_packcoord`, whose keys
# would then collide and silently merge unrelated nodes across tree boundaries. Called from the
# `OctreeBWG` constructors, i.e. on every path into `ForestBWG`.
@inline function _check_maxlevel(dim::Integer, b::Integer)
    (0 <= b <= DEFAULT_MAXLEVEL[dim]) || _throw_maxlevel(dim, b)
    return b
end

struct OctantBWG{dim, N, T <: Integer} <: Ferrite.AbstractCell{Ferrite.RefHypercube{dim}}
    #Refinement level
    l::T
    #x,y,z \in {0,...,2^b} where (0 ≤ l ≤ b)
    xyz::NTuple{dim, T}
end

"""
    OctantBWG(dim::Integer, l::Integer, m::Integer, b::Integer)
Construct an `octant` based on dimension `dim`, level `l`, morton index `m` and maximum refinement level `b`
"""
function OctantBWG(dim::Integer, l::T1, m::T2, b::T1 = DEFAULT_MAXLEVEL[dim]) where {T1 <: Integer, T2 <: Integer}
    @assert l ≤ b #maximum refinement level exceeded
    @assert m ≤ (one(T1) + one(T1))^(dim * l)
    x, y, z = (zero(T1), zero(T1), zero(T1))
    # `Int32` is fixed here rather than following from the octant's coordinate type; see #1406.
    h = Int32(_compute_size(b, l))
    _zero = zero(T1)
    _one = one(T1)
    _two = _one + _one
    for i in _zero:(l - _one)
        x = x | (h * ((m - _one) & _two^(dim * i)) ÷ _two^((dim - _one) * i))
        y = y | (h * ((m - _one) & _two^(dim * i + _one)) ÷ _two^((dim - _one) * i + _one))
        z = z | (h * ((m - _one) & _two^(dim * i + _two)) ÷ _two^((dim - _one) * i + _two))
    end
    return if dim < 3
        OctantBWG{2, 4, T1}(l, (x, y))
    else
        OctantBWG{3, 8, T1}(l, (x, y, z))
    end
end

function OctantBWG(level::T, coords::NTuple) where {T <: Integer}
    dim = length(coords)
    nnodes = 2^dim
    return OctantBWG{dim, nnodes, eltype(coords)}(level, coords)
end

"""
From [BWG2011](@citet);
> The octant coordinates are stored as integers of a fixed number b of bits,
> where the highest (leftmost) bit represents the first vertical level of the
> octree (counting the root as level zero), the second highest bit the second level of the octree, and so on.
A morton index can thus be constructed by interleaving the integer bits (2D):
\$m(\\text{Oct}) := (y_b,x_b,y_{b-1},x_{b-1},...y_0,x_0)_2\$
further we assume the following
> Due to the two-complement representation of integers in practically all current hardware,
> where the highest digit denotes the negated appropriate power of two, bitwise operations as used,
> for example, in Algorithm 1 yield the correct result even for negative coordinates.
also from [BWG2011](@citet)
"""
function morton(octant::OctantBWG{dim, N, T}, l::T, b::T) where {dim, N, T <: Integer}
    o = one(T)
    z = zero(T)
    id = zero(widen(eltype(octant.xyz)))
    loop_length = (sizeof(typeof(id)) * T(8)) ÷ dim - o
    for i in z:loop_length
        for d in z:(dim - o)
            # first shift extract i-th bit and second shift inserts it at interleaved index
            id = id | ((octant.xyz[d + o] & (o << i)) << ((dim - o) * i + d))
        end
    end
    # discard the bit information about deeper levels
    return (id >> ((b - l) * dim)) + o
end
morton(octant::OctantBWG{dim, N, T1}, l::T2, b::T3) where {dim, N, T1 <: Integer, T2 <: Integer, T3 <: Integer} = morton(octant, T1(l), T1(b))

Base.zero(::Type{OctantBWG{3, 8}}) = OctantBWG(3, 0, 1)
Base.zero(::Type{OctantBWG{2, 4}}) = OctantBWG(2, 0, 1)
root(dim::T) where {T <: Integer} = zero(OctantBWG{dim, 2^dim})

nchilds(::Type{OctantBWG{dim, N, T}}) where {dim, N, T} = N
nchilds(o::OctantBWG) = nchilds(typeof(o))

Base.isequal(o1::OctantBWG, o2::OctantBWG) = (o1.l == o2.l) && (o1.xyz == o2.xyz)
Base.hash(o::OctantBWG, h::UInt) = hash(o.xyz, hash(o.l, h))
"""
    o1::OctantBWG < o2::OctantBWG
Implements Algorithm 2.1 of [IBWG2015](@citet).
Checks first if mortonid is smaller and later if level is smaller.
Thus, ancestors precede descendants (preordering).
"""
function Base.isless(o1::OctantBWG, o2::OctantBWG)
    if o1.xyz != o2.xyz
        # morton(o, o.l, o.l) shifts by (b - l)*dim = 0, so it returns the full,
        # level-independent anchor interleave: this compares anchors in Z-order.
        return morton(o1, o1.l, o1.l) < morton(o2, o2.l, o2.l)
    else
        return o1.l < o2.l
    end
end

"""
    children(octant::OctantBWG{dim, N, T}, b::Integer) -> NTuple{N, OctantBWG}
Compute the `N = 2^dim` children of `octant`, returned in z-order (x before y before z).

The first child's vertices of `octant` are utilized. Its `2^dim` vertices coincide
exactly with the anchors (lower-left corners) of all children, so each child is simply the
level-`l+1` octant placed at the corresponding vertex of `family`.

In 2D, with parent anchor `⊙` at `(x,y)` and edge length `H = 2h`:

    (x,y+H) +───────┬───────+ (x+H,y+H)
            │  c3   │  c4   │            family = first child (c1), anchored at (x,y),
    (x,y+h) ├───────┼───────┤            edge length h. Its four vertices
            │  c1   │  c2   │              v1=(x,  y  )  v2=(x+h, y  )
      (x,y) ⊙───────┴───────+ (x+H,y)      v3=(x,  y+h)  v4=(x+h, y+h)
               (x+h,y)                 are exactly the anchors of c1..c4.
"""
function children(octant::OctantBWG{dim, N, T}, b::Integer) where {dim, N, T}
    family = OctantBWG{dim, N, T}(octant.l + one(T), octant.xyz)
    return ntuple(i -> OctantBWG{dim, N, T}(family.l, vertex(family, i, b)), nchilds(octant))
end

"""
    vertex(octant::OctantBWG{dim}, c::Integer, b::Integer) -> NTuple{dim, Int}

Octree (integer) coordinate of the `c`-th corner of `octant`, `c ∈ 1:2^dim` in z-order. Along
axis `d`, bit `d-1` of `c-1` selects the low corner `octant.xyz[d]` (bit 0) or the high corner
`octant.xyz[d] + h` (bit 1), where `h = _compute_size(b, octant.l)` is the octant's edge length.
"""
function vertex(octant::OctantBWG{dim, N, T}, c::Integer, b::Integer) where {dim, N, T}
    h = T(_compute_size(b, octant.l))
    return ntuple(d -> ((c - 1) & (2^(d - 1))) == 0 ? octant.xyz[d] : octant.xyz[d] + h, dim)
end

"""
    vertices(octant::OctantBWG{dim}, b::Integer)

Computes all vertices of a given `octant`. Each vertex is encoded within the octree coordinates i.e. by integers.
"""
function vertices(octant::OctantBWG{dim}, b::Integer) where {dim}
    _nvertices = 2^dim
    return ntuple(i -> vertex(octant, i, b), _nvertices)
end

"""
    face(octant::OctantBWG{dim}, f::Integer, b::Integer) -> NTuple{2^(dim - 1), NTuple{dim, Int}}

Octree (integer) coordinates of the corners of local face `f` of `octant` (`f ∈ 1:2dim`, p4est
face order). Two corners in 2D, four in 3D, obtained from the face→corner lookup `𝒱₂`/`𝒱₃` and
[`vertex`](@ref). The plural [`faces`](@ref) builds the full set.
"""
function face(octant::OctantBWG{2}, f::Integer, b::Integer)
    return ntuple(i -> vertex(octant, 𝒱₂[f, i], b), 2)
end

function face(octant::OctantBWG{3}, f::Integer, b::Integer)
    return ntuple(i -> vertex(octant, 𝒱₃[f, i], b), 4)
end

"""
    faces(octant::OctantBWG{dim}, b::Integer)

Computes all faces of a given `octant`. Each face is encoded within the octree coordinates i.e. by integers.
Further, each face consists of either two two-dimensional integer coordinates or four three-dimensional integer coordinates.
"""
function faces(octant::OctantBWG{dim}, b::Integer) where {dim}
    _nfaces = 2 * dim
    return ntuple(i -> face(octant, i, b), _nfaces)
end

"""
    edge(octant::OctantBWG{3}, e::Integer, b::Integer) -> NTuple{2, NTuple{3, Int}}

Octree (integer) coordinates of the two endpoints of local edge `e` of a 3D `octant`
(`e ∈ 1:12`), via the edge→corner lookup `𝒰` and [`vertex`](@ref).
"""
function edge(octant::OctantBWG{3}, e::Integer, b::Integer)
    return ntuple(i -> vertex(octant, 𝒰[e, i], b), 2)
end

"""
    edges(octant::OctantBWG{dim}, b::Integer)

Computes all edges of a given `octant`. Each edge is encoded within the octree coordinates i.e. by integers.
Further, each edge consists of two three-dimensional integer coordinates.
"""
edges(octant::OctantBWG{3}, b::Integer) = ntuple(i -> edge(octant, i, b), Val(12))

struct OctreeBWG{dim, N, T <: Integer} <: Ferrite.AbstractCell{Ferrite.RefHypercube{dim}}
    leaves::Vector{OctantBWG{dim, N, T}}
    #maximum refinement level
    b::T
    nodes::NTuple{N, Int}
end

# Number of children an octant of this tree splits into (== 2^dim == N, the corner count),
# straight from the type so it works regardless of the leaves' levels.
_nchildren(::OctreeBWG{dim, N}) where {dim, N} = N

"""
    refine_octant!(octree::OctreeBWG, pivot_octant::OctantBWG)

Internal, octree-level refinement primitive; the user-facing entry point is
[`refine!`](@ref)`(forest, cellids)`.

Replace the leaf `pivot_octant` in `octree.leaves` by its `2^dim`
[`children`](@ref Ferrite.AMR.children), spliced in z-order into the parent's slot so the
Morton order of `leaves` is preserved. A no-op if `pivot_octant` is already at the tree's
maximum level `octree.b`.

`pivot_octant` is located with a `searchsortedfirst` binary search, which requires
`octree.leaves` to be Morton-sorted (the `Base.isless` total order) — the standard
invariant for a BWG octree. A single call is `O(n)` because of the in-place `insert!`;
to refine many leaves at once prefer [`refine_all!`](@ref Ferrite.AMR.refine_all!) or the
`refine!(forest, cellids)` vector method, which rebuild the leaf list in one linear pass
instead of `n` shifts.
"""
function refine_octant!(octree::OctreeBWG{dim, N, T}, pivot_octant::OctantBWG{dim, N, T}) where {dim, N, T <: Integer}
    if !(pivot_octant.l + 1 <= octree.b)
        return
    end
    o = one(T)
    # leaves are Morton-sorted (the `Base.isless` total order), so locate the
    # pivot with a binary search instead of an O(N) linear scan.
    leave_idx = searchsortedfirst(octree.leaves, pivot_octant)
    old_octant = popat!(octree.leaves, leave_idx)
    _children = children(pivot_octant, octree.b)
    for child in _children
        insert!(octree.leaves, leave_idx, child)
        leave_idx += 1
    end
    return
end

"""
    coarsen_octant!(octree::OctreeBWG, o::OctantBWG)

Internal, octree-level coarsening primitive; the user-facing entry point is
[`coarsen!`](@ref)`(forest, cellids)`.

Replace the `2^dim`-sibling family that `o` belongs to with their common `parent` in the
tree's Morton-sorted `leaves`. `o` is snapped back to the family's first sibling (via its
`child_id`/morton), the parent is written in its slot, and the remaining `2^dim - 1`
siblings are deleted — the inverse of [`refine_octant!`](@ref Ferrite.AMR.refine_octant!).
Assumes the whole family is present and at the same level (e.g. after
[`balanceforest!`](@ref Ferrite.AMR.balanceforest!)).
"""
function coarsen_octant!(octree::OctreeBWG{dim, N, T}, o::OctantBWG{dim, N, T}) where {dim, N, T <: Integer}
    _two = T(2)
    leave_idx = findfirst(x -> x == o, octree.leaves)
    shift = child_id(o, octree.b) - one(T)
    coarsen_target = o
    if shift != zero(T)
        old_morton = morton(o, o.l, octree.b)
        coarsen_target = OctantBWG(dim, o.l, old_morton, octree.b)
    end
    window_start = leave_idx - shift
    window_length = _two^dim - one(T)
    new_octant = parent(coarsen_target, octree.b)
    octree.leaves[leave_idx - shift] = new_octant
    return deleteat!(octree.leaves, (leave_idx - shift + one(T)):(leave_idx - shift + window_length))
end

OctreeBWG{3, 8}(nodes::NTuple, b = DEFAULT_MAXLEVEL[3]) = OctreeBWG{3, 8, Int64}([zero(OctantBWG{3, 8})], Int64(_check_maxlevel(3, b)), nodes)
OctreeBWG{2, 4}(nodes::NTuple, b = DEFAULT_MAXLEVEL[2]) = OctreeBWG{2, 4, Int64}([zero(OctantBWG{2, 4})], Int64(_check_maxlevel(2, b)), nodes)
OctreeBWG(cell::Quadrilateral, b = DEFAULT_MAXLEVEL[2]) = OctreeBWG{2, 4}(cell.nodes, b)
OctreeBWG(cell::Hexahedron, b = DEFAULT_MAXLEVEL[3]) = OctreeBWG{3, 8}(cell.nodes, b)

Base.length(tree::OctreeBWG) = length(tree.leaves)
Base.eltype(::Type{OctreeBWG{dim, N, T}}) where {dim, N, T} = T

"""
    inside(oct::OctantBWG{dim}, b) -> Bool

Whether `oct` lies within its tree's root domain `[0, 2^b)^dim` (see [`_maximum_size`](@ref)).
A `false` means the octant has crossed a tree boundary — the signal that an inter-tree transform
([`transform_facet`](@ref)/[`transform_corner`](@ref)) is needed to express it in the
neighbouring tree's coordinate system.
"""
function inside(oct::OctantBWG{dim}, b) where {dim}
    maxsize = _maximum_size(b)
    outside = any(xyz -> xyz >= maxsize, oct.xyz) || any(xyz -> xyz < 0, oct.xyz)
    return !outside
end

# [`ancestor_id`](@ref) without the level assertion, for the traversal hot paths where
# `l ≤ leaf.l` holds by construction (the range's leaves are strict descendants of the
# split octant), and without the `T(2)^j` power in the loop.
@inline function _ancestor_id_fast(octant::OctantBWG{dim, N, T}, l::Integer, b::Integer) where {dim, N, T}
    h = T(_compute_size(b, l))
    z = zero(T)
    i = 1
    for d in 1:dim
        i += Int((octant.xyz[d] & h) != z) << (d - 1)
    end
    return i
end

"""
    split_bounds(leaves, lo, hi, a::OctantBWG, b) -> NTuple{2^dim + 1, Int}

Algorithm 3.3 of [IBWG2015](@citet). Given the contiguous, Morton-sorted leaf
sub-range `leaves[lo:hi]` (all strict descendants of `a`), return boundary indices
`k` such that child `i` of `a` occupies `leaves[k[i]:k[i+1]-1]`. Non-allocating:
returns a stack `NTuple` — no `𝐤` vector and no `SubArray` views, so the recursive
descent that calls it at every internal octant stays allocation-free. Short ranges
(the vast majority: the recursion halves the range per level) are split by a single
linear `ancestor_id` sweep; long ranges by `2^dim - 1` binary searches (the child
key `ancestor_id` is monotone along the range).

Shared descent helper of the point iterator (`_iterate_interior!`,
`_descend_to_corner`) and the interface descent (`_iter_interface!`).
"""
function split_bounds(leaves, lo::Integer, hi::Integer, a::OctantBWG{dim, N, T}, b::Integer) where {dim, N, T}
    l1 = a.l + one(T)
    ilo = Int(lo); ihi = Int(hi)
    if ihi - ilo == N - 1 && leaves[ilo].l == l1 && leaves[ihi].l == l1
        # exactly N leaves one level down -> they are precisely the N children in z-order
        # (the fully-refined-once case, the bulk of all calls): pure arithmetic split
        return ntuple(i -> ilo + i - 1, Val(N + 1))
    end
    if ihi - ilo < 8 * N
        k = ntuple(i -> i == 1 ? ilo : ihi + 1, Val(N + 1))
        prev = 1
        for pos in ilo:ihi
            cid = _ancestor_id_fast(leaves[pos], l1, b)
            while prev < cid
                prev += 1
                k = Base.setindex(k, pos, prev)
            end
        end
        return k
    end
    return ntuple(Val(N + 1)) do i
        i == 1 && return ilo
        i == N + 1 && return ihi + 1
        # first index j ∈ lo:hi with ancestor_id(leaves[j], l1) ≥ i
        alo = ilo; ahi = ihi + 1
        while alo < ahi
            mid = (alo + ahi) >>> 1
            _ancestor_id_fast(leaves[mid], l1, b) < i ? (alo = mid + 1) : (ahi = mid)
        end
        return alo
    end
end

_isleaf(leaves, lo::Integer, hi::Integer, oct::OctantBWG) = lo == hi && leaves[lo] == oct

"""
    rotation_permutation(r, i) -> i′
computes based on the rotation indicator `r` ∈ {0,1} and a given corner index `i` ∈ {1,2} the permuted corner index `i′`
"""
function rotation_permutation(r, i)
    i′ = r == 0 ? i : 3 - i
    return i′
end

"""
    rotation_permutation(f, f′, r, i) -> i′
computes based on the rotation indicator `r` ∈ {0,...,3} and a given corner index `i` ∈ {1,...,4} the permuted corner index `i′`
See Table 3 and Theorem 2.2 [BWG2011](@citet).
"""
function rotation_permutation(f, f′, r, i)
    return 𝒫[𝒬[ℛ[f, f′], r + 1], i]
end

p4est_opposite_face_index(f) = ((f - 1) ⊻ 0b1) + 1
p4est_opposite_edge_index(e) = ((e - 1) ⊻ 0b11) + 1

# Named decodings of the p4est face/edge/corner index encodings (Section 2.1 of [BWG2011]),
# so that the transform kernels below read as geometry instead of bit magic. All returned
# axes/sides are 0-based, matching the coordinate arithmetic they feed.
# Faces come in axis pairs: 1/2 = x∓, 3/4 = y∓, 5/6 = z∓.
@inline _face_axis(f) = (f - one(f)) ÷ 2 # the axis of the face normal
@inline _face_side(f) = (f - one(f)) & one(f) # 0 = lower face on that axis, 1 = upper
# 3D edges come in axis quadruples: 1-4 along x, 5-8 along y, 9-12 along z.
@inline _edge_axis(e) = (e - one(e)) ÷ 4 # the along-edge axis
# An edge's position on its i-th transverse axis (i ∈ (1, 2), ascending axis order):
@inline _edge_side(e, i) = ((e - one(e)) >> (i - 1)) & one(e) # 0 = lower end, 1 = upper
# Corner c sits at the upper end of (1-based) axis i iff bit i-1 of c-1 is set:
@inline _corner_side(c, i) = ((c - one(c)) >> (i - 1)) & one(c) # 0 = lower end, 1 = upper

# Membership test on a Morton-sorted leaf vector. Leaves are kept in `Base.isless` order
# everywhere (balancetree linearises + sorts, refine! splices children in z-order), so this is an
# O(log n) binary search instead of the O(n) linear scan that `∈ Vector` would do — the same idea
# as in `refine!`. Used in the inter-tree balance hot path, once per boundary-leaf neighbour.
#
# The leaves are ordered by the `(morton-anchor, level)` key (== the `Base.isless` order). We
# compute the *target* key once and compare against each probed leaf's key, rather than going
# through `searchsortedfirst`/`isless` which recomputes `morton` for both sides every step
# (`morton` is a ~b·dim-bit interleave — the dominant per-comparison cost on small per-tree arrays).
@inline function _in_leaves(leaves::Vector{<:OctantBWG}, o::OctantBWG)
    okey = (morton(o, o.l, o.l), o.l)
    lo = 1
    hi = length(leaves)
    @inbounds while lo <= hi
        mid = (lo + hi) >>> 1
        lf = leaves[mid]
        lkey = (morton(lf, lf.l, lf.l), lf.l)
        if lkey < okey
            lo = mid + 1
        elseif lkey > okey
            hi = mid - 1
        else
            return true                 # equal key uniquely identifies the octant
        end
    end
    return false
end

"""
    possibleneighbors(o::OctantBWG{2}, l, b)
Returns a tuple of possible neighbors, where the first four are corner neighbors that are exclusively connected via a corner.
The other four possible neighbors are face neighbors. Always returns the full `NTuple` (type-stable);
callers that want only in-tree neighbours filter with `inside`.
"""
function possibleneighbors(o::OctantBWG{2}, l, b)
    return ntuple(Val(8)) do i
        if i > 4
            j = i - 4
            facet_neighbor(o, j, b)
        else
            corner_neighbor(o, i, b)
        end
    end
end

"""
    possibleneighbors(o::OctantBWG{3}, l, b)
Returns a tuple of possible neighbors, where the first eight are corner neighbors that are exclusively connected via a corner.
After the first eight corner neighbors, the 6 possible face neighbors follow and after them, the edge neighbors.
Always returns the full `NTuple` (type-stable); callers that want only in-tree neighbours filter with `inside`.
"""
function possibleneighbors(o::OctantBWG{3}, l, b)
    return ntuple(Val(26)) do i
        if 8 < i ≤ 14
            j = i - 8
            facet_neighbor(o, j, b)
        elseif 14 < i ≤ 26
            j = i - 14
            edge_neighbor(o, j, b)
        else
            corner_neighbor(o, i, b)
        end
    end
end

"""
    isancestor(o1, o2, b) -> Bool
Is `o1` a strict ancestor of `o2`, i.e. is `o2` a descendant of `o1`? Walks `o2`'s parent
chain up to the root, which is itself a valid ancestor (`parent` of the root is the root,
hence the level-0 stop instead of a levels-remaining counter).
"""
function isancestor(o1, o2, b)
    o2.l > o1.l || return false     # an ancestor is strictly coarser
    p = parent(o2, b)
    while p.l > zero(p.l)
        p == o1 && return true
        p = parent(p, b)
    end
    return p == o1                  # the walk ended at the root, itself a valid ancestor
end

"""
    contains_facet(mface, sface) -> Bool

Whether the sub-facet `sface` is geometrically contained in the master facet `mface`, both given
as octree corner coordinates. In 2D the facets are axis-aligned segments and containment is an
interval test along the shared axis; in 3D `mface` is a face and `sface` is accepted iff its
[`center`](@ref) lies inside `mface`'s bounding box. Used to decide whether a leaf face lies on a
coarse/root face (refinement interfaces, boundary-set transfer).
"""
function contains_facet(mface::Tuple{Tuple{T1, T1}, Tuple{T1, T1}}, sface::Tuple{Tuple{T2, T2}, Tuple{T2, T2}}) where {T1 <: Integer, T2 <: Integer}
    if mface[1][1] == sface[1][1] && mface[2][1] == sface[2][1] # vertical
        return mface[1][2] ≤ sface[1][2] ≤ sface[2][2] ≤ mface[2][2]
    elseif mface[1][2] == sface[1][2] && mface[2][2] == sface[2][2] # horizontal
        return mface[1][1] ≤ sface[1][1] ≤ sface[2][1] ≤ mface[2][1]
    else
        return false
    end
end

# The center test is exact here (not just a heuristic): octant faces are dyadic — their size is
# a power of two and their position a multiple of that size — so two axis-aligned dyadic faces
# either nest or have disjoint interiors. `sface`'s center is an interior point of `sface`, hence
# it lies in `mface`'s (closed, degenerate-in-normal-direction) bounding box iff `sface ⊆ mface`.
function contains_facet(mface::NTuple{4, Tuple{T1, T1, T1}}, sface::NTuple{4, Tuple{T2, T2, T2}}) where {T1 <: Integer, T2 <: Integer}
    sface_center = center(sface)
    lower_left = ntuple(i -> minimum(getindex.(mface, i)), 3)
    top_right = ntuple(i -> maximum(getindex.(mface, i)), 3)
    if (lower_left[1] ≤ sface_center[1] ≤ top_right[1]) && (lower_left[2] ≤ sface_center[2] ≤ top_right[2]) && (lower_left[3] ≤ sface_center[3] ≤ top_right[3])
        return true
    else
        return false
    end
end

"""
    face_contains_edge(f, e) -> Bool

Whether edge `e` lies on face `f` (3D), tested by checking `e`'s [`center`](@ref) against `f`'s
bounding box.
"""
function face_contains_edge(f::NTuple{4, Tuple{T1, T1, T1}}, e::Tuple{Tuple{T2, T2, T2}, Tuple{T2, T2, T2}}) where {T1 <: Integer, T2 <: Integer}
    edge_center = center(e)
    lower_left = ntuple(i -> minimum(getindex.(f, i)), 3)
    top_right = ntuple(i -> maximum(getindex.(f, i)), 3)
    if (lower_left[1] ≤ edge_center[1] ≤ top_right[1]) && (lower_left[2] ≤ edge_center[2] ≤ top_right[2]) && (lower_left[3] ≤ edge_center[3] ≤ top_right[3])
        return true
    else
        return false
    end
end

"""
    contains_edge(medge, sedge) -> Bool

3D analogue of [`contains_facet`](@ref) for edges: `sedge ⊆ medge` when the two share their two
constant coordinates and `sedge` lies within `medge` along the varying axis.
"""
function contains_edge(medge::Tuple{Tuple{T1, T1, T1}, Tuple{T1, T1, T1}}, sedge::Tuple{Tuple{T2, T2, T2}, Tuple{T2, T2, T2}}) where {T1 <: Integer, T2 <: Integer}
    if (medge[1][1] == sedge[1][1] && medge[2][1] == sedge[2][1]) && (medge[1][2] == sedge[1][2] && medge[2][2] == sedge[2][2]) # x1 & x2 aligned
        return medge[1][3] ≤ sedge[1][3] ≤ sedge[2][3] ≤ medge[2][3]
    elseif (medge[1][1] == sedge[1][1] && medge[2][1] == sedge[2][1]) && (medge[1][3] == sedge[1][3] && medge[2][3] == sedge[2][3]) # x1 & x3 aligned
        return medge[1][2] ≤ sedge[1][2] ≤ sedge[2][2] ≤ medge[2][2]
    elseif (medge[1][2] == sedge[1][2] && medge[2][2] == sedge[2][2]) && (medge[1][3] == sedge[1][3] && medge[2][3] == sedge[2][3]) # x2 & x3 aligned
        return medge[1][1] ≤ sedge[1][1] ≤ sedge[2][1] ≤ medge[2][1]
    else
        return false
    end
end

"""
    center(pivot_face) -> NTuple{dim, Int}

Integer centroid of a set of octree coordinates (the corners of a face or edge): the elementwise
sum divided by the number of points. For a coarse face bordering a refined neighbour this is
exactly the hanging node — the face center (4 corners) or edge midpoint (2 corners) — in the same
integer/topological frame (no physical coordinates).
"""
function center(pivot_face)
    centerpoint = ntuple(i -> 0, length(pivot_face[1]))
    for c in pivot_face
        centerpoint = c .+ centerpoint
    end
    return centerpoint .÷ length(pivot_face)
end

"""
    child_id(octant::OctantBWG, b::Integer)
Given some OctantBWG `octant` and maximum refinement level `b`, compute the child_id of `octant`
note the following quote from Burstedde et al:
  children are numbered from 0 for the front lower left child,
  to 1 for the front lower right child, to 2 for the back lower left, and so on, with
  4, . . . , 7 being the four children on top of the children 0, . . . , 3.
shifted by 1 due to julia 1 based indexing
"""
function child_id(octant::OctantBWG{dim, N, T}, b::Integer = DEFAULT_MAXLEVEL[dim]) where {dim, N, T <: Integer}
    i = 0x00
    t = T(2)
    z = zero(T)
    h = T(_compute_size(b, octant.l))
    xyz = octant.xyz
    for j in 0:(dim - 1)
        i = i | ((xyz[j + 1] & h) != z ? t^j : z)
    end
    return i + 0x01
end

"""
    ancestor_id(octant::OctantBWG, l::Integer, b::Integer)
Algorithm 3.2 of [IBWG2015](@citet) that generalizes `child_id` for different queried levels.
Applied to a single octree, i.e. the array of leaves, yields a monotonic sequence
"""
function ancestor_id(octant::OctantBWG{dim, N, T}, l::Integer, b::Integer = DEFAULT_MAXLEVEL[dim]) where {dim, N, T <: Integer}
    @assert 0 < l ≤ octant.l
    i = 0x00
    t = T(2)
    z = zero(T)
    h = T(_compute_size(b, l))
    for j in 0:(dim - 1)
        i = i | ((octant.xyz[j + 1] & h) != z ? t^j : z)
    end
    return i + 0x01
end

"""
    parent(octant::OctantBWG{dim}, b::Integer = DEFAULT_MAXLEVEL[dim]) -> OctantBWG

The level-`(l-1)` ancestor of `octant`. Computed by clearing the bit that distinguishes the octant
from its parent: `xyz .& ~h` with `h = _compute_size(b, l)` snaps the anchor back to the parent's
anchor. The root returns itself.
"""
function parent(octant::OctantBWG{dim, N, T}, b::Integer = DEFAULT_MAXLEVEL[dim]) where {dim, N, T}
    if octant.l > zero(T)
        h = T(_compute_size(b, octant.l))
        l = octant.l - one(T)
        return OctantBWG(l, octant.xyz .& ~h)
    else
        root(dim)
    end
end

"""
    descendants(octant::OctantBWG, b::Integer)
Given an `octant`, computes the two smallest possible octants that fit into the first and last corners
of `octant`, respectively. These computed octants are called first and last descendants of `octant`
since they are connected to `octant` by a path down the octree to the maximum level  `b`
"""
function descendants(octant::OctantBWG{dim, N, T}, b::Integer = DEFAULT_MAXLEVEL[dim]) where {dim, N, T}
    l1 = b; l2 = b
    h = T(_compute_size(b, octant.l))
    return OctantBWG(l1, octant.xyz), OctantBWG(l2, octant.xyz .+ (h - one(T)))
end

"""
    facet_neighbor(octant::OctantBWG{dim, N, T}, f::T, b::T = DEFAULT_MAXLEVEL[dim]) -> OctantBWG{dim, N, T}
Intraoctree face neighbor for a given faceindex `f` (in p4est, i.e. z order convention) and specified maximum refinement level `b`.
Implements Algorithm 5 of [BWG2011](@citet).

    x-------x-------x
    |       |       |
    |   3   |   4   |
    |       |       |
    x-------x-------x
    |       |       |
    o   1   *   2   |
    |       |       |
    x-------x-------x

Consider octant 1 at `xyz=(0,0)`, a maximum refinement level of 1 and faceindex 2 (marked as `*`).
Then, the computed face neighbor will be octant 2 with `xyz=(1,0)`.
Note that the function is not sensitive in terms of leaving the octree boundaries.
For the above example, a query for face index 1 (marked as `o`) will return an octant outside of the octree with `xyz=(-1,0)`.
"""
function facet_neighbor(octant::OctantBWG{3, N, T}, f::T, b::T = DEFAULT_MAXLEVEL[3]) where {N, T <: Integer}
    l = octant.l
    h = T(_compute_size(b, octant.l))
    x, y, z = octant.xyz
    x += ((f == T(1)) ? -h : ((f == T(2)) ? h : zero(T)))
    y += ((f == T(3)) ? -h : ((f == T(4)) ? h : zero(T)))
    z += ((f == T(5)) ? -h : ((f == T(6)) ? h : zero(T)))
    return OctantBWG(l, (x, y, z))
end
function facet_neighbor(octant::OctantBWG{2, N, T}, f::T, b::T = DEFAULT_MAXLEVEL[2]) where {N, T <: Integer}
    l = octant.l
    h = T(_compute_size(b, octant.l))
    x, y = octant.xyz
    x += ((f == T(1)) ? -h : ((f == T(2)) ? h : zero(T)))
    y += ((f == T(3)) ? -h : ((f == T(4)) ? h : zero(T)))
    return OctantBWG(l, (x, y))
end
facet_neighbor(o::OctantBWG{dim, N, T1}, f::T2, b::T3) where {dim, N, T1 <: Integer, T2 <: Integer, T3 <: Integer} = facet_neighbor(o, T1(f), T1(b))

reference_faces_bwg(::Type{Ferrite.RefHypercube{2}}) = ((1, 3), (2, 4), (1, 2), (3, 4))
reference_faces_bwg(::Type{Ferrite.RefHypercube{3}}) = ((1, 3, 5, 7), (2, 4, 6, 8), (1, 2, 5, 6), (3, 4, 7, 8), (1, 2, 3, 4), (5, 6, 7, 8)) # p4est consistent ordering

"""
    edge_neighbor(octant::OctantBWG, e::Integer, b::Integer)
Computes the edge neighbor octant which is only connected by the edge `e` to `octant`.
"""
function edge_neighbor(octant::OctantBWG{3, N, T}, e::T, b::T = DEFAULT_MAXLEVEL[3]) where {N, T <: Integer}
    @assert 1 ≤ e ≤ 12
    _one = one(T)
    _two = T(2)
    e -= _one
    l = octant.l
    h = T(_compute_size(b, octant.l))
    ox, oy, oz = octant.xyz
    𝐚 = (
        e ÷ 4,
        e < 4 ? 1 : 0,
        e < 8 ? 2 : 1,
    )
    xyz = zeros(T, 3)
    xyz[𝐚[1] + _one] = octant.xyz[𝐚[1] + _one]
    xyz[𝐚[2] + _one] = octant.xyz[𝐚[2] + _one] + (_two * (e & _one) - _one) * h
    xyz[𝐚[3] + _one] = octant.xyz[𝐚[3] + _one] + ((e & _two) - _one) * h
    return OctantBWG(l, (xyz[1], xyz[2], xyz[3]))
end
edge_neighbor(o::OctantBWG{3, N, T1}, e::T2, b::T3) where {N, T1 <: Integer, T2 <: Integer, T3 <: Integer} = edge_neighbor(o, T1(e), T1(b))

"""
    corner_neighbor(octant::OctantBWG, c::Integer, b::Integer)
Computes the corner neighbor octant which is only connected by the corner `c` to `octant`
"""
function corner_neighbor(octant::OctantBWG{3, N, T}, c::T, b::T = DEFAULT_MAXLEVEL[3]) where {N, T <: Integer}
    c -= one(T)
    l = octant.l
    h = T(_compute_size(b, octant.l))
    ox, oy, oz = octant.xyz
    _one = one(T)
    _two = T(2)
    x = ox + (_two * (c & _one) - _one) * h
    y = oy + ((c & _two) - _one) * h
    z = oz + ((c & T(4)) ÷ _two - _one) * h
    return OctantBWG(l, (x, y, z))
end

function corner_neighbor(octant::OctantBWG{2, N, T}, c::T, b::T = DEFAULT_MAXLEVEL[2]) where {N, T <: Integer}
    c -= one(T)
    l = octant.l
    h = _compute_size(b, octant.l)
    ox, oy = octant.xyz
    _one = one(T)
    _two = T(2)
    x = ox + (_two * (c & _one) - _one) * h
    y = oy + ((c & _two) - _one) * h
    return OctantBWG(l, (x, y))
end
corner_neighbor(o::OctantBWG{dim, N, T1}, c::T2, b::T3) where {dim, N, T1 <: Integer, T2 <: Integer, T3 <: Integer} = corner_neighbor(o, T1(c), T1(b))

function Base.show(io::IO, ::MIME"text/plain", o::OctantBWG{3, N, M}) where {N, M}
    x, y, z = o.xyz
    println(io, "OctantBWG{3,$N,$M}")
    println(io, "   l = $(o.l)")
    return println(io, "   xyz = $x,$y,$z")
end

function Base.show(io::IO, ::MIME"text/plain", o::OctantBWG{2, N, M}) where {N, M}
    x, y = o.xyz
    println(io, "OctantBWG{2,$N,$M}")
    println(io, "   l = $(o.l)")
    return println(io, "   xy = $x,$y")
end

# Cold-path error so the hot `_compute_size` stays tiny and inlinable.
@noinline _throw_octant_level(l, b) = throw(DomainError(l, "octant level $l exceeds the tree's maximum refinement level b = $b; keep refinement below level b (set when constructing `ForestBWG(grid, b)`)"))
"""
    _compute_size(b::Integer, l::Integer) -> Int

Edge length, in integer octree coordinates, of a level-`l` octant in a tree using `b` bits:
``2^{b-l}``, computed as a bit shift `1 << (b - l)`. The root (`l = 0`) spans the whole domain
(`2^b`); a leaf at the maximum level `b` has size `1` (the finest representable octant).

A level `l > b` is invalid (it would need a sub-atom octant) and throws a `DomainError` — this is
the guard against refining or iterating past the tree's maximum refinement level `b`.
"""
function _compute_size(b::Integer, l::Integer)
    l <= b || _throw_octant_level(l, b)
    return 1 << (b - l)
end
"""
    _maximum_size(b::Integer) -> Int

Integer extent `2^b` of the root octant along each axis (computed as `1 << b`): the octree
coordinate domain of a tree is `[0, 2^b)^dim`. Used by [`inside`](@ref) to detect when an octant
leaves its tree.
"""
_maximum_size(b::Integer) = 1 << b
# return the two adjacent faces $f_i$ adjacent to edge `edge`
_face(edge::Int) = 𝒮[edge, :]
# return the `i`-th adjacent face fᵢ to edge `edge`
_face(edge::Int, i::Int) = 𝒮[edge, i]
# return two face corners ξᵢ of the face `face` along edge `edge`
_face_edge_corners(edge::Int, face::Int) = 𝒯[edge, face]
# return the two `edge` corners cᵢ
_edge_corners(edge::Int) = 𝒰[edge, :]
# return the `i`-th edge corner of `edge`
_edge_corners(edge::Int, i::Int) = 𝒰[edge, i]
# finds face corner ξ′ in f′ for two associated faces f,f′ in {1,...,6} and their orientation r in {1,...,4}
_neighbor_corner(f::Int, f′::Int, r::Int, ξ::Int) = 𝒫[𝒬[ℛ[f, f′], r], ξ]

# map given `face` to its corners `c`. Need to provide dim for different lookup
function _face_corners(dim::Int, face::Int)
    if dim == 2
        return 𝒱₂[face, :]
    elseif dim == 3
        return 𝒱₃[face, :]
    else
        error("No corner-lookup table available")
    end
end

##### OCTANT LOOK UP TABLES ######
# All indices in these tables are in p4est/BWG numbering (one-based), see Section 2 and
# Tables 1-3 of [BWG2011]. The `*_perm`/`*_inv` tables at the end translate between the
# p4est and Ferrite orderings.

# Edge -> the two faces containing that edge: 𝒮[e, :] ("S" in Section 2.1 of [BWG2011]).
const 𝒮 = [
    3  5
    4  5
    3  6
    4  6
    1  5
    2  5
    1  6
    2  6
    1  3
    2  3
    1  4
    2  4
]

# (Edge, face) -> the face-local corner pair spanning that edge within the face:
# 𝒯[e, f] ("T" in Section 2.1 of [BWG2011]); (0, 0) marks edges not contained in the face.
const 𝒯 = [
    (0, 0)  (0, 0)  (1, 2)  (0, 0)  (1, 2)  (0, 0)
    (0, 0)  (0, 0)  (0, 0)  (1, 2)  (3, 4)  (0, 0)
    (0, 0)  (0, 0)  (3, 4)  (0, 0)  (0, 0)  (1, 2)
    (0, 0)  (0, 0)  (0, 0)  (3, 4)  (0, 0)  (3, 4)
    (1, 2)  (0, 0)  (0, 0)  (0, 0)  (1, 3)  (0, 0)
    (0, 0)  (1, 2)  (0, 0)  (0, 0)  (2, 4)  (0, 0)
    (3, 4)  (0, 0)  (0, 0)  (0, 0)  (0, 0)  (1, 3)
    (0, 0)  (3, 4)  (0, 0)  (0, 0)  (0, 0)  (2, 4)
    (1, 3)  (0, 0)  (1, 3)  (0, 0)  (0, 0)  (0, 0)
    (0, 0)  (1, 3)  (2, 4)  (0, 0)  (0, 0)  (0, 0)
    (2, 4)  (0, 0)  (0, 0)  (1, 3)  (0, 0)  (0, 0)
    (0, 0)  (2, 4)  (0, 0)  (2, 4)  (0, 0)  (0, 0)
]

# Edge -> the two octant corners bounding that edge: 𝒰[e, :] ("U" in Section 2.1 of [BWG2011]).
const 𝒰 = [
    1  2
    3  4
    5  6
    7  8
    1  3
    2  4
    5  7
    6  8
    1  5
    2  6
    3  7
    4  8
]

# Inverse of 𝒰: row = corner, entries = the three edges touching that corner.
const 𝒰_inv = [
    1  5  9
    1  6  10
    2  5  11
    2  6  12
    3  7  9
    3  8  10
    4  7  11
    4  8  12
]

# Face -> the corners on that face, 2D: 𝒱₂[f, :] ("V" in Section 2.1 of [BWG2011]; faces 1/2 = x∓, 3/4 = y∓).
const 𝒱₂ = [
    1  3
    2  4
    1  2
    3  4
]

# Inverse of 𝒱₂: row = corner, entries = the two faces touching that corner.
const 𝒱₂_inv = [
    1  3
    2  3
    1  4
    2  4
]

# Face -> the four corners on that face, 3D: 𝒱₃[f, :] ("V" in Section 2.1 of [BWG2011]; faces 1/2 = x∓,
# 3/4 = y∓, 5/6 = z∓).
const 𝒱₃ = [
    1  3  5  7
    2  4  6  8
    1  2  5  6
    3  4  7  8
    1  2  3  4
    5  6  7  8
]

# Inverse of 𝒱₃: row = corner, entries = the three faces touching that corner.
const 𝒱₃_inv = [
    1  3  5
    2  3  5
    1  4  5
    2  4  5
    1  3  6
    2  3  6
    1  4  6
    2  4  6
]

# Face indices permutation from p4est idx to Ferrite idx
const 𝒱₂_perm = [
    4
    2
    1
    3
]

# Face indices permutation from Ferrite idx to p4est idx
const 𝒱₂_perm_inv = [
    3
    2
    4
    1
]

# Face indices permutation from p4est idx to Ferrite idx (3D)
const 𝒱₃_perm = [
    5
    3
    2
    4
    1
    6
]

# Face indices permutation from Ferrite idx to p4est idx (3D)
const 𝒱₃_perm_inv = [
    5
    3
    2
    4
    1
    6
]

# edge indices permutation from p4est idx to Ferrite idx
const edge_perm = [
    1
    3
    5
    7
    4
    2
    8
    6
    9
    10
    12
    11
]

# edge indices permutation from Ferrite idx to p4est idx
const edge_perm_inv = [
    1
    6
    2
    5
    3
    8
    4
    7
    9
    10
    12
    11
]

# The corner-permutation machinery of Theorem 2.2 and Table 3 of [BWG2011]: for two touching faces
# (f, f′) with orientation r, corner ξ of f maps to corner 𝒫[𝒬[ℛ[f, f′], r + 1], ξ] of f′
# (see `rotation_permutation` / `_neighbor_corner`).
# ℛ[f, f′]: which of the three alignment cases of the face pair applies ("R", one-based).
const ℛ = [
    1  2  2  1  1  2
    3  1  1  2  2  1
    3  1  1  2  2  1
    1  3  3  1  1  2
    1  3  3  1  1  2
    3  1  1  3  3  1
]

# 𝒬[ℛ-case, r + 1]: selects the row of 𝒫 for face orientation r ∈ {0,...,3} ("Q").
const 𝒬 = [
    2  3  6  7
    1  4  5  8
    1  5  4  8
]

# The eight corner permutations of a square face ("P"); row selected via 𝒬.
const 𝒫 = [
    1  2  3  4
    1  3  2  4
    2  1  4  3
    2  4  1  3
    3  1  4  2
    3  4  1  2
    4  2  3  1
    4  3  2  1
]

# Node indices permutation from p4est idx to Ferrite idx
const node_map₂ = [
    1,
    2,
    4,
    3,
]

# Node indices permutation from Ferrite idx to p4est idx
const node_map₂_inv = [
    1,
    2,
    4,
    3,
]

const node_map₃ = [
    1,
    2,
    4,
    3,
    5,
    6,
    8,
    7,
]

const node_map₃_inv = [
    1,
    2,
    4,
    3,
    5,
    6,
    8,
    7,
]
