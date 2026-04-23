# TODO we should remove the mixture of indices. Maybe with these:
# - struct FacetIndexBWG ... end
# - struct QuadrilateralBWG ... end
# - struct HexahedronBWG ... end

abstract type AbstractAdaptiveGrid{dim} <: Ferrite.AbstractGrid{dim} end
abstract type AbstractAdaptiveCell{refshape <: Ferrite.AbstractRefShape} <: Ferrite.AbstractCell{refshape} end

const ncorners_face3D = 4
const ncorners_face2D = 2
const ncorners_edge = ncorners_face2D

_maxlevel = [30, 19]

function set_maxlevel(dim::Integer, maxlevel::Integer)
    return _maxlevel[dim - 1] = maxlevel
end

struct OctantBWG{dim, N, T} <: Ferrite.AbstractCell{Ferrite.RefHypercube{dim}}
    #Refinement level
    l::T
    #x,y,z \in {0,...,2^b} where (0 ≤ l ≤ b)}
    xyz::NTuple{dim, T}
end

"""
    OctantBWG(dim::Integer, l::Integer, b::Integer, m::Integer)
Construct an `octant` based on dimension `dim`, level `l`, amount of levels `b` and morton index `m`
"""
function OctantBWG(dim::Integer, l::T1, m::T2, b::T1 = _maxlevel[dim - 1]) where {T1 <: Integer, T2 <: Integer}
    @assert l ≤ b #maximum refinement level exceeded
    @assert m ≤ (one(T1) + one(T1))^(dim * l)
    x, y, z = (zero(T1), zero(T1), zero(T1))
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

#OctantBWG(dim::Int,l::Int,m::Int,b::Int=_maxlevel[dim-1]) = OctantBWG(dim,l,m,b)
#OctantBWG(dim::Int,l::Int,m::Int,b::Int32) = OctantBWG(dim,l,m,b)
#OctantBWG(dim::Int,l::Int32,m::Int,b::Int32) = OctantBWG(dim,l,Int32(m),b)
function OctantBWG(level::T, coords::NTuple) where {T <: Integer}
    dim = length(coords)
    nnodes = 2^dim
    return OctantBWG{dim, nnodes, eltype(coords)}(level, coords)
end
#OctantBWG(level::Int32,coords::NTuple) = OctantBWG(level,Int32.(coords))
#OctantBWG(level::Int32, coords::NTuple{dim,Int32}) where dim = OctantBWG{dim,2^dim,2*dim,Int32}(level,coords)

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

TODO: use LUT method from https://www.forceflow.be/2013/10/07/morton-encodingdecoding-through-bit-interleaving-implementations/
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
Base.eltype(::Type{OctantBWG{dim, N, T}}) where {dim, N, T} = T

ncorners(::Type{OctantBWG{dim, N, T}}) where {dim, N, T} = N # TODO change to how many corners
ncorners(o::OctantBWG) = ncorners(typeof(o))
nnodes(::Type{OctantBWG{dim, N, T}}) where {dim, N, T} = N
nnodes(o::OctantBWG) = ncorners(typeof(o))
nchilds(::Type{OctantBWG{dim, N, T}}) where {dim, N, T} = N
nchilds(o::OctantBWG) = nchilds(typeof(o)) # Follow z order, x before y before z for faces, edges and corners

Base.isequal(o1::OctantBWG, o2::OctantBWG) = (o1.l == o2.l) && (o1.xyz == o2.xyz)
"""
    o1::OctantBWG < o2::OctantBWG
Implements Algorithm 2.1 of [IBWG2015](@citet).
Checks first if mortonid is smaller and later if level is smaller.
Thus, ancestors precede descendants (preordering).
"""
function Base.isless(o1::OctantBWG, o2::OctantBWG)
    if o1.xyz != o2.xyz
        #TODO verify b=o1.l/b=o2.l as argument potential bug otherwise
        return morton(o1, o1.l, o1.l) < morton(o2, o2.l, o2.l)
    else
        return o1.l < o2.l
    end
end

"""
    children(octant::OctantBWG{dim,N,T}, b::Integer) -> NTuple{M,OctantBWG}
Computes all childern of `octant`
"""
function children(octant::OctantBWG{dim, N, T}, b::Integer) where {dim, N, T}
    o = one(T)
    _nchilds = nchilds(octant)
    startid = morton(octant, octant.l + o, b)
    endid = startid + _nchilds + o
    return ntuple(i -> OctantBWG(dim, octant.l + o, (startid:endid)[i], b), _nchilds)
end

abstract type OctantIndex{T <: Integer} end
Base.isequal(i1::T, i2::T) where {T <: OctantIndex} = i1.idx == i2.idx #same type
Base.isequal(i1::T1, i2::T2) where {T1 <: OctantIndex, T2 <: OctantIndex} = false #different type

struct OctantCornerIndex{T} <: OctantIndex{T}
    idx::T
end
Base.hash(idx::OctantCornerIndex) = Base.hash((0, idx.idx))
Base.show(io::IO, ::MIME"text/plain", c::OctantCornerIndex) = print(io, "O-Corner $(c.idx)")
Base.show(io::IO, c::OctantCornerIndex) = print(io, "O-Corner $(c.idx)")

struct OctantEdgeIndex{T} <: OctantIndex{T}
    idx::T
end
Base.hash(idx::OctantEdgeIndex) = Base.hash((1, idx.idx))
Base.show(io::IO, ::MIME"text/plain", e::OctantEdgeIndex) = print(io, "O-Edge $(e.idx)")
Base.show(io::IO, e::OctantEdgeIndex) = print(io, "O-Edge $(e.idx)")

struct OctantFaceIndex{T} <: OctantIndex{T}
    idx::T
end
Base.hash(idx::OctantFaceIndex) = Base.hash((2, idx.idx))
Base.show(io::IO, ::MIME"text/plain", f::OctantFaceIndex) = print(io, "O-Face $(f.idx)")
Base.show(io::IO, f::OctantFaceIndex) = print(io, "O-Face $(f.idx)")

vertex(octant::OctantBWG, c::OctantCornerIndex, b::Integer) = vertex(octant, c.idx, b)
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

face(octant::OctantBWG, f::OctantFaceIndex, b::Integer) = face(octant, f.idx, b)
function face(octant::OctantBWG{2}, f::Integer, b::Integer)
    cornerid = view(𝒱₂, f, :)
    return ntuple(i -> vertex(octant, cornerid[i], b), 2)
end

function face(octant::OctantBWG{3}, f::Integer, b::Integer)
    cornerid = view(𝒱₃, f, :)
    return ntuple(i -> vertex(octant, cornerid[i], b), 4)
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

edge(octant::OctantBWG, e::OctantEdgeIndex, b::Integer) = edge(octant, e.idx, b)
function edge(octant::OctantBWG{3}, e::Integer, b::Integer)
    cornerid = view(𝒰, e, :)
    return ntuple(i -> vertex(octant, cornerid[i], b), 2)
end

"""
    edges(octant::OctantBWG{dim}, b::Integer)

Computes all edges of a given `octant`. Each edge is encoded within the octree coordinates i.e. by integers.
Further, each edge consists of two three-dimensional integer coordinates.
"""
edges(octant::OctantBWG{3}, b::Integer) = ntuple(i -> edge(octant, i, b), 12)

"""
    boundaryset(o::OctantBWG{2}, i::Integer, b::Integer
implements two dimensional boundaryset table from Fig.4.1 [IBWG2015](@citet)
TODO: could be done little bit less ugly
"""
function boundaryset(o::OctantBWG{2, N, T}, i::Integer, b::Integer) where {N, T}
    if i == 1
        return Set((OctantCornerIndex(1), OctantFaceIndex(1), OctantFaceIndex(3)))
    elseif i == 2
        return Set((OctantCornerIndex(2), OctantFaceIndex(2), OctantFaceIndex(3)))
    elseif i == 3
        return Set((OctantCornerIndex(3), OctantFaceIndex(1), OctantFaceIndex(4)))
    elseif i == 4
        return Set((OctantCornerIndex(4), OctantFaceIndex(2), OctantFaceIndex(4)))
    else
        throw("no boundary")
    end
end

"""
    boundaryset(o::OctantBWG{3}, i::Integer, b::Integer
implements three dimensional boundaryset table from Fig.4.1 [IBWG2015](@citet)
TODO: could be done little bit less ugly
"""
function boundaryset(o::OctantBWG{3, N, T}, i::Integer, b::Integer) where {N, T}
    if i == 1
        return Set((OctantCornerIndex(1), OctantEdgeIndex(1), OctantEdgeIndex(5), OctantEdgeIndex(9), OctantFaceIndex(1), OctantFaceIndex(3), OctantFaceIndex(5)))
    elseif i == 2
        return Set((OctantCornerIndex(2), OctantEdgeIndex(1), OctantEdgeIndex(6), OctantEdgeIndex(10), OctantFaceIndex(2), OctantFaceIndex(3), OctantFaceIndex(5)))
    elseif i == 3
        return Set((OctantCornerIndex(3), OctantEdgeIndex(2), OctantEdgeIndex(5), OctantEdgeIndex(11), OctantFaceIndex(1), OctantFaceIndex(4), OctantFaceIndex(5)))
    elseif i == 4
        return Set((OctantCornerIndex(4), OctantEdgeIndex(2), OctantEdgeIndex(6), OctantEdgeIndex(12), OctantFaceIndex(2), OctantFaceIndex(4), OctantFaceIndex(5)))
    elseif i == 5
        return Set((OctantCornerIndex(5), OctantEdgeIndex(3), OctantEdgeIndex(7), OctantEdgeIndex(9), OctantFaceIndex(1), OctantFaceIndex(3), OctantFaceIndex(6)))
    elseif i == 6
        return Set((OctantCornerIndex(6), OctantEdgeIndex(3), OctantEdgeIndex(8), OctantEdgeIndex(10), OctantFaceIndex(2), OctantFaceIndex(3), OctantFaceIndex(6)))
    elseif i == 7
        return Set((OctantCornerIndex(7), OctantEdgeIndex(4), OctantEdgeIndex(7), OctantEdgeIndex(11), OctantFaceIndex(1), OctantFaceIndex(4), OctantFaceIndex(6)))
    elseif i == 8
        return Set((OctantCornerIndex(8), OctantEdgeIndex(4), OctantEdgeIndex(8), OctantEdgeIndex(12), OctantFaceIndex(2), OctantFaceIndex(4), OctantFaceIndex(6)))
    else
        throw("no boundary")
    end
end

"""
    find_range_boundaries(f::OctantBWG{dim,N,T}, l::OctantBWG{dim,N,T}, s::OctantBWG{dim,N,T}, idxset, b)
    find_range_boundaries(s::OctantBWG{dim,N,T}, idxset, b)
Algorithm 4.2 of [IBWG2015](@citet)
TODO: write tests
"""
function find_range_boundaries(f::OctantBWG{dim, N, T1}, l::OctantBWG{dim, N, T1}, s::OctantBWG{dim, N, T1}, idxset::Set{OctantIndex{T2}}, b) where {dim, N, T1, T2}
    o = one(T1)
    if isempty(idxset) || s.l == b
        return idxset
    end
    j = ancestor_id(f, s.l + o, b); k = ancestor_id(l, s.l + o, b)
    boundary_j = boundaryset(s, j, b)
    kidz = children(s, b)
    if j == k
        return find_range_boundaries(f, l, kidz[j], idxset ∩ boundary_j, b)
    end
    idxset_match = Set{OctantIndex{T2}}()
    for i in (j + o):(k - o)
        union!(idxset_match, idxset ∩ boundaryset(s, i, b))
    end
    boundary_k = boundaryset(s, k, b)
    idxset_match_j = setdiff((idxset ∩ boundary_j), idxset_match)
    fj, lj = descendants(kidz[j], b)
    if fj != f
        idxset_match_j = find_range_boundaries(f, lj, kidz[j], idxset_match_j, b)
    end
    idxset_match_k = setdiff(setdiff((idxset ∩ boundary_k), idxset_match), idxset_match_j)
    fk, lk = descendants(kidz[k], b)
    if lk != l
        idxset_match_k = find_range_boundaries(fk, l, kidz[k], idxset_match_k, b)
    end
    return idxset_match ∪ idxset_match_j ∪ idxset_match_k
end

#for convenience, should probably changed to parent(s) until parent(s)==root and then descendants(root)
function find_range_boundaries(s::OctantBWG, idxset, b)
    f, l = descendants(s, b)
    return find_range_boundaries(f, l, s, idxset, b)
end

function isrelevant(xyz::NTuple{dim, T}, leafsuppₚ::Set{<:OctantBWG}) where {dim, T}
    ###### only relevant for distributed
    #for all s in leafsuppₚ
    #    if s in 𝒪ₚ
    #        return true
    #    else
    #        check stuff Algorithm 5.1 line 4-5
    #    end
    #end
    return true
end

struct OctreeBWG{dim, N, T} <: AbstractAdaptiveCell{Ferrite.RefHypercube{dim}}
    leaves::Vector{OctantBWG{dim, N, T}}
    #maximum refinement level
    b::T
    nodes::NTuple{N, Int}
end

function refine!(octree::OctreeBWG{dim, N, T}, pivot_octant::OctantBWG{dim, N, T}) where {dim, N, T <: Integer}
    if !(pivot_octant.l + 1 <= octree.b)
        return
    end
    o = one(T)
    # TODO replace this with recursive search function
    leave_idx = findfirst(x -> x == pivot_octant, octree.leaves)
    old_octant = popat!(octree.leaves, leave_idx)
    _children = children(pivot_octant, octree.b)
    for child in _children
        insert!(octree.leaves, leave_idx, child)
        leave_idx += 1
    end
    return
end

function coarsen!(octree::OctreeBWG{dim, N, T}, o::OctantBWG{dim, N, T}) where {dim, N, T <: Integer}
    _two = T(2)
    leave_idx = findfirst(x -> x == o, octree.leaves)
    shift = child_id(o, octree.b) - one(T)
    if shift != zero(T)
        old_morton = morton(o, o.l, octree.b)
        o = OctantBWG(dim, o.l, old_morton, octree.b)
    end
    window_start = leave_idx - shift
    window_length = _two^dim - one(T)
    new_octant = parent(o, octree.b)
    octree.leaves[leave_idx - shift] = new_octant
    return deleteat!(octree.leaves, (leave_idx - shift + one(T)):(leave_idx - shift + window_length))
end

OctreeBWG{3, 8}(nodes::NTuple, b = _maxlevel[2]) = OctreeBWG{3, 8, Int64}([zero(OctantBWG{3, 8})], Int64(b), nodes)
OctreeBWG{2, 4}(nodes::NTuple, b = _maxlevel[1]) = OctreeBWG{2, 4, Int64}([zero(OctantBWG{2, 4})], Int64(b), nodes)
OctreeBWG(cell::Quadrilateral, b = _maxlevel[2]) = OctreeBWG{2, 4}(cell.nodes, b)
OctreeBWG(cell::Hexahedron, b = _maxlevel[1]) = OctreeBWG{3, 8}(cell.nodes, b)

Base.length(tree::OctreeBWG) = length(tree.leaves)
Base.eltype(::Type{OctreeBWG{dim, N, T}}) where {dim, N, T} = T

function inside(oct::OctantBWG{dim}, b) where {dim}
    maxsize = _maximum_size(b)
    outside = any(xyz -> xyz >= maxsize, oct.xyz) || any(xyz -> xyz < 0, oct.xyz)
    return !outside
end

inside(tree::OctreeBWG{dim}, oct::OctantBWG{dim}) where {dim} = inside(oct, tree.b)

"""
    split_array(octree::OctreeBWG, a::OctantBWG)
    split_array(octantarray, a::OctantBWG, b::Integer)
Algorithm 3.3 of [IBWG2015](@citet). Efficient binary search.
"""
function split_array(octantarray, a::OctantBWG{dim, N, T}, b::Integer) where {dim, N, T}
    o = one(T)
    𝐤 = T[i == 1 ? 1 : length(octantarray) + 1 for i in 1:(2^dim + 1)]
    for i in 2:(2^dim)
        m = 𝐤[i - 1]
        while m < 𝐤[i]
            n = m + (𝐤[i] - m) ÷ 2
            c = ancestor_id(octantarray[n], a.l + o, b)
            if c < i
                m = n + 1
            else
                for j in i:c
                    𝐤[j] = n
                end
            end
        end
    end
    #TODO non-allocating way?
    return ntuple(i -> view(octantarray, 𝐤[i]:(𝐤[i + 1] - 1)), 2^dim)
end

split_array(tree::OctreeBWG, a::OctantBWG) = split_array(tree.leaves, a, tree.b)

function search(octantarray, a::OctantBWG{dim, N, T1}, idxset::Vector{T2}, b::Integer, Match = match) where {dim, N, T1 <: Integer, T2}
    isempty(octantarray) && return
    isleaf = (length(octantarray) == 1 && a ∈ octantarray) ? true : false
    idxset_match = eltype(idxset)[]
    for q in idxset
        if Match(a, isleaf, q, b)
            push!(idxset_match, q)
        end
    end
    if isempty(idxset_match) && !isleaf
        𝐇 = split_array(octantarray, a, b)
        _children = children(a, b)
        for (child, h) in zip(_children, 𝐇)
            search(h, child, idxset_match, b)
        end
    end
    return idxset_match
end

search(tree::OctreeBWG, a::OctantBWG, idxset, Match = match) = search(tree.leaves, a, idxset, tree.b, match)

"""
    match(o::OctantBWG, isleaf::Bool, q)
from [IBWG2015](@citet)
> match returns true if there is a leaf r ∈ 𝒪 that is a descendant of o
> such that match_q(r) = true, and is allowed to return a false positive
> (i.e., true even if match_q(r) = false for all descendants leaves of o)
> if isleaf=true, then the return  value of match is irrelevant
I don't understand what of a to check against index q
"""
function match(o::OctantBWG, isleaf::Bool, q, b)
    isleaf && (return true)
    println(q)
    println(o)
    return false
end

"""
    ForestBWG{dim, C<:AbstractAdaptiveCell, T<:Real} <: AbstractAdaptiveGrid{dim}
`p4est` adaptive grid implementation based on [BWG2011](@citet)
and [IBWG2015](@citet).

## Constructor
    ForestBWG(grid::AbstractGrid{dim}, b=_maxlevel[dim-1]) where dim
Builds an adaptive grid based on a non-adaptive one `grid` and a given max refinement level `b`.
"""
struct ForestBWG{dim, C <: OctreeBWG, T <: Real} <: AbstractAdaptiveGrid{dim}
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

function ForestBWG(grid::Ferrite.AbstractGrid{dim}, b = _maxlevel[dim - 1]) where {dim}
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

function refine_all!(forest::ForestBWG, l)
    for tree in forest.cells
        for leaf in tree.leaves
            if leaf.l != l - 1 #maxlevel
                continue
            else
                refine!(tree, leaf)
            end
        end
    end
    return
end

function refine!(forest::ForestBWG, cellid::Integer)
    nleaves_k = length(forest.cells[1].leaves)
    prev_nleaves_k = 0
    k = 1
    while nleaves_k < cellid
        k += 1
        prev_nleaves_k = nleaves_k
        nleaves_k += length(forest.cells[k].leaves)
    end
    return refine!(forest.cells[k], forest.cells[k].leaves[cellid - prev_nleaves_k])
end

function refine!(forest::ForestBWG, cellids::Vector{<:Integer})
    sort!(cellids)
    ncells = getncells(forest)
    shift = 0
    for cellid in cellids
        refine!(forest, cellid + shift)
        shift += getncells(forest) - ncells
        ncells = getncells(forest)
    end
    return
end

function coarsen_all!(forest::ForestBWG)
    for tree in forest.cells
        for leaf in tree.leaves
            if child_id(leaf, tree.b) == 1
                coarsen!(tree, leaf)
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

function Ferrite.getcells(forest::ForestBWG{dim, C}) where {dim, C}
    treetype = C
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

function Ferrite.getcells(forest::ForestBWG{dim}, cellid::Int) where {dim}
    @warn "Slow dispatch, consider to call `getcells(forest)` once instead" maxlog = 1 #TODO doc page for performance
    #TODO should nleaves be saved by forest?
    nleaves = length.(forest.cells) # cells=trees
    #TODO remove that later by for loop or [IBWG2015](@citet) iterator approach
    nleaves_cumsum = cumsum(nleaves)
    k = findfirst(x -> cellid <= x, nleaves_cumsum)
    #TODO is this actually correct?
    leafid = k == 1 ? cellid : cellid - (nleaves_cumsum[k] - nleaves[k])
    return forest.cells[k].leaves[leafid]
end

Ferrite.getcelltype(grid::ForestBWG) = eltype(grid.cells)
Ferrite.getcelltype(grid::ForestBWG, i::Int) = eltype(grid.cells) # assume for now same cell type TODO

"""
    transform_pointBWG(forest, vertices) -> Vector{Vec{dim}}
    transform_pointBWG(forest::ForestBWG{dim}, k::Integer, vertex::NTuple{dim,T}) where {dim,T} -> Vec{dim}

Transformation of a octree coordinate system point `vertex` (or a collection `vertices`) to the corresponding physical coordinate system.
"""
function transform_pointBWG(forest::ForestBWG{dim}, k::Integer, vertex::NTuple{dim, T}) where {dim, T}
    tree = forest.cells[k]
    cellnodes = getnodes(forest, collect(tree.nodes)) .|> get_node_coordinate
    vertex = vertex .* (2 / (2^tree.b)) .- 1
    octant_physical_coordinates = sum(j -> cellnodes[j] * Ferrite.reference_shape_value(Lagrange{Ferrite.RefHypercube{dim}, 1}(), Vec{dim}(vertex), j), 1:length(cellnodes))
    return Vec{dim}(octant_physical_coordinates)
end

transform_pointBWG(forest, vertices) = transform_pointBWG.((forest,), first.(vertices), last.(vertices))

"""
    rotation_permutation(::Val{2},r,i) -> i′
computes based on the rotation indicator `r` ∈ {0,1} and a given corner index `i` ∈ {1,2} the permuted corner index `i′`
"""
function rotation_permutation(r, i)
    i′ = r == 0 ? i : 3 - i
    return i′
end

"""
    rotation_permutation(f,f′,r,i) -> i′
computes based on the rotation indicator `r` ∈ {0,...,3} and a given corner index `i` ∈ {1,...,4} the permuted corner index `i′`
See Table 3 and Theorem 2.2 [BWG2011](@citet).
"""
function rotation_permutation(f, f′, r, i)
    return 𝒫[𝒬[ℛ[f, f′], r + 1], i]
end

p4est_opposite_face_index(f) = ((f - 1) ⊻ 0b1) + 1
p4est_opposite_edge_index(e) = ((e - 1) ⊻ 0b11) + 1

#TODO: this function should wrap the LNodes Iterator of [IBWG2015](@citet)
"""
    creategrid(forest::ForestBWG) -> NonConformingGrid
Materializes a p4est forest into a NonConformingGrid that can be used as usual to solve a finite element problem.
"""
function creategrid(forest::ForestBWG{dim, C, T}) where {dim, C, T}
    nodes = Vector{Tuple{Int, NTuple{dim, Int32}}}()
    sizehint!(nodes, getncells(forest) * 2^dim)
    _perm = dim == 2 ? 𝒱₂_perm : 𝒱₃_perm
    _perminv = dim == 2 ? 𝒱₂_perm_inv : 𝒱₃_perm_inv
    node_map = dim < 3 ? node_map₂ : node_map₃
    node_map_inv = dim < 3 ? node_map₂_inv : node_map₃_inv
    nodeids = Dict{Tuple{Int, NTuple{dim, Int32}}, Int}()
    nodeowners = Dict{Tuple{Int, NTuple{dim, Int32}}, Tuple{Int, NTuple{dim, Int32}}}()
    facet_neighborhood = Ferrite.Ferrite.get_facet_facet_neighborhood(forest)

    # Phase 1: Assign node owners intra-octree
    pivot_nodeid = 1
    for (k, tree) in enumerate(forest.cells)
        for leaf in tree.leaves
            _vertices = vertices(leaf, tree.b)
            for v in _vertices
                push!(nodes, (k, v))
                nodeids[(k, v)] = pivot_nodeid
                pivot_nodeid += 1
                nodeowners[(k, v)] = (k, v)
            end
        end
    end

    # Phase 2: Assign node owners inter-octree
    for (k, tree) in enumerate(forest.cells)
        _vertices = vertices(root(dim), tree.b)
        # Vertex neighbors
        @debug println("Setting vertex neighbors for octree $k")
        for (v, vc) in enumerate(_vertices)
            vertex_neighbor = forest.topology.vertex_vertex_neighbor[k, node_map[v]]
            for (k′, v′) in vertex_neighbor
                @debug println("  pair $v $v′")
                if k > k′
                    #delete!(nodes,(k,v))
                    new_v = vertex(root(dim), node_map[v′], tree.b)
                    nodeids[(k, vc)] = nodeids[(k′, new_v)]
                    nodeowners[(k, vc)] = (k′, new_v)
                    @debug println("    Matching $vc (local) to $new_v (neighbor)")
                end
            end
            # TODO check if we need to also update the face neighbors
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
                    for leaf in tree.leaves
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
                                if haskey(nodeids, (k′, fnodes_neighbor[i′]))
                                    @debug println("    Updating $((k, fnodes[i])) $(nodeids[(k, fnodes[i])]) -> $(nodeids[(k′, fnodes_neighbor[i′])])")
                                    nodeids[(k, fnodes[i])] = nodeids[(k′, fnodes_neighbor[i′])]
                                    nodeowners[(k, fnodes[i])] = (k′, fnodes_neighbor[i′])
                                end
                            end
                        else
                            for i in 1:ncorners_face3D
                                rotated_ξ = rotation_permutation(f′, f, r, i)
                                if haskey(nodeids, (k′, fnodes_neighbor[i]))
                                    @debug println("    Updating $((k, fnodes[i])) $(nodeids[(k, fnodes[rotated_ξ])]) -> $(nodeids[(k′, fnodes_neighbor[i])])")
                                    nodeids[(k, fnodes[rotated_ξ])] = nodeids[(k′, fnodes_neighbor[i])]
                                    nodeowners[(k, fnodes[rotated_ξ])] = (k′, fnodes_neighbor[i])
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
                    for leaf in tree.leaves
                        # First we skip edges which are not on the current edge of the root element
                        enodes = edge(leaf, e, tree.b)
                        if !contains_edge(ec, enodes)
                            @debug println("  Rejecting leaf $leaf because its edge $enodes is not on the octant boundary")
                            continue
                        end
                        neighbor_candidate = transform_edge(forest, k′, e′, leaf, false)
                        # Candidate must be the edge opposite to e'
                        e′candidate = p4est_opposite_edge_index(e′)

                        enodes_neighbor = edge(neighbor_candidate, e′candidate, tree′.b)
                        r = compute_edge_orientation(forest, k, e)
                        @debug println("  Trying to match $enodes (local) to $enodes_neighbor (neighbor $neighbor_candidate)")
                        for i in 1:ncorners_edge
                            i′ = rotation_permutation(r, i)
                            if haskey(nodeids, (k′, enodes_neighbor[i′]))
                                @debug println("    Updating $((k, enodes[i])) $(nodeids[(k, enodes[i])]) -> $(nodeids[(k′, enodes_neighbor[i′])])")
                                nodeids[(k, enodes[i])] = nodeids[(k′, enodes_neighbor[i′])]
                                nodeowners[(k, enodes[i])] = (k′, enodes_neighbor[i′])
                            end
                        end
                    end
                end
            end
        end
    end

    # Phase 3: Compute unique physical nodes
    nodeids_dedup = Dict{Int, Int}()
    next_nodeid = 1
    for (kv, nodeid) in nodeids
        if !haskey(nodeids_dedup, nodeid)
            nodeids_dedup[nodeid] = next_nodeid
            next_nodeid += 1
        end
    end
    nodes_physical_all = transform_pointBWG(forest, nodes)
    nodes_physical = zeros(eltype(nodes_physical_all), next_nodeid - 1)
    for (ni, (kv, nodeid)) in enumerate(nodeids)
        nodes_physical[nodeids_dedup[nodeid]] = nodes_physical_all[nodeid]
    end

    # Phase 4: Generate cells
    celltype = dim < 3 ? Quadrilateral : Hexahedron
    cells = celltype[]
    cellnodes = zeros(Int, 2^dim)
    for (k, tree) in enumerate(forest.cells)
        for leaf in tree.leaves
            _vertices = vertices(leaf, tree.b)
            cellnodes = ntuple(i -> nodeids_dedup[nodeids[nodeowners[(k, _vertices[i])]]], length(_vertices))
            push!(cells, celltype(ntuple(i -> cellnodes[node_map[i]], length(cellnodes))))
        end
    end

    # Phase 5: Generate grid and haning nodes
    facetsets = reconstruct_facetsets(forest) #TODO edge, node and cellsets
    #facesets_ordered = OrderedSet(facetsets)
    hnodes = hangingnodes(forest, nodeids, nodeowners)
    hnodes_dedup = Dict{Int64, Vector{Int64}}()
    for (constrained, constainers) in hnodes
        hnodes_dedup[nodeids_dedup[constrained]] = [nodeids_dedup[constainer] for constainer in constainers]
    end
    return NonConformingGrid(cells, nodes_physical .|> Node, facetsets = facetsets, conformity_info = hnodes_dedup)
end

function reconstruct_facetsets(forest::ForestBWG{dim}) where {dim}
    _perm = dim == 2 ? 𝒱₂_perm : 𝒱₃_perm
    _perm_inv = dim == 2 ? 𝒱₂_perm_inv : 𝒱₃_perm_inv
    new_facesets = typeof(forest.facetsets)()
    for (facetsetname, facetset) in forest.facetsets
        new_facetset = typeof(facetset)()
        for facetidx in facetset
            pivot_tree = forest.cells[facetidx[1]]
            last_cellid = facetidx[1] != 1 ? sum(length, @view(forest.cells[1:(facetidx[1] - 1)])) : 0
            pivot_faceid = facetidx[2]
            pivot_face = faces(root(dim), pivot_tree.b)[_perm_inv[pivot_faceid]]
            for (leaf_idx, leaf) in enumerate(pivot_tree.leaves)
                for (leaf_face_idx, leaf_face) in enumerate(faces(leaf, pivot_tree.b))
                    if contains_facet(pivot_face, leaf_face)
                        ferrite_leaf_face_idx = _perm[leaf_face_idx]
                        push!(new_facetset, FacetIndex(last_cellid + leaf_idx, ferrite_leaf_face_idx))
                    end
                end
            end
        end
        new_facesets[facetsetname] = new_facetset
    end
    return new_facesets
end

"""
    hangingnodes(forest,nodeids,nodeowners)
Constructs a map from constrained nodeids to the ones that constraint
"""
function hangingnodes(forest::ForestBWG{dim}, nodeids, nodeowners) where {dim}
    _perm = dim == 2 ? 𝒱₂_perm : 𝒱₃_perm
    _perminv = dim == 2 ? 𝒱₂_perm_inv : 𝒱₃_perm_inv
    facetable = dim == 2 ? 𝒱₂ : 𝒱₃
    opposite_face = dim == 2 ? opposite_face_2 : opposite_face_3
    hnodes = Dict{Int, Vector{Int}}()
    facet_neighborhood = Ferrite.Ferrite.get_facet_facet_neighborhood(forest)
    for (k, tree) in enumerate(forest.cells)
        rootfaces = faces(root(dim), tree.b)
        for (l, leaf) in enumerate(tree.leaves)
            c̃ = child_id(leaf, tree.b)
            if leaf == root(dim)
                continue
            end
            for (ci, c) in enumerate(vertices(leaf, tree.b))
                parent_ = parent(leaf, tree.b)
                parentfaces = faces(parent_, tree.b)
                for (pface_i, pface) in enumerate(parentfaces)
                    if iscenter(c, pface) #hanging node candidate
                        neighbor_candidate = facet_neighbor(parent_, pface_i, tree.b)
                        if inside(tree, neighbor_candidate) #intraoctree branch
                            neighbor_candidate_idx = findfirst(x -> x == neighbor_candidate, tree.leaves)
                            if neighbor_candidate_idx !== nothing
                                neighbor_candidate_faces = faces(neighbor_candidate, tree.b)
                                nf = findfirst(x -> x == pface, neighbor_candidate_faces)
                                hnodes[nodeids[nodeowners[(k, c)]]] = [nodeids[nodeowners[(k, nc)]] for nc in neighbor_candidate_faces[nf]]
                                if dim > 2
                                    vs = vertices(leaf, tree.b)
                                    for ξ in 1:ncorners_face3D
                                        c′ = facetable[pface_i, ξ]
                                        if c′ ∉ (c̃, ci)
                                            neighbor_candidate_edges = edges(neighbor_candidate, tree.b)
                                            ne = findfirst(x -> iscenter(vs[c′], x), neighbor_candidate_edges)
                                            if ne !== nothing
                                                hnodes[nodeids[nodeowners[(k, vs[c′])]]] = [nodeids[nodeowners[(k, ne)]] for ne in neighbor_candidate_edges[ne]]
                                            end
                                        end
                                    end
                                end
                                break
                            end
                        else #interoctree branch
                            #for (ri,rf) in enumerate(rootfaces)
                            #for (pi,pf) in enumerate(parentfaces)
                            facet_neighbor_ = facet_neighborhood[k, _perm[pface_i]]
                            if length(facet_neighbor_) == 0
                                continue
                            end
                            if any(x -> contains_facet(x, pface), rootfaces)
                                k′ = facet_neighbor_[1][1]
                                ri′ = _perminv[facet_neighbor_[1][2]]
                                interoctree_neighbor = transform_facet(forest, k′, ri′, neighbor_candidate)
                                interoctree_neighbor_candidate_idx = findfirst(x -> x == interoctree_neighbor, forest.cells[k′].leaves)
                                if interoctree_neighbor_candidate_idx !== nothing
                                    r = compute_face_orientation(forest, k, pface_i)
                                    neighbor_candidate_faces = faces(neighbor_candidate, forest.cells[k′].b)
                                    transformed_neighbor_faces = faces(interoctree_neighbor, forest.cells[k′].b)

                                    fnodes = transformed_neighbor_faces[ri′]
                                    vs = vertices(leaf, tree.b)
                                    if dim > 2
                                        rotated_ξ = ntuple(i -> rotation_permutation(ri′, pface_i, r, i), ncorners_face3D)
                                    else
                                        rotated_ξ = ntuple(i -> rotation_permutation(r, i), ncorners_face2D)
                                    end
                                    hnodes[nodeids[nodeowners[(k, c)]]] = [nodeids[nodeowners[(k′, fnodes[ξ])]] for ξ in rotated_ξ]

                                    if dim > 2
                                        for ξ in rotated_ξ
                                            c′ = facetable[pface_i, ξ]
                                            if c′ ∉ (c̃, ci)
                                                #neighbor_candidate_edges = edges(interoctree_neighbor,tree.b)
                                                #ne = findfirst(x->iscenter(vs[c′],x),neighbor_candidate_edges)
                                                neighbor_candidate_edges = edges(parent_, tree.b)
                                                ne = findfirst(x -> iscenter(vs[c′], x), neighbor_candidate_edges)
                                                if ne !== nothing
                                                    hnodes[nodeids[nodeowners[(k, vs[c′])]]] = [nodeids[nodeowners[(k, ne)]] for ne in neighbor_candidate_edges[ne]]
                                                end
                                            end
                                        end
                                    end
                                    break
                                end
                            end
                            #end
                        end
                    end
                end
                if dim > 2
                    parentedges = edges(parent_, tree.b)
                    for (pedge_i, pedge) in enumerate(parentedges)
                        if iscenter(c, pedge) #hanging node candidate
                            neighbor_candidate = edge_neighbor(parent_, pedge_i, tree.b)
                            for (ri, re) in enumerate(edges(root(dim), tree.b))
                                edge_neighbor_ = forest.topology.edge_edge_neighbor[k, edge_perm[ri]]
                                if length(edge_neighbor_) == 0
                                    continue
                                end
                                if contains_edge(re, pedge)
                                    k′ = edge_neighbor_[1][1]
                                    ri′ = edge_perm_inv[edge_neighbor_[1][2]]
                                    interoctree_neighbor = transform_edge(forest, k′, ri′, neighbor_candidate, true)
                                    interoctree_neighbor_candidate_idx = findfirst(x -> x == interoctree_neighbor, forest.cells[k′].leaves)
                                    if interoctree_neighbor_candidate_idx !== nothing
                                        neighbor_candidate_edges = edges(neighbor_candidate, forest.cells[k′].b)
                                        transformed_neighbor_edges = edges(interoctree_neighbor, forest.cells[k′].b)
                                        ne = findfirst(x -> iscenter(c, x), neighbor_candidate_edges)
                                        if ne !== nothing
                                            hnodes[nodeids[nodeowners[(k, c)]]] = [nodeids[nodeowners[(k′, nc)]] for nc in transformed_neighbor_edges[ne]]
                                        else
                                            @error "things are messed up for hanging edge constraint interoctree at octree $k edge $(_perm[ri])"
                                        end
                                        break
                                    end
                                end
                            end
                        end
                    end
                end
            end
        end
    end
    return hnodes
end

function balance_corner(forest, k′, c′, o, s)
    o.l == 1 && return # no balancing needed for pivot octant level == 1
    o′ = transform_corner(forest, k′, c′, o, false)
    s′ = transform_corner(forest, k′, c′, s, true) #TODO verify the bool here; I think it's correct
    neighbor_tree = forest.cells[k′]
    return if s′ ∉ neighbor_tree.leaves && parent(s′, neighbor_tree.b) ∉ neighbor_tree.leaves
        if parent(parent(s′, neighbor_tree.b), neighbor_tree.b) ∈ neighbor_tree.leaves
            refine!(neighbor_tree, parent(parent(s′, neighbor_tree.b), neighbor_tree.b))
        end
    end
end

function balance_face(forest, k′, f′, o, s)
    o.l == 1 && return # no balancing needed for pivot octant level == 1
    o′ = transform_facet(forest, k′, f′, o)
    s′ = transform_facet(forest, k′, f′, s)
    neighbor_tree = forest.cells[k′]
    return if s′ ∉ neighbor_tree.leaves && parent(s′, neighbor_tree.b) ∉ neighbor_tree.leaves
        if parent(parent(s′, neighbor_tree.b), neighbor_tree.b) ∈ neighbor_tree.leaves
            refine!(neighbor_tree, parent(parent(s′, neighbor_tree.b), neighbor_tree.b))
        end
    end
end

function balance_edge(forest, k′, e′, o, s)
    o.l == 1 && return # no balancing needed for pivot octant level == 1
    o′ = transform_edge(forest, k′, e′, o, false)
    s′ = transform_edge(forest, k′, e′, s, true)
    neighbor_tree = forest.cells[k′]
    return if s′ ∉ neighbor_tree.leaves && parent(s′, neighbor_tree.b) ∉ neighbor_tree.leaves
        if parent(parent(s′, neighbor_tree.b), neighbor_tree.b) ∈ neighbor_tree.leaves
            refine!(neighbor_tree, parent(parent(s′, neighbor_tree.b), neighbor_tree.b))
        end
    end
end
"""
    balanceforest!(forest)
Algorithm 17 of [BWG2011](@citet)
"""
function balanceforest!(forest::ForestBWG{dim}) where {dim}
    perm_face = dim == 2 ? 𝒱₂_perm : 𝒱₃_perm
    perm_face_inv = dim == 2 ? 𝒱₂_perm_inv : 𝒱₃_perm_inv
    perm_corner = dim == 2 ? node_map₂ : node_map₃
    perm_corner_inv = dim == 2 ? node_map₂_inv : node_map₃_inv
    root_ = root(dim)
    nrefcells = 0
    facet_neighborhood = Ferrite.Ferrite.get_facet_facet_neighborhood(forest)
    while nrefcells - getncells(forest) != 0
        nrefcells = getncells(forest)
        for k in 1:length(forest.cells)
            tree = forest.cells[k]
            rootfaces = faces(root_, tree.b)
            rootedges = dim == 3 ? edges(root_, tree.b) : nothing #TODO change
            rootvertices = vertices(root_, tree.b)
            balanced = balancetree(tree)
            forest.cells[k] = balanced
            for (o_i, o) in enumerate(forest.cells[k].leaves)
                ss = possibleneighbors(o, o.l, tree.b, ; insidetree = false)
                isinside = inside.(ss, (tree.b,))
                notinsideidx = findall(.! isinside)
                if !isempty(notinsideidx)
                    # s_i encodes the type of neighborhood, since it is the index of possibleneighbors
                    # see the docs of this function
                    for s_i in notinsideidx
                        s = ss[s_i]
                        if dim == 2 # need more clever s_i encoding
                            if s_i <= 4 #corner neighbor, only true for 2D see possibleneighbors
                                cc = forest.topology.vertex_vertex_neighbor[k, perm_corner[s_i]]
                                participating_faces_idx = findall(x -> any(x .== s_i), 𝒱₂) #TODO! optimize by using inverted table
                                pivot_faces = faces(o, tree.b)
                                if isempty(cc)
                                    # the branch below checks if we are in a newly introduced topologic tree connection
                                    # by checking if the corner neighbor is only accesible by transforming through a face
                                    # TODO: enable a bool that either activates or deactivates the balancing over a corner
                                    for face_idx in participating_faces_idx
                                        face_idx = face_idx[1]
                                        contained = contains_facet(rootfaces[face_idx], pivot_faces[face_idx])
                                        if contained
                                            fc = facet_neighborhood[k, perm_face[face_idx]]
                                            isempty(fc) && continue
                                            @assert length(fc) == 1
                                            fc = fc[1]
                                            k′, f′ = fc[1], perm_face_inv[fc[2]]
                                            balance_face(forest, k′, f′, o, s)
                                        end
                                    end
                                    continue
                                else
                                    for corner_connection in cc
                                        !(vertex(o, s_i, tree.b) == rootvertices[s_i]) && continue
                                        k′, c′ = corner_connection[1], perm_corner_inv[corner_connection[2]]
                                        balance_corner(forest, k′, c′, o, s)
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
                        else #TODO collapse this 3D branch with more clever s_i encoding into the 2D branch
                            if s_i <= 8 #corner neighbor, only true for 2D see possibleneighbors
                                #TODO a check of new introduced corner neighbors aka corner balancing, see 2D branch
                                cc = forest.topology.vertex_vertex_neighbor[k, perm_corner[s_i]]
                                isempty(cc) && continue
                                for corner_connection in cc
                                    !(vertex(o, s_i, tree.b) == rootvertices[s_i]) && continue
                                    k′, c′ = corner_connection[1], perm_corner_inv[corner_connection[2]]
                                    balance_corner(forest, k′, c′, o, s)
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
                                contained_face = findall(x -> face_contains_edge(x, pivot_edge), rootfaces)
                                if !isempty(contained_face) && !contains_edge(rootedges[s_i], pivot_edge) #check if pivot edge in interior of rootface and not octree edge
                                    for face_idx in contained_face
                                        fc = facet_neighborhood[k, perm_face[face_idx]]
                                        isempty(fc) && continue
                                        @assert length(fc) == 1
                                        fc = fc[1]
                                        k′, f′ = fc[1], perm_face_inv[fc[2]]
                                        balance_face(forest, k′, f′, o, s)
                                    end
                                    continue
                                end
                                isempty(ec) && continue
                                for edge_connection in ec
                                    !contains_edge(rootedges[s_i], pivot_edge) && continue
                                    k′, e′ = edge_connection[1], edge_perm_inv[edge_connection[2]]
                                    balance_edge(forest, k′, e′, o, s)
                                end
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
Algorithm 7 of [SSB2008](@citet)

TODO optimise the unnecessary allocations
"""
function balancetree(tree::OctreeBWG)
    if length(tree.leaves) == 1
        return tree
    end
    W = copy(tree.leaves); P = eltype(tree.leaves)[]; R = eltype(tree.leaves)[]
    for l in tree.b:-1:1 #TODO verify to do this until level 1
        Q = [o for o in W if o.l == l]
        sort!(Q)
        #construct T
        T = eltype(Q)[]
        for x in Q
            if isempty(T)
                push!(T, x)
                continue
            end
            p = parent(x, tree.b)
            if p ∉ parent.(T, (tree.b,))
                push!(T, x)
            end
        end
        for t in T
            push!(R, t, siblings(t, tree.b)...)
            push!(P, possibleneighbors(parent(t, tree.b), l - 1, tree.b)...)
        end
        append!(P, x for x in W if x.l == l - 1)
        filter!(x -> !(x.l == l - 1), W) #don't know why I have to negotiate like this, otherwise behaves weird
        unique!(P)
        append!(W, P)
        empty!(P)
    end
    sort!(R) # be careful with sort, by=morton doesn't work due to ambuigity at max depth level
    linearise!(R, tree.b)
    return OctreeBWG(R, tree.b, tree.nodes)
end

"""
Algorithm 8 of [SSB2008](@citet)

Inverted the algorithm to delete! instead of add incrementally to a new array
"""
function linearise!(leaves::Vector{T}, b) where {T <: OctantBWG}
    inds = [i for i in 1:(length(leaves) - 1) if isancestor(leaves[i], leaves[i + 1], b)]
    return deleteat!(leaves, inds)
end

function siblings(o::OctantBWG, b; include_self = false)
    siblings = children(parent(o, b), b)
    if !include_self
        siblings = filter(x -> x !== o, siblings)
    end
    return siblings
end

"""
    possibleneighbors(o::OctantBWG{2},l,b;insidetree=true)
Returns a list of possible neighbors, where the first four are corner neighbors that are exclusively connected via a corner.
The other four possible neighbors are face neighbors.
"""
function possibleneighbors(o::OctantBWG{2}, l, b; insidetree = true)
    neighbors = ntuple(8) do i
        if i > 4
            j = i - 4
            facet_neighbor(o, j, b)
        else
            corner_neighbor(o, i, b)
        end
    end
    if insidetree
        neighbors = filter(x -> inside(x, b), neighbors)
    end
    return neighbors
end

"""
    possibleneighbors(o::OctantBWG{3},l,b;insidetree=true)
Returns a list of possible neighbors, where the first eight are corner neighbors that are exclusively connected via a corner.
After the first eight corner neighbors, the 6 possible face neighbors follow and after them, the edge neighbors.
"""
function possibleneighbors(o::OctantBWG{3}, l, b; insidetree = true)
    neighbors = ntuple(26) do i
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
    if insidetree
        neighbors = filter(x -> inside(x, b), neighbors)
    end
    return neighbors
end

"""
    isancestor(o1,o2,b) -> Bool
Is o2 an ancestor of o1
"""
function isancestor(o1, o2, b)
    ancestor = false
    l = o2.l - 1
    p = parent(o2, b)
    while l > 0
        if p == o1
            ancestor = true
            break
        end
        l -= 1
        p = parent(p, b)
    end
    return ancestor
end

function contains_facet(mface::Tuple{Tuple{T1, T1}, Tuple{T1, T1}}, sface::Tuple{Tuple{T2, T2}, Tuple{T2, T2}}) where {T1 <: Integer, T2 <: Integer}
    if mface[1][1] == sface[1][1] && mface[2][1] == sface[2][1] # vertical
        return mface[1][2] ≤ sface[1][2] ≤ sface[2][2] ≤ mface[2][2]
    elseif mface[1][2] == sface[1][2] && mface[2][2] == sface[2][2] # horizontal
        return mface[1][1] ≤ sface[1][1] ≤ sface[2][1] ≤ mface[2][1]
    else
        return false
    end
end

# currently checking if sface centroid lies in mface
# TODO should be checked if applicaple in general, I guess yes
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

function center(pivot_face)
    centerpoint = ntuple(i -> 0, length(pivot_face[1]))
    for c in pivot_face
        centerpoint = c .+ centerpoint
    end
    return centerpoint .÷ length(pivot_face)
end

iscenter(c, f) = c == center(f)

function Base.show(io::IO, ::MIME"text/plain", agrid::ForestBWG)
    println(io, "ForestBWG with ")
    println(io, "   $(getncells(agrid)) cells")
    return println(io, "   $(length(agrid.cells)) trees")
end

"""
    child_id(octant::OctantBWG, b::Integer)
Given some OctantBWG `octant` and maximum refinement level `b`, compute the child_id of `octant`
note the following quote from Bursedde et al:
  children are numbered from 0 for the front lower left child,
  to 1 for the front lower right child, to 2 for the back lower left, and so on, with
  4, . . . , 7 being the four children on top of the children 0, . . . , 3.
shifted by 1 due to julia 1 based indexing
"""
function child_id(octant::OctantBWG{dim, N, T}, b::Integer = _maxlevel[2]) where {dim, N, T <: Integer}
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
function ancestor_id(octant::OctantBWG{dim, N, T}, l::Integer, b::Integer = _maxlevel[dim - 1]) where {dim, N, T <: Integer}
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

function parent(octant::OctantBWG{dim, N, T}, b::Integer = _maxlevel[dim - 1]) where {dim, N, T}
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
function descendants(octant::OctantBWG{dim, N, T}, b::Integer = _maxlevel[dim - 1]) where {dim, N, T}
    l1 = b; l2 = b
    h = T(_compute_size(b, octant.l))
    return OctantBWG(l1, octant.xyz), OctantBWG(l2, octant.xyz .+ (h - one(T)))
end

"""
    facet_neighbor(octant::OctantBWG{dim,N,T}, f::T, b::T=_maxlevel[2]) -> OctantBWG{dim,N,T}
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
function facet_neighbor(octant::OctantBWG{3, N, T}, f::T, b::T = _maxlevel[2]) where {N, T <: Integer}
    l = octant.l
    h = T(_compute_size(b, octant.l))
    x, y, z = octant.xyz
    x += ((f == T(1)) ? -h : ((f == T(2)) ? h : zero(T)))
    y += ((f == T(3)) ? -h : ((f == T(4)) ? h : zero(T)))
    z += ((f == T(5)) ? -h : ((f == T(6)) ? h : zero(T)))
    return OctantBWG(l, (x, y, z))
end
function facet_neighbor(octant::OctantBWG{2, N, T}, f::T, b::T = _maxlevel[1]) where {N, T <: Integer}
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
reference_edges_bwg(::Type{Ferrite.RefHypercube{3}}) = (
    (𝒰[1, 1], 𝒰[1, 2]), (𝒰[2, 1], 𝒰[2, 2]), (𝒰[3, 1], 𝒰[3, 2]),
    (𝒰[4, 1], 𝒰[4, 2]), (𝒰[5, 1], 𝒰[5, 2]), (𝒰[6, 1], 𝒰[6, 2]),
    (𝒰[7, 1], 𝒰[7, 2]), (𝒰[8, 1], 𝒰[8, 2]), (𝒰[9, 1], 𝒰[9, 2]),
    (𝒰[10, 1], 𝒰[10, 2]), (𝒰[11, 1], 𝒰[11, 2]), (𝒰[12, 1], 𝒰[12, 2]),
) # TODO maybe remove, unnecessary, can directly use the table
# reference_faces_bwg(::Type{RefHypercube{3}}) = ((1,3,7,5) , (2,4,8,6), (1,2,6,5), (3,4,8,7), (1,2,4,4), (5,6,8,7)) # Note that this does NOT follow P4est order!

"""
    compute_face_orientation(forest::ForestBWG, k::Integer, f::Integer)
Slow implementation for the determination of the face orientation of face `f` from octree `k` following definition 2.1 from [BWG2011](@citet).

TODO use table 3 for more vroom
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
    nodes_f = [forest.cells[k].nodes[n_perm[ni]] for ni in reffacenodes[f]]
    nodes_f′ = [forest.cells[k′].nodes[n_perm[ni]] for ni in reffacenodes[f′]]
    if f > f′
        return T2(findfirst(isequal(nodes_f′[1]), nodes_f) - 1)
    else
        return T2(findfirst(isequal(nodes_f[1]), nodes_f′) - 1)
    end
end

"""
    compute_edge_orientation(forest::ForestBWG, k::Integer, e::Integer)
Slow implementation for the determination of the edge orientation of edge `e` from octree `k` following definition below Table 3 [BWG2011](@citet).

TODO use some table?
"""
function compute_edge_orientation(forest::ForestBWG{<:Any, <:OctreeBWG{3, <:Any, T2}}, k::T1, e::T1) where {T1, T2}
    n_perm = node_map₃
    n_perminv = node_map₃_inv
    e_perm = edge_perm
    e_perminv = edge_perm_inv

    e_ferrite = e_perm[e]
    k′, e′_ferrite = forest.topology.edge_edge_neighbor[k, e_ferrite][1]
    e′ = e_perminv[e′_ferrite]
    refedgenodes = reference_edges_bwg(Ferrite.RefHypercube{3})
    nodes_e = ntuple(i -> forest.cells[k].nodes[n_perm[refedgenodes[e][i]]], length(refedgenodes[e]))
    nodes_e′ = ntuple(i -> forest.cells[k′].nodes[n_perm[refedgenodes[e′][i]]], length(refedgenodes[e′]))
    if nodes_e == nodes_e′
        s = T2(0)
    else
        s = T2(1)
    end
    return s
end

"""
    transform_facet_remote(forest::ForestBWG, k::T1, f::T1, o::OctantBWG{dim,N,T2}) -> OctantBWG{dim,N,T1,T2}
    transform_facet_remote(forest::ForestBWG, f::FacetIndex, o::OctantBWG{dim,N,T2}) -> OctantBWG{dim,N,T2}
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
    s′ = _one - (((f - _one) & _one) ⊻ ((f′ - _one) & _one))
    s = zeros(T2, dim - 1)
    a = zeros(T2, 3) # Coordinate axes of f
    b = zeros(T2, 3) # Coordinate axes of f'
    r = compute_face_orientation(forest, k, f)
    a[3] = (f - _one) ÷ 2; b[3] = (f′ - _one) ÷ 2 # origin and target normal axis
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
    xyz[b[3] + _one] = T2(((_two * ((f′ - _one) & 1)) - _one) * 2^maxlevel + s′ * g + (1 - 2 * s′) * o.xyz[a[3] + _one])
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
    a_sign = _two * ((f - _one) & 1) - _one
    # Coordinate axes of f'
    b = (
        f′ ≤ 2, # tangent
        f′ > 2,  # normal
    )
    # b_sign = _two*(f′ & 1) - _one

    maxlevel = forest.cells[1].b
    depth_offset = 2^maxlevel - 2^(maxlevel - o.l)

    s′ = _one - (((f - _one) & _one) ⊻ ((f′ - _one) & _one)) # arithmetic switch: TODO understand this.

    # xyz = zeros(T2, 2)
    # xyz[a[1] + _one] = T2((r == 0) ? o.xyz[b[1] + _one] : depth_offset - o.xyz[b[1] + _one])
    # xyz[a[2] + _one] = T2(a_sign*2^maxlevel + s′*depth_offset + (1-2*s′)*o.xyz[b[2] + _one])
    # return OctantBWG(o.l,(xyz[1],xyz[2]))

    # We can do this because the permutation and inverse permutation are the same
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
    s′ = _one - (((f - _one) & _one) ⊻ ((f′ - _one) & _one))
    r = compute_face_orientation(forest, k, f)

    # Coordinate axes of f
    a = (
        (f ≤ 2) ? 1 : 0,
        (f ≤ 4) ? 2 : 1,
        (f - _one) ÷ 2,
    )
    a_sign = _two * ((f - _one) & 1) - _one

    # Coordinate axes of f'
    b = if Bool(ℛ[1, f] - _one) ⊻ Bool(ℛ[1, f′] - _one) ⊻ (((r == 0) || (r == 3))) # What is this condition exactly?
        (
            (f′ < 5) ? 2 : 1,
            (f′ < 3) ? 1 : 0,
            (f′ - _one) ÷ 2,
        )
    else
        (
            (f′ < 3) ? 1 : 0,
            (f′ < 5) ? 2 : 1,
            (f′ - _one) ÷ 2,
        )
    end
    # b_sign = _two*(f′ & 1) - _one

    s = if ℛ[f, f′] == 1 + 1 # R is one-based
        (r & 2, r & 1)
    else
        (r & 1, r & 2)
    end
    maxlevel = forest.cells[1].b
    depth_offset = 2^maxlevel - 2^(maxlevel - o.l)
    xyz = zeros(T2, 3)
    xyz[a[1] + _one] = T2((s[1] == 0) ? o.xyz[b[1] + _one] : depth_offset - o.xyz[b[1] + _one])
    xyz[a[2] + _one] = T2((s[2] == 0) ? o.xyz[b[2] + _one] : depth_offset - o.xyz[b[2] + _one])
    xyz[a[3] + _one] = T2(a_sign * 2^maxlevel + s′ * depth_offset + (1 - 2 * s′) * o.xyz[b[3] + _one])
    return OctantBWG(o.l, (xyz[1], xyz[2], xyz[3]))

    # xyz = (
    #     T2((s[1] == 0) ? o.xyz[b[1] + _one] : depth_offset - o.xyz[b[1] + _one]),
    #     T2((s[2] == 0) ? o.xyz[b[2] + _one] : depth_offset - o.xyz[b[2] + _one]),
    #     T2(a_sign*2^maxlevel + s′*depth_offset + (1-2*s′)*o.xyz[b[3] + _one])
    # )
    # return OctantBWG(o.l,(xyz[a[1] + _one],xyz[a[2] + _one],xyz[a[3] + _one]))
end

transform_facet(forest::ForestBWG, f::FacetIndex, oct::OctantBWG) = transform_facet(forest, f[1], f[2], oct)

"""
    transform_corner(forest,k,c',oct,inside::Bool)
    transform_corner(forest,v::VertexIndex,oct,inside::Bool)

Algorithm 12 but with flipped logic in [BWG2011](@citet) to transform corner into different octree coordinate system
Implements flipped logic in the sense of pushing the Octant `oct` through vertex v and stays within octree coordinate system `k`.
"""
function transform_corner(forest::ForestBWG, k::T1, c::T1, oct::OctantBWG{dim, N, T2}, inside::Bool) where {dim, N, T1 <: Integer, T2 <: Integer}
    _perm = dim == 2 ? node_map₂ : node_map₃
    _perminv = dim == 2 ? node_map₂_inv : node_map₃_inv
    k′, c′ = forest.topology.vertex_vertex_neighbor[k, _perm[c]][1]
    k′, c′ = forest.topology.vertex_vertex_neighbor[k′, c′][1] #get the corner connection of neighbor to pivot oct
    c′ = _perminv[c′]
    # make a dispatch that returns only the coordinates?
    b = forest.cells[k].b
    l = oct.l; g = 2^b - 2^(b - l)
    h⁻ = inside ? 0 : -2^(b - l); h⁺ = inside ? g : 2^b
    xyz = ntuple(i -> ((c′ - 1) & 2^(i - 1) == 0) ? h⁻ : h⁺, dim)
    return OctantBWG(l, xyz)
end

transform_corner(forest::ForestBWG, v::VertexIndex, oct::OctantBWG, inside) = transform_corner(forest, v[1], v[2], oct, inside)

"""
    transform_corner_remote(forest,k,c',oct,inside::Bool)
    transform_corner_remote(forest,v::VertexIndex,oct,inside::Bool)

Algorithm 12 in [BWG2011](@citet) to transform corner into different octree coordinate system.
Follows exactly the version of the paper by taking `oct` and looking from the neighbor octree coordinate system (neighboring to `k`,`v`) at `oct`.
"""
function transform_corner_remote(forest::ForestBWG, k::T1, c::T1, oct::OctantBWG{dim, N, T2}, inside::Bool) where {dim, N, T1 <: Integer, T2 <: Integer}
    _perm = dim == 2 ? node_map₂ : node_map₃
    _perminv = dim == 2 ? node_map₂_inv : node_map₃_inv
    k′, c′ = forest.topology.vertex_vertex_neighbor[k, _perm[c]][1]
    c′ = _perminv[c′]
    # make a dispatch that returns only the coordinates?
    b = forest.cells[k].b
    l = oct.l; g = 2^b - 2^(b - l)
    h⁻ = inside ? 0 : -2^(b - l); h⁺ = inside ? g : 2^b
    xyz = ntuple(i -> ((c′ - 1) & 2^(i - 1) == 0) ? h⁻ : h⁺, dim)
    return OctantBWG(l, xyz)
end

transform_corner_remote(forest::ForestBWG, v::VertexIndex, oct::OctantBWG, inside) = transform_corner_remote(forest, v[1], v[2], oct, inside)


"""
    transform_edge_remote(forest,k,e,oct,inside::Bool)
    transform_edge_remote(forest,e::Edgeindex,oct,inside::Bool)

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
        ((e′ - _one) ÷ _four),
        e′ - _one < 4 ? 1 : 0,
        e′ - _one < 8 ? 2 : 1,
    )
    a₀ = ((e - _one) ÷ _four) #subtract 1 based index
    a₀ += _one #add it again
    b = forest.cells[k].b
    l = oct.l; g = _two^b - _two^(b - l)
    h⁻ = inside ? z : -_two^(b - l); h⁺ = inside ? g : _two^b
    s = compute_edge_orientation(forest, k, e)
    xyz = zeros(T2, 3)
    xyz[𝐛[1] + _one] = s * g + (_one - (_two * s)) * oct.xyz[a₀]
    xyz[𝐛[2] + _one] = ((e′ - _one) & 1) == 0 ? h⁻ : h⁺
    xyz[𝐛[3] + _one] = ((e′ - _one) & 2) == 0 ? h⁻ : h⁺
    return OctantBWG(l, (xyz[1], xyz[2], xyz[3]))
end

transform_edge_remote(forest::ForestBWG, e::EdgeIndex, oct::OctantBWG, inside) = transform_edge_remote(forest, e[1], e[2], oct, inside)

"""
    transform_edge(forest,k,e,oct,inside::Bool)
    transform_edge(forest,e::Edgeindex,oct,inside::Bool)

Algorithm 10 in [BWG2011](@citet) to transform cedge into different octree coordinate system but reversed logic.
See `transform_edge_remote` with logic from paper.
In this function we stick to the coordinate system of the pivot tree k and transform an octant through edge e into this k-th octree coordinate system.
"""
function transform_edge(forest::ForestBWG, k::T1, e::T1, oct::OctantBWG{3, N, T2}, inside::Bool) where {N, T1 <: Integer, T2 <: Integer}
    _four = T2(4)
    _one = T2(1)
    _two = T2(2)
    z = zero(T2)
    e_perm = edge_perm
    e_perminv = edge_perm_inv

    e_ferrite = e_perm[e]
    k′, e′_ferrite = forest.topology.edge_edge_neighbor[k, e_ferrite][1]
    k′, e′_ferrite = forest.topology.edge_edge_neighbor[k′, e′_ferrite][1] #get pivot connection from neighbor perspective
    e′ = e_perminv[e′_ferrite]
    #see Algorithm 9, line 18
    𝐛 = (
        ((e′ - _one) ÷ _four),
        e′ - _one < 4 ? 1 : 0,
        e′ - _one < 8 ? 2 : 1,
    )
    a₀ = ((e - _one) ÷ _four) #subtract 1 based index
    a₀ += _one #add it again
    b = forest.cells[k].b
    l = oct.l; g = _two^b - _two^(b - l)
    h⁻ = inside ? z : -_two^(b - l); h⁺ = inside ? g : _two^b
    s = compute_edge_orientation(forest, k′, e′)
    xyz = zeros(T2, 3)
    xyz[𝐛[1] + _one] = s * g + (_one - (_two * s)) * oct.xyz[a₀]
    xyz[𝐛[2] + _one] = ((e′ - _one) & 1) == 0 ? h⁻ : h⁺
    xyz[𝐛[3] + _one] = ((e′ - _one) & 2) == 0 ? h⁻ : h⁺
    return OctantBWG(l, (xyz[1], xyz[2], xyz[3]))
end

transform_edge(forest::ForestBWG, e::EdgeIndex, oct::OctantBWG, inside) = transform_edge(forest, e[1], e[2], oct, inside)

"""
    edge_neighbor(octant::OctantBWG, e::Integer, b::Integer)
Computes the edge neighbor octant which is only connected by the edge `e` to `octant`.
"""
function edge_neighbor(octant::OctantBWG{3, N, T}, e::T, b::T = _maxlevel[2]) where {N, T <: Integer}
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
function corner_neighbor(octant::OctantBWG{3, N, T}, c::T, b::T = _maxlevel[2]) where {N, T <: Integer}
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

function corner_neighbor(octant::OctantBWG{2, N, T}, c::T, b::T = _maxlevel[1]) where {N, T <: Integer}
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

function corner_face_participation(dim::T, c::T) where {T <: Integer}
    if dim == 2
        return 𝒱₂_perm[findall(x -> c ∈ x, eachrow(𝒱₂))]
    else
        return 𝒱₃_perm[findall(x -> c ∈ x, eachrow(𝒱₃))]
    end
end

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

_compute_size(b::Integer, l::Integer) = 2^(b - l)
_maximum_size(b::Integer) = 2^(b)
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
# finds face corner ξ′ in f′ for two associated faces f,f′ in {1,...,6} and their orientation r in {1,...,4}}
_neighbor_corner(f::Int, f′::Int, r::Int, ξ::Int) = 𝒫[𝒬[ℛ[f, f′], r], ξ]

# map given `face` and `ξ` to corner `c`. Need to provide dim for different lookup
function _face_corners(dim::Int, face::Int, ξ::Int)
    if dim == 2
        return 𝒱₂[face, ξ]
    elseif dim == 3
        return 𝒱₃[face, ξ]
    else
        error("No corner-lookup table available")
    end
end

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

# (0,0) non existing connections
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

const 𝒱₂ = [
    1  3
    2  4
    1  2
    3  4
]

const 𝒱₃ = [
    1  3  5  7
    2  4  6  8
    1  2  5  6
    3  4  7  8
    1  2  3  4
    5  6  7  8
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

const 𝒱₃_perm = [
    5
    3
    2
    4
    1
    6
]

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

const ℛ = [
    1  2  2  1  1  2
    3  1  1  2  2  1
    3  1  1  2  2  1
    1  3  3  1  1  2
    1  3  3  1  1  2
    3  1  1  3  3  1
]

const 𝒬 = [
    2  3  6  7
    1  4  5  8
    1  5  4  8
]

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

const opposite_corner_2 = [
    4,
    3,
    2,
    1,
]

const opposite_corner_3 = [
    8,
    7,
    6,
    5,
    4,
    3,
    2,
    1,
]

const opposite_face_2 = [
    2,
    1,
    4,
    3,
]

const opposite_face_3 = [
    2,
    1,
    4,
    3,
    6,
    5,
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
