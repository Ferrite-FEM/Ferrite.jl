# The coarse mesh a forest lives on: its topology (`Connectivity`), the `AbstractForest`
# interface, and `Forest`, the minimal forest.

"""
    NeighborTable

`table[k, i]` is the (possibly empty) list of `(tree, local index)` pairs connected to local
entity `i` (vertex, edge or facet) of tree `k`. Stored compressed: one flat `data` vector and a
range per `(k, i)`.
"""
struct NeighborTable
    data::Vector{NTuple{2, Int}}
    ranges::Matrix{UnitRange{Int}}
end

Base.@propagate_inbounds Base.getindex(t::NeighborTable, k::Integer, i::Integer) = view(t.data, t.ranges[k, i])
Base.size(t::NeighborTable) = size(t.ranges)
Base.size(t::NeighborTable, d::Integer) = size(t.ranges, d)
Base.:(==)(a::NeighborTable, b::NeighborTable) =
    size(a) == size(b) && all(i -> a[Tuple(i)...] == b[Tuple(i)...], CartesianIndices(a.ranges))

"""
    NeighborTable(f, ntrees, nlocal)

Build a table by calling `f(k, i)` for each tree `k ∈ 1:ntrees` and local entity
`i ∈ 1:nlocal`; `f` returns an iterable of `(tree, local index)` pairs.
"""
function NeighborTable(f::F, ntrees::Integer, nlocal::Integer) where {F}
    data = NTuple{2, Int}[]
    ranges = Matrix{UnitRange{Int}}(undef, ntrees, nlocal)
    for i in 1:nlocal, k in 1:ntrees
        lo = length(data) + 1
        for (k′, i′) in f(k, i)
            push!(data, (Int(k′), Int(i′)))
        end
        ranges[k, i] = lo:length(data)
    end
    return NeighborTable(data, ranges)
end

# Local vertices of the facets and edges of a tree, in the counter-clockwise (VTK) numbering
# of the coarse quadrilateral/hexahedron.
const QUAD_FACETS = ((1, 2), (2, 3), (3, 4), (4, 1))
const HEX_FACETS = ((1, 4, 3, 2), (1, 2, 6, 5), (2, 3, 7, 6), (3, 4, 8, 7), (1, 5, 8, 4), (5, 6, 7, 8))
const HEX_EDGES = ((1, 2), (2, 3), (3, 4), (4, 1), (5, 6), (6, 7), (7, 8), (8, 5), (1, 5), (2, 6), (3, 7), (4, 8))

"""
    Connectivity{dim}

Topology of the coarse mesh whose cells are the trees of a forest (p4est's
`p4est_connectivity_t`). Local vertices, edges and facets of a tree use the counter-clockwise
(VTK) numbering of the quadrilateral/hexahedron, see `QUAD_FACETS`, `HEX_FACETS` and
`HEX_EDGES`; the z-order used inside the octrees is translated internally.

- `facet_neighbors[k, f]`: the `(k′, f′)` sharing facet `f` of tree `k` (at most one).
- `edge_neighbors[k, e]` (3D only, no columns in 2D): the `(k′, e′)` sharing edge `e` of tree
  `k` that share no facet with it.
- `vertex_neighbors[k, v]`: the `(k′, v′)` sharing vertex `v` of tree `k` that share neither a
  facet nor an edge with it.

The coarse vertex ids of each tree, from which the relative orientation of neighbouring trees is
deduced, are stored in the trees themselves (the `nodes` of each `OctreeBWG`).

    Connectivity(tree_vertices::AbstractVector{NTuple{N, Int}})

Deduce the connectivity of a conforming coarse mesh of quadrilaterals (`N = 4`) or hexahedra
(`N = 8`) from the global vertex ids of each tree (counter-clockwise/VTK order): two trees are
facet, edge or vertex neighbours if they share the vertices of a facet, of an edge, or only
single vertices. Degenerate cells (repeated vertex ids) are rejected.
"""
struct Connectivity{dim}
    facet_neighbors::NeighborTable
    edge_neighbors::NeighborTable
    vertex_neighbors::NeighborTable
end

function Connectivity(tree_vertices::AbstractVector{NTuple{N, Int}}) where {N}
    N == 4 || N == 8 || throw(ArgumentError("expected 4 (quadrilateral) or 8 (hexahedron) vertices per tree, got $N"))
    return N == 4 ? _connectivity(tree_vertices, Val(2), QUAD_FACETS, ()) :
        _connectivity(tree_vertices, Val(3), HEX_FACETS, HEX_EDGES)
end

function _connectivity(tree_vertices::AbstractVector{NTuple{N, Int}}, ::Val{dim}, facets, edges) where {N, dim}
    ntrees = length(tree_vertices)
    for (k, vs) in enumerate(tree_vertices)
        allunique(vs) || throw(ArgumentError("tree $k has repeated vertex ids $vs; degenerate coarse cells are not supported"))
        all(>(0), vs) || throw(ArgumentError("tree $k has non-positive vertex ids $vs"))
    end
    vertex_to_tree = _vertex_to_tree(tree_vertices)
    # (tree, local entity, neighbour tree, neighbour local entity), in order of discovery
    fentries = sizehint!(NTuple{4, Int}[], ntrees * length(facets))
    eentries = sizehint!(NTuple{4, Int}[], ntrees * length(edges))
    ventries = sizehint!(NTuple{4, Int}[], ntrees * N)
    lastseen = zeros(Int, ntrees)
    for (k, vs) in enumerate(tree_vertices)
        # Neighbour trees in order of discovery over the tree's vertices (as `ExclusiveTopology`).
        for v in vs, k′ in vertex_to_tree[v]
            (k′ == k || lastseen[k′] == k) && continue
            lastseen[k′] = k
            vs′ = tree_vertices[k′]
            # The number of shared vertices tells which entity can be shared (as in
            # `ExclusiveTopology`); fall back to lower-dimensional ones on degenerate meshes.
            nshared = _num_shared(vs, vs′)
            nshared >= length(facets[1]) && _add_entity_neighbor!(fentries, k, vs, k′, vs′, facets) && continue
            nshared >= 2 && _add_entity_neighbor!(eentries, k, vs, k′, vs′, edges) && continue
            for (c, v) in enumerate(vs)
                c′ = findfirst(==(v), vs′)
                c′ === nothing || push!(ventries, (k, c, k′, c′))
            end
        end
    end
    return Connectivity{dim}(
        _neighbor_table(fentries, ntrees, length(facets)),
        _neighbor_table(eentries, ntrees, length(edges)),
        _neighbor_table(ventries, ntrees, N),
    )
end

# Vertex -> trees touching it, in ascending tree order (compressed, two passes).
function _vertex_to_tree(tree_vertices::AbstractVector{<:NTuple})
    nverts = maximum(maximum, tree_vertices; init = 0)
    offsets = zeros(Int, nverts + 1)
    for vs in tree_vertices, v in vs
        offsets[v + 1] += 1
    end
    offsets[1] = 1
    cumsum!(offsets, offsets)
    data = Vector{Int}(undef, offsets[end] - 1)
    next = offsets[1:(end - 1)]
    for (k, vs) in enumerate(tree_vertices), v in vs
        data[next[v]] = k
        next[v] += 1
    end
    return [view(data, offsets[v]:(offsets[v + 1] - 1)) for v in 1:nverts]
end

function _num_shared(vs::NTuple{N, Int}, vs′::NTuple{N, Int}) where {N}
    n = 0
    for v in vs, v′ in vs′
        n += Int(v == v′)
    end
    return n
end

# If trees `k` and `k′` share all vertices of one of `entities` (local vertex tuples), record
# the first such entity pair and return `true`.
function _add_entity_neighbor!(entries, k, vs, k′, vs′, entities)
    for (e, ev) in enumerate(entities)
        gv = map(i -> vs[i], ev)
        all(in(vs′), gv) || continue
        for (e′, ev′) in enumerate(entities)
            if all(i -> vs′[i] in gv, ev′)
                push!(entries, (k, e, k′, e′))
                return true
            end
        end
    end
    return false
end

# Compress `(k, i, k′, i′)` entries into a `NeighborTable` with a stable counting sort, keeping
# the order of discovery within each `(k, i)`.
function _neighbor_table(entries::Vector{NTuple{4, Int}}, ntrees::Int, nlocal::Int)
    lin = LinearIndices((ntrees, nlocal))
    counts = zeros(Int, ntrees * nlocal + 1)
    for (k, i, _, _) in entries
        counts[lin[k, i] + 1] += 1
    end
    counts[1] = 1
    cumsum!(counts, counts)
    ranges = Matrix{UnitRange{Int}}(undef, ntrees, nlocal)
    for j in 1:(ntrees * nlocal)
        ranges[j] = counts[j]:(counts[j + 1] - 1)
    end
    data = Vector{NTuple{2, Int}}(undef, length(entries))
    for (k, i, k′, i′) in entries
        j = lin[k, i]
        data[counts[j]] = (k′, i′)
        counts[j] += 1
    end
    return NeighborTable(data, ranges)
end

"""
    AbstractForest{dim, C <: OctreeBWG}

A forest of octrees over a coarse mesh. Subtypes implement

- `trees(forest)::Vector{C}`: the trees, one per coarse cell,
- `connectivity(forest)::Connectivity{dim}`: the coarse-mesh topology,

and may carry further data (geometry, named sets, ...). [`Forest`](@ref) is the minimal
implementation.
"""
abstract type AbstractForest{dim, C <: OctreeBWG} end

"""
    trees(forest::AbstractForest) -> Vector{<:OctreeBWG}

The trees of `forest`, one per coarse cell.
"""
function trees end

"""
    connectivity(forest::AbstractForest) -> Connectivity

The coarse-mesh topology of `forest`.
"""
function connectivity end

"""
    Forest(trees::Vector{<:OctreeBWG}, connectivity::Connectivity)
    Forest(tree_vertices::Vector{NTuple{N, Int}}, b = DEFAULT_MAXLEVEL[dim])

Minimal [`AbstractForest`](@ref): just the trees and the coarse-mesh topology. The second
form starts from one unrefined tree per coarse cell with maximum refinement level `b`, given
the cells' global vertex ids in counter-clockwise (VTK) order, and deduces the connectivity
from them (see [`Connectivity`](@ref)).
"""
struct Forest{dim, C <: OctreeBWG} <: AbstractForest{dim, C}
    trees::Vector{C}
    connectivity::Connectivity{dim}
end

trees(f::Forest) = f.trees
connectivity(f::Forest) = f.connectivity

function Forest(tree_vertices::AbstractVector{NTuple{N, Int}}, b::Integer = DEFAULT_MAXLEVEL[N == 4 ? 2 : 3]) where {N}
    conn = Connectivity(tree_vertices)
    dim = N == 4 ? 2 : 3
    return Forest(OctreeBWG{dim, N}.(tree_vertices, b), conn)
end
