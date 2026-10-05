# Translation layer between Ferrite's grids and the forest of octrees of PureP4est: the
# `ForestBWG` (coarse mesh geometry and named sets on top of the trees), and the
# materialization of a forest into a `NonConformingGrid`.

"""
    ForestBWG{dim, C <: OctreeBWG, T <: Real} <: PureP4est.AbstractForest{dim, C}
`p4est` adaptive grid implementation based on [BWG2011](@citet)
and [IBWG2015](@citet).

## Constructor
    ForestBWG(grid::AbstractGrid{dim}, b) where dim
Builds an adaptive grid based on a non-adaptive one `grid` and a given max refinement level `b`,
i.e. no leaf may be refined beyond level `b`.

`b` must satisfy `0 ≤ b ≤ 30` in 2D and `0 ≤ b ≤ 19` in 3D, and defaults to those upper bounds
(p4est's `P4EST_MAXLEVEL`/`P8EST_MAXLEVEL`). They are hard limits, not just defaults: a larger
`b` makes octree coordinates exceed the per-axis bit budget of the `UInt64` boundary-table keys
that [`creategrid`](@ref) uses to identify nodes across tree boundaries. An out-of-range `b`
therefore throws a `DomainError` rather than silently producing a grid with wrongly merged nodes.

The forest itself (refinement, balancing, node numbering) is implemented in the
Ferrite-independent PureP4est package; `ForestBWG` adds the coarse mesh's geometry and named
sets, which [`creategrid`](@ref) carries over to the refined grid.
"""
struct ForestBWG{dim, C <: OctreeBWG, T <: Real} <: AbstractForest{dim, C}
    cells::Vector{C}
    connectivity::Connectivity{dim}
    nodes::Vector{Node{dim, T}}
    # Sets
    cellsets::Dict{String, OrderedSet{Int}}
    nodesets::Dict{String, OrderedSet{Int}}
    facetsets::Dict{String, OrderedSet{FacetIndex}}
    vertexsets::Dict{String, OrderedSet{VertexIndex}}
end

PureP4est.trees(forest::ForestBWG) = forest.cells
PureP4est.connectivity(forest::ForestBWG) = forest.connectivity

function ForestBWG(grid::Ferrite.AbstractGrid{dim}, b = DEFAULT_MAXLEVEL[dim]) where {dim}
    cells = getcells(grid)
    C = eltype(cells)
    @assert isconcretetype(C)
    @assert (C == Quadrilateral && dim == 2) || (C == Hexahedron && dim == 3)
    # Ferrite's local numbering of `Quadrilateral`/`Hexahedron` vertices, edges and facets is
    # the counter-clockwise (VTK) one PureP4est expects, so the cells map over as they are.
    tree_vertices = [cell.nodes for cell in cells]
    trees = OctreeBWG{dim, 2^dim}.(tree_vertices, b)
    return ForestBWG(
        trees, Connectivity(tree_vertices), getnodes(grid),
        Ferrite.getcellsets(grid), Ferrite.getnodesets(grid),
        Ferrite.getfacetsets(grid), Ferrite.getvertexsets(grid),
    )
end

Ferrite.getncells(forest::ForestBWG) = nleaves(forest)
Ferrite.getcellsets(forest::ForestBWG) = forest.cellsets
Ferrite.getnodesets(forest::ForestBWG) = forest.nodesets
Ferrite.getfacetsets(forest::ForestBWG) = forest.facetsets
Ferrite.getvertexsets(forest::ForestBWG) = forest.vertexsets
Ferrite.getcellset(forest::ForestBWG, name::String) = forest.cellsets[name]
Ferrite.getnodeset(forest::ForestBWG, name::String) = forest.nodesets[name]
Ferrite.getfacetset(forest::ForestBWG, name::String) = forest.facetsets[name]
Ferrite.getvertexset(forest::ForestBWG, name::String) = forest.vertexsets[name]
# The coarse grid's nodes
Ferrite.getnodes(forest::ForestBWG) = forest.nodes
Ferrite.getnnodes(forest::ForestBWG) = length(forest.nodes)
Ferrite.getspatialdim(::ForestBWG{dim}) where {dim} = dim

"""
    getcells(forest::ForestBWG) -> Vector{OctantBWG}

Collect the leaf octants of all trees of `forest` into a single vector, in ascending cell id
order (tree by tree, Morton order within each tree) — i.e. the octant `getcells(forest)[i]`
materializes into cell `i` of [`creategrid`](@ref)`(forest)`.

!!! warning "Allocates on every call"
    This materializes a fresh vector of all leaves each time it is called — `O(n)` in the
    number of cells. Call it once and reuse the result instead of calling it inside loops.

The returned octants live in the coordinate system of their respective tree, so a scalar
`getcells(forest, cellid)` is deliberately not supported (an octant is not interpretable without
its tree). Take `forest.cells[k].leaves` for tree-local work.
"""
Ferrite.getcells(forest::ForestBWG) = forest_leaves(forest)

function Ferrite.getcells(forest::ForestBWG, cellid::Union{Int, AbstractVector{Int}})
    throw(ArgumentError("getcells(forest, cellid) is not supported: a leaf octant is not interpretable without its tree. Use getcells(forest) for all leaves, or forest.cells[k].leaves for tree-local work."))
end

function Base.show(io::IO, ::MIME"text/plain", forest::ForestBWG)
    println(io, "ForestBWG with ")
    println(io, "   $(getncells(forest)) cells")
    return print(io, "   $(length(forest.cells)) trees")
end

# Ferrite's entity indices for the `(tree, local index)` based inter-tree transforms.
PureP4est.transform_facet_remote(forest::ForestBWG, f::FacetIndex, oct::OctantBWG) = transform_facet_remote(forest, f[1], f[2], oct)
PureP4est.transform_facet(forest::ForestBWG, f::FacetIndex, oct::OctantBWG) = transform_facet(forest, f[1], f[2], oct)
PureP4est.transform_corner(forest::ForestBWG, v::VertexIndex, oct::OctantBWG, inside) = transform_corner(forest, v[1], v[2], oct, inside)
PureP4est.transform_corner_remote(forest::ForestBWG, v::VertexIndex, oct::OctantBWG, inside) = transform_corner_remote(forest, v[1], v[2], oct, inside)
PureP4est.transform_edge_remote(forest::ForestBWG, e::EdgeIndex, oct::OctantBWG, inside) = transform_edge_remote(forest, e[1], e[2], oct, inside)
PureP4est.transform_edge(forest::ForestBWG, e::EdgeIndex, oct::OctantBWG, inside) = transform_edge(forest, e[1], e[2], oct, inside)

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

1. affine-scale the octree coordinate (in `[0, 2^b]^dim`) to the reference cube
   ``\\xi \\in [-1,1]^{dim}`` via ``\\xi = \\texttt{vertex} \\cdot 2/2^b - 1``;
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

# Physical node coordinates from the integer `(tree, coord)` node references of `lnodes`.
function _nodes_from_refs(forest::ForestBWG{dim, C, T}, noderefs::Vector{Tuple{Int32, NTuple{dim, Int32}}}) where {dim, C, T}
    nodes = Vector{Node{dim, T}}(undef, length(noderefs))
    kprev = 0
    corners = _treecorners(forest, 1)
    b = forest.cells[1].b
    @inbounds for (i, (k, xyz)) in enumerate(noderefs)
        if k != kprev
            corners = _treecorners(forest, k)
            b = forest.cells[k].b
            kprev = k
        end
        nodes[i] = Node(_interp_treepoint(corners, b, xyz))
    end
    return nodes
end

"""
    _build_cells(::Type{CT}, E, node_map, ::Val{NV}) -> Vector{CT}

Materialize the cell vector from the element-node matrix `E` (z-order slots): column `gid` is
cell `gid`'s connectivity, reordered to Ferrite's vertex order via `node_map` and wrapped in
cell type `CT` (`Quadrilateral`/`Hexahedron`).
"""
function _build_cells(::Type{CT}, E::Matrix{Int}, node_map, ::Val{NV}) where {CT, NV}
    ncells = size(E, 2)
    cells = Vector{CT}(undef, ncells)
    @inbounds for gid in 1:ncells
        cells[gid] = CT(ntuple(i -> E[node_map[i], gid], Val(NV)))
    end
    return cells
end

"""
    reconstruct_facetsets(forest::ForestBWG) -> Dict{String, OrderedSet{FacetIndex}}

Transfer the macro-mesh facet sets onto the materialized (refined) grid: each original
`FacetIndex` (tree, facet) becomes a `FacetIndex` for every leaf of that tree lying on the root
facet (see `PureP4est.foreach_root_facet_leaf`). This keeps named boundaries (e.g.
Dirichlet/Neumann sets) valid after refinement.
"""
function reconstruct_facetsets(forest::ForestBWG)
    offsets = _element_offsets(forest)
    new_facetsets = typeof(forest.facetsets)()
    for (name, facetset) in forest.facetsets
        new_facetset = typeof(facetset)()
        for (k, f) in facetset
            off = offsets[k]
            foreach_root_facet_leaf(i -> push!(new_facetset, FacetIndex(off + i, f)), forest.cells[k], f)
        end
        new_facetsets[name] = new_facetset
    end
    return new_facetsets
end

"""
    reconstruct_cellsets(forest::ForestBWG) -> Dict{String, OrderedSet{Int}}

Transfer the macro-mesh cell sets onto the materialized (refined) grid: every leaf inherits
the set membership of its tree (macro cell), so each macro cell id in a set is replaced by
the cell ids of that tree's leaves. This keeps named subdomains (e.g. material regions) valid
after refinement.
"""
function reconstruct_cellsets(forest::ForestBWG)
    offsets = _element_offsets(forest)
    new_cellsets = typeof(forest.cellsets)()
    for (name, cellset) in forest.cellsets
        new_cellset = typeof(cellset)()
        for k in cellset
            for leaf_idx in 1:length(forest.cells[k].leaves)
                push!(new_cellset, offsets[k] + leaf_idx)
            end
        end
        new_cellsets[name] = new_cellset
    end
    return new_cellsets
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

Node numbering and hanging-node detection are done on the integer octree coordinates by
`PureP4est.lnodes` ([IBWG2015](@citet) §6). This maps the nodes to physical space
([`_interp_treepoint`](@ref)), builds the cells in Ferrite's vertex order, turns the hanging-node
records into `conformity_info`, and carries the named boundaries and subdomains over
([`reconstruct_facetsets`](@ref), [`reconstruct_cellsets`](@ref)).

Requires a 2:1-balanced forest (see [`balanceforest!`](@ref)) — balance is what guarantees
hanging vertices are simple feature midpoints with non-hanging masters, and it is checked (an
unbalanced forest leaves element vertices without node ids, which raises an error).
"""
function creategrid(forest::ForestBWG{dim}) where {dim}
    ln = lnodes(forest)
    node_map = dim == 2 ? node_map₂ : node_map₃
    celltype = dim == 2 ? Quadrilateral : Hexahedron
    cells = _build_cells(celltype, ln.E, node_map, Val(2^dim))
    hnodes = Dict{Int, Vector{Int}}()
    for (p, m1, m2) in ln.hanging2
        hnodes[p] = [m1, m2]
    end
    for (p, m1, m2, m3, m4) in ln.hanging4
        hnodes[p] = [m1, m2, m3, m4]
    end
    return NonConformingGrid(
        cells, _nodes_from_refs(forest, ln.noderefs);
        conformity_info = hnodes,
        facetsets = reconstruct_facetsets(forest),
        cellsets = reconstruct_cellsets(forest),
    )
end

"""
    facetskeleton(forest::ForestBWG) -> Vector{NTuple{2, FacetIndex}}

Materialize the interior facet skeleton of the *refined* forest: one entry per leaf-level facet
interface, as a pair of `FacetIndex` into the grid returned by [`creategrid`](@ref). For a
conforming interface the pair holds the two equal-size cells sharing the facet; for a
non-conforming (hanging) interface each **fine subfacet** gets its own pair, fine side first and
the coarse cell's facet second. Domain-boundary facets are not part of the skeleton. See
`PureP4est.facetskeleton`.

Requires a 2:1-balanced forest (see [`balanceforest!`](@ref)), like [`creategrid`](@ref).
"""
Ferrite.facetskeleton(forest::ForestBWG) = PureP4est.facetskeleton(forest, FacetIndex)
