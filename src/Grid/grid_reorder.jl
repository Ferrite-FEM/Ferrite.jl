"""
    reorder_cells!(grid::Grid, perm::AbstractVector{Int}) -> grid

Reorder the cells of `grid` according to the permutation `perm`, such that
`grid.cells[i]` after the call equals `old_cells[perm[i]]` before.

All cell-indexed data structures are updated consistently:
- `cellsets` — cell IDs are mapped via the inverse permutation.
- `facetsets` — the cell-ID component of each `FacetIndex` is remapped.
- `vertexsets` — the cell-ID component of each `VertexIndex` is remapped.
- `nodesets` — unchanged (node IDs are independent of cell ordering).

!!! warning
    Any structure built from the grid after the last call to `reorder_cells!`
    (e.g. `ExclusiveTopology`, `DofHandler`, `ConstraintHandler`) becomes
    stale and must be rebuilt from scratch.
"""
function reorder_cells!(grid::Grid, perm::AbstractVector{Int})
    if !(isperm(perm) && length(perm) == getncells(grid))
        throw(ArgumentError("perm must be a permutation of 1:$(getncells(grid))"))
    end
    iperm = invperm(perm)

    grid.cells = grid.cells[perm]

    for (name, cellset) in grid.cellsets
        new_ids = sort!([iperm[id] for id in cellset])
        grid.cellsets[name] = OrderedSet(new_ids)
    end

    for (name, facetset) in grid.facetsets
        new_facets = [FacetIndex(iperm[fi[1]], fi[2]) for fi in facetset]
        sort!(new_facets; by = fi -> fi.idx)
        grid.facetsets[name] = OrderedSet(new_facets)
    end

    for (name, vertexset) in grid.vertexsets
        new_vertices = [VertexIndex(iperm[vi[1]], vi[2]) for vi in vertexset]
        sort!(new_vertices; by = vi -> vi.idx)
        grid.vertexsets[name] = OrderedSet(new_vertices)
    end

    return grid
end

"""
    compute_sfc_ordering(grid::AbstractGrid) -> Vector{Int}

Return a permutation of cell indices based on a space-filling curve (SFC) ordering.

The algorithm builds an adaptive octree (quadtree in 2D, binary tree in 1D) over the
bounding box of `grid`. Starting from the root, any leaf node that contains more than one
element — determined by the element's centroid — is subdivided into `2^dim` children.
Subdivision continues until every leaf holds at most one element. The ordering is then
obtained by traversing the leaves in Morton (Z-curve) order.

The returned permutation `perm` satisfies `perm[new_pos] = old_cell_id`, so that
`reorder_cells!(grid, compute_sfc_ordering(grid))` reorders the grid in SFC order.

!!! note
    Cell centroids are approximated as the arithmetic mean of all node coordinates of
    the cell.  For higher-order elements this is an approximation, not the exact centroid,
    but it is sufficient for spatial partitioning purposes.
"""
function compute_sfc_ordering(grid::AbstractGrid)
    n = getncells(grid)
    n == 0 && return Int[]
    centers = [cellcenter(grid, i) for i in 1:n]
    lo, hi = bounding_box(grid)
    ordering = Int[]
    sizehint!(ordering, n)
    compute_morton_ordering!(ordering, centers, lo, hi, collect(1:n), 0)
    return ordering
end

# Compute the centroid of cell `cell_id` as the mean of its node coordinates.
function cellcenter(grid::AbstractGrid, cell_id::Int)
    node_ids = get_node_ids(getcells(grid, cell_id))
    s = sum(get_node_coordinate(grid, nid) for nid in node_ids)
    return s / length(node_ids)
end

# Recursive Morton-order (Z-curve) traversal of an adaptive octree.
# `centers[id]` is the centroid of element `id`.
# `lo` / `hi` define the current axis-aligned box.
# `cell_ids` is the set of element IDs assigned to this box.
# `depth` is used as a safety guard against infinite recursion.
function compute_morton_ordering!(
        ordering::Vector{Int},
        centers::Vector{Vec{dim, T}},
        lo::Vec{dim, T},
        hi::Vec{dim, T},
        cell_ids::Vector{Int},
        depth::Int,
    ) where {dim, T}
    if length(cell_ids) <= 1
        append!(ordering, cell_ids)
        return
    end

    mid = (lo + hi) / 2
    n_children = 1 << dim  # 2^dim
    children = [Int[] for _ in 1:n_children]

    for id in cell_ids
        c = centers[id]
        octant = 0
        for d in 1:dim
            if c[d] >= mid[d]
                octant += 1 << (d - 1)
            end
        end
        push!(children[octant + 1], id)
    end

    # Detect degenerate split: if all elements fall in one child (e.g. coincident
    # centroids), further subdivision will make no progress — stop here.
    if any(c -> length(c) == length(cell_ids), children)
        append!(ordering, cell_ids)
        @warn "Unable to compute SFC: Cell centroids coincide."
        return
    end

    # Visit children in Morton (Z-curve) order: child index i encodes which side of
    # mid each dimension is on via bit (d-1) of (i-1).
    for child_idx in 1:n_children
        isempty(children[child_idx]) && continue
        child_lo = Vec{dim}(d -> isodd((child_idx - 1) >> (d - 1)) ? mid[d] : lo[d])
        child_hi = Vec{dim}(d -> isodd((child_idx - 1) >> (d - 1)) ? hi[d] : mid[d])
        _sfc_ordering!(ordering, centers, child_lo, child_hi, children[child_idx], depth + 1)
    end
    return
end
