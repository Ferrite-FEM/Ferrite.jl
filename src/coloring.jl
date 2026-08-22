# Split `1:n` into at most `maxchunks` contiguous ranges of similar size.
function _color_chunks(n::Int, maxchunks::Int)
    return Iterators.partition(1:n, max(1, cld(n, maxchunks)))
end

# We need a sorted collection without duplicates. The default case (`cells = 1:ncells`)
# fulfills this already.
_sorted_cellvec(cellset::AbstractUnitRange{Int}) = cellset
_sorted_cellvec(cellset) = unique!(sort!(collect(Int, cellset)))

# Sort the gathered candidates and append them, deduplicated, to `buf`, counting into
# `colcount[cellid]`. The candidate list must not contain `cellid` itself.
function _append_candidates_sorted_unique!(buf, colcount, candidates, cellid)
    # QuickSort sorts in-place without allocations. The default algorithm dispatches
    # to counting/radix sort which allocates a workspace on each call so we use QuickSort
    # which performs better here.
    sort!(candidates; alg = QuickSort)
    # A candidate may occur multiple times (e.g. a neighbor sharing k nodes with the cell
    # occurs k times). After sorting, duplicates are adjacent and can be skipped by
    # comparing with the previous entry (a unique! fused with the counting).
    prev = 0
    for cell_neighbour in candidates
        if cell_neighbour != prev
            push!(buf, cell_neighbour)
            colcount[cellid] += 1
            prev = cell_neighbour
        end
    end
    return buf
end

function _gather_neighbor_chunk!(colcount, grid, cellvec, chunk, nodeptr, nodecells)
    buf = Int[]
    candidates = Int[]
    # Loop over cells in the chunk
    for i in chunk
        cellid = cellvec[i]
        empty!(candidates)
        # Loop over nodes of the cell
        for v in get_node_ids(getcells(grid, cellid))
            # Loop over the cells connected to this node
            for r in nodeptr[v]:(nodeptr[v + 1] - 1)
                cell_neighbour = nodecells[r]
                cell_neighbour == cellid || push!(candidates, cell_neighbour)
            end
        end
        _append_candidates_sorted_unique!(buf, colcount, candidates, cellid)
    end
    return buf
end

# Map from node id to the cells in `cellvec` containing it, in CSR-like form.
function _build_node_to_cell_map(grid::AbstractGrid, cellvec)
    nnodes = getnnodes(grid)
    nodeptr = zeros(Int, nnodes + 1)
    nodeptr[1] = 1
    for cellid in cellvec
        for v in get_node_ids(getcells(grid, cellid))
            nodeptr[v + 1] += 1
        end
    end
    for i in 2:(nnodes + 1)
        nodeptr[i] += nodeptr[i - 1]
    end
    nodecells = Vector{Int}(undef, nodeptr[end] - 1)
    cursor = copy(nodeptr)
    for cellid in cellvec
        for v in get_node_ids(getcells(grid, cellid))
            nodecells[cursor[v]] = cellid
            cursor[v] += 1
        end
    end
    return nodeptr, nodecells
end

# Run `gather!(counts, chunk) -> buf` over contiguous chunks of `cellvec` in parallel and
# assemble the per-chunk buffers into a CSR-like (ptr, adj) structure over all cells.
# Since `cellvec` is sorted and the chunks are contiguous, each chunk's buffer is a
# contiguous range of the output, which makes the result independent of the number of
# threads.
function _chunked_gather!(gather!::F, ncells::Int, cellvec) where {F}
    chunks = collect(_color_chunks(length(cellvec), Threads.nthreads()))
    counts = zeros(Int, ncells)
    buffers = Vector{Vector{Int}}(undef, length(chunks))
    @sync for (ci, chunk) in enumerate(chunks)
        Threads.@spawn begin
            buffers[$ci] = gather!(counts, $chunk)
        end
    end
    ptr = Vector{Int}(undef, ncells + 1)
    ptr[1] = 1
    for c in 1:ncells
        ptr[c + 1] = ptr[c] + counts[c]
    end
    adj = Vector{Int}(undef, ptr[end] - 1)
    @assert length(adj) == sum(length, buffers; init = 0)
    # This loop is trivially parallelizable but it is just a memcpy so there is no
    # measurable speedup from doing so.
    for (ci, chunk) in enumerate(chunks)
        buf = buffers[ci]
        copyto!(adj, ptr[cellvec[first(chunk)]], buf, 1, length(buf))
    end
    return ptr, adj
end

# Incidence matrix for element connections in the grid
function create_incidence_matrix(grid::AbstractGrid, cellset = 1:getncells(grid))
    ncells = getncells(grid)
    cellvec = _sorted_cellvec(cellset)
    if isempty(cellvec)
        return SparseArrays.spzeros(Bool, Int, ncells, ncells)
    end

    nodeptr, nodecells = _build_node_to_cell_map(grid, cellvec)

    # For each cell, gather the unique cells sharing at least one node with it. The
    # chunked gather makes the result independent of the number of threads (see
    # `_chunked_gather!`).
    colptr, rowval = _chunked_gather!(ncells, cellvec) do counts, chunk
        _gather_neighbor_chunk!(counts, grid, cellvec, chunk, nodeptr, nodecells)
    end
    nzval = fill(true, length(rowval))
    return SparseMatrixCSC(ncells, ncells, colptr, rowval, nzval)
end

# Greedy coloring of the cells in `cells` such that no two connected cells (for which
# `is_member` returns `true` for both) have the same color. `cell_colors` (indexed by cell
# id, zeroed for the cells to color) and `occupied` are scratch data.
function _greedy_coloring!(cell_colors::Vector{Int}, occupied::Vector{Bool}, incidence_matrix, cells, is_member::F) where {F}
    final_colors = Vector{Int}[]
    total_colors = 0
    for cellid in cells
        for i in 1:total_colors
            occupied[i] = false
        end
        # loop over neighbors
        for r in nzrange(incidence_matrix, cellid)
            cell_neighbour = incidence_matrix.rowval[r]
            is_member(cell_neighbour) || continue # Only care about the given subset
            color = cell_colors[cell_neighbour]
            if color != 0
                occupied[color] = true
            end
        end

        # occupied now contains all the colors we are not allowed to use
        free_color = 0
        for attempt_color in 1:total_colors
            if !occupied[attempt_color]
                free_color = attempt_color
                break
            end
        end
        if free_color == 0 # no free color found, need to bump max colors
            total_colors += 1
            free_color = total_colors
            total_colors > length(occupied) && push!(occupied, false)
            push!(final_colors, Int[])
        end
        cell_colors[cellid] = free_color
        push!(final_colors[free_color], cellid)
    end
    return final_colors
end

# Greedy algorithm for coloring a grid such that no two cells with the same node
# have the same color
function greedy_coloring(incidence_matrix, cells = 1:size(incidence_matrix, 1))
    ncells = size(incidence_matrix, 1)
    cell_colors = zeros(Int, ncells)
    occupied = Bool[]
    if cells isa AbstractUnitRange{Int}
        return _greedy_coloring!(cell_colors, occupied, incidence_matrix, cells, c -> c in cells)
    else
        mask = zeros(Bool, ncells)
        for c in cells
            mask[c] = true
        end
        return _greedy_coloring!(cell_colors, occupied, incidence_matrix, cells, c -> mask[c])
    end
end

function _color_zone_chunk!(zone_colors, chunk, zones, zone_of, cell_colors, incidence_matrix)
    occupied = Bool[]
    for zi in chunk
        zone_colors[zi] = _greedy_coloring!(cell_colors, occupied, incidence_matrix, zones[zi], c -> zone_of[c] == zi)
    end
    return
end

# See Appendix A in https://www.math.colostate.edu/%7Ebangerth/publications/2013-pattern.pdf
function workstream_coloring(incidence_matrix, cellset)

    cellvec = _sorted_cellvec(cellset)
    if length(cellvec) == 0
        return Vector{Int}[]
    elseif length(cellvec) == 1
        return Vector{Int}[Int[first(cellvec)]]
    end
    ncells = size(incidence_matrix, 1)

    ###################
    # 1. Partitioning #
    ###################
    # Note: the incidence matrix is assumed to be created with the same cellset, so all
    # neighbors found through it are members of the cellset.
    zone_of = zeros(Int, ncells) # Zero represents no zone assigned yet
    zones = Vector{Int}[]
    n_visited = 0
    seed_idx = 1
    while n_visited < length(cellvec)
        ## Zone 1: Just the first unvisited element (starts a new part of the cellset,
        ## disconnected from the already zoned cells)
        while zone_of[cellvec[seed_idx]] != 0
            seed_idx += 1
        end
        seed = cellvec[seed_idx]
        push!(zones, Int[seed])
        zone_of[seed] = length(zones)
        n_visited += 1
        ## Zone N: All elements with connection to elements in Zone N-1
        while true
            s = Int[]
            Z = length(zones) + 1
            for c in zones[end]
                for r in nzrange(incidence_matrix, c)
                    cell_neighbour = incidence_matrix.rowval[r]
                    if zone_of[cell_neighbour] == 0
                        zone_of[cell_neighbour] = Z
                        push!(s, cell_neighbour)
                    end
                end
            end
            isempty(s) && break # no more cells connected to previous zone
            push!(zones, s)
            n_visited += length(s)
        end
    end

    ###############
    # 2. Coloring #
    ###############
    # TODO: The reference uses DSATUR algorithm instead of greedy
    # Zones are colored in parallel: cells in a zone only ever compare colors with cells
    # in the same zone, so each task reads and writes a disjoint part of `cell_colors`.
    # Zone sizes vary wildly (they are levels of a breadth-first traversal), so
    # oversubscribe with 4x more tasks than threads to get some load balancing from the
    # scheduler.
    zone_colors = Vector{Vector{Vector{Int}}}(undef, length(zones))
    cell_colors = zeros(Int, ncells)
    @sync for chunk in _color_chunks(length(zones), 4 * Threads.nthreads())
        Threads.@spawn _color_zone_chunk!(zone_colors, $chunk, zones, zone_of, cell_colors, incidence_matrix)
    end

    ################
    # 3. Gathering #
    ################
    Nodd, Zodd = findmax(x -> isodd(x) ? length(zone_colors[x]) : typemin(Int), 1:length(zone_colors))
    Neven, Zeven = findmax(x -> iseven(x) ? length(zone_colors[x]) : typemin(Int), 1:length(zone_colors))
    N = Nodd + Neven
    final_colors = append!(zone_colors[Zodd], zone_colors[Zeven]) # Reuse these for output
    color_sizes = map(length, final_colors)
    used_for_zone = Set{Int}()
    for Z in 1:length(zone_colors)
        (Z == Zodd || Z == Zeven) && continue
        zone_color_vectors = zone_colors[Z]
        odd = isodd(Z)

        empty!(used_for_zone)

        for local_color in sortperm(zone_color_vectors; by = length, rev = true)
            cond = odd ? (x -> x <= Nodd) : (x -> x > Nodd)
            _, global_color = findmin(x -> (cond(x) && x ∉ used_for_zone) ? color_sizes[x] : typemax(Int), 1:N)
            push!(used_for_zone, global_color)
            append!(final_colors[global_color], zone_color_vectors[local_color])
            color_sizes[global_color] = length(final_colors[global_color])
        end
    end

    # Maybe nice to sort?
    foreach(sort!, final_colors)

    return final_colors
end

@enumx ColoringAlgorithm Greedy WorkStream
# For backwards compatibility
const GREEDY = ColoringAlgorithm.Greedy
const WORKSTREAM = ColoringAlgorithm.WorkStream

"""
    create_coloring(g::Grid, cellset = 1:getncells(g); alg::ColoringAlgorithm)

Create a coloring of the cells in grid `g` such that no neighboring cells
have the same color. If only a subset of cells should be colored, the cells to color can be specified by `cellset`.

Returns a vector of vectors with cell indexes, e.g.:

```julia
ret = [
   [1, 3, 5, 10, ...], # cells for color 1
   [2, 4, 6, 12, ...], # cells for color 2
]
```

Two different algorithms are available, specified with the `alg` keyword argument:
 - `alg = ColoringAlgorithm.WorkStream` (default): Three step algorithm from
   Turcksin et al. [Turcksin2016](@cite), albeit with a greedy coloring in the second step. Generally results in more colors than
   `ColoringAlgorithm.Greedy`, however the cells are more equally distributed among the colors.
 - `alg = ColoringAlgorithm.Greedy`: greedy algorithm that works well for structured quadrilateral grids such as
   e.g. quadrilateral grids from `generate_grid`.

The resulting colors can be visualized using [`Ferrite.write_cell_colors`](@ref).

!!! note "Cell to color mapping"
    In a previous version of Ferrite this function returned a dictionary mapping
    cell ID to color numbers as the first argument. If you need this mapping you
    can create it using the following construct:
    ```julia
    colors = create_coloring(...)
    cell_colormap = Dict{Int,Int}(
        cellid => color for (color, cellids) in enumerate(final_colors) for cellid in cellids
    )
    ```

# References
 - [Turcksin2016](@cite) Turcksin et al. ACM Trans. Math. Softw. 43 (2016).
"""
function create_coloring(g::AbstractGrid, cellset = 1:getncells(g); alg::ColoringAlgorithm.T = ColoringAlgorithm.WorkStream)
    incidence_matrix = create_incidence_matrix(g, cellset)
    return _color_incidence_matrix(incidence_matrix, cellset, alg)
end

function _color_incidence_matrix(incidence_matrix, cellset, alg::ColoringAlgorithm.T)
    if alg === ColoringAlgorithm.WorkStream
        return workstream_coloring(incidence_matrix, cellset)
    elseif alg === ColoringAlgorithm.Greedy
        return greedy_coloring(incidence_matrix, cellset)
    else
        error("impossible")
    end
end

######################
# Interface coloring #
######################

# Enumerate the interfaces of the grid -- pairs of facets `(facet_here, facet_there)` --
# restricted to interfaces where both cells are in the cellset, in the same order as
# `InterfaceIterator` visits them.
function _enumerate_interfaces(grid::AbstractGrid, topology, cellvec)
    neighborhood = get_facet_facet_neighborhood(topology, grid)
    interfaces = NTuple{2, FacetIndex}[]
    for facet_a in facetskeleton(topology, grid)
        neighbors = neighborhood[facet_a[1], facet_a[2]]
        isempty(neighbors) && continue
        length(neighbors) > 1 && error("multiple neighboring facets not supported yet")
        facet_b = neighbors[1]
        (insorted(facet_a[1], cellvec) && insorted(facet_b[1], cellvec)) || continue
        # Canonicalize to FacetIndex: depending on the grid dimension the skeleton and
        # neighborhood are in terms of e.g. EdgeIndex.
        push!(interfaces, (FacetIndex(facet_a[1], facet_a[2]), FacetIndex(facet_b[1], facet_b[2])))
    end
    return interfaces
end

# Map from cell id to the ids (indices into `interfaces`) of the interfaces incident to
# it, in CSR-like form. Since interfaces are enumerated in order the per-cell lists are
# sorted.
function _cell_to_interface_map(ncells::Int, interfaces)
    ptr = zeros(Int, ncells + 1)
    ptr[1] = 1
    for (facet_a, facet_b) in interfaces
        ptr[facet_a[1] + 1] += 1
        ptr[facet_b[1] + 1] += 1
    end
    for i in 2:(ncells + 1)
        ptr[i] += ptr[i - 1]
    end
    adj = Vector{Int}(undef, ptr[end] - 1)
    cursor = copy(ptr)
    for (k, (facet_a, facet_b)) in pairs(interfaces)
        for c in (facet_a[1], facet_b[1])
            adj[cursor[c]] = k
            cursor[c] += 1
        end
    end
    return ptr, adj
end

# Conflict gather for interface items when all dofs written by the interface terms are
# cell-interior (purely discontinuous fields): an interface writes the dofs of its two
# cells, so two interfaces conflict iff they share a cell (the "line graph" of the facet
# adjacency). This needs at most Δ + 1 colors where Δ is the maximum number of facet
# neighbors of a cell.
function _gather_interface_cell_chunk!(count, interfaces, chunk, iptr, iadj)
    buf = Int[]
    candidates = Int[]
    for k in chunk
        facet_a, facet_b = interfaces[k]
        empty!(candidates)
        for c in (facet_a[1], facet_b[1])
            for r in iptr[c]:(iptr[c + 1] - 1)
                j = iadj[r]
                j == k || push!(candidates, j)
            end
        end
        _append_candidates_sorted_unique!(buf, count, candidates, k)
    end
    return buf
end

# Conservative conflict gather for interface items: when continuous fields are written
# by the interface terms the item also writes dofs shared with the node neighbors of its
# two cells, so two interfaces conflict iff a cell of one is node-adjacent to (or equal
# to) a cell of the other. This is a superset of the share-a-cell graph above.
function _gather_interface_node_chunk!(count, grid, interfaces, chunk, nodeptr, nodecells, iptr, iadj)
    buf = Int[]
    candidates = Int[]
    for k in chunk
        facet_a, facet_b = interfaces[k]
        empty!(candidates)
        for c in (facet_a[1], facet_b[1])
            for v in get_node_ids(getcells(grid, c))
                for r in nodeptr[v]:(nodeptr[v + 1] - 1)
                    # Cell node-adjacent to (or equal to) c: all its interfaces conflict
                    d = nodecells[r]
                    for r2 in iptr[d]:(iptr[d + 1] - 1)
                        j = iadj[r2]
                        j == k || push!(candidates, j)
                    end
                end
            end
        end
        _append_candidates_sorted_unique!(buf, count, candidates, k)
    end
    return buf
end

"""
    create_interface_coloring(
        grid::AbstractGrid, [topology::ExclusiveTopology], [cellset];
        alg::ColoringAlgorithm, discontinuous::Bool = false,
    )

Create a coloring of the *interfaces* of the grid such that no two conflicting
interfaces -- interfaces whose (concurrent) assembly may write to the same entries of
the global matrix and vector -- have the same color. This is the interface-loop
counterpart of [`create_coloring`](@ref), for threading assembly loops over
[`InterfaceIterator`](@ref) (e.g. interface terms in DG methods).

Returns a vector of vectors of interfaces, where each interface is a tuple of the two
facets `(facet_here, facet_there)`. Each color can be iterated with
`InterfaceIterator(grid_or_dh, color)`:

```julia
colors = create_interface_coloring(grid, topology)
for color in colors
    # (interfaces within a color are independent -- loop below can be parallelized)
    for ic in InterfaceIterator(dh, color)
        # assemble interface terms
    end
end
```

An interface writes to the dofs of its two cells. With the default
`discontinuous = false` the coloring is conservative: it is safe also when the interface
terms write to dofs of continuous fields, which are shared with all node neighbors of
the two cells. If *all* fields written by the interface assembly are discontinuous (all
dofs interior to the cells) this can be sharpened by passing `discontinuous = true`, in
which case two interfaces conflict only if they share a cell, resulting in
significantly fewer colors.

If `cellset` is given, only interfaces between two cells of the set are colored (cf.
[`create_coloring`](@ref)).

Note that for a purely discontinuous discretization the accompanying *cell* loop needs
no coloring at all -- no dofs are shared between cells -- and with continuous fields
present the standard [`create_coloring`](@ref) covers it. Constraint condensation
during assembly ([`apply_assemble!`](@ref) with e.g. [`AffineConstraint`](@ref)s) can
write outside of the interface dofs and is not accounted for here.
"""
function create_interface_coloring(
        grid::AbstractGrid, topology::ExclusiveTopology = ExclusiveTopology(grid),
        cellset = 1:getncells(grid);
        alg::ColoringAlgorithm.T = ColoringAlgorithm.WorkStream,
        discontinuous::Bool = false,
    )
    cellvec = _sorted_cellvec(cellset)
    interfaces = _enumerate_interfaces(grid, topology, cellvec)
    ninterfaces = length(interfaces)
    if ninterfaces == 0
        return Vector{NTuple{2, FacetIndex}}[]
    end
    iptr, iadj = _cell_to_interface_map(getncells(grid), interfaces)
    local colptr, rowval
    if discontinuous
        colptr, rowval = _chunked_gather!(ninterfaces, 1:ninterfaces) do counts, chunk
            _gather_interface_cell_chunk!(counts, interfaces, chunk, iptr, iadj)
        end
    else
        nodeptr, nodecells = _build_node_to_cell_map(grid, cellvec)
        colptr, rowval = _chunked_gather!(ninterfaces, 1:ninterfaces) do counts, chunk
            _gather_interface_node_chunk!(counts, grid, interfaces, chunk, nodeptr, nodecells, iptr, iadj)
        end
    end
    incidence_matrix = SparseMatrixCSC(ninterfaces, ninterfaces, colptr, rowval, fill(true, length(rowval)))
    id_colors = _color_incidence_matrix(incidence_matrix, 1:ninterfaces, alg)
    return [interfaces[ids] for ids in id_colors]
end
