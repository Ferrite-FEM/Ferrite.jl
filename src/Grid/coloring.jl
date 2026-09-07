# Split `1:n` into at most `maxchunks` contiguous ranges of similar size.
function _color_chunks(n::Int, maxchunks::Int)
    return Iterators.partition(1:n, max(1, cld(n, maxchunks)))
end

# We need a sorted collection without duplicates. The default case (`cells = 1:ncells`)
# fulfills this already.
_sorted_cellvec(cellset::AbstractUnitRange{Int}) = cellset
_sorted_cellvec(cellset) = unique!(sort!(collect(Int, cellset)))

# Append the extra conflict cells (constraint condensation and/or user provided) for
# `cellid` to the candidate list. The lists are constructed such that they never contain
# `cellid` itself and only cells in the cellset.
_append_extra_conflicts!(candidates, ::Nothing, cellid) = candidates
function _append_extra_conflicts!(candidates, extras::Dict{Int, Vector{Int}}, cellid)
    l = get(extras, cellid, nothing)
    l === nothing || append!(candidates, l)
    return candidates
end

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

function _gather_neighbor_chunk!(colcount, grid, cellvec, chunk, nodeptr, nodecells, extras = nothing)
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
        _append_extra_conflicts!(candidates, extras, cellid)
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

# Two cells sharing at least this many vertices share a facet: 1 vertex in 1D, an edge
# (2 vertices) in 2D, a triangular or quadrilateral face (>= 3 vertices) in 3D. On
# conforming grids this matches the facet classification in `ExclusiveTopology`. The
# `min` makes the relation symmetric and conservative (over-inclusive, which only adds
# conflict edges and is thus safe) for grids with mixed reference dimensions.
_facet_conflict_threshold(a::AbstractCell, b::AbstractCell) = min(getrefdim(a), getrefdim(b))

# Like `_gather_neighbor_chunk!` but only keeping the facet neighbors: node-sharing
# candidates that share at least `_facet_conflict_threshold` *vertices* with the cell.
# Counting corner vertices (`_num_shared_vertices`) rather than nodes matters for
# higher-order cells: e.g. a `QuadraticTetrahedron` edge neighbor shares 3 nodes (2
# vertices + midside node) but only 2 vertices, and must not classify as a facet
# neighbor.
function _gather_facet_neighbor_chunk!(facetcount, grid, cellvec, chunk, nodeptr, nodecells)
    buf = Int[]
    candidates = Int[]
    for i in chunk
        cellid = cellvec[i]
        cell = getcells(grid, cellid)
        empty!(candidates)
        for v in get_node_ids(cell)
            for r in nodeptr[v]:(nodeptr[v + 1] - 1)
                cell_neighbour = nodecells[r]
                cell_neighbour == cellid || push!(candidates, cell_neighbour)
            end
        end
        sort!(candidates; alg = QuickSort)
        prev = 0
        for cell_neighbour in candidates
            cell_neighbour == prev && continue
            prev = cell_neighbour
            other = getcells(grid, cell_neighbour)
            if _num_shared_vertices(cell, other) >= _facet_conflict_threshold(cell, other)
                push!(buf, cell_neighbour)
                facetcount[cellid] += 1
            end
        end
    end
    return buf
end

# Facet adjacency A_f restricted to the cellset, in CSR-like form (ptr indexed by cell
# id over all cells; only cellset cells have entries).
function _build_facet_adjacency(grid::AbstractGrid, cellvec, nodeptr, nodecells)
    return _chunked_gather!(getncells(grid), cellvec) do counts, chunk
        _gather_facet_neighbor_chunk!(counts, grid, cellvec, chunk, nodeptr, nodecells)
    end
end

# Facet adjacency from an existing topology, using the same helper as the sparsity
# pattern's interface entries (`add_interface_entries!`) so that the conflict graph is
# consistent with the pattern by construction.
function _facet_adjacency_from_topology(grid::AbstractGrid, cellvec, topology)
    ncells = getncells(grid)
    neighbor_cells = create_cell_to_neighbors(grid, topology)
    ptr = Vector{Int}(undef, ncells + 1)
    ptr[1] = 1
    for cellid in 1:ncells
        cnt = 0
        if insorted(cellid, cellvec)
            for n in neighbor_cells[cellid]
                cnt += insorted(n, cellvec)
            end
        end
        ptr[cellid + 1] = ptr[cellid] + cnt
    end
    adj = Vector{Int}(undef, ptr[end] - 1)
    for cellid in cellvec
        k = ptr[cellid]
        for n in neighbor_cells[cellid]
            insorted(n, cellvec) || continue
            adj[k] = n
            k += 1
        end
        sort!(view(adj, ptr[cellid]:(ptr[cellid + 1] - 1)))
    end
    return ptr, adj
end

# Conflict gather for interface (DG) assembly where only discontinuous (L2) fields cross
# the interfaces: an assembly item writes dofs of the cell and its facet neighbors, and
# with cell-interior dofs two items' interface writes conflict iff their facet closures
# intersect: A_f ∪ A_f² ("cells sharing a facet neighbor" -- both items write the shared
# neighbor's dofs). Note that e.g. 2D corner neighbors conflict through their two shared
# facet neighbors, while 3D vertex-only diagonal neighbors share no facet neighbor and
# their interface writes do not conflict.
# `include_node_conflicts` unions in the node-sharing graph, which is required whenever
# the DofHandler carries any continuous field (its *cell* assembly writes shared dofs of
# all node neighbors, including 3D vertex diagonals). For a purely discontinuous
# discretization no dofs are shared between cells and the node graph is dropped
# ("facet" mode, fewest colors).
function _gather_interface_chunk!(colcount, grid, cellvec, chunk, nodeptr, nodecells, facetptr, facetadj, extras, include_node_conflicts::Bool)
    buf = Int[]
    candidates = Int[]
    for i in chunk
        cellid = cellvec[i]
        empty!(candidates)
        if include_node_conflicts
            # Node neighbors (this includes the facet neighbors A_f)
            for v in get_node_ids(getcells(grid, cellid))
                for r in nodeptr[v]:(nodeptr[v + 1] - 1)
                    cell_neighbour = nodecells[r]
                    cell_neighbour == cellid || push!(candidates, cell_neighbour)
                end
            end
        else
            # A_f: the facet neighbors themselves
            for r in facetptr[cellid]:(facetptr[cellid + 1] - 1)
                push!(candidates, facetadj[r])
            end
        end
        # A_f²: facet neighbors of facet neighbors
        for r in facetptr[cellid]:(facetptr[cellid + 1] - 1)
            j = facetadj[r]
            for r2 in facetptr[j]:(facetptr[j + 1] - 1)
                k = facetadj[r2]
                k == cellid || push!(candidates, k)
            end
        end
        _append_extra_conflicts!(candidates, extras, cellid)
        _append_candidates_sorted_unique!(buf, colcount, candidates, cellid)
    end
    return buf
end

# Conflict gather for interface (DG) assembly where a continuous field crosses the
# interfaces ("product" mode): an item writes dofs of its facet closure {i} ∪ Nf(i),
# and continuous dofs are shared between node neighbors, so two items conflict iff
# their facet closures contain node-adjacent (or equal) cells. This is the closure
# product (I ∪ A_f)·(I ∪ A_n)·(I ∪ A_f), which strictly contains node-distance-2 (e.g.
# 3D vertex diagonals) *and* some node-distance-3 pairs (two cells three apart in a row
# conflict through the shared dofs of the two cells between them).
function _gather_product_chunk!(colcount, cellvec, chunk, an_ptr, an_adj, facetptr, facetadj, extras)
    buf = Int[]
    candidates = Int[]
    closure = Int[]
    for ii in chunk
        cellid = cellvec[ii]
        empty!(candidates)
        # X ∈ {i} ∪ Nf(i)
        empty!(closure)
        push!(closure, cellid)
        for r in facetptr[cellid]:(facetptr[cellid + 1] - 1)
            push!(closure, facetadj[r])
        end
        for X in closure
            # Y = X: append {X} ∪ Nf(X)
            X == cellid || push!(candidates, X)
            for r in facetptr[X]:(facetptr[X + 1] - 1)
                k = facetadj[r]
                k == cellid || push!(candidates, k)
            end
            # Y ∈ Nn(X): append {Y} ∪ Nf(Y)
            for r in an_ptr[X]:(an_ptr[X + 1] - 1)
                Y = an_adj[r]
                Y == cellid || push!(candidates, Y)
                for r2 in facetptr[Y]:(facetptr[Y + 1] - 1)
                    k = facetadj[r2]
                    k == cellid || push!(candidates, k)
                end
            end
        end
        _append_extra_conflicts!(candidates, extras, cellid)
        _append_candidates_sorted_unique!(buf, colcount, candidates, cellid)
    end
    return buf
end

# Normalize user-provided extra conflicts into the internal Dict form: symmetrized,
# self-edge free, restricted to the cellset (so that one conflict specification can be
# reused for different sub-colorings). Accepts a Dict (cell => conflicting cells) or an
# iterable of cell groups where all cells in a group mutually conflict ("cliques").
_normalize_extra_conflicts(::Nothing, ncells::Int, cellvec) = nothing
function _normalize_extra_conflicts(spec, ncells::Int, cellvec)
    extras = Dict{Int, Vector{Int}}()
    add_edge! = function (a::Int, b::Int)
        if !(1 <= a <= ncells && 1 <= b <= ncells)
            throw(ArgumentError("cell id out of range in extra_conflicts: got cells ($a, $b) for a grid with $ncells cells"))
        end
        a == b && return
        # Conflicts with cells outside the colored set are irrelevant for the coloring
        (insorted(a, cellvec) && insorted(b, cellvec)) || return
        push!(get!(() -> Int[], extras, a), b)
        push!(get!(() -> Int[], extras, b), a)
        return
    end
    if spec isa AbstractDict
        for (a, bs) in spec, b in bs
            add_edge!(Int(a), Int(b))
        end
    else
        for group in spec
            cells = collect(Int, group)
            for i in eachindex(cells)
                for j in (i + 1):lastindex(cells)
                    add_edge!(cells[i], cells[j])
                end
            end
        end
    end
    return isempty(extras) ? nothing : extras
end

function _merge_extras(a::Union{Nothing, Dict{Int, Vector{Int}}}, b::Union{Nothing, Dict{Int, Vector{Int}}})
    a === nothing && return b
    b === nothing && return a
    for (k, v) in b
        append!(get!(() -> Int[], a, k), v)
    end
    return a
end

# Conflict edges from constraint condensation during assembly. The method for
# `ch::ConstraintHandler` is defined in src/Dofs/ConstraintHandler.jl (this file is
# included before the DofHandler/ConstraintHandler definitions).
_incidence_constraint_extras(::Nothing, grid, cellvec, facetptr, facetadj) = nothing

# Incidence matrix for element connections in the grid: cells i and j are connected if
# assembly items for i and j may write to the same global matrix/vector entries.
#  - `interface_mode = :none`: conflict iff sharing a node (cell-local assembly of
#    continuous fields).
#  - `interface_mode = :facet`: interface (DG) assembly for a purely discontinuous
#    discretization: A_f ∪ A_f² (see `_gather_interface_chunk!`).
#  - `interface_mode = :sharp`: interface (DG) assembly where only discontinuous fields
#    cross interfaces but continuous fields exist: node graph ∪ A_f².
#  - `interface_mode = :product`: interface assembly where a continuous field crosses
#    interfaces, or field information is unavailable (see `_gather_product_chunk!`).
#  - `topology`: optional `ExclusiveTopology` used for exact facet adjacency (otherwise
#    facet neighbors are classified by shared-vertex counting).
#  - `ch`: optional `ConstraintHandler`; affine constraints (e.g. `PeriodicDirichlet`)
#    add conflicts between cells writing to the same master dofs during condensation.
#  - `extra_conflicts`: optional user-provided conflicts (see `create_coloring`).
# Note: the conflict graph is restricted to the cellset -- interfaces to cells outside
# the cellset are not considered part of the assembly loop.
function create_incidence_matrix(
        grid::AbstractGrid, cellset = 1:getncells(grid);
        interface_mode::Symbol = :none,
        topology = nothing,
        ch = nothing,
        extra_conflicts = nothing,
    )
    if interface_mode !== :none && interface_mode !== :facet && interface_mode !== :sharp && interface_mode !== :product
        throw(ArgumentError("invalid interface_mode: $(repr(interface_mode))"))
    end
    ncells = getncells(grid)
    cellvec = _sorted_cellvec(cellset)
    if isempty(cellvec)
        return SparseArrays.spzeros(Bool, Int, ncells, ncells)
    end

    nodeptr, nodecells = _build_node_to_cell_map(grid, cellvec)

    # Facet adjacency (within the cellset) for interface (DG) conflicts
    if interface_mode === :none
        facetptr = facetadj = nothing
    elseif topology === nothing
        facetptr, facetadj = _build_facet_adjacency(grid, cellvec, nodeptr, nodecells)
    else
        facetptr, facetadj = _facet_adjacency_from_topology(grid, cellvec, topology)
    end

    # Extra conflict edges from constraint condensation and/or user input
    extras = _merge_extras(
        _incidence_constraint_extras(ch, grid, cellvec, facetptr, facetadj),
        _normalize_extra_conflicts(extra_conflicts, ncells, cellvec),
    )

    # For each cell, gather the unique conflicting cells. The chunked gather makes the
    # result independent of the number of threads (see `_chunked_gather!`).
    local colptr, rowval
    if interface_mode === :none
        colptr, rowval = _chunked_gather!(ncells, cellvec) do counts, chunk
            _gather_neighbor_chunk!(counts, grid, cellvec, chunk, nodeptr, nodecells, extras)
        end
    elseif interface_mode === :facet || interface_mode === :sharp
        include_node_conflicts = interface_mode === :sharp
        colptr, rowval = _chunked_gather!(ncells, cellvec) do counts, chunk
            _gather_interface_chunk!(counts, grid, cellvec, chunk, nodeptr, nodecells, facetptr, facetadj, extras, include_node_conflicts)
        end
    else # :product
        # The closure product composition needs the node adjacency in queryable form
        an_ptr, an_adj = _chunked_gather!(ncells, cellvec) do counts, chunk
            _gather_neighbor_chunk!(counts, grid, cellvec, chunk, nodeptr, nodecells)
        end
        colptr, rowval = _chunked_gather!(ncells, cellvec) do counts, chunk
            _gather_product_chunk!(counts, cellvec, chunk, an_ptr, an_adj, facetptr, facetadj, extras)
        end
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
    # Note: the incidence matrix is assumed to be created with the same cellset and the
    # same conflict options, so all neighbors found through it are members of the
    # cellset. In particular, all conflict edges (including long-range ones from e.g.
    # periodic constraints or user input) must already be present in the matrix here:
    # the odd/even zone merge in step 3 is only correct because a breadth-first-search
    # edge never spans more than one zone, which holds for any edge of the graph being
    # traversed but not for edges added afterwards.
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
    create_coloring(g::Grid, cellset = 1:getncells(g); alg::ColoringAlgorithm, kwargs...)
    create_coloring(dh::DofHandler, [ch::ConstraintHandler]; alg::ColoringAlgorithm, kwargs...)

Create a coloring of the cells in grid `g` such that no two conflicting cells -- cells
whose (concurrent) assembly may write to the same entries of the global matrix and
vector -- have the same color. If only a subset of cells should be colored, the cells to
color can be specified by `cellset`.

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

By default two cells conflict when they share a node, which is correct for cell-local
assembly of continuous fields. Additional conflict sources can be enabled with keyword
arguments:

 - `interface_coupling`: also account for interface (DG) assembly, where an assembly
   item writes to dofs of the cell *and its facet neighbors* (e.g. via
   [`InterfaceValues`](@ref)). For the grid based method this is a `Bool`, and the
   conflict graph is conservative: it must be safe also when continuous fields are
   written from the interface terms, since the grid carries no information about the
   discretization. The `DofHandler` based method additionally accepts the same coupling
   matrix as [`allocate_matrix`](@ref)/[`init_sparsity_pattern`](@ref) and uses the
   interpolations to build a sharper conflict graph (fewer colors) when all fields that
   couple across interfaces are discontinuous.
 - `topology`: optional [`ExclusiveTopology`](@ref), used to determine facet neighbors
   for `interface_coupling`. If not given, facet neighbors are classified directly from
   the grid.
 - `ch::ConstraintHandler` (second positional argument of the `DofHandler` method):
   account for constraint condensation during assembly (i.e.
   [`apply_assemble!`](@ref)/[`apply_local!`](@ref)): cells whose condensation writes to
   the same master dofs conflict, even if they are far apart in the grid (e.g. cells on
   opposite sides with [`PeriodicDirichlet`](@ref), or arbitrary
   [`AffineConstraint`](@ref)s). Only affine constraints add conflicts -- plain
   `Dirichlet` conditions do not write to other dofs during condensation.
 - `extra_conflicts`: user-provided conflicts for couplings that Ferrite cannot infer.
   Accepts a `Dict{Int, Vector{Int}}` (cell id => conflicting cell ids; symmetrized
   automatically) or an iterable of cell id collections where all cells in a collection
   mutually conflict, e.g. `[[1, 17, 203], [4, 99]]`.

The `DofHandler` method accepts (and validates) the full set of keyword arguments of
[`add_sparsity_entries!`](@ref) so that the arguments used for matrix allocation can be
forwarded as-is; `coupling` and `keep_constrained` do not influence the conflict graph
and are ignored.

Note that conflicts are only considered between cells in `cellset`: interface terms
between a cell in the set and a cell outside of it are not accounted for.

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
function create_coloring(
        g::AbstractGrid, cellset = 1:getncells(g);
        alg::ColoringAlgorithm.T = ColoringAlgorithm.WorkStream,
        interface_coupling::Bool = false,
        topology = nothing,
        extra_conflicts = nothing,
    )
    # Without a DofHandler there is no information about the discretization, so for
    # interface coupling the conservative conflict graph must be used (see
    # `_gather_product_chunk!`). Pass a DofHandler to get a sharper coloring for purely
    # discontinuous interpolations.
    incidence_matrix = create_incidence_matrix(
        g, cellset;
        interface_mode = interface_coupling ? :product : :none,
        topology, extra_conflicts,
    )
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
