# Split `1:n` into at most `maxchunks` contiguous ranges of similar size.
function _color_chunks(n::Int, maxchunks::Int)
    return Iterators.partition(1:n, max(1, cld(n, maxchunks)))
end

# We need a sorted collection without duplicates. The default case (`cells = 1:ncells`)
# fulfills this already.
_sorted_cellvec(cellset::AbstractUnitRange{Int}) = cellset
_sorted_cellvec(cellset) = unique!(sort!(collect(Int, cellset)))

# Candidates are gathered for one item `k` at a time and may occur multiple times (e.g. a
# neighbor sharing several nodes with a cell). Duplicates are skipped with the marker array
# `mark`: `mark[j] == k` if `j` has already been added for item `k`. Since every item is
# gathered once, `mark` never has to be reset. Setting `mark[k] = k` before gathering
# excludes the item itself.
@inline function _push_candidate!(candidates, mark, j, k)
    if mark[j] != k
        mark[j] = k
        push!(candidates, j)
    end
    return candidates
end

# Append the (unique) candidates of item `k` to `buf`, counting into `colcount[k]`
function _append_candidates!(buf, colcount, candidates, k)
    append!(buf, candidates)
    colcount[k] += length(candidates)
    return buf
end

# Sort the (unique) candidates of item `k` and append them to `buf`, counting into
# `colcount[k]`.
function _append_candidates_sorted!(buf, colcount, candidates, k)
    # QuickSort sorts in place without allocating, unlike the default (radix) algorithm,
    # which allocates a workspace on each call.
    sort!(candidates; alg = QuickSort)
    return _append_candidates!(buf, colcount, candidates, k)
end

function _gather_neighbor_chunk!(colcount, grid, cellvec, chunk, nodeptr, nodecells)
    buf = Int[]
    candidates = Int[]
    mark = zeros(Int, getncells(grid))
    # Loop over cells in the chunk
    for i in chunk
        cellid = cellvec[i]
        empty!(candidates)
        mark[cellid] = cellid
        # Loop over nodes of the cell
        for v in get_node_ids(getcells(grid, cellid))
            # Loop over the cells connected to this node
            for r in nodeptr[v]:(nodeptr[v + 1] - 1)
                _push_candidate!(candidates, mark, nodecells[r], cellid)
            end
        end
        _append_candidates_sorted!(buf, colcount, candidates, cellid)
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

# Conflict graph in CSR-like form, i.e. the sparsity pattern of a symmetric incidence
# matrix without the values: the neighbors of vertex `i` are
# `adj[ptr[i]:(ptr[i + 1] - 1)]`, which do not have to be sorted.
struct ConflictGraph
    ptr::Vector{Int}
    adj::Vector{Int}
end
_nvertices(g::ConflictGraph) = length(g.ptr) - 1
_neighbor_range(g::ConflictGraph, i::Integer) = g.ptr[i]:(g.ptr[i + 1] - 1)

# Greedy coloring of the cells in `cells` such that no two connected cells (for which
# `is_member` returns `true` for both) have the same color. `cell_colors` (indexed by cell
# id, zeroed for the cells to color) and `occupied` are scratch data.
function _greedy_coloring!(cell_colors::Vector{Int}, occupied::Vector{Bool}, graph::ConflictGraph, cells, is_member::F) where {F}
    final_colors = Vector{Int}[]
    total_colors = 0
    for cellid in cells
        for i in 1:total_colors
            occupied[i] = false
        end
        # loop over neighbors
        for r in _neighbor_range(graph, cellid)
            cell_neighbour = graph.adj[r]
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
function greedy_coloring(graph::ConflictGraph, cells = 1:_nvertices(graph))
    ncells = _nvertices(graph)
    cell_colors = zeros(Int, ncells)
    occupied = Bool[]
    if cells isa AbstractUnitRange{Int}
        return _greedy_coloring!(cell_colors, occupied, graph, cells, c -> c in cells)
    else
        mask = zeros(Bool, ncells)
        for c in cells
            mask[c] = true
        end
        return _greedy_coloring!(cell_colors, occupied, graph, cells, c -> mask[c])
    end
end

function _color_zone_chunk!(zone_colors, chunk, zones, zone_of, cell_colors, graph::ConflictGraph)
    occupied = Bool[]
    for zi in chunk
        zone_colors[zi] = _greedy_coloring!(cell_colors, occupied, graph, zones[zi], c -> zone_of[c] == zi)
    end
    return
end

# See Appendix A in https://www.math.colostate.edu/%7Ebangerth/publications/2013-pattern.pdf
function workstream_coloring(graph::ConflictGraph, cellset)

    cellvec = _sorted_cellvec(cellset)
    if length(cellvec) == 0
        return Vector{Int}[]
    elseif length(cellvec) == 1
        return Vector{Int}[Int[first(cellvec)]]
    end
    ncells = _nvertices(graph)

    ###################
    # 1. Partitioning #
    ###################
    # Zero represents no zone assigned yet, and -1 cells that are not in the cellset (and
    # therefore never assigned a zone, i.e. the conflict graph may contain other cells)
    zone_of = fill(-1, ncells)
    zone_of[cellvec] .= 0
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
                for r in _neighbor_range(graph, c)
                    cell_neighbour = graph.adj[r]
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
        Threads.@spawn _color_zone_chunk!(zone_colors, $chunk, zones, zone_of, cell_colors, graph)
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
    create_coloring(
        grid::AbstractGrid, set = 1:getncells(grid);
        alg::ColoringAlgorithm
    )

Create a coloring of the items in `set` such that no two conflicting items -- items
whose (concurrent) assembly may write to the same entries of the global matrix and
vector -- have the same color. The returned colors partition `set`, and all items of
one color can therefore be assembled concurrently.

The items to color are determined by the element type of `set`:
 - cells (`Int`), e.g. a cellset (default: all cells of the grid). A cell writes to the
   dofs of the cell.
 - facets (`FacetIndex`), e.g. a facetset for assembly of boundary terms. A facet
   writes to the dofs of its cell.
 - interfaces (`NTuple{2, FacetIndex}`), e.g. [`interfaceskeleton`](@ref) or a subset
   of it for assembly of interface terms in DG methods. An interface writes to the dofs
   of its two cells.

Returns a vector of vectors of items, e.g. for cells:

```julia
ret = [
   [1, 3, 5, 10, ...], # cells for color 1
   [2, 4, 6, 12, ...], # cells for color 2
]
```

The colors are processed one after another, with the items of each color distributed
over concurrent tasks. See the [multithreaded assembly how-to](@ref howto-threaded-assembly)
for complete examples.

The coloring is conservative: two items conflict if a cell of one shares a node with (or
is equal to) a cell of the other, which is safe for any discretization (continuous fields
place dofs on vertices, edges, and faces, shared with all node neighbors of the cell).
Conflicts are only considered among the items in `set`, since only these are assembled
by the corresponding loop. Duplicate cells in a cellset are ignored, while facets and
interfaces must appear at most once in `set` (and interfaces in one orientation).

For an exact coloring, based on the dofs that the items write to, use
[`create_coloring(dh::DofHandler, [ch], set)`](@ref create_coloring(::DofHandler))
instead. This typically results in fewer colors, e.g. for discontinuous fields or fields
with dofs on facets or edges only, and can also account for constraint condensation
during assembly ([`apply_assemble!`](@ref)) of affine constraints (e.g.
[`PeriodicDirichlet`](@ref)), which writes outside of the item dofs. This function is
equivalent to `create_coloring(AssemblyConflicts(grid), set; alg)`; construct the
[`AssemblyConflicts`](@ref) explicitly to reuse them when coloring multiple sets (e.g. both
the cells and the interfaces).

Two different algorithms are available, specified with the `alg` keyword argument:
 - `alg = ColoringAlgorithm.WorkStream` (default): Three step algorithm from
   Turcksin et al. [Turcksin2016](@cite), albeit with a greedy coloring in the second step. Generally results in more colors than
   `ColoringAlgorithm.Greedy`, however the items are more equally distributed among the colors.
 - `alg = ColoringAlgorithm.Greedy`: greedy algorithm that works well for structured quadrilateral grids such as
   e.g. quadrilateral grids from `generate_grid`.

The resulting cell colors can be visualized using [`Ferrite.write_cell_colors`](@ref).

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
        grid::AbstractGrid, set = 1:getncells(grid);
        alg::ColoringAlgorithm.T = ColoringAlgorithm.WorkStream,
    )
    items = _collect_items(eltype(set), getncells(grid), set)
    isempty(items) && return Vector{eltype(items)}[]
    # Only the cells written by the items can cause conflicts between them, so only
    # compute the conflicts for these (instead of the full AssemblyConflicts(grid))
    cellvec = eltype(items) === Int ? items : _sorted_cellvec(Iterators.flatten(_item_cells(item) for item in items))
    return _create_coloring(_grid_conflicts(grid, cellvec), items, alg)
end

# The items of `set` as a vector: sorted unique cells, or validated facets/interfaces
_collect_items(::Type{<:Integer}, ncells::Int, cellset) = collect(Int, _sorted_cellvec(cellset))
function _collect_items(
        ::Type{T}, ncells::Int, set
    ) where {T <: Union{FacetIndex, NTuple{2, FacetIndex}}}
    items = set isa AbstractVector{T} ? set : collect(T, set)
    for item in items
        if !all(c -> 1 <= c <= ncells, _item_cells(item))
            throw(ArgumentError("$item refers to a cell id out of range for a grid with $ncells cells"))
        end
    end
    if !allunique(items)
        throw(ArgumentError("the set contains duplicate entries"))
    end
    return items
end

function _color_conflict_graph(graph::ConflictGraph, cellset, alg::ColoringAlgorithm.T)
    if alg === ColoringAlgorithm.WorkStream
        return workstream_coloring(graph, cellset)
    elseif alg === ColoringAlgorithm.Greedy
        return greedy_coloring(graph, cellset)
    else
        error("impossible")
    end
end

#########
# Items #
#########

# The cells an item writes to
_item_cells(cell::Int) = (cell,)
_item_cells(facet::FacetIndex) = (facet[1],)
_item_cells((facet_a, facet_b)::NTuple{2, FacetIndex}) = (facet_a[1], facet_b[1])

# Map from cell id to the ids (indices into `items`) of the items writing to it, in
# CSR-like form. Since items are enumerated in order the per-cell lists are sorted. An item
# is listed twice for a cell if it writes to that cell twice (an interface with the same
# cell on both sides, which `interfaceskeleton` never contains); such duplicates are
# skipped by the conflict gathers (see `_push_candidate!`).
function _cell_to_item_map(ncells::Int, items)
    ptr = zeros(Int, ncells + 1)
    ptr[1] = 1
    for item in items
        for c in _item_cells(item)
            ptr[c + 1] += 1
        end
    end
    for i in 2:(ncells + 1)
        ptr[i] += ptr[i - 1]
    end
    adj = Vector{Int}(undef, ptr[end] - 1)
    cursor = copy(ptr)
    for (k, item) in pairs(items)
        for c in _item_cells(item)
            adj[cursor[c]] = k
            cursor[c] += 1
        end
    end
    return ptr, adj
end

#####################################
# Item coloring from cell conflicts #
#####################################

"""
    AssemblyConflicts(grid::AbstractGrid)
    AssemblyConflicts(dh::DofHandler, [ch::ConstraintHandler])

Precompute the conflicts between the cells of the grid for coloring assembly loops. Two
cells conflict if their assembly may write to a common entry of the global matrix and
vector. The conflicts are used to compute colorings of cells, facets, and interfaces with
[`create_coloring(::AssemblyConflicts, set)`](@ref create_coloring(::AssemblyConflicts)),
and the same `AssemblyConflicts` can be reused for coloring multiple sets (e.g. the
cells, a facetset and the interfaces).

 - `AssemblyConflicts(grid)`: conservative conflicts from the grid only: two cells conflict
   if they share a node (cf. [`create_coloring(::AbstractGrid)`](@ref create_coloring)).
 - `AssemblyConflicts(dh, [ch])`: exact conflicts from the dofs: two cells conflict if
   they write to a common dof. The write set of a cell is given by its dofs
   ([`celldofs`](@ref)). If the `ConstraintHandler` `ch` is given the write set also
   includes the master dofs of the affinely constrained dofs of the cell, since
   constraint condensation during assembly ([`apply_assemble!`](@ref)) writes to these.
"""
struct AssemblyConflicts{G <: AbstractGrid}
    grid::G
    # Whether a cell writes to any dof at all
    cell_writes::BitVector
    # Cell conflict graph in CSR-like form: the (other) cells whose write sets intersect
    # the write set of the cell
    conflict_ptr::Vector{Int}
    conflict_cells::Vector{Int}
end

AssemblyConflicts(grid::AbstractGrid) = _grid_conflicts(grid, 1:getncells(grid))

# Conservative (node based) conflicts between the cells in `cellvec` (sorted): cells
# conflict if they share a node. Cells not in `cellvec` have no conflicts.
function _grid_conflicts(grid::AbstractGrid, cellvec)
    ncells = getncells(grid)
    nodeptr, nodecells = _build_node_to_cell_map(grid, cellvec)
    conflict_ptr, conflict_cells = _chunked_gather!(ncells, cellvec) do counts, chunk
        _gather_neighbor_chunk!(counts, grid, cellvec, chunk, nodeptr, nodecells)
    end
    return AssemblyConflicts(grid, trues(ncells), conflict_ptr, conflict_cells)
end

# Sort and deduplicate `v[start:end]` in place
function _sort_unique_tail!(v::Vector{Int}, start::Int)
    sort!(view(v, start:length(v)))
    last = start - 1
    for r in start:length(v)
        if last < start || v[r] != v[last]
            last += 1
            v[last] = v[r]
        end
    end
    return resize!(v, last)
end

# TODO: Support restricting the write sets to a subset of the fields, for loops that only
#       assemble into the dofs of some fields. For example, with a continuous field u and
#       a discontinuous field p, where the interface loop only assembles p terms (and only
#       passes the p dofs to `assemble!`), the interfaces would only conflict through a
#       common cell. This could be a `fields` keyword argument of the constructor, or of
#       `create_coloring` if a conflict graph per field is stored.
# TODO: Writes to algebraic dofs (see `algebraic_dofs`) are not accounted for, since they
#       are not part of the cell dofs, so a colored loop assembling e.g. cells coupled to
#       an algebraic variable silently races on its entries. Support an
#       `algebraic_couplings` keyword argument (as for `allocate_matrix`) adding the
#       algebraic dofs to the write sets of the coupled cells. Together with detection of
#       groups with many cells (see below) this would result in either a correct coloring
#       or an error suggesting atomic assembly.
function AssemblyConflicts(dh::DofHandler, ch::Union{Nothing, ConstraintHandler} = nothing)
    isclosed(dh) || throw(ArgumentError("the DofHandler must be closed"))
    if ch !== nothing
        isclosed(ch) || throw(ArgumentError("the ConstraintHandler must be closed"))
        ch.dh === dh || throw(ArgumentError("the ConstraintHandler belongs to a different DofHandler"))
    end
    grid = get_grid(dh)
    ncells = getncells(grid)
    n = ndofs(dh)
    # Map affinely constrained dofs to their index in the constraint handler. Note that
    # `close!(ch)` has already resolved recursive constraints, so the coefficient lists
    # (masters) are final.
    affine_of = zeros(Int, ch === nothing ? 0 : n)
    if ch !== nothing
        for (i, d) in pairs(ch.prescribed_dofs)
            coefficients = ch.dofcoefficients[i]
            (coefficients === nothing || isempty(coefficients)) && continue
            affine_of[d] = i
        end
    end
    # 1. Cell -> written dofs (sorted, unique)
    wptr, wdofs = _chunked_gather!(ncells, 1:ncells) do counts, chunk
        _gather_cell_write_set_chunk!(counts, dh, ch, affine_of, chunk)
    end
    cell_writes = BitVector(wptr[c + 1] > wptr[c] for c in 1:ncells)
    # 2. Dof -> writing cells (transpose, sorted since cells are visited in order)
    dptr = zeros(Int, n + 1)
    dptr[1] = 1
    for d in wdofs
        dptr[d + 1] += 1
    end
    for i in 2:(n + 1)
        dptr[i] += dptr[i - 1]
    end
    dcells = Vector{Int}(undef, length(wdofs))
    cursor = copy(dptr)
    for c in 1:ncells, r in wptr[c]:(wptr[c + 1] - 1)
        d = wdofs[r]
        dcells[cursor[d]] = c
        cursor[d] += 1
    end
    # 3. Group the dofs written by more than one cell by their set of cells. The group is
    # represented by its smallest dof, found independently (in parallel) for each dof.
    rep = zeros(Int, n) # dof -> representative dof, or 0 if written by at most one cell
    @sync for chunk in _color_chunks(n, Threads.nthreads())
        Threads.@spawn _dof_representative_chunk!(rep, $chunk, wptr, wdofs, dptr, dcells)
    end
    # Number the groups and gather their cells
    dof_group = zeros(Int, n)
    group_ptr = Int[1]
    group_cells = Int[]
    for d in 1:n
        rep[d] == 0 && continue
        if rep[d] == d
            append!(group_cells, view(dcells, dptr[d]:(dptr[d + 1] - 1)))
            push!(group_ptr, length(group_cells) + 1)
            dof_group[d] = length(group_ptr) - 1
        else
            dof_group[d] = dof_group[rep[d]] # rep[d] < d is already numbered
        end
    end
    # 4. Cell -> groups
    cell_ptr, cell_groups = _chunked_gather!(ncells, 1:ncells) do counts, chunk
        _gather_cell_groups_chunk!(counts, dof_group, wptr, wdofs, chunk)
    end
    # 5. Cell conflict graph: cells conflict iff they write to a common dof group
    # TODO: All cells of a group conflict pairwise, so a group with many cells, e.g. from
    #       an affine constraint where many dofs share a master (such as an algebraic dof
    #       for a rigid body constraint on a boundary), results in a quadratic number of
    #       conflicts (and a nearly serial coloring of these cells). Detect such groups
    #       and e.g. throw an error suggesting atomic assembly instead.
    conflict_ptr, conflict_cells = _chunked_gather!(ncells, 1:ncells) do counts, chunk
        _gather_cell_conflict_chunk!(counts, group_ptr, group_cells, cell_ptr, cell_groups, chunk)
    end
    return AssemblyConflicts(grid, cell_writes, conflict_ptr, conflict_cells)
end

function _gather_cell_conflict_chunk!(counts, group_ptr, group_cells, cell_ptr, cell_groups, chunk)
    buf = Int[]
    candidates = Int[]
    mark = zeros(Int, length(counts))
    for c in chunk
        empty!(candidates)
        mark[c] = c
        for rg in cell_ptr[c]:(cell_ptr[c + 1] - 1)
            g = cell_groups[rg]
            for rc in group_ptr[g]:(group_ptr[g + 1] - 1)
                _push_candidate!(candidates, mark, group_cells[rc], c)
            end
        end
        _append_candidates_sorted!(buf, counts, candidates, c)
    end
    return buf
end

# The (sorted, unique) dofs written by the cells in the chunk: the cell dofs and the
# masters of the affinely constrained dofs
function _gather_cell_write_set_chunk!(counts, dh::DofHandler, ch, affine_of, chunk)
    buf = Int[]
    dofbuf = Int[]
    for c in chunk
        start = length(buf) + 1
        resize!(dofbuf, ndofs_per_cell(dh, c))
        isempty(dofbuf) || celldofs!(dofbuf, dh, c)
        for d in dofbuf
            push!(buf, d)
            if ch !== nothing && affine_of[d] != 0
                for (m, _) in ch.dofcoefficients[affine_of[d]]
                    push!(buf, m)
                end
            end
        end
        _sort_unique_tail!(buf, start)
        counts[c] = length(buf) - start + 1
    end
    return buf
end

# For each dof in the chunk written by more than one cell, find the smallest dof written
# by exactly the same cells. Such a dof is written by the first cell of the list, so it
# suffices to search the (sorted) write set of that cell.
function _dof_representative_chunk!(rep, chunk, wptr, wdofs, dptr, dcells)
    for d in chunk
        rd = dptr[d]:(dptr[d + 1] - 1)
        length(rd) < 2 && continue
        rep[d] = d
        c = dcells[first(rd)]
        for r in wptr[c]:(wptr[c + 1] - 1)
            d2 = wdofs[r]
            d2 >= d && break
            rd2 = dptr[d2]:(dptr[d2 + 1] - 1)
            length(rd2) == length(rd) || continue
            if all(i -> dcells[rd[i]] == dcells[rd2[i]], eachindex(rd))
                rep[d] = d2
                break
            end
        end
    end
    return rep
end

# The (sorted, unique) dof groups written by the cells in the chunk
function _gather_cell_groups_chunk!(counts, dof_group, wptr, wdofs, chunk)
    buf = Int[]
    for c in chunk
        start = length(buf) + 1
        for r in wptr[c]:(wptr[c + 1] - 1)
            g = dof_group[wdofs[r]]
            g == 0 || push!(buf, g)
        end
        _sort_unique_tail!(buf, start)
        counts[c] = length(buf) - start + 1
    end
    return buf
end

# Exact conflict gather: two items conflict iff their write sets (the union of the write
# sets of their cells) intersect, i.e. iff they write to a common cell (with a nonempty
# write set) or a cell of one conflicts with a cell of the other.
function _gather_item_conflict_chunk!(count, cc::AssemblyConflicts, items, chunk, iptr, iadj)
    buf = Int[]
    candidates = Int[]
    mark = zeros(Int, length(count))
    for k in chunk
        empty!(candidates)
        mark[k] = k
        for c in _item_cells(items[k])
            cc.cell_writes[c] || continue
            # Items writing to the same cell
            for r in iptr[c]:(iptr[c + 1] - 1)
                _push_candidate!(candidates, mark, iadj[r], k)
            end
            # Items writing to a cell conflicting with c
            for rc in cc.conflict_ptr[c]:(cc.conflict_ptr[c + 1] - 1)
                c2 = cc.conflict_cells[rc]
                for r in iptr[c2]:(iptr[c2 + 1] - 1)
                    _push_candidate!(candidates, mark, iadj[r], k)
                end
            end
        end
        # The neighbors are not sorted since the coloring algorithms do not require it
        _append_candidates!(buf, count, candidates, k)
    end
    return buf
end

# Conflict graph of the items, indexed by item number. The cell conflict graph (the
# expensive, set independent part) is cached in the `AssemblyConflicts`; this only
# translates it to the items of the given set, which is different for every set.
function _item_conflict_graph(cc::AssemblyConflicts, items)
    nitems = length(items)
    iptr, iadj = _cell_to_item_map(getncells(cc.grid), items)
    colptr, rowval = _chunked_gather!(nitems, 1:nitems) do counts, chunk
        _gather_item_conflict_chunk!(counts, cc, items, chunk, iptr, iadj)
    end
    return ConflictGraph(colptr, rowval)
end

"""
    create_coloring(conflicts::AssemblyConflicts, set = 1:getncells(grid); alg::ColoringAlgorithm)

Create a coloring of the items (cells, facets, or interfaces, see
[`create_coloring(::AbstractGrid)`](@ref create_coloring)) in `set` from the precomputed
`conflicts` (see [`AssemblyConflicts`](@ref)). An item writes to its cells (facets to the
cell of the facet, and interfaces to the two cells of the interface), and two items
conflict if they write to a common cell or to two conflicting cells. The same
`AssemblyConflicts` can be reused to color multiple sets, e.g. both the cells and the
interfaces.
"""
function create_coloring(
        conflicts::AssemblyConflicts, set = 1:getncells(conflicts.grid);
        alg::ColoringAlgorithm.T = ColoringAlgorithm.WorkStream,
    )
    items = _collect_items(eltype(set), getncells(conflicts.grid), set)
    return _create_coloring(conflicts, items, alg)
end

# Coloring of the (collected, see `_collect_items`) items
function _create_coloring(conflicts::AssemblyConflicts, items::AbstractVector{T}, alg::ColoringAlgorithm.T) where {T}
    isempty(items) && return Vector{T}[]
    if T === Int
        # Cells: the cell conflict graph is the conflict graph of the items (the
        # coloring algorithms ignore conflicts with cells that are not in `items`)
        graph = ConflictGraph(conflicts.conflict_ptr, conflicts.conflict_cells)
        # (`items` are sorted and unique, so all cells if the lengths match)
        cellset = length(items) == _nvertices(graph) ? (1:length(items)) : items
        return _color_conflict_graph(graph, cellset, alg)
    end
    graph = _item_conflict_graph(conflicts, items)
    id_colors = _color_conflict_graph(graph, 1:length(items), alg)
    return Vector{T}[items[ids] for ids in id_colors]
end

"""
    create_coloring(
        dh::DofHandler, [ch::ConstraintHandler], set = 1:getncells(grid);
        alg::ColoringAlgorithm
    )

Create a coloring of the items (cells, facets, or interfaces, see
[`create_coloring(::AbstractGrid)`](@ref create_coloring)) in `set` based on the dofs
that the items write to. This is equivalent to
`create_coloring(AssemblyConflicts(dh, [ch]), set; alg)`, see [`AssemblyConflicts`](@ref)
and [`create_coloring(::AssemblyConflicts)`](@ref); construct the `AssemblyConflicts`
explicitly to reuse them when coloring multiple sets.

The resulting coloring is exact (two items conflict if and only if their assembly writes
to a common dof), and therefore typically results in fewer colors than the grid based
coloring, e.g. for discretizations where neighboring cells only share dofs on facets or
edges (e.g. [`Nedelec`](@ref) or [`CrouzeixRaviart`](@ref)), or no dofs at all (e.g.
[`DiscontinuousLagrange`](@ref), in which case the cells need a single color). If the
`ConstraintHandler` `ch` is given, writes to the master dofs of affinely constrained dofs
are accounted for, which is required for constraint condensation during assembly (see
[`apply_assemble!`](@ref)).

Note that algebraic dofs (see [`algebraic_dofs`](@ref)) are not part of the cell dofs,
and writes to them are therefore not accounted for. Since they are typically shared by
all cells, use atomic assembly for those instead.
"""
function create_coloring(
        dh::DofHandler, set = 1:getncells(get_grid(dh));
        alg::ColoringAlgorithm.T = ColoringAlgorithm.WorkStream,
    )
    return create_coloring(AssemblyConflicts(dh), set; alg)
end
function create_coloring(
        dh::DofHandler, ch::ConstraintHandler, set = 1:getncells(get_grid(dh));
        alg::ColoringAlgorithm.T = ColoringAlgorithm.WorkStream,
    )
    return create_coloring(AssemblyConflicts(dh, ch), set; alg)
end
