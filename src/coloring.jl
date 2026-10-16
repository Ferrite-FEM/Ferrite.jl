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
    create_coloring(
        grid::AbstractGrid, set = 1:getncells(grid);
        alg::ColoringAlgorithm, cells_share_dofs::Bool = true,
        ch::Union{Nothing, ConstraintHandler} = nothing,
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

With the default `cells_share_dofs = true` the coloring is conservative: two items conflict
if a cell of one is node-adjacent to (or equal to) a cell of the other, which is safe
also when dofs are shared between cells (continuous fields place dofs on vertices,
edges, and faces, shared with all node neighbors of the cell). If *no* dofs are shared
between cells -- every field written by the assembly has all dofs interior to the
cells, e.g. [`DiscontinuousLagrange`](@ref) -- this can be sharpened by passing
`cells_share_dofs = false`, in which case two items conflict only if they write to the same
cell, resulting in significantly fewer colors. In particular, cells then never conflict
and the cell loop needs no coloring (a single color is returned). Note that a
discretization being "discontinuous" is not sufficient: e.g. `CrouzeixRaviart` is
discontinuous but places (shared) dofs on the facets.

Conflicts are only considered among the items in `set`, since only these are assembled
by the corresponding loop. Duplicate cells in a cellset are ignored, while facets and
interfaces must appear at most once in `set` (and interfaces in one orientation).

Constraint condensation during assembly ([`apply_assemble!`](@ref)) of affine
constraints (e.g. [`PeriodicDirichlet`](@ref) or [`AffineConstraint`](@ref)) also
writes to the master dofs of constrained dofs, which can be far away from the item in
the mesh (e.g. on the opposite side of the domain for periodic constraints). To account
for this, pass the (closed) `ConstraintHandler` as the `ch` keyword argument: items
writing to a common master dof then also conflict. This is not needed if the
constraints are applied after assembly (e.g. with [`apply!`](@ref)), or if there are
no affine constraints (plain [`Dirichlet`](@ref) constraints cause no writes outside
of the item dofs).

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
        cells_share_dofs::Bool = true,
        ch::Union{Nothing, ConstraintHandler} = nothing,
    )
    if ch !== nothing
        isclosed(ch) || throw(ArgumentError("the ConstraintHandler must be closed"))
        if get_grid(ch.dh) !== grid
            throw(ArgumentError("the ConstraintHandler belongs to a DofHandler with a different grid"))
        end
    end
    return _create_coloring(eltype(set), grid, set, alg, cells_share_dofs, ch)
end

# Cells
function _create_coloring(
        ::Type{<:Integer}, grid::AbstractGrid, cellset, alg::ColoringAlgorithm.T,
        cells_share_dofs::Bool, ch::Union{Nothing, ConstraintHandler}
    )
    if ch !== nothing
        # Conflicts from constraint condensation are computed by the generic item
        # coloring. The item ids of the sorted cells are a monotone relabeling of the
        # cell ids, so in absence of affine constraints this gives the same coloring as
        # the cell id based path below (for a sorted cellset; Greedy colors an unsorted
        # cellset in the given order).
        cellvec = collect(Int, _sorted_cellvec(cellset))
        return _color_items(grid, cellvec, alg, cells_share_dofs, ch)
    end
    if !cells_share_dofs
        # The write sets of any two (distinct) cells are disjoint
        cellvec = _sorted_cellvec(cellset)
        return isempty(cellvec) ? Vector{Int}[] : Vector{Int}[collect(Int, cellvec)]
    end
    incidence_matrix = create_incidence_matrix(grid, cellset)
    return _color_incidence_matrix(incidence_matrix, cellset, alg)
end

# Facets and interfaces
function _create_coloring(
        ::Type{T}, grid::AbstractGrid, set, alg::ColoringAlgorithm.T, cells_share_dofs::Bool,
        ch::Union{Nothing, ConstraintHandler}
    ) where {T <: Union{FacetIndex, NTuple{2, FacetIndex}}}
    items = set isa AbstractVector{T} ? set : collect(T, set)
    ncells = getncells(grid)
    for item in items
        if !all(c -> 1 <= c <= ncells, _item_cells(item))
            throw(ArgumentError("$item refers to a cell id out of range for a grid with $ncells cells"))
        end
    end
    if !allunique(items)
        throw(ArgumentError("the set contains duplicate entries"))
    end
    return _color_items(grid, items, alg, cells_share_dofs, ch)
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

##########################################
# Item (cell, facet, interface) coloring #
##########################################

# The cells an item writes to
_item_cells(cell::Int) = (cell,)
_item_cells(facet::FacetIndex) = (facet[1],)
_item_cells((facet_a, facet_b)::NTuple{2, FacetIndex}) = (facet_a[1], facet_b[1])

# Map from cell id to the ids (indices into `items`) of the items writing to it, in
# CSR-like form. Since items are enumerated in order the per-cell lists are sorted.
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

# Conflict gather for items when all dofs written are cell-interior (purely
# discontinuous fields): two items conflict iff they write to a common cell. For
# interfaces this is the "line graph" of the facet adjacency, which needs at most Δ + 1
# colors where Δ is the maximum number of facet neighbors of a cell.
function _gather_item_cell_chunk!(count, items, chunk, iptr, iadj, masters)
    buf = Int[]
    candidates = Int[]
    for k in chunk
        empty!(candidates)
        for c in _item_cells(items[k])
            for r in iptr[c]:(iptr[c + 1] - 1)
                j = iadj[r]
                j == k || push!(candidates, j)
            end
        end
        _append_master_conflicts!(candidates, masters, k)
        _append_candidates_sorted_unique!(buf, count, candidates, k)
    end
    return buf
end

# Conservative conflict gather for items: when continuous fields are written the item
# also writes dofs shared with the node neighbors of its cells, so two items conflict iff
# a cell of one is node-adjacent to (or equal to) a cell of the other. This is a superset
# of the common-cell graph above.
function _gather_item_node_chunk!(count, grid, items, chunk, nodeptr, nodecells, iptr, iadj, masters)
    buf = Int[]
    candidates = Int[]
    for k in chunk
        empty!(candidates)
        for c in _item_cells(items[k])
            for v in get_node_ids(getcells(grid, c))
                for r in nodeptr[v]:(nodeptr[v + 1] - 1)
                    # Cell node-adjacent to (or equal to) c: all its items conflict
                    d = nodecells[r]
                    for r2 in iptr[d]:(iptr[d + 1] - 1)
                        j = iadj[r2]
                        j == k || push!(candidates, j)
                    end
                end
            end
        end
        _append_master_conflicts!(candidates, masters, k)
        _append_candidates_sorted_unique!(buf, count, candidates, k)
    end
    return buf
end

# Conflicts from condensation of affine constraints during assembly (`apply_assemble!`):
# condensing the local matrix and vector of an item with (global) dofs S writes to the
# global entries of S ∪ M(S), where M(S) are the master dofs of the affinely constrained
# dofs in S (see `_condense_local!`). The S ∩ S conflicts are covered by the cell/node
# based gathers above, so the remaining conflicts are exactly the pairs of items
# writing to a common master dof (through either S or M(S)). Note that these can be far
# apart in the mesh, e.g. on opposite sides of the domain for periodic constraints.
#
# Returns `nothing` if there are no affine constraints, and otherwise the map from item
# to (indices of) masters it writes to, and the inverse map from master to items, both
# in CSR-like form. Masters are indexed in order of discovery.
function _item_master_map(ch::ConstraintHandler, items)
    dh = ch.dh
    n = ndofs(dh)
    # Map dofs to master indices and affinely constrained dofs to the indices of their
    # masters. Note that `close!(ch)` has already resolved recursive constraints, so the
    # coefficient lists are final.
    master_of = zeros(Int, n) # dof -> master index, or 0
    affine_of = zeros(Int, n) # dof -> index into affine_masters, or 0
    nmasters = 0
    affine_masters = Vector{Int}[]
    for (i, d) in pairs(ch.prescribed_dofs)
        coefficients = ch.dofcoefficients[i]
        (coefficients === nothing || isempty(coefficients)) && continue
        master_ids = Int[]
        for (m, _) in coefficients
            if master_of[m] == 0
                nmasters += 1
                master_of[m] = nmasters
            end
            push!(master_ids, master_of[m])
        end
        push!(affine_masters, master_ids)
        affine_of[d] = length(affine_masters)
    end
    nmasters == 0 && return nothing
    # Item -> masters
    kptr = Vector{Int}(undef, length(items) + 1)
    kptr[1] = 1
    kadj = Int[]
    dofbuf = Int[]
    for (k, item) in pairs(items)
        start = length(kadj) + 1
        for c in _item_cells(item)
            resize!(dofbuf, ndofs_per_cell(dh, c))
            celldofs!(dofbuf, dh, c)
            for dof in dofbuf
                mi = master_of[dof]
                mi == 0 || push!(kadj, mi)
                ai = affine_of[dof]
                ai == 0 || append!(kadj, affine_masters[ai])
            end
        end
        # Sort and deduplicate the masters of this item
        sort!(view(kadj, start:length(kadj)))
        last = start - 1
        for r in start:length(kadj)
            if last < start || kadj[r] != kadj[last]
                last += 1
                kadj[last] = kadj[r]
            end
        end
        resize!(kadj, last)
        kptr[k + 1] = length(kadj) + 1
    end
    # Master -> items (transpose)
    mptr = zeros(Int, nmasters + 1)
    mptr[1] = 1
    for mi in kadj
        mptr[mi + 1] += 1
    end
    for i in 2:(nmasters + 1)
        mptr[i] += mptr[i - 1]
    end
    madj = Vector{Int}(undef, length(kadj))
    cursor = copy(mptr)
    for k in eachindex(items)
        for r in kptr[k]:(kptr[k + 1] - 1)
            mi = kadj[r]
            madj[cursor[mi]] = k
            cursor[mi] += 1
        end
    end
    return (kptr, kadj, mptr, madj)
end

_append_master_conflicts!(candidates, ::Nothing, k) = candidates
function _append_master_conflicts!(candidates, (kptr, kadj, mptr, madj), k)
    for r in kptr[k]:(kptr[k + 1] - 1)
        mi = kadj[r]
        for r2 in mptr[mi]:(mptr[mi + 1] - 1)
            j = madj[r2]
            j == k || push!(candidates, j)
        end
    end
    return candidates
end

# The sorted unique cells written to by the items
function _item_cellvec(items)
    cells = Int[]
    for item in items
        append!(cells, _item_cells(item))
    end
    return unique!(sort!(cells))
end

function _color_items(
        grid::AbstractGrid, items::AbstractVector{T}, alg::ColoringAlgorithm.T,
        cells_share_dofs::Bool, ch::Union{Nothing, ConstraintHandler}
    ) where {T}
    nitems = length(items)
    if nitems == 0
        return Vector{T}[]
    end
    iptr, iadj = _cell_to_item_map(getncells(grid), items)
    masters = ch === nothing ? nothing : _item_master_map(ch, items)
    local colptr, rowval
    if !cells_share_dofs
        colptr, rowval = _chunked_gather!(nitems, 1:nitems) do counts, chunk
            _gather_item_cell_chunk!(counts, items, chunk, iptr, iadj, masters)
        end
    else
        # Only node sharing between cells of the given items matters (other cells are
        # not assembled), so the node map over those cells captures all conflicts.
        cellvec = _item_cellvec(items)
        nodeptr, nodecells = _build_node_to_cell_map(grid, cellvec)
        colptr, rowval = _chunked_gather!(nitems, 1:nitems) do counts, chunk
            _gather_item_node_chunk!(counts, grid, items, chunk, nodeptr, nodecells, iptr, iadj, masters)
        end
    end
    incidence_matrix = SparseMatrixCSC(nitems, nitems, colptr, rowval, fill(true, length(rowval)))
    id_colors = _color_incidence_matrix(incidence_matrix, 1:nitems, alg)
    return Vector{T}[items[ids] for ids in id_colors]
end
