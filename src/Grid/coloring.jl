# Split `1:n` into at most `maxchunks` contiguous ranges of similar size.
function _color_chunks(n::Int, maxchunks::Int)
    return Iterators.partition(1:n, cld(n, min(n, maxchunks)))
end

function _gather_neighbor_chunk!(colcount, g, cellvec, chunk, nodeptr, nodecells)
    buf = Int[]
    candidates = Int[]
    for i in chunk
        cellid = cellvec[i]
        empty!(candidates)
        for v in get_node_ids(getcells(g, cellid))
            for r in nodeptr[v]:(nodeptr[v + 1] - 1)
                cell_neighbour = nodecells[r]
                cell_neighbour == cellid || push!(candidates, cell_neighbour)
            end
        end
        sort!(candidates)
        prev = 0
        for cell_neighbour in candidates
            if cell_neighbour != prev
                push!(buf, cell_neighbour)
                colcount[cellid] += 1
                prev = cell_neighbour
            end
        end
    end
    return buf
end

# Incidence matrix for element connections in the grid
function create_incidence_matrix(g::AbstractGrid, cellset = 1:getncells(g))
    ncells = getncells(g)
    cellvec = sort!(collect(Int, cellset))
    if isempty(cellvec)
        return SparseArrays.spzeros(Bool, Int, ncells, ncells)
    end

    # Map from node id to the cells in the cellset containing it, in CSR-like form.
    # Since cells are visited in ascending order each per-node cell list is sorted.
    nnodes = getnnodes(g)
    nodeptr = zeros(Int, nnodes + 1)
    nodeptr[1] = 1
    for cellid in cellvec
        for v in get_node_ids(getcells(g, cellid))
            nodeptr[v + 1] += 1
        end
    end
    for i in 2:(nnodes + 1)
        nodeptr[i] += nodeptr[i - 1]
    end
    nodecells = Vector{Int}(undef, nodeptr[end] - 1)
    cursor = copy(nodeptr)
    for cellid in cellvec
        for v in get_node_ids(getcells(g, cellid))
            nodecells[cursor[v]] = cellid
            cursor[v] += 1
        end
    end

    # For each cell, gather the unique cells sharing at least one node with it.
    # Since `cellvec` is sorted, each chunk corresponds to a contiguous range of
    # columns in the CSC structure, so the chunk buffers can be computed in
    # parallel and copied over without synchronization.
    chunks = collect(_color_chunks(length(cellvec), Threads.nthreads()))
    colcount = zeros(Int, ncells)
    buffers = Vector{Vector{Int}}(undef, length(chunks))
    @sync for (ci, chunk) in enumerate(chunks)
        Threads.@spawn begin
            buffers[$ci] = _gather_neighbor_chunk!(colcount, g, cellvec, $chunk, nodeptr, nodecells)
        end
    end

    colptr = Vector{Int}(undef, ncells + 1)
    colptr[1] = 1
    for c in 1:ncells
        colptr[c + 1] = colptr[c] + colcount[c]
    end
    rowval = Vector{Int}(undef, colptr[end] - 1)
    @sync for (ci, chunk) in enumerate(chunks)
        Threads.@spawn begin
            buf = buffers[$ci]
            copyto!(rowval, colptr[cellvec[first($chunk)]], buf, 1, length(buf))
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

    if length(cellset) == 0
        return Vector{Int}[]
    elseif length(cellset) == 1
        return Vector{Int}[Int[first(cellset)]]
    end
    ncells = size(incidence_matrix, 1)

    ###################
    # 1. Partitioning #
    ###################
    # Note: the incidence matrix is assumed to be created with the same cellset, so all
    # neighbors found through it are members of the cellset.
    cellvec = sort!(collect(Int, cellset))
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
    if alg === ColoringAlgorithm.WorkStream
        return workstream_coloring(incidence_matrix, cellset)
    elseif alg === ColoringAlgorithm.Greedy
        return greedy_coloring(incidence_matrix, cellset)
    else
        error("impossible")
    end
end
