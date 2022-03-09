# Incidence matrix for element connections in the grid
function create_incidence_matrix(g::Grid, cellset::Set{Int}=Set{Int}(1:getncells(g)))
    cell_containing_node = Dict{Int, Set{Int}}()
    for cellid in cellset
        cell = getcells(g, cellid)
        for v in cell.nodes
            if !haskey(cell_containing_node, v)
                cell_containing_node[v] = Set{Int}()
            end
            push!(cell_containing_node[v], cellid)
        end
    end

    I, J, V = Int[], Int[], Bool[]
    for (_, cells) in cell_containing_node
        for cell1 in cells # All these cells have a neighboring node
            for cell2 in cells
                # if true # cell1 != cell2
                if cell1 != cell2
                    push!(I, cell1)
                    push!(J, cell2)
                    push!(V, true)
                end
            end
        end
    end

    incidence_matrix = sparse(I, J, V)
    return incidence_matrix
end

# Greedy algorithm for coloring a grid such that no two cells with the same node
# have the same color
function greedy_coloring(incidence_matrix, cells=1:size(incidence_matrix, 1))
    cell_colors = Dict{Int,Int}(i => 0 for i in cells) # Zero represents no color set yet
    occupied_colors = Set{Int}()
    final_colors = Vector{Int}[]
    total_colors = 0
    for cellid in cells
        empty!(occupied_colors)
        # loop over neighbors
        for r in nzrange(incidence_matrix, cellid)
            cell_neighbour = incidence_matrix.rowval[r]
            cell_neighbour in cells || continue # Only care about the subset given in cells
            color = cell_colors[cell_neighbour]
            if color != 0
                push!(occupied_colors, color)
            end
        end

        # occupied colors now contains all the colors we are not allowed to use
        free_color = 0
        for attempt_color in 1:total_colors
            if attempt_color ∉ occupied_colors
                free_color = attempt_color
                break
            end
        end
        if free_color == 0 # no free color found, need to bump max colors
            total_colors += 1
            free_color = total_colors
            push!(final_colors, Int[])
        end
        @assert free_color != 0
        cell_colors[cellid] = free_color
        push!(final_colors[free_color], cellid)
    end
    return final_colors
end

# See Appendix A in https://www.math.colostate.edu/%7Ebangerth/publications/2013-pattern.pdf
function workstream_coloring(incidence_matrix, cellset::Set{Int})
    ###################
    # 1. Partitioning #
    ###################
    zones = Set{Int}[]
    n_visited = 0
    Z = 1
    Z0 = Set{Int}() # Dummy zone
    while n_visited < length(cellset)
        remaining_cellset = setdiff(cellset, zones...)
        ## Zone 1: Just the first element
        push!(zones, Set{Int}(first(remaining_cellset)))
        Z += 1
        n_visited += 1
        ## Zone N: All elements with connection to elements in Zone N-1
        while true
            s = Set{Int}()
            # Loop over all elements in previous zone and add their neigbouring elements
            # unless they are in any of the previous 2 zones.
            empty_zone = true
            for c in get(zones, Z-1, Z0)
                c ∈ cellset || continue # technically not needed as long as incidence matrix is created with cellset
                for r in nzrange(incidence_matrix, c)
                    cell_neighbour = incidence_matrix.rowval[r]
                    if !(cell_neighbour in get(zones, Z-2, Z0) || cell_neighbour in get(zones, Z-1, Z0))
                        push!(s, cell_neighbour)
                        empty_zone = false
                    end
                end
            end
            empty_zone && break # no more cells connected to previous zone

            push!(zones, s)
            n_visited += length(s)
            Z += 1
        end
    end

    ###############
    # 2. Coloring #
    ###############
    # TODO: The reference uses DSATUR algorithm instead of greedy
    # TODO: Zones can be colorized in parallel
    zone_colors = Vector{Vector{Int}}[greedy_coloring(incidence_matrix, z) for z in zones]

    ################
    # 3. Gathering #
    ################
    Nodd,  Zodd  = findmax(x -> isodd(x)  ? length(zone_colors[x]) : typemin(Int), 1:length(zone_colors))
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

        for local_color in sortperm(zone_color_vectors; by=length, rev=true)
            cond = odd ? (x -> x <= Nodd) : (x -> x > Nodd)
            _, global_color = findmin(x -> (cond(x) && x ∉ used_for_zone) ? color_sizes[x] : typemax(Int), 1:N)
            push!(used_for_zone, global_color)
            append!(final_colors[global_color], zone_color_vectors[local_color])
            map!(length, color_sizes, final_colors)
        end
    end

    # Maybe nice to sort?
    foreach(sort!, final_colors)

    return final_colors
end

# @enum ColoringAlgorithm GREEDY WORKSTREAM
baremodule ColoringAlgorithm
    struct T x::Int end
    const Greedy = T(0)
    const WorkStream = T(1)
end
const GREEDY = ColoringAlgorithm.Greedy
const WORKSTREAM = ColoringAlgorithm.WorkStream

"""
    create_coloring(g::Grid, cellset::Set{Int}=Set(1:getncells(g)); alg::ColoringAlgorithm)

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
 - `alg = Ferrite.ColoringAlgorithm.WorkStream` (default): Three step algorithm from
   [*WorkStream*](https://www.math.colostate.edu/%7Ebangerth/publications/2013-pattern.pdf)
   , albeit with a greedy coloring in the second step. Generally results in more colors than `Ferrite.GREEDY`,
   however the cells are more equally distributed among the colors.
 - `alg = Ferrite.ColoringAlgorithm.Greedy`: greedy algorithm that works well for structured quadrilateral grids such as
   e.g. quadrilateral grids from `generate_grid`.

The resulting colors can be visualized using [`vtk_cell_data_colors`](@ref).

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
"""
function create_coloring(g::Grid, cellset::Set{Int}=Set{Int}(1:getncells(g)); alg::ColoringAlgorithm.T=ColoringAlgorithm.WorkStream)
    incidence_matrix = create_incidence_matrix(g, cellset)
    if alg === ColoringAlgorithm.WorkStream
        return workstream_coloring(incidence_matrix, cellset)
    elseif alg === ColoringAlgorithm.Greedy
        return greedy_coloring(incidence_matrix, cellset)
    else
        error("impossible")
    end
end

"""
    vtk_cell_data_colors(vtkfile, cell_colors, name="coloring")

Write cell colors (see [`create_coloring`](@ref)) to a VTK file for visualization.

In case of coloring a subset, the cells which are not part of the subset are represented as color 0.
"""
function vtk_cell_data_colors(vtkfile, cell_colors::AbstractVector{<:AbstractVector{<:Integer}}, name="coloring")
    color_vector = zeros(Int, vtkfile.Ncls)
    for (i, cells_color) in enumerate(cell_colors)
        for cell in cells_color
            color_vector[cell] = i
        end
    end
    vtk_cell_data(vtkfile, color_vector, name)
end
