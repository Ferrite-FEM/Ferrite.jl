#----------------------------------------------------------------------#
# Performance benchmarks for AMR (Adaptive Mesh Refinement) in Ferrite
#----------------------------------------------------------------------#
# Run standalone:
#   julia --project benchmark/benchmarks-amr.jl
# Or via the benchmark suite:
#   FERRITE_SELECTED_BENCHMARKS=amr julia --project benchmark/runbenchmarks.jl benchmark/benchmarks.jl

using Ferrite
using Ferrite.AMR
using Printf

#==============================================================================#
# Helpers
#==============================================================================#

"""
    create_forest(dim, n; max_level)
Create a ForestBWG from an n^dim hypercube grid.
"""
function create_forest(::Val{2}, n; max_level = 5)
    grid = generate_grid(Quadrilateral, (n, n))
    return ForestBWG(grid, max_level)
end

function create_forest(::Val{3}, n; max_level = 5)
    grid = generate_grid(Hexahedron, (n, n, n))
    return ForestBWG(grid, max_level)
end

"""
    create_refined_forest(dim, n, levels; kwargs...)
Create a forest, uniformly refine it `levels` times, then refine a subset
of cells (the first quarter) to create a non-trivial adaptive state.
"""
function create_refined_forest(dim::Val{D}, n, uniform_levels; adaptive_fraction = 0.25, max_level = 8) where {D}
    forest = create_forest(dim, n; max_level)
    for l in 1:uniform_levels
        refine_all!(forest, l)
    end
    # Refine a fraction of cells to create non-uniformity
    ncells = Ferrite.getncells(forest)
    n_to_refine = max(1, round(Int, ncells * adaptive_fraction))
    cells_to_refine = collect(1:n_to_refine)
    refine!(forest, cells_to_refine)
    return forest
end

#==============================================================================#
# Individual component benchmarks
#==============================================================================#

function benchmark_refine_all(dim, n, levels; max_level = 8)
    forest = create_forest(dim, n; max_level)
    # Warm up
    forest_copy = deepcopy(forest)
    for l in 1:levels
        refine_all!(forest_copy, l)
    end
    # Timed
    times = Float64[]
    for _ in 1:5
        forest_run = create_forest(dim, n; max_level)
        t = @elapsed begin
            for l in 1:levels
                refine_all!(forest_run, l)
            end
        end
        push!(times, t)
    end
    ncells = Ferrite.getncells(forest_copy)
    return minimum(times), ncells
end

function benchmark_adaptive_refine(forest_template, fraction)
    # Refine a fraction of cells
    times = Float64[]
    for _ in 1:5
        forest = deepcopy(forest_template)
        ncells = Ferrite.getncells(forest)
        n_to_refine = max(1, round(Int, ncells * fraction))
        cells_to_refine = collect(1:n_to_refine)
        t = @elapsed refine!(forest, cells_to_refine)
        push!(times, t)
    end
    return minimum(times)
end

function benchmark_balancetree(forest_template)
    times = Float64[]
    for _ in 1:5
        forest = deepcopy(forest_template)
        t = @elapsed begin
            for k in 1:length(forest.cells)
                forest.cells[k] = Ferrite.AMR.balancetree(forest.cells[k])
            end
        end
        push!(times, t)
    end
    return minimum(times)
end

function benchmark_balanceforest(forest_template)
    times = Float64[]
    for _ in 1:5
        forest = deepcopy(forest_template)
        t = @elapsed balanceforest!(forest)
        push!(times, t)
    end
    return minimum(times)
end

function benchmark_creategrid(forest_template)
    # Forest must be balanced first
    forest = deepcopy(forest_template)
    balanceforest!(forest)
    times = Float64[]
    for _ in 1:5
        t = @elapsed creategrid(forest)
        push!(times, t)
    end
    ncells = Ferrite.getncells(forest)
    return minimum(times), ncells
end

#==============================================================================#
# Full pipeline benchmark
#==============================================================================#

function benchmark_full_pipeline(dim, n, uniform_levels; max_level = 8, adaptive_fraction = 0.25)
    # Step 1: Forest creation
    t_create = @elapsed forest = create_forest(dim, n; max_level)
    ncells_base = Ferrite.getncells(forest)

    # Step 2: Uniform refinement
    t_uniform = @elapsed begin
        for l in 1:uniform_levels
            refine_all!(forest, l)
        end
    end
    ncells_uniform = Ferrite.getncells(forest)

    # Step 3: Adaptive refinement
    n_to_refine = max(1, round(Int, ncells_uniform * adaptive_fraction))
    cells_to_refine = collect(1:n_to_refine)
    t_adaptive = @elapsed refine!(forest, cells_to_refine)
    ncells_adaptive = Ferrite.getncells(forest)

    # Step 4: Balancing (individual tree + inter-tree)
    t_balance_trees = @elapsed begin
        for k in 1:length(forest.cells)
            forest.cells[k] = Ferrite.AMR.balancetree(forest.cells[k])
        end
    end
    # Now full forest balance (inter-tree)
    t_balance_forest = @elapsed balanceforest!(forest)
    ncells_balanced = Ferrite.getncells(forest)

    # Step 5: Grid creation (includes hanging nodes)
    t_creategrid = @elapsed grid = creategrid(forest)

    return (;
        dim = dim isa Val{2} ? 2 : 3,
        base_grid_n = n,
        uniform_levels,
        max_level,
        adaptive_fraction,
        ncells_base,
        ncells_uniform,
        ncells_adaptive,
        ncells_balanced,
        ncells_grid = length(grid.cells),
        n_hanging_nodes = length(grid.conformity_info),
        t_create,
        t_uniform,
        t_adaptive,
        t_balance_trees,
        t_balance_forest,
        t_creategrid,
        t_total = t_create + t_uniform + t_adaptive + t_balance_trees + t_balance_forest + t_creategrid,
    )
end

#==============================================================================#
# Scaling study
#==============================================================================#

function scaling_study(dim; ns = [2, 4, 8], levels = [1, 2, 3], max_level = 8)
    results = []
    for n in ns
        for lvl in levels
            # Skip combinations that would create too many cells
            estimated_cells = n^(dim isa Val{2} ? 2 : 3) * (2^(dim isa Val{2} ? 2 : 3))^lvl
            if estimated_cells > 500_000
                continue
            end
            r = benchmark_full_pipeline(dim, n, lvl; max_level)
            push!(results, r)
        end
    end
    return results
end

#==============================================================================#
# Pretty printing
#==============================================================================#

function print_results(results)
    println("="^120)
    println("AMR Performance Benchmark Results")
    println("="^120)
    header = @sprintf(
        "%-5s %-6s %-6s %-10s %-10s %-10s %-10s %-8s | %-10s %-10s %-10s %-10s %-10s %-10s",
        "dim", "n", "lvls", "#base", "#uniform", "#adapted", "#balanced", "#hang",
        "create", "uniform", "adaptive", "bal_tree", "bal_forest", "creategrid"
    )
    println(header)
    println("-"^120)
    for r in results
        line = @sprintf(
            "%-5d %-6d %-6d %-10d %-10d %-10d %-10d %-8d | %-10s %-10s %-10s %-10s %-10s %-10s",
            r.dim, r.base_grid_n, r.uniform_levels,
            r.ncells_base, r.ncells_uniform, r.ncells_adaptive, r.ncells_balanced, r.n_hanging_nodes,
            fmt_time(r.t_create), fmt_time(r.t_uniform), fmt_time(r.t_adaptive),
            fmt_time(r.t_balance_trees), fmt_time(r.t_balance_forest), fmt_time(r.t_creategrid)
        )
        println(line)
    end
    println("="^120)

    # Print percentage breakdown for the largest case
    return if !isempty(results)
        r = results[end]
        total = r.t_total
        println("\nPercentage breakdown (largest case: $(r.dim)D, n=$(r.base_grid_n), $(r.uniform_levels) levels, $(r.ncells_balanced) cells):")
        println("  Forest creation:       $(fmt_time(r.t_create))  ($(@sprintf("%.1f", 100 * r.t_create / total))%)")
        println("  Uniform refinement:    $(fmt_time(r.t_uniform))  ($(@sprintf("%.1f", 100 * r.t_uniform / total))%)")
        println("  Adaptive refinement:   $(fmt_time(r.t_adaptive))  ($(@sprintf("%.1f", 100 * r.t_adaptive / total))%)")
        println("  Balance (per-tree):    $(fmt_time(r.t_balance_trees))  ($(@sprintf("%.1f", 100 * r.t_balance_trees / total))%)")
        println("  Balance (forest):      $(fmt_time(r.t_balance_forest))  ($(@sprintf("%.1f", 100 * r.t_balance_forest / total))%)")
        println("  creategrid:            $(fmt_time(r.t_creategrid))  ($(@sprintf("%.1f", 100 * r.t_creategrid / total))%)")
        println("  TOTAL:                 $(fmt_time(total))")
    end
end

function fmt_time(t)
    if t < 1.0e-3
        return @sprintf("%.1f μs", t * 1.0e6)
    elseif t < 1.0
        return @sprintf("%.2f ms", t * 1.0e3)
    else
        return @sprintf("%.3f s", t)
    end
end

#==============================================================================#
# Detailed profiling of balancetree internals
#==============================================================================#

"""
    profile_balancetree(forest)
Profiles the internal operations of balancetree to identify hotspots.
Returns timing breakdown of: sorting, parent computation, possibleneighbors, linearise.
"""
function profile_balancetree(forest)
    println("\n" * "="^80)
    println("Detailed balancetree profiling")
    println("="^80)

    for (k, tree) in enumerate(forest.cells)
        nleaves = length(tree.leaves)
        nleaves < 10 && continue  # Skip trivial trees

        println("\nTree $k: $(nleaves) leaves, max_level=$(tree.b)")

        # Profile the components
        W = copy(tree.leaves)
        t_sort = 0.0
        t_parent = 0.0
        t_neighbors = 0.0
        t_filter = 0.0
        n_parent_checks = 0
        n_neighbor_calls = 0

        P = eltype(tree.leaves)[]
        R = eltype(tree.leaves)[]
        for l in tree.b:-1:1
            Q = [o for o in W if o.l == l]
            isempty(Q) && continue

            t_sort += @elapsed sort!(Q)

            T_set = eltype(Q)[]
            t_parent += @elapsed begin
                for x in Q
                    if isempty(T_set)
                        push!(T_set, x)
                        continue
                    end
                    p = Ferrite.AMR.parent(x, tree.b)
                    # This is an O(|T|) check per element!
                    n_parent_checks += length(T_set)
                    if p ∉ Ferrite.AMR.parent.(T_set, (tree.b,))
                        push!(T_set, x)
                    end
                end
            end

            t_neighbors += @elapsed begin
                for t in T_set
                    push!(R, t, Ferrite.AMR.siblings(t, tree.b)...)
                    n_neighbor_calls += 1
                    push!(P, Ferrite.AMR.possibleneighbors(Ferrite.AMR.parent(t, tree.b), l - 1, tree.b)...)
                end
            end

            t_filter += @elapsed begin
                append!(P, x for x in W if x.l == l - 1)
                filter!(x -> !(x.l == l - 1), W)
                unique!(P)
                append!(W, P)
                empty!(P)
            end
        end

        t_final_sort = @elapsed sort!(R)
        t_linearise = @elapsed Ferrite.AMR.linearise!(R, tree.b)

        total = t_sort + t_parent + t_neighbors + t_filter + t_final_sort + t_linearise
        println("  Sorting Q:          $(fmt_time(t_sort))  ($(@sprintf("%.1f", 100 * t_sort / max(total, 1.0e-15)))%)")
        println("  Parent checks:      $(fmt_time(t_parent))  ($(@sprintf("%.1f", 100 * t_parent / max(total, 1.0e-15)))%) [$(n_parent_checks) comparisons]")
        println("  Neighbors+siblings: $(fmt_time(t_neighbors))  ($(@sprintf("%.1f", 100 * t_neighbors / max(total, 1.0e-15)))%) [$(n_neighbor_calls) calls]")
        println("  Filter/unique/W:    $(fmt_time(t_filter))  ($(@sprintf("%.1f", 100 * t_filter / max(total, 1.0e-15)))%)")
        println("  Final sort:         $(fmt_time(t_final_sort))  ($(@sprintf("%.1f", 100 * t_final_sort / max(total, 1.0e-15)))%)")
        println("  Linearise:          $(fmt_time(t_linearise))  ($(@sprintf("%.1f", 100 * t_linearise / max(total, 1.0e-15)))%)")
        println("  Total:              $(fmt_time(total))")
        println("  Result leaves:      $(length(R))")

        # Only profile first non-trivial tree
        break
    end
    return
end

"""
    profile_balanceforest(forest_template)
Profiles the forest balancing algorithm to identify inter-tree overhead.
"""
function profile_balanceforest(forest_template)
    println("\n" * "="^80)
    println("Detailed balanceforest profiling")
    println("="^80)

    forest = deepcopy(forest_template)

    dim = forest.cells[1] isa Ferrite.AMR.OctreeBWG{3} ? 3 : 2
    root_ = Ferrite.AMR.root(dim)

    iteration = 0
    nrefcells = 0
    t_pertree_balance = 0.0
    t_possibleneighbors = 0.0
    t_inside_check = 0.0
    t_interoctree = 0.0
    n_interoctree_calls = 0
    n_total_leaves_processed = 0

    facet_neighborhood = Ferrite.get_facet_facet_neighborhood(forest)

    while nrefcells - Ferrite.getncells(forest) != 0
        iteration += 1
        nrefcells = Ferrite.getncells(forest)

        t_iter_tree = @elapsed begin
            for k in 1:length(forest.cells)
                forest.cells[k] = Ferrite.AMR.balancetree(forest.cells[k])
            end
        end
        t_pertree_balance += t_iter_tree

        t_iter_neigh = 0.0
        t_iter_inside = 0.0
        t_iter_inter = 0.0
        n_iter_inter = 0
        n_iter_leaves = 0

        for k in 1:length(forest.cells)
            tree = forest.cells[k]
            for (o_i, o) in enumerate(tree.leaves)
                n_iter_leaves += 1
                t0 = time_ns()
                ss = Ferrite.AMR.possibleneighbors(o, o.l, tree.b; insidetree = false)
                t_iter_neigh += (time_ns() - t0) / 1.0e9

                t0 = time_ns()
                isinside = Ferrite.AMR.inside.(ss, (tree.b,))
                notinsideidx = findall(.! isinside)
                t_iter_inside += (time_ns() - t0) / 1.0e9

                if !isempty(notinsideidx)
                    t0 = time_ns()
                    # Count inter-octree operations (don't actually do them to avoid double-balancing)
                    n_iter_inter += length(notinsideidx)
                    t_iter_inter += (time_ns() - t0) / 1.0e9
                end
            end
        end

        t_possibleneighbors += t_iter_neigh
        t_inside_check += t_iter_inside
        t_interoctree += t_iter_inter
        n_interoctree_calls += n_iter_inter
        n_total_leaves_processed += n_iter_leaves

        println("  Iteration $iteration: $(Ferrite.getncells(forest)) cells, per-tree balance: $(fmt_time(t_iter_tree))")
    end

    println("\nSummary after $iteration iterations:")
    println("  Per-tree balance total:   $(fmt_time(t_pertree_balance))")
    println("  possibleneighbors total:  $(fmt_time(t_possibleneighbors))")
    println("  inside check total:       $(fmt_time(t_inside_check))")
    println("  Inter-octree overhead:    $(fmt_time(t_interoctree)) [$(n_interoctree_calls) calls]")
    return println("  Total leaves processed:   $(n_total_leaves_processed)")
end

"""
    profile_creategrid(forest)
Profiles the grid creation phases individually.
"""
function profile_creategrid(forest_template)
    println("\n" * "="^80)
    println("Detailed creategrid profiling")
    println("="^80)

    forest = deepcopy(forest_template)
    balanceforest!(forest)
    ncells = Ferrite.getncells(forest)

    # The IBWG2015 LNodes `creategrid` is a single integer pass (no separate node-assignment
    # phases to profile in isolation). Time the whole call plus its two public sub-components:
    # the base Morton descent, hanging-node detection, and facetset reconstruction.
    t_descent = @elapsed (n = 0; Ferrite.AMR.iterate_leaves(_ -> (n += 1), forest))
    println("  iterate_leaves (descent):  $(fmt_time(t_descent))  [$(n) leaves visited]")

    t_hang = @elapsed hnodes = Ferrite.AMR.iterate_hanging(forest)
    println("  iterate_hanging:           $(fmt_time(t_hang))  [$(length(hnodes)) constraints]")

    t_facets = @elapsed Ferrite.AMR.reconstruct_facetsets(forest)
    println("  reconstruct_facetsets:     $(fmt_time(t_facets))")

    t_total = @elapsed grid = creategrid(forest)
    println("  creategrid (TOTAL):        $(fmt_time(t_total))  [$(ncells) cells, $(getnnodes(grid)) nodes]")
    return
end

#==============================================================================#
# Algorithmic complexity analysis
#==============================================================================#

"""
    analyze_balance_complexity(dim)
Tests how balancetree and balanceforest scale with problem size.
Identifies whether we see O(n), O(n log n), or worse behavior.
"""
function analyze_balance_complexity(dim; ns = [2, 4, 8], levels = [1, 2, 3], max_level = 8)
    println("\n" * "="^80)
    println("Balance algorithm complexity analysis ($(dim isa Val{2} ? "2" : "3")D)")
    println("="^80)
    D = dim isa Val{2} ? 2 : 3

    println(
        @sprintf(
            "\n%-10s %-12s %-15s %-15s %-15s %-15s",
            "n", "#trees", "#leaves_pre", "#leaves_post", "t_balancetree", "t_balanceforest"
        )
    )
    println("-"^85)

    for n in ns
        for lvl in levels
            estimated = n^D * (2^D)^lvl
            estimated > 200_000 && continue

            forest = create_forest(dim, n; max_level)
            for l in 1:lvl
                refine_all!(forest, l)
            end
            # Add some non-uniformity
            ncells = Ferrite.getncells(forest)
            n_to_refine = max(1, round(Int, ncells * 0.25))
            refine!(forest, collect(1:n_to_refine))

            nleaves_pre = Ferrite.getncells(forest)
            ntrees = length(forest.cells)

            # Benchmark balancetree (individual trees)
            forest_copy = deepcopy(forest)
            t_bt = @elapsed begin
                for k in 1:length(forest_copy.cells)
                    forest_copy.cells[k] = Ferrite.AMR.balancetree(forest_copy.cells[k])
                end
            end

            # Benchmark full forest balance
            forest_copy2 = deepcopy(forest)
            t_bf = @elapsed balanceforest!(forest_copy2)
            nleaves_post = Ferrite.getncells(forest_copy2)

            println(
                @sprintf(
                    "%-10d %-12d %-15d %-15d %-15s %-15s",
                    n, ntrees, nleaves_pre, nleaves_post, fmt_time(t_bt), fmt_time(t_bf)
                )
            )
        end
    end
    return
end

#==============================================================================#
# Memory analysis
#==============================================================================#

function analyze_memory(dim; n = 4, levels = 2, max_level = 8)
    println("\n" * "="^80)
    println("Memory analysis ($(dim isa Val{2} ? "2" : "3")D, n=$n, $levels levels)")
    println("="^80)

    forest = create_forest(dim, n; max_level)
    for l in 1:levels
        refine_all!(forest, l)
    end
    ncells = Ferrite.getncells(forest)
    n_to_refine = max(1, round(Int, ncells * 0.25))
    refine!(forest, collect(1:n_to_refine))

    GC.gc()
    mem_before = Base.gc_live_bytes()

    forest_balanced = deepcopy(forest)
    balanceforest!(forest_balanced)

    GC.gc()
    mem_after_balance = Base.gc_live_bytes()

    grid = creategrid(forest_balanced)

    GC.gc()
    mem_after_grid = Base.gc_live_bytes()

    println("  Cells before balance: $(Ferrite.getncells(forest))")
    println("  Cells after balance:  $(Ferrite.getncells(forest_balanced))")
    println("  Grid cells:           $(length(grid.cells))")
    println("  Hanging nodes:        $(length(grid.conformity_info))")
    println("  Memory (approx):")
    println("    After balance: ~$(round((mem_after_balance - mem_before) / 1024, digits = 1)) KB")
    return println("    After grid:    ~$(round((mem_after_grid - mem_before) / 1024, digits = 1)) KB")
end

#==============================================================================#
# Identify specific hotspots
#==============================================================================#

"""
    identify_hotspots(dim, n, levels)
Runs the complete pipeline with detailed timing to identify the main bottlenecks.
"""
function identify_hotspots(dim; n = 4, levels = 2, max_level = 8)
    println("\n" * "="^80)
    println("HOTSPOT ANALYSIS ($(dim isa Val{2} ? "2" : "3")D, n=$n, $levels levels)")
    println("="^80)

    # Create non-trivially refined forest
    forest = create_refined_forest(dim, n, levels; max_level)
    ncells = Ferrite.getncells(forest)
    println("Forest: $(length(forest.cells)) trees, $ncells leaves total")

    # 1. Profile balancetree internals
    profile_balancetree(deepcopy(forest))

    # 2. Profile balanceforest
    profile_balanceforest(deepcopy(forest))

    # 3. Profile creategrid phases
    return profile_creategrid(deepcopy(forest))
end

#==============================================================================#
# Main entry point
#==============================================================================#

function run_all_benchmarks()
    println("Ferrite.jl AMR Performance Benchmark Suite")
    println("Julia $(VERSION)")
    println("Date: $(Dates.now())")
    println()

    # Warm-up (compile everything)
    println("Warming up...")
    _ = benchmark_full_pipeline(Val(2), 2, 1; max_level = 5)
    _ = benchmark_full_pipeline(Val(3), 2, 1; max_level = 5)
    println("Warm-up complete.\n")

    # 2D benchmarks
    results_2d = []
    for n in [2, 4, 8, 16]
        for lvl in [1, 2, 3]
            estimated = n^2 * 4^lvl
            estimated > 200_000 && continue
            r = benchmark_full_pipeline(Val(2), n, lvl; max_level = 8)
            push!(results_2d, r)
        end
    end

    # 3D benchmarks
    results_3d = []
    for n in [2, 4, 8]
        for lvl in [1, 2, 3]
            estimated = n^3 * 8^lvl
            estimated > 200_000 && continue
            r = benchmark_full_pipeline(Val(3), n, lvl; max_level = 8)
            push!(results_3d, r)
        end
    end

    println("\n2D RESULTS:")
    print_results(results_2d)

    println("\n\n3D RESULTS:")
    print_results(results_3d)

    # Detailed hotspot analysis on medium-sized problems
    identify_hotspots(Val(2); n = 8, levels = 3, max_level = 8)
    identify_hotspots(Val(3); n = 4, levels = 2, max_level = 8)

    # Complexity analysis
    analyze_balance_complexity(Val(2); ns = [2, 4, 8, 16], levels = [1, 2, 3])
    analyze_balance_complexity(Val(3); ns = [2, 4], levels = [1, 2, 3])

    # Memory analysis
    analyze_memory(Val(2); n = 8, levels = 3)
    analyze_memory(Val(3); n = 4, levels = 2)

    return (results_2d, results_3d)
end

# If run as a standalone script
if abspath(PROGRAM_FILE) == @__FILE__
    using Dates
    run_all_benchmarks()
end
