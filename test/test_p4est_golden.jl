# Regression + correctness tests for AMR (ForestBWG) grid materialization.
#
# TWO independent layers, because the current code may itself contain latent bugs:
#
#  1. GOLDEN (regression): a canonical, renumbering-invariant fingerprint of the
#     balanced forest and the materialized NonConformingGrid, frozen to files in
#     test/amr_golden/. Detects whether a refactor/optimization of `creategrid`,
#     `balanceforest!`, or `hangingnodes` *changed* the output. It pins current
#     behavior — it does NOT certify that behavior is correct.
#
#  2. INVARIANTS (correctness): properties that must hold for ANY valid mesh of
#     these inputs, independent of the current code. These can catch bugs that
#     the golden snapshot would otherwise enshrine. If one fails, fix the code —
#     do not regenerate the golden file.
#
# Regenerate golden references after an INTENTIONAL output change:
#   FERRITE_REGEN_AMR_GOLDEN=true julia --project test/runtests.jl

const _AMR = Ferrite.AMR
const GOLDEN_DIR = joinpath(@__DIR__, "amr_golden")
const REGEN_GOLDEN = get(ENV, "FERRITE_REGEN_AMR_GOLDEN", "false") == "true"

_r(x) = round(Float64(x); digits = 10)
_coord(c) = ntuple(i -> _r(c[i]), length(c))

# ---------------------------------------------------------------- golden layer

function canonical_grid(grid)
    nc = [_coord(Ferrite.get_node_coordinate(n)) for n in grid.nodes]
    dim = length(nc[1])
    cells = sort!([ntuple(i -> nc[cell.nodes[i]], length(cell.nodes)) for cell in grid.cells])
    nodes = sort(nc)
    hanging = sort!([(nc[h], sort([nc[m] for m in mas])) for (h, mas) in grid.conformity_info])
    facets = Tuple{String, Vector}[]
    for (name, set) in grid.facetsets
        cents = NTuple{dim, Float64}[]
        for fi in set
            fv = Ferrite.facets(grid.cells[fi[1]])[fi[2]]
            push!(cents, ntuple(d -> _r(sum(nc[v][d] for v in fv) / length(fv)), dim))
        end
        push!(facets, (name, sort!(cents)))
    end
    sort!(facets; by = first)
    return (; ncells = length(grid.cells), nnodes = length(grid.nodes),
        nhanging = length(grid.conformity_info), cells, nodes, hanging, facets)
end

function fingerprint_grid(grid)
    cg = canonical_grid(grid)
    io = IOBuffer()
    println(io, "ncells=", cg.ncells, " nnodes=", cg.nnodes, " nhanging=", cg.nhanging)
    println(io, "[cells]"); for c in cg.cells; println(io, c); end
    println(io, "[nodes]"); for n in cg.nodes; println(io, n); end
    println(io, "[hanging]"); for (h, ms) in cg.hanging; println(io, h, " <= ", ms); end
    println(io, "[facets]"); for (name, cents) in cg.facets; println(io, name, ": ", cents); end
    return String(take!(io))
end

function fingerprint_forest(forest)
    io = IOBuffer()
    for (k, tree) in enumerate(forest.cells)
        leaves = sort([(Int(o.l), Int.(o.xyz)) for o in tree.leaves])
        println(io, "tree ", k, " b=", tree.b, " nleaves=", length(leaves))
        for l in leaves; println(io, l); end
    end
    return String(take!(io))
end

function check_golden(name, forest, grid)
    fp = "===FOREST===\n" * fingerprint_forest(forest) * "===GRID===\n" * fingerprint_grid(grid)
    path = joinpath(GOLDEN_DIR, name * ".txt")
    if REGEN_GOLDEN || !isfile(path)
        mkpath(GOLDEN_DIR)
        write(path, fp)
        @test true  # bootstrapped reference
    else
        ref = read(path, String)
        fp == ref || @info "Golden mismatch for $name; if intentional, regen with FERRITE_REGEN_AMR_GOLDEN=true."
        @test fp == ref
    end
end

# ------------------------------------------------------------ invariant layer

# signed measure in cell-local (Ferrite) node order; > 0 means positively oriented
function signed_measure(v, ::Val{2}) # z-component of (v2-v1) × (v4-v1)
    return (v[2][1] - v[1][1]) * (v[4][2] - v[1][2]) - (v[2][2] - v[1][2]) * (v[4][1] - v[1][1])
end
function signed_measure(v, ::Val{3}) # triple product of edges from v1
    a = v[2] - v[1]; b = v[4] - v[1]; c = v[5] - v[1]
    return a[1] * (b[2] * c[3] - b[3] * c[2]) - a[2] * (b[1] * c[3] - b[3] * c[1]) + a[3] * (b[1] * c[2] - b[2] * c[1])
end

# vertex set forms an axis-aligned square/cube (order-independent)
function is_axis_aligned_box(verts, dim)
    length(unique(w -> ntuple(d -> _r(w[d]), dim), verts)) == 2^dim || return false
    for d in 1:dim
        length(unique(_r(w[d]) for w in verts)) == 2 || return false
    end
    sides = ntuple(d -> maximum(w[d] for w in verts) - minimum(w[d] for w in verts), dim)
    return all(s -> s > 1.0e-12, sides) # axis-aligned rectangular box (octree cells need not be cubes)
end

function check_invariants(name, grid; box::Bool)
    nc = [_coord(Ferrite.get_node_coordinate(n)) for n in grid.nodes]
    dim = length(nc[1])
    @testset "$name" begin
        # (1) no duplicate physical nodes
        @test length(unique(nc)) == length(nc)
        for cell in grid.cells
            ids = cell.nodes
            # (2) node indices in range
            @test all(i -> 1 <= i <= length(nc), ids)
            verts = [Ferrite.get_node_coordinate(grid.nodes[i]) for i in ids]
            # (3) non-degenerate, positively oriented (the det J > 0 check, incl. rotated trees)
            @test signed_measure(verts, Val(dim)) > 0
            # (4) correct geometric shape for structured grids
            box && @test is_axis_aligned_box(verts, dim)
        end
        # (5) each hanging node sits at the mean of its constrainers (geometric
        #     consistency of the affine constraint — independent of current code)
        for (h, mas) in grid.conformity_info
            mean = ntuple(d -> sum(nc[m][d] for m in mas) / length(mas), dim)
            @test all(d -> abs(nc[h][d] - mean[d]) < 1.0e-8, 1:dim)
        end
    end
end

# ------------------------------------------------------------------- cases

rotcell(c::Quadrilateral) = Quadrilateral((c.nodes[2], c.nodes[3], c.nodes[4], c.nodes[1]))
rotcell(c::Hexahedron) = Hexahedron((c.nodes[2], c.nodes[3], c.nodes[4], c.nodes[1], c.nodes[6], c.nodes[7], c.nodes[8], c.nodes[5]))

# Build every case once; returns (name, forest, grid, box-shaped?).
function build_amr_cases()
    cases = Tuple{String, Any, Any, Bool}[]
    add!(name, f, box) = push!(cases, (name, f, _AMR.creategrid(f), box))

    # ---- 2D ----
    let f = ForestBWG(generate_grid(Quadrilateral, (1, 1)), 3)
        _AMR.refine_all!(f, 1)
        _AMR.refine!(f.cells[1], f.cells[1].leaves[1])
        _AMR.refine!(f.cells[1], f.cells[1].leaves[1])
        add!("2d_single_tree", f, true)
    end
    let f = ForestBWG(generate_grid(Quadrilateral, (2, 2)), 3)
        _AMR.refine_all!(f, 1)
        _AMR.refine!(f.cells[1], f.cells[1].leaves[1])
        _AMR.refine!(f.cells[1], f.cells[1].leaves[1])
        add!("2d_four_trees", f, true)
    end
    let f = ForestBWG(generate_grid(Quadrilateral, (3, 3)), 3)
        for _ in 1:3; _AMR.refine!(f.cells[1], f.cells[1].leaves[1]); end
        _AMR.refine!(f.cells[3], f.cells[3].leaves[1])
        _AMR.refine!(f.cells[3], f.cells[3].leaves[2])
        _AMR.refine!(f.cells[3], f.cells[3].leaves[3])
        _AMR.refine!(f.cells[7], f.cells[7].leaves[1])
        _AMR.refine!(f.cells[7], f.cells[7].leaves[3])
        _AMR.refine!(f.cells[7], f.cells[7].leaves[5])
        _AMR.refine!(f.cells[9], f.cells[9].leaves[end])
        _AMR.refine!(f.cells[9], f.cells[9].leaves[end])
        _AMR.refine!(f.cells[9], f.cells[9].leaves[end])
        add!("2d_3x3_irregular", f, true)
    end
    let grid = generate_grid(Quadrilateral, (2, 2))
        grid.cells[2] = rotcell(grid.cells[2])
        f = ForestBWG(grid, 3)
        _AMR.refine!(f.cells[2], f.cells[2].leaves[1])
        add!("2d_rotated", f, true)
    end
    let f = ForestBWG(generate_grid(Quadrilateral, (2, 2)), 4)
        _AMR.refine_all!(f, 1)
        _AMR.refine!(f.cells[1], f.cells[1].leaves[1])
        _AMR.refine!(f.cells[1], f.cells[1].leaves[1])
        _AMR.balanceforest!(f)
        add!("2d_balanced", f, true)
    end
    let f = ForestBWG(Ferrite.generate_simple_disc_grid(Quadrilateral, 10), 3)
        Ferrite.refine!(f.cells[1], f.cells[1].leaves[1])
        Ferrite.refine!(f.cells[1], f.cells[1].leaves[3])
        Ferrite.balanceforest!(f)
        add!("2d_disc", f, false)
    end

    # ---- 3D ----
    let f = ForestBWG(generate_grid(Hexahedron, (1, 1, 1)), 3)
        _AMR.refine_all!(f, 1)
        _AMR.refine!(f.cells[1], f.cells[1].leaves[1])
        _AMR.refine!(f.cells[1], f.cells[1].leaves[1])
        add!("3d_single_tree", f, true)
    end
    for (nx, ny, nz) in ((2, 1, 1), (1, 2, 1), (1, 1, 2))
        f = ForestBWG(generate_grid(Hexahedron, (nx, ny, nz)), 3)
        _AMR.refine_all!(f, 1)
        add!("3d_face_$(nx)x$(ny)x$(nz)", f, true)
    end
    let f = ForestBWG(generate_grid(Hexahedron, (2, 2, 2)), 3)
        _AMR.refine_all!(f, 1)
        add!("3d_2x2x2", f, true)
    end
    let grid = generate_grid(Hexahedron, (2, 2, 2))
        grid.cells[2] = rotcell(rotcell(grid.cells[2]))
        f = ForestBWG(grid, 3)
        _AMR.refine_all!(f, 1)
        add!("3d_2x2x2_rotated", f, true)
    end
    let f = ForestBWG(generate_grid(Hexahedron, (1, 1, 1)), 3)
        _AMR.refine_all!(f, 1)
        _AMR.refine!(f.cells[1], f.cells[1].leaves[1])
        add!("3d_hanging_intra", f, true)
    end
    let grid = generate_grid(Hexahedron, (2, 2, 2))
        grid.cells[1] = rotcell(rotcell(grid.cells[1]))
        f = ForestBWG(grid, 3)
        _AMR.refine!(f.cells[1], f.cells[1].leaves[1])
        add!("3d_hanging_inter_rotated", f, true)
    end
    let f = ForestBWG(generate_grid(Hexahedron, (2, 2, 2)), 4)
        _AMR.refine_all!(f, 1)
        _AMR.refine!(f.cells[1], f.cells[1].leaves[1])
        _AMR.refine!(f.cells[1], f.cells[1].leaves[1])
        _AMR.balanceforest!(f)
        add!("3d_balanced", f, true)
    end
    let f = ForestBWG(Ferrite.generate_simple_disc_grid(Hexahedron, 10), 3)
        Ferrite.refine!(f.cells[1], f.cells[1].leaves[1])
        Ferrite.refine!(f.cells[1], f.cells[1].leaves[3])
        Ferrite.balanceforest!(f)
        add!("3d_disc", f, false)
    end
    return cases
end

const AMR_CASES = build_amr_cases()

@testset "AMR golden (regression)" begin
    for (name, forest, grid, _) in AMR_CASES
        check_golden(name, forest, grid)
    end
end

@testset "AMR correctness invariants" begin
    for (name, _, grid, box) in AMR_CASES
        check_invariants(name, grid; box = box)
    end
end
