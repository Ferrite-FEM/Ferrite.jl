# The Ferrite side of the forest: `ForestBWG` (the `AbstractGrid` interface, accessors and
# constructor checks), the coarse connectivity it hands to PureP4est, and creategrid checks on
# forests that need physical coordinates. The forest algorithms themselves (inter-tree
# transforms, refinement/coarsening, 2:1 balancing) are tested in `lib/PureP4est/test`.
using Ferrite, Test
import PureP4est

include(joinpath(@__DIR__, "test_utils.jl"))

@testset "OctreeBWG Operations" begin
    # Locally refined 3D forests on a rotated root mesh, materialized: the forest-level
    # part (transforms, refinement, node and hanging-node counts) lives in PureP4est's suite.

    # Rotate three dimensional case
    # This is our root mesh top view
    # x-----------x-----------x
    # |6    3    5|8    4    7|
    # |           |           |
    # |     ^     |     ^     |
    # |2    |    1|1    |    2|
    # |  <--+     |     +-->  |
    # |           |           |
    # |7    4    8|5    3    6|
    # x-----------x-----------x
    # |8    4    7|8    4    7|
    # |           |           |
    # |     ^     |     ^     |
    # |1    |    2|1    |    2|
    # |     +-->  |     +-->  |
    # |           |           |
    # |5    3    6|5    3    6|
    # x-----------x-----------x
    grid = generate_grid(Hexahedron, (2, 2, 2))
    # Rotate face topologically as decscribed in the ascii picture above
    grid.cells[7] = Hexahedron((grid.cells[7].nodes[2], grid.cells[7].nodes[3], grid.cells[7].nodes[4], grid.cells[7].nodes[1], grid.cells[7].nodes[4 + 2], grid.cells[7].nodes[4 + 3], grid.cells[7].nodes[4 + 4], grid.cells[7].nodes[4 + 1]))
    grid.cells[7] = Hexahedron((grid.cells[7].nodes[2], grid.cells[7].nodes[3], grid.cells[7].nodes[4], grid.cells[7].nodes[1], grid.cells[7].nodes[4 + 2], grid.cells[7].nodes[4 + 3], grid.cells[7].nodes[4 + 4], grid.cells[7].nodes[4 + 1]))

    # Single
    adaptive_grid = ForestBWG(grid, 3)
    PureP4est.refine_all!(adaptive_grid, 1)
    PureP4est.refine_octant!(adaptive_grid.cells[1], adaptive_grid.cells[1].leaves[8])
    transferred_grid = Ferrite.creategrid(adaptive_grid)
    @test unique(transferred_grid.nodes) == transferred_grid.nodes
    # Unrefined grid has 5 ^ dim nodes and the refined element introduces 6 face center, 12 edge center and 1 volume center nodes
    @test length(transferred_grid.nodes) == 5^3 + (6 + 12 + 1)
    # 6 faces and 12 edges of the single refined element induces one hanging node each
    @test length(transferred_grid.conformity_info) == 6 + 12

    adaptive_grid = ForestBWG(grid, 3)
    PureP4est.refine_all!(adaptive_grid, 1)
    PureP4est.refine_octant!(adaptive_grid.cells[1], adaptive_grid.cells[1].leaves[1])
    transferred_grid = Ferrite.creategrid(adaptive_grid)
    @test unique(transferred_grid.nodes) == transferred_grid.nodes
    # Unrefined grid has 5 ^ dim nodes and the refined element introduces 6 face center, 12 edge center and 1 volume center nodes
    @test length(transferred_grid.nodes) == 5^3 + (6 + 12 + 1)
    # 6 faces and 12 edges of the single refined element induces one hanging node each - minus 3 faces and 3 edges on the outer boundary
    @test length(transferred_grid.conformity_info) == 6 + 12 - 2 * 3

    adaptive_grid = ForestBWG(grid, 3)
    PureP4est.refine_all!(adaptive_grid, 1)
    PureP4est.refine_octant!(adaptive_grid.cells[8], adaptive_grid.cells[8].leaves[8])
    transferred_grid = Ferrite.creategrid(adaptive_grid)
    @test unique(transferred_grid.nodes) == transferred_grid.nodes
    # Unrefined grid has 5 ^ dim nodes and the refined element introduces 6 face center, 12 edge center and 1 volume center nodes
    @test length(transferred_grid.nodes) == 5^3 + (6 + 12 + 1)
    # 6 faces and 12 edges of the single refined element induces one hanging node each - minus 3 faces and 3 edges on the outer boundary
    @test length(transferred_grid.conformity_info) == 6 + 12 - 2 * 3

    adaptive_grid = ForestBWG(grid, 3)
    PureP4est.refine_all!(adaptive_grid, 1)
    PureP4est.refine_octant!(adaptive_grid.cells[8], adaptive_grid.cells[8].leaves[1])
    transferred_grid = Ferrite.creategrid(adaptive_grid)
    # Unrefined grid has 5 ^ dim nodes and the refined element introduces 6 face center, 12 edge center and 1 volume center nodes
    @test length(transferred_grid.nodes) == 5^3 + (6 + 12 + 1)
    # 6 faces and 12 edges of the single refined element induces one hanging node each
    @test length(transferred_grid.conformity_info) == 6 + 12

    # Combined
    adaptive_grid = ForestBWG(grid, 3)
    PureP4est.refine_all!(adaptive_grid, 1)
    PureP4est.refine_octant!(adaptive_grid.cells[1], adaptive_grid.cells[1].leaves[8])
    PureP4est.refine_octant!(adaptive_grid.cells[1], adaptive_grid.cells[1].leaves[1])
    transferred_grid = Ferrite.creategrid(adaptive_grid)
    @test unique(transferred_grid.nodes) == transferred_grid.nodes
    @test length(transferred_grid.nodes) == 5^3 + 2 * (6 + 12 + 1)
    @test length(transferred_grid.conformity_info) == 2 * (6 + 12) - 2 * 3

    adaptive_grid = ForestBWG(grid, 3)
    PureP4est.refine_all!(adaptive_grid, 1)
    PureP4est.refine_octant!(adaptive_grid.cells[8], adaptive_grid.cells[8].leaves[8])
    PureP4est.refine_octant!(adaptive_grid.cells[8], adaptive_grid.cells[8].leaves[1])
    transferred_grid = Ferrite.creategrid(adaptive_grid)
    @test unique(transferred_grid.nodes) == transferred_grid.nodes
    @test length(transferred_grid.nodes) == 5^3 + 2 * (6 + 12 + 1)
    @test length(transferred_grid.conformity_info) == 2 * (6 + 12) - 2 * 3

    # Combined
    adaptive_grid = ForestBWG(grid, 3)
    PureP4est.refine_all!(adaptive_grid, 1)
    PureP4est.refine_octant!(adaptive_grid.cells[1], adaptive_grid.cells[1].leaves[8])
    PureP4est.refine_octant!(adaptive_grid.cells[1], adaptive_grid.cells[1].leaves[1])
    PureP4est.refine_octant!(adaptive_grid.cells[8], adaptive_grid.cells[8].leaves[8])
    PureP4est.refine_octant!(adaptive_grid.cells[8], adaptive_grid.cells[8].leaves[1])
    transferred_grid = Ferrite.creategrid(adaptive_grid)
    @test unique(transferred_grid.nodes) == transferred_grid.nodes
    @test length(transferred_grid.nodes) == 5^3 + 4 * (6 + 12 + 1)
    @test length(transferred_grid.conformity_info) == 4 * (6 + 12) - 2 * 3 - 2 * 3

    # Combined and not rotated
    adaptive_grid = ForestBWG(grid, 3)
    PureP4est.refine_all!(adaptive_grid, 1)
    PureP4est.refine_octant!(adaptive_grid.cells[1], adaptive_grid.cells[1].leaves[8])
    PureP4est.refine_octant!(adaptive_grid.cells[1], adaptive_grid.cells[1].leaves[1])
    PureP4est.refine_octant!(adaptive_grid.cells[6], adaptive_grid.cells[6].leaves[6])
    PureP4est.refine_octant!(adaptive_grid.cells[6], adaptive_grid.cells[6].leaves[3])
    transferred_grid = Ferrite.creategrid(adaptive_grid)
    @test unique(transferred_grid.nodes) == transferred_grid.nodes
    # +5^3 on the coarse grid
    # +4 refined elements a 6 face nodes, 12 edge nodes and 1 volume nodes
    # -1 shared node between tree 1 and 6
    @test length(transferred_grid.nodes) == 5^3 + 4 * (6 + 12 + 1) - 1
    # 30 constraints from tree 1 (2*18 - 6 boundary) + 30 from tree 6 (2*18 - 6 boundary)
    # - 1 shared on common edge
    @test length(transferred_grid.conformity_info) == 59

    # Combined and rotated
    adaptive_grid = ForestBWG(grid, 3)
    PureP4est.refine_all!(adaptive_grid, 1)
    PureP4est.refine_octant!(adaptive_grid.cells[1], adaptive_grid.cells[1].leaves[8])
    PureP4est.refine_octant!(adaptive_grid.cells[1], adaptive_grid.cells[1].leaves[1])
    PureP4est.refine_octant!(adaptive_grid.cells[7], adaptive_grid.cells[7].leaves[6])
    PureP4est.refine_octant!(adaptive_grid.cells[7], adaptive_grid.cells[7].leaves[3])
    transferred_grid = Ferrite.creategrid(adaptive_grid)
    @test unique(transferred_grid.nodes) == transferred_grid.nodes
    # +5^3 on the coarse grid
    # +4 refined elements a 6 face nodes, 12 edge nodes and 1 volume nodes
    # -1 shared node between tree 1 and 7
    @test length(transferred_grid.nodes) == 5^3 + 4 * (6 + 12 + 1) - 1
    # 30 constraints from tree 1 + 30 from rotated tree 7 - 1 shared on common edge
    @test length(transferred_grid.conformity_info) == 59
end

@testset "ForestBWG AbstractGrid Interfacing" begin
    maxlevel = 3
    grid = generate_grid(Quadrilateral, (2, 2))
    adaptive_grid = ForestBWG(grid, maxlevel)
    for l in 1:maxlevel
        PureP4est.refine_all!(adaptive_grid, l)
        @test getncells(adaptive_grid) == 2^(2 * l) * 4 == length(getcells(adaptive_grid))
    end
end

@testset "edge balance at a multi-tree edge with mixed orientations" begin
    # Five hexes sharing the central vertical edge (0,0,0)-(0,0,1) — the 5-quad fan extruded in
    # z — with trees 3 and 5 listed "upside down" (reversed node tuple = 180° rotation, still
    # positively oriented), so the trees traverse the shared macro edge in opposite directions.
    # `transform_edge` must then take the along-edge flip from the actual (pivot, neighbor) pair:
    # orienting against `edge_edge_neighbor[..][1]` pairs with an arbitrary incident tree and
    # mirrors the balance refinement to the far end of the edge.
    nq = 5
    base = [Vec((0.0, 0.0))]
    for i in 0:(nq - 1)
        θ1 = 2π * i / nq
        θm = 2π * (i + 0.5) / nq
        push!(base, Vec((cos(θ1), sin(θ1))))
        push!(base, Vec(1.3 .* (cos(θm), sin(θm))))
    end
    nodes3 = Node{3, Float64}[]
    for z in (0.0, 1.0), p in base
        push!(nodes3, Node(Vec((p[1], p[2], z))))
    end
    nb = length(base)
    cells3 = map(0:(nq - 1)) do i
        q = (1, 2 + 2i, 3 + 2i, i == nq - 1 ? 2 : 4 + 2i)
        t = (q..., (q .+ nb)...)
        return (i + 1) in (3, 5) ? Hexahedron(reverse(t)) : Hexahedron(t)
    end
    forest = ForestBWG(Grid(cells3, nodes3), 6)
    # refine all trees toward the bottom end of the central edge, balancing in between
    target = Vec((0.0, 0.0, 0.0))
    for _ in 1:3
        g = Ferrite.AMR.creategrid(forest)
        marked = [c for c in 1:getncells(g) if any(n -> norm(n - target) < 1.0e-12, getcoordinates(g, c))]
        PureP4est.refine!(forest, marked)
        PureP4est.balanceforest!(forest)
    end
    g = Ferrite.AMR.creategrid(forest)
    minsz_at_target = Inf
    for c in 1:getncells(g)
        coords = getcoordinates(g, c)
        sz = maximum(maximum(x -> x[d], coords) - minimum(x -> x[d], coords) for d in 1:3)
        ctr = sum(coords) / length(coords)
        # fine cells may only exist in the graded halo around the refined bottom vertex — a
        # mirrored edge balance plants them near the top end of the central edge instead
        sz < 0.3 && @test norm(ctr - target) <= 0.75
        any(n -> norm(n - target) < 1.0e-12, coords) && (minsz_at_target = min(minsz_at_target, sz))
    end
    @test minsz_at_target < 0.2 # the target refinement itself happened
end

@testset "ForestBWG accessors and error paths" begin
    grid = generate_grid(Quadrilateral, (2, 2))
    forest = ForestBWG(grid, 3)
    PureP4est.refine_all!(forest, 1)

    # getcells collects the leaves of all trees in cell id order (tree by tree, Morton
    # order within each tree); the scalar getcells(forest, cellid) deliberately throws
    # instead of hitting the generic fallback, which would return a whole tree
    @test_throws ArgumentError getcells(forest, 7)
    @test_throws ArgumentError getcells(forest, [1, 2])

    leaves = getcells(forest)
    @test length(leaves) == getncells(forest)
    @test leaves[7] == forest.cells[2].leaves[3]
    @test leaves[1] == forest.cells[1].leaves[1]

    # The maximum refinement level is bounded by p4est's P4EST_MAXLEVEL/P8EST_MAXLEVEL:
    # beyond it an octree coordinate no longer fits the per-axis bit budget of the UInt64
    # boundary-table keys, whose collisions would silently merge unrelated nodes across
    # tree boundaries (creategrid used to return a grid with too many nodes instead).
    @test_throws DomainError ForestBWG(generate_grid(Quadrilateral, (2, 1)), 31)
    @test_throws DomainError ForestBWG(generate_grid(Hexahedron, (2, 1, 1)), 20)
    @test_throws DomainError ForestBWG(generate_grid(Quadrilateral, (2, 1)), -1)
    # the bounds themselves are admissible and are the defaults
    @test ForestBWG(generate_grid(Quadrilateral, (2, 1)), 30).cells[1].b == 30
    @test ForestBWG(generate_grid(Hexahedron, (2, 1, 1)), 19).cells[1].b == 19
    @test ForestBWG(generate_grid(Quadrilateral, (2, 1))).cells[1].b == 30
    @test ForestBWG(generate_grid(Hexahedron, (2, 1, 1))).cells[1].b == 19

    # the forest keeps the base grid's sets and nodes
    @test Ferrite.getfacetsets(forest) == Ferrite.getfacetsets(grid)
    @test Ferrite.getcellsets(forest) == Ferrite.getcellsets(grid)
    @test getfacetset(forest, "left") == getfacetset(grid, "left")
    @test getnodes(forest) == getnodes(grid)
    @test getnnodes(forest) == getnnodes(grid)
    @test Ferrite.getspatialdim(forest) == 2
    addcellset!(grid, "A", [1, 3])
    addnodeset!(grid, "N", [2])
    addvertexset!(grid, "V", x -> x[1] ≈ -1)
    forest_sets = ForestBWG(grid, 3)
    @test getcellset(forest_sets, "A") == getcellset(grid, "A")
    @test getnodeset(forest_sets, "N") == getnodeset(grid, "N")
    @test getvertexset(forest_sets, "V") == getvertexset(grid, "V")

    # Ferrite's entity indices are accepted by the inter-tree transforms
    let f = ForestBWG(generate_grid(Hexahedron, (2, 2, 2)), 3), o = f.cells[1].leaves[1]
        @test PureP4est.transform_facet(f, FacetIndex(1, 2), o) == PureP4est.transform_facet(f, 1, 2, o)
        @test PureP4est.transform_facet_remote(f, FacetIndex(1, 2), o) == PureP4est.transform_facet_remote(f, 1, 2, o)
        @test PureP4est.transform_corner(f, VertexIndex(1, 8), o, false) == PureP4est.transform_corner(f, 1, 8, o, false)
        @test PureP4est.transform_corner_remote(f, VertexIndex(1, 8), o, false) == PureP4est.transform_corner_remote(f, 1, 8, o, false)
        @test PureP4est.transform_edge(f, EdgeIndex(1, 4), o, false) == PureP4est.transform_edge(f, 1, 4, o, false)
        @test PureP4est.transform_edge_remote(f, EdgeIndex(1, 4), o, false) == PureP4est.transform_edge_remote(f, 1, 4, o, false)
    end
end

# The coarse connectivity deduced by PureP4est from the cells' vertex ids must agree with
# Ferrite's ExclusiveTopology, entry by entry and in the same order (balancing and the
# cross-tree node merge iterate the neighbour lists in order).
@testset "Connectivity agrees with ExclusiveTopology" begin
    function connectivity_from_topology(grid)
        dim = Ferrite.getspatialdim(grid)
        top = ExclusiveTopology(grid)
        tuples(t) = (k, i) -> ((x[1], x[2]) for x in t[k, i])
        fn = Ferrite.get_facet_facet_neighborhood(top, grid)
        n = size(fn, 1)
        edges = dim == 3 ? PureP4est.NeighborTable(tuples(top.edge_edge_neighbor), n, 12) :
            PureP4est.NeighborTable(NTuple{2, Int}[], Matrix{UnitRange{Int}}(undef, n, 0))
        return (PureP4est.NeighborTable(tuples(fn), n, size(fn, 2)), edges, PureP4est.NeighborTable(tuples(top.vertex_vertex_neighbor), n, 2^dim))
    end
    # renumber the local vertices of every other cell, so neighbours meet at all orientations
    function rotate_every_other(grid)
        cells = map(enumerate(getcells(grid))) do (i, c)
            n = c.nodes
            iseven(i) || return c
            c isa Quadrilateral ? Quadrilateral((n[2], n[3], n[4], n[1])) : Hexahedron((n[2], n[3], n[4], n[1], n[6], n[7], n[8], n[5]))
        end
        return Grid(cells, getnodes(grid))
    end
    for grid in (
            generate_grid(Quadrilateral, (5, 4)), generate_grid(Hexahedron, (3, 2, 4)),
            rotate_every_other(generate_grid(Quadrilateral, (4, 4))), rotate_every_other(generate_grid(Hexahedron, (3, 3, 2))),
            generate_grid(Quadrilateral, (1, 1)), generate_grid(Hexahedron, (1, 1, 1)),
        )
        conn = PureP4est.connectivity(ForestBWG(grid))
        @test (conn.facet_neighbors, conn.edge_neighbors, conn.vertex_neighbors) == connectivity_from_topology(grid)
    end
end
