# Materializing a forest into a `NonConformingGrid`: node numbering, set transfer,
# hanging nodes, the facet skeleton and the point iterator driving them. Mirrors the
# `creategrid`/`facetskeleton` half of `src/Adaptivity/forest.jl`.
using Ferrite, Test

include(joinpath(@__DIR__, "test_utils.jl"))

@testset "Materializing Grid" begin
    #################################################
    ############ structured 2D examples #############
    #################################################

    # 2D case with a single tree
    grid = generate_grid(Quadrilateral, (1, 1))
    adaptive_grid = ForestBWG(grid, 3)
    Ferrite.AMR.refine_all!(adaptive_grid, 1)
    Ferrite.AMR.refine_octant!(adaptive_grid.cells[1], adaptive_grid.cells[1].leaves[1])
    Ferrite.AMR.refine_octant!(adaptive_grid.cells[1], adaptive_grid.cells[1].leaves[1])
    transferred_grid = Ferrite.AMR.creategrid(adaptive_grid)
    @test length(transferred_grid.cells) == 10
    @test length(transferred_grid.nodes) == 19
    @test unique(transferred_grid.nodes) == transferred_grid.nodes

    #2D case with four trees and a nonuniform refinement pattern
    grid = generate_grid(Quadrilateral, (2, 2))
    adaptive_grid = ForestBWG(grid, 3)
    Ferrite.AMR.refine_all!(adaptive_grid, 1)
    Ferrite.AMR.refine_octant!(adaptive_grid.cells[1], adaptive_grid.cells[1].leaves[1])
    Ferrite.AMR.refine_octant!(adaptive_grid.cells[1], adaptive_grid.cells[1].leaves[1])
    transferred_grid = Ferrite.AMR.creategrid(adaptive_grid)
    @test length(transferred_grid.cells) == 22
    @test length(transferred_grid.nodes) == 35
    @test unique(transferred_grid.nodes) == transferred_grid.nodes

    #more random refinement
    grid = generate_grid(Quadrilateral, (3, 3))
    adaptive_grid = ForestBWG(grid, 3)
    Ferrite.AMR.refine_octant!(adaptive_grid.cells[1], adaptive_grid.cells[1].leaves[1])
    Ferrite.AMR.refine_octant!(adaptive_grid.cells[1], adaptive_grid.cells[1].leaves[1])
    Ferrite.AMR.refine_octant!(adaptive_grid.cells[1], adaptive_grid.cells[1].leaves[1])
    Ferrite.AMR.refine_octant!(adaptive_grid.cells[3], adaptive_grid.cells[3].leaves[1])
    Ferrite.AMR.refine_octant!(adaptive_grid.cells[3], adaptive_grid.cells[3].leaves[2])
    Ferrite.AMR.refine_octant!(adaptive_grid.cells[3], adaptive_grid.cells[3].leaves[3])
    Ferrite.AMR.refine_octant!(adaptive_grid.cells[7], adaptive_grid.cells[7].leaves[1])
    Ferrite.AMR.refine_octant!(adaptive_grid.cells[7], adaptive_grid.cells[7].leaves[3])
    Ferrite.AMR.refine_octant!(adaptive_grid.cells[7], adaptive_grid.cells[7].leaves[5])
    Ferrite.AMR.refine_octant!(adaptive_grid.cells[9], adaptive_grid.cells[9].leaves[end])
    Ferrite.AMR.refine_octant!(adaptive_grid.cells[9], adaptive_grid.cells[9].leaves[end])
    Ferrite.AMR.refine_octant!(adaptive_grid.cells[9], adaptive_grid.cells[9].leaves[end])
    transferred_grid = Ferrite.AMR.creategrid(adaptive_grid)
    @test length(transferred_grid.cells) == 45
    @test length(transferred_grid.nodes) == 76
    @test unique(transferred_grid.nodes) == transferred_grid.nodes

    #################################################
    ############ structured 3D examples #############
    #################################################

    # 3D case with a single tree
    grid = generate_grid(Hexahedron, (1, 1, 1))
    adaptive_grid = ForestBWG(grid, 3)
    Ferrite.AMR.refine_all!(adaptive_grid, 1)
    Ferrite.AMR.refine_octant!(adaptive_grid.cells[1], adaptive_grid.cells[1].leaves[1])
    Ferrite.AMR.refine_octant!(adaptive_grid.cells[1], adaptive_grid.cells[1].leaves[1])
    transferred_grid = Ferrite.AMR.creategrid(adaptive_grid)
    @test length(transferred_grid.cells) == 8 + 7 + 7
    @test length(transferred_grid.nodes) == 65
    @test unique(transferred_grid.nodes) == transferred_grid.nodes

    # Test only Interoctree by face connection
    grid = generate_grid(Hexahedron, (2, 1, 1))
    adaptive_grid = ForestBWG(grid, 3)
    Ferrite.AMR.refine_all!(adaptive_grid, 1)
    transferred_grid = Ferrite.AMR.creategrid(adaptive_grid)
    @test length(transferred_grid.cells) == 16
    @test length(transferred_grid.nodes) == 45
    @test unique(transferred_grid.nodes) == transferred_grid.nodes
    #rotate the case around
    grid = generate_grid(Hexahedron, (1, 2, 1))
    adaptive_grid = ForestBWG(grid, 3)
    Ferrite.AMR.refine_all!(adaptive_grid, 1)
    transferred_grid = Ferrite.AMR.creategrid(adaptive_grid)
    @test length(transferred_grid.cells) == 16
    @test length(transferred_grid.nodes) == 45
    @test unique(transferred_grid.nodes) == transferred_grid.nodes
    grid = generate_grid(Hexahedron, (1, 1, 2))
    adaptive_grid = ForestBWG(grid, 3)
    Ferrite.AMR.refine_all!(adaptive_grid, 1)
    transferred_grid = Ferrite.AMR.creategrid(adaptive_grid)
    @test length(transferred_grid.cells) == 16
    @test length(transferred_grid.nodes) == 45
    @test unique(transferred_grid.nodes) == transferred_grid.nodes

    grid = generate_grid(Hexahedron, (2, 2, 2))
    adaptive_grid = ForestBWG(grid, 3)
    Ferrite.AMR.refine_all!(adaptive_grid, 1)
    transferred_grid = Ferrite.AMR.creategrid(adaptive_grid)
    @test length(transferred_grid.cells) == 8^2
    @test length(transferred_grid.nodes) == 125 # 5 per edge
    @test unique(transferred_grid.nodes) == transferred_grid.nodes

    # Rotate three dimensional case
    grid = generate_grid(Hexahedron, (2, 2, 2))
    # Rotate face topologically
    grid.cells[2] = Hexahedron((grid.cells[2].nodes[2], grid.cells[2].nodes[3], grid.cells[2].nodes[4], grid.cells[2].nodes[1], grid.cells[2].nodes[4 + 2], grid.cells[2].nodes[4 + 3], grid.cells[2].nodes[4 + 4], grid.cells[2].nodes[4 + 1]))
    grid.cells[2] = Hexahedron((grid.cells[2].nodes[2], grid.cells[2].nodes[3], grid.cells[2].nodes[4], grid.cells[2].nodes[1], grid.cells[2].nodes[4 + 2], grid.cells[2].nodes[4 + 3], grid.cells[2].nodes[4 + 4], grid.cells[2].nodes[4 + 1]))
    # This is our root mesh bottom view
    # x-----------x-----------x
    # |4    4    3|4    4    3|
    # |           |           |
    # |     ^     |     ^     |
    # |1    |    2|1    |    2|
    # |     +-->  |     +-->  |
    # |           |           |
    # |1    3    2|1    3    2|
    # x-----------x-----------x
    # |4    4    3|3    2    2|
    # |           |           |
    # |     ^     |     ^     |
    # |1    |    2|4    |    3|
    # |     +-->  |  <--+     |
    # |           |           |
    # |1    3    2|4    1    1|
    # x-----------x-----------x
    adaptive_grid = ForestBWG(grid, 3)
    Ferrite.AMR.refine_all!(adaptive_grid, 1)
    transferred_grid = Ferrite.AMR.creategrid(adaptive_grid)
    @test length(transferred_grid.cells) == 8^2
    @test length(transferred_grid.nodes) == 125 # 5 per edge
    @test unique(transferred_grid.nodes) == transferred_grid.nodes
    #TODO iterate over all rotated versions and check if det J > 0
end

@testset "Materializing Float32 grid $dim D" for (dim, CT) in ((2, Quadrilateral), (3, Hexahedron))
    # ForestBWG is generic in the coordinate type T: a Float32 grid must survive the whole
    # pipeline (forest -> refine -> balance -> creategrid -> conformity constraints) and come
    # out Float32-typed, with coordinates matching the Float64 reference up to eps(Float32).
    grid64 = generate_grid(CT, ntuple(_ -> 2, dim))
    nodes32 = [Node(Vec{dim, Float32}(n.x)) for n in Ferrite.getnodes(grid64)]
    grid32 = Grid(getcells(grid64), nodes32; facetsets = Ferrite.getfacetsets(grid64))

    forests = map((grid64, grid32)) do grid
        forest = ForestBWG(grid, 3)
        Ferrite.AMR.refine_octant!(forest.cells[1], forest.cells[1].leaves[1])
        Ferrite.AMR.balanceforest!(forest)
        forest
    end
    @test forests[2] isa ForestBWG{dim, <:Any, Float32}

    transferred64, transferred32 = Ferrite.AMR.creategrid.(forests)
    @test transferred32 isa Ferrite.AMR.NonConformingGrid{dim, <:Any, Float32}
    @test eltype(transferred32.nodes) == Node{dim, Float32}
    @test transferred32.conformity_info == transferred64.conformity_info
    @test all(
        maximum(abs.(n32.x .- n64.x)) <= 4 * eps(Float32)
            for (n32, n64) in zip(transferred32.nodes, transferred64.nodes)
    )

    # conformity constraints on top of the Float32 grid
    dh = DofHandler(transferred32)
    add!(dh, :u, Lagrange{Ferrite.RefHypercube{dim}, 1}())
    close!(dh)
    ch = ConstraintHandler(dh)
    add!(ch, ConformityConstraint(:u))
    close!(ch)
    @test length(ch.prescribed_dofs) == length(transferred32.conformity_info)
end

@testset "cellset transfer $dim D" for (dim, CT) in ((2, Quadrilateral), (3, Hexahedron))
    # Every leaf inherits the cellset membership of its macro (tree) cell, so the macro
    # cellsets must survive creategrid: same names, covering exactly the leaves of their trees.
    grid = generate_grid(CT, ntuple(_ -> 2, dim))
    # generate_grid spans [-1,1]^dim; addcellset! keeps cells where ALL nodes satisfy the
    # predicate, so the half-space tests must include the x = 0 interface plane.
    addcellset!(grid, "left", x -> x[1] <= 0)
    addcellset!(grid, "right", x -> x[1] >= 0)
    forest = ForestBWG(grid, 3)
    Ferrite.AMR.refine_octant!(forest.cells[1], forest.cells[1].leaves[1])
    Ferrite.AMR.refine_octant!(forest.cells[1], forest.cells[1].leaves[1])
    Ferrite.AMR.balanceforest!(forest)
    transferred_grid = Ferrite.AMR.creategrid(forest)

    left = getcellset(transferred_grid, "left")
    right = getcellset(transferred_grid, "right")
    # the two sets partition the refined grid
    @test !isempty(left) && !isempty(right)
    @test isempty(intersect(left, right))
    @test sort!(union(collect(left), collect(right))) == 1:getncells(transferred_grid)
    # membership is geometric: every leaf lies inside its macro cell's half
    for (set, sgn) in ((left, -1), (right, +1))
        for cellid in set
            centroid = sum(getcoordinates(transferred_grid, cellid)) / 2^dim
            @test sgn * centroid[1] > 0
        end
    end
    # an unrefined forest reproduces the macro cellsets verbatim
    unrefined = Ferrite.AMR.creategrid(ForestBWG(grid, 3))
    @test Ferrite.getcellsets(unrefined) == Ferrite.getcellsets(grid)
end

@testset "vertex/node sets are not transferred $dim D" for (dim, CT) in ((2, Quadrilateral), (3, Hexahedron))
    # creategrid only carries cellsets and facetsets onto the refined grid. The forest keeps
    # the macro vertex/node sets, but the materialized grid has both empty -- documented on
    # `creategrid`/`NonConformingGrid`, pinned here so the behaviour cannot drift silently.
    grid = generate_grid(CT, ntuple(_ -> 2, dim))
    addvertexset!(grid, "vs", x -> x[1] ≈ -1.0)
    grid.nodesets["ns"] = Ferrite.OrderedSet([1, 2])
    forest = ForestBWG(grid, 3)
    @test Ferrite.getvertexsets(forest) == Ferrite.getvertexsets(grid)
    @test Ferrite.getnodesets(forest) == Ferrite.getnodesets(grid)

    Ferrite.AMR.refine!(forest, [1])
    Ferrite.AMR.balanceforest!(forest)
    transferred_grid = Ferrite.AMR.creategrid(forest)
    @test isempty(Ferrite.getvertexsets(transferred_grid))
    @test isempty(Ferrite.getnodesets(transferred_grid))
    # the sets that *are* transferred still are
    @test keys(Ferrite.getfacetsets(transferred_grid)) == keys(Ferrite.getfacetsets(grid))
end

@testset "hanging nodes" begin
    #Easy Intraoctree
    grid = generate_grid(Hexahedron, (1, 1, 1))
    adaptive_grid = ForestBWG(grid, 3)
    Ferrite.AMR.refine_all!(adaptive_grid, 1)
    Ferrite.AMR.refine_octant!(adaptive_grid.cells[1], adaptive_grid.cells[1].leaves[1])
    # x-----------x-----------x
    # |           |           |
    # |           |           |
    # |           |           |
    # |           |           |
    # |           |           |
    # |           |           |
    # |           |           |
    # x-----x-----x-----------x
    # |     |     |           |
    # |     |     |           |
    # |     |     |           |
    # x-----x-----x           |
    # |     |     |           |
    # |     |     |           |
    # |     |     |           |
    # x-----x-----x-----------x
    transferred_grid = Ferrite.AMR.creategrid(adaptive_grid)
    @test length(transferred_grid.conformity_info) == 12

    # Easy Interoctree
    grid = generate_grid(Hexahedron, (2, 2, 2))
    adaptive_grid = ForestBWG(grid, 3)
    Ferrite.AMR.refine_octant!(adaptive_grid.cells[1], adaptive_grid.cells[1].leaves[1])
    # x-----------x-----------x
    # |           |           |
    # |           |           |
    # |           |           |
    # |           |           |
    # |           |           |
    # |           |           |
    # |           |           |
    # x-----x-----x-----------x
    # |     |     |           |
    # |     |     |           |
    # |     |     |           |
    # x-----x-----x           |
    # |     |     |           |
    # |     |     |           |
    # |     |     |           |
    # x-----x-----x-----------x
    transferred_grid = Ferrite.AMR.creategrid(adaptive_grid)
    @test length(transferred_grid.conformity_info) == 12

    #rotate the case from above in the first cell around
    grid = generate_grid(Hexahedron, (2, 2, 2))
    # Rotate face topologically
    grid.cells[1] = Hexahedron((grid.cells[1].nodes[2], grid.cells[1].nodes[3], grid.cells[1].nodes[4], grid.cells[1].nodes[1], grid.cells[1].nodes[4 + 2], grid.cells[1].nodes[4 + 3], grid.cells[1].nodes[4 + 4], grid.cells[1].nodes[4 + 1]))
    grid.cells[1] = Hexahedron((grid.cells[1].nodes[2], grid.cells[1].nodes[3], grid.cells[1].nodes[4], grid.cells[1].nodes[1], grid.cells[1].nodes[4 + 2], grid.cells[1].nodes[4 + 3], grid.cells[1].nodes[4 + 4], grid.cells[1].nodes[4 + 1]))
    adaptive_grid = ForestBWG(grid, 3)
    Ferrite.AMR.refine_octant!(adaptive_grid.cells[1], adaptive_grid.cells[1].leaves[1])
    transferred_grid_rotated = Ferrite.AMR.creategrid(adaptive_grid)
    @test Set(transferred_grid_rotated.conformity_info[2]) == Set([1, 9])
    @test Set(transferred_grid_rotated.conformity_info[3]) == Set([1, 13])
    @test Set(transferred_grid_rotated.conformity_info[5]) == Set([1, 19])
    @test Set(transferred_grid_rotated.conformity_info[6]) == Set([1, 9, 19, 23])
    @test Set(transferred_grid_rotated.conformity_info[7]) == Set([1, 13, 19, 25])
    @test Set(transferred_grid_rotated.conformity_info[11]) == Set([9, 23])
    @test Set(transferred_grid_rotated.conformity_info[15]) == Set([13, 25])
    @test Set(transferred_grid_rotated.conformity_info[20]) == Set([19, 23])
    @test Set(transferred_grid_rotated.conformity_info[21]) == Set([19, 25])
    @test Set(transferred_grid_rotated.conformity_info[22]) == Set([19, 23, 25, 27])
    @test Set(transferred_grid_rotated.conformity_info[24]) == Set([23, 27])
    @test Set(transferred_grid_rotated.conformity_info[26]) == Set([25, 27])
    @test length(transferred_grid_rotated.conformity_info) == 12

    #2D rotated case
    grid = generate_grid(Quadrilateral, (2, 2))
    # Rotate face topologically
    grid.cells[2] = Quadrilateral((grid.cells[2].nodes[2], grid.cells[2].nodes[3], grid.cells[2].nodes[4], grid.cells[2].nodes[1]))
    # This is our root mesh
    # x-----------x-----------x
    # |4    4    3|4    4    3|
    # |           |           |
    # |     ^     |     ^     |
    # |1    |    2|1    |    2|
    # |     +-->  |     +-->  |
    # |           |           |
    # |1    3    2|1    3    2|
    # x-----------x-----------x
    # |4    4    3|3    2    2|
    # |           |           |
    # |     ^     |     ^     |
    # |1    |    2|4    |    3|
    # |     +-->  |  <--+     |
    # |           |           |
    # |1    3    2|4    1    1|
    # x-----------x-----------x
    adaptive_grid = ForestBWG(grid, 3)
    Ferrite.AMR.refine_octant!(adaptive_grid.cells[2], adaptive_grid.cells[2].leaves[1])
    transferred_grid_rotated = Ferrite.AMR.creategrid(adaptive_grid)
    @test Set(transferred_grid_rotated.conformity_info[10]) == Set([4, 9])
    @test Set(transferred_grid_rotated.conformity_info[11]) == Set([2, 4])
    @test length(transferred_grid_rotated.conformity_info) == 2

    # multiple corner connections in 2D by disc discretization
    grid = generate_simple_disc_grid(Quadrilateral, 10)
    adaptive_grid = ForestBWG(grid, 3)
    @test getncells(adaptive_grid) == 10
    Ferrite.AMR.refine_octant!(adaptive_grid.cells[1], adaptive_grid.cells[1].leaves[1])
    Ferrite.AMR.refine_octant!(adaptive_grid.cells[1], adaptive_grid.cells[1].leaves[3])
    @test getncells(adaptive_grid) == 16
    Ferrite.balanceforest!(adaptive_grid)
    @test getncells(adaptive_grid) == 9 * 4 + 3 + 4

    # multiple corner connections in 3D by cylinder discretization
    grid = generate_simple_disc_grid(Hexahedron, 10)
    adaptive_grid = ForestBWG(grid, 3)
    @test getncells(adaptive_grid) == 10
    Ferrite.AMR.refine_octant!(adaptive_grid.cells[1], adaptive_grid.cells[1].leaves[1])
    @test getncells(adaptive_grid) == 17
    Ferrite.AMR.refine_octant!(adaptive_grid.cells[1], adaptive_grid.cells[1].leaves[3])
    @test getncells(adaptive_grid) == 24
    Ferrite.balanceforest!(adaptive_grid)
    @test getncells(adaptive_grid) == 9 * 8 + 7 + 8
end

@testset "facet skeleton" begin
    ≈ₐ(a, b) = isapprox(a, b; atol = 1.0e-12)

    # Geometric checks for axis-aligned [-1,1]^dim grids: every pair shares a plane, the
    # first (fine) facet lies inside the second's bounding box, and the facets not covered
    # by the skeleton are exactly the domain-boundary facets.
    function check_skeleton_geometry(grid, skel, dim)
        X = [Ferrite.get_node_coordinate(n) for n in Ferrite.getnodes(grid)]
        fcoords(fi) = [X[n] for n in Ferrite.facets(Ferrite.getcells(grid, fi[1]))[fi[2]]]
        covered = Set{FacetIndex}()
        for (fa, fb) in skel
            ca = fcoords(fa); cb = fcoords(fb)
            # both facets degenerate at the same coordinate along some axis (shared plane)
            @test any(a -> all(x -> x[a] ≈ₐ cb[1][a], ca) && all(x -> x[a] ≈ₐ cb[1][a], cb), 1:dim)
            # fine ⊆ coarse (equality for conforming pairs) — also checks fine-side-first order
            for a in 1:dim
                @test minimum(x -> x[a], ca) >= minimum(x -> x[a], cb) - 1.0e-12
                @test maximum(x -> x[a], ca) <= maximum(x -> x[a], cb) + 1.0e-12
            end
            push!(covered, fa); push!(covered, fb)
        end
        for c in 1:getncells(grid), f in 1:Ferrite.nfacets(Ferrite.getcells(grid, c))
            fi = FacetIndex(c, f)
            onboundary = any(a -> all(x -> x[a] ≈ₐ 1.0, fcoords(fi)) || all(x -> x[a] ≈ₐ -1.0, fcoords(fi)), 1:dim)
            @test onboundary == !(fi in covered)
        end
        return
    end

    # Independent 2D ground truth from the materialized grid (a 2D facet is a node pair):
    # two owners -> conforming pair; single owner whose facet spans a hanging node and its
    # master -> fine side of a hanging interface, coarse owner via the master pair.
    function skeleton_groundtruth_2d(grid)
        cells = Ferrite.getcells(grid)
        owner = Dict{Tuple{Int, Int}, Vector{FacetIndex}}()
        for c in 1:getncells(grid), (f, fnodes) in enumerate(Ferrite.facets(cells[c]))
            push!(get!(() -> FacetIndex[], owner, minmax(fnodes...)), FacetIndex(c, f))
        end
        hang = grid.conformity_info
        pairs = Set{NTuple{2, Tuple{Int, Int}}}()
        for (key, owners) in owner
            a, b = key
            if length(owners) == 2
                p1 = (owners[1][1], owners[1][2]); p2 = (owners[2][1], owners[2][2])
                push!(pairs, p1 < p2 ? (p1, p2) : (p2, p1))     # conforming: unordered
            else
                for (hn, oth) in ((a, b), (b, a))
                    if haskey(hang, hn) && length(hang[hn]) == 2 && oth in hang[hn]
                        ck = minmax(hang[hn][1], hang[hn][2])
                        haskey(owner, ck) || continue
                        cfi = owner[ck][1]
                        push!(pairs, ((owners[1][1], owners[1][2]), (cfi[1], cfi[2])))
                        break
                    end
                end
            end
        end
        return pairs
    end

    function skeleton_canonical_2d(grid, skel)
        cells = Ferrite.getcells(grid)
        fkey(fi) = minmax(Ferrite.facets(cells[fi[1]])[fi[2]]...)
        pairs = Set{NTuple{2, Tuple{Int, Int}}}()
        for (fa, fb) in skel
            p1 = (fa[1], fa[2]); p2 = (fb[1], fb[2])
            if fkey(fa) == fkey(fb)                              # conforming: unordered
                push!(pairs, p1 < p2 ? (p1, p2) : (p2, p1))
            else                                                 # hanging: fine first
                push!(pairs, (p1, p2))
            end
        end
        @test length(pairs) == length(skel)                      # no duplicates
        return pairs
    end

    # 2x1: refine cell 1 once -> 4 intra-tree conforming + 2 inter-tree hanging pairs
    grid = generate_grid(Quadrilateral, (2, 1))
    forest = ForestBWG(grid, 3)
    Ferrite.refine!(forest, [1])
    Ferrite.balanceforest!(forest)
    tg = Ferrite.AMR.creategrid(forest)
    skel = Ferrite.facetskeleton(forest)
    @test length(skel) == 6
    @test skeleton_canonical_2d(tg, skel) == skeleton_groundtruth_2d(tg)
    check_skeleton_geometry(tg, skel, 2)

    # 2x1 both refined -> 4 + 4 intra-tree + 2 inter-tree conforming pairs
    forest = ForestBWG(generate_grid(Quadrilateral, (2, 1)), 3)
    Ferrite.refine_all!(forest, 1)
    tg = Ferrite.AMR.creategrid(forest)
    skel = Ferrite.facetskeleton(forest)
    @test length(skel) == 10
    @test skeleton_canonical_2d(tg, skel) == skeleton_groundtruth_2d(tg)
    check_skeleton_geometry(tg, skel, 2)

    # 2D multi-level: intra- and inter-tree hanging + conforming interfaces mixed
    forest = ForestBWG(generate_grid(Quadrilateral, (2, 2)), 5)
    Ferrite.refine!(forest, [1])
    Ferrite.balanceforest!(forest)
    Ferrite.refine!(forest, [1, 2, 7])
    Ferrite.balanceforest!(forest)
    Ferrite.refine!(forest, [3])
    Ferrite.balanceforest!(forest)
    tg = Ferrite.AMR.creategrid(forest)
    skel = Ferrite.facetskeleton(forest)
    @test skeleton_canonical_2d(tg, skel) == skeleton_groundtruth_2d(tg)
    check_skeleton_geometry(tg, skel, 2)

    # 2D rotated macro element (cf. "hanging nodes" testset): topology-only rotation, so
    # the node-pair ground truth and the geometric checks both still apply
    grid = generate_grid(Quadrilateral, (2, 2))
    grid.cells[2] = Quadrilateral((grid.cells[2].nodes[2], grid.cells[2].nodes[3], grid.cells[2].nodes[4], grid.cells[2].nodes[1]))
    forest = ForestBWG(grid, 3)
    Ferrite.AMR.refine_octant!(forest.cells[2], forest.cells[2].leaves[1])
    tg = Ferrite.AMR.creategrid(forest)
    skel = Ferrite.facetskeleton(forest)
    @test skeleton_canonical_2d(tg, skel) == skeleton_groundtruth_2d(tg)
    check_skeleton_geometry(tg, skel, 2)

    # 3D intra-octree hanging (cf. "hanging nodes" testset)
    forest = ForestBWG(generate_grid(Hexahedron, (1, 1, 1)), 3)
    Ferrite.AMR.refine_all!(forest, 1)
    Ferrite.AMR.refine_octant!(forest.cells[1], forest.cells[1].leaves[1])
    tg = Ferrite.AMR.creategrid(forest)
    skel = Ferrite.facetskeleton(forest)
    # 8 octants -> 12 coarse interfaces; refining one octant replaces 3 of them by 4
    # hanging subfacet pairs each and adds 12 interfaces between its children
    @test length(skel) == 12 - 3 + 3 * 4 + 12
    check_skeleton_geometry(tg, skel, 3)

    # 3D inter-octree, multi-level
    forest = ForestBWG(generate_grid(Hexahedron, (2, 2, 2)), 4)
    Ferrite.refine!(forest, [1])
    Ferrite.balanceforest!(forest)
    Ferrite.refine!(forest, [1])
    Ferrite.balanceforest!(forest)
    tg = Ferrite.AMR.creategrid(forest)
    skel = Ferrite.facetskeleton(forest)
    check_skeleton_geometry(tg, skel, 3)
end

@testset "unbalanced forest errors" begin
    forest = ForestBWG(generate_grid(Quadrilateral, (1, 1)), 4)
    Ferrite.AMR.refine!(forest, 1) # 4 level-1 leaves
    Ferrite.AMR.refine!(forest, 2) # second quadrant -> level 2
    Ferrite.AMR.refine!(forest, 2) # its first child -> level 3, faces the level-1 first quadrant: 2:1 violated
    @test_throws ArgumentError Ferrite.AMR.creategrid(forest)
    @test_throws ArgumentError Ferrite.facetskeleton(forest)
end

@testset "point iterator LeafSupport iteration" begin
    forest = ForestBWG(generate_grid(Quadrilateral, (1, 1)), 3)
    Ferrite.AMR.refine!(forest, 1)
    tree = forest.cells[1]
    sc = Ferrite.AMR.IterScratch(tree)
    octs = Ferrite.AMR.OctantBWG{2, 4, Int64}[]
    Ferrite.AMR.iterate_points(tree, sc; mindim = 0, maxdim = 1) do c, ls
        @test collect(ls) == [ls[i] for i in 1:length(ls)] # Base.iterate/eltype agree with getindex
        append!(octs, ls)
        return
    end
    @test !isempty(octs)
    @test all(o -> o ∈ tree.leaves, octs)
end

@testset "3D creategrid across rotated tree faces" begin
    # 90° rotations of the second hexahedron about the z- and x-axis: the shared macro
    # face pairs up with faces of different local axes/orientations, exercising the
    # inter-tree coordinate transforms for all axis permutations.
    ρz = (2, 3, 4, 1, 6, 7, 8, 5)
    ρx = (5, 6, 2, 1, 8, 7, 3, 4)
    for ρ in (ρz, ρx), nrot in 1:3, refine_tree in (1, 2)
        grid = generate_grid(Hexahedron, (2, 1, 1))
        c = grid.cells[2].nodes
        for _ in 1:nrot
            c = ntuple(i -> c[ρ[i]], 8)
        end
        grid.cells[2] = Hexahedron(c)
        forest = ForestBWG(grid, 3)
        Ferrite.AMR.refine_octant!(forest.cells[refine_tree], forest.cells[refine_tree].leaves[1])
        Ferrite.AMR.balanceforest!(forest)
        g = Ferrite.AMR.creategrid(forest)
        @test getncells(g) == getncells(forest)
        # cross-tree node identification must not leave duplicate physical nodes
        coords = [round.(n.x; digits = 8) for n in g.nodes]
        @test length(unique(coords)) == length(coords)
        # every cell references valid, distinct nodes
        @test all(cell -> all(n -> 1 <= n <= length(g.nodes), cell.nodes) && allunique(cell.nodes), g.cells)
        # hanging constraints reference valid nodes
        @test all(p -> 1 <= p.first <= length(g.nodes) && all(m -> 1 <= m <= length(g.nodes), p.second), g.conformity_info)
    end
end

@testset "creategrid on a deep uniform tree" begin
    # 64 leaves under the root: exercises the binary-search branch of split_bounds
    forest = ForestBWG(generate_grid(Quadrilateral, (1, 1)), 4)
    for l in 1:3
        Ferrite.AMR.refine_all!(forest, l)
    end
    g = Ferrite.AMR.creategrid(forest)
    @test getncells(g) == 64
    @test length(g.nodes) == 81 # (2^3 + 1)^2
    @test isempty(g.conformity_info)
end
