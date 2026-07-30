# The forest data structure: inter-tree transforms, the Ferrite `AbstractGrid`
# interface, refinement/coarsening and 2:1 balancing. Mirrors the forest half of
# `src/Adaptivity/forest.jl`.
using Ferrite, Test

include(joinpath(@__DIR__, "test_utils.jl"))

@testset "OctreeBWG Operations" begin
    # maximum level == 3
    # Octant level 0 size == 2^3=8
    # Octant level 1 size == 2^3/2 = 4
    # Octant level 2 size == 2^3/2 = 2
    # Octant level 3 size == 2^3/2 = 1
    # test translation constructor
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
    for cell in adaptive_grid.cells
        @test cell isa Ferrite.AMR.OctreeBWG
        @test cell.leaves[1] == Ferrite.AMR.OctantBWG(2, 0, 1, cell.b)
    end
    @test Ferrite.AMR.transform_facet_remote(adaptive_grid, FacetIndex(2, 4), adaptive_grid.cells[1].leaves[1]) == Ferrite.AMR.OctantBWG(0, (8, 0))
    @test Ferrite.AMR.transform_facet_remote(adaptive_grid, FacetIndex(1, 2), adaptive_grid.cells[1].leaves[1]) == Ferrite.AMR.OctantBWG(0, (0, 8))
    @test Ferrite.AMR.transform_facet_remote(adaptive_grid, FacetIndex(4, 1), adaptive_grid.cells[3].leaves[1]) == Ferrite.AMR.OctantBWG(0, (8, 0))
    @test Ferrite.AMR.transform_facet_remote(adaptive_grid, FacetIndex(3, 2), adaptive_grid.cells[4].leaves[1]) == Ferrite.AMR.OctantBWG(0, (-8, 0))
    @test Ferrite.AMR.transform_facet_remote(adaptive_grid, FacetIndex(3, 3), adaptive_grid.cells[1].leaves[1]) == Ferrite.AMR.OctantBWG(0, (0, 8))
    @test Ferrite.AMR.transform_facet_remote(adaptive_grid, FacetIndex(1, 4), adaptive_grid.cells[3].leaves[1]) == Ferrite.AMR.OctantBWG(0, (0, -8))
    @test Ferrite.AMR.transform_facet_remote(adaptive_grid, FacetIndex(4, 3), adaptive_grid.cells[2].leaves[1]) == Ferrite.AMR.OctantBWG(0, (8, 0))
    @test Ferrite.AMR.transform_facet_remote(adaptive_grid, FacetIndex(2, 2), adaptive_grid.cells[4].leaves[1]) == Ferrite.AMR.OctantBWG(0, (0, -8))
    o = adaptive_grid.cells[1].leaves[1]
    @test Ferrite.AMR.transform_facet(adaptive_grid, FacetIndex(1, 2), o) == Ferrite.AMR.OctantBWG(0, (8, 0))
    @test Ferrite.AMR.transform_facet(adaptive_grid, FacetIndex(1, 4), o) == Ferrite.AMR.OctantBWG(0, (0, 8))
    @test Ferrite.AMR.transform_facet(adaptive_grid, FacetIndex(2, 4), o) == Ferrite.AMR.OctantBWG(0, (0, 8))
    @test Ferrite.AMR.transform_facet(adaptive_grid, FacetIndex(2, 2), o) == Ferrite.AMR.OctantBWG(0, (8, 0))
    @test Ferrite.AMR.transform_facet(adaptive_grid, FacetIndex(3, 2), o) == Ferrite.AMR.OctantBWG(0, (8, 0))
    @test Ferrite.AMR.transform_facet(adaptive_grid, FacetIndex(3, 3), o) == Ferrite.AMR.OctantBWG(0, (0, -8))
    @test Ferrite.AMR.transform_facet(adaptive_grid, FacetIndex(4, 1), o) == Ferrite.AMR.OctantBWG(0, (-8, 0))
    @test Ferrite.AMR.transform_facet(adaptive_grid, FacetIndex(4, 3), o) == Ferrite.AMR.OctantBWG(0, (0, -8))

    grid_new = Ferrite.AMR.creategrid(adaptive_grid)
    @test length(grid_new.nodes) == 9
    @test length(grid_new.conformity_info) == 0

    grid.cells[4] = Quadrilateral((grid.cells[4].nodes[2], grid.cells[4].nodes[3], grid.cells[4].nodes[4], grid.cells[4].nodes[1]))
    grid.cells[4] = Quadrilateral((grid.cells[4].nodes[2], grid.cells[4].nodes[3], grid.cells[4].nodes[4], grid.cells[4].nodes[1]))
    # root mesh in Ferrite.AMR notation                        in p4est notation
    # x-----------x-----------x                         x-----------x-----------x
    # |4    3    3|2    1    1|                         |3    4    4|2    3    1|
    # |           |           |                         |           |           |
    # |     ^     |  <--+     |                         |     ^     |  <--+     |
    # |4    |    2|2    |    4|                         |1    |    2|2    |    1|
    # |     +-->  |     v     |                         |     +-->  |     v     |
    # |           |           |                         |           |           |
    # |1    1    2|3    3    4|                         |1    3    2|4    4    3|
    # x-----------x-----------x                         x-----------x-----------x
    # |4    3    3|3    2    2|                         |3    4    4|4    4    1|
    # |           |           |                         |           |           |
    # |     ^     |     ^     |                         |     ^     |     ^     |
    # |4    |    2|3    |    1|                         |1    |    2|2    |    1|
    # |     +-->  |  <--+     |                         |     +-->  |  <--+     |
    # |           |           |                         |           |           |
    # |1    1    2|4    4    1|                         |1    3    2|3    3    1|
    # x-----------x-----------x                         x-----------x-----------x
    adaptive_grid = ForestBWG(grid, 3)
    for cell in adaptive_grid.cells
        @test cell isa Ferrite.AMR.OctreeBWG
        @test cell.leaves[1] == Ferrite.AMR.OctantBWG(2, 0, 1, cell.b)
    end
    @test Ferrite.AMR.transform_facet_remote(adaptive_grid, FacetIndex(2, 4), adaptive_grid.cells[1].leaves[1]) == Ferrite.AMR.OctantBWG(0, (8, 0))
    @test Ferrite.AMR.transform_facet_remote(adaptive_grid, FacetIndex(1, 2), adaptive_grid.cells[1].leaves[1]) == Ferrite.AMR.OctantBWG(0, (0, 8))
    @test Ferrite.AMR.transform_facet_remote(adaptive_grid, FacetIndex(4, 2), adaptive_grid.cells[3].leaves[1]) == Ferrite.AMR.OctantBWG(0, (8, 0))
    @test Ferrite.AMR.transform_facet_remote(adaptive_grid, FacetIndex(3, 2), adaptive_grid.cells[4].leaves[1]) == Ferrite.AMR.OctantBWG(0, (8, 0))
    @test Ferrite.AMR.transform_facet_remote(adaptive_grid, FacetIndex(3, 3), adaptive_grid.cells[1].leaves[1]) == Ferrite.AMR.OctantBWG(0, (0, 8))
    @test Ferrite.AMR.transform_facet_remote(adaptive_grid, FacetIndex(1, 4), adaptive_grid.cells[3].leaves[1]) == Ferrite.AMR.OctantBWG(0, (0, -8))
    @test Ferrite.AMR.transform_facet_remote(adaptive_grid, FacetIndex(4, 4), adaptive_grid.cells[2].leaves[1]) == Ferrite.AMR.OctantBWG(0, (8, 0))
    @test Ferrite.AMR.transform_facet_remote(adaptive_grid, FacetIndex(2, 2), adaptive_grid.cells[4].leaves[1]) == Ferrite.AMR.OctantBWG(0, (0, 8))

    #@test Ferrite.AMR.transform_corner(adaptive_grid, VertexIndex(4,4), adaptive_grid.cells[1].leaves[1],false) == Ferrite.AMR.transform_corner_remote(adaptive_grid, VertexIndex(1,4), adaptive_grid.cells[1].leaves[1], false) == Ferrite.AMR.OctantBWG(0,(8,8))
    #@test Ferrite.AMR.transform_corner_remote(adaptive_grid, VertexIndex(3,2), adaptive_grid.cells[1].leaves[1], false) == Ferrite.AMR.transform_corner_remote(adaptive_grid, VertexIndex(2,4), adaptive_grid.cells[1].leaves[1], false) == Ferrite.AMR.OctantBWG(0,(8,-8))
    #@test Ferrite.AMR.transform_corner(adaptive_grid, VertexIndex(4,4), adaptive_grid.cells[1].leaves[1],false) == Ferrite.AMR.transform_corner(adaptive_grid, VertexIndex(1,4), adaptive_grid.cells[1].leaves[1],false) == Ferrite.AMR.OctantBWG(0,(8,8))
    #@test Ferrite.AMR.transform_corner(adaptive_grid, VertexIndex(3,2), adaptive_grid.cells[1].leaves[1], false) == Ferrite.AMR.transform_corner(adaptive_grid, VertexIndex(2,4), adaptive_grid.cells[1].leaves[1], false) == Ferrite.AMR.OctantBWG(0,(8,-8))

    o = adaptive_grid.cells[1].leaves[1]
    @test Ferrite.AMR.transform_facet(adaptive_grid, FacetIndex(1, 2), o) == Ferrite.AMR.OctantBWG(0, (8, 0))
    @test Ferrite.AMR.transform_facet(adaptive_grid, FacetIndex(1, 4), o) == Ferrite.AMR.OctantBWG(0, (0, 8))
    @test Ferrite.AMR.transform_facet(adaptive_grid, FacetIndex(2, 4), o) == Ferrite.AMR.OctantBWG(0, (0, 8))
    @test Ferrite.AMR.transform_facet(adaptive_grid, FacetIndex(2, 2), o) == Ferrite.AMR.OctantBWG(0, (8, 0))
    @test Ferrite.AMR.transform_facet(adaptive_grid, FacetIndex(3, 2), o) == Ferrite.AMR.OctantBWG(0, (8, 0))
    @test Ferrite.AMR.transform_facet(adaptive_grid, FacetIndex(3, 3), o) == Ferrite.AMR.OctantBWG(0, (0, -8))
    @test Ferrite.AMR.transform_facet(adaptive_grid, FacetIndex(4, 2), o) == Ferrite.AMR.OctantBWG(0, (8, 0))
    @test Ferrite.AMR.transform_facet(adaptive_grid, FacetIndex(4, 4), o) == Ferrite.AMR.OctantBWG(0, (0, 8))


    #simple first and second level refinement
    # first case
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
    # x--x--x-----x           |
    # |  |  |     |           |
    # x--x--x     |           |
    # |  |  |     |           |
    # x--x--x-----x-----------x
    Ferrite.AMR.refine_octant!(adaptive_grid.cells[1], adaptive_grid.cells[1].leaves[1])
    @test length(adaptive_grid.cells[1].leaves) == 4
    for (m, octant) in zip(1:4, adaptive_grid.cells[1].leaves)
        @test octant == Ferrite.AMR.OctantBWG(2, 1, m, adaptive_grid.cells[1].b)
    end
    Ferrite.AMR.refine_octant!(adaptive_grid.cells[1], adaptive_grid.cells[1].leaves[1])

    @test Ferrite.AMR.transform_facet(adaptive_grid, FacetIndex(2, 4), adaptive_grid.cells[1].leaves[5]) == Ferrite.AMR.OctantBWG(1, (0, 8))
    @test Ferrite.AMR.transform_facet(adaptive_grid, FacetIndex(2, 4), adaptive_grid.cells[1].leaves[7]) == Ferrite.AMR.OctantBWG(1, (4, 8))
    @test Ferrite.AMR.transform_facet(adaptive_grid, FacetIndex(3, 3), adaptive_grid.cells[1].leaves[6]) == Ferrite.AMR.OctantBWG(1, (0, -4))
    @test Ferrite.AMR.transform_facet(adaptive_grid, FacetIndex(3, 3), adaptive_grid.cells[1].leaves[7]) == Ferrite.AMR.OctantBWG(1, (4, -4))

    grid_new = Ferrite.AMR.creategrid(adaptive_grid)
    @test length(grid_new.nodes) == 19
    @test length(grid_new.conformity_info) == 4

    # octree holds now 3 first level and 4 second level
    @test length(adaptive_grid.cells[1].leaves) == 7
    for (m, octant) in zip(1:4, adaptive_grid.cells[1].leaves)
        @test octant == Ferrite.AMR.OctantBWG(2, 2, m, adaptive_grid.cells[1].b)
    end


    # second case
    # x-----------x-----------x
    # |           |           |
    # |           |           |
    # |           |           |
    # |           |           |
    # |           |           |
    # x-----x--x--x-----------x
    # |     |  |  |           |
    # |     x--x--x           |
    # |     |  |  |           |
    # x-----x--x--x           |
    # |     |     |           |
    # |     |     |           |
    # x-----x-----x-----------x
    adaptive_grid = ForestBWG(grid, 3)
    Ferrite.AMR.refine_octant!(adaptive_grid.cells[1], adaptive_grid.cells[1].leaves[1])
    Ferrite.AMR.refine_octant!(adaptive_grid.cells[1], adaptive_grid.cells[1].leaves[4])
    @test length(adaptive_grid.cells[1].leaves) == 7
    @test all(getproperty.(adaptive_grid.cells[1].leaves[1:3], :l) .== 1)

    @test Ferrite.AMR.transform_facet(adaptive_grid, FacetIndex(2, 4), adaptive_grid.cells[1].leaves[2]) == Ferrite.AMR.OctantBWG(1, (0, 8))
    @test Ferrite.AMR.transform_facet(adaptive_grid, FacetIndex(2, 4), adaptive_grid.cells[1].leaves[5]) == Ferrite.AMR.OctantBWG(2, (4, 8))
    @test Ferrite.AMR.transform_facet(adaptive_grid, FacetIndex(2, 4), adaptive_grid.cells[1].leaves[7]) == Ferrite.AMR.OctantBWG(2, (6, 8))
    @test Ferrite.AMR.transform_facet(adaptive_grid, FacetIndex(3, 3), adaptive_grid.cells[1].leaves[3]) == Ferrite.AMR.OctantBWG(1, (0, -4))
    @test Ferrite.AMR.transform_facet(adaptive_grid, FacetIndex(3, 3), adaptive_grid.cells[1].leaves[6]) == Ferrite.AMR.OctantBWG(2, (4, -2))
    @test Ferrite.AMR.transform_facet(adaptive_grid, FacetIndex(3, 3), adaptive_grid.cells[1].leaves[7]) == Ferrite.AMR.OctantBWG(2, (6, -2))

    grid_new = Ferrite.AMR.creategrid(adaptive_grid)
    @test length(grid_new.nodes) == 19
    @test length(grid_new.conformity_info) == 4

    # more complex neighborhoods
    grid = generate_simple_disc_grid(Quadrilateral, 6)
    grid.cells[2] = Quadrilateral((grid.cells[2].nodes[2], grid.cells[2].nodes[3], grid.cells[2].nodes[4], grid.cells[2].nodes[1]))
    adaptive_grid = ForestBWG(grid, 3)
    Ferrite.AMR.refine_octant!(adaptive_grid.cells[3], adaptive_grid.cells[3].leaves[1])
    Ferrite.AMR.refine_octant!(adaptive_grid.cells[5], adaptive_grid.cells[5].leaves[1])

    grid_new = Ferrite.AMR.creategrid(adaptive_grid)
    @test length(grid_new.nodes) == 23
    @test length(grid_new.conformity_info) == 4

    ##################################################################
    ####uniform refinement and coarsening for all cells and levels####
    ##################################################################
    adaptive_grid = ForestBWG(grid, 8)
    for l in 1:8
        Ferrite.AMR.refine_all!(adaptive_grid, l)
        for tree in adaptive_grid.cells
            @test all(Ferrite.AMR.morton.(tree.leaves, l, 8) == collect(1:(2^(2 * l))))
        end
    end
    #check montonicity of ancestor_id
    for tree in adaptive_grid.cells
        ids = Ferrite.AMR.ancestor_id.(tree.leaves, (1,), (tree.b,))
        @test issorted(ids)
    end
    #now go back from finest to coarsest
    for l in 7:-1:0
        Ferrite.AMR._coarsen_all!(adaptive_grid)
        for tree in adaptive_grid.cells
            @test all(Ferrite.AMR.morton.(tree.leaves, l, 8) == collect(1:(2^(2 * l))))
        end
    end
    #########################
    # now do the same with 3D
    # some ascii picasso can insert here something beautiful
    #########################
    # TODO add some test with higher refinement level which failed in my REPl (I think 8 should fail)
    # TODO add some rotation and more elaborate case
    grid = generate_grid(Hexahedron, (2, 2, 2))
    adaptive_grid = ForestBWG(grid, 3)
    o = adaptive_grid.cells[1].leaves[1]

    # faces
    @test Ferrite.AMR.transform_facet(adaptive_grid, FacetIndex(1, 2), o) == Ferrite.AMR.OctantBWG(0, (8, 0, 0))
    @test Ferrite.AMR.transform_facet_remote(adaptive_grid, FacetIndex(1, 2), o) == Ferrite.AMR.OctantBWG(0, (-8, 0, 0))
    @test Ferrite.AMR.transform_facet(adaptive_grid, FacetIndex(1, 4), o) == Ferrite.AMR.OctantBWG(0, (0, 8, 0))
    @test Ferrite.AMR.transform_facet_remote(adaptive_grid, FacetIndex(1, 4), o) == Ferrite.AMR.OctantBWG(0, (0, -8, 0))
    @test Ferrite.AMR.transform_facet(adaptive_grid, FacetIndex(1, 6), o) == Ferrite.AMR.OctantBWG(0, (0, 0, 8))
    @test Ferrite.AMR.transform_facet_remote(adaptive_grid, FacetIndex(1, 6), o) == Ferrite.AMR.OctantBWG(0, (0, 0, -8))
    @test Ferrite.AMR.transform_facet(adaptive_grid, FacetIndex(8, 1), o) == Ferrite.AMR.OctantBWG(0, (-8, 0, 0))
    @test Ferrite.AMR.transform_facet_remote(adaptive_grid, FacetIndex(8, 1), o) == Ferrite.AMR.OctantBWG(0, (8, 0, 0))
    @test Ferrite.AMR.transform_facet(adaptive_grid, FacetIndex(8, 3), o) == Ferrite.AMR.OctantBWG(0, (0, -8, 0))
    @test Ferrite.AMR.transform_facet_remote(adaptive_grid, FacetIndex(8, 3), o) == Ferrite.AMR.OctantBWG(0, (0, 8, 0))
    @test Ferrite.AMR.transform_facet(adaptive_grid, FacetIndex(8, 5), o) == Ferrite.AMR.OctantBWG(0, (0, 0, -8))
    @test Ferrite.AMR.transform_facet_remote(adaptive_grid, FacetIndex(8, 5), o) == Ferrite.AMR.OctantBWG(0, (0, 0, 8))

    @test_throws BoundsError Ferrite.AMR.transform_facet(adaptive_grid, FacetIndex(1, 1), o)
    @test_throws BoundsError Ferrite.AMR.transform_facet_remote(adaptive_grid, FacetIndex(1, 1), o)
    @test_throws BoundsError Ferrite.AMR.transform_facet(adaptive_grid, FacetIndex(1, 3), o)
    @test_throws BoundsError Ferrite.AMR.transform_facet_remote(adaptive_grid, FacetIndex(1, 3), o)
    @test_throws BoundsError Ferrite.AMR.transform_facet(adaptive_grid, FacetIndex(1, 5), o)
    @test_throws BoundsError Ferrite.AMR.transform_facet_remote(adaptive_grid, FacetIndex(1, 5), o)
    @test_throws BoundsError Ferrite.AMR.transform_facet(adaptive_grid, FacetIndex(8, 2), o)
    @test_throws BoundsError Ferrite.AMR.transform_facet_remote(adaptive_grid, FacetIndex(8, 2), o)
    @test_throws BoundsError Ferrite.AMR.transform_facet(adaptive_grid, FacetIndex(8, 4), o)
    @test_throws BoundsError Ferrite.AMR.transform_facet_remote(adaptive_grid, FacetIndex(8, 4), o)
    @test_throws BoundsError Ferrite.AMR.transform_facet(adaptive_grid, FacetIndex(8, 6), o)
    @test_throws BoundsError Ferrite.AMR.transform_facet_remote(adaptive_grid, FacetIndex(8, 6), o)

    #corners
    @test Ferrite.AMR.transform_corner(adaptive_grid, VertexIndex(1, 1), o, false) == Ferrite.AMR.OctantBWG(0, (-8, -8, -8))
    @test Ferrite.AMR.transform_corner(adaptive_grid, VertexIndex(1, 2), o, false) == Ferrite.AMR.OctantBWG(0, (8, -8, -8))
    @test Ferrite.AMR.transform_corner(adaptive_grid, VertexIndex(1, 3), o, false) == Ferrite.AMR.OctantBWG(0, (-8, 8, -8))
    @test Ferrite.AMR.transform_corner(adaptive_grid, VertexIndex(1, 4), o, false) == Ferrite.AMR.OctantBWG(0, (8, 8, -8))
    @test Ferrite.AMR.transform_corner(adaptive_grid, VertexIndex(1, 5), o, false) == Ferrite.AMR.OctantBWG(0, (-8, -8, 8))
    @test Ferrite.AMR.transform_corner(adaptive_grid, VertexIndex(1, 6), o, false) == Ferrite.AMR.OctantBWG(0, (8, -8, 8))
    @test Ferrite.AMR.transform_corner(adaptive_grid, VertexIndex(1, 7), o, false) == Ferrite.AMR.OctantBWG(0, (-8, 8, 8))
    @test Ferrite.AMR.transform_corner(adaptive_grid, VertexIndex(1, 8), o, false) == Ferrite.AMR.OctantBWG(0, (8, 8, 8))
    @test_throws BoundsError Ferrite.AMR.transform_corner_remote(adaptive_grid, VertexIndex(1, 1), o, false)
    @test_throws BoundsError Ferrite.AMR.transform_corner_remote(adaptive_grid, VertexIndex(1, 2), o, false)
    @test_throws BoundsError Ferrite.AMR.transform_corner_remote(adaptive_grid, VertexIndex(1, 3), o, false)
    @test_throws BoundsError Ferrite.AMR.transform_corner_remote(adaptive_grid, VertexIndex(1, 4), o, false)
    @test_throws BoundsError Ferrite.AMR.transform_corner_remote(adaptive_grid, VertexIndex(1, 5), o, false)
    @test_throws BoundsError Ferrite.AMR.transform_corner_remote(adaptive_grid, VertexIndex(1, 6), o, false)
    @test_throws BoundsError Ferrite.AMR.transform_corner_remote(adaptive_grid, VertexIndex(1, 7), o, false)
    Ferrite.AMR.transform_corner_remote(adaptive_grid, VertexIndex(1, 8), o, false) == Ferrite.AMR.OctantBWG(0, (-8, -8, -8))

    #edges
    @test_throws BoundsError Ferrite.AMR.transform_edge(adaptive_grid, EdgeIndex(1, 1), o, false)
    @test_throws BoundsError Ferrite.AMR.transform_edge(adaptive_grid, EdgeIndex(1, 2), o, false)
    @test_throws BoundsError Ferrite.AMR.transform_edge(adaptive_grid, EdgeIndex(1, 3), o, false)
    @test_throws BoundsError Ferrite.AMR.transform_edge_remote(adaptive_grid, EdgeIndex(1, 1), o, false)
    @test_throws BoundsError Ferrite.AMR.transform_edge_remote(adaptive_grid, EdgeIndex(1, 2), o, false)
    @test_throws BoundsError Ferrite.AMR.transform_edge_remote(adaptive_grid, EdgeIndex(1, 3), o, false)
    @test Ferrite.AMR.transform_edge(adaptive_grid, EdgeIndex(1, 4), o, false) == Ferrite.AMR.OctantBWG(0, (0, 8, 8))
    @test Ferrite.AMR.transform_edge_remote(adaptive_grid, EdgeIndex(1, 4), o, false) == Ferrite.AMR.OctantBWG(0, (0, -8, -8))
    @test_throws BoundsError Ferrite.AMR.transform_edge(adaptive_grid, EdgeIndex(1, 5), o, false)
    @test_throws BoundsError Ferrite.AMR.transform_edge(adaptive_grid, EdgeIndex(1, 6), o, false)
    @test_throws BoundsError Ferrite.AMR.transform_edge(adaptive_grid, EdgeIndex(1, 7), o, false)
    @test_throws BoundsError Ferrite.AMR.transform_edge_remote(adaptive_grid, EdgeIndex(1, 5), o, false)
    @test_throws BoundsError Ferrite.AMR.transform_edge_remote(adaptive_grid, EdgeIndex(1, 6), o, false)
    @test_throws BoundsError Ferrite.AMR.transform_edge_remote(adaptive_grid, EdgeIndex(1, 7), o, false)
    @test Ferrite.AMR.transform_edge(adaptive_grid, EdgeIndex(1, 8), o, false) == Ferrite.AMR.OctantBWG(0, (8, 0, 8))
    @test Ferrite.AMR.transform_edge_remote(adaptive_grid, EdgeIndex(1, 8), o, false) == Ferrite.AMR.OctantBWG(0, (-8, 0, -8))
    @test_throws BoundsError Ferrite.AMR.transform_edge(adaptive_grid, EdgeIndex(1, 9), o, false)
    @test_throws BoundsError Ferrite.AMR.transform_edge(adaptive_grid, EdgeIndex(1, 10), o, false)
    @test_throws BoundsError Ferrite.AMR.transform_edge(adaptive_grid, EdgeIndex(1, 11), o, false)
    @test_throws BoundsError Ferrite.AMR.transform_edge_remote(adaptive_grid, EdgeIndex(1, 9), o, false)
    @test_throws BoundsError Ferrite.AMR.transform_edge_remote(adaptive_grid, EdgeIndex(1, 10), o, false)
    @test_throws BoundsError Ferrite.AMR.transform_edge_remote(adaptive_grid, EdgeIndex(1, 11), o, false)
    @test Ferrite.AMR.transform_edge(adaptive_grid, EdgeIndex(1, 12), o, false) == Ferrite.AMR.OctantBWG(0, (8, 8, 0))
    @test Ferrite.AMR.transform_edge_remote(adaptive_grid, EdgeIndex(1, 12), o, false) == Ferrite.AMR.OctantBWG(0, (-8, -8, 0))

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
    adaptive_grid = ForestBWG(grid, 3)
    @test Ferrite.AMR.transform_corner(adaptive_grid, 7, 3, Ferrite.AMR.OctantBWG(0, (0, 0, 0)), false) == Ferrite.AMR.OctantBWG(0, (-8, 8, -8))

    #refinement
    Ferrite.AMR.refine_octant!(adaptive_grid.cells[1], adaptive_grid.cells[1].leaves[1])
    @test length(adaptive_grid.cells[1].leaves) == 8
    for (m, octant) in zip(1:8, adaptive_grid.cells[1].leaves)
        @test octant == Ferrite.AMR.OctantBWG(3, 1, m, adaptive_grid.cells[1].b)
    end
    Ferrite.AMR.refine_octant!(adaptive_grid.cells[1], adaptive_grid.cells[1].leaves[1])
    @test length(adaptive_grid.cells[1].leaves) == 15
    for (m, octant) in zip(1:8, adaptive_grid.cells[1].leaves)
        @test octant == Ferrite.AMR.OctantBWG(3, 2, m, adaptive_grid.cells[1].b)
    end
    adaptive_grid = ForestBWG(grid, 3)
    Ferrite.AMR.refine_octant!(adaptive_grid.cells[1], adaptive_grid.cells[1].leaves[1])
    Ferrite.AMR.refine_octant!(adaptive_grid.cells[1], adaptive_grid.cells[1].leaves[4])
    @test length(adaptive_grid.cells[1].leaves) == 15
    @test all(getproperty.(adaptive_grid.cells[1].leaves[1:3], :l) .== 1)
    @test all(getproperty.(adaptive_grid.cells[1].leaves[4:11], :l) .== 2)
    @test all(getproperty.(adaptive_grid.cells[1].leaves[12:end], :l) .== 1)
    adaptive_grid = ForestBWG(grid, 5)
    #go from coarsest to finest uniformly
    for l in 1:5
        Ferrite.AMR.refine_all!(adaptive_grid, l)
        for tree in adaptive_grid.cells
            @test all(Ferrite.AMR.morton.(tree.leaves, l, 5) == collect(1:(2^(3 * l))))
        end
    end
    #now go back from finest to coarsest
    for l in 4:-1:0
        Ferrite.AMR._coarsen_all!(adaptive_grid)
        for tree in adaptive_grid.cells
            @test all(Ferrite.AMR.morton.(tree.leaves, l, 5) == collect(1:(2^(3 * l))))
        end
    end

    # Single
    adaptive_grid = ForestBWG(grid, 3)
    Ferrite.AMR.refine_all!(adaptive_grid, 1)
    Ferrite.AMR.refine_octant!(adaptive_grid.cells[1], adaptive_grid.cells[1].leaves[8])
    transferred_grid = Ferrite.creategrid(adaptive_grid)
    @test unique(transferred_grid.nodes) == transferred_grid.nodes
    # Unrefined grid has 5 ^ dim nodes and the refined element introduces 6 face center, 12 edge center and 1 volume center nodes
    @test length(transferred_grid.nodes) == 5^3 + (6 + 12 + 1)
    # 6 faces and 12 edges of the single refined element induces one hanging node each
    @test length(transferred_grid.conformity_info) == 6 + 12

    adaptive_grid = ForestBWG(grid, 3)
    Ferrite.AMR.refine_all!(adaptive_grid, 1)
    Ferrite.AMR.refine_octant!(adaptive_grid.cells[1], adaptive_grid.cells[1].leaves[1])
    transferred_grid = Ferrite.creategrid(adaptive_grid)
    @test unique(transferred_grid.nodes) == transferred_grid.nodes
    # Unrefined grid has 5 ^ dim nodes and the refined element introduces 6 face center, 12 edge center and 1 volume center nodes
    @test length(transferred_grid.nodes) == 5^3 + (6 + 12 + 1)
    # 6 faces and 12 edges of the single refined element induces one hanging node each - minus 3 faces and 3 edges on the outer boundary
    @test length(transferred_grid.conformity_info) == 6 + 12 - 2 * 3

    adaptive_grid = ForestBWG(grid, 3)
    Ferrite.AMR.refine_all!(adaptive_grid, 1)
    Ferrite.AMR.refine_octant!(adaptive_grid.cells[8], adaptive_grid.cells[8].leaves[8])
    transferred_grid = Ferrite.creategrid(adaptive_grid)
    @test unique(transferred_grid.nodes) == transferred_grid.nodes
    # Unrefined grid has 5 ^ dim nodes and the refined element introduces 6 face center, 12 edge center and 1 volume center nodes
    @test length(transferred_grid.nodes) == 5^3 + (6 + 12 + 1)
    # 6 faces and 12 edges of the single refined element induces one hanging node each - minus 3 faces and 3 edges on the outer boundary
    @test length(transferred_grid.conformity_info) == 6 + 12 - 2 * 3

    adaptive_grid = ForestBWG(grid, 3)
    Ferrite.AMR.refine_all!(adaptive_grid, 1)
    Ferrite.AMR.refine_octant!(adaptive_grid.cells[8], adaptive_grid.cells[8].leaves[1])
    transferred_grid = Ferrite.creategrid(adaptive_grid)
    # Unrefined grid has 5 ^ dim nodes and the refined element introduces 6 face center, 12 edge center and 1 volume center nodes
    @test length(transferred_grid.nodes) == 5^3 + (6 + 12 + 1)
    # 6 faces and 12 edges of the single refined element induces one hanging node each
    @test length(transferred_grid.conformity_info) == 6 + 12

    # Combined
    adaptive_grid = ForestBWG(grid, 3)
    Ferrite.AMR.refine_all!(adaptive_grid, 1)
    Ferrite.AMR.refine_octant!(adaptive_grid.cells[1], adaptive_grid.cells[1].leaves[8])
    Ferrite.AMR.refine_octant!(adaptive_grid.cells[1], adaptive_grid.cells[1].leaves[1])
    transferred_grid = Ferrite.creategrid(adaptive_grid)
    @test unique(transferred_grid.nodes) == transferred_grid.nodes
    @test length(transferred_grid.nodes) == 5^3 + 2 * (6 + 12 + 1)
    @test length(transferred_grid.conformity_info) == 2 * (6 + 12) - 2 * 3

    adaptive_grid = ForestBWG(grid, 3)
    Ferrite.AMR.refine_all!(adaptive_grid, 1)
    Ferrite.AMR.refine_octant!(adaptive_grid.cells[8], adaptive_grid.cells[8].leaves[8])
    Ferrite.AMR.refine_octant!(adaptive_grid.cells[8], adaptive_grid.cells[8].leaves[1])
    transferred_grid = Ferrite.creategrid(adaptive_grid)
    @test unique(transferred_grid.nodes) == transferred_grid.nodes
    @test length(transferred_grid.nodes) == 5^3 + 2 * (6 + 12 + 1)
    @test length(transferred_grid.conformity_info) == 2 * (6 + 12) - 2 * 3

    # Combined
    adaptive_grid = ForestBWG(grid, 3)
    Ferrite.AMR.refine_all!(adaptive_grid, 1)
    Ferrite.AMR.refine_octant!(adaptive_grid.cells[1], adaptive_grid.cells[1].leaves[8])
    Ferrite.AMR.refine_octant!(adaptive_grid.cells[1], adaptive_grid.cells[1].leaves[1])
    Ferrite.AMR.refine_octant!(adaptive_grid.cells[8], adaptive_grid.cells[8].leaves[8])
    Ferrite.AMR.refine_octant!(adaptive_grid.cells[8], adaptive_grid.cells[8].leaves[1])
    transferred_grid = Ferrite.creategrid(adaptive_grid)
    @test unique(transferred_grid.nodes) == transferred_grid.nodes
    @test length(transferred_grid.nodes) == 5^3 + 4 * (6 + 12 + 1)
    @test length(transferred_grid.conformity_info) == 4 * (6 + 12) - 2 * 3 - 2 * 3

    # Combined and not rotated
    adaptive_grid = ForestBWG(grid, 3)
    Ferrite.AMR.refine_all!(adaptive_grid, 1)
    Ferrite.AMR.refine_octant!(adaptive_grid.cells[1], adaptive_grid.cells[1].leaves[8])
    Ferrite.AMR.refine_octant!(adaptive_grid.cells[1], adaptive_grid.cells[1].leaves[1])
    Ferrite.AMR.refine_octant!(adaptive_grid.cells[6], adaptive_grid.cells[6].leaves[6])
    Ferrite.AMR.refine_octant!(adaptive_grid.cells[6], adaptive_grid.cells[6].leaves[3])
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
    Ferrite.AMR.refine_all!(adaptive_grid, 1)
    Ferrite.AMR.refine_octant!(adaptive_grid.cells[1], adaptive_grid.cells[1].leaves[8])
    Ferrite.AMR.refine_octant!(adaptive_grid.cells[1], adaptive_grid.cells[1].leaves[1])
    Ferrite.AMR.refine_octant!(adaptive_grid.cells[7], adaptive_grid.cells[7].leaves[6])
    Ferrite.AMR.refine_octant!(adaptive_grid.cells[7], adaptive_grid.cells[7].leaves[3])
    transferred_grid = Ferrite.creategrid(adaptive_grid)
    @test unique(transferred_grid.nodes) == transferred_grid.nodes
    # +5^3 on the coarse grid
    # +4 refined elements a 6 face nodes, 12 edge nodes and 1 volume nodes
    # -1 shared node between tree 1 and 7
    @test length(transferred_grid.nodes) == 5^3 + 4 * (6 + 12 + 1) - 1
    # 30 constraints from tree 1 + 30 from rotated tree 7 - 1 shared on common edge
    @test length(transferred_grid.conformity_info) == 59

    # Reproducer test for Fig.3 BWG 11
    grid = generate_grid(Hexahedron, (2, 1, 1))
    # (a)
    adaptive_grid = ForestBWG(grid, 3)
    Ferrite.AMR.refine_octant!(adaptive_grid.cells[2], adaptive_grid.cells[2].leaves[1])
    Ferrite.AMR.refine_octant!(adaptive_grid.cells[2], adaptive_grid.cells[2].leaves[3])
    @test adaptive_grid.cells[2].leaves[3 + 4] == Ferrite.AMR.OctantBWG(2, (0, 4, 2))
    @test Ferrite.AMR.transform_facet(adaptive_grid, FacetIndex(1, 2), adaptive_grid.cells[2].leaves[3 + 4]) == Ferrite.AMR.OctantBWG(2, (8, 4, 2))
    # (b) Rotate elements topologically
    grid.cells[1] = Hexahedron((grid.cells[1].nodes[2], grid.cells[1].nodes[3], grid.cells[1].nodes[4], grid.cells[1].nodes[1], grid.cells[1].nodes[6], grid.cells[1].nodes[7], grid.cells[1].nodes[8], grid.cells[1].nodes[5]))
    grid.cells[2] = Hexahedron((grid.cells[2].nodes[4], grid.cells[2].nodes[1], grid.cells[2].nodes[2], grid.cells[2].nodes[3], grid.cells[2].nodes[8], grid.cells[2].nodes[5], grid.cells[2].nodes[6], grid.cells[2].nodes[7]))
    # grid.cells[2] = Hexahedron((grid.cells[2].nodes[1], grid.cells[2].nodes[3], grid.cells[2].nodes[4], grid.cells[2].nodes[8], grid.cells[2].nodes[6], grid.cells[2].nodes[2], grid.cells[2].nodes[7], grid.cells[2].nodes[5])) How to rotate along diagonal? :)
    adaptive_grid = ForestBWG(grid, 3)
    Ferrite.AMR.refine_octant!(adaptive_grid.cells[2], adaptive_grid.cells[2].leaves[1])
    Ferrite.AMR.refine_octant!(adaptive_grid.cells[2], adaptive_grid.cells[2].leaves[1])
    @test adaptive_grid.cells[2].leaves[6] == Ferrite.AMR.OctantBWG(2, (2, 0, 2))
    @test Ferrite.AMR.transform_facet(adaptive_grid, FacetIndex(1, 3), adaptive_grid.cells[2].leaves[6]) == Ferrite.AMR.OctantBWG(2, (4, -2, 2))
end

@testset "ForestBWG AbstractGrid Interfacing" begin
    maxlevel = 3
    grid = generate_grid(Quadrilateral, (2, 2))
    adaptive_grid = ForestBWG(grid, maxlevel)
    for l in 1:maxlevel
        Ferrite.AMR.refine_all!(adaptive_grid, l)
        @test getncells(adaptive_grid) == 2^(2 * l) * 4 == length(getcells(adaptive_grid))
    end
end

@testset "Balancing" begin
    #2D cases
    #simple one quad with one additional non-allowed non-conformity level
    grid = generate_grid(Quadrilateral, (1, 1))
    adaptive_grid = ForestBWG(grid, 3)
    Ferrite.AMR.refine_all!(adaptive_grid, 1)
    Ferrite.AMR.refine_octant!(adaptive_grid.cells[1], adaptive_grid.cells[1].leaves[2])
    Ferrite.AMR.refine_octant!(adaptive_grid.cells[1], adaptive_grid.cells[1].leaves[6])
    Ferrite.AMR.refine_octant!(adaptive_grid.cells[1], adaptive_grid.cells[1].leaves[6])
    balanced = Ferrite.AMR.balancetree(adaptive_grid.cells[1])
    @test length(balanced.leaves) == 16

    #more complex non-conformity level 3 and 4 that needs to be balanced
    adaptive_grid = ForestBWG(grid, 5)
    Ferrite.AMR.refine_all!(adaptive_grid, 1)
    Ferrite.AMR.refine_octant!(adaptive_grid.cells[1], adaptive_grid.cells[1].leaves[2])
    Ferrite.AMR.refine_octant!(adaptive_grid.cells[1], adaptive_grid.cells[1].leaves[4])
    Ferrite.AMR.refine_octant!(adaptive_grid.cells[1], adaptive_grid.cells[1].leaves[7])
    Ferrite.AMR.refine_octant!(adaptive_grid.cells[1], adaptive_grid.cells[1].leaves[12])
    Ferrite.AMR.refine_octant!(adaptive_grid.cells[1], adaptive_grid.cells[1].leaves[12])
    Ferrite.AMR.refine_octant!(adaptive_grid.cells[1], adaptive_grid.cells[1].leaves[15])
    Ferrite.AMR.refine_octant!(adaptive_grid.cells[1], adaptive_grid.cells[1].leaves[16])
    balanced = Ferrite.AMR.balancetree(adaptive_grid.cells[1])
    @test length(balanced.leaves) == 64

    grid = generate_grid(Quadrilateral, (2, 1))
    adaptive_grid = ForestBWG(grid, 2)
    Ferrite.AMR.refine_octant!(adaptive_grid.cells[1], adaptive_grid.cells[1].leaves[1])
    Ferrite.AMR.refine_octant!(adaptive_grid.cells[1], adaptive_grid.cells[1].leaves[2])
    Ferrite.AMR.balanceforest!(adaptive_grid)
    @test Ferrite.AMR.getncells(adaptive_grid) == 11

    grid = generate_grid(Quadrilateral, (2, 2))
    adaptive_grid = ForestBWG(grid, 2)
    Ferrite.AMR.refine_octant!(adaptive_grid.cells[1], adaptive_grid.cells[1].leaves[1])
    Ferrite.AMR.refine_octant!(adaptive_grid.cells[1], adaptive_grid.cells[1].leaves[4])
    Ferrite.AMR.balanceforest!(adaptive_grid)
    @test Ferrite.AMR.getncells(adaptive_grid) == 19

    # 2D example with balancing over a corner connection that is not within the topology tables
    grid = generate_grid(Quadrilateral, (2, 1))
    adaptive_grid = ForestBWG(grid, 3)
    Ferrite.AMR.refine_octant!(adaptive_grid.cells[1], adaptive_grid.cells[1].leaves[1])
    Ferrite.AMR.refine_octant!(adaptive_grid.cells[1], adaptive_grid.cells[1].leaves[2])
    Ferrite.AMR.refine_octant!(adaptive_grid.cells[1], adaptive_grid.cells[1].leaves[5])
    Ferrite.AMR.balanceforest!(adaptive_grid)
    @test Ferrite.AMR.getncells(adaptive_grid) == 23

    #corner balance case but rotated
    grid = generate_grid(Quadrilateral, (2, 1))
    grid.cells[1] = Quadrilateral((grid.cells[1].nodes[2], grid.cells[1].nodes[3], grid.cells[1].nodes[4], grid.cells[1].nodes[1]))
    adaptive_grid = ForestBWG(grid, 3)
    Ferrite.AMR.refine_octant!(adaptive_grid.cells[1], adaptive_grid.cells[1].leaves[1])
    Ferrite.AMR.refine_octant!(adaptive_grid.cells[1], adaptive_grid.cells[1].leaves[1])
    Ferrite.AMR.refine_octant!(adaptive_grid.cells[1], adaptive_grid.cells[1].leaves[2])
    Ferrite.AMR.balanceforest!(adaptive_grid)
    @test Ferrite.AMR.getncells(adaptive_grid) == 23

    # 3D case intra tree simple test, non conformity level 2
    grid = generate_grid(Hexahedron, (1, 1, 1))
    adaptive_grid = ForestBWG(grid, 3)
    Ferrite.AMR.refine_all!(adaptive_grid, 1)
    Ferrite.AMR.refine_octant!(adaptive_grid.cells[1], adaptive_grid.cells[1].leaves[2])
    Ferrite.AMR.refine_octant!(adaptive_grid.cells[1], adaptive_grid.cells[1].leaves[6])
    Ferrite.AMR.refine_octant!(adaptive_grid.cells[1], adaptive_grid.cells[1].leaves[6])
    balanced = Ferrite.AMR.balancetree(adaptive_grid.cells[1])
    @test length(balanced.leaves) == 43

    #3D case intra tree non conformity level 3 at two different places
    adaptive_grid = ForestBWG(grid, 4)
    Ferrite.AMR.refine_all!(adaptive_grid, 1)
    Ferrite.AMR.refine_octant!(adaptive_grid.cells[1], adaptive_grid.cells[1].leaves[2])
    Ferrite.AMR.refine_octant!(adaptive_grid.cells[1], adaptive_grid.cells[1].leaves[4])
    Ferrite.AMR.refine_octant!(adaptive_grid.cells[1], adaptive_grid.cells[1].leaves[7])
    Ferrite.AMR.refine_octant!(adaptive_grid.cells[1], adaptive_grid.cells[1].leaves[12])
    Ferrite.AMR.refine_octant!(adaptive_grid.cells[1], adaptive_grid.cells[1].leaves[28])
    Ferrite.AMR.refine_octant!(adaptive_grid.cells[1], adaptive_grid.cells[1].leaves[29])
    Ferrite.AMR.refine_octant!(adaptive_grid.cells[1], adaptive_grid.cells[1].leaves[37])
    Ferrite.AMR.refine_octant!(adaptive_grid.cells[1], adaptive_grid.cells[1].leaves[39])
    balanced = Ferrite.AMR.balancetree(adaptive_grid.cells[1])
    @test length(balanced.leaves) == 127

    #3D case inter tree non conformity level 3 at two different places
    grid = generate_grid(Hexahedron, (2, 2, 2))
    adaptive_grid = ForestBWG(grid, 4)
    Ferrite.AMR.refine_octant!(adaptive_grid.cells[1], adaptive_grid.cells[1].leaves[1])
    Ferrite.AMR.refine_octant!(adaptive_grid.cells[1], adaptive_grid.cells[1].leaves[2])
    Ferrite.AMR.refine_octant!(adaptive_grid.cells[1], adaptive_grid.cells[1].leaves[4])
    #Ferrite.AMR.refine_octant!(adaptive_grid.cells[1],adaptive_grid.cells[1].leaves[7])
    Ferrite.AMR.refine_octant!(adaptive_grid.cells[7], adaptive_grid.cells[7].leaves[1])
    Ferrite.AMR.refine_octant!(adaptive_grid.cells[7], adaptive_grid.cells[7].leaves[1])
    #Ferrite.AMR.refine_octant!(adaptive_grid.cells[7],adaptive_grid.cells[7].leaves[1])
    Ferrite.AMR.balanceforest!(adaptive_grid)
    transferred_grid_ref = Ferrite.AMR.creategrid(adaptive_grid)

    # Rotate three dimensional case
    grid = generate_grid(Hexahedron, (2, 2, 2))
    # This is our root mesh top view
    # x-----------x-----------x
    # |7    2    6|8    4    7|
    # |           |           |
    # |     ^     |     ^     |
    # |4    |    3|1    |    2|
    # |  <--+     |     +-->  |
    # |           |           |
    # |8    1    5|5    3    6|
    # x-----------x-----------x
    # |8    4    7|8    4    7|
    # |           |           |
    # |     ^     |     ^     |
    # |1    |    2|1    |    2|
    # |     +-->  |     +-->  |
    # |           |           |
    # |5    3    6|5    3    6|
    # x-----------x-----------x
    # Rotate face topologically
    grid.cells[7] = Hexahedron((grid.cells[7].nodes[2], grid.cells[7].nodes[3], grid.cells[7].nodes[4], grid.cells[7].nodes[1], grid.cells[7].nodes[4 + 2], grid.cells[7].nodes[4 + 3], grid.cells[7].nodes[4 + 4], grid.cells[7].nodes[4 + 1]))
    grid.cells[7] = Hexahedron((grid.cells[7].nodes[2], grid.cells[7].nodes[3], grid.cells[7].nodes[4], grid.cells[7].nodes[1], grid.cells[7].nodes[4 + 2], grid.cells[7].nodes[4 + 3], grid.cells[7].nodes[4 + 4], grid.cells[7].nodes[4 + 1]))
    adaptive_grid = ForestBWG(grid, 3)
    Ferrite.AMR.refine_octant!(adaptive_grid.cells[1], adaptive_grid.cells[1].leaves[1])
    Ferrite.AMR.refine_octant!(adaptive_grid.cells[1], adaptive_grid.cells[1].leaves[2])
    Ferrite.AMR.refine_octant!(adaptive_grid.cells[1], adaptive_grid.cells[1].leaves[4])
    #Ferrite.AMR.refine_octant!(adaptive_grid.cells[1],adaptive_grid.cells[1].leaves[7])
    Ferrite.AMR.refine_octant!(adaptive_grid.cells[7], adaptive_grid.cells[7].leaves[1])
    Ferrite.AMR.refine_octant!(adaptive_grid.cells[7], adaptive_grid.cells[7].leaves[1])
    Ferrite.AMR.balanceforest!(adaptive_grid)
    transferred_grid = Ferrite.AMR.creategrid(adaptive_grid)
    @test length(transferred_grid.cells) == length(transferred_grid_ref.cells)
    @test length(transferred_grid.cells) == 92

    # edge balancing for new introduced connection that is not within topology table
    grid = generate_grid(Hexahedron, (2, 1, 1))
    adaptive_grid = ForestBWG(grid, 3)
    Ferrite.AMR.refine!(adaptive_grid, [1, 2])
    Ferrite.AMR.balanceforest!(adaptive_grid)
    Ferrite.AMR.refine!(adaptive_grid, [4])
    Ferrite.AMR.balanceforest!(adaptive_grid)
    Ferrite.AMR.refine!(adaptive_grid, [5])
    Ferrite.AMR.balanceforest!(adaptive_grid)
    @test Ferrite.AMR.getncells(adaptive_grid) == 51

    #another edge balancing case
    grid = generate_grid(Hexahedron, (2, 2, 2))
    adaptive_grid = ForestBWG(grid, 3)
    Ferrite.AMR.refine!(adaptive_grid, 1)
    Ferrite.AMR.balanceforest!(adaptive_grid)
    Ferrite.AMR.refine!(adaptive_grid, [2, 4, 6, 8])
    Ferrite.AMR.balanceforest!(adaptive_grid)
    Ferrite.AMR.refine!(adaptive_grid, 34)
    Ferrite.AMR.balanceforest!(adaptive_grid)
    # 141 = 134 + 7: balancing corner connections introduced by refinement (not present in
    # the macro topology) refines one additional leaf in this configuration
    @test Ferrite.AMR.getncells(adaptive_grid) == 141

    #yet another edge balancing case
    grid = generate_grid(Hexahedron, (2, 2, 2))
    adaptive_grid = ForestBWG(grid, 3)
    Ferrite.AMR.refine!(adaptive_grid, 1)
    Ferrite.AMR.balanceforest!(adaptive_grid)
    Ferrite.AMR.refine!(adaptive_grid, [2, 4, 6, 8])
    Ferrite.AMR.balanceforest!(adaptive_grid)
    Ferrite.AMR.refine!(adaptive_grid, 30)
    Ferrite.AMR.balanceforest!(adaptive_grid)
    # 127 = 120 + 7: one additional leaf from refinement-introduced corner balancing
    @test Ferrite.AMR.getncells(adaptive_grid) == 127

    grid = generate_grid(Hexahedron, (2, 2, 2))
    adaptive_grid = ForestBWG(grid, 3)
    Ferrite.AMR.refine_octant!(adaptive_grid.cells[4], adaptive_grid.cells[4].leaves[1])
    Ferrite.AMR.balanceforest!(adaptive_grid)
    @test Ferrite.AMR.getncells(adaptive_grid) == 15

    #yet another edge balancing case
    grid = generate_grid(Hexahedron, (2, 2, 2))
    adaptive_grid = ForestBWG(grid, 3)
    Ferrite.AMR.refine_octant!(adaptive_grid.cells[4], adaptive_grid.cells[4].leaves[1])
    Ferrite.AMR.balanceforest!(adaptive_grid)
    Ferrite.AMR.refine_octant!(adaptive_grid.cells[4], adaptive_grid.cells[4].leaves[1])
    Ferrite.AMR.balanceforest!(adaptive_grid)
    @test Ferrite.AMR.getncells(adaptive_grid) == 43

    #yet another edge balancing case
    grid = generate_grid(Hexahedron, (2, 2, 2))
    adaptive_grid = ForestBWG(grid, 3)
    Ferrite.AMR.refine_octant!(adaptive_grid.cells[4], adaptive_grid.cells[4].leaves[1])
    Ferrite.AMR.balanceforest!(adaptive_grid)
    Ferrite.AMR.refine_octant!(adaptive_grid.cells[4], adaptive_grid.cells[4].leaves[1])
    Ferrite.AMR.balanceforest!(adaptive_grid)
    Ferrite.AMR.refine_octant!(adaptive_grid.cells[3], adaptive_grid.cells[3].leaves[2])
    Ferrite.AMR.refine_octant!(adaptive_grid.cells[4], adaptive_grid.cells[4].leaves[10])
    Ferrite.AMR.refine_octant!(adaptive_grid.cells[4], adaptive_grid.cells[4].leaves[3])
    Ferrite.AMR.balanceforest!(adaptive_grid)
    @test Ferrite.AMR.getncells(adaptive_grid) == 71

    #yet another edge balancing case
    grid = generate_grid(Hexahedron, (2, 2, 2))
    adaptive_grid = ForestBWG(grid, 3)
    Ferrite.AMR.refine_octant!(adaptive_grid.cells[4], adaptive_grid.cells[4].leaves[1])
    Ferrite.AMR.balanceforest!(adaptive_grid)
    Ferrite.AMR.refine_octant!(adaptive_grid.cells[4], adaptive_grid.cells[4].leaves[1])
    Ferrite.AMR.balanceforest!(adaptive_grid)
    Ferrite.AMR.refine_octant!(adaptive_grid.cells[4], adaptive_grid.cells[4].leaves[7])
    Ferrite.AMR.balanceforest!(adaptive_grid)
    # 127 = 120 + 7: one additional leaf from refinement-introduced corner balancing
    @test Ferrite.AMR.getncells(adaptive_grid) == 127
end

@testset "corner balance across refinement-introduced connections" begin
    # Exhaustive 2:1 audit on the physical leaf boxes (valid for the axis-aligned macro cells
    # of generate_grid): counts leaf pairs whose closed bounding boxes touch (face, edge or
    # corner contact) while their levels differ by 2 or more — zero for a fully balanced forest.
    function count_unbalanced_contacts(forest::ForestBWG{dim}) where {dim}
        b = forest.cells[1].b
        m = Float64(Ferrite.AMR._maximum_size(b))
        boxes = Tuple{Int, Vector{Float64}, Vector{Float64}}[]
        for (k, tree) in enumerate(forest.cells)
            corners = collect(Ferrite.AMR._treecorners(forest, k))
            lo_t = [minimum(c[d] for c in corners) for d in 1:dim]
            hi_t = [maximum(c[d] for c in corners) for d in 1:dim]
            for o in tree.leaves
                h = Float64(Ferrite.AMR._compute_size(b, o.l))
                lo = lo_t .+ (collect(Float64.(o.xyz)) ./ m) .* (hi_t .- lo_t)
                hi = lo_t .+ ((collect(Float64.(o.xyz)) .+ h) ./ m) .* (hi_t .- lo_t)
                push!(boxes, (Int(o.l), lo, hi))
            end
        end
        nviol = 0
        for i in eachindex(boxes), j in (i + 1):length(boxes)
            li, loi, hii = boxes[i]
            lj, loj, hij = boxes[j]
            abs(li - lj) < 2 && continue
            touching = all(d -> min(hii[d], hij[d]) - max(loi[d], loj[d]) >= -1.0e-12, 1:dim)
            nviol += touching
        end
        return nviol
    end
    # Repeatedly refine the leaf of `forest.cells[treeid]` selected by `pred`, then balance.
    function refine_towards_and_balance!(forest, treeid, nsteps, pred)
        for _ in 1:nsteps
            t = forest.cells[treeid]
            Ferrite.AMR.refine_octant!(t, only(filter(pred, t.leaves)))
        end
        Ferrite.AMR.balanceforest!(forest)
        return forest
    end

    # A refined leaf's corner can touch another tree at a point that is NOT a macro vertex —
    # a corner connection "newly introduced" by refinement, absent from the macro topology.
    # Balancing must route these through the face (2D/3D) or edge (3D) the corner lies on.

    # 3D corner in the middle of the shared macro face
    forest = ForestBWG(generate_grid(Hexahedron, (2, 1, 1)), 4)
    m = Int(Ferrite.AMR._maximum_size(forest.cells[1].b))
    Ferrite.AMR.refine_octant!(forest.cells[2], forest.cells[2].leaves[1])
    refine_towards_and_balance!(forest, 2, 2, o -> Int(o.xyz[1]) == 0 && Int(o.xyz[2]) == m ÷ 2 && Int(o.xyz[3]) == m ÷ 2)
    @test count_unbalanced_contacts(forest) == 0

    # 3D corner in the middle of the shared macro edge (diagonal tree is an exclusive edge neighbour)
    forest = ForestBWG(generate_grid(Hexahedron, (2, 2, 1)), 4)
    Ferrite.AMR.refine_octant!(forest.cells[4], forest.cells[4].leaves[1])
    refine_towards_and_balance!(forest, 4, 2, o -> Int(o.xyz[1]) == 0 && Int(o.xyz[2]) == 0 && Int(o.xyz[3]) == m ÷ 2)
    @test count_unbalanced_contacts(forest) == 0
    # the balanced forest must still materialize into a conforming constrained space
    g = Ferrite.AMR.creategrid(forest)
    lin(x) = 1.0 + 2.0 * x[1] - 3.0 * x[2] + 0.5 * x[3]
    @test all(lin(g.nodes[h].x) ≈ sum(lin(g.nodes[mm].x) for mm in ms) / length(ms) for (h, ms) in g.conformity_info)
    @test all(!haskey(g.conformity_info, mm) for (h, ms) in g.conformity_info for mm in ms)

    # 2D corner in the middle of the shared macro face while the tree's root corner in that
    # direction has an exclusive vertex neighbour (grid center) — must not shadow the fallback
    forest = ForestBWG(generate_grid(Quadrilateral, (2, 2)), 4)
    Ferrite.AMR.refine_octant!(forest.cells[2], forest.cells[2].leaves[1])
    refine_towards_and_balance!(forest, 2, 2, o -> Int(o.xyz[1]) == 0 && Int(o.xyz[2]) + Int(Ferrite.AMR._compute_size(forest.cells[2].b, o.l)) == m ÷ 2)
    @test count_unbalanced_contacts(forest) == 0

    # regression: macro-corner refinement (handled via vertex_vertex_neighbor) stays balanced
    forest = ForestBWG(generate_grid(Hexahedron, (2, 2, 2)), 4)
    for _ in 1:3
        t = forest.cells[4]
        leaf = only(filter(o -> o.xyz[1] == 0 && o.xyz[2] == 0 && Int(o.xyz[3]) + Int(Ferrite.AMR._compute_size(t.b, o.l)) == m, t.leaves))
        Ferrite.AMR.refine_octant!(t, leaf)
    end
    Ferrite.AMR.balanceforest!(forest)
    @test count_unbalanced_contacts(forest) == 0

    # regression: 2D through-face corner with no macro corner connection at all
    forest = ForestBWG(generate_grid(Quadrilateral, (2, 1)), 4)
    Ferrite.AMR.refine_octant!(forest.cells[2], forest.cells[2].leaves[1])
    refine_towards_and_balance!(forest, 2, 2, o -> Int(o.xyz[1]) == 0 && Int(o.xyz[2]) == m ÷ 2)
    @test count_unbalanced_contacts(forest) == 0
end

@testset "corner balance at a multi-tree vertex" begin
    # Five quads sharing a central vertex, with rotated connectivity so the center sits at a
    # different local corner in each tree (as in unstructured meshes). The vertex-only
    # neighbor lists at the center then have two entries whose corner indices differ, so
    # `transform_corner` must place the balance octant at the corner the caller resolved from
    # the connection — re-deriving it from `vertex_vertex_neighbor[..][1]` picks an arbitrary
    # incident tree and plants refinement at a wrong (far-away) corner. See the report in
    # PR #1349: spurious refinement clusters one macro cell away from the refined notch tip.
    nodes = [Node(Vec((0.0, 0.0)))]  # center
    nquads = 5
    for i in 0:(nquads - 1)
        θ1 = 2π * i / nquads
        θm = 2π * (i + 0.5) / nquads
        push!(nodes, Node(Vec((cos(θ1), sin(θ1)))))          # ring node shared by quads i-1, i
        push!(nodes, Node(Vec(1.3 .* (cos(θm), sin(θm)))))   # outer kite node of quad i
    end
    cells = map(0:(nquads - 1)) do i
        t = (1, 2 + 2i, 3 + 2i, i == nquads - 1 ? 2 : 4 + 2i)
        r = i % 4 # cyclic rotation keeps orientation but moves the center's local index
        return Quadrilateral(ntuple(j -> t[mod1(j + r, 4)], 4))
    end
    forest = ForestBWG(Grid(cells, nodes), 6)
    # refine all trees toward the central vertex three times, balancing in between
    for _ in 1:3
        g = Ferrite.AMR.creategrid(forest)
        marked = [c for c in 1:getncells(g) if any(n -> norm(n) < 1.0e-12, getcoordinates(g, c))]
        Ferrite.AMR.refine!(forest, marked)
        Ferrite.AMR.balanceforest!(forest)
    end
    for (k, tree) in enumerate(forest.cells)
        b = tree.b
        c_bwg = Ferrite.AMR.node_map₂_inv[findfirst(==(1), cells[k].nodes)]
        vc = Ferrite.AMR.vertex(Ferrite.AMR.root(2), c_bwg, b)
        maxlvl_at_corner = 0
        for leaf in tree.leaves
            h = Ferrite.AMR._compute_size(b, leaf.l)
            # Chebyshev distance from the leaf's box to the tree corner at the central vertex
            cheb = maximum(d -> max(leaf.xyz[d] - vc[d], vc[d] - (leaf.xyz[d] + h), 0), 1:2)
            cheb == 0 && (maxlvl_at_corner = max(maxlvl_at_corner, Int(leaf.l)))
            # deep leaves may only appear in the graded halo around the central vertex
            leaf.l >= 2 && @test cheb <= 2h
        end
        # ... and the 2:1 balance against the level-3 corner leaves must actually hold
        @test maxlvl_at_corner >= 2
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
        Ferrite.AMR.refine!(forest, marked)
        Ferrite.AMR.balanceforest!(forest)
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
    Ferrite.AMR.refine_all!(forest, 1)

    # getcells collects the leaves of all trees in cell id order (tree by tree, Morton
    # order within each tree); the scalar getcells(forest, cellid) deliberately throws
    # instead of hitting the generic fallback, which would return a whole tree
    @test_throws ArgumentError getcells(forest, 7)
    @test_throws ArgumentError getcells(forest, [1, 2])

    # marking a cell already at the maximum level is a documented no-op — also when the
    # tree's *first* leaf is the max-level one (used to throw from the 2^dim size hint)
    let f = ForestBWG(generate_grid(Quadrilateral, (2, 2)), 2)
        Ferrite.AMR.refine!(f, [1])
        Ferrite.AMR.balanceforest!(f)
        Ferrite.AMR.refine!(f, [1])
        Ferrite.AMR.balanceforest!(f)
        n = getncells(f)
        Ferrite.AMR.refine!(f, [1])            # cell 1 = first leaf of tree 1, at max level
        @test getncells(f) == n
        Ferrite.AMR.refine_and_coarsen!(f, [1], Int[]) # same guard in the fused path
        @test getncells(f) == n
        g = Ferrite.AMR.creategrid(f)
        @test getncells(g) == n
    end
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
    # a two-tree forest refined once has 45 (3D) / 15 (2D) nodes; a b past the limit used
    # to inflate these because the shared face nodes failed to merge
    let f = ForestBWG(generate_grid(Hexahedron, (2, 1, 1)), 19)
        Ferrite.AMR.refine_all!(f, 1)
        @test getnnodes(Ferrite.AMR.creategrid(f)) == 45
    end

    # NOTE: getcelltype currently exposes the octree (tree) type; this will change
    @test getcelltype(forest) === eltype(forest.cells) === getcelltype(forest, 1)
    @test getcelltype(forest) <: Ferrite.AMR.OctreeBWG

    # getneighborhood forwards to the macro topology
    top = ExclusiveTopology(grid)
    @test Ferrite.getneighborhood(forest, FacetIndex(1, 2)) == Ferrite.getneighborhood(top, grid, FacetIndex(1, 2))

    # integer-type-promoting convenience methods
    o2 = Ferrite.AMR.OctantBWG(1, (0, 2))
    o3 = Ferrite.AMR.OctantBWG(1, (0, 2, 4))
    #TODO: for gpu probably should change to propagate the type
    @test Ferrite.AMR.morton(o2, Int32(1), Int32(3)) == Ferrite.AMR.morton(o2, 1, 3)
    @test Ferrite.AMR.facet_neighbor(o2, Int32(1), Int32(3)) == Ferrite.AMR.facet_neighbor(o2, 1, 3)
    @test Ferrite.AMR.corner_neighbor(o2, Int32(1), Int32(3)) == Ferrite.AMR.corner_neighbor(o2, 1, 3)
    @test Ferrite.AMR.edge_neighbor(o3, Int32(1), Int32(3)) == Ferrite.AMR.edge_neighbor(o3, 1, 3)

    # face -> corner lookup for both dimensions, and the error path
    @test Ferrite.AMR._face_corners(2, 1) == Ferrite.AMR.𝒱₂[1, :]
    @test Ferrite.AMR._face_corners(3, 1) == Ferrite.AMR.𝒱₃[1, :]
    @test_throws ErrorException Ferrite.AMR._face_corners(4, 1)

    # octant level beyond the tree's maximum refinement level b
    @test_throws DomainError Ferrite.AMR._compute_size(2, 3)
end

@testset "refine!/coarsen! edge branches" begin
    # scalar refine! of a cell in a later tree walks the per-tree leaf counts
    forest = ForestBWG(generate_grid(Quadrilateral, (2, 2)), 3)
    Ferrite.AMR.refine!(forest, 3)
    @test getncells(forest) == 7
    @test length(forest.cells[3].leaves) == 4
    @test length(forest.cells[1].leaves) == 1

    # coarsen! from a non-first sibling snaps back to the family's first sibling
    forest = ForestBWG(generate_grid(Quadrilateral, (1, 1)), 3)
    Ferrite.AMR.refine_all!(forest, 1)
    tree = forest.cells[1]
    @test length(tree.leaves) == 4
    Ferrite.AMR.coarsen_octant!(tree, tree.leaves[2])
    @test length(tree.leaves) == 1
    @test tree.leaves[1].l == 0

    # refine_all! on mixed levels keeps the leaves that are not at level l-1
    forest = ForestBWG(generate_grid(Quadrilateral, (1, 1)), 3)
    Ferrite.AMR.refine!(forest, 1)
    Ferrite.AMR.refine!(forest, 1)
    @test getncells(forest) == 7 # 4 at level 2, 3 at level 1
    Ferrite.AMR.refine_all!(forest, 3) # only the level-2 leaves refine
    @test getncells(forest) == 19 # 16 at level 3, 3 at level 1
end

@testset "batch coarsen!(forest, cellids) $dim D" for (dim, CT) in ((2, Quadrilateral), (3, Hexahedron))
    nchild = 2^dim

    # batch coarsen! inverts batch refine!: refining cell 1 then coarsening its children
    # (ids 1:nchild) recovers the original forest octant-for-octant
    forest = ForestBWG(generate_grid(CT, ntuple(_ -> 2, dim)), 4)
    Ferrite.AMR.refine_all!(forest, 1)
    base = getcells(forest)
    Ferrite.AMR.refine!(forest, [1])
    @test getncells(forest) == length(base) + (nchild - 1)
    Ferrite.AMR.coarsen!(forest, collect(1:nchild))
    @test getcells(forest) == base
    for tree in forest.cells
        @test issorted(tree.leaves)
    end

    # on a uniformly refined forest, coarsening every cell with require_all_siblings
    # reproduces _coarsen_all!
    f1 = ForestBWG(generate_grid(CT, ntuple(_ -> 1, dim)), 4)
    Ferrite.AMR.refine_all!(f1, 1)
    Ferrite.AMR.refine_all!(f1, 2)
    f2 = deepcopy(f1)
    Ferrite.AMR._coarsen_all!(f1)
    Ferrite.AMR.coarsen!(f2, collect(1:getncells(f2)); require_all_siblings = true)
    @test getcells(f1) == getcells(f2)

    # policy modularity: one marked sibling is a no-op under all-siblings, collapses the
    # family under any-sibling
    f = ForestBWG(generate_grid(CT, ntuple(_ -> 1, dim)), 4)
    Ferrite.AMR.refine_all!(f, 1)
    n0 = getncells(f)
    fa = deepcopy(f)
    Ferrite.AMR.coarsen!(fa, [1]; require_all_siblings = true)
    @test getncells(fa) == n0
    fb = deepcopy(f)
    Ferrite.AMR.coarsen!(fb, [1]; require_all_siblings = false)
    @test getncells(fb) == n0 - (nchild - 1)

    # incomplete family (one child refined further) cannot be coarsened -> silently skipped
    fi = ForestBWG(generate_grid(CT, ntuple(_ -> 1, dim)), 4)
    Ferrite.AMR.refine_all!(fi, 1)
    Ferrite.AMR.refine!(fi, [1]) # child 1 -> nchild grandchildren; family now incomplete
    n1 = getncells(fi)
    Ferrite.AMR.coarsen!(fi, collect((nchild + 1):n1); require_all_siblings = false) # the surviving level-1 leaves
    @test getncells(fi) == n1

    # the caller's id vector is not mutated
    ids = collect(nchild:-1:1)
    fc = ForestBWG(generate_grid(CT, ntuple(_ -> 1, dim)), 4)
    Ferrite.AMR.refine_all!(fc, 1)
    Ferrite.AMR.coarsen!(fc, ids)
    @test ids == collect(nchild:-1:1)

    # empty ids is a no-op
    fe = ForestBWG(generate_grid(CT, ntuple(_ -> 1, dim)), 4)
    Ferrite.AMR.refine_all!(fe, 1)
    m = getncells(fe)
    Ferrite.AMR.coarsen!(fe, Int[])
    @test getncells(fe) == m
end

@testset "refine_and_coarsen! $dim D" for (dim, CT) in ((2, Quadrilateral), (3, Hexahedron))
    nchild = 2^dim
    ntree = 2^dim # (2,2[,2]) grid: one tree per octant

    # fused single pass, no balancing: coarsen tree 1's family (ids 1:nchild) while refining
    # the first leaf of tree 3 (global id 2*nchild + 1). Both ids resolve against the
    # ORIGINAL numbering, so the coarsen is not thrown off by the refine.
    forest = ForestBWG(generate_grid(CT, ntuple(_ -> 2, dim)), 4)
    Ferrite.AMR.refine_all!(forest, 1)
    n0 = getncells(forest)
    refid = 2 * nchild + 1
    Ferrite.AMR.refine_and_coarsen!(forest, collect(1:nchild), [refid]; balance = false)
    @test getncells(forest) == n0 - (nchild - 1) + (nchild - 1)
    @test length(forest.cells[1].leaves) == 1              # tree 1 coarsened to root
    @test length(forest.cells[3].leaves) == 2 * nchild - 1 # tree 3 leaf 1 refined
    for tree in forest.cells
        @test issorted(tree.leaves)
    end

    # with balancing (default), the coarsened tree 1 sits next to tree 3's finer cells and is
    # re-refined to restore 2:1; the result is a valid grid creategrid accepts
    balanced = ForestBWG(generate_grid(CT, ntuple(_ -> 2, dim)), 4)
    Ferrite.AMR.refine_all!(balanced, 1)
    Ferrite.AMR.refine_and_coarsen!(balanced, collect(1:nchild), [refid])
    @test Ferrite.AMR.creategrid(balanced) isa Ferrite.AMR.NonConformingGrid

    # roundtrip: refine cell 1, then coarsen its children back to the original forest
    h = ForestBWG(generate_grid(CT, ntuple(_ -> 2, dim)), 4)
    base = getcells(h)
    Ferrite.AMR.refine_and_coarsen!(h, Int[], [1]; balance = false)
    @test getncells(h) == length(base) + (nchild - 1)
    Ferrite.AMR.refine_and_coarsen!(h, collect(1:nchild), Int[]; balance = false)
    @test getcells(h) == base

    # conflict: overlapping refine/coarsen ids
    c = ForestBWG(generate_grid(CT, ntuple(_ -> 2, dim)), 4)
    Ferrite.AMR.refine_all!(c, 1)
    @test_throws ArgumentError Ferrite.AMR.refine_and_coarsen!(c, [1, 2], [2, nchild + 1])

    # conflict: a refine id inside a family being coarsened (reachable with any-sibling policy)
    c2 = ForestBWG(generate_grid(CT, ntuple(_ -> 2, dim)), 4)
    Ferrite.AMR.refine_all!(c2, 1)
    @test_throws ArgumentError Ferrite.AMR.refine_and_coarsen!(c2, [1], [3]; require_all_siblings = false)
end
