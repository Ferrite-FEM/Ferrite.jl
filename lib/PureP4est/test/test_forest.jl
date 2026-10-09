# The forest data structure: inter-tree transforms, refinement/coarsening and 2:1
# balancing. Mirrors the forest half of `src/forest.jl`.
using PureP4est, Test

include(joinpath(@__DIR__, "utils.jl"))

@testset "Connectivity input validation" begin
    # degenerate cells (repeated vertex ids) would give non-reciprocal neighbour relations
    @test_throws ArgumentError Connectivity([(1, 2, 3, 4), (1, 1, 2, 5)])
    @test_throws ArgumentError Forest([(1, 2, 3, 4), (1, 1, 2, 5)])
    @test_throws ArgumentError Connectivity([(0, 1, 2, 3)])
    @test_throws ArgumentError Connectivity([(1, 2, 3)])
    # neighbour relations are reciprocal
    conn = Connectivity(brick((3, 2, 2)))
    for t in (conn.facet_neighbors, conn.edge_neighbors, conn.vertex_neighbors), k in 1:size(t, 1), i in 1:size(t, 2), (k′, i′) in t[k, i]
        @test (k, i) in t[k′, i′]
    end
end

@testset "OctreeBWG Operations" begin
    # maximum level == 3
    # Octant level 0 size == 2^3=8
    # Octant level 1 size == 2^3/2 = 4
    # Octant level 2 size == 2^3/4 = 2
    # Octant level 3 size == 2^3/8 = 1
    # test translation constructor
    cells = brick((2, 2))
    # Rotate face topologically
    cells[2] = cells[2][[2, 3, 4, 1]]
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
    forest = Forest(cells, 3)
    for tree in trees(forest)
        @test tree isa PureP4est.OctreeBWG
        @test tree.leaves[1] == PureP4est.OctantBWG(2, 0, 1, tree.b)
    end
    @test PureP4est.transform_facet_remote(forest, 2, 4, forest.trees[1].leaves[1]) == PureP4est.OctantBWG(0, (8, 0))
    @test PureP4est.transform_facet_remote(forest, 1, 2, forest.trees[1].leaves[1]) == PureP4est.OctantBWG(0, (0, 8))
    @test PureP4est.transform_facet_remote(forest, 4, 1, forest.trees[3].leaves[1]) == PureP4est.OctantBWG(0, (8, 0))
    @test PureP4est.transform_facet_remote(forest, 3, 2, forest.trees[4].leaves[1]) == PureP4est.OctantBWG(0, (-8, 0))
    @test PureP4est.transform_facet_remote(forest, 3, 3, forest.trees[1].leaves[1]) == PureP4est.OctantBWG(0, (0, 8))
    @test PureP4est.transform_facet_remote(forest, 1, 4, forest.trees[3].leaves[1]) == PureP4est.OctantBWG(0, (0, -8))
    @test PureP4est.transform_facet_remote(forest, 4, 3, forest.trees[2].leaves[1]) == PureP4est.OctantBWG(0, (8, 0))
    @test PureP4est.transform_facet_remote(forest, 2, 2, forest.trees[4].leaves[1]) == PureP4est.OctantBWG(0, (0, -8))
    o = forest.trees[1].leaves[1]
    @test PureP4est.transform_facet(forest, 1, 2, o) == PureP4est.OctantBWG(0, (8, 0))
    @test PureP4est.transform_facet(forest, 1, 4, o) == PureP4est.OctantBWG(0, (0, 8))
    @test PureP4est.transform_facet(forest, 2, 4, o) == PureP4est.OctantBWG(0, (0, 8))
    @test PureP4est.transform_facet(forest, 2, 2, o) == PureP4est.OctantBWG(0, (8, 0))
    @test PureP4est.transform_facet(forest, 3, 2, o) == PureP4est.OctantBWG(0, (8, 0))
    @test PureP4est.transform_facet(forest, 3, 3, o) == PureP4est.OctantBWG(0, (0, -8))
    @test PureP4est.transform_facet(forest, 4, 1, o) == PureP4est.OctantBWG(0, (-8, 0))
    @test PureP4est.transform_facet(forest, 4, 3, o) == PureP4est.OctantBWG(0, (0, -8))

    ln = lnodes(forest)
    @test nnodes(ln) == 9
    @test nhanging(ln) == 0

    cells[4] = cells[4][[2, 3, 4, 1]]
    cells[4] = cells[4][[2, 3, 4, 1]]
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
    forest = Forest(cells, 3)
    for tree in trees(forest)
        @test tree isa PureP4est.OctreeBWG
        @test tree.leaves[1] == PureP4est.OctantBWG(2, 0, 1, tree.b)
    end
    @test PureP4est.transform_facet_remote(forest, 2, 4, forest.trees[1].leaves[1]) == PureP4est.OctantBWG(0, (8, 0))
    @test PureP4est.transform_facet_remote(forest, 1, 2, forest.trees[1].leaves[1]) == PureP4est.OctantBWG(0, (0, 8))
    @test PureP4est.transform_facet_remote(forest, 4, 2, forest.trees[3].leaves[1]) == PureP4est.OctantBWG(0, (8, 0))
    @test PureP4est.transform_facet_remote(forest, 3, 2, forest.trees[4].leaves[1]) == PureP4est.OctantBWG(0, (8, 0))
    @test PureP4est.transform_facet_remote(forest, 3, 3, forest.trees[1].leaves[1]) == PureP4est.OctantBWG(0, (0, 8))
    @test PureP4est.transform_facet_remote(forest, 1, 4, forest.trees[3].leaves[1]) == PureP4est.OctantBWG(0, (0, -8))
    @test PureP4est.transform_facet_remote(forest, 4, 4, forest.trees[2].leaves[1]) == PureP4est.OctantBWG(0, (8, 0))
    @test PureP4est.transform_facet_remote(forest, 2, 2, forest.trees[4].leaves[1]) == PureP4est.OctantBWG(0, (0, 8))

    @test PureP4est.transform_corner(forest, 4, 4, forest.trees[1].leaves[1], false) == PureP4est.transform_corner_remote(forest, 1, 4, forest.trees[1].leaves[1], false) == PureP4est.OctantBWG(0, (8, 8))
    @test PureP4est.transform_corner(forest, 4, 4, forest.trees[1].leaves[1], false) == PureP4est.transform_corner(forest, 1, 4, forest.trees[1].leaves[1], false) == PureP4est.OctantBWG(0, (8, 8))

    o = forest.trees[1].leaves[1]
    @test PureP4est.transform_facet(forest, 1, 2, o) == PureP4est.OctantBWG(0, (8, 0))
    @test PureP4est.transform_facet(forest, 1, 4, o) == PureP4est.OctantBWG(0, (0, 8))
    @test PureP4est.transform_facet(forest, 2, 4, o) == PureP4est.OctantBWG(0, (0, 8))
    @test PureP4est.transform_facet(forest, 2, 2, o) == PureP4est.OctantBWG(0, (8, 0))
    @test PureP4est.transform_facet(forest, 3, 2, o) == PureP4est.OctantBWG(0, (8, 0))
    @test PureP4est.transform_facet(forest, 3, 3, o) == PureP4est.OctantBWG(0, (0, -8))
    @test PureP4est.transform_facet(forest, 4, 2, o) == PureP4est.OctantBWG(0, (8, 0))
    @test PureP4est.transform_facet(forest, 4, 4, o) == PureP4est.OctantBWG(0, (0, 8))


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
    PureP4est.refine_octant!(forest.trees[1], forest.trees[1].leaves[1])
    @test length(forest.trees[1].leaves) == 4
    for (m, octant) in zip(1:4, forest.trees[1].leaves)
        @test octant == PureP4est.OctantBWG(2, 1, m, forest.trees[1].b)
    end
    PureP4est.refine_octant!(forest.trees[1], forest.trees[1].leaves[1])

    @test PureP4est.transform_facet(forest, 2, 4, forest.trees[1].leaves[5]) == PureP4est.OctantBWG(1, (0, 8))
    @test PureP4est.transform_facet(forest, 2, 4, forest.trees[1].leaves[7]) == PureP4est.OctantBWG(1, (4, 8))
    @test PureP4est.transform_facet(forest, 3, 3, forest.trees[1].leaves[6]) == PureP4est.OctantBWG(1, (0, -4))
    @test PureP4est.transform_facet(forest, 3, 3, forest.trees[1].leaves[7]) == PureP4est.OctantBWG(1, (4, -4))

    ln = lnodes(forest)
    @test nnodes(ln) == 19
    @test nhanging(ln) == 4

    # octree holds now 3 first level and 4 second level
    @test length(forest.trees[1].leaves) == 7
    for (m, octant) in zip(1:4, forest.trees[1].leaves)
        @test octant == PureP4est.OctantBWG(2, 2, m, forest.trees[1].b)
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
    forest = Forest(cells, 3)
    PureP4est.refine_octant!(forest.trees[1], forest.trees[1].leaves[1])
    PureP4est.refine_octant!(forest.trees[1], forest.trees[1].leaves[4])
    @test length(forest.trees[1].leaves) == 7
    @test all(getproperty.(forest.trees[1].leaves[1:3], :l) .== 1)

    @test PureP4est.transform_facet(forest, 2, 4, forest.trees[1].leaves[2]) == PureP4est.OctantBWG(1, (0, 8))
    @test PureP4est.transform_facet(forest, 2, 4, forest.trees[1].leaves[5]) == PureP4est.OctantBWG(2, (4, 8))
    @test PureP4est.transform_facet(forest, 2, 4, forest.trees[1].leaves[7]) == PureP4est.OctantBWG(2, (6, 8))
    @test PureP4est.transform_facet(forest, 3, 3, forest.trees[1].leaves[3]) == PureP4est.OctantBWG(1, (0, -4))
    @test PureP4est.transform_facet(forest, 3, 3, forest.trees[1].leaves[6]) == PureP4est.OctantBWG(2, (4, -2))
    @test PureP4est.transform_facet(forest, 3, 3, forest.trees[1].leaves[7]) == PureP4est.OctantBWG(2, (6, -2))

    ln = lnodes(forest)
    @test nnodes(ln) == 19
    @test nhanging(ln) == 4

    # more complex neighborhoods
    cells = disc(6)
    cells[2] = cells[2][[2, 3, 4, 1]]
    forest = Forest(cells, 3)
    PureP4est.refine_octant!(forest.trees[3], forest.trees[3].leaves[1])
    PureP4est.refine_octant!(forest.trees[5], forest.trees[5].leaves[1])

    ln = lnodes(forest)
    @test nnodes(ln) == 23
    @test nhanging(ln) == 4

    ##################################################################
    ####uniform refinement and coarsening for all cells and levels####
    ##################################################################
    forest = Forest(cells, 8)
    for l in 1:8
        PureP4est.refine_all!(forest, l)
        for tree in forest.trees
            @test all(PureP4est.morton.(tree.leaves, l, 8) == collect(1:(2^(2 * l))))
        end
    end
    #check montonicity of ancestor_id
    for tree in forest.trees
        ids = PureP4est.ancestor_id.(tree.leaves, (1,), (tree.b,))
        @test issorted(ids)
    end
    #now go back from finest to coarsest
    for l in 7:-1:0
        PureP4est._coarsen_all!(forest)
        for tree in forest.trees
            @test all(PureP4est.morton.(tree.leaves, l, 8) == collect(1:(2^(2 * l))))
        end
    end
    #########################
    # now do the same with 3D
    #########################
    cells = brick((2, 2, 2))
    forest = Forest(cells, 3)
    o = forest.trees[1].leaves[1]

    # faces
    @test PureP4est.transform_facet(forest, 1, 2, o) == PureP4est.OctantBWG(0, (8, 0, 0))
    @test PureP4est.transform_facet_remote(forest, 1, 2, o) == PureP4est.OctantBWG(0, (-8, 0, 0))
    @test PureP4est.transform_facet(forest, 1, 4, o) == PureP4est.OctantBWG(0, (0, 8, 0))
    @test PureP4est.transform_facet_remote(forest, 1, 4, o) == PureP4est.OctantBWG(0, (0, -8, 0))
    @test PureP4est.transform_facet(forest, 1, 6, o) == PureP4est.OctantBWG(0, (0, 0, 8))
    @test PureP4est.transform_facet_remote(forest, 1, 6, o) == PureP4est.OctantBWG(0, (0, 0, -8))
    @test PureP4est.transform_facet(forest, 8, 1, o) == PureP4est.OctantBWG(0, (-8, 0, 0))
    @test PureP4est.transform_facet_remote(forest, 8, 1, o) == PureP4est.OctantBWG(0, (8, 0, 0))
    @test PureP4est.transform_facet(forest, 8, 3, o) == PureP4est.OctantBWG(0, (0, -8, 0))
    @test PureP4est.transform_facet_remote(forest, 8, 3, o) == PureP4est.OctantBWG(0, (0, 8, 0))
    @test PureP4est.transform_facet(forest, 8, 5, o) == PureP4est.OctantBWG(0, (0, 0, -8))
    @test PureP4est.transform_facet_remote(forest, 8, 5, o) == PureP4est.OctantBWG(0, (0, 0, 8))

    @test_throws BoundsError PureP4est.transform_facet(forest, 1, 1, o)
    @test_throws BoundsError PureP4est.transform_facet_remote(forest, 1, 1, o)
    @test_throws BoundsError PureP4est.transform_facet(forest, 1, 3, o)
    @test_throws BoundsError PureP4est.transform_facet_remote(forest, 1, 3, o)
    @test_throws BoundsError PureP4est.transform_facet(forest, 1, 5, o)
    @test_throws BoundsError PureP4est.transform_facet_remote(forest, 1, 5, o)
    @test_throws BoundsError PureP4est.transform_facet(forest, 8, 2, o)
    @test_throws BoundsError PureP4est.transform_facet_remote(forest, 8, 2, o)
    @test_throws BoundsError PureP4est.transform_facet(forest, 8, 4, o)
    @test_throws BoundsError PureP4est.transform_facet_remote(forest, 8, 4, o)
    @test_throws BoundsError PureP4est.transform_facet(forest, 8, 6, o)
    @test_throws BoundsError PureP4est.transform_facet_remote(forest, 8, 6, o)

    #corners
    @test PureP4est.transform_corner(forest, 1, 1, o, false) == PureP4est.OctantBWG(0, (-8, -8, -8))
    @test PureP4est.transform_corner(forest, 1, 2, o, false) == PureP4est.OctantBWG(0, (8, -8, -8))
    @test PureP4est.transform_corner(forest, 1, 3, o, false) == PureP4est.OctantBWG(0, (-8, 8, -8))
    @test PureP4est.transform_corner(forest, 1, 4, o, false) == PureP4est.OctantBWG(0, (8, 8, -8))
    @test PureP4est.transform_corner(forest, 1, 5, o, false) == PureP4est.OctantBWG(0, (-8, -8, 8))
    @test PureP4est.transform_corner(forest, 1, 6, o, false) == PureP4est.OctantBWG(0, (8, -8, 8))
    @test PureP4est.transform_corner(forest, 1, 7, o, false) == PureP4est.OctantBWG(0, (-8, 8, 8))
    @test PureP4est.transform_corner(forest, 1, 8, o, false) == PureP4est.OctantBWG(0, (8, 8, 8))
    @test_throws BoundsError PureP4est.transform_corner_remote(forest, 1, 1, o, false)
    @test_throws BoundsError PureP4est.transform_corner_remote(forest, 1, 2, o, false)
    @test_throws BoundsError PureP4est.transform_corner_remote(forest, 1, 3, o, false)
    @test_throws BoundsError PureP4est.transform_corner_remote(forest, 1, 4, o, false)
    @test_throws BoundsError PureP4est.transform_corner_remote(forest, 1, 5, o, false)
    @test_throws BoundsError PureP4est.transform_corner_remote(forest, 1, 6, o, false)
    @test_throws BoundsError PureP4est.transform_corner_remote(forest, 1, 7, o, false)
    @test PureP4est.transform_corner_remote(forest, 1, 8, o, false) == PureP4est.OctantBWG(0, (-8, -8, -8))

    #edges
    @test_throws BoundsError PureP4est.transform_edge(forest, 1, 1, o, false)
    @test_throws BoundsError PureP4est.transform_edge(forest, 1, 2, o, false)
    @test_throws BoundsError PureP4est.transform_edge(forest, 1, 3, o, false)
    @test_throws BoundsError PureP4est.transform_edge_remote(forest, 1, 1, o, false)
    @test_throws BoundsError PureP4est.transform_edge_remote(forest, 1, 2, o, false)
    @test_throws BoundsError PureP4est.transform_edge_remote(forest, 1, 3, o, false)
    @test PureP4est.transform_edge(forest, 1, 4, o, false) == PureP4est.OctantBWG(0, (0, 8, 8))
    @test PureP4est.transform_edge_remote(forest, 1, 4, o, false) == PureP4est.OctantBWG(0, (0, -8, -8))
    @test_throws BoundsError PureP4est.transform_edge(forest, 1, 5, o, false)
    @test_throws BoundsError PureP4est.transform_edge(forest, 1, 6, o, false)
    @test_throws BoundsError PureP4est.transform_edge(forest, 1, 7, o, false)
    @test_throws BoundsError PureP4est.transform_edge_remote(forest, 1, 5, o, false)
    @test_throws BoundsError PureP4est.transform_edge_remote(forest, 1, 6, o, false)
    @test_throws BoundsError PureP4est.transform_edge_remote(forest, 1, 7, o, false)
    @test PureP4est.transform_edge(forest, 1, 8, o, false) == PureP4est.OctantBWG(0, (8, 0, 8))
    @test PureP4est.transform_edge_remote(forest, 1, 8, o, false) == PureP4est.OctantBWG(0, (-8, 0, -8))
    @test_throws BoundsError PureP4est.transform_edge(forest, 1, 9, o, false)
    @test_throws BoundsError PureP4est.transform_edge(forest, 1, 10, o, false)
    @test_throws BoundsError PureP4est.transform_edge(forest, 1, 11, o, false)
    @test_throws BoundsError PureP4est.transform_edge_remote(forest, 1, 9, o, false)
    @test_throws BoundsError PureP4est.transform_edge_remote(forest, 1, 10, o, false)
    @test_throws BoundsError PureP4est.transform_edge_remote(forest, 1, 11, o, false)
    @test PureP4est.transform_edge(forest, 1, 12, o, false) == PureP4est.OctantBWG(0, (8, 8, 0))
    @test PureP4est.transform_edge_remote(forest, 1, 12, o, false) == PureP4est.OctantBWG(0, (-8, -8, 0))

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
    cells = brick((2, 2, 2))
    # Rotate face topologically as decscribed in the ascii picture above
    cells[7] = cells[7][[2, 3, 4, 1, 6, 7, 8, 5]]
    cells[7] = cells[7][[2, 3, 4, 1, 6, 7, 8, 5]]
    forest = Forest(cells, 3)
    @test PureP4est.transform_corner(forest, 7, 3, PureP4est.OctantBWG(0, (0, 0, 0)), false) == PureP4est.OctantBWG(0, (-8, 8, -8))

    #refinement
    PureP4est.refine_octant!(forest.trees[1], forest.trees[1].leaves[1])
    @test length(forest.trees[1].leaves) == 8
    for (m, octant) in zip(1:8, forest.trees[1].leaves)
        @test octant == PureP4est.OctantBWG(3, 1, m, forest.trees[1].b)
    end
    PureP4est.refine_octant!(forest.trees[1], forest.trees[1].leaves[1])
    @test length(forest.trees[1].leaves) == 15
    for (m, octant) in zip(1:8, forest.trees[1].leaves)
        @test octant == PureP4est.OctantBWG(3, 2, m, forest.trees[1].b)
    end
    forest = Forest(cells, 3)
    PureP4est.refine_octant!(forest.trees[1], forest.trees[1].leaves[1])
    PureP4est.refine_octant!(forest.trees[1], forest.trees[1].leaves[4])
    @test length(forest.trees[1].leaves) == 15
    @test all(getproperty.(forest.trees[1].leaves[1:3], :l) .== 1)
    @test all(getproperty.(forest.trees[1].leaves[4:11], :l) .== 2)
    @test all(getproperty.(forest.trees[1].leaves[12:end], :l) .== 1)
    forest = Forest(cells, 5)
    #go from coarsest to finest uniformly
    for l in 1:5
        PureP4est.refine_all!(forest, l)
        for tree in forest.trees
            @test all(PureP4est.morton.(tree.leaves, l, 5) == collect(1:(2^(3 * l))))
        end
    end
    #now go back from finest to coarsest
    for l in 4:-1:0
        PureP4est._coarsen_all!(forest)
        for tree in forest.trees
            @test all(PureP4est.morton.(tree.leaves, l, 5) == collect(1:(2^(3 * l))))
        end
    end

    # Single
    forest = Forest(cells, 3)
    PureP4est.refine_all!(forest, 1)
    PureP4est.refine_octant!(forest.trees[1], forest.trees[1].leaves[8])
    ln = lnodes(forest)
    # Unrefined grid has 5 ^ dim nodes and the refined element introduces 6 face center, 12 edge center and 1 volume center nodes
    @test nnodes(ln) == 5^3 + (6 + 12 + 1)
    # 6 faces and 12 edges of the single refined element induces one hanging node each
    @test nhanging(ln) == 6 + 12

    forest = Forest(cells, 3)
    PureP4est.refine_all!(forest, 1)
    PureP4est.refine_octant!(forest.trees[1], forest.trees[1].leaves[1])
    ln = lnodes(forest)
    # Unrefined grid has 5 ^ dim nodes and the refined element introduces 6 face center, 12 edge center and 1 volume center nodes
    @test nnodes(ln) == 5^3 + (6 + 12 + 1)
    # 6 faces and 12 edges of the single refined element induces one hanging node each - minus 3 faces and 3 edges on the outer boundary
    @test nhanging(ln) == 6 + 12 - 2 * 3

    forest = Forest(cells, 3)
    PureP4est.refine_all!(forest, 1)
    PureP4est.refine_octant!(forest.trees[8], forest.trees[8].leaves[8])
    ln = lnodes(forest)
    # Unrefined grid has 5 ^ dim nodes and the refined element introduces 6 face center, 12 edge center and 1 volume center nodes
    @test nnodes(ln) == 5^3 + (6 + 12 + 1)
    # 6 faces and 12 edges of the single refined element induces one hanging node each - minus 3 faces and 3 edges on the outer boundary
    @test nhanging(ln) == 6 + 12 - 2 * 3

    forest = Forest(cells, 3)
    PureP4est.refine_all!(forest, 1)
    PureP4est.refine_octant!(forest.trees[8], forest.trees[8].leaves[1])
    ln = lnodes(forest)
    # Unrefined grid has 5 ^ dim nodes and the refined element introduces 6 face center, 12 edge center and 1 volume center nodes
    @test nnodes(ln) == 5^3 + (6 + 12 + 1)
    # 6 faces and 12 edges of the single refined element induces one hanging node each
    @test nhanging(ln) == 6 + 12

    # Combined
    forest = Forest(cells, 3)
    PureP4est.refine_all!(forest, 1)
    PureP4est.refine_octant!(forest.trees[1], forest.trees[1].leaves[8])
    PureP4est.refine_octant!(forest.trees[1], forest.trees[1].leaves[1])
    ln = lnodes(forest)
    @test nnodes(ln) == 5^3 + 2 * (6 + 12 + 1)
    @test nhanging(ln) == 2 * (6 + 12) - 2 * 3

    forest = Forest(cells, 3)
    PureP4est.refine_all!(forest, 1)
    PureP4est.refine_octant!(forest.trees[8], forest.trees[8].leaves[8])
    PureP4est.refine_octant!(forest.trees[8], forest.trees[8].leaves[1])
    ln = lnodes(forest)
    @test nnodes(ln) == 5^3 + 2 * (6 + 12 + 1)
    @test nhanging(ln) == 2 * (6 + 12) - 2 * 3

    # Combined
    forest = Forest(cells, 3)
    PureP4est.refine_all!(forest, 1)
    PureP4est.refine_octant!(forest.trees[1], forest.trees[1].leaves[8])
    PureP4est.refine_octant!(forest.trees[1], forest.trees[1].leaves[1])
    PureP4est.refine_octant!(forest.trees[8], forest.trees[8].leaves[8])
    PureP4est.refine_octant!(forest.trees[8], forest.trees[8].leaves[1])
    ln = lnodes(forest)
    @test nnodes(ln) == 5^3 + 4 * (6 + 12 + 1)
    @test nhanging(ln) == 4 * (6 + 12) - 2 * 3 - 2 * 3

    # Combined and not rotated
    forest = Forest(cells, 3)
    PureP4est.refine_all!(forest, 1)
    PureP4est.refine_octant!(forest.trees[1], forest.trees[1].leaves[8])
    PureP4est.refine_octant!(forest.trees[1], forest.trees[1].leaves[1])
    PureP4est.refine_octant!(forest.trees[6], forest.trees[6].leaves[6])
    PureP4est.refine_octant!(forest.trees[6], forest.trees[6].leaves[3])
    ln = lnodes(forest)
    # +5^3 on the coarse grid
    # +4 refined elements a 6 face nodes, 12 edge nodes and 1 volume nodes
    # -1 shared node between tree 1 and 6
    @test nnodes(ln) == 5^3 + 4 * (6 + 12 + 1) - 1
    # 30 constraints from tree 1 (2*18 - 6 boundary) + 30 from tree 6 (2*18 - 6 boundary)
    # - 1 shared on common edge
    @test nhanging(ln) == 59

    # Combined and rotated
    forest = Forest(cells, 3)
    PureP4est.refine_all!(forest, 1)
    PureP4est.refine_octant!(forest.trees[1], forest.trees[1].leaves[8])
    PureP4est.refine_octant!(forest.trees[1], forest.trees[1].leaves[1])
    PureP4est.refine_octant!(forest.trees[7], forest.trees[7].leaves[6])
    PureP4est.refine_octant!(forest.trees[7], forest.trees[7].leaves[3])
    ln = lnodes(forest)
    # +5^3 on the coarse grid
    # +4 refined elements a 6 face nodes, 12 edge nodes and 1 volume nodes
    # -1 shared node between tree 1 and 7
    @test nnodes(ln) == 5^3 + 4 * (6 + 12 + 1) - 1
    # 30 constraints from tree 1 + 30 from rotated tree 7 - 1 shared on common edge
    @test nhanging(ln) == 59

    # Reproducer test for Fig.3 BWG 11
    cells = brick((2, 1, 1))
    # (a)
    forest = Forest(cells, 3)
    PureP4est.refine_octant!(forest.trees[2], forest.trees[2].leaves[1])
    PureP4est.refine_octant!(forest.trees[2], forest.trees[2].leaves[3])
    @test forest.trees[2].leaves[3 + 4] == PureP4est.OctantBWG(2, (0, 4, 2))
    @test PureP4est.transform_facet(forest, 1, 2, forest.trees[2].leaves[3 + 4]) == PureP4est.OctantBWG(2, (8, 4, 2))
    # (b) Rotate elements topologically
    cells[1] = cells[1][[2, 3, 4, 1, 6, 7, 8, 5]]
    cells[2] = cells[2][[4, 1, 2, 3, 8, 5, 6, 7]]
    forest = Forest(cells, 3)
    PureP4est.refine_octant!(forest.trees[2], forest.trees[2].leaves[1])
    PureP4est.refine_octant!(forest.trees[2], forest.trees[2].leaves[1])
    @test forest.trees[2].leaves[6] == PureP4est.OctantBWG(2, (2, 0, 2))
    @test PureP4est.transform_facet(forest, 1, 3, forest.trees[2].leaves[6]) == PureP4est.OctantBWG(2, (4, -2, 2))
end

@testset "Balancing" begin
    #2D cases
    #simple one quad with one additional non-allowed non-conformity level
    cells = brick((1, 1))
    forest = Forest(cells, 3)
    PureP4est.refine_all!(forest, 1)
    PureP4est.refine_octant!(forest.trees[1], forest.trees[1].leaves[2])
    PureP4est.refine_octant!(forest.trees[1], forest.trees[1].leaves[6])
    PureP4est.refine_octant!(forest.trees[1], forest.trees[1].leaves[6])
    balanced = PureP4est.balancetree(forest.trees[1])
    @test length(balanced.leaves) == 16

    #more complex non-conformity level 3 and 4 that needs to be balanced
    forest = Forest(cells, 5)
    PureP4est.refine_all!(forest, 1)
    PureP4est.refine_octant!(forest.trees[1], forest.trees[1].leaves[2])
    PureP4est.refine_octant!(forest.trees[1], forest.trees[1].leaves[4])
    PureP4est.refine_octant!(forest.trees[1], forest.trees[1].leaves[7])
    PureP4est.refine_octant!(forest.trees[1], forest.trees[1].leaves[12])
    PureP4est.refine_octant!(forest.trees[1], forest.trees[1].leaves[12])
    PureP4est.refine_octant!(forest.trees[1], forest.trees[1].leaves[15])
    PureP4est.refine_octant!(forest.trees[1], forest.trees[1].leaves[16])
    balanced = PureP4est.balancetree(forest.trees[1])
    @test length(balanced.leaves) == 64

    cells = brick((2, 1))
    forest = Forest(cells, 2)
    PureP4est.refine_octant!(forest.trees[1], forest.trees[1].leaves[1])
    PureP4est.refine_octant!(forest.trees[1], forest.trees[1].leaves[2])
    PureP4est.balanceforest!(forest)
    @test nleaves(forest) == 11

    cells = brick((2, 2))
    forest = Forest(cells, 2)
    PureP4est.refine_octant!(forest.trees[1], forest.trees[1].leaves[1])
    PureP4est.refine_octant!(forest.trees[1], forest.trees[1].leaves[4])
    PureP4est.balanceforest!(forest)
    @test nleaves(forest) == 19

    # 2D example with balancing over a corner connection that is not within the topology tables
    cells = brick((2, 1))
    forest = Forest(cells, 3)
    PureP4est.refine_octant!(forest.trees[1], forest.trees[1].leaves[1])
    PureP4est.refine_octant!(forest.trees[1], forest.trees[1].leaves[2])
    PureP4est.refine_octant!(forest.trees[1], forest.trees[1].leaves[5])
    PureP4est.balanceforest!(forest)
    @test nleaves(forest) == 23

    #corner balance case but rotated
    cells = brick((2, 1))
    cells[1] = cells[1][[2, 3, 4, 1]]
    forest = Forest(cells, 3)
    PureP4est.refine_octant!(forest.trees[1], forest.trees[1].leaves[1])
    PureP4est.refine_octant!(forest.trees[1], forest.trees[1].leaves[1])
    PureP4est.refine_octant!(forest.trees[1], forest.trees[1].leaves[2])
    PureP4est.balanceforest!(forest)
    @test nleaves(forest) == 23

    # 3D case intra tree simple test, non conformity level 2
    cells = brick((1, 1, 1))
    forest = Forest(cells, 3)
    PureP4est.refine_all!(forest, 1)
    PureP4est.refine_octant!(forest.trees[1], forest.trees[1].leaves[2])
    PureP4est.refine_octant!(forest.trees[1], forest.trees[1].leaves[6])
    PureP4est.refine_octant!(forest.trees[1], forest.trees[1].leaves[6])
    balanced = PureP4est.balancetree(forest.trees[1])
    @test length(balanced.leaves) == 43

    #3D case intra tree non conformity level 3 at two different places
    forest = Forest(cells, 4)
    PureP4est.refine_all!(forest, 1)
    PureP4est.refine_octant!(forest.trees[1], forest.trees[1].leaves[2])
    PureP4est.refine_octant!(forest.trees[1], forest.trees[1].leaves[4])
    PureP4est.refine_octant!(forest.trees[1], forest.trees[1].leaves[7])
    PureP4est.refine_octant!(forest.trees[1], forest.trees[1].leaves[12])
    PureP4est.refine_octant!(forest.trees[1], forest.trees[1].leaves[28])
    PureP4est.refine_octant!(forest.trees[1], forest.trees[1].leaves[29])
    PureP4est.refine_octant!(forest.trees[1], forest.trees[1].leaves[37])
    PureP4est.refine_octant!(forest.trees[1], forest.trees[1].leaves[39])
    balanced = PureP4est.balancetree(forest.trees[1])
    @test length(balanced.leaves) == 127

    #3D case inter tree non conformity level 3 at two different places
    cells = brick((2, 2, 2))
    forest = Forest(cells, 4)
    PureP4est.refine_octant!(forest.trees[1], forest.trees[1].leaves[1])
    PureP4est.refine_octant!(forest.trees[1], forest.trees[1].leaves[2])
    PureP4est.refine_octant!(forest.trees[1], forest.trees[1].leaves[4])
    PureP4est.refine_octant!(forest.trees[7], forest.trees[7].leaves[1])
    PureP4est.refine_octant!(forest.trees[7], forest.trees[7].leaves[1])
    PureP4est.balanceforest!(forest)
    nleaves_ref = nleaves(forest)

    # Rotate three dimensional case
    cells = brick((2, 2, 2))
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
    cells[7] = cells[7][[2, 3, 4, 1, 6, 7, 8, 5]]
    cells[7] = cells[7][[2, 3, 4, 1, 6, 7, 8, 5]]
    forest = Forest(cells, 3)
    PureP4est.refine_octant!(forest.trees[1], forest.trees[1].leaves[1])
    PureP4est.refine_octant!(forest.trees[1], forest.trees[1].leaves[2])
    PureP4est.refine_octant!(forest.trees[1], forest.trees[1].leaves[4])
    PureP4est.refine_octant!(forest.trees[7], forest.trees[7].leaves[1])
    PureP4est.refine_octant!(forest.trees[7], forest.trees[7].leaves[1])
    PureP4est.balanceforest!(forest)
    @test nleaves(forest) == nleaves_ref
    @test nleaves(forest) == 92

    # edge balancing for new introduced connection that is not within topology table
    cells = brick((2, 1, 1))
    forest = Forest(cells, 3)
    PureP4est.refine!(forest, [1, 2])
    PureP4est.balanceforest!(forest)
    PureP4est.refine!(forest, [4])
    PureP4est.balanceforest!(forest)
    PureP4est.refine!(forest, [5])
    PureP4est.balanceforest!(forest)
    @test nleaves(forest) == 51

    #another edge balancing case
    cells = brick((2, 2, 2))
    forest = Forest(cells, 3)
    PureP4est.refine!(forest, 1)
    PureP4est.balanceforest!(forest)
    PureP4est.refine!(forest, [2, 4, 6, 8])
    PureP4est.balanceforest!(forest)
    PureP4est.refine!(forest, 34)
    PureP4est.balanceforest!(forest)
    # 141 = 134 + 7: balancing corner connections introduced by refinement (not present in
    # the macro topology) refines one additional leaf in this configuration
    @test nleaves(forest) == 141

    #yet another edge balancing case
    cells = brick((2, 2, 2))
    forest = Forest(cells, 3)
    PureP4est.refine!(forest, 1)
    PureP4est.balanceforest!(forest)
    PureP4est.refine!(forest, [2, 4, 6, 8])
    PureP4est.balanceforest!(forest)
    PureP4est.refine!(forest, 30)
    PureP4est.balanceforest!(forest)
    # 127 = 120 + 7: one additional leaf from refinement-introduced corner balancing
    @test nleaves(forest) == 127

    cells = brick((2, 2, 2))
    forest = Forest(cells, 3)
    PureP4est.refine_octant!(forest.trees[4], forest.trees[4].leaves[1])
    PureP4est.balanceforest!(forest)
    @test nleaves(forest) == 15

    #yet another edge balancing case
    cells = brick((2, 2, 2))
    forest = Forest(cells, 3)
    PureP4est.refine_octant!(forest.trees[4], forest.trees[4].leaves[1])
    PureP4est.balanceforest!(forest)
    PureP4est.refine_octant!(forest.trees[4], forest.trees[4].leaves[1])
    PureP4est.balanceforest!(forest)
    @test nleaves(forest) == 43

    #yet another edge balancing case
    cells = brick((2, 2, 2))
    forest = Forest(cells, 3)
    PureP4est.refine_octant!(forest.trees[4], forest.trees[4].leaves[1])
    PureP4est.balanceforest!(forest)
    PureP4est.refine_octant!(forest.trees[4], forest.trees[4].leaves[1])
    PureP4est.balanceforest!(forest)
    PureP4est.refine_octant!(forest.trees[3], forest.trees[3].leaves[2])
    PureP4est.refine_octant!(forest.trees[4], forest.trees[4].leaves[10])
    PureP4est.refine_octant!(forest.trees[4], forest.trees[4].leaves[3])
    PureP4est.balanceforest!(forest)
    @test nleaves(forest) == 71

    #yet another edge balancing case
    cells = brick((2, 2, 2))
    forest = Forest(cells, 3)
    PureP4est.refine_octant!(forest.trees[4], forest.trees[4].leaves[1])
    PureP4est.balanceforest!(forest)
    PureP4est.refine_octant!(forest.trees[4], forest.trees[4].leaves[1])
    PureP4est.balanceforest!(forest)
    PureP4est.refine_octant!(forest.trees[4], forest.trees[4].leaves[7])
    PureP4est.balanceforest!(forest)
    # 127 = 120 + 7: one additional leaf from refinement-introduced corner balancing
    @test nleaves(forest) == 127
end


@testset "corner balance across refinement-introduced connections" begin
    # Exhaustive 2:1 audit on the leaf boxes in brick coordinates (valid for the axis-aligned
    # trees of an unpermuted `brick`): counts leaf pairs whose closed boxes touch (face, edge or
    # corner contact) while their levels differ by 2 or more — zero for a fully balanced forest.
    function count_unbalanced_contacts(forest::Forest{dim}, dims::NTuple{dim, Int}) where {dim}
        boxes = Tuple{Int, NTuple{dim, Int}, NTuple{dim, Int}}[]
        for (k, tree) in enumerate(trees(forest))
            lo_t = brick_position(dims, k) .* Int(PureP4est._maximum_size(tree.b))
            for o in tree.leaves
                h = Int(PureP4est._compute_size(tree.b, o.l))
                lo = lo_t .+ Int.(o.xyz)
                push!(boxes, (Int(o.l), lo, lo .+ h))
            end
        end
        nviol = 0
        for i in eachindex(boxes), j in (i + 1):length(boxes)
            li, loi, hii = boxes[i]
            lj, loj, hij = boxes[j]
            abs(li - lj) < 2 && continue
            touching = all(d -> min(hii[d], hij[d]) >= max(loi[d], loj[d]), 1:dim)
            nviol += touching
        end
        return nviol
    end
    # Repeatedly refine the leaf of `trees(forest)[treeid]` selected by `pred`, then balance.
    function refine_towards_and_balance!(forest, treeid, nsteps, pred)
        for _ in 1:nsteps
            t = trees(forest)[treeid]
            PureP4est.refine_octant!(t, only(filter(pred, t.leaves)))
        end
        PureP4est.balanceforest!(forest)
        return forest
    end

    # A refined leaf's corner can touch another tree at a point that is NOT a macro vertex —
    # a corner connection "newly introduced" by refinement, absent from the macro topology.
    # Balancing must route these through the face (2D/3D) or edge (3D) the corner lies on.

    # 3D corner in the middle of the shared macro face
    forest = Forest(brick((2, 1, 1)), 4)
    m = Int(PureP4est._maximum_size(forest.trees[1].b))
    PureP4est.refine_octant!(forest.trees[2], forest.trees[2].leaves[1])
    refine_towards_and_balance!(forest, 2, 2, o -> Int(o.xyz[1]) == 0 && Int(o.xyz[2]) == m ÷ 2 && Int(o.xyz[3]) == m ÷ 2)
    @test count_unbalanced_contacts(forest, (2, 1, 1)) == 0

    # 3D corner in the middle of the shared macro edge (diagonal tree is an exclusive edge neighbour)
    forest = Forest(brick((2, 2, 1)), 4)
    PureP4est.refine_octant!(forest.trees[4], forest.trees[4].leaves[1])
    refine_towards_and_balance!(forest, 4, 2, o -> Int(o.xyz[1]) == 0 && Int(o.xyz[2]) == 0 && Int(o.xyz[3]) == m ÷ 2)
    @test count_unbalanced_contacts(forest, (2, 2, 1)) == 0
    # the balanced forest must still yield a conforming constrained space: every hanging node
    # is the mean of its masters (exact in integer brick coordinates, i.e. any linear function
    # is reproduced), and no master is itself hanging
    ln = lnodes(forest)
    X = [brick_position((2, 2, 1), k) .* m .+ Int.(xyz) for (k, xyz) in ln.noderefs]
    @test all(2 .* X[h] == X[m1] .+ X[m2] for (h, m1, m2) in ln.hanging2)
    @test all(4 .* X[h] == X[m1] .+ X[m2] .+ X[m3] .+ X[m4] for (h, m1, m2, m3, m4) in ln.hanging4)
    hanging = Set(first.(ln.hanging2)) ∪ Set(first.(ln.hanging4))
    @test all(mm ∉ hanging for hs in (ln.hanging2, ln.hanging4) for hm in hs for mm in hm[2:end])

    # 2D corner in the middle of the shared macro face while the tree's root corner in that
    # direction has an exclusive vertex neighbour (grid center) — must not shadow the fallback
    forest = Forest(brick((2, 2)), 4)
    PureP4est.refine_octant!(forest.trees[2], forest.trees[2].leaves[1])
    refine_towards_and_balance!(forest, 2, 2, o -> Int(o.xyz[1]) == 0 && Int(o.xyz[2]) + Int(PureP4est._compute_size(forest.trees[2].b, o.l)) == m ÷ 2)
    @test count_unbalanced_contacts(forest, (2, 2)) == 0

    # regression: macro-corner refinement (handled via vertex_vertex_neighbor) stays balanced
    forest = Forest(brick((2, 2, 2)), 4)
    for _ in 1:3
        t = forest.trees[4]
        leaf = only(filter(o -> o.xyz[1] == 0 && o.xyz[2] == 0 && Int(o.xyz[3]) + Int(PureP4est._compute_size(t.b, o.l)) == m, t.leaves))
        PureP4est.refine_octant!(t, leaf)
    end
    PureP4est.balanceforest!(forest)
    @test count_unbalanced_contacts(forest, (2, 2, 2)) == 0

    # regression: 2D through-face corner with no macro corner connection at all
    forest = Forest(brick((2, 1)), 4)
    PureP4est.refine_octant!(forest.trees[2], forest.trees[2].leaves[1])
    refine_towards_and_balance!(forest, 2, 2, o -> Int(o.xyz[1]) == 0 && Int(o.xyz[2]) == m ÷ 2)
    @test count_unbalanced_contacts(forest, (2, 1)) == 0
end

@testset "corner balance at a multi-tree vertex" begin
    # Five quads sharing a central vertex, with rotated connectivity so the center sits at a
    # different local corner in each tree (as in unstructured meshes). The vertex-only
    # neighbor lists at the center then have two entries whose corner indices differ, so
    # `transform_corner` must place the balance octant at the corner the caller resolved from
    # the connection — re-deriving it from `vertex_vertex_neighbor[..][1]` picks an arbitrary
    # incident tree and plants refinement at a wrong (far-away) corner. See the report in
    # PR #1349: spurious refinement clusters one macro cell away from the refined notch tip.
    # Vertex 1 is the center, 2 + 2i the ring vertex shared by quads i-1 and i, 3 + 2i the
    # outer kite vertex of quad i.
    nquads = 5
    cells = map(0:(nquads - 1)) do i
        t = (1, 2 + 2i, 3 + 2i, i == nquads - 1 ? 2 : 4 + 2i)
        r = i % 4 # cyclic rotation keeps orientation but moves the center's local index
        return ntuple(j -> t[mod1(j + r, 4)], 4)
    end
    forest = Forest(cells, 6)
    # the corner of each tree at the central vertex, in the tree's integer coordinates
    vcs = map(1:nquads) do k
        c_bwg = PureP4est.node_map₂_inv[findfirst(==(1), cells[k])]
        return PureP4est.vertex(PureP4est.root(2), c_bwg, forest.trees[k].b)
    end
    # Chebyshev distance from a leaf's box to its tree's corner vc at the central vertex
    function cheb(leaf, b, vc)
        h = PureP4est._compute_size(b, leaf.l)
        return maximum(d -> max(leaf.xyz[d] - vc[d], vc[d] - (leaf.xyz[d] + h), 0), 1:2)
    end
    # refine all trees toward the central vertex three times, balancing in between
    for _ in 1:3
        leaves = [(k, leaf) for (k, tree) in enumerate(trees(forest)) for leaf in tree.leaves]
        marked = [i for (i, (k, leaf)) in enumerate(leaves) if cheb(leaf, forest.trees[k].b, vcs[k]) == 0]
        PureP4est.refine!(forest, marked)
        PureP4est.balanceforest!(forest)
    end
    for (k, tree) in enumerate(trees(forest))
        b = tree.b
        maxlvl_at_corner = 0
        for leaf in tree.leaves
            h = PureP4est._compute_size(b, leaf.l)
            c = cheb(leaf, b, vcs[k])
            c == 0 && (maxlvl_at_corner = max(maxlvl_at_corner, Int(leaf.l)))
            # deep leaves may only appear in the graded halo around the central vertex
            leaf.l >= 2 && @test c <= 2h
        end
        # ... and the 2:1 balance against the level-3 corner leaves must actually hold
        @test maxlvl_at_corner >= 2
    end
end

@testset "refine!/coarsen! edge branches" begin
    # scalar refine! of a cell in a later tree walks the per-tree leaf counts
    forest = Forest(brick((2, 2)), 3)
    PureP4est.refine!(forest, 3)
    @test nleaves(forest) == 7
    @test length(forest.trees[3].leaves) == 4
    @test length(forest.trees[1].leaves) == 1

    # coarsen! from a non-first sibling snaps back to the family's first sibling
    forest = Forest(brick((1, 1)), 3)
    PureP4est.refine_all!(forest, 1)
    tree = forest.trees[1]
    @test length(tree.leaves) == 4
    PureP4est.coarsen_octant!(tree, tree.leaves[2])
    @test length(tree.leaves) == 1
    @test tree.leaves[1].l == 0

    # refine_all! on mixed levels keeps the leaves that are not at level l-1
    forest = Forest(brick((1, 1)), 3)
    PureP4est.refine!(forest, 1)
    PureP4est.refine!(forest, 1)
    @test nleaves(forest) == 7 # 4 at level 2, 3 at level 1
    PureP4est.refine_all!(forest, 3) # only the level-2 leaves refine
    @test nleaves(forest) == 19 # 16 at level 3, 3 at level 1
end

@testset "batch coarsen!(forest, cellids) $dim D" for dim in (2, 3)
    nchild = 2^dim

    # batch coarsen! inverts batch refine!: refining cell 1 then coarsening its children
    # (ids 1:nchild) recovers the original forest octant-for-octant
    forest = Forest(brick(ntuple(_ -> 2, dim)), 4)
    PureP4est.refine_all!(forest, 1)
    base = forest_leaves(forest)
    PureP4est.refine!(forest, [1])
    @test nleaves(forest) == length(base) + (nchild - 1)
    PureP4est.coarsen!(forest, collect(1:nchild))
    @test forest_leaves(forest) == base
    for tree in forest.trees
        @test issorted(tree.leaves)
    end

    # on a uniformly refined forest, coarsening every cell with require_all_siblings
    # reproduces _coarsen_all!
    f1 = Forest(brick(ntuple(_ -> 1, dim)), 4)
    PureP4est.refine_all!(f1, 1)
    PureP4est.refine_all!(f1, 2)
    f2 = deepcopy(f1)
    PureP4est._coarsen_all!(f1)
    PureP4est.coarsen!(f2, collect(1:nleaves(f2)); require_all_siblings = true)
    @test forest_leaves(f1) == forest_leaves(f2)

    # policy modularity: one marked sibling is a no-op under all-siblings, collapses the
    # family under any-sibling
    f = Forest(brick(ntuple(_ -> 1, dim)), 4)
    PureP4est.refine_all!(f, 1)
    n0 = nleaves(f)
    fa = deepcopy(f)
    PureP4est.coarsen!(fa, [1]; require_all_siblings = true)
    @test nleaves(fa) == n0
    fb = deepcopy(f)
    PureP4est.coarsen!(fb, [1]; require_all_siblings = false)
    @test nleaves(fb) == n0 - (nchild - 1)

    # incomplete family (one child refined further) cannot be coarsened -> silently skipped
    fi = Forest(brick(ntuple(_ -> 1, dim)), 4)
    PureP4est.refine_all!(fi, 1)
    PureP4est.refine!(fi, [1]) # child 1 -> nchild grandchildren; family now incomplete
    n1 = nleaves(fi)
    PureP4est.coarsen!(fi, collect((nchild + 1):n1); require_all_siblings = false) # the surviving level-1 leaves
    @test nleaves(fi) == n1

    # the caller's id vector is not mutated
    ids = collect(nchild:-1:1)
    fc = Forest(brick(ntuple(_ -> 1, dim)), 4)
    PureP4est.refine_all!(fc, 1)
    PureP4est.coarsen!(fc, ids)
    @test ids == collect(nchild:-1:1)

    # empty ids is a no-op
    fe = Forest(brick(ntuple(_ -> 1, dim)), 4)
    PureP4est.refine_all!(fe, 1)
    m = nleaves(fe)
    PureP4est.coarsen!(fe, Int[])
    @test nleaves(fe) == m
end

@testset "refine_and_coarsen! $dim D" for dim in (2, 3)
    nchild = 2^dim
    ntree = 2^dim # (2,2[,2]) grid: one tree per octant

    # fused single pass, no balancing: coarsen tree 1's family (ids 1:nchild) while refining
    # the first leaf of tree 3 (global id 2*nchild + 1). Both ids resolve against the
    # ORIGINAL numbering, so the coarsen is not thrown off by the refine.
    forest = Forest(brick(ntuple(_ -> 2, dim)), 4)
    PureP4est.refine_all!(forest, 1)
    n0 = nleaves(forest)
    refid = 2 * nchild + 1
    PureP4est.refine_and_coarsen!(forest, collect(1:nchild), [refid]; balance = false)
    @test nleaves(forest) == n0 - (nchild - 1) + (nchild - 1)
    @test length(forest.trees[1].leaves) == 1              # tree 1 coarsened to root
    @test length(forest.trees[3].leaves) == 2 * nchild - 1 # tree 3 leaf 1 refined
    for tree in forest.trees
        @test issorted(tree.leaves)
    end

    # with balancing (default), the coarsened tree 1 sits next to tree 3's finer cells and is
    # re-refined to restore 2:1; the result is a balanced forest lnodes accepts
    balanced = Forest(brick(ntuple(_ -> 2, dim)), 4)
    PureP4est.refine_all!(balanced, 1)
    PureP4est.refine_and_coarsen!(balanced, collect(1:nchild), [refid])
    @test lnodes(balanced) isa PureP4est.LNodes

    # roundtrip: refine cell 1, then coarsen its children back to the original forest
    h = Forest(brick(ntuple(_ -> 2, dim)), 4)
    base = forest_leaves(h)
    PureP4est.refine_and_coarsen!(h, Int[], [1]; balance = false)
    @test nleaves(h) == length(base) + (nchild - 1)
    PureP4est.refine_and_coarsen!(h, collect(1:nchild), Int[]; balance = false)
    @test forest_leaves(h) == base

    # conflict: overlapping refine/coarsen ids
    c = Forest(brick(ntuple(_ -> 2, dim)), 4)
    PureP4est.refine_all!(c, 1)
    @test_throws ArgumentError PureP4est.refine_and_coarsen!(c, [1, 2], [2, nchild + 1])

    # conflict: a refine id inside a family being coarsened (reachable with any-sibling policy)
    c2 = Forest(brick(ntuple(_ -> 2, dim)), 4)
    PureP4est.refine_all!(c2, 1)
    @test_throws ArgumentError PureP4est.refine_and_coarsen!(c2, [1], [3]; require_all_siblings = false)
end

@testset "Forest accessors and error paths" begin
    forest = Forest(brick((2, 2)), 3)
    PureP4est.refine_all!(forest, 1)

    # marking a cell already at the maximum level is a documented no-op — also when the
    # tree's *first* leaf is the max-level one (used to throw from the 2^dim size hint)
    let f = Forest(brick((2, 2)), 2)
        PureP4est.refine!(f, [1])
        PureP4est.balanceforest!(f)
        PureP4est.refine!(f, [1])
        PureP4est.balanceforest!(f)
        n = nleaves(f)
        PureP4est.refine!(f, [1])            # cell 1 = first leaf of tree 1, at max level
        @test nleaves(f) == n
        PureP4est.refine_and_coarsen!(f, [1], Int[]) # same guard in the fused path
        @test nleaves(f) == n
        @test size(lnodes(f).E, 2) == n
    end
    # forest_leaves collects the leaves of all trees in cell id order (tree by tree, Morton
    # order within each tree)
    leaves = forest_leaves(forest)
    @test length(leaves) == nleaves(forest)
    @test leaves[7] == forest.trees[2].leaves[3]
    @test leaves[1] == forest.trees[1].leaves[1]

    # The maximum refinement level is bounded by p4est's P8EST_MAXLEVEL: beyond it an octree
    # coordinate no longer fits the per-axis bit budget of the UInt64 boundary-table keys,
    # whose collisions would silently merge unrelated nodes across tree boundaries. A
    # two-tree forest refined once at the bound has 45 nodes; a b past the limit used to
    # inflate this because the shared face nodes failed to merge.
    let f = Forest(brick((2, 1, 1)), 19)
        PureP4est.refine_all!(f, 1)
        @test nnodes(lnodes(f)) == 45
    end
end
