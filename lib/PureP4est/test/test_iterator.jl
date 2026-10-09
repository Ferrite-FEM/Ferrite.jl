# The point iterator and its consumers: node numbering with hanging-node detection
# (`lnodes`) and the facet skeleton. Mirrors the iterator/`lnodes`/`facetskeleton` half of
# `src/forest.jl`.
using PureP4est, Test

include(joinpath(@__DIR__, "utils.jl"))

@testset "point iterator LeafSupport iteration" begin
    forest = Forest(brick((1, 1)), 3)
    PureP4est.refine!(forest, 1)
    tree = forest.trees[1]
    sc = PureP4est.IterScratch(tree)
    octs = PureP4est.OctantBWG{2, 4, Int64}[]
    PureP4est.iterate_points(tree, sc; mindim = 0, maxdim = 1) do c, ls
        @test collect(ls) == [ls[i] for i in 1:length(ls)] # Base.iterate/eltype agree with getindex
        append!(octs, ls)
        return
    end
    @test !isempty(octs)
    @test all(o -> o ∈ tree.leaves, octs)
end

@testset "lnodes node numbering" begin
    #################################################
    ############ structured 2D examples #############
    #################################################

    # 2D case with a single tree
    cells = brick((1, 1))
    forest = Forest(cells, 3)
    PureP4est.refine_all!(forest, 1)
    PureP4est.refine_octant!(forest.trees[1], forest.trees[1].leaves[1])
    PureP4est.refine_octant!(forest.trees[1], forest.trees[1].leaves[1])
    ln = lnodes(forest)
    @test size(ln.E, 2) == 10
    @test nnodes(ln) == 19

    #2D case with four trees and a nonuniform refinement pattern
    cells = brick((2, 2))
    forest = Forest(cells, 3)
    PureP4est.refine_all!(forest, 1)
    PureP4est.refine_octant!(forest.trees[1], forest.trees[1].leaves[1])
    PureP4est.refine_octant!(forest.trees[1], forest.trees[1].leaves[1])
    ln = lnodes(forest)
    @test size(ln.E, 2) == 22
    @test nnodes(ln) == 35

    #more random refinement
    cells = brick((3, 3))
    forest = Forest(cells, 3)
    PureP4est.refine_octant!(forest.trees[1], forest.trees[1].leaves[1])
    PureP4est.refine_octant!(forest.trees[1], forest.trees[1].leaves[1])
    PureP4est.refine_octant!(forest.trees[1], forest.trees[1].leaves[1])
    PureP4est.refine_octant!(forest.trees[3], forest.trees[3].leaves[1])
    PureP4est.refine_octant!(forest.trees[3], forest.trees[3].leaves[2])
    PureP4est.refine_octant!(forest.trees[3], forest.trees[3].leaves[3])
    PureP4est.refine_octant!(forest.trees[7], forest.trees[7].leaves[1])
    PureP4est.refine_octant!(forest.trees[7], forest.trees[7].leaves[3])
    PureP4est.refine_octant!(forest.trees[7], forest.trees[7].leaves[5])
    PureP4est.refine_octant!(forest.trees[9], forest.trees[9].leaves[end])
    PureP4est.refine_octant!(forest.trees[9], forest.trees[9].leaves[end])
    PureP4est.refine_octant!(forest.trees[9], forest.trees[9].leaves[end])
    ln = lnodes(forest)
    @test size(ln.E, 2) == 45
    @test nnodes(ln) == 76

    #################################################
    ############ structured 3D examples #############
    #################################################

    # 3D case with a single tree
    cells = brick((1, 1, 1))
    forest = Forest(cells, 3)
    PureP4est.refine_all!(forest, 1)
    PureP4est.refine_octant!(forest.trees[1], forest.trees[1].leaves[1])
    PureP4est.refine_octant!(forest.trees[1], forest.trees[1].leaves[1])
    ln = lnodes(forest)
    @test size(ln.E, 2) == 8 + 7 + 7
    @test nnodes(ln) == 65

    # Test only Interoctree by face connection
    cells = brick((2, 1, 1))
    forest = Forest(cells, 3)
    PureP4est.refine_all!(forest, 1)
    ln = lnodes(forest)
    @test size(ln.E, 2) == 16
    @test nnodes(ln) == 45
    #rotate the case around
    cells = brick((1, 2, 1))
    forest = Forest(cells, 3)
    PureP4est.refine_all!(forest, 1)
    ln = lnodes(forest)
    @test size(ln.E, 2) == 16
    @test nnodes(ln) == 45
    cells = brick((1, 1, 2))
    forest = Forest(cells, 3)
    PureP4est.refine_all!(forest, 1)
    ln = lnodes(forest)
    @test size(ln.E, 2) == 16
    @test nnodes(ln) == 45

    cells = brick((2, 2, 2))
    forest = Forest(cells, 3)
    PureP4est.refine_all!(forest, 1)
    ln = lnodes(forest)
    @test size(ln.E, 2) == 8^2
    @test nnodes(ln) == 125 # 5 per edge

    # Rotate three dimensional case
    cells = brick((2, 2, 2))
    # Rotate face topologically
    cells[2] = cells[2][[2, 3, 4, 1, 6, 7, 8, 5]]
    cells[2] = cells[2][[2, 3, 4, 1, 6, 7, 8, 5]]
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
    forest = Forest(cells, 3)
    PureP4est.refine_all!(forest, 1)
    ln = lnodes(forest)
    @test size(ln.E, 2) == 8^2
    @test nnodes(ln) == 125 # 5 per edge
end

@testset "hanging nodes" begin
    #Easy Intraoctree
    cells = brick((1, 1, 1))
    forest = Forest(cells, 3)
    PureP4est.refine_all!(forest, 1)
    PureP4est.refine_octant!(forest.trees[1], forest.trees[1].leaves[1])
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
    ln = lnodes(forest)
    @test nhanging(ln) == 12

    # Easy Interoctree
    cells = brick((2, 2, 2))
    forest = Forest(cells, 3)
    PureP4est.refine_octant!(forest.trees[1], forest.trees[1].leaves[1])
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
    ln = lnodes(forest)
    @test nhanging(ln) == 12

    #rotate the case from above in the first cell around
    cells = brick((2, 2, 2))
    # Rotate face topologically
    cells[1] = cells[1][[2, 3, 4, 1, 6, 7, 8, 5]]
    cells[1] = cells[1][[2, 3, 4, 1, 6, 7, 8, 5]]
    forest = Forest(cells, 3)
    PureP4est.refine_octant!(forest.trees[1], forest.trees[1].leaves[1])
    ln = lnodes(forest)
    @test nhanging(ln) == 12

    #2D rotated case
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
    PureP4est.refine_octant!(forest.trees[2], forest.trees[2].leaves[1])
    ln = lnodes(forest)
    @test nhanging(ln) == 2

    # multiple corner connections in 2D by disc discretization
    cells = disc(10)
    forest = Forest(cells, 3)
    @test nleaves(forest) == 10
    PureP4est.refine_octant!(forest.trees[1], forest.trees[1].leaves[1])
    PureP4est.refine_octant!(forest.trees[1], forest.trees[1].leaves[3])
    @test nleaves(forest) == 16
    PureP4est.balanceforest!(forest)
    @test nleaves(forest) == 9 * 4 + 3 + 4

    # multiple corner connections in 3D by cylinder discretization
    cells = disc3(10)
    forest = Forest(cells, 3)
    @test nleaves(forest) == 10
    PureP4est.refine_octant!(forest.trees[1], forest.trees[1].leaves[1])
    @test nleaves(forest) == 17
    PureP4est.refine_octant!(forest.trees[1], forest.trees[1].leaves[3])
    @test nleaves(forest) == 24
    PureP4est.balanceforest!(forest)
    @test nleaves(forest) == 9 * 8 + 7 + 8
end

@testset "facet skeleton" begin
    # Leaf-level facet interfaces of the refined forest, as (cell id, local facet) pairs.
    # The pairs themselves are checked against the materialized grid in Ferrite's suite.

    # 2x1: refine cell 1 once -> 4 intra-tree conforming + 2 inter-tree hanging pairs
    forest = Forest(brick((2, 1)), 3)
    PureP4est.refine!(forest, [1])
    PureP4est.balanceforest!(forest)
    @test length(PureP4est.facetskeleton(forest)) == 6

    # 2x1 both refined -> 4 + 4 intra-tree + 2 inter-tree conforming pairs
    forest = Forest(brick((2, 1)), 3)
    PureP4est.refine_all!(forest, 1)
    @test length(PureP4est.facetskeleton(forest)) == 10

    # 3D intra-octree hanging (cf. "hanging nodes" testset)
    forest = Forest(brick((1, 1, 1)), 3)
    PureP4est.refine_all!(forest, 1)
    PureP4est.refine_octant!(forest.trees[1], forest.trees[1].leaves[1])
    # 8 octants -> 12 coarse interfaces; refining one octant replaces 3 of them by 4
    # hanging subfacet pairs each and adds 12 interfaces between its children
    @test length(PureP4est.facetskeleton(forest)) == 12 - 3 + 3 * 4 + 12
end

@testset "unbalanced forest errors" begin
    forest = Forest(brick((1, 1)), 4)
    PureP4est.refine!(forest, 1) # 4 level-1 leaves
    PureP4est.refine!(forest, 2) # second quadrant -> level 2
    PureP4est.refine!(forest, 2) # its first child -> level 3, faces the level-1 first quadrant: 2:1 violated
    @test_throws ArgumentError lnodes(forest)
    @test_throws ArgumentError PureP4est.facetskeleton(forest)
end

@testset "lnodes on a deep uniform tree" begin
    # 64 leaves under the root: exercises the binary-search branch of split_bounds
    forest = Forest(brick((1, 1)), 4)
    for l in 1:3
        PureP4est.refine_all!(forest, l)
    end
    ln = lnodes(forest)
    @test size(ln.E, 2) == 64
    @test nnodes(ln) == 81 # (2^3 + 1)^2
    @test isempty(ln.hanging2) && isempty(ln.hanging4)
end
