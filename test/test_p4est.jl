using Ferrite, Test

include(joinpath(@__DIR__, "test_utils.jl"))

@testset "OctantBWG Lookup Tables" begin
    @test Ferrite.AMR._face(1) == [3, 5]
    @test Ferrite.AMR._face(5) == [1, 5]
    @test Ferrite.AMR._face(12) == [2, 4]
    @test Ferrite.AMR._face(1, 1) == 3  && Ferrite.AMR._face(1, 2) == 5
    @test Ferrite.AMR._face(5, 1) == 1  && Ferrite.AMR._face(5, 2) == 5
    @test Ferrite.AMR._face(12, 1) == 2 && Ferrite.AMR._face(12, 2) == 4
    @test Ferrite.AMR._face(3, 1) == 3  && Ferrite.AMR._face(3, 2) == 6

    @test Ferrite.AMR._face_edge_corners(1, 1) == (0, 0)
    @test Ferrite.AMR._face_edge_corners(3, 3) == (3, 4)
    @test Ferrite.AMR._face_edge_corners(8, 6) == (2, 4)
    @test Ferrite.AMR._face_edge_corners(4, 5) == (0, 0)
    @test Ferrite.AMR._face_edge_corners(5, 4) == (0, 0)
    @test Ferrite.AMR._face_edge_corners(7, 1) == (3, 4)
    @test Ferrite.AMR._face_edge_corners(11, 1) == (2, 4)
    @test Ferrite.AMR._face_edge_corners(9, 1) == (1, 3)
    @test Ferrite.AMR._face_edge_corners(10, 2) == (1, 3)
    @test Ferrite.AMR._face_edge_corners(12, 2) == (2, 4)

    @test Ferrite.AMR.𝒱₃[1, :] == Ferrite.AMR.𝒰[1:4, 1] == Ferrite.AMR._face_corners(3, 1)
    @test Ferrite.AMR.𝒱₃[2, :] == Ferrite.AMR.𝒰[1:4, 2] == Ferrite.AMR._face_corners(3, 2)
    @test Ferrite.AMR.𝒱₃[3, :] == Ferrite.AMR.𝒰[5:8, 1] == Ferrite.AMR._face_corners(3, 3)
    @test Ferrite.AMR.𝒱₃[4, :] == Ferrite.AMR.𝒰[5:8, 2] == Ferrite.AMR._face_corners(3, 4)
    @test Ferrite.AMR.𝒱₃[5, :] == Ferrite.AMR.𝒰[9:12, 1] == Ferrite.AMR._face_corners(3, 5)
    @test Ferrite.AMR.𝒱₃[6, :] == Ferrite.AMR.𝒰[9:12, 2] == Ferrite.AMR._face_corners(3, 6)

    @test Ferrite.AMR._edge_corners(1) == [1, 2]
    @test Ferrite.AMR._edge_corners(4) == [7, 8]
    @test Ferrite.AMR._edge_corners(12, 2) == 8

    #Test Figure 3a) of Burstedde, Wilcox, Ghattas [2011]
    test_ξs = (1, 2, 3, 4)
    @test Ferrite.AMR._neighbor_corner.((1,), (2,), (1,), test_ξs) == test_ξs
    #Test Figure 3b)
    @test Ferrite.AMR._neighbor_corner.((3,), (5,), (3,), test_ξs) == (Ferrite.AMR.𝒫[5, :]...,)
end

@testset "Index Permutation" begin
    for i in 1:length(Ferrite.AMR.edge_perm)
        @test i == Ferrite.AMR.edge_perm_inv[Ferrite.AMR.edge_perm[i]]
    end
    for i in 1:length(Ferrite.AMR.𝒱₂_perm)
        @test i == Ferrite.AMR.𝒱₂_perm_inv[Ferrite.AMR.𝒱₂_perm[i]]
    end
    for i in 1:length(Ferrite.AMR.𝒱₃_perm)
        @test i == Ferrite.AMR.𝒱₃_perm_inv[Ferrite.AMR.𝒱₃_perm[i]]
    end
    for i in 1:length(Ferrite.AMR.node_map₂)
        @test i == Ferrite.AMR.node_map₂_inv[Ferrite.AMR.node_map₂[i]]
    end
    for i in 1:length(Ferrite.AMR.node_map₃)
        @test i == Ferrite.AMR.node_map₃_inv[Ferrite.AMR.node_map₃[i]]
    end
end

@testset "OctantBWG Encoding" begin
    #    # Tests from Figure 3a) and 3b) of Burstedde et al
    o = Ferrite.AMR.Ferrite.AMR.OctantBWG(3, 2, 21, 3)
    b = 3
    @test Ferrite.AMR.child_id(o, b) == 5
    @test Ferrite.AMR.child_id(Ferrite.AMR.parent(o, b), b) == 3
    @test Ferrite.AMR.parent(Ferrite.AMR.parent(o, b), b) == Ferrite.AMR.Ferrite.AMR.OctantBWG(3, 0, 1, b)
    @test Ferrite.AMR.parent(Ferrite.AMR.parent(Ferrite.AMR.parent(o, b), b), b) == Ferrite.AMR.root(3)
    o = Ferrite.AMR.Ferrite.AMR.OctantBWG(3, 2, 4, 3)
    @test Ferrite.AMR.child_id(o, b) == 4
    @test Ferrite.AMR.child_id(Ferrite.AMR.parent(o, b), b) == 1
    @test Ferrite.AMR.parent(Ferrite.AMR.parent(o, b), b) == Ferrite.AMR.Ferrite.AMR.OctantBWG(3, 0, 1, b)
    @test Ferrite.AMR.parent(Ferrite.AMR.parent(Ferrite.AMR.parent(o, b), b), b) == Ferrite.AMR.root(3)

    @test Ferrite.AMR.child_id(Ferrite.AMR.Ferrite.AMR.OctantBWG(2, 1, 1, 3), 3) == 1
    @test Ferrite.AMR.child_id(Ferrite.AMR.Ferrite.AMR.OctantBWG(2, 1, 2, 3), 3) == 2
    @test Ferrite.AMR.child_id(Ferrite.AMR.Ferrite.AMR.OctantBWG(2, 1, 3, 3), 3) == 3
    @test Ferrite.AMR.child_id(Ferrite.AMR.Ferrite.AMR.OctantBWG(2, 1, 4, 3), 3) == 4
    @test Ferrite.AMR.child_id(Ferrite.AMR.Ferrite.AMR.OctantBWG(2, 2, 1, 3), 3) == 1
    @test Ferrite.AMR.child_id(Ferrite.AMR.Ferrite.AMR.OctantBWG(3, 2, 1, 3), 3) == 1
    @test Ferrite.AMR.child_id(Ferrite.AMR.Ferrite.AMR.OctantBWG(3, 2, 2, 3), 3) == 2
    @test Ferrite.AMR.child_id(Ferrite.AMR.Ferrite.AMR.OctantBWG(3, 2, 3, 3), 3) == 3
    @test Ferrite.AMR.child_id(Ferrite.AMR.Ferrite.AMR.OctantBWG(3, 2, 4, 3), 3) == 4
    @test Ferrite.AMR.child_id(Ferrite.AMR.Ferrite.AMR.OctantBWG(3, 2, 16, 3), 3) == 8
    @test Ferrite.AMR.child_id(Ferrite.AMR.Ferrite.AMR.OctantBWG(3, 2, 24, 3), 3) == 8
    @test Ferrite.AMR.child_id(Ferrite.AMR.Ferrite.AMR.OctantBWG(3, 2, 64, 3), 3) == 8
    @test Ferrite.AMR.child_id(Ferrite.AMR.Ferrite.AMR.OctantBWG(3, 2, 9, 3), 3) == 1
    #maxlevel = 10 takes too long
    maxlevel = 6
    levels = collect(1:maxlevel)
    morton_ids = [1:(2^(2 * l)) for l in levels]
    for (level, morton_range) in zip(levels, morton_ids)
        for morton_id in morton_range
            @test Int(Ferrite.AMR.morton(Ferrite.AMR.OctantBWG(2, level, morton_id, maxlevel), level, maxlevel)) == morton_id
        end
    end
    morton_ids = [1:(2^(3 * l)) for l in levels]
    for (level, morton_range) in zip(levels, morton_ids)
        for morton_id in morton_range
            @test Int(Ferrite.AMR.morton(Ferrite.AMR.OctantBWG(3, level, morton_id, maxlevel), level, maxlevel)) == morton_id
        end
    end
end

@testset "OctantBWG Operations" begin
    o = Ferrite.AMR.Ferrite.AMR.OctantBWG(1, (2, 0, 0))
    @test Ferrite.AMR.facet_neighbor(o, 1, 2) == Ferrite.AMR.Ferrite.AMR.OctantBWG(1, (0, 0, 0))
    @test Ferrite.AMR.facet_neighbor(o, 2, 2) == Ferrite.AMR.Ferrite.AMR.OctantBWG(1, (4, 0, 0))
    @test Ferrite.AMR.facet_neighbor(o, 3, 2) == Ferrite.AMR.Ferrite.AMR.OctantBWG(1, (2, -2, 0))
    @test Ferrite.AMR.facet_neighbor(o, 4, 2) == Ferrite.AMR.Ferrite.AMR.OctantBWG(1, (2, 2, 0))
    @test Ferrite.AMR.facet_neighbor(o, 5, 2) == Ferrite.AMR.Ferrite.AMR.OctantBWG(1, (2, 0, -2))
    @test Ferrite.AMR.facet_neighbor(o, 6, 2) == Ferrite.AMR.Ferrite.AMR.OctantBWG(1, (2, 0, 2))
    @test Ferrite.AMR.descendants(o, 2) == (Ferrite.AMR.Ferrite.AMR.OctantBWG(2, (2, 0, 0)), Ferrite.AMR.Ferrite.AMR.OctantBWG(2, (3, 1, 1)))
    @test Ferrite.AMR.descendants(o, 3) == (Ferrite.AMR.Ferrite.AMR.OctantBWG(3, (2, 0, 0)), Ferrite.AMR.Ferrite.AMR.OctantBWG(3, (5, 3, 3)))

    o = Ferrite.AMR.Ferrite.AMR.OctantBWG(1, (0, 0, 0))
    @test Ferrite.AMR.facet_neighbor(o, 1, 2) == Ferrite.AMR.Ferrite.AMR.OctantBWG(1, (-2, 0, 0))
    @test Ferrite.AMR.facet_neighbor(o, 2, 2) == Ferrite.AMR.Ferrite.AMR.OctantBWG(1, (2, 0, 0))
    @test Ferrite.AMR.facet_neighbor(o, 3, 2) == Ferrite.AMR.Ferrite.AMR.OctantBWG(1, (0, -2, 0))
    @test Ferrite.AMR.facet_neighbor(o, 4, 2) == Ferrite.AMR.Ferrite.AMR.OctantBWG(1, (0, 2, 0))
    @test Ferrite.AMR.facet_neighbor(o, 5, 2) == Ferrite.AMR.Ferrite.AMR.OctantBWG(1, (0, 0, -2))
    @test Ferrite.AMR.facet_neighbor(o, 6, 2) == Ferrite.AMR.Ferrite.AMR.OctantBWG(1, (0, 0, 2))
    o = Ferrite.AMR.Ferrite.AMR.OctantBWG(0, (0, 0, 0))
    @test Ferrite.AMR.descendants(o, 2) == (Ferrite.AMR.Ferrite.AMR.OctantBWG(2, (0, 0, 0)), Ferrite.AMR.Ferrite.AMR.OctantBWG(2, (3, 3, 3)))
    @test Ferrite.AMR.descendants(o, 3) == (Ferrite.AMR.Ferrite.AMR.OctantBWG(3, (0, 0, 0)), Ferrite.AMR.Ferrite.AMR.OctantBWG(3, (7, 7, 7)))

    @test Ferrite.AMR.edge_neighbor(Ferrite.AMR.Ferrite.AMR.OctantBWG(2, (2, 0, 0)), 1, 3) == Ferrite.AMR.Ferrite.AMR.OctantBWG(2, (2, -2, -2))
    @test Ferrite.AMR.edge_neighbor(Ferrite.AMR.Ferrite.AMR.OctantBWG(2, (2, 0, 0)), 4, 3) == Ferrite.AMR.Ferrite.AMR.OctantBWG(2, (2, 2, 2))
    @test Ferrite.AMR.edge_neighbor(Ferrite.AMR.Ferrite.AMR.OctantBWG(2, (2, 0, 0)), 6, 3) == Ferrite.AMR.Ferrite.AMR.OctantBWG(2, (4, 0, -2))
    @test Ferrite.AMR.edge_neighbor(Ferrite.AMR.Ferrite.AMR.OctantBWG(2, (2, 0, 0)), 9, 3) == Ferrite.AMR.Ferrite.AMR.OctantBWG(2, (0, -2, 0))
    @test Ferrite.AMR.edge_neighbor(Ferrite.AMR.Ferrite.AMR.OctantBWG(2, (2, 0, 0)), 12, 3) == Ferrite.AMR.Ferrite.AMR.OctantBWG(2, (4, 2, 0))

    @test Ferrite.AMR.edge_neighbor(Ferrite.AMR.Ferrite.AMR.OctantBWG(3, (0, 0, 0)), 1, 4) == Ferrite.AMR.Ferrite.AMR.OctantBWG(3, (0, -2, -2))
    @test Ferrite.AMR.edge_neighbor(Ferrite.AMR.Ferrite.AMR.OctantBWG(3, (0, 0, 0)), 12, 4) == Ferrite.AMR.Ferrite.AMR.OctantBWG(3, (2, 2, 0))

    @test Ferrite.AMR.edge_neighbor(Ferrite.AMR.Ferrite.AMR.OctantBWG(2, (0, 0, 0)), 1, 4) == Ferrite.AMR.Ferrite.AMR.OctantBWG(2, (0, -4, -4))
    @test Ferrite.AMR.edge_neighbor(Ferrite.AMR.Ferrite.AMR.OctantBWG(2, (0, 0, 0)), 2, 4) == Ferrite.AMR.Ferrite.AMR.OctantBWG(2, (0, 4, -4))
    @test Ferrite.AMR.edge_neighbor(Ferrite.AMR.Ferrite.AMR.OctantBWG(2, (0, 0, 0)), 3, 4) == Ferrite.AMR.Ferrite.AMR.OctantBWG(2, (0, -4, 4))
    @test Ferrite.AMR.edge_neighbor(Ferrite.AMR.Ferrite.AMR.OctantBWG(2, (0, 0, 0)), 4, 4) == Ferrite.AMR.Ferrite.AMR.OctantBWG(2, (0, 4, 4))
    @test Ferrite.AMR.edge_neighbor(Ferrite.AMR.Ferrite.AMR.OctantBWG(2, (0, 0, 0)), 5, 4) == Ferrite.AMR.Ferrite.AMR.OctantBWG(2, (-4, 0, -4))
    @test Ferrite.AMR.edge_neighbor(Ferrite.AMR.Ferrite.AMR.OctantBWG(2, (0, 0, 0)), 6, 4) == Ferrite.AMR.Ferrite.AMR.OctantBWG(2, (4, 0, -4))
    @test Ferrite.AMR.edge_neighbor(Ferrite.AMR.Ferrite.AMR.OctantBWG(2, (0, 0, 0)), 7, 4) == Ferrite.AMR.Ferrite.AMR.OctantBWG(2, (-4, 0, 4))
    @test Ferrite.AMR.edge_neighbor(Ferrite.AMR.Ferrite.AMR.OctantBWG(2, (0, 0, 0)), 8, 4) == Ferrite.AMR.Ferrite.AMR.OctantBWG(2, (4, 0, 4))
    @test Ferrite.AMR.edge_neighbor(Ferrite.AMR.Ferrite.AMR.OctantBWG(2, (0, 0, 0)), 9, 4) == Ferrite.AMR.Ferrite.AMR.OctantBWG(2, (-4, -4, 0))
    @test Ferrite.AMR.edge_neighbor(Ferrite.AMR.Ferrite.AMR.OctantBWG(2, (0, 0, 0)), 10, 4) == Ferrite.AMR.Ferrite.AMR.OctantBWG(2, (4, -4, 0))
    @test Ferrite.AMR.edge_neighbor(Ferrite.AMR.Ferrite.AMR.OctantBWG(2, (0, 0, 0)), 11, 4) == Ferrite.AMR.Ferrite.AMR.OctantBWG(2, (-4, 4, 0))
    @test Ferrite.AMR.edge_neighbor(Ferrite.AMR.Ferrite.AMR.OctantBWG(2, (0, 0, 0)), 12, 4) == Ferrite.AMR.Ferrite.AMR.OctantBWG(2, (4, 4, 0))

    @test Ferrite.AMR.edge_neighbor(Ferrite.AMR.Ferrite.AMR.OctantBWG(1, (0, 0, 0)), 1, 4) == Ferrite.AMR.Ferrite.AMR.OctantBWG(1, (0, -8, -8))
    @test Ferrite.AMR.edge_neighbor(Ferrite.AMR.Ferrite.AMR.OctantBWG(1, (0, 0, 0)), 12, 4) == Ferrite.AMR.Ferrite.AMR.OctantBWG(1, (8, 8, 0))

    @test Ferrite.AMR.corner_neighbor(Ferrite.AMR.Ferrite.AMR.OctantBWG(2, (2, 0, 0)), 1, 3) == Ferrite.AMR.Ferrite.AMR.OctantBWG(2, (0, -2, -2))
    @test Ferrite.AMR.corner_neighbor(Ferrite.AMR.Ferrite.AMR.OctantBWG(2, (2, 0, 0)), 4, 3) == Ferrite.AMR.Ferrite.AMR.OctantBWG(2, (4, 2, -2))
    @test Ferrite.AMR.corner_neighbor(Ferrite.AMR.Ferrite.AMR.OctantBWG(2, (2, 0, 0)), 8, 3) == Ferrite.AMR.Ferrite.AMR.OctantBWG(2, (4, 2, 2))

    @test Ferrite.AMR.corner_neighbor(Ferrite.AMR.Ferrite.AMR.OctantBWG(2, (2, 0)), 1, 3) == Ferrite.AMR.Ferrite.AMR.OctantBWG(2, (0, -2))
    @test Ferrite.AMR.corner_neighbor(Ferrite.AMR.Ferrite.AMR.OctantBWG(2, (2, 0)), 2, 3) == Ferrite.AMR.Ferrite.AMR.OctantBWG(2, (4, -2))
    @test Ferrite.AMR.corner_neighbor(Ferrite.AMR.Ferrite.AMR.OctantBWG(2, (2, 0)), 4, 3) == Ferrite.AMR.Ferrite.AMR.OctantBWG(2, (4, 2))
end

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
    Ferrite.AMR.refine!(adaptive_grid.cells[1], adaptive_grid.cells[1].leaves[1])
    @test length(adaptive_grid.cells[1].leaves) == 4
    for (m, octant) in zip(1:4, adaptive_grid.cells[1].leaves)
        @test octant == Ferrite.AMR.OctantBWG(2, 1, m, adaptive_grid.cells[1].b)
    end
    Ferrite.AMR.refine!(adaptive_grid.cells[1], adaptive_grid.cells[1].leaves[1])

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
    Ferrite.AMR.refine!(adaptive_grid.cells[1], adaptive_grid.cells[1].leaves[1])
    Ferrite.AMR.refine!(adaptive_grid.cells[1], adaptive_grid.cells[1].leaves[4])
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
    Ferrite.AMR.refine!(adaptive_grid.cells[3], adaptive_grid.cells[3].leaves[1])
    Ferrite.AMR.refine!(adaptive_grid.cells[5], adaptive_grid.cells[5].leaves[1])

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
        Ferrite.AMR.coarsen_all!(adaptive_grid)
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
    @test_throws BoundsError Ferrite.AMR.transform_corner(adaptive_grid, VertexIndex(1, 1), o, false) == Ferrite.AMR.OctantBWG(0, (-8, -8, -8))
    @test_throws BoundsError Ferrite.AMR.transform_corner(adaptive_grid, VertexIndex(1, 2), o, false) == Ferrite.AMR.OctantBWG(0, (8, -8, -8))
    @test_throws BoundsError Ferrite.AMR.transform_corner(adaptive_grid, VertexIndex(1, 3), o, false) == Ferrite.AMR.OctantBWG(0, (-8, 8, -8))
    @test_throws BoundsError Ferrite.AMR.transform_corner(adaptive_grid, VertexIndex(1, 4), o, false) == Ferrite.AMR.OctantBWG(0, (8, 8, -8))
    @test_throws BoundsError Ferrite.AMR.transform_corner(adaptive_grid, VertexIndex(1, 5), o, false) == Ferrite.AMR.OctantBWG(0, (-8, -8, 8))
    @test_throws BoundsError Ferrite.AMR.transform_corner(adaptive_grid, VertexIndex(1, 6), o, false) == Ferrite.AMR.OctantBWG(0, (8, -8, 8))
    @test_throws BoundsError Ferrite.AMR.transform_corner(adaptive_grid, VertexIndex(1, 7), o, false) == Ferrite.AMR.OctantBWG(0, (-8, 8, 8))
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
    Ferrite.AMR.refine!(adaptive_grid.cells[1], adaptive_grid.cells[1].leaves[1])
    @test length(adaptive_grid.cells[1].leaves) == 8
    for (m, octant) in zip(1:8, adaptive_grid.cells[1].leaves)
        @test octant == Ferrite.AMR.OctantBWG(3, 1, m, adaptive_grid.cells[1].b)
    end
    Ferrite.AMR.refine!(adaptive_grid.cells[1], adaptive_grid.cells[1].leaves[1])
    @test length(adaptive_grid.cells[1].leaves) == 15
    for (m, octant) in zip(1:8, adaptive_grid.cells[1].leaves)
        @test octant == Ferrite.AMR.OctantBWG(3, 2, m, adaptive_grid.cells[1].b)
    end
    adaptive_grid = ForestBWG(grid, 3)
    Ferrite.AMR.refine!(adaptive_grid.cells[1], adaptive_grid.cells[1].leaves[1])
    Ferrite.AMR.refine!(adaptive_grid.cells[1], adaptive_grid.cells[1].leaves[4])
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
        Ferrite.AMR.coarsen_all!(adaptive_grid)
        for tree in adaptive_grid.cells
            @test all(Ferrite.AMR.morton.(tree.leaves, l, 5) == collect(1:(2^(3 * l))))
        end
    end

    # Single
    adaptive_grid = ForestBWG(grid, 3)
    Ferrite.AMR.refine_all!(adaptive_grid, 1)
    Ferrite.AMR.refine!(adaptive_grid.cells[1], adaptive_grid.cells[1].leaves[8])
    transferred_grid = Ferrite.creategrid(adaptive_grid)
    @test unique(transferred_grid.nodes) == transferred_grid.nodes
    # Unrefined grid has 5 ^ dim nodes and the refined element introduces 6 face center, 12 edge center and 1 volume center nodes
    @test length(transferred_grid.nodes) == 5^3 + (6 + 12 + 1)
    # 6 faces and 12 edges of the single refined element induces one hanging node each
    @test length(transferred_grid.conformity_info) == 6 + 12

    adaptive_grid = ForestBWG(grid, 3)
    Ferrite.AMR.refine_all!(adaptive_grid, 1)
    Ferrite.AMR.refine!(adaptive_grid.cells[1], adaptive_grid.cells[1].leaves[1])
    transferred_grid = Ferrite.creategrid(adaptive_grid)
    @test unique(transferred_grid.nodes) == transferred_grid.nodes
    # Unrefined grid has 5 ^ dim nodes and the refined element introduces 6 face center, 12 edge center and 1 volume center nodes
    @test length(transferred_grid.nodes) == 5^3 + (6 + 12 + 1)
    # 6 faces and 12 edges of the single refined element induces one hanging node each - minus 3 faces and 3 edges on the outer boundary
    @test length(transferred_grid.conformity_info) == 6 + 12 - 2 * 3

    adaptive_grid = ForestBWG(grid, 3)
    Ferrite.AMR.refine_all!(adaptive_grid, 1)
    Ferrite.AMR.refine!(adaptive_grid.cells[8], adaptive_grid.cells[8].leaves[8])
    transferred_grid = Ferrite.creategrid(adaptive_grid)
    @test unique(transferred_grid.nodes) == transferred_grid.nodes
    # Unrefined grid has 5 ^ dim nodes and the refined element introduces 6 face center, 12 edge center and 1 volume center nodes
    @test length(transferred_grid.nodes) == 5^3 + (6 + 12 + 1)
    # 6 faces and 12 edges of the single refined element induces one hanging node each - minus 3 faces and 3 edges on the outer boundary
    @test length(transferred_grid.conformity_info) == 6 + 12 - 2 * 3

    adaptive_grid = ForestBWG(grid, 3)
    Ferrite.AMR.refine_all!(adaptive_grid, 1)
    Ferrite.AMR.refine!(adaptive_grid.cells[8], adaptive_grid.cells[8].leaves[1])
    transferred_grid = Ferrite.creategrid(adaptive_grid)
    # Unrefined grid has 5 ^ dim nodes and the refined element introduces 6 face center, 12 edge center and 1 volume center nodes
    @test length(transferred_grid.nodes) == 5^3 + (6 + 12 + 1)
    # 6 faces and 12 edges of the single refined element induces one hanging node each
    @test length(transferred_grid.conformity_info) == 6 + 12

    # Combined
    adaptive_grid = ForestBWG(grid, 3)
    Ferrite.AMR.refine_all!(adaptive_grid, 1)
    Ferrite.AMR.refine!(adaptive_grid.cells[1], adaptive_grid.cells[1].leaves[8])
    Ferrite.AMR.refine!(adaptive_grid.cells[1], adaptive_grid.cells[1].leaves[1])
    transferred_grid = Ferrite.creategrid(adaptive_grid)
    @test unique(transferred_grid.nodes) == transferred_grid.nodes
    @test length(transferred_grid.nodes) == 5^3 + 2 * (6 + 12 + 1)
    @test length(transferred_grid.conformity_info) == 2 * (6 + 12) - 2 * 3

    adaptive_grid = ForestBWG(grid, 3)
    Ferrite.AMR.refine_all!(adaptive_grid, 1)
    Ferrite.AMR.refine!(adaptive_grid.cells[8], adaptive_grid.cells[8].leaves[8])
    Ferrite.AMR.refine!(adaptive_grid.cells[8], adaptive_grid.cells[8].leaves[1])
    transferred_grid = Ferrite.creategrid(adaptive_grid)
    @test unique(transferred_grid.nodes) == transferred_grid.nodes
    @test length(transferred_grid.nodes) == 5^3 + 2 * (6 + 12 + 1)
    @test length(transferred_grid.conformity_info) == 2 * (6 + 12) - 2 * 3

    # Combined
    adaptive_grid = ForestBWG(grid, 3)
    Ferrite.AMR.refine_all!(adaptive_grid, 1)
    Ferrite.AMR.refine!(adaptive_grid.cells[1], adaptive_grid.cells[1].leaves[8])
    Ferrite.AMR.refine!(adaptive_grid.cells[1], adaptive_grid.cells[1].leaves[1])
    Ferrite.AMR.refine!(adaptive_grid.cells[8], adaptive_grid.cells[8].leaves[8])
    Ferrite.AMR.refine!(adaptive_grid.cells[8], adaptive_grid.cells[8].leaves[1])
    transferred_grid = Ferrite.creategrid(adaptive_grid)
    @test unique(transferred_grid.nodes) == transferred_grid.nodes
    @test length(transferred_grid.nodes) == 5^3 + 4 * (6 + 12 + 1)
    @test length(transferred_grid.conformity_info) == 4 * (6 + 12) - 2 * 3 - 2 * 3

    # Combined and not rotated
    adaptive_grid = ForestBWG(grid, 3)
    Ferrite.AMR.refine_all!(adaptive_grid, 1)
    Ferrite.AMR.refine!(adaptive_grid.cells[1], adaptive_grid.cells[1].leaves[8])
    Ferrite.AMR.refine!(adaptive_grid.cells[1], adaptive_grid.cells[1].leaves[1])
    Ferrite.AMR.refine!(adaptive_grid.cells[6], adaptive_grid.cells[6].leaves[6])
    Ferrite.AMR.refine!(adaptive_grid.cells[6], adaptive_grid.cells[6].leaves[3])
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
    Ferrite.AMR.refine!(adaptive_grid.cells[1], adaptive_grid.cells[1].leaves[8])
    Ferrite.AMR.refine!(adaptive_grid.cells[1], adaptive_grid.cells[1].leaves[1])
    Ferrite.AMR.refine!(adaptive_grid.cells[7], adaptive_grid.cells[7].leaves[6])
    Ferrite.AMR.refine!(adaptive_grid.cells[7], adaptive_grid.cells[7].leaves[3])
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
    Ferrite.AMR.refine!(adaptive_grid.cells[2], adaptive_grid.cells[2].leaves[1])
    Ferrite.AMR.refine!(adaptive_grid.cells[2], adaptive_grid.cells[2].leaves[3])
    @test adaptive_grid.cells[2].leaves[3 + 4] == Ferrite.AMR.OctantBWG(2, (0, 4, 2))
    @test Ferrite.AMR.transform_facet(adaptive_grid, FacetIndex(1, 2), adaptive_grid.cells[2].leaves[3 + 4]) == Ferrite.AMR.OctantBWG(2, (8, 4, 2))
    # (b) Rotate elements topologically
    grid.cells[1] = Hexahedron((grid.cells[1].nodes[2], grid.cells[1].nodes[3], grid.cells[1].nodes[4], grid.cells[1].nodes[1], grid.cells[1].nodes[6], grid.cells[1].nodes[7], grid.cells[1].nodes[8], grid.cells[1].nodes[5]))
    grid.cells[2] = Hexahedron((grid.cells[2].nodes[4], grid.cells[2].nodes[1], grid.cells[2].nodes[2], grid.cells[2].nodes[3], grid.cells[2].nodes[8], grid.cells[2].nodes[5], grid.cells[2].nodes[6], grid.cells[2].nodes[7]))
    # grid.cells[2] = Hexahedron((grid.cells[2].nodes[1], grid.cells[2].nodes[3], grid.cells[2].nodes[4], grid.cells[2].nodes[8], grid.cells[2].nodes[6], grid.cells[2].nodes[2], grid.cells[2].nodes[7], grid.cells[2].nodes[5])) How to rotate along diagonal? :)
    adaptive_grid = ForestBWG(grid, 3)
    Ferrite.AMR.refine!(adaptive_grid.cells[2], adaptive_grid.cells[2].leaves[1])
    Ferrite.AMR.refine!(adaptive_grid.cells[2], adaptive_grid.cells[2].leaves[1])
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
    Ferrite.AMR.refine!(adaptive_grid.cells[1], adaptive_grid.cells[1].leaves[2])
    Ferrite.AMR.refine!(adaptive_grid.cells[1], adaptive_grid.cells[1].leaves[6])
    Ferrite.AMR.refine!(adaptive_grid.cells[1], adaptive_grid.cells[1].leaves[6])
    balanced = Ferrite.AMR.balancetree(adaptive_grid.cells[1])
    @test length(balanced.leaves) == 16

    #more complex non-conformity level 3 and 4 that needs to be balanced
    adaptive_grid = ForestBWG(grid, 5)
    Ferrite.AMR.refine_all!(adaptive_grid, 1)
    Ferrite.AMR.refine!(adaptive_grid.cells[1], adaptive_grid.cells[1].leaves[2])
    Ferrite.AMR.refine!(adaptive_grid.cells[1], adaptive_grid.cells[1].leaves[4])
    Ferrite.AMR.refine!(adaptive_grid.cells[1], adaptive_grid.cells[1].leaves[7])
    Ferrite.AMR.refine!(adaptive_grid.cells[1], adaptive_grid.cells[1].leaves[12])
    Ferrite.AMR.refine!(adaptive_grid.cells[1], adaptive_grid.cells[1].leaves[12])
    Ferrite.AMR.refine!(adaptive_grid.cells[1], adaptive_grid.cells[1].leaves[15])
    Ferrite.AMR.refine!(adaptive_grid.cells[1], adaptive_grid.cells[1].leaves[16])
    balanced = Ferrite.AMR.balancetree(adaptive_grid.cells[1])
    @test length(balanced.leaves) == 64

    grid = generate_grid(Quadrilateral, (2, 1))
    adaptive_grid = ForestBWG(grid, 2)
    Ferrite.AMR.refine!(adaptive_grid.cells[1], adaptive_grid.cells[1].leaves[1])
    Ferrite.AMR.refine!(adaptive_grid.cells[1], adaptive_grid.cells[1].leaves[2])
    Ferrite.AMR.balanceforest!(adaptive_grid)
    @test Ferrite.AMR.getncells(adaptive_grid) == 11

    grid = generate_grid(Quadrilateral, (2, 2))
    adaptive_grid = ForestBWG(grid, 2)
    Ferrite.AMR.refine!(adaptive_grid.cells[1], adaptive_grid.cells[1].leaves[1])
    Ferrite.AMR.refine!(adaptive_grid.cells[1], adaptive_grid.cells[1].leaves[4])
    Ferrite.AMR.balanceforest!(adaptive_grid)
    @test Ferrite.AMR.getncells(adaptive_grid) == 19

    # 2D example with balancing over a corner connection that is not within the topology tables
    grid = generate_grid(Quadrilateral, (2, 1))
    adaptive_grid = ForestBWG(grid, 3)
    Ferrite.AMR.refine!(adaptive_grid.cells[1], adaptive_grid.cells[1].leaves[1])
    Ferrite.AMR.refine!(adaptive_grid.cells[1], adaptive_grid.cells[1].leaves[2])
    Ferrite.AMR.refine!(adaptive_grid.cells[1], adaptive_grid.cells[1].leaves[5])
    Ferrite.AMR.balanceforest!(adaptive_grid)
    @test Ferrite.AMR.getncells(adaptive_grid) == 23

    #corner balance case but rotated
    grid = generate_grid(Quadrilateral, (2, 1))
    grid.cells[1] = Quadrilateral((grid.cells[1].nodes[2], grid.cells[1].nodes[3], grid.cells[1].nodes[4], grid.cells[1].nodes[1]))
    adaptive_grid = ForestBWG(grid, 3)
    Ferrite.AMR.refine!(adaptive_grid.cells[1], adaptive_grid.cells[1].leaves[1])
    Ferrite.AMR.refine!(adaptive_grid.cells[1], adaptive_grid.cells[1].leaves[1])
    Ferrite.AMR.refine!(adaptive_grid.cells[1], adaptive_grid.cells[1].leaves[2])
    Ferrite.AMR.balanceforest!(adaptive_grid)
    @test Ferrite.AMR.getncells(adaptive_grid) == 23

    # 3D case intra tree simple test, non conformity level 2
    grid = generate_grid(Hexahedron, (1, 1, 1))
    adaptive_grid = ForestBWG(grid, 3)
    Ferrite.AMR.refine_all!(adaptive_grid, 1)
    Ferrite.AMR.refine!(adaptive_grid.cells[1], adaptive_grid.cells[1].leaves[2])
    Ferrite.AMR.refine!(adaptive_grid.cells[1], adaptive_grid.cells[1].leaves[6])
    Ferrite.AMR.refine!(adaptive_grid.cells[1], adaptive_grid.cells[1].leaves[6])
    balanced = Ferrite.AMR.balancetree(adaptive_grid.cells[1])
    @test length(balanced.leaves) == 43

    #3D case intra tree non conformity level 3 at two different places
    adaptive_grid = ForestBWG(grid, 4)
    Ferrite.AMR.refine_all!(adaptive_grid, 1)
    Ferrite.AMR.refine!(adaptive_grid.cells[1], adaptive_grid.cells[1].leaves[2])
    Ferrite.AMR.refine!(adaptive_grid.cells[1], adaptive_grid.cells[1].leaves[4])
    Ferrite.AMR.refine!(adaptive_grid.cells[1], adaptive_grid.cells[1].leaves[7])
    Ferrite.AMR.refine!(adaptive_grid.cells[1], adaptive_grid.cells[1].leaves[12])
    Ferrite.AMR.refine!(adaptive_grid.cells[1], adaptive_grid.cells[1].leaves[28])
    Ferrite.AMR.refine!(adaptive_grid.cells[1], adaptive_grid.cells[1].leaves[29])
    Ferrite.AMR.refine!(adaptive_grid.cells[1], adaptive_grid.cells[1].leaves[37])
    Ferrite.AMR.refine!(adaptive_grid.cells[1], adaptive_grid.cells[1].leaves[39])
    balanced = Ferrite.AMR.balancetree(adaptive_grid.cells[1])
    @test length(balanced.leaves) == 127

    #3D case inter tree non conformity level 3 at two different places
    grid = generate_grid(Hexahedron, (2, 2, 2))
    adaptive_grid = ForestBWG(grid, 4)
    Ferrite.AMR.refine!(adaptive_grid.cells[1], adaptive_grid.cells[1].leaves[1])
    Ferrite.AMR.refine!(adaptive_grid.cells[1], adaptive_grid.cells[1].leaves[2])
    Ferrite.AMR.refine!(adaptive_grid.cells[1], adaptive_grid.cells[1].leaves[4])
    #Ferrite.AMR.refine!(adaptive_grid.cells[1],adaptive_grid.cells[1].leaves[7])
    Ferrite.AMR.refine!(adaptive_grid.cells[7], adaptive_grid.cells[7].leaves[1])
    Ferrite.AMR.refine!(adaptive_grid.cells[7], adaptive_grid.cells[7].leaves[1])
    #Ferrite.AMR.refine!(adaptive_grid.cells[7],adaptive_grid.cells[7].leaves[1])
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
    Ferrite.AMR.refine!(adaptive_grid.cells[1], adaptive_grid.cells[1].leaves[1])
    Ferrite.AMR.refine!(adaptive_grid.cells[1], adaptive_grid.cells[1].leaves[2])
    Ferrite.AMR.refine!(adaptive_grid.cells[1], adaptive_grid.cells[1].leaves[4])
    #Ferrite.AMR.refine!(adaptive_grid.cells[1],adaptive_grid.cells[1].leaves[7])
    Ferrite.AMR.refine!(adaptive_grid.cells[7], adaptive_grid.cells[7].leaves[1])
    Ferrite.AMR.refine!(adaptive_grid.cells[7], adaptive_grid.cells[7].leaves[1])
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
    Ferrite.AMR.refine!(adaptive_grid.cells[4], adaptive_grid.cells[4].leaves[1])
    Ferrite.AMR.balanceforest!(adaptive_grid)
    @test Ferrite.AMR.getncells(adaptive_grid) == 15

    #yet another edge balancing case
    grid = generate_grid(Hexahedron, (2, 2, 2))
    adaptive_grid = ForestBWG(grid, 3)
    Ferrite.AMR.refine!(adaptive_grid.cells[4], adaptive_grid.cells[4].leaves[1])
    Ferrite.AMR.balanceforest!(adaptive_grid)
    Ferrite.AMR.refine!(adaptive_grid.cells[4], adaptive_grid.cells[4].leaves[1])
    Ferrite.AMR.balanceforest!(adaptive_grid)
    @test Ferrite.AMR.getncells(adaptive_grid) == 43

    #yet another edge balancing case
    grid = generate_grid(Hexahedron, (2, 2, 2))
    adaptive_grid = ForestBWG(grid, 3)
    Ferrite.AMR.refine!(adaptive_grid.cells[4], adaptive_grid.cells[4].leaves[1])
    Ferrite.AMR.balanceforest!(adaptive_grid)
    Ferrite.AMR.refine!(adaptive_grid.cells[4], adaptive_grid.cells[4].leaves[1])
    Ferrite.AMR.balanceforest!(adaptive_grid)
    Ferrite.AMR.refine!(adaptive_grid.cells[3], adaptive_grid.cells[3].leaves[2])
    Ferrite.AMR.refine!(adaptive_grid.cells[4], adaptive_grid.cells[4].leaves[10])
    Ferrite.AMR.refine!(adaptive_grid.cells[4], adaptive_grid.cells[4].leaves[3])
    Ferrite.AMR.balanceforest!(adaptive_grid)
    @test Ferrite.AMR.getncells(adaptive_grid) == 71

    #yet another edge balancing case
    grid = generate_grid(Hexahedron, (2, 2, 2))
    adaptive_grid = ForestBWG(grid, 3)
    Ferrite.AMR.refine!(adaptive_grid.cells[4], adaptive_grid.cells[4].leaves[1])
    Ferrite.AMR.balanceforest!(adaptive_grid)
    Ferrite.AMR.refine!(adaptive_grid.cells[4], adaptive_grid.cells[4].leaves[1])
    Ferrite.AMR.balanceforest!(adaptive_grid)
    Ferrite.AMR.refine!(adaptive_grid.cells[4], adaptive_grid.cells[4].leaves[7])
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
            Ferrite.AMR.refine!(t, only(filter(pred, t.leaves)))
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
    Ferrite.AMR.refine!(forest.cells[2], forest.cells[2].leaves[1])
    refine_towards_and_balance!(forest, 2, 2, o -> Int(o.xyz[1]) == 0 && Int(o.xyz[2]) == m ÷ 2 && Int(o.xyz[3]) == m ÷ 2)
    @test count_unbalanced_contacts(forest) == 0

    # 3D corner in the middle of the shared macro edge (diagonal tree is an exclusive edge neighbour)
    forest = ForestBWG(generate_grid(Hexahedron, (2, 2, 1)), 4)
    Ferrite.AMR.refine!(forest.cells[4], forest.cells[4].leaves[1])
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
    Ferrite.AMR.refine!(forest.cells[2], forest.cells[2].leaves[1])
    refine_towards_and_balance!(forest, 2, 2, o -> Int(o.xyz[1]) == 0 && Int(o.xyz[2]) + Int(Ferrite.AMR._compute_size(forest.cells[2].b, o.l)) == m ÷ 2)
    @test count_unbalanced_contacts(forest) == 0

    # regression: macro-corner refinement (handled via vertex_vertex_neighbor) stays balanced
    forest = ForestBWG(generate_grid(Hexahedron, (2, 2, 2)), 4)
    for _ in 1:3
        t = forest.cells[4]
        leaf = only(filter(o -> o.xyz[1] == 0 && o.xyz[2] == 0 && Int(o.xyz[3]) + Int(Ferrite.AMR._compute_size(t.b, o.l)) == m, t.leaves))
        Ferrite.AMR.refine!(t, leaf)
    end
    Ferrite.AMR.balanceforest!(forest)
    @test count_unbalanced_contacts(forest) == 0

    # regression: 2D through-face corner with no macro corner connection at all
    forest = ForestBWG(generate_grid(Quadrilateral, (2, 1)), 4)
    Ferrite.AMR.refine!(forest.cells[2], forest.cells[2].leaves[1])
    refine_towards_and_balance!(forest, 2, 2, o -> Int(o.xyz[1]) == 0 && Int(o.xyz[2]) == m ÷ 2)
    @test count_unbalanced_contacts(forest) == 0
end

@testset "Materializing Grid" begin
    #################################################
    ############ structured 2D examples #############
    #################################################

    # 2D case with a single tree
    grid = generate_grid(Quadrilateral, (1, 1))
    adaptive_grid = ForestBWG(grid, 3)
    Ferrite.AMR.refine_all!(adaptive_grid, 1)
    Ferrite.AMR.refine!(adaptive_grid.cells[1], adaptive_grid.cells[1].leaves[1])
    Ferrite.AMR.refine!(adaptive_grid.cells[1], adaptive_grid.cells[1].leaves[1])
    transferred_grid = Ferrite.AMR.creategrid(adaptive_grid)
    @test length(transferred_grid.cells) == 10
    @test length(transferred_grid.nodes) == 19
    @test unique(transferred_grid.nodes) == transferred_grid.nodes

    #2D case with four trees and somewhat refinement pattern
    grid = generate_grid(Quadrilateral, (2, 2))
    adaptive_grid = ForestBWG(grid, 3)
    Ferrite.AMR.refine_all!(adaptive_grid, 1)
    Ferrite.AMR.refine!(adaptive_grid.cells[1], adaptive_grid.cells[1].leaves[1])
    Ferrite.AMR.refine!(adaptive_grid.cells[1], adaptive_grid.cells[1].leaves[1])
    transferred_grid = Ferrite.AMR.creategrid(adaptive_grid)
    @test length(transferred_grid.cells) == 22
    @test length(transferred_grid.nodes) == 35
    @test unique(transferred_grid.nodes) == transferred_grid.nodes

    #more random refinement
    grid = generate_grid(Quadrilateral, (3, 3))
    adaptive_grid = ForestBWG(grid, 3)
    Ferrite.AMR.refine!(adaptive_grid.cells[1], adaptive_grid.cells[1].leaves[1])
    Ferrite.AMR.refine!(adaptive_grid.cells[1], adaptive_grid.cells[1].leaves[1])
    Ferrite.AMR.refine!(adaptive_grid.cells[1], adaptive_grid.cells[1].leaves[1])
    Ferrite.AMR.refine!(adaptive_grid.cells[3], adaptive_grid.cells[3].leaves[1])
    Ferrite.AMR.refine!(adaptive_grid.cells[3], adaptive_grid.cells[3].leaves[2])
    Ferrite.AMR.refine!(adaptive_grid.cells[3], adaptive_grid.cells[3].leaves[3])
    Ferrite.AMR.refine!(adaptive_grid.cells[7], adaptive_grid.cells[7].leaves[1])
    Ferrite.AMR.refine!(adaptive_grid.cells[7], adaptive_grid.cells[7].leaves[3])
    Ferrite.AMR.refine!(adaptive_grid.cells[7], adaptive_grid.cells[7].leaves[5])
    Ferrite.AMR.refine!(adaptive_grid.cells[9], adaptive_grid.cells[9].leaves[end])
    Ferrite.AMR.refine!(adaptive_grid.cells[9], adaptive_grid.cells[9].leaves[end])
    Ferrite.AMR.refine!(adaptive_grid.cells[9], adaptive_grid.cells[9].leaves[end])
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
    Ferrite.AMR.refine!(adaptive_grid.cells[1], adaptive_grid.cells[1].leaves[1])
    Ferrite.AMR.refine!(adaptive_grid.cells[1], adaptive_grid.cells[1].leaves[1])
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
        Ferrite.AMR.refine!(forest.cells[1], forest.cells[1].leaves[1])
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
    Ferrite.AMR.refine!(forest.cells[1], forest.cells[1].leaves[1])
    Ferrite.AMR.refine!(forest.cells[1], forest.cells[1].leaves[1])
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

@testset "ConformityConstraint subdomain guard" begin
    # Subdomains (fields not living on the whole grid) are not supported with conformity
    # constraints yet: the hanging/master node lookup assumes every vertex carries a dof.
    # Requesting one must throw a descriptive error instead of crashing in close!(ch) with
    # `AffineConstraint(0, ...)` (BoundsError at isconstrained[0]).
    grid = generate_grid(Quadrilateral, (2, 2))
    forest = ForestBWG(grid, 3)
    Ferrite.AMR.refine!(forest.cells[1], forest.cells[1].leaves[1])
    Ferrite.AMR.balanceforest!(forest)
    transferred_grid = Ferrite.AMR.creategrid(forest)
    ip = Lagrange{RefQuadrilateral, 1}()

    # partial coverage through a plain SubDofHandler
    dh = DofHandler(transferred_grid)
    sdh = SubDofHandler(dh, Ferrite.OrderedSet(1:(getncells(transferred_grid) - 1)))
    add!(sdh, :u, ip)
    close!(dh)
    ch = ConstraintHandler(dh)
    @test_throws ArgumentError add!(ch, ConformityConstraint(:u))

    # partial coverage through the documented L2Projector set kwarg
    @test_throws ArgumentError L2Projector(ip, transferred_grid; set = Ferrite.OrderedSet(1:(getncells(transferred_grid) - 1)))

    # full coverage split over two SubDofHandlers is legitimate and must keep working
    dh2 = DofHandler(transferred_grid)
    half = getncells(transferred_grid) ÷ 2
    sdh1 = SubDofHandler(dh2, Ferrite.OrderedSet(1:half))
    add!(sdh1, :u, ip)
    sdh2 = SubDofHandler(dh2, Ferrite.OrderedSet((half + 1):getncells(transferred_grid)))
    add!(sdh2, :u, ip)
    close!(dh2)
    ch2 = ConstraintHandler(dh2)
    add!(ch2, ConformityConstraint(:u))
    close!(ch2)
    @test length(ch2.prescribed_dofs) == length(transferred_grid.conformity_info)
end

@testset "hanging nodes" begin
    #Easy Intraoctree
    grid = generate_grid(Hexahedron, (1, 1, 1))
    adaptive_grid = ForestBWG(grid, 3)
    Ferrite.AMR.refine_all!(adaptive_grid, 1)
    Ferrite.AMR.refine!(adaptive_grid.cells[1], adaptive_grid.cells[1].leaves[1])
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
    Ferrite.AMR.refine!(adaptive_grid.cells[1], adaptive_grid.cells[1].leaves[1])
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
    Ferrite.AMR.refine!(adaptive_grid.cells[1], adaptive_grid.cells[1].leaves[1])
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
    Ferrite.AMR.refine!(adaptive_grid.cells[2], adaptive_grid.cells[2].leaves[1])
    transferred_grid_rotated = Ferrite.AMR.creategrid(adaptive_grid)
    @test Set(transferred_grid_rotated.conformity_info[10]) == Set([4, 9])
    @test Set(transferred_grid_rotated.conformity_info[11]) == Set([2, 4])
    @test length(transferred_grid_rotated.conformity_info) == 2

    # multiple corner connections in 2D by disc discretization
    grid = generate_simple_disc_grid(Quadrilateral, 10)
    adaptive_grid = ForestBWG(grid, 3)
    @test getncells(adaptive_grid) == 10
    Ferrite.refine!(adaptive_grid.cells[1], adaptive_grid.cells[1].leaves[1])
    Ferrite.refine!(adaptive_grid.cells[1], adaptive_grid.cells[1].leaves[3])
    @test getncells(adaptive_grid) == 16
    Ferrite.balanceforest!(adaptive_grid)
    @test getncells(adaptive_grid) == 9 * 4 + 3 + 4

    # multiple corner connections in 3D by cylinder discretization
    grid = generate_simple_disc_grid(Hexahedron, 10)
    adaptive_grid = ForestBWG(grid, 3)
    @test getncells(adaptive_grid) == 10
    Ferrite.refine!(adaptive_grid.cells[1], adaptive_grid.cells[1].leaves[1])
    @test getncells(adaptive_grid) == 17
    Ferrite.refine!(adaptive_grid.cells[1], adaptive_grid.cells[1].leaves[3])
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
    Ferrite.AMR.refine!(forest.cells[2], forest.cells[2].leaves[1])
    tg = Ferrite.AMR.creategrid(forest)
    skel = Ferrite.facetskeleton(forest)
    @test skeleton_canonical_2d(tg, skel) == skeleton_groundtruth_2d(tg)
    check_skeleton_geometry(tg, skel, 2)

    # 3D intra-octree hanging (cf. "hanging nodes" testset)
    forest = ForestBWG(generate_grid(Hexahedron, (1, 1, 1)), 3)
    Ferrite.AMR.refine_all!(forest, 1)
    Ferrite.AMR.refine!(forest.cells[1], forest.cells[1].leaves[1])
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

@testset "InterfaceValues on facet skeleton" begin
    # Every skeleton pair — conforming, hanging, across trees, rotated macro elements —
    # must reinit! an InterfaceValues via AffineInterfaceTransformation such that the two
    # sides' quadrature points coincide physically, and a globally linear field (whose
    # nodal values automatically satisfy the hanging midpoint constraints) is continuous
    # with continuous gradient across the interface.
    function check_interfacevalues(forest, refshape)
        dim = refshape === RefQuadrilateral ? 2 : 3
        grid = Ferrite.AMR.creategrid(forest)
        skel = Ferrite.facetskeleton(forest)
        cells = Ferrite.getcells(grid)
        iv = InterfaceValues(FacetQuadratureRule{refshape}(2), Lagrange{refshape, 1}())
        u_lin(x) = 1.0 + 2.0 * x[1] - 3.0 * x[dim] + (dim == 3 ? 0.5 * x[2] : 0.0)
        ∇u_lin = Tensors.gradient(u_lin, zero(Vec{dim}))
        for (fiA, fiB) in skel
            cA, fA = fiA[1], fiA[2]
            cB, fB = fiB[1], fiB[2]
            coordsA = getcoordinates(grid, cA)
            coordsB = getcoordinates(grid, cB)
            trans = Ferrite.AffineInterfaceTransformation(cells[cA], coordsA, fA, cells[cB], coordsB, fB)
            reinit!(iv, cells[cA], coordsA, fA, cells[cB], coordsB, fB, trans)
            ue = [u_lin.(coordsA); u_lin.(coordsB)]
            for qp in 1:getnquadpoints(iv)
                xh = spatial_coordinate(iv, qp, coordsA, coordsB; here = true)
                xt = spatial_coordinate(iv, qp, coordsA, coordsB; here = false)
                @test xh ≈ xt
                @test function_value(iv, qp, ue; here = true) ≈ u_lin(xh)
                @test function_value(iv, qp, ue; here = false) ≈ u_lin(xh)
                @test function_gradient(iv, qp, ue; here = true) ≈ ∇u_lin
                @test function_gradient(iv, qp, ue; here = false) ≈ ∇u_lin
                # opposing outward normals (straight facets for these grids)
                @test getnormal(iv, qp; here = true) ≈ -getnormal(iv, qp; here = false)
            end
        end
        return
    end

    # 2D multi-level, intra- + inter-tree, conforming + hanging
    forest = ForestBWG(generate_grid(Quadrilateral, (2, 2)), 5)
    Ferrite.refine!(forest, [1])
    Ferrite.balanceforest!(forest)
    Ferrite.refine!(forest, [1, 2, 7])
    Ferrite.balanceforest!(forest)
    check_interfacevalues(forest, RefQuadrilateral)

    # 2D rotated macro element
    grid = generate_grid(Quadrilateral, (2, 2))
    grid.cells[2] = Quadrilateral((grid.cells[2].nodes[2], grid.cells[2].nodes[3], grid.cells[2].nodes[4], grid.cells[2].nodes[1]))
    forest = ForestBWG(grid, 3)
    Ferrite.AMR.refine!(forest.cells[2], forest.cells[2].leaves[1])
    check_interfacevalues(forest, RefQuadrilateral)

    # 3D multi-level, intra- + inter-tree
    forest = ForestBWG(generate_grid(Hexahedron, (2, 2, 2)), 4)
    Ferrite.refine!(forest, [1])
    Ferrite.balanceforest!(forest)
    Ferrite.refine!(forest, [1])
    Ferrite.balanceforest!(forest)
    check_interfacevalues(forest, RefHexahedron)

    # 3D rotated macro element (cf. "hanging nodes" testset)
    grid = generate_grid(Hexahedron, (2, 2, 2))
    grid.cells[1] = Hexahedron((grid.cells[1].nodes[2], grid.cells[1].nodes[3], grid.cells[1].nodes[4], grid.cells[1].nodes[1], grid.cells[1].nodes[6], grid.cells[1].nodes[7], grid.cells[1].nodes[8], grid.cells[1].nodes[5]))
    grid.cells[1] = Hexahedron((grid.cells[1].nodes[2], grid.cells[1].nodes[3], grid.cells[1].nodes[4], grid.cells[1].nodes[1], grid.cells[1].nodes[6], grid.cells[1].nodes[7], grid.cells[1].nodes[8], grid.cells[1].nodes[5]))
    forest = ForestBWG(grid, 3)
    Ferrite.AMR.refine!(forest.cells[1], forest.cells[1].leaves[1])
    check_interfacevalues(forest, RefHexahedron)
end

@testset "ForestBWG accessors and error paths" begin
    grid = generate_grid(Quadrilateral, (2, 2))
    forest = ForestBWG(grid, 3)
    Ferrite.AMR.refine_all!(forest, 1)

    # scalar getcells must warn about its slow dispatch and agree with the vector variant
    leaves = getcells(forest)
    c7 = @test_logs (:warn, r"Slow dispatch") getcells(forest, 7)
    @test c7 == leaves[7] == forest.cells[2].leaves[3]
    c1 = @test_logs (:warn, r"Slow dispatch") getcells(forest, 1)
    @test c1 == leaves[1] == forest.cells[1].leaves[1]

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
    Ferrite.AMR.coarsen!(tree, tree.leaves[2])
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
    # reproduces coarsen_all!
    f1 = ForestBWG(generate_grid(CT, ntuple(_ -> 1, dim)), 4)
    Ferrite.AMR.refine_all!(f1, 1)
    Ferrite.AMR.refine_all!(f1, 2)
    f2 = deepcopy(f1)
    Ferrite.AMR.coarsen_all!(f1)
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
        Ferrite.AMR.refine!(forest.cells[refine_tree], forest.cells[refine_tree].leaves[1])
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

@testset "ConformityConstraint fallbacks and vector fields" begin
    # warn-and-skip on a conforming grid
    grid = generate_grid(Quadrilateral, (2, 2))
    dh = DofHandler(grid)
    add!(dh, :u, Lagrange{RefQuadrilateral, 1}())
    close!(dh)
    ch = ConstraintHandler(dh)
    @test_logs (:warn, r"conforming grid") add!(ch, ConformityConstraint(:u))

    # a non-conforming grid with two hanging nodes
    forest = ForestBWG(generate_grid(Quadrilateral, (2, 2)), 3)
    Ferrite.refine!(forest, [1])
    Ferrite.balanceforest!(forest)
    ncgrid = Ferrite.AMR.creategrid(forest)
    @test Ferrite.get_coordinate_type(ncgrid) == Vec{2, Float64}
    nhanging = length(ncgrid.conformity_info)
    @test nhanging == 2

    # only linear Lagrange (and vectorizations) are supported
    dh2 = DofHandler(ncgrid)
    add!(dh2, :u, Lagrange{RefQuadrilateral, 2}())
    close!(dh2)
    ch2 = ConstraintHandler(dh2)
    @test_throws ArgumentError add!(ch2, ConformityConstraint(:u))

    # vector-valued field: one affine constraint per component per hanging node
    dhv = DofHandler(ncgrid)
    add!(dhv, :u, Lagrange{RefQuadrilateral, 1}()^2)
    close!(dhv)
    chv = ConstraintHandler(dhv)
    add!(chv, ConformityConstraint(:u))
    close!(chv)
    @test length(chv.prescribed_dofs) == 2 * nhanging
    pdofs = Set(chv.prescribed_dofs)
    for (i, d) in pairs(chv.prescribed_dofs)
        dc = chv.dofcoefficients[i]
        # edge midpoint: average of the 2 edge endpoints, componentwise
        @test dc !== nothing && length(dc) == 2
        @test all(p -> p.second ≈ 0.5, dc)
        # the sibling component dof of the same hanging vertex is constrained as well
        partner = isodd(d) ? d + 1 : d - 1
        @test partner in pdofs
    end
end

@testset "ConformityConstraint multiple fields" begin
    # unit test for the per-field coverage guard: a hanging vertex (or one of its masters)
    # that maps to dof 0 (field absent there) must be rejected.
    ci = Dict(3 => [1, 2])
    @test Ferrite.AMR._has_uncovered_hanging_vertex(ci, [10, 11, 12]) == false
    @test Ferrite.AMR._has_uncovered_hanging_vertex(ci, [10, 11, 0]) == true  # hanging uncovered
    @test Ferrite.AMR._has_uncovered_hanging_vertex(ci, [0, 11, 12]) == true  # master uncovered

    # a non-conforming grid with two hanging nodes
    forest = ForestBWG(generate_grid(Quadrilateral, (2, 2)), 3)
    Ferrite.refine!(forest, [1])
    Ferrite.balanceforest!(forest)
    ncgrid = Ferrite.AMR.creategrid(forest)
    nhanging = length(ncgrid.conformity_info)
    @test nhanging == 2
    ip = Lagrange{RefQuadrilateral, 1}()

    # collect the global dofs belonging to a given field
    fielddofs(dh, name) = mapreduce(union!, dh.subdofhandlers; init = Set{Int}()) do sdh
        name in sdh.field_names || return Set{Int}()
        r = Ferrite.dof_range(sdh, name)
        Set{Int}(d for c in sdh.cellset for d in celldofs(dh, c)[r])
    end

    # two scalar fields on the whole grid: each field is constrained independently and only
    # against dofs of its own field (regression for the global-vs-local field index).
    dh = DofHandler(ncgrid)
    add!(dh, :u, ip)
    add!(dh, :p, ip)
    close!(dh)
    udofs = fielddofs(dh, :u)
    pdofs = fielddofs(dh, :p)
    @test isempty(intersect(udofs, pdofs))

    ch = ConstraintHandler(dh)
    add!(ch, ConformityConstraint(:u))
    add!(ch, ConformityConstraint(:p))
    close!(ch)
    @test length(ch.prescribed_dofs) == 2 * nhanging
    @test count(in(udofs), ch.prescribed_dofs) == nhanging
    @test count(in(pdofs), ch.prescribed_dofs) == nhanging
    # every constrained dof and all its masters live in the same field
    for (i, d) in pairs(ch.prescribed_dofs)
        fieldset = d in udofs ? udofs : pdofs
        @test all(p -> p.first in fieldset, ch.dofcoefficients[i])
    end

    # constraining only one field must leave the other field's dofs untouched
    chu = ConstraintHandler(dh)
    add!(chu, ConformityConstraint(:u))
    close!(chu)
    @test length(chu.prescribed_dofs) == nhanging
    @test all(in(udofs), chu.prescribed_dofs)
    @test isempty(intersect(Set(chu.prescribed_dofs), pdofs))

    # Taylor–Hood-like mix: vector displacement + scalar pressure, both linear
    dhth = DofHandler(ncgrid)
    add!(dhth, :u, ip^2)
    add!(dhth, :p, ip)
    close!(dhth)
    chth = ConstraintHandler(dhth)
    add!(chth, ConformityConstraint(:u))
    add!(chth, ConformityConstraint(:p))
    close!(chth)
    # 2 displacement components + 1 pressure per hanging node
    @test length(chth.prescribed_dofs) == 3 * nhanging
    @test count(in(fielddofs(dhth, :u)), chth.prescribed_dofs) == 2 * nhanging
    @test count(in(fielddofs(dhth, :p)), chth.prescribed_dofs) == nhanging
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
