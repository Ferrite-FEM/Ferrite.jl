# Octants: the p4est reference numbering tables, the (level, coordinate) <-> morton
# encoding, and the octant-local operations. Mirrors `src/Adaptivity/octree.jl`.
using Ferrite, Test

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

@testset "isancestor" begin
    b = 5
    for dim in (2, 3)
        r = Ferrite.AMR.root(dim)
        first_child = Ferrite.AMR.children(r, b)[1]
        grandchild = Ferrite.AMR.children(first_child, b)[1]
        other = Ferrite.AMR.children(r, b)[end]

        # the root is an ancestor of everything below it (used to be missed because the
        # parent walk stopped before reaching level 0)
        @test Ferrite.AMR.isancestor(r, first_child, b)
        @test Ferrite.AMR.isancestor(r, grandchild, b)
        # direct parent and grandparent
        @test Ferrite.AMR.isancestor(first_child, grandchild, b)
        # strict: an octant is not its own ancestor, and finer is never an ancestor of coarser
        @test !Ferrite.AMR.isancestor(r, r, b)
        @test !Ferrite.AMR.isancestor(grandchild, first_child, b)
        @test !Ferrite.AMR.isancestor(first_child, r, b)
        # a different branch at the same level is unrelated
        @test !Ferrite.AMR.isancestor(other, grandchild, b)
    end
end
