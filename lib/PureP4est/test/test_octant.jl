# Octants: the p4est reference numbering tables, the (level, coordinate) <-> morton
# encoding, and the octant-local operations. Mirrors `src/octree.jl`.
using PureP4est, Test

@testset "OctantBWG Lookup Tables" begin
    @test PureP4est._face(1) == [3, 5]
    @test PureP4est._face(5) == [1, 5]
    @test PureP4est._face(12) == [2, 4]
    @test PureP4est._face(1, 1) == 3  && PureP4est._face(1, 2) == 5
    @test PureP4est._face(5, 1) == 1  && PureP4est._face(5, 2) == 5
    @test PureP4est._face(12, 1) == 2 && PureP4est._face(12, 2) == 4
    @test PureP4est._face(3, 1) == 3  && PureP4est._face(3, 2) == 6

    @test PureP4est._face_edge_corners(1, 1) == (0, 0)
    @test PureP4est._face_edge_corners(3, 3) == (3, 4)
    @test PureP4est._face_edge_corners(8, 6) == (2, 4)
    @test PureP4est._face_edge_corners(4, 5) == (0, 0)
    @test PureP4est._face_edge_corners(5, 4) == (0, 0)
    @test PureP4est._face_edge_corners(7, 1) == (3, 4)
    @test PureP4est._face_edge_corners(11, 1) == (2, 4)
    @test PureP4est._face_edge_corners(9, 1) == (1, 3)
    @test PureP4est._face_edge_corners(10, 2) == (1, 3)
    @test PureP4est._face_edge_corners(12, 2) == (2, 4)

    @test PureP4est.𝒱₃[1, :] == PureP4est.𝒰[1:4, 1] == PureP4est._face_corners(3, 1)
    @test PureP4est.𝒱₃[2, :] == PureP4est.𝒰[1:4, 2] == PureP4est._face_corners(3, 2)
    @test PureP4est.𝒱₃[3, :] == PureP4est.𝒰[5:8, 1] == PureP4est._face_corners(3, 3)
    @test PureP4est.𝒱₃[4, :] == PureP4est.𝒰[5:8, 2] == PureP4est._face_corners(3, 4)
    @test PureP4est.𝒱₃[5, :] == PureP4est.𝒰[9:12, 1] == PureP4est._face_corners(3, 5)
    @test PureP4est.𝒱₃[6, :] == PureP4est.𝒰[9:12, 2] == PureP4est._face_corners(3, 6)

    @test PureP4est._edge_corners(1) == [1, 2]
    @test PureP4est._edge_corners(4) == [7, 8]
    @test PureP4est._edge_corners(12, 2) == 8

    #Test Figure 3a) of Burstedde, Wilcox, Ghattas [2011]
    test_ξs = (1, 2, 3, 4)
    @test PureP4est._neighbor_corner.((1,), (2,), (1,), test_ξs) == test_ξs
    #Test Figure 3b)
    @test PureP4est._neighbor_corner.((3,), (5,), (3,), test_ξs) == (PureP4est.𝒫[5, :]...,)
end

@testset "Index Permutation" begin
    for i in 1:length(PureP4est.edge_perm)
        @test i == PureP4est.edge_perm_inv[PureP4est.edge_perm[i]]
    end
    for i in 1:length(PureP4est.𝒱₂_perm)
        @test i == PureP4est.𝒱₂_perm_inv[PureP4est.𝒱₂_perm[i]]
    end
    for i in 1:length(PureP4est.𝒱₃_perm)
        @test i == PureP4est.𝒱₃_perm_inv[PureP4est.𝒱₃_perm[i]]
    end
    for i in 1:length(PureP4est.node_map₂)
        @test i == PureP4est.node_map₂_inv[PureP4est.node_map₂[i]]
    end
    for i in 1:length(PureP4est.node_map₃)
        @test i == PureP4est.node_map₃_inv[PureP4est.node_map₃[i]]
    end
end

@testset "OctantBWG Encoding" begin
    # Tests from Figure 3a) and 3b) of Burstedde et al
    o = PureP4est.OctantBWG(3, 2, 21, 3)
    b = 3
    @test PureP4est.child_id(o, b) == 5
    @test PureP4est.child_id(PureP4est.parent(o, b), b) == 3
    @test PureP4est.parent(PureP4est.parent(o, b), b) == PureP4est.OctantBWG(3, 0, 1, b)
    @test PureP4est.parent(PureP4est.parent(PureP4est.parent(o, b), b), b) == PureP4est.root(3)
    o = PureP4est.OctantBWG(3, 2, 4, 3)
    @test PureP4est.child_id(o, b) == 4
    @test PureP4est.child_id(PureP4est.parent(o, b), b) == 1
    @test PureP4est.parent(PureP4est.parent(o, b), b) == PureP4est.OctantBWG(3, 0, 1, b)
    @test PureP4est.parent(PureP4est.parent(PureP4est.parent(o, b), b), b) == PureP4est.root(3)

    @test PureP4est.child_id(PureP4est.OctantBWG(2, 1, 1, 3), 3) == 1
    @test PureP4est.child_id(PureP4est.OctantBWG(2, 1, 2, 3), 3) == 2
    @test PureP4est.child_id(PureP4est.OctantBWG(2, 1, 3, 3), 3) == 3
    @test PureP4est.child_id(PureP4est.OctantBWG(2, 1, 4, 3), 3) == 4
    @test PureP4est.child_id(PureP4est.OctantBWG(2, 2, 1, 3), 3) == 1
    @test PureP4est.child_id(PureP4est.OctantBWG(3, 2, 1, 3), 3) == 1
    @test PureP4est.child_id(PureP4est.OctantBWG(3, 2, 2, 3), 3) == 2
    @test PureP4est.child_id(PureP4est.OctantBWG(3, 2, 3, 3), 3) == 3
    @test PureP4est.child_id(PureP4est.OctantBWG(3, 2, 4, 3), 3) == 4
    @test PureP4est.child_id(PureP4est.OctantBWG(3, 2, 16, 3), 3) == 8
    @test PureP4est.child_id(PureP4est.OctantBWG(3, 2, 24, 3), 3) == 8
    @test PureP4est.child_id(PureP4est.OctantBWG(3, 2, 64, 3), 3) == 8
    @test PureP4est.child_id(PureP4est.OctantBWG(3, 2, 9, 3), 3) == 1
    #maxlevel = 10 takes too long
    maxlevel = 6
    levels = collect(1:maxlevel)
    morton_ids = [1:(2^(2 * l)) for l in levels]
    for (level, morton_range) in zip(levels, morton_ids)
        for morton_id in morton_range
            @test Int(PureP4est.morton(PureP4est.OctantBWG(2, level, morton_id, maxlevel), level, maxlevel)) == morton_id
        end
    end
    morton_ids = [1:(2^(3 * l)) for l in levels]
    for (level, morton_range) in zip(levels, morton_ids)
        for morton_id in morton_range
            @test Int(PureP4est.morton(PureP4est.OctantBWG(3, level, morton_id, maxlevel), level, maxlevel)) == morton_id
        end
    end
end

@testset "OctantBWG Operations" begin
    o = PureP4est.OctantBWG(1, (2, 0, 0))
    @test PureP4est.facet_neighbor(o, 1, 2) == PureP4est.OctantBWG(1, (0, 0, 0))
    @test PureP4est.facet_neighbor(o, 2, 2) == PureP4est.OctantBWG(1, (4, 0, 0))
    @test PureP4est.facet_neighbor(o, 3, 2) == PureP4est.OctantBWG(1, (2, -2, 0))
    @test PureP4est.facet_neighbor(o, 4, 2) == PureP4est.OctantBWG(1, (2, 2, 0))
    @test PureP4est.facet_neighbor(o, 5, 2) == PureP4est.OctantBWG(1, (2, 0, -2))
    @test PureP4est.facet_neighbor(o, 6, 2) == PureP4est.OctantBWG(1, (2, 0, 2))
    @test PureP4est.descendants(o, 2) == (PureP4est.OctantBWG(2, (2, 0, 0)), PureP4est.OctantBWG(2, (3, 1, 1)))
    @test PureP4est.descendants(o, 3) == (PureP4est.OctantBWG(3, (2, 0, 0)), PureP4est.OctantBWG(3, (5, 3, 3)))

    o = PureP4est.OctantBWG(1, (0, 0, 0))
    @test PureP4est.facet_neighbor(o, 1, 2) == PureP4est.OctantBWG(1, (-2, 0, 0))
    @test PureP4est.facet_neighbor(o, 2, 2) == PureP4est.OctantBWG(1, (2, 0, 0))
    @test PureP4est.facet_neighbor(o, 3, 2) == PureP4est.OctantBWG(1, (0, -2, 0))
    @test PureP4est.facet_neighbor(o, 4, 2) == PureP4est.OctantBWG(1, (0, 2, 0))
    @test PureP4est.facet_neighbor(o, 5, 2) == PureP4est.OctantBWG(1, (0, 0, -2))
    @test PureP4est.facet_neighbor(o, 6, 2) == PureP4est.OctantBWG(1, (0, 0, 2))
    o = PureP4est.OctantBWG(0, (0, 0, 0))
    @test PureP4est.descendants(o, 2) == (PureP4est.OctantBWG(2, (0, 0, 0)), PureP4est.OctantBWG(2, (3, 3, 3)))
    @test PureP4est.descendants(o, 3) == (PureP4est.OctantBWG(3, (0, 0, 0)), PureP4est.OctantBWG(3, (7, 7, 7)))

    @test PureP4est.edge_neighbor(PureP4est.OctantBWG(2, (2, 0, 0)), 1, 3) == PureP4est.OctantBWG(2, (2, -2, -2))
    @test PureP4est.edge_neighbor(PureP4est.OctantBWG(2, (2, 0, 0)), 4, 3) == PureP4est.OctantBWG(2, (2, 2, 2))
    @test PureP4est.edge_neighbor(PureP4est.OctantBWG(2, (2, 0, 0)), 6, 3) == PureP4est.OctantBWG(2, (4, 0, -2))
    @test PureP4est.edge_neighbor(PureP4est.OctantBWG(2, (2, 0, 0)), 9, 3) == PureP4est.OctantBWG(2, (0, -2, 0))
    @test PureP4est.edge_neighbor(PureP4est.OctantBWG(2, (2, 0, 0)), 12, 3) == PureP4est.OctantBWG(2, (4, 2, 0))

    @test PureP4est.edge_neighbor(PureP4est.OctantBWG(3, (0, 0, 0)), 1, 4) == PureP4est.OctantBWG(3, (0, -2, -2))
    @test PureP4est.edge_neighbor(PureP4est.OctantBWG(3, (0, 0, 0)), 12, 4) == PureP4est.OctantBWG(3, (2, 2, 0))

    @test PureP4est.edge_neighbor(PureP4est.OctantBWG(2, (0, 0, 0)), 1, 4) == PureP4est.OctantBWG(2, (0, -4, -4))
    @test PureP4est.edge_neighbor(PureP4est.OctantBWG(2, (0, 0, 0)), 2, 4) == PureP4est.OctantBWG(2, (0, 4, -4))
    @test PureP4est.edge_neighbor(PureP4est.OctantBWG(2, (0, 0, 0)), 3, 4) == PureP4est.OctantBWG(2, (0, -4, 4))
    @test PureP4est.edge_neighbor(PureP4est.OctantBWG(2, (0, 0, 0)), 4, 4) == PureP4est.OctantBWG(2, (0, 4, 4))
    @test PureP4est.edge_neighbor(PureP4est.OctantBWG(2, (0, 0, 0)), 5, 4) == PureP4est.OctantBWG(2, (-4, 0, -4))
    @test PureP4est.edge_neighbor(PureP4est.OctantBWG(2, (0, 0, 0)), 6, 4) == PureP4est.OctantBWG(2, (4, 0, -4))
    @test PureP4est.edge_neighbor(PureP4est.OctantBWG(2, (0, 0, 0)), 7, 4) == PureP4est.OctantBWG(2, (-4, 0, 4))
    @test PureP4est.edge_neighbor(PureP4est.OctantBWG(2, (0, 0, 0)), 8, 4) == PureP4est.OctantBWG(2, (4, 0, 4))
    @test PureP4est.edge_neighbor(PureP4est.OctantBWG(2, (0, 0, 0)), 9, 4) == PureP4est.OctantBWG(2, (-4, -4, 0))
    @test PureP4est.edge_neighbor(PureP4est.OctantBWG(2, (0, 0, 0)), 10, 4) == PureP4est.OctantBWG(2, (4, -4, 0))
    @test PureP4est.edge_neighbor(PureP4est.OctantBWG(2, (0, 0, 0)), 11, 4) == PureP4est.OctantBWG(2, (-4, 4, 0))
    @test PureP4est.edge_neighbor(PureP4est.OctantBWG(2, (0, 0, 0)), 12, 4) == PureP4est.OctantBWG(2, (4, 4, 0))

    @test PureP4est.edge_neighbor(PureP4est.OctantBWG(1, (0, 0, 0)), 1, 4) == PureP4est.OctantBWG(1, (0, -8, -8))
    @test PureP4est.edge_neighbor(PureP4est.OctantBWG(1, (0, 0, 0)), 12, 4) == PureP4est.OctantBWG(1, (8, 8, 0))

    @test PureP4est.corner_neighbor(PureP4est.OctantBWG(2, (2, 0, 0)), 1, 3) == PureP4est.OctantBWG(2, (0, -2, -2))
    @test PureP4est.corner_neighbor(PureP4est.OctantBWG(2, (2, 0, 0)), 4, 3) == PureP4est.OctantBWG(2, (4, 2, -2))
    @test PureP4est.corner_neighbor(PureP4est.OctantBWG(2, (2, 0, 0)), 8, 3) == PureP4est.OctantBWG(2, (4, 2, 2))

    @test PureP4est.corner_neighbor(PureP4est.OctantBWG(2, (2, 0)), 1, 3) == PureP4est.OctantBWG(2, (0, -2))
    @test PureP4est.corner_neighbor(PureP4est.OctantBWG(2, (2, 0)), 2, 3) == PureP4est.OctantBWG(2, (4, -2))
    @test PureP4est.corner_neighbor(PureP4est.OctantBWG(2, (2, 0)), 4, 3) == PureP4est.OctantBWG(2, (4, 2))
end

@testset "isancestor" begin
    b = 5
    for dim in (2, 3)
        r = PureP4est.root(dim)
        first_child = PureP4est.children(r, b)[1]
        grandchild = PureP4est.children(first_child, b)[1]
        other = PureP4est.children(r, b)[end]

        # the root is an ancestor of everything below it (used to be missed because the
        # parent walk stopped before reaching level 0)
        @test PureP4est.isancestor(r, first_child, b)
        @test PureP4est.isancestor(r, grandchild, b)
        # direct parent and grandparent
        @test PureP4est.isancestor(first_child, grandchild, b)
        # strict: an octant is not its own ancestor, and finer is never an ancestor of coarser
        @test !PureP4est.isancestor(r, r, b)
        @test !PureP4est.isancestor(grandchild, first_child, b)
        @test !PureP4est.isancestor(first_child, r, b)
        # a different branch at the same level is unrelated
        @test !PureP4est.isancestor(other, grandchild, b)
    end
end

@testset "OctantBWG convenience methods and error paths" begin
    # integer-type-promoting convenience methods
    o2 = PureP4est.OctantBWG(1, (0, 2))
    o3 = PureP4est.OctantBWG(1, (0, 2, 4))
    @test PureP4est.morton(o2, Int32(1), Int32(3)) == PureP4est.morton(o2, 1, 3)
    @test PureP4est.facet_neighbor(o2, Int32(1), Int32(3)) == PureP4est.facet_neighbor(o2, 1, 3)
    @test PureP4est.corner_neighbor(o2, Int32(1), Int32(3)) == PureP4est.corner_neighbor(o2, 1, 3)
    @test PureP4est.edge_neighbor(o3, Int32(1), Int32(3)) == PureP4est.edge_neighbor(o3, 1, 3)

    # face -> corner lookup for both dimensions, and the error path
    @test PureP4est._face_corners(2, 1) == PureP4est.𝒱₂[1, :]
    @test PureP4est._face_corners(3, 1) == PureP4est.𝒱₃[1, :]
    @test_throws ErrorException PureP4est._face_corners(4, 1)

    # octant level beyond the tree's maximum refinement level b
    @test_throws DomainError PureP4est._compute_size(2, 3)
end
