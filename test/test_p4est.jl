@testset "OctantBWG Lookup Tables" begin
    @test Ferrite._face(1) == [3,5]
    @test Ferrite._face(5) == [1,5]
    @test Ferrite._face(12) == [2,4]
    @test Ferrite._face(1,1) == 3  && Ferrite._face(1,2) == 5
    @test Ferrite._face(5,1) == 1  && Ferrite._face(5,2) == 5
    @test Ferrite._face(12,1) == 2 && Ferrite._face(12,2) == 4
    @test Ferrite._face(3,1) == 3  && Ferrite._face(3,2) == 6

    @test Ferrite._face_edge_corners(1,1) == (0,0)
    @test Ferrite._face_edge_corners(3,3) == (3,4)
    @test Ferrite._face_edge_corners(8,6) == (2,4)
    @test Ferrite._face_edge_corners(4,5) == (0,0)
    @test Ferrite._face_edge_corners(5,4) == (0,0)
    @test Ferrite._face_edge_corners(7,1) == (3,4)
    @test Ferrite._face_edge_corners(11,1) == (2,4)
    @test Ferrite._face_edge_corners(9,1) == (1,3)
    @test Ferrite._face_edge_corners(10,2) == (1,3)
    @test Ferrite._face_edge_corners(12,2) == (2,4)
   
    @test Ferrite.𝒱₃[1,:] == Ferrite.𝒰[1:4,1] == Ferrite._face_corners(3,1)
    @test Ferrite.𝒱₃[2,:] == Ferrite.𝒰[1:4,2] == Ferrite._face_corners(3,2)
    @test Ferrite.𝒱₃[3,:] == Ferrite.𝒰[5:8,1] == Ferrite._face_corners(3,3)
    @test Ferrite.𝒱₃[4,:] == Ferrite.𝒰[5:8,2] == Ferrite._face_corners(3,4)
    @test Ferrite.𝒱₃[5,:] == Ferrite.𝒰[9:12,1] == Ferrite._face_corners(3,5)
    @test Ferrite.𝒱₃[6,:] == Ferrite.𝒰[9:12,2] == Ferrite._face_corners(3,6)

    @test Ferrite._edge_corners(1) == [1,2]
    @test Ferrite._edge_corners(4) == [7,8]
    @test Ferrite._edge_corners(12,2) == 8
   
    #Test Figure 3a) of Burstedde, Wilcox, Ghattas [2011]
    test_ξs = (1,2,3,4)
    @test Ferrite._neighbor_corner.((1,),(2,),(1,),test_ξs) == test_ξs
    #Test Figure 3b)
    @test Ferrite._neighbor_corner.((3,),(5,),(3,),test_ξs) == (Ferrite.𝒫[5,:]...,)
end

@testset "OctantBWG Encoding" begin
#    # Tests from Figure 3a) and 3b) of Burstedde et al
    o = Ferrite.OctantBWG(3,2,21,3)
    b = 3
    @test Ferrite.child_id(o,b) == 5
    @test Ferrite.child_id(Ferrite.parent(o,b),b) == 3
    @test Ferrite.parent(Ferrite.parent(o,b),b) == Ferrite.OctantBWG(3,0,1,b)
    @test Ferrite.parent(Ferrite.parent(Ferrite.parent(o,b),b),b) == Ferrite.root(3)
    o = Ferrite.OctantBWG(3,2,4,3)
    @test Ferrite.child_id(o,b) == 4
    @test Ferrite.child_id(Ferrite.parent(o,b),b) == 1
    @test Ferrite.parent(Ferrite.parent(o,b),b) == Ferrite.OctantBWG(3,0,1,b)
    @test Ferrite.parent(Ferrite.parent(Ferrite.parent(o,b),b),b) == Ferrite.root(3)

    @test Ferrite.child_id(Ferrite.OctantBWG(2,1,1,3),3) == 1
    @test Ferrite.child_id(Ferrite.OctantBWG(2,1,2,3),3) == 2
    @test Ferrite.child_id(Ferrite.OctantBWG(2,1,3,3),3) == 3
    @test Ferrite.child_id(Ferrite.OctantBWG(2,1,4,3),3) == 4
    @test Ferrite.child_id(Ferrite.OctantBWG(2,2,1,3),3) == 1
    @test Ferrite.child_id(Ferrite.OctantBWG(3,2,1,3),3) == 1
    @test Ferrite.child_id(Ferrite.OctantBWG(3,2,2,3),3) == 2
    @test Ferrite.child_id(Ferrite.OctantBWG(3,2,3,3),3) == 3
    @test Ferrite.child_id(Ferrite.OctantBWG(3,2,4,3),3) == 4
    @test Ferrite.child_id(Ferrite.OctantBWG(3,2,16,3),3) == 8
    @test Ferrite.child_id(Ferrite.OctantBWG(3,2,24,3),3) == 8
    @test Ferrite.child_id(Ferrite.OctantBWG(3,2,64,3),3) == 8
    @test Ferrite.child_id(Ferrite.OctantBWG(3,2,9,3),3) == 1
    #maxlevel = 10 takes too long
    maxlevel = 6
    levels = collect(1:maxlevel)
    morton_ids = [1:2^(2*l) for l in levels]
    for (level,morton_range) in zip(levels,morton_ids)
        for morton_id in morton_range
            @test Int(Ferrite.morton(OctantBWG(2,level,morton_id,maxlevel),level,maxlevel)) == morton_id
        end
    end
    morton_ids = [1:2^(3*l) for l in levels]
    for (level,morton_range) in zip(levels,morton_ids)
        for morton_id in morton_range
            @test Int(Ferrite.morton(OctantBWG(3,level,morton_id,maxlevel),level,maxlevel)) == morton_id
        end
    end
end

@testset "OctantBWG Operations" begin
    o = Ferrite.OctantBWG(1,(2,0,0))
    @test Ferrite.face_neighbor(o,1,2) == Ferrite.OctantBWG(1,(0,0,0))
    @test Ferrite.face_neighbor(o,2,2) == Ferrite.OctantBWG(1,(4,0,0))
    @test Ferrite.face_neighbor(o,3,2) == Ferrite.OctantBWG(1,(2,-2,0))
    @test Ferrite.face_neighbor(o,4,2) == Ferrite.OctantBWG(1,(2,2,0))
    @test Ferrite.face_neighbor(o,5,2) == Ferrite.OctantBWG(1,(2,0,-2))
    @test Ferrite.face_neighbor(o,6,2) == Ferrite.OctantBWG(1,(2,0,2))
    @test Ferrite.descendants(o,2) == (Ferrite.OctantBWG(2,(2,0,0)), Ferrite.OctantBWG(2,(3,1,1)))
    @test Ferrite.descendants(o,3) == (Ferrite.OctantBWG(3,(2,0,0)), Ferrite.OctantBWG(3,(5,3,3)))

    o = Ferrite.OctantBWG(1,(0,0,0))
    @test Ferrite.face_neighbor(o,1,2) == Ferrite.OctantBWG(1,(-2,0,0))
    @test Ferrite.face_neighbor(o,2,2) == Ferrite.OctantBWG(1,(2,0,0))
    @test Ferrite.face_neighbor(o,3,2) == Ferrite.OctantBWG(1,(0,-2,0))
    @test Ferrite.face_neighbor(o,4,2) == Ferrite.OctantBWG(1,(0,2,0))
    @test Ferrite.face_neighbor(o,5,2) == Ferrite.OctantBWG(1,(0,0,-2))
    @test Ferrite.face_neighbor(o,6,2) == Ferrite.OctantBWG(1,(0,0,2))
    o = Ferrite.OctantBWG(0,(0,0,0))
    @test Ferrite.descendants(o,2) == (Ferrite.OctantBWG(2,(0,0,0)), Ferrite.OctantBWG(2,(3,3,3)))
    @test Ferrite.descendants(o,3) == (Ferrite.OctantBWG(3,(0,0,0)), Ferrite.OctantBWG(3,(7,7,7)))

    @test Ferrite.edge_neighbor(Ferrite.OctantBWG(2,(2,0,0)),1,3) == Ferrite.OctantBWG(2,(2,-2,-2))
    @test Ferrite.edge_neighbor(Ferrite.OctantBWG(2,(2,0,0)),4,3) == Ferrite.OctantBWG(2,(2,2,2))
    @test Ferrite.edge_neighbor(Ferrite.OctantBWG(2,(2,0,0)),6,3) == Ferrite.OctantBWG(2,(4,0,-2))
    @test Ferrite.edge_neighbor(Ferrite.OctantBWG(2,(2,0,0)),9,3) == Ferrite.OctantBWG(2,(0,-2,0))
    @test Ferrite.edge_neighbor(Ferrite.OctantBWG(2,(2,0,0)),12,3) == Ferrite.OctantBWG(2,(4,2,0))

    @test Ferrite.edge_neighbor(Ferrite.OctantBWG(3,(0,0,0)),1,4)  == Ferrite.OctantBWG(3,(0,-2,-2))
    @test Ferrite.edge_neighbor(Ferrite.OctantBWG(3,(0,0,0)),12,4) == Ferrite.OctantBWG(3,(2,2,0))
    @test Ferrite.edge_neighbor(Ferrite.OctantBWG(2,(0,0,0)),1,4)  == Ferrite.OctantBWG(2,(0,-4,-4))
    @test Ferrite.edge_neighbor(Ferrite.OctantBWG(2,(0,0,0)),12,4) == Ferrite.OctantBWG(2,(4,4,0))
    @test Ferrite.edge_neighbor(Ferrite.OctantBWG(1,(0,0,0)),1,4)  == Ferrite.OctantBWG(1,(0,-8,-8))
    @test Ferrite.edge_neighbor(Ferrite.OctantBWG(1,(0,0,0)),12,4) == Ferrite.OctantBWG(1,(8,8,0))

    @test Ferrite.corner_neighbor(Ferrite.OctantBWG(2,(2,0,0)),1,3) == Ferrite.OctantBWG(2,(0,-2,-2))
    @test Ferrite.corner_neighbor(Ferrite.OctantBWG(2,(2,0,0)),4,3) == Ferrite.OctantBWG(2,(4,2,-2))
    @test Ferrite.corner_neighbor(Ferrite.OctantBWG(2,(2,0,0)),8,3) == Ferrite.OctantBWG(2,(4,2,2))

    @test Ferrite.corner_neighbor(Ferrite.OctantBWG(2,(2,0)),1,3) == Ferrite.OctantBWG(2,(0,-2))
    @test Ferrite.corner_neighbor(Ferrite.OctantBWG(2,(2,0)),2,3) == Ferrite.OctantBWG(2,(4,-2))
    @test Ferrite.corner_neighbor(Ferrite.OctantBWG(2,(2,0)),4,3) == Ferrite.OctantBWG(2,(4,2))
end

@testset "OctreeBWG Operations" begin
    # maximum level == 3
    # Octant level 0 size == 2^3=8
    # Octant level 1 size == 2^3/2 = 4
    # Octant level 2 size == 2^3/2 = 2
    # Octant level 3 size == 2^3/2 = 1
    # test translation constructor
    grid = generate_grid(Quadrilateral,(2,2))
    grid.cells[2] = Quadrilateral((grid.cells[2].nodes[2], grid.cells[2].nodes[3], grid.cells[2].nodes[4], grid.cells[2].nodes[1]))
    # x-----------x-----------x
    # |4    4    3|4    4    3|
    # |           |           |
    # |     ^     |     ^     |
    # |1    |    2|1    |    2|
    # |     +-->  |     +-->  |
    # |           |           |
    # |1    3    2|1    3    2|
    # x-----------x-----------|
    # |4    4    3|3    2    2|
    # |           |           |
    # |     ^     |     ^     |
    # |1    |    2|4    |    3|
    # |     +-->  |  <--+     |
    # |           |           |
    # |1    3    2|4    1    1|
    # x-----------x-----------x
    adaptive_grid = ForestBWG(grid,3)
    for cell in adaptive_grid.cells
        @test cell isa OctreeBWG
        @test cell.leaves[1] == OctantBWG(2,0,1,cell.b)
    end
    # Looks like some sign error sneaked in :)
    @test Ferrite.transform_face(adaptive_grid, FaceIndex(2,4), adaptive_grid.cells[1].leaves[1]) == OctantBWG(0,(-8,0))
    @test Ferrite.transform_face(adaptive_grid, FaceIndex(1,2), adaptive_grid.cells[2].leaves[1]) == OctantBWG(0,(0,-8))
    @test Ferrite.transform_face(adaptive_grid, FaceIndex(4,1), adaptive_grid.cells[3].leaves[1]) == OctantBWG(0,(-8,0))
    @test Ferrite.transform_face(adaptive_grid, FaceIndex(3,2), adaptive_grid.cells[4].leaves[1]) == OctantBWG(0,(8,0))
    @test Ferrite.transform_face(adaptive_grid, FaceIndex(3,3), adaptive_grid.cells[1].leaves[1]) == OctantBWG(0,(0,-8))
    @test Ferrite.transform_face(adaptive_grid, FaceIndex(1,4), adaptive_grid.cells[3].leaves[1]) == OctantBWG(0,(0,8))
    @test Ferrite.transform_face(adaptive_grid, FaceIndex(4,3), adaptive_grid.cells[2].leaves[1]) == OctantBWG(0,(-8,0))
    @test Ferrite.transform_face(adaptive_grid, FaceIndex(2,2), adaptive_grid.cells[4].leaves[1]) == OctantBWG(0,(0,8))
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
    # x-----x-----x-----------|
    # |     |     |           |
    # |     |     |           |
    # |     |     |           |
    # x--x--x-----x           |
    # |  |  |     |           |
    # x--x--x     |           |
    # |  |  |     |           |
    # x--x--x-----x-----------x
    Ferrite.refine!(adaptive_grid.cells[1],adaptive_grid.cells[1].leaves[1])
    @test length(adaptive_grid.cells[1].leaves) == 4
    for (m,octant) in zip(1:4,adaptive_grid.cells[1].leaves)
        @test octant == OctantBWG(2,1,m,adaptive_grid.cells[1].b)
    end
    Ferrite.refine!(adaptive_grid.cells[1],adaptive_grid.cells[1].leaves[1])
    # octree holds now 3 first level and 4 second level
    @test length(adaptive_grid.cells[1].leaves) == 7
    for (m,octant) in zip(1:4,adaptive_grid.cells[1].leaves)
        @test octant == OctantBWG(2,2,m,adaptive_grid.cells[1].b)
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
    adaptive_grid = ForestBWG(grid,3)
    Ferrite.refine!(adaptive_grid.cells[1],adaptive_grid.cells[1].leaves[1])
    Ferrite.refine!(adaptive_grid.cells[1],adaptive_grid.cells[1].leaves[4])
    @test length(adaptive_grid.cells[1].leaves) == 7
    @test all(getproperty.(adaptive_grid.cells[1].leaves[1:3],:l) .== 1)
    ##################################################################
    ####uniform refinement and coarsening for all cells and levels####
    ##################################################################
    adaptive_grid = ForestBWG(grid,8)
    for l in 1:8
        Ferrite.refine_all!(adaptive_grid,l)
        for tree in adaptive_grid.cells
            @test all(Ferrite.morton.(tree.leaves,l,8) == collect(1:2^(2*l)))
        end
    end
    #check montonicity of ancestor_id
    for tree in adaptive_grid.cells
        ids = Ferrite.ancestor_id.(tree.leaves,(1,),(tree.b,))
        @test issorted(ids)
    end
    #now go back from finest to coarsest
    for l in 7:-1:0
        Ferrite.coarsen_all!(adaptive_grid)
        for tree in adaptive_grid.cells
            @test all(Ferrite.morton.(tree.leaves,l,8) == collect(1:2^(2*l)))
        end
    end
    #########################
    # now do the same with 3D
    # some ascii picasso can insert here something beautiful
    #########################
    grid = generate_grid(Hexahedron,(2,2,2))
    adaptive_grid = ForestBWG(grid,3)
    Ferrite.refine!(adaptive_grid.cells[1],adaptive_grid.cells[1].leaves[1])
    @test length(adaptive_grid.cells[1].leaves) == 8
    for (m,octant) in zip(1:8,adaptive_grid.cells[1].leaves)
        @test octant == OctantBWG(3,1,m,adaptive_grid.cells[1].b)
    end
    Ferrite.refine!(adaptive_grid.cells[1],adaptive_grid.cells[1].leaves[1])
    @test length(adaptive_grid.cells[1].leaves) == 15
    for (m,octant) in zip(1:8,adaptive_grid.cells[1].leaves)
        @test octant == OctantBWG(3,2,m,adaptive_grid.cells[1].b)
    end
    adaptive_grid = ForestBWG(grid,3)
    Ferrite.refine!(adaptive_grid.cells[1],adaptive_grid.cells[1].leaves[1])
    Ferrite.refine!(adaptive_grid.cells[1],adaptive_grid.cells[1].leaves[4])
    @test length(adaptive_grid.cells[1].leaves) == 15
    @test all(getproperty.(adaptive_grid.cells[1].leaves[1:3],:l) .== 1)
    @test all(getproperty.(adaptive_grid.cells[1].leaves[4:11],:l) .== 2)
    @test all(getproperty.(adaptive_grid.cells[1].leaves[12:end],:l) .== 1)
    adaptive_grid = ForestBWG(grid,5)
    #go from coarsest to finest uniformly
    for l in 1:5
        Ferrite.refine_all!(adaptive_grid,l)
        for tree in adaptive_grid.cells
            @test all(Ferrite.morton.(tree.leaves,l,5) == collect(1:2^(3*l)))
        end
    end
    #now go back from finest to coarsest
    for l in 4:-1:0
        Ferrite.coarsen_all!(adaptive_grid)
        for tree in adaptive_grid.cells
            @test all(Ferrite.morton.(tree.leaves,l,5) == collect(1:2^(3*l)))
        end
    end
end

@testset "ForestBWG AbstractGrid Interfacing" begin
    maxlevel = 3
    grid = generate_grid(Quadrilateral,(2,2))
    adaptive_grid = ForestBWG(grid,maxlevel)
    for l in 1:maxlevel
        Ferrite.refine_all!(adaptive_grid,l)
        @test getncells(adaptive_grid) == 2^(2*l) * 4 == length(getcells(adaptive_grid))
    end
end

@testset "Balancing" begin
    grid = generate_grid(Quadrilateral,(1,1))
    adaptive_grid = ForestBWG(grid,3)
    Ferrite.refine_all!(adaptive_grid,1)
    Ferrite.refine!(adaptive_grid.cells[1],adaptive_grid.cells[1].leaves[2])
    Ferrite.refine!(adaptive_grid.cells[1],adaptive_grid.cells[1].leaves[6])
    Ferrite.refine!(adaptive_grid.cells[1],adaptive_grid.cells[1].leaves[6])
    balanced = Ferrite.balancetree(adaptive_grid.cells[1])
    @test length(balanced.leaves) == 16

    adaptive_grid = ForestBWG(grid,5)
    Ferrite.refine_all!(adaptive_grid,1)
    Ferrite.refine!(adaptive_grid.cells[1],adaptive_grid.cells[1].leaves[2])
    Ferrite.refine!(adaptive_grid.cells[1],adaptive_grid.cells[1].leaves[4])
    Ferrite.refine!(adaptive_grid.cells[1],adaptive_grid.cells[1].leaves[7])
    Ferrite.refine!(adaptive_grid.cells[1],adaptive_grid.cells[1].leaves[12])
    Ferrite.refine!(adaptive_grid.cells[1],adaptive_grid.cells[1].leaves[12])
    Ferrite.refine!(adaptive_grid.cells[1],adaptive_grid.cells[1].leaves[15])
    Ferrite.refine!(adaptive_grid.cells[1],adaptive_grid.cells[1].leaves[16])
    balanced = Ferrite.balancetree(adaptive_grid.cells[1])
    @test length(balanced.leaves) == 64


    grid = generate_grid(Quadrilateral,(2,1))
    adaptive_grid = ForestBWG(grid,2)
    Ferrite.refine!(adaptive_grid.cells[1],adaptive_grid.cells[1].leaves[1])
    Ferrite.refine!(adaptive_grid.cells[1],adaptive_grid.cells[1].leaves[2])
    Ferrite.balanceforest!(adaptive_grid)
    @test Ferrite.getncells(adaptive_grid) == 11

    grid = generate_grid(Quadrilateral,(2,2))
    adaptive_grid = ForestBWG(grid,2)
    Ferrite.refine!(adaptive_grid.cells[1],adaptive_grid.cells[1].leaves[1])
    Ferrite.refine!(adaptive_grid.cells[1],adaptive_grid.cells[1].leaves[4])
    Ferrite.balanceforest!(adaptive_grid)
    @test Ferrite.getncells(adaptive_grid) == 19
end
