@testset "Octant Lookup Tables" begin
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

@testset "Octant Encoding" begin
    # Tests from Figure 3a) and 3b) of Burstedde et al
    o = Ferrite.Octant{3,8,6}(2,(0,4,2))
    b = 0x03
    @test Ferrite.child_id(o,b) == 5
    @test Ferrite.child_id(Ferrite.parent(o,b),b) == 3
    @test Ferrite.parent(Ferrite.parent(o,b),b) == Ferrite.Octant{3,8,6}(0,(0,0,0))
    @test_throws ErrorException Ferrite.parent(Ferrite.parent(Ferrite.parent(o,b),b),b)
    o = Ferrite.Octant{3,8,6}(2,(2,2,0))
    @test Ferrite.child_id(o,b) == 4 
    @test Ferrite.child_id(Ferrite.parent(o,b),b) == 1 
    @test Ferrite.parent(Ferrite.parent(o,b),b) == Ferrite.Octant{3,8,6}(0,(0,0,0))
    @test_throws ErrorException Ferrite.parent(Ferrite.parent(Ferrite.parent(o,b),b),b)    

    # Now I shift the root about (1,1,1)
    o = Ferrite.Octant{3,8,6}(2,(0,4,2) .+ 1)
    b = 0x03
    @test Ferrite.child_id(o,b) == 5
    @test Ferrite.child_id(Ferrite.parent(o,b),b) == 3
    @test Ferrite.parent(Ferrite.parent(o,b),b) == Ferrite.Octant{3,8,6}(0,(0,0,0) .+ 1)
    @test_throws ErrorException Ferrite.parent(Ferrite.parent(Ferrite.parent(o,b),b),b)
    o = Ferrite.Octant{3,8,6}(2,(2,2,0) .+ 1)
    @test Ferrite.child_id(o,b) == 4 
    @test Ferrite.child_id(Ferrite.parent(o,b),b) == 1 
    @test Ferrite.parent(Ferrite.parent(o,b),b) == Ferrite.Octant{3,8,6}(0,(0,0,0) .+ 1)
    @test_throws ErrorException Ferrite.parent(Ferrite.parent(Ferrite.parent(o,b),b),b)

    # coordinate system always on lowest level
    # dim 3, level 2, morton id 2, number of levels 3
    @test Ferrite.Octant(3,2,2,3) == Ferrite.Octant{3,8,6}(2,(2,0,0)) 
    # dim 3, level 1, morton id 2, number of levels 3
    @test Ferrite.Octant(3,1,2,3) == Ferrite.Octant{3,8,6}(1,(4,0,0)) 
    # dim 3, level 0, morton id 2, number of levels 3
    @test_throws AssertionError Ferrite.Octant(3,0,2,3)
    # dim 3, level 2, morton id 4, number of levels 3
    @test Ferrite.Octant(3,2,4,3) == Ferrite.Octant{3,8,6}(2,(2,2,0)) 
    @test Ferrite.Octant(3,1,4,3) == Ferrite.Octant{3,8,6}(1,(4,4,0)) 
    @test Ferrite.Octant(3,2,5,3) == Ferrite.Octant{3,8,6}(2,(0,0,2)) 
    @test Ferrite.Octant(3,1,5,3) == Ferrite.Octant{3,8,6}(1,(0,0,4)) 
    @test Ferrite.Octant(2,1,1,3) == Ferrite.Octant{2,8,6}(1,(0,0))
    @test Ferrite.Octant(2,1,2,3) == Ferrite.Octant{2,8,6}(1,(4,0))
    @test Ferrite.Octant(2,1,3,3) == Ferrite.Octant{2,8,6}(1,(0,4))
    @test Ferrite.Octant(2,1,4,3) == Ferrite.Octant{2,8,6}(1,(4,4))
    @test Ferrite.child_id(Ferrite.Octant(2,1,1,3),3) == 1
    @test Ferrite.child_id(Ferrite.Octant(2,1,2,3),3) == 2
    @test Ferrite.child_id(Ferrite.Octant(2,1,3,3),3) == 3
    @test Ferrite.child_id(Ferrite.Octant(2,1,4,3),3) == 4
    @test Ferrite.child_id(Ferrite.Octant(2,2,1,3),3) == 1
    @test Ferrite.child_id(Ferrite.Octant(3,2,1,3),3) == 1
    @test Ferrite.child_id(Ferrite.Octant(3,2,2,3),3) == 2
    @test Ferrite.child_id(Ferrite.Octant(3,2,3,3),3) == 3
    @test Ferrite.child_id(Ferrite.Octant(3,2,4,3),3) == 4
    @test Ferrite.child_id(Ferrite.Octant(3,2,16,3),3) == 8
    @test Ferrite.child_id(Ferrite.Octant(3,2,24,3),3) == 8
    @test Ferrite.child_id(Ferrite.Octant(3,2,64,3),3) == 8
    @test Ferrite.child_id(Ferrite.Octant(3,2,9,3),3) == 1
end

@testset "Octant Operations" begin
    o = Ferrite.Octant{3,8,6}(1,(2,0,0))
    @test Ferrite.face_neighbor(o,0x01,0x02) == Ferrite.Octant{3,8,6}(1,(0,0,0)) 
    @test Ferrite.face_neighbor(o,0x02,0x02) == Ferrite.Octant{3,8,6}(1,(4,0,0)) 
    @test Ferrite.face_neighbor(o,0x03,0x02) == Ferrite.Octant{3,8,6}(1,(2,-2,0))
    @test Ferrite.face_neighbor(o,0x04,0x02) == Ferrite.Octant{3,8,6}(1,(2,2,0)) 
    @test Ferrite.face_neighbor(o,0x05,0x02) == Ferrite.Octant{3,8,6}(1,(2,0,-2))
    @test Ferrite.face_neighbor(o,0x06,0x02) == Ferrite.Octant{3,8,6}(1,(2,0,2)) 
    @test Ferrite.descendants(o,2) == (Ferrite.Octant{3,8,6}(1,(2,0,0)), Ferrite.Octant{3,8,6}(1,(2,0,0)))
    @test Ferrite.descendants(o,3) == (Ferrite.Octant{3,8,6}(2,(2,0,0)), Ferrite.Octant{3,8,6}(2,(4,2,2)))

    o = Ferrite.Octant{3,8,6}(1,(0,0,0))
    @test Ferrite.face_neighbor(o,1,2) == Ferrite.Octant{3,8,6}(1,(-2,0,0)) 
    @test Ferrite.face_neighbor(o,2,2) == Ferrite.Octant{3,8,6}(1,(2,0,0)) 
    @test Ferrite.face_neighbor(o,3,2) == Ferrite.Octant{3,8,6}(1,(0,-2,0))
    @test Ferrite.face_neighbor(o,4,2) == Ferrite.Octant{3,8,6}(1,(0,2,0)) 
    @test Ferrite.face_neighbor(o,5,2) == Ferrite.Octant{3,8,6}(1,(0,0,-2))
    @test Ferrite.face_neighbor(o,6,2) == Ferrite.Octant{3,8,6}(1,(0,0,2)) 
    o = Ferrite.Octant{3,8,6}(0,(0,0,0))
    @test Ferrite.descendants(o,2) == (Ferrite.Octant{3,8,6}(1,(0,0,0)), Ferrite.Octant{3,8,6}(1,(2,2,2)))
    @test Ferrite.descendants(o,3) == (Ferrite.Octant{3,8,6}(2,(0,0,0)), Ferrite.Octant{3,8,6}(2,(6,6,6)))
    
    @test Ferrite.edge_neighbor(Ferrite.Octant{3,8,6}(2,(2,0,0)),1,3) == Ferrite.Octant{3,8,6}(2,(2,-2,-2))
    @test Ferrite.edge_neighbor(Ferrite.Octant{3,8,6}(2,(2,0,0)),4,3) == Ferrite.Octant{3,8,6}(2,(2,2,2))
    @test Ferrite.edge_neighbor(Ferrite.Octant{3,8,6}(2,(2,0,0)),6,3) == Ferrite.Octant{3,8,6}(2,(4,0,-2))
    @test Ferrite.edge_neighbor(Ferrite.Octant{3,8,6}(2,(2,0,0)),9,3) == Ferrite.Octant{3,8,6}(2,(0,-2,0))
    @test Ferrite.edge_neighbor(Ferrite.Octant{3,8,6}(2,(2,0,0)),12,3) == Ferrite.Octant{3,8,6}(2,(4,2,0))

    @test Ferrite.edge_neighbor(Ferrite.Octant{3,8,6}(3,(0,0,0)),1,4)  == Ferrite.Octant{3,8,6}(3,(0,-2,-2))
    @test Ferrite.edge_neighbor(Ferrite.Octant{3,8,6}(3,(0,0,0)),12,4) == Ferrite.Octant{3,8,6}(3,(2,2,0))
    @test Ferrite.edge_neighbor(Ferrite.Octant{3,8,6}(2,(0,0,0)),1,4)  == Ferrite.Octant{3,8,6}(2,(0,-4,-4))
    @test Ferrite.edge_neighbor(Ferrite.Octant{3,8,6}(2,(0,0,0)),12,4) == Ferrite.Octant{3,8,6}(2,(4,4,0))
    @test Ferrite.edge_neighbor(Ferrite.Octant{3,8,6}(1,(0,0,0)),1,4)  == Ferrite.Octant{3,8,6}(1,(0,-8,-8))
    @test Ferrite.edge_neighbor(Ferrite.Octant{3,8,6}(1,(0,0,0)),12,4) == Ferrite.Octant{3,8,6}(1,(8,8,0))

    @test Ferrite.corner_neighbor(Ferrite.Octant{3,8,6}(2,(2,0,0)),1,3) == Ferrite.Octant{3,8,6}(2,(0,-2,-2))
    @test Ferrite.corner_neighbor(Ferrite.Octant{3,8,6}(2,(2,0,0)),4,3) == Ferrite.Octant{3,8,6}(2,(4,2,-2))
    @test Ferrite.corner_neighbor(Ferrite.Octant{3,8,6}(2,(2,0,0)),8,3) == Ferrite.Octant{3,8,6}(2,(4,2,2))

    @test Ferrite.corner_neighbor(Ferrite.Octant{2,8,6}(2,(2,0)),1,3) == Ferrite.Octant{2,8,6}(2,(0,-2))
    @test Ferrite.corner_neighbor(Ferrite.Octant{2,8,6}(2,(2,0)),2,3) == Ferrite.Octant{2,8,6}(2,(4,-2))
    @test Ferrite.corner_neighbor(Ferrite.Octant{2,8,6}(2,(2,0)),4,3) == Ferrite.Octant{2,8,6}(2,(4,2))
end
