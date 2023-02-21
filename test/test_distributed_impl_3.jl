using Ferrite, MPI, Metis
using Test

MPI.Init()
@testset "setup check 2" begin
    @test MPI.Comm_size(MPI.COMM_WORLD) == 3
end


FerriteMPI = Base.get_extension(Ferrite, :FerriteMPI)

# In this test battery we check for several invariants
# 1. Elements are not rotated or flipped during distribution
# 2. Element index order is preserved during distribution
# 3. Shared entities are setup correctly
# Note that there is no strict ordering requirement on the shared entities!
@testset "distributed grid construction 3" begin
    # We do not cover subcommunicators for now.
    comm = MPI.COMM_WORLD

    # y
    # ^
    # |
    # |
    #  ----> x
    #
    # Global grid:
    # +------+------+
    # |   3  |   4  |
    # +------+------+
    # |   1  |   2  |
    # +------+------+
    # 
    # Distributed grid:
    # +------+------+
    # | 2[3] | 1[2] |
    # +------+------+
    # | 1[1] | 1[3] |
    # +------+------+
    #
    # With the notation "a[b]" where
    # - a denotes the local element index 
    # - b denotes the rank
    #
    @testset "Quadrilateral" begin
        grid = generate_grid(Quadrilateral, (2,2))
        topo = CoverTopology(grid)
        dgrid = FerriteMPI.DistributedGrid(grid, topo, comm, Int32[1,3,3,2])

        my_rank = Ferrite.global_rank(dgrid)
        if my_rank == 1
            lgrid = getlocalgrid(dgrid)
            @test getncells(lgrid) == 1
            non_shared_vertices = [VertexIndex(1,1)]
            non_shared_faces = [FaceIndex(1,1), FaceIndex(1,4)]
        elseif my_rank == 2
            lgrid = getlocalgrid(dgrid)
            @test getncells(lgrid) == 1
            non_shared_vertices = [VertexIndex(1,3)]
            non_shared_faces = [FaceIndex(1,2), FaceIndex(1,3)]
        elseif my_rank == 3
            lgrid = getlocalgrid(dgrid)
            @test getncells(lgrid) == 2
            non_shared_vertices = [VertexIndex(1,2), VertexIndex(2,4)]
            non_shared_faces = [FaceIndex(1,1), FaceIndex(1,2), FaceIndex(2,3), FaceIndex(2,4)]
        else
            # Abstract machine or memory corruption during exectution above.
            @test false
        end

        for sv ∈ get_shared_vertices(dgrid)
            @test sv.local_idx ∉ Set(non_shared_vertices)
        end
        for v ∈ non_shared_vertices
            @test !is_shared_vertex(dgrid, v)
        end
        for sf ∈ get_shared_faces(dgrid)
            @test sf.local_idx ∉ Set(non_shared_faces)
        end
        for f ∈ non_shared_faces
            @test !is_shared_face(dgrid, f)
        end
    end

    # y
    # ^  z
    # | /
    # |/
    #  ----> x
    # Global grid:
    #     +------+------+
    #    /|   3  |   4  |
    #   + +------+------+
    #   |/|   1  |   2  |
    #   + +------+------+
    #   |/      /      /
    #   +------+------+
    #  
    # Distributed grid:
    #     +------+------+
    #    /| 2[3] | 1[2] |
    #   + +------+------+
    #   |/| 1[1] | 1[3] |
    #   + +------+------+
    #   |/      /      /
    #   +------+------+
    #
    # With the notation "a[b]" where
    # - a denotes the local element index 
    # - b denotes the rank
    #
    @testset "Hexahedron" begin
        grid = generate_grid(Hexahedron, (2,2,1))
        topo = CoverTopology(grid)
        dgrid = FerriteMPI.DistributedGrid(grid, topo, comm, Int32[1,3,3,2])

        my_rank = Ferrite.global_rank(dgrid)
        if my_rank == 1
            lgrid = getlocalgrid(dgrid)
            @test getncells(lgrid) == 1
            non_shared_vertices = [VertexIndex(1,1), VertexIndex(1,5)]
            non_shared_faces    = [FaceIndex(1,1), FaceIndex(1,6), FaceIndex(1,2), FaceIndex(1,5)]
            non_shared_edges    = [EdgeIndex(1,1), EdgeIndex(1,4), EdgeIndex(1,5), EdgeIndex(1,8), EdgeIndex(1,9)]
        elseif my_rank == 2
            lgrid = getlocalgrid(dgrid)
            @test getncells(lgrid) == 1
            non_shared_vertices = [VertexIndex(1,3), VertexIndex(1,7)]
            non_shared_faces    = [FaceIndex(1,1), FaceIndex(1,6), FaceIndex(1,3), FaceIndex(1,4)]
            non_shared_edges    = [EdgeIndex(1,2), EdgeIndex(1,6), EdgeIndex(1,3), EdgeIndex(1,7), EdgeIndex(1,11)]
        elseif my_rank == 3
            lgrid = getlocalgrid(dgrid)
            @test getncells(lgrid) == 2
            non_shared_vertices = [VertexIndex(1,2), VertexIndex(1,6), VertexIndex(2,4), VertexIndex(2,8)]
            non_shared_faces    = [FaceIndex(1,1), FaceIndex(1,6), FaceIndex(1,2), FaceIndex(1,3), FaceIndex(2,1), FaceIndex(2,6), FaceIndex(2,5), FaceIndex(2,4)]
            non_shared_edges    = [EdgeIndex(1,1), EdgeIndex(1,5), EdgeIndex(1,2), EdgeIndex(1,6), EdgeIndex(1,10), EdgeIndex(2,4), EdgeIndex(2,8), EdgeIndex(2,3), EdgeIndex(2,7), EdgeIndex(2,12)]
        else
            # Abstract machine or memory corruption during exectution above.
            @test false
        end

        for sv ∈ get_shared_vertices(dgrid)
            @test sv.local_idx ∉ Set(non_shared_vertices)
        end
        for v ∈ non_shared_vertices
            @test !is_shared_vertex(dgrid, v)
        end
        for sf ∈ get_shared_faces(dgrid)
            @test sf.local_idx ∉ Set(non_shared_faces)
        end
        for f ∈ non_shared_faces
            @test !is_shared_face(dgrid, f)
        end
        for se ∈ get_shared_edges(dgrid)
            @test se.local_idx ∉ Set(non_shared_edges)
        end
        for e ∈ non_shared_edges
            @test !is_shared_edge(dgrid, e)
        end
    end
end
