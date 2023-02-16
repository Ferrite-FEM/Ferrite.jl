using Test, Ferrite, MPI

# @testset "dof distribution" begin
#     MPI.Init()
#     my_rank = MPI.Comm_rank(MPI.COMM_WORLD)+1

#     dim = 2
#     ref = RefCube
#     ip = Lagrange{dim, ref, 1}()
#     global_grid = generate_grid(Quadrilateral, (3, 3))
#     global_topology = ExclusiveTopology(global_grid)
#     dgrid = DistributedGrid(global_grid, global_topology, MPI.COMM_WORLD, Int32[3,3,4,2,5,4,1,2,5])

#     dh = DistributedDofHandler(dgrid)
#     push!(dh, :u, 1, ip)
#     close!(dh);

#     @test length(dh.ldof_to_gdof) == length(dh.ldof_to_rank)
#     if my_rank == 1
#         @test dh.ldof_to_gdof == [1,2,3,4]
#         @test dh.ldof_to_rank == [1,1,1,1]
#     elseif my_rank == 2
#         @test dh.ldof_to_gdof == [5,6,2,1,7,8,3]
#         @test dh.ldof_to_rank == [2,2,1,1,2,2,1]
#     elseif my_rank == 3
#         @test dh.ldof_to_gdof == [9,10, 6, 5,11,12]
#         @test dh.ldof_to_rank == [3, 3, 2, 2, 3, 3]
#     elseif my_rank == 4
#         @test dh.ldof_to_gdof == [11,13,14,12,15, 7]
#         @test dh.ldof_to_rank == [ 3, 4, 4, 3, 4, 2]
#     elseif my_rank == 5
#         @test dh.ldof_to_gdof == [6,12, 7, 2,15,16, 8]
#         @test dh.ldof_to_rank == [2, 3, 2, 1, 4, 5, 2]
#     end
#     MPI.Finalize()
# end

@testset "distributed grid generation" begin
    MPI.Init()
    my_rank = MPI.Comm_rank(MPI.COMM_WORLD)+1

    global_grid = generate_grid(Hexahedron, (2, 1, 1))
    global_topology = ExclusiveTopology(global_grid)
    dgrid = DistributedGrid(global_grid, global_topology, MPI.COMM_WORLD, Int32[2, 1])
    if my_rank == 1
        @test length(Ferrite.get_shared_edges(dgrid)) == 4
        function check_edge_correctly_shared_1(idx_local, idx_nonlocal)
            se = Ferrite.get_shared_edge(dgrid, idx_local)
            @test Ferrite.remote_entities(se) == Dict(2 => [idx_nonlocal])
        end
        check_edge_correctly_shared_1(EdgeIndex(1,4), EdgeIndex(1,2))
        check_edge_correctly_shared_1(EdgeIndex(1,9), EdgeIndex(1,10))
        check_edge_correctly_shared_1(EdgeIndex(1,12), EdgeIndex(1,11))
        check_edge_correctly_shared_1(EdgeIndex(1,8), EdgeIndex(1,6))
    elseif my_rank == 2
        @test length(Ferrite.get_shared_edges(dgrid)) == 4
        function check_edge_correctly_shared_2(idx_nonlocal, idx_local)
            se = Ferrite.get_shared_edge(dgrid, idx_local)
            @test Ferrite.remote_entities(se) == Dict(1 => [idx_nonlocal])
        end
        check_edge_correctly_shared_2(EdgeIndex(1,4), EdgeIndex(1,2))
        check_edge_correctly_shared_2(EdgeIndex(1,9), EdgeIndex(1,10))
        check_edge_correctly_shared_2(EdgeIndex(1,12), EdgeIndex(1,11))
        check_edge_correctly_shared_2(EdgeIndex(1,8), EdgeIndex(1,6))
    end
    MPI.Finalize()
end
