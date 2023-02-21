using Ferrite, MPI, Metis
using Test

MPI.Init()
@testset "setup check 2" begin
    @test MPI.Comm_size(MPI.COMM_WORLD) == 5
end

FerriteMPI = Base.get_extension(Ferrite, :FerriteMPI)

# We arbitrarily test a quite hard case with many corner 
# cases in 2D to catch regressions.
@testset "distributed dof distribution 5" begin
    # We do not cover subcommunicators for now.
    comm = MPI.COMM_WORLD

    dim = 2
    ref = RefCube
    ip = Lagrange{dim, ref, 1}()
    global_grid = generate_grid(Quadrilateral, (3, 3))
    global_topology = CoverTopology(global_grid)
    dgrid = FerriteMPI.DistributedGrid(global_grid, global_topology, comm, Int32[3,3,4,2,5,4,1,2,5])
    my_rank = Ferrite.global_rank(dgrid)
    
    dh = DofHandler(dgrid)
    push!(dh, :u, 1, ip)
    close!(dh);

    @test length(dh.ldof_to_gdof) == length(dh.ldof_to_rank)
    if my_rank == 1
        @test dh.ldof_to_gdof == [1,2,3,4]
        @test dh.ldof_to_rank == [1,1,1,1]
    elseif my_rank == 2
        @test dh.ldof_to_gdof == [5,6,2,1,7,8,3]
        @test dh.ldof_to_rank == [2,2,1,1,2,2,1]
    elseif my_rank == 3
        @test dh.ldof_to_gdof == [9,10, 6, 5,11,12]
        @test dh.ldof_to_rank == [3, 3, 2, 2, 3, 3]
    elseif my_rank == 4
        @test dh.ldof_to_gdof == [11,13,14,12,15, 7]
        @test dh.ldof_to_rank == [ 3, 4, 4, 3, 4, 2]
    elseif my_rank == 5
        @test dh.ldof_to_gdof == [6,12, 7, 2,15,16, 8]
        @test dh.ldof_to_rank == [2, 3, 2, 1, 4, 5, 2]
    end
end
