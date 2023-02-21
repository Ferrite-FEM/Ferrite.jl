using MPI
using Test

@testset "FerriteMPI n=2" begin
    n = 2  # number of processes
    mpiexec() do exe  # MPI wrapper
        run(`$exe -n $n $(Base.julia_cmd()) test_distributed_impl_2.jl`)
    end
end

@testset "FerriteMPI n=3" begin
    n = 3  # number of processes
    mpiexec() do exe  # MPI wrapper
        run(`$exe -n $n $(Base.julia_cmd()) test_distributed_impl_3.jl`)
    end
end

@testset "FerriteMPI n=5" begin
    n = 5  # number of processes
    mpiexec() do exe  # MPI wrapper
        run(`$exe -n $n $(Base.julia_cmd()) test_distributed_impl_5.jl`)
    end
end
