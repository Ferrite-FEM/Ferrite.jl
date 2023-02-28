using PkgBenchmark

benchmarkpkg(
    dirname(@__DIR__),
    BenchmarkConfig(
        env = Dict(
            # Julia algorithms run in serial
            "JULIA_NUM_THREADS" => "1",
            # External solvers run in serial
            "OMP_NUM_THREADS" => "1",
        )
    ),
    resultfile = joinpath(@__DIR__, "result.json"),
)
