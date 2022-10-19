using PkgBenchmark

benchmarkpkg(
    dirname(@__DIR__),
    BenchmarkConfig(
        env = Dict(
            "JULIA_NUM_THREADS" => "1",
            "OMP_NUM_THREADS" => "1",
        )
    ),
    resultfile = joinpath(@__DIR__, "result.json"),
)
