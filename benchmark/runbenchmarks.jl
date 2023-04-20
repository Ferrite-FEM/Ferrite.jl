if !(length(ARGS) == 1 || length(ARGS) == 2)
    error("Usage: runbenchmarks.jl <script> <benchmarks>")
end

using PkgBenchmark

benchmarkpkg(
    dirname(@__DIR__),
    BenchmarkConfig(
        env = Dict(
            # Selected benchmarks
            "FERRITE_SELECTED_BENCHMARKS" => get(ARGS, 2, "all"),
            # Julia algorithms run in serial
            "JULIA_NUM_THREADS" => "1",
            # External solvers run in serial
            "OMP_NUM_THREADS" => "1",
        )
    );
    resultfile = joinpath(@__DIR__, "result.json"),
    script = ARGS[1],
)
