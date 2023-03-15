if !(length(ARGS) == 3 || length(ARGS) == 4)
    error("Usage: runcomparison.jl <target-commit> <baseline-commit> <script> <benchmarks>")
end

using BenchmarkTools, PkgBenchmark

const env = Dict(
    # Selected benchmarks
    "FERRITE_SELECTED_BENCHMARKS" => get(ARGS, 4, "all"),
    # Julia algorithms run in serial
    "JULIA_NUM_THREADS" => "1",
    # External solvers run in serial
    "OMP_NUM_THREADS" => "1",
)

BenchmarkTools.judge("..",
    BenchmarkConfig(; id = ARGS[1], env = env),
    BenchmarkConfig(; id = ARGS[2], env = env);
    script = ARGS[3],
)
