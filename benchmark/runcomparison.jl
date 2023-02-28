if length(ARGS) != 3
    @error "Usage: runcomparison.jl <target-commit> <baseline-commit> <script>"
    exit(-1)
end

using BenchmarkTools, PkgBenchmark

env = Dict(
            # Julia algorithms run in serial
            "JULIA_NUM_THREADS" => "1",
            # External solvers run in serial
            "OMP_NUM_THREADS" => "1",
        )

BenchmarkTools.judge("..", 
    BenchmarkConfig(;id=ARGS[1], env=env),
    BenchmarkConfig(;id=ARGS[2], env=env);
    script=ARGS[3]
)
