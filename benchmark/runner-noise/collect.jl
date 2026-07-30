# One measurement job of the runner-noise experiment (.github/workflows/RunnerNoise.yml):
# run the benchmark suite PASSES times at a single commit and record the raw per-benchmark
# minima of every pass. Since every pass measures the same code, any difference between
# passes is runner noise, and the analysis (analyze.jl) can replay any tolerance/nruns
# policy against it offline.
#
# Each pass goes through Tachometer.run_revision, i.e. a fresh worktree + subprocess +
# BenchmarkTools tuning, so a pass here costs and behaves exactly like a production
# baseline-or-target run in Benchmarks.yml.
#
# Driven by env vars (set by the workflow):
#   NOISE_REPO    path to the package working copy (default ".")
#   NOISE_REF     git rev to benchmark (default GITHUB_SHA, else HEAD)
#   NOISE_PASSES  number of suite runs (default 8)
#   NOISE_OUT     output JSON path (default "runner-noise.json")
#   RUNNER_LABEL / RUNNER_REP  identify the matrix job in the output

using Tachometer
using JSON

repo = get(ENV, "NOISE_REPO", ".")
ref = get(ENV, "NOISE_REF", get(ENV, "GITHUB_SHA", "HEAD"))
npasses = parse(Int, get(ENV, "NOISE_PASSES", "8"))
outfile = get(ENV, "NOISE_OUT", "runner-noise.json")

runs = Vector{Any}()
for i in 1:npasses
    println("[runner-noise] pass $(i)/$(npasses)")
    t0 = time()
    r = Tachometer.run_revision(repo, ref, "benchmark/benchmarks.jl")
    r.ok || error("pass $i failed:\n" * r.log)
    push!(
        runs, Dict(
            "pass" => i,
            "wall_s" => time() - t0,
            "estimates" => Dict(
                k => Dict("time" => v.time, "memory" => v.memory, "allocs" => v.allocs)
                    for (k, v) in r.estimates
            ),
        )
    )
end

open(outfile, "w") do io
    JSON.print(
        io, Dict(
            "runner" => get(ENV, "RUNNER_LABEL", "unknown"),
            "rep" => get(ENV, "RUNNER_REP", "0"),
            "ref" => ref,
            "julia" => string(VERSION),
            "os" => string(Sys.KERNEL),
            "arch" => string(Sys.ARCH),
            "cpu" => Sys.cpu_info()[1].model,
            "runs" => runs,
        ), 2
    )
end
println("[runner-noise] wrote $(outfile)")
