# Regenerate benchmark/tune.json, the committed parameters for the suite. Invoke with
# `make tune`. Every benchmark declares `evals` explicitly (see the comment at the bottom of
# benchmarks.jl), so `tune!` is a no-op and this just snapshots the declared parameters --
# the file only exists because the CI runner falls back to a slow `tune!` without it.
using BenchmarkTools

const paramsfile = joinpath(@__DIR__, "tune.json")

# Tune from scratch rather than from the currently committed values.
isfile(paramsfile) && rm(paramsfile)

include(joinpath(@__DIR__, "benchmarks.jl"))

@info "Snapshotting parameters for $(length(BenchmarkTools.leaves(SUITE))) benchmarks..."
tune!(SUITE)
BenchmarkTools.save(paramsfile, BenchmarkTools.params(SUITE))
open(paramsfile, "a") do io # `save` does not write one, and pre-commit requires it
    write(io, "\n")
end
@info "Wrote $(paramsfile)"
