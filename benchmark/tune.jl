# Regenerate benchmark/tune.json, the committed `evals` parameters for the suite. Invoke with
# `make tune`. See the comment at the bottom of benchmarks.jl for why these are committed.
using BenchmarkTools

const paramsfile = joinpath(@__DIR__, "tune.json")

# Tune from scratch rather than from the currently committed values.
isfile(paramsfile) && rm(paramsfile)

include(joinpath(@__DIR__, "benchmarks.jl"))

@info "Tuning $(length(BenchmarkTools.leaves(SUITE))) benchmarks, this takes a few minutes..."
tune!(SUITE)
BenchmarkTools.save(paramsfile, BenchmarkTools.params(SUITE))
@info "Wrote $(paramsfile)"
