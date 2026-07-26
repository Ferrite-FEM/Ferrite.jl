using BenchmarkTools
using Ferrite

const selected = get(ENV, "FERRITE_SELECTED_BENCHMARKS", "all")
const runall = selected == "all"

include("helper.jl")

const SUITE = BenchmarkGroup()

if runall || selected == "mesh"
    include("benchmarks-mesh.jl")
end
if runall || selected == "dofs"
    include("benchmarks-dofs.jl")
end
if runall || selected == "assembly"
    include("benchmarks-assembly.jl")
end
if runall || selected == "boundary-conditions"
    include("benchmarks-boundary-conditions.jl")
end
if runall || selected == "sparsity-pattern"
    include("benchmarks-sparsity-pattern.jl")
end

# `gctrial` runs a full `gcscrub()` before every benchmark, which costs ~0.7 s each and thus
# dominates a suite of this size (~300 of the ~340 benchmarks measure well under a
# millisecond). Disabling it cuts a full pass from ~410 s to ~120 s; the minimum estimator
# already discards samples that a GC pause landed in. The budget is capped for the same
# reason -- 1 s is far more than these need.
for (_, b) in BenchmarkTools.leaves(SUITE)
    b.params.seconds = min(b.params.seconds, 1.0)
    b.params.samples = min(b.params.samples, 1000)
    b.params.gctrial = false
end

# `evals` is expensive to derive (`tune!` over the whole suite takes ~5 min, more than twice
# what running it costs) but changes rarely, so it is committed in tune.json and loaded here.
# Regenerate with `make tune` after adding or substantially changing benchmarks.
let paramsfile = joinpath(@__DIR__, "tune.json")
    isfile(paramsfile) && loadparams!(SUITE, BenchmarkTools.load(paramsfile)[1], :evals, :samples)
end
