# Entry point for the benchmark suite, defining `SUITE` as expected by Tachometer.
# See benchmark/README.md for the design of the suite.
#
# The suite deliberately does NOT sweep over all combinations of element type,
# interpolation, order etc. -- each benchmark group picks the few configurations that
# exercise structurally different code paths for the functionality it covers. Per-cell
# operations are measured over a batch of cells, because Tachometer discards differences
# below ~1 us as noise (`time-floor` in .github/workflows/Benchmarks.yml); a benchmark
# below that floor can never report a regression and only wastes CI time. Keep every
# benchmark above ~5 us and below ~10 ms where possible.

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
if runall || selected == "fevalues"
    include("benchmarks-fevalues.jl")
end
if runall || selected == "assembly"
    include("benchmarks-assembly.jl")
end
if runall || selected == "constraints"
    include("benchmarks-constraints.jl")
end
if runall || selected == "sparsity-pattern"
    include("benchmarks-sparsity-pattern.jl")
end
if runall || selected == "postprocessing"
    include("benchmarks-postprocessing.jl")
end
if runall || selected == "amr"
    include("benchmarks-amr.jl")
end

# Runtime is controlled by explicit parameters instead of tuning:
#  - `evals = 1` is declared on every benchmark (everything measures well above the timer
#    resolution). This also marks `evals_set`, which makes Tachometer skip tuning
#    entirely.
#  - `samples`/`seconds` default to 1000/0.5 s below, which bounds a full pass by roughly
#    0.5 s per benchmark. Benchmarks that declare other values (e.g. `seconds = 1.0` for
#    the ones with millisecond runtimes or expensive `setup`) keep them.
#  - `gctrial = false` because a full `gcscrub()` before every benchmark costs ~0.7 s each
#    and would dominate the suite; the minimum estimator already discards samples that a
#    GC pause landed in.
for (_, b) in BenchmarkTools.leaves(SUITE)
    p = b.params
    p.gctrial = false
    if !p.evals_set
        error("benchmark without explicit `evals`; declare `evals = 1` (see the comment above)")
    end
    if p.seconds == BenchmarkTools.DEFAULT_PARAMETERS.seconds
        p.seconds = 0.5
    end
    if p.samples == BenchmarkTools.DEFAULT_PARAMETERS.samples
        p.samples = 1000
    end
end
