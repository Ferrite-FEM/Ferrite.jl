# Aggregate the collect.jl outputs of the runner-noise experiment into a per-runner
# noise profile, written as markdown. Reads every *.json in NOISE_RESULTS (default
# "results") and writes NOISE_SUMMARY (default "summary.md").
#
# All passes in a job measured the same commit, so:
#   * "pair |Δ|" — relative time change between consecutive passes, the raw
#     run-to-run jitter of a benchmark on that runner. Only changes that clear the
#     production 1 µs floor are counted; sub-floor wiggles can never be flagged.
#   * "false flags" — the production Tachometer policy (5% tolerance, 1 µs floor,
#     nruns=2 with both pairs required to agree) replayed on consecutive blocks of
#     4 passes: (b₁ t₁ b₂ t₂). Every flag is a false positive by construction.
#   * "zero-FP tol" — the smallest tolerance at which this data would have produced
#     no false flags under the nruns=2 policy: the largest direction-consistent,
#     floor-clearing excess that any benchmark showed in both pairs of a block.
#   * "drift" — per pass, the median time ratio against pass 1 across all
#     benchmarks: whole-machine speed change within a job (tenancy/thermal), which
#     interleaving alone cannot fully absorb.

using JSON
using Statistics: median, quantile
using Printf: @sprintf

const TOL = 0.05          # production time tolerance
const FLOOR_NS = 1000.0   # production time floor

resultsdir = get(ENV, "NOISE_RESULTS", "results")
outfile = get(ENV, "NOISE_SUMMARY", "summary.md")

jobs = [JSON.parsefile(joinpath(resultsdir, f)) for f in readdir(resultsdir) if endswith(f, ".json")]
isempty(jobs) && error("no result JSONs found in $(resultsdir)")

pct(x) = @sprintf("%.1f%%", 100 * x)

# The per-pass minimum time of each benchmark, for benchmarks present in every pass.
function job_times(job)
    runs = sort(job["runs"]; by = r -> r["pass"])
    names = intersect([Set(keys(r["estimates"])) for r in runs]...)
    return Dict(name => [Float64(r["estimates"][name]["time"]) for r in runs] for name in names)
end

# Production verdict for one simulated comparison from two (baseline, target) pairs:
# flagged only if both pairs agree in direction, clear the tolerance, and clear the floor.
function flagged(b1, t1, b2, t2, tol)
    red(b, t) = t / b > 1 + tol && t - b > FLOOR_NS
    grn(b, t) = b / t > 1 + tol && b - t > FLOOR_NS
    return (red(b1, t1) && red(b2, t2)) || (grn(b1, t1) && grn(b2, t2))
end

# The excess that survives the nruns=2 agreement gate for one block: the smaller of the
# two pairs' relative changes when both pairs move the same way and clear the floor, else
# 0. The max of this over all benchmarks/blocks is the tolerance needed for zero flags.
function agreed_excess(b1, t1, b2, t2)
    r1, r2 = t1 / b1 - 1, t2 / b2 - 1
    sign(r1) == sign(r2) || return 0.0
    min(abs(t1 - b1), abs(t2 - b2)) > FLOOR_NS || return 0.0
    return min(abs(r1), abs(r2))
end

rows = String[]
for runner in sort(unique(j["runner"] for j in jobs))
    group = filter(j -> j["runner"] == runner, jobs)
    cpus = sort(unique(j["cpu"] for j in group))
    walls = [r["wall_s"] for j in group for r in j["runs"]]

    pair_deltas = Float64[]     # |Δ| between consecutive passes, floor-clearing only
    excesses = Float64[]        # agreement-gated excess per benchmark and block
    nflags = 0                  # false flags under the production policy
    ncomparisons = 0            # simulated nruns=2 comparisons
    drifts = Float64[]          # max whole-suite drift within each job

    for job in group
        times = job_times(job)
        n = length(first(values(times)))
        for t in values(times)
            for i in 1:(n - 1)
                abs(t[i + 1] - t[i]) > FLOOR_NS && push!(pair_deltas, abs(t[i + 1] / t[i] - 1))
            end
        end
        for lo in 1:4:(n - 3)  # disjoint blocks of 4 passes: b1 t1 b2 t2
            ncomparisons += 1
            for t in values(times)
                b1, t1, b2, t2 = t[lo], t[lo + 1], t[lo + 2], t[lo + 3]
                nflags += flagged(b1, t1, b2, t2, TOL)
                push!(excesses, agreed_excess(b1, t1, b2, t2))
            end
        end
        push!(drifts, maximum(abs(median(t[i] / t[1] for t in values(times)) - 1) for i in 2:n))
    end

    deltas = isempty(pair_deltas) ? "all < floor" :
        "$(pct(median(pair_deltas))) / $(pct(quantile(pair_deltas, 0.95))) / $(pct(maximum(pair_deltas)))"
    push!(
        rows,
        "| `$(runner)` | $(length(group)) | $(join(cpus, "<br>")) | $(round(Int, median(walls))) s " *
            "| $(deltas) " *
            "| $(nflags) in $(ncomparisons) | $(pct(maximum(excesses))) | $(pct(maximum(drifts))) |"
    )
end

open(outfile, "w") do io
    println(io, "## Runner-noise experiment — same-commit (null) comparisons\n")
    println(io, "Every flag is a false positive: all passes measured the same commit.\n")
    println(io, "| Runner | Jobs | CPU(s) | Pass wall | Pair \\|Δt\\| p50/p95/max | False flags @ prod policy | Zero-FP tolerance | Max drift |")
    println(io, "|:--|--:|:--|--:|:--|:--|--:|--:|")
    foreach(r -> println(io, r), rows)
    println(io, "\n- **Pair |Δt|** — run-to-run change of a benchmark between consecutive passes (floor-clearing only).")
    println(io, "- **False flags @ prod policy** — benchmark flags under the production 5% / 1 µs / nruns=2 rule, per simulated comparisons.")
    println(io, "- **Zero-FP tolerance** — smallest time tolerance that would have produced no flags on this data (nruns=2).")
    println(io, "- **Max drift** — largest whole-suite median shift relative to the first pass within a job.")
end
println("wrote $(outfile)")
