# [Performance analysis](@id devdocs-performance)

Ferrite maintains a benchmark suite in the `benchmark/` folder that covers meshes, dof
distribution, `FEValues`, assembly, constraints, sparsity patterns, postprocessing and
adaptive mesh refinement. It is run automatically on CI and can be run locally, both to
benchmark a single revision and to compare two revisions against each other.

## Benchmark dashboard

Every push to `master` runs the full suite and records the results to the benchmark
dashboard at <https://ferrite-fem.github.io/Ferrite.jl/benchmarks/>. The dashboard shows
the time series of every benchmark over the history of `master`, with releases marked, and
is the first place to look when tracking down *when* a performance regression was
introduced.

## Benchmarks on pull requests

Every pull request against `master` is benchmarked by
[Tachometer.jl](https://github.com/KristofferC/Tachometer.jl): the suite is run on both the
PR and its merge base with interleaved passes, and the comparison is posted as a PR
comment. Regressions are judged against a per-benchmark noise band derived from the
dashboard time series, so a red entry in the report means the difference cleared the
historical run-to-run noise of that particular benchmark. Docs-only changes skip the
benchmark run.

## Running the benchmarks locally

Local benchmarking also goes through
[Tachometer.jl](https://github.com/KristofferC/Tachometer.jl) -- the same tool that runs the
suite on CI -- so the report matches what a PR comment will show. From the repository root:

```julia
using Tachometer

# compare the working tree (including uncommitted changes) against HEAD
compare()

# compare the current branch against master, or two explicit revisions
compare(; baseline = "master")
compare(; baseline = "<baseline-commit>", target = "<target-commit>")
```

On Julia 1.12+ it can also be installed as a shell command via
`pkg> app add https://github.com/KristofferC/Tachometer.jl`, after which

```sh
tachometer                    # compare HEAD vs the working tree
tachometer --baseline=master
```

does the same thing from the shell: it prints a one-line summary, writes the full Markdown
report to `tachometer-report/` (configurable with `--output-dir`), and exits non-zero on a
regression.

Set the environment variable `FERRITE_SELECTED_BENCHMARKS=<group>` to restrict a run to a
single group. The following groups are currently available: `mesh`, `dofs`, `fevalues`,
`assembly`, `constraints`, `sparsity-pattern`, `postprocessing`, and `amr`. See the table in
[`benchmark/README.md`](https://github.com/Ferrite-FEM/Ferrite.jl/blob/master/benchmark/README.md)
for what each group covers.

To run the suite without a comparison, e.g. while writing a benchmark, activate the
`benchmark/` project (with the parent Ferrite checkout `Pkg.develop`ed into it), include
`benchmarks.jl`, and `run(SUITE)` (or `run(SUITE["<group>"])`).

## Adding benchmarks

The design principles of the suite -- covering structurally different code paths rather
than sweeping all parameter combinations, keeping every benchmark above the CI noise floor,
and staying within the CI runtime budget -- are documented in
[`benchmark/README.md`](https://github.com/Ferrite-FEM/Ferrite.jl/blob/master/benchmark/README.md).
Read it before adding benchmarks.
