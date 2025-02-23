# [Performance analysis](@id devdocs-performance)

In the benchmark folder we provide basic infrastructure to analyze the performance of
Ferrite to help tracking down performance regression issues. Two basic tools can be
directly executed via make: A basic benchmark for the current branch and a comparison
between two commits. To execute the benchmark on the current branch only open a shell
in the benchmark folder and call

```
make benchmark
```

whereas for the comparison of two commits simply call

```
make compare target=<target-commit> baseline=<baseline-commit>
```

If you have a custom julia executable that is not accessible via the `julia` command,
then you can pass the executable via

```
JULIA_CMD=<path-to-julia-executable> make compare target=<target-commit> baseline=<baseline-commit>
```

!!! note
    For the performance comparison between two commits you must not have any uncommitted
    or untracked files in your Ferrite.jl folder! Otherwise the `PkgBenchmark.jl` will
    fail to setup the comparison.

For more fine grained control you can run subsets of the benchmarks via by appending `-<subset>`
to compare or benchmark, e.g.

```
make benchmark-mesh
```

to benchmark only the mesh functionality. The following subsets are currently available:
 - `assembly`
 - `boundary-conditions`
 - `dofs`
 - `mesh`

!!! note
    It is recommended to run all benchmarks before running subsets to get the
    correct tuning parameters for each benchmark.
