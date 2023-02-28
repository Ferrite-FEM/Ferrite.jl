```@meta
DocTestSetup = :(using Ferrite)
```

# Ferrite.jl
*A simple finite element toolbox written in Julia.*

## Introduction
`Ferrite` is a finite element toolbox that provides functionalities to implement finite element analysis in [Julia](https://github.com/JuliaLang/julia). The aim is to be general and to keep mathematical abstractions.
The main functionalities of the package include:

* Facilitate integration using different quadrature rules.
* Define different finite element interpolations.
* Evaluate shape functions, derivatives of shape functions etc. for the different interpolations and quadrature rules.
* Evaluate functions and derivatives in the finite element space.
* Generate simple grids.
* Export grids and solutions to VTK.

The best way to get started with `Ferrite` is to look at the documented examples.


!!! note

    `Ferrite` is still under development. If you find a bug, or have
    ideas for improvements, feel free to open an issue or make a
    pull request on the [`Ferrite` GitHub page](https://github.com/Ferrite-FEM/Ferrite.jl).

## Installation

You can install Ferrite from the Pkg REPL (press `]` in the Julia
REPL to enter `pkg>` mode):

```
pkg> add Ferrite
```

!!! note
    Alternative installation method:
    ```julia
    julia> import Pkg; Pkg.add("Ferrite")
    ```

To load the package, use

```julia
using Ferrite
```

## Debugging Information

There is a debug mode to aid the development of new algorithms in Ferrite, as well as to
help tracking down bugs. It can be turned on via

```julia
using Ferrite
Ferrite.debug_mode()
```

followed by restarting the Julia process. It can be turned off again by calling

```julia
using Ferrite
Ferrite.debug_mode(enable=false)
```

also followed by restarting the Julia process.

## Performance Analysis

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
    For the performance comparison between two commits you must not have any uncommited
    or untracked files in your Ferrite.jl folder! Otherwise the `PkgBenchmark.jl` will 
    fail to setup the comparison.

For more fine grained control you can run subsets of the benchmarks via by appending `-<subset>`
to compare or benchmark, e.g.

```
make benchmark-mesh
```

to benchmark only the mesh functionality. Here `subset` is either
* assembly
* boundary-conditions
* dofs
* mesh

!!! note
    It is recommended to run all benchmarks before running subsets to get the
    correct tuning parameters for each benchmark.
