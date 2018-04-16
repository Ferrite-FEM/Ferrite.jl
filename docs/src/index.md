```@meta
DocTestSetup = :(using JuAFEM)
```

# JuAFEM.jl
*A simple finite element toolbox written in Julia.*

## Introduction
`JuAFEM` is a finite element toolbox that provides functionalities to implement finite element analysis in [Julia](https://github.com/JuliaLang/julia). The aim is to be general and to keep mathematical abstractions.
The main functionalities of the package include:

* Facilitate integration using different quadrature rules.
* Define different finite element interpolations.
* Evaluate shape functions, derivatives of shape functions etc. for the different interpolations and quadrature rules.
* Evaluate functions and derivatives in the finite element space.
* Generate simple grids.
* Export grids and solutions to VTK.

The best way to get started with `JuAFEM` is to look at the documented examples.


!!! note

    `JuAFEM` is still under development. If you find a bug, or have
    ideas for improvements, feel free to open an issue or make a
    pull request on the [`JuAFEM` GitHub page](https://github.com/KristofferC/JuAFEM.jl).

## Installation

To install, simply run the following in the Julia REPL:
```julia
Pkg.clone("https://github.com/KristofferC/JuAFEM.jl")
```
and then run
```julia
using JuAFEM
```
to load the package.