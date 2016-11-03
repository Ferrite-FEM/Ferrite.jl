# JuAFEM.jl

*A simple finite element toolbox written in Julia.*

## Introduction
JuAFEM is a finite element toolbox that provides functionalities to implement finite element analysis in [Julia](https://github.com/JuliaLang/julia). The aim is to be general and to keep mathematical abstractions.
The main functionalities of the package includes:

* Facilitate integration using different quadrature rules.
* Define different finite element spaces
* Evaluate shape functions, derivatives of shape functions etc. for the different finite element spaces and quadrature rules.
* Evaluate functions and derivatives in the finite element space.
* Generate simple grids, importing grids from common file formats.
* Facilitate degree of freedom numbering and keeping track of different fields.
* Export grids and solutions to VTK.

The types and functionalities of JuAFEM is described in more detail in the manual, see below.

JuAFEM is still under heavy development. If you find a bug, or have ideas on how to improve the package, feel free to open an issue or to make a pull request on the [JuAFEM GitHub page](https://github.com/KristofferC/JuAFEM.jl).

## Installation

To install, simply run the following in the Julia REPL:

    Pkg.clone("https://github.com/KristofferC/JuAFEM.jl")

and then run

    using JuAFEM

to load the package.


```@contents
Pages = ["man/basic_types.md"]
```


## API

```@contents
Pages = ["lib/maintypes.md", "lib/utility_functions.md"]
Depth = 2
```
