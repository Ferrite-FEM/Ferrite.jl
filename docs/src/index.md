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

In Julia v1.0 (and v0.7) you can install Ferrite from the Pkg REPL (press `]` in the Julia
REPL to enter `pkg>` mode):

```
pkg> add https://github.com/Ferrite-FEM/Ferrite.jl.git
```

!!! note
    Alternative installation method:
    ```julia
    julia> import Pkg; Pkg.add(PackageSpec(url = "https://github.com/Ferrite-FEM/Ferrite.jl.git"))
    ```

To load the package, use

```julia
using Ferrite
```

!!! note
    In Julia v0.6 you need to checkout the `release-0.3` branch when installing:
    ```
    Pkg.clone("https://github.com/Ferrite-FEM/Ferrite.jl.git")
    Pkg.checkout("Ferrite", "release-0.3")
    ```
