<picture>
  <source media="(prefers-color-scheme: light)" srcset="https://github.com/Ferrite-FEM/Ferrite.jl/blob/master/docs/src/assets/logo-horizontal.svg">
  <source media="(prefers-color-scheme: dark)" srcset="https://github.com/Ferrite-FEM/Ferrite.jl/blob/master/docs/src/assets/logo-horizontal-dark.svg">
  <img alt="Ferrite.jl logo." src="https://github.com/Ferrite-FEM/Ferrite.jl/blob/master/docs/src/assets/logo-horizontal.svg">
</picture>

[![Test Status](https://github.com/Ferrite-FEM/Ferrite.jl/actions/workflows/Test.yml/badge.svg?branch=master&event=push)](https://github.com/Ferrite-FEM/Ferrite.jl/actions/workflows/Test.yml)
[![codecov.io](https://codecov.io/github/Ferrite-FEM/Ferrite.jl/coverage.svg?branch=master)](https://codecov.io/github/Ferrite-FEM/Ferrite.jl?branch=master)
[![code style: runic](https://img.shields.io/badge/code_style-%E1%9A%B1%E1%9A%A2%E1%9A%BE%E1%9B%81%E1%9A%B2-black)](https://github.com/fredrikekre/Runic.jl)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.13862652.svg)](https://doi.org/10.5281/zenodo.13862652)

Ferrite.jl is a comprehensive toolbox for finite element analysis (FEA) in Julia. It is designed for researchers and engineers who want full control over their simulation pipelines—from the weak form definition to the assembly and solution strategies. If you want to implement your own finite element solver, perform novel multiphysics simulations, or experiment with new element types, Ferrite provides the high-performance building blocks you need.

## Scope & Core Features

Ferrite provides a general, performant, and mathematically abstract foundation for solving partial differential equations (PDEs). Key features include:

*   **Versatile Grid Management**: Support for 1D, 2D, and 3D grids with mixed element types.
*   **Flexible Degree of Freedom Handling**: Robust `DofHandler` for scalar and vector fields, supporting mixed interpolations and multiple fields.
*   **Efficient Assembly**: Optimised routines for iterating over cells and faces, calculating shape functions, gradients, and numerical integration.
*   **Boundary Conditions**: Easy definition of Dirichlet, Neumann, and Periodic boundary conditions.
*   **High Performance**: Design that leverages Julia's multiple dispatch and type system for speed comparable to C/C++.
*   **Visualization**: Native export to VTK (Paraview/VisIt) and integration with [FerriteViz.jl](https://github.com/Ferrite-FEM/FerriteViz.jl) for Makie-based visualization.

## When NOT to use Ferrite

While Ferrite is powerful, it might not be the right tool if:

*   You are looking for a commercial-style "black box" solver with a graphical user interface (GUI) to merely configure and run standard simulations.
*   You require a library with a vast built-in catalogue of material models or structural components (Ferrite focuses on the *engine*, letting you define the physics).
*   You need complex built-in mesh generation. (Ferrite handles grid data structures excellently but relies on external tools like [Gmsh](https://gmsh.info/) (via [FerriteGmsh.jl](https://github.com/Ferrite-FEM/FerriteGmsh.jl)) for generating complex geometries).

## Documentation

[![][docs-stable-img]][docs-stable-url]

## Installation
You can install Ferrite from the Pkg REPL:
```
pkg> add Ferrite
```

## Contributing

Contributions in all forms (bug reports, documentation, features, suggestions, ...) are very
welcome. See [CONTRIBUTING](CONTRIBUTING.md) for more details.

## Questions

If you have questions about Ferrite you're welcome to reach out to us on the [Julia
Slack][julia-slack] under `#ferrite-fem` or on [Zulip][julia-zulip] under `#Ferrite.jl`.
Alternatively you can start a [new discussion][gh-discussion] in the discussion forum on the
repository. Feel free to ask us even if you are not sure the problem is with Ferrite.

If you encounter what you think is a bug please report it, see
[CONTRIBUTING.md](CONTRIBUTING.md#reporting-issues) for more information.

## Community Standards

Please keep in mind that we are part of the Julia community and adhere to the
[Julia Community Standards][standards].

## Related packages
The following registered packages are part of the Ferrite ecosystem in addition to Ferrite itself:
* [Tensors.jl][Tensors]: Used throughout Ferrite for efficient tensor manipulation.
* [FerriteViz.jl][FerriteViz]: [Makie.jl][Makie]-based visualization of Ferrite data.
* [FerriteGmsh.jl][FerriteGmsh]: Create, interact with, and import [Gmsh][Gmsh] meshes into Ferrite.
* [FerriteMeshParser.jl][FerriteMeshParser]: Parse the mesh from Abaqus input files into a Ferrite mesh.


[docs-stable-img]: https://img.shields.io/badge/docs-latest%20release-blue
[docs-stable-url]: http://ferrite-fem.github.io/Ferrite.jl/

[standards]: https://julialang.org/community/standards/
[julia-slack]: https://julialang.org/slack/
[julia-zulip]: https://julialang.zulipchat.com/
[gh-discussion]: https://github.com/Ferrite-FEM/Ferrite.jl/discussions/new

[Tensors]: https://github.com/Ferrite-FEM/Tensors.jl
[FerriteViz]: https://github.com/Ferrite-FEM/FerriteViz.jl
[FerriteGmsh]: https://github.com/Ferrite-FEM/FerriteGmsh.jl
[FerriteMeshParser]: https://github.com/Ferrite-FEM/FerriteMeshParser.jl
[Makie]: https://docs.makie.org/stable/
[Gmsh]: https://gmsh.info/
