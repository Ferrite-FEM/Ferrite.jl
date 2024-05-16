<picture>
  <source media="(prefers-color-scheme: light)" srcset="https://github.com/Ferrite-FEM/Ferrite.jl/blob/master/docs/src/assets/logo-horizontal.svg">
  <source media="(prefers-color-scheme: dark)" srcset="https://github.com/Ferrite-FEM/Ferrite.jl/blob/master/docs/src/assets/logo-horizontal-dark.svg">
  <img alt="Ferrite.jl logo." src="https://github.com/Ferrite-FEM/Ferrite.jl/blob/master/docs/src/assets/logo-horizontal.svg">
</picture>

![Build Status](https://github.com/Ferrite-FEM/Ferrite.jl/workflows/CI/badge.svg?event=push)
[![codecov.io](http://codecov.io/github/Ferrite-FEM/Ferrite.jl/coverage.svg?branch=master)](http://codecov.io/github/Ferrite-FEM/Ferrite.jl?branch=master)

A simple finite element toolbox written in Julia.

**Note:** This package was originally called JuAFEM.jl, but has now been renamed to Ferrite.jl.

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

If you have questions about Ferrite.jl you're welcome to reach out to us on the [Julia
Slack][julia-slack] under `#ferrite-fem` or on [Zulip][julia-zulip] under `#Ferrite.jl`.
Alternatively you can start a [new discussion][gh-discussion] in the discussion forum on the
repository. Feel free to ask us even if you are not sure the problem is with Ferrite.jl.

If you encounter what you think is a bug please report it, see
[CONTRIBUTING.md](CONTRIBUTING.md#reporting-issues) for more information.

## Community Standards

Please keep in mind that we are part of the Julia community and adhere to the
[Julia Community Standards][standards].

## Related packages
The following registered packages are part of the `Ferrite.jl` ecosystem in addition to Ferrite itself:
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
