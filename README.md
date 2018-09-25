# JuAFEM

[![Build Status](https://travis-ci.org/KristofferC/JuAFEM.jl.svg?branch=master)](https://travis-ci.org/KristofferC/JuAFEM.jl)
[![Build status](https://ci.appveyor.com/api/projects/status/e93hh431emoj5410?svg=true)](https://ci.appveyor.com/project/KristofferC/juafem-jl)
[![codecov.io](http://codecov.io/github/KristofferC/JuAFEM.jl/coverage.svg?branch=master)](http://codecov.io/github/KristofferC/JuAFEM.jl?branch=master)

A simple finite element toolbox written in Julia.

## Documentation

[![][docs-dev-img]][docs-dev-url]

## Installation
In Julia v1.0 (and v0.7) you can install JuAFEM from the Pkg REPL:
```
pkg> add https://github.com/KristofferC/JuAFEM.jl.git
```
which will track the `master` branch of the package.

In Julia v0.6 you need to checkout the `release-0.3` branch when installing:
```
Pkg.clone("https://github.com/KristofferC/JuAFEM.jl.git")
Pkg.checkout("JuAFEM", "release-0.3")
```

[docs-dev-img]: https://img.shields.io/badge/docs-dev-blue.svg
[docs-dev-url]: http://kristofferc.github.io/JuAFEM.jl/dev/
