# Ferrite.jl changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]
### Added
- Functionality for querying information about the grid topology (e.g. neighboring cells,
  boundaries, ...). ([#363][github-363])
### Fixed
- Fix application of boundary conditions when combining RHSData and affine constraints.
  ([#431][github-431])


## [0.3.4] - 2022-02-25
### Added
- Affine (linear) constraints between degrees-of-freedom. ([#401][github-401])
- Periodic Dirichlet boundary conditions. ([#418][github-418])
- Evaluation of arbitrary quantities in FE space. ([#425][github-425])
### Changed
- Interpolation(s) and the quadrature rule are now stored as part of the `CellValues`
  structs (`cv.func_interp`, `cv.geo_interp`, and `cv.qr`). ([#428][github-428])

## [0.3.3] - 2022-02-04
### Changed
- Verify user input in various functions to eliminate possible out-of-bounds accesses.
  ([#407][github-407], [#411][github-411])

## [0.3.2] - 2022-01-18
### Added
- Support for new interpolation types: `DiscontinuousLagrange`, `BubbleEnrichedLagrange`,
  and `CrouzeixRaviart`. ([#352][github-352], [#392][github-392])
### Changed
- Julia version 1.0 is no longer supported for Ferrite versions >= 0.3.2. Use Julia version
  >= 1.6. ([#385][github-385])
- Quadrature data for L2 projection can now be given as a matrix of size "number of
  elements" x "number of quadrature points per element". ([#386][github-386])
- Projected values from L2 projection can now be exported directly to VTK.
  ([#390][github-390])
- Grid coloring can now act on a subset of cells. ([#402][github-402])
- Various functions related to cell values now use traits to make it easier to extend and
  reuse functionality in external code. ([#404][github-404])
### Fixed
- Exporting tensors to VTK now use correct names for the components. ([#406][github-406])


[github-352]: https://github.com/Ferrite-FEM/Ferrite.jl/issues/352
[github-363]: https://github.com/Ferrite-FEM/Ferrite.jl/pull/363
[github-385]: https://github.com/Ferrite-FEM/Ferrite.jl/pull/385
[github-386]: https://github.com/Ferrite-FEM/Ferrite.jl/pull/386
[github-390]: https://github.com/Ferrite-FEM/Ferrite.jl/pull/390
[github-392]: https://github.com/Ferrite-FEM/Ferrite.jl/pull/392
[github-401]: https://github.com/Ferrite-FEM/Ferrite.jl/pull/401
[github-402]: https://github.com/Ferrite-FEM/Ferrite.jl/pull/402
[github-404]: https://github.com/Ferrite-FEM/Ferrite.jl/pull/404
[github-406]: https://github.com/Ferrite-FEM/Ferrite.jl/pull/406
[github-407]: https://github.com/Ferrite-FEM/Ferrite.jl/pull/407
[github-411]: https://github.com/Ferrite-FEM/Ferrite.jl/pull/411
[github-418]: https://github.com/Ferrite-FEM/Ferrite.jl/pull/418
[github-425]: https://github.com/Ferrite-FEM/Ferrite.jl/pull/425
[github-428]: https://github.com/Ferrite-FEM/Ferrite.jl/pull/428

[Unreleased]: https://github.com/Ferrite-FEM/Ferrite.jl/compare/v0.3.4...HEAD
[0.3.4]: https://github.com/Ferrite-FEM/Ferrite.jl/compare/v0.3.3...v0.3.4
[0.3.3]: https://github.com/Ferrite-FEM/Ferrite.jl/compare/v0.3.2...v0.3.3
[0.3.2]: https://github.com/Ferrite-FEM/Ferrite.jl/compare/v0.3.1...v0.3.2
