# Ferrite.jl changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]
### Added
 - New higher order function interpolations for triangles (`Lagrange{2,RefTetrahedron,3}`,
   `Lagrange{2,RefTetrahedron,4}`, and `Lagrange{2,RefTetrahedron,5}`). ([#482][github-482],
   [#512][github-512])
 - New Gaussian quadrature formula for triangles up to order 15. ([#514][github-514])
### Changed
 - The default components to constrain in `Dirichlet` and `PeriodicDirichlet` have changed
   from component 1 to all components of the field. For scalar problems this has no effect.
   ([#506][github-506], [#509][github-509])

## [0.3.8] - 2022-10-05
### Added
 - Ferrite.jl now has a logo! ([#464][github-464])
 - New keyword argument `search_nneighbors::Int` in `PointEvalHandler` for specifying how
   many neighboring elements to consider in the kNN search. The default is still 3 (usually
   sufficient). ([#466][github-466])
 - The IJV-assembler now support assembling non-square matrices. ([#471][github-471])
 - Periodic boundary conditions have been reworked and generalized. It now supports
   arbitrary relations between the mirror and image boundaries (e.g. not only translations
   in x/y/z direction). ([#478][github-478], [#481][github-481], [#496][github-496],
   [#501][github-501])
### Fixed
 - Fix `PointEvalHandler` when the first point is missing. ([#466][github-466])
 - Fix the ordering of nodes on the face for `(Quadratic)Tetrahedron` cells.
   ([#475][github-475])
### Other improvements
 - Many improvements to the documentation. ([#467][github-467], [#473][github-473],
   [#487][github-487], [#494][github-494], [#500][github-500])
 - Improved error messages in `reinit!` when number of geometric base functions and number
   of element coordinates mismatch. ([#469][github-469])
 - Remove some unnecessary function parametrizations. ([#503][github-503])
 - Remove some unnecessary allocations in grid coloring. ([#505][github-505])
 - More efficient way of creating the sparsity pattern when using `AffineConstraints` and/or
   `PeriodicDirichlet`. ([#436][github-436])

## [0.3.7] - 2022-07-05
### Fixed
- Fix tests for newer version of WriteVTK (no functional change). ([#462][github-462])
### Other improvements
 - Various improvements to the heat equation example and the hyperelasticity example in the
   documentation. ([#460][github-460], [#461][github-461])

## [0.3.6] - 2022-06-30
### Fixed
- Fix a bug with `L2Projection` of mixed grid. ([#456][github-456])
### Other improvements
 - Expanded manual section of Dirichlet BCs. ([#458][github-458])

## [0.3.5] - 2022-05-30
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
[github-436]: https://github.com/Ferrite-FEM/Ferrite.jl/pull/436
[github-456]: https://github.com/Ferrite-FEM/Ferrite.jl/pull/456
[github-458]: https://github.com/Ferrite-FEM/Ferrite.jl/pull/458
[github-460]: https://github.com/Ferrite-FEM/Ferrite.jl/pull/460
[github-461]: https://github.com/Ferrite-FEM/Ferrite.jl/pull/461
[github-462]: https://github.com/Ferrite-FEM/Ferrite.jl/pull/462
[github-464]: https://github.com/Ferrite-FEM/Ferrite.jl/pull/464
[github-466]: https://github.com/Ferrite-FEM/Ferrite.jl/pull/466
[github-466]: https://github.com/Ferrite-FEM/Ferrite.jl/pull/466
[github-467]: https://github.com/Ferrite-FEM/Ferrite.jl/pull/467
[github-469]: https://github.com/Ferrite-FEM/Ferrite.jl/pull/469
[github-471]: https://github.com/Ferrite-FEM/Ferrite.jl/pull/471
[github-473]: https://github.com/Ferrite-FEM/Ferrite.jl/pull/473
[github-475]: https://github.com/Ferrite-FEM/Ferrite.jl/pull/475
[github-478]: https://github.com/Ferrite-FEM/Ferrite.jl/pull/478
[github-481]: https://github.com/Ferrite-FEM/Ferrite.jl/pull/481
[github-482]: https://github.com/Ferrite-FEM/Ferrite.jl/pull/482
[github-487]: https://github.com/Ferrite-FEM/Ferrite.jl/pull/487
[github-494]: https://github.com/Ferrite-FEM/Ferrite.jl/pull/494
[github-496]: https://github.com/Ferrite-FEM/Ferrite.jl/pull/496
[github-500]: https://github.com/Ferrite-FEM/Ferrite.jl/pull/500
[github-501]: https://github.com/Ferrite-FEM/Ferrite.jl/pull/501
[github-503]: https://github.com/Ferrite-FEM/Ferrite.jl/pull/503
[github-505]: https://github.com/Ferrite-FEM/Ferrite.jl/pull/505
[github-506]: https://github.com/Ferrite-FEM/Ferrite.jl/pull/506
[github-509]: https://github.com/Ferrite-FEM/Ferrite.jl/pull/509
[github-512]: https://github.com/Ferrite-FEM/Ferrite.jl/pull/512
[github-514]: https://github.com/Ferrite-FEM/Ferrite.jl/pull/514

[Unreleased]: https://github.com/Ferrite-FEM/Ferrite.jl/compare/v0.3.8...HEAD
[0.3.8]: https://github.com/Ferrite-FEM/Ferrite.jl/compare/v0.3.7...v0.3.8
[0.3.7]: https://github.com/Ferrite-FEM/Ferrite.jl/compare/v0.3.6...v0.3.7
[0.3.6]: https://github.com/Ferrite-FEM/Ferrite.jl/compare/v0.3.5...v0.3.6
[0.3.5]: https://github.com/Ferrite-FEM/Ferrite.jl/compare/v0.3.4...v0.3.5
[0.3.4]: https://github.com/Ferrite-FEM/Ferrite.jl/compare/v0.3.3...v0.3.4
[0.3.3]: https://github.com/Ferrite-FEM/Ferrite.jl/compare/v0.3.2...v0.3.3
[0.3.2]: https://github.com/Ferrite-FEM/Ferrite.jl/compare/v0.3.1...v0.3.2
