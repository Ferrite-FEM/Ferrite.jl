# Ferrite.jl changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]
### Added
 - [Metis.jl](https://github.com/JuliaSparse/Metis.jl) extension for fill-reducing DoF
   permutation. This uses Julias new package extension mechanism (requires Julia 1.10) to
   support a new DoF renumbering order `DofOrder.Ext{Metis}()` that can be passed to
   `renumber!` to renumber DoFs using the Metis.jl library. ([#393][github-393],
   [#549][github-549])
 - New function `apply_analytical!` for setting the values of the degrees of freedom for a 
   specific field according to a spatial function `f(x)`. ([#532][github-532])
### Deprecated
 - Adding fields to a DoF handler with `push!(dh, ...)` has been deprecated in favor of
   `add!(dh, ...)`. This is to make it consistent with how constraints are added to a
   constraint handler. ([#578][github-578])

## [0.3.10] - 2022-12-11
### Added
 - New functions `apply_local!` and `apply_assemble!` for applying constraints locally on
   the element level before assembling to the global system. ([#528][github-528])
 - New functionality to renumber DoFs by fields or by components. This is useful when you
   need the global matrix to be blocked. ([#378][github-378], [#545][github-545])
 - Functionality to renumber DoFs in DofHandler and ConstraintHandler simultaneously:
   `renumber!(dh::DofHandler, ch::ConstraintHandler, order)`. Previously renumbering had to
   be done *before* creating the ConstraintHandler since otherwise DoF numbers would be
   inconsistent. However, this was inconvenient in cases where the constraints impact the
   new DoF order permutation. ([#542][github-542])
 - The coupling between fields can now be specified when creating the global matrix with
   `create_sparsity_pattern` by passing a `Matrix{Bool}`. For example, in a problem with
   unknowns `(u, p)` and corresponding test functions `(v, q)`, if there is no coupling
   between `p` and `q` it is unnecessary to allocate entries in the global matrix
   corresponding to these DoFs. This can now be communicated to `create_sparsity_pattern` by
   passing the coupling matrix `[true true; true false]` in the keyword argument `coupling`.
   ([#544][github-544])
### Changed
 - Runtime and allocations for application of boundary conditions in `apply!` and
   `apply_zero!` have been improved. As a result, the `strategy` keyword argument is
   obsolete and thus ignored. ([#489][github-489])
 - The internal representation of `Dirichlet` boundary conditions and `AffineConstraint`s in
   the `ConstraintHandler` have been unified. As a result, conflicting constraints on DoFs
   are handled more consistently: the constraint added last to the `ConstraintHandler` now
   always override any previous constraints. Conflicting constraints could previously cause
   problems when a DoF where prescribed by both `Dirichlet` and `AffineConstraint`.
   ([#529][github-529])
 - Entries in local matrix/vector are now ignored in the assembly procedure. This allows,
   for example, using a dense local matrix `[a b; c d]` even if no entries exist in the
   global matrix for the `d` block, i.e. in `[A B; C D]` the `D` block is zero, and these
   global entries might not exist in the sparse matrix. (Such sparsity patterns can now be
   created by `create_sparsity_pattern`, see [#544][github-544].) ([#543][github-543])
### Fixed
 - Fix affine constraints with prescribed DoFs in the right-hand-side. In particular, DoFs
   that are prescribed by just an inhomogeneity are now handled correctly, and nested affine
   constraints now give an error instead of silently giving the wrong result.
   ([#530][github-530], [#535][github-535])
 - Fixed internal inconsistency in edge ordering for 2nd order RefTetrahedron and RefCube.
   ([#520][github-520], [#523][github-523])
### Other improvements
 - Performance improvements:
    - Reduced time and memory allocations in DoF distribution for `MixedDofHandler`.
      ([#533][github-533])
    - Reduced time and memory allocations reductions in `getcoordinates!`.
      ([#536][github-536])
    - Reduced time and memory allocations in affine constraint condensation.
      ([#537][github-537], [#541][github-541], [#550][github-550])
 - Documentation improvements:
    - Use `:static` scheduling for threaded `for`-loop ([#534][github-534])
    - Remove use of `@inbounds` ([#547][github-547])
 - Unification of `create_sparsity_pattern` methods to remove code duplication between
   `DofHandler` and `MixedDofHandler`. ([#538][github-538], [#540][github-540])

## [0.3.9] - 2022-10-19
### Added
 - New higher order function interpolations for triangles (`Lagrange{2,RefTetrahedron,3}`,
   `Lagrange{2,RefTetrahedron,4}`, and `Lagrange{2,RefTetrahedron,5}`). ([#482][github-482],
   [#512][github-512])
 - New Gaussian quadrature formula for triangles up to order 15. ([#514][github-514])
 - Add debug mode for working with Ferrite internals. ([#524][github-524])
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
[github-378]: https://github.com/Ferrite-FEM/Ferrite.jl/issues/378
[github-385]: https://github.com/Ferrite-FEM/Ferrite.jl/pull/385
[github-386]: https://github.com/Ferrite-FEM/Ferrite.jl/pull/386
[github-390]: https://github.com/Ferrite-FEM/Ferrite.jl/pull/390
[github-392]: https://github.com/Ferrite-FEM/Ferrite.jl/pull/392
[github-393]: https://github.com/Ferrite-FEM/Ferrite.jl/issues/393
[github-401]: https://github.com/Ferrite-FEM/Ferrite.jl/pull/401
[github-402]: https://github.com/Ferrite-FEM/Ferrite.jl/pull/402
[github-404]: https://github.com/Ferrite-FEM/Ferrite.jl/pull/404
[github-406]: https://github.com/Ferrite-FEM/Ferrite.jl/pull/406
[github-407]: https://github.com/Ferrite-FEM/Ferrite.jl/pull/407
[github-411]: https://github.com/Ferrite-FEM/Ferrite.jl/pull/411
[github-418]: https://github.com/Ferrite-FEM/Ferrite.jl/pull/418
[github-425]: https://github.com/Ferrite-FEM/Ferrite.jl/pull/425
[github-428]: https://github.com/Ferrite-FEM/Ferrite.jl/pull/428
[github-431]: https://github.com/Ferrite-FEM/Ferrite.jl/pull/431
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
[github-489]: https://github.com/Ferrite-FEM/Ferrite.jl/pull/489
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
[github-520]: https://github.com/Ferrite-FEM/Ferrite.jl/issues/520
[github-523]: https://github.com/Ferrite-FEM/Ferrite.jl/pull/523
[github-524]: https://github.com/Ferrite-FEM/Ferrite.jl/pull/524
[github-528]: https://github.com/Ferrite-FEM/Ferrite.jl/pull/528
[github-529]: https://github.com/Ferrite-FEM/Ferrite.jl/pull/529
[github-530]: https://github.com/Ferrite-FEM/Ferrite.jl/issues/530
[github-532]: https://github.com/Ferrite-FEM/Ferrite.jl/pull/532
[github-533]: https://github.com/Ferrite-FEM/Ferrite.jl/pull/533
[github-534]: https://github.com/Ferrite-FEM/Ferrite.jl/pull/534
[github-535]: https://github.com/Ferrite-FEM/Ferrite.jl/pull/535
[github-536]: https://github.com/Ferrite-FEM/Ferrite.jl/pull/536
[github-537]: https://github.com/Ferrite-FEM/Ferrite.jl/pull/537
[github-538]: https://github.com/Ferrite-FEM/Ferrite.jl/pull/538
[github-540]: https://github.com/Ferrite-FEM/Ferrite.jl/pull/540
[github-541]: https://github.com/Ferrite-FEM/Ferrite.jl/pull/541
[github-542]: https://github.com/Ferrite-FEM/Ferrite.jl/pull/542
[github-543]: https://github.com/Ferrite-FEM/Ferrite.jl/pull/543
[github-544]: https://github.com/Ferrite-FEM/Ferrite.jl/pull/544
[github-545]: https://github.com/Ferrite-FEM/Ferrite.jl/pull/545
[github-547]: https://github.com/Ferrite-FEM/Ferrite.jl/pull/547
[github-549]: https://github.com/Ferrite-FEM/Ferrite.jl/pull/549
[github-550]: https://github.com/Ferrite-FEM/Ferrite.jl/pull/550
[github-578]: https://github.com/Ferrite-FEM/Ferrite.jl/pull/578

[Unreleased]: https://github.com/Ferrite-FEM/Ferrite.jl/compare/v0.3.10...HEAD
[0.3.10]: https://github.com/Ferrite-FEM/Ferrite.jl/compare/v0.3.9...v0.3.10
[0.3.9]: https://github.com/Ferrite-FEM/Ferrite.jl/compare/v0.3.8...v0.3.9
[0.3.8]: https://github.com/Ferrite-FEM/Ferrite.jl/compare/v0.3.7...v0.3.8
[0.3.7]: https://github.com/Ferrite-FEM/Ferrite.jl/compare/v0.3.6...v0.3.7
[0.3.6]: https://github.com/Ferrite-FEM/Ferrite.jl/compare/v0.3.5...v0.3.6
[0.3.5]: https://github.com/Ferrite-FEM/Ferrite.jl/compare/v0.3.4...v0.3.5
[0.3.4]: https://github.com/Ferrite-FEM/Ferrite.jl/compare/v0.3.3...v0.3.4
[0.3.3]: https://github.com/Ferrite-FEM/Ferrite.jl/compare/v0.3.2...v0.3.3
[0.3.2]: https://github.com/Ferrite-FEM/Ferrite.jl/compare/v0.3.1...v0.3.2
