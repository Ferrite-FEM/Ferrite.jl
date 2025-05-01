# Ferrite changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [v1.1.0] - 2025-05-01

### Added
 - New vector interpolations for H(div) and H(curl) spaces (`Nedelec`, `RaviartThomas`,
   `BrezziDouglasMarini`). ([#1045],[#1162])
 - New boundary condition type, `ProjectedDirichlet`, for H(div) and H(curl) interpolations. ([#1151])
   It can be added to a constraint handler the same way as for a regular `Dirichlet`.
 - Support for exporting discontinuous fields to VTK. ([#867])
   This happens automatically when a DofHandler with a discontinuous field is used
   to construct the `VTKGridFile`, but can be requested with the `write_discontinuous`
   keyword argument.

### Fixes
 - `addboundaryfacetset` has been fixed for mixed grids. ([#1176])
 -  `evaluate_at_grid_nodes` now respects the precision of the input dof vector. ([#1044])

### Removed
 - The deprecated third type parameter for interpolations has been removed. ([#1083])
   Old code which tries to use three parameters will now throw the somewhat cryptic error:
   ```
   julia> Lagrange{2, RefCube, 1}()
   ERROR: too many parameters for type
   ```

### Documentation updates
 - A comparison between different assembly strategies have been added to the docs. ([#1063])
 - The affine constraints docs have been extended. ([#1146])

### Other
 - Ferrite now uses [Runic.jl](https://github.com/fredrikekre/Runic.jl) for code formatting.
   ([#1096])
 - Ferrite now supports `ForwardDiff` v1. ([#1178])

## [v1.0.0] - 2024-09-30

Ferrite version 1.0 is a relatively large release, with a lot of new features, improvements,
deprecations and some removals. These changes are made to make the code base more consistent
and more suitable for future improvements. With this 1.0 release we are aiming for long time
stability, and there is no breaking release 2.0 on the horizon.

Unfortunately this means that code written for Ferrite version 0.3 will have to be updated.
All changes, with upgrade paths, are listed in the sections below. Since these sections
include a lot of other information as well (new features, internal changes, ...) there is
also a dedicated section about upgrading code from Ferrite 0.3 to 1.0 (see below) which
include the most common changes that are required. In addition, in all cases where possible,
you will be presented with a descriptive error message telling you what needs to change.

Deprecations for 1.0 will be removed during the 1.x release series. When upgrading old code
it is therefore recommended to use Ferrite 1.0 as a first stepping stone since this release
contain descriptive deprecation error messages that might not exist in e.g. Ferrite version
1.2.

### Upgrading code from Ferrite 0.3 to 1.0

This section give a short overview of the most common required changes. More details and
motivation are included in the following sections (with links to issues/pull request for
more discussion).

- **Interpolations**: remove the first parameter (the reference dimension) and use new
  reference shapes.

  Examples:
  ```diff
  # Linear Lagrange interpolation for a line
  - Lagrange{1, RefCube, 1}()
  + Lagrange{RefLine, 1}()

  # Linear Lagrange interpolation for a quadrilateral
  - Lagrange{2, RefCube, 1}()
  + Lagrange{RefQuadrilateral, 1}()

  # Quadratic Lagrange interpolation for a triangle
  - Lagrange{2, RefTetrahedron, 2}()
  + Lagrange{RefTriangle, 2}()
  ```

  For vector valued problems it is now required to explicitly vectorize the interpolation
  using the new `VectorizedInterpolation`. This is required when passing the interpolation
  to `CellValues` and when adding fields to the `DofHandler` using `add!`. In both of these
  places the interpolation was implicitly vectorized in Ferrite 0.3.

  Examples:
  ```julia
  # Linear Lagrange interpolation for a vector problem on the triangle (vector dimension
  # same as the reference dimension)
  ip_scalar = Lagrange{RefTriangle, 1}()
  ip_vector = ip_scalar ^ 2 # or VectorizedInterpolation{2}(ip_scalar)
  ```

- **Quadrature**: remove the first parameter (the reference dimension) and use new reference
  shapes.

  Examples:
  ```diff
  # Quadrature for a line
  - QuadratureRule{1, RefCube}(quadrature_order)
  + QuadratureRule{RefLine}(quadrature_order)

  # Quadrature for a quadrilateral
  - QuadratureRule{2, RefCube}(quadrature_order)
  + QuadratureRule{RefQuadrilateral}(quadrature_order)

  # Quadrature for a tetrahedron
  - QuadratureRule{3, RefTetrahedron}(quadrature_order)
  + QuadratureRule{RefTetrahedron}(quadrature_order)
  ```

- **Quadrature for face integration (FacetValues)**: replace `QuadratureRule{dim-1,
  reference_shape}(quadrature_order)` with
  `FacetQuadratureRule{reference_shape}(quadrature_order)`.

  Examples:
  ```diff
  # Quadrature for the facets of a quadrilateral
  - QuadratureRule{1, RefCube}(quadrature_order)
  + FacetQuadratureRule{RefQuadrilateral}(quadrature_order)

  # Quadrature for the facets of a triangle
  - QuadratureRule{1, RefTetrahedron}(quadrature_order)
  + FacetQuadratureRule{RefTriangle}(quadrature_order)

  # Quadrature for the facets of a hexhedron
  - QuadratureRule{2, RefCube}(quadrature_order)
  + FacetQuadratureRule{RefHexahedron}(quadrature_order)
  ```

- **CellValues**: replace usage of `CellScalarValues` *and* `CellVectorValues` with
  `CellValues`. For vector valued problems the interpolation passed to `CellValues` should
  be vectorized to a `VectorizedInterpolation` (see above).

  Examples:
  ```diff
  # CellValues for a scalar problem with triangle elements
  - qr = QuadratureRule{2, RefTetrahedron}(quadrature_order)
  - ip = Lagrange{2, RefTetrahedron, 1}()
  - cv = CellScalarValues(qr, ip)
  + qr = QuadratureRule{RefTriangle}(quadrature_order)
  + ip = Lagrange{RefTriangle, 1}()
  + cv = CellValues(qr, ip)

  # CellValues for a vector problem with hexahedronal elements
  - qr = QuadratureRule{3, RefCube}(quadrature_order)
  - ip = Lagrange{3, RefCube, 1}()
  - cv = CellVectorValues(qr, ip)
  + qr = QuadratureRule{RefHexahedron}(quadrature_order)
  + ip = Lagrange{RefHexahedron, 1}() ^ 3
  + cv = CellValues(qr, ip)
  ```

  If you use `CellScalarValues` or `CellVectorValues` in method signature you must replace
  them with `CellValues`. Note that the type parameters are different.

  Examples:
  ```diff
  - function do_something(cvs::CellScalarValues, cvv::CellVectorValues)
  + function do_something(cvs::CellValues, cvv::CellValues)
  ```

  The default geometric interpolation have changed from the function interpolation to always
  use linear Lagrange interpolation. If you use linear elements in the grid, and a higher
  order interpolation for the function you can now rely on the new default:
  ```diff
  qr = QuadratureRule(...)
  - ip_function = Lagrange{2, RefTetrahedron, 2}()
  - ip_geometry = Lagrange{2, RefTetrahedron, 1}()
  - cv = CellScalarValues(qr, ip_function, ip_geometry)
  + ip_function = Lagrange{RefTriangle, 2}()
  + cv = CellValues(qr, ip_function)
  ```
  and if you have quadratic (or higher order) elements in the grid you must now pass the
  corresponding interpolation to the constructor:
  ```diff
  qr = QuadratureRule(...)
  - ip_function = Lagrange{2, RefTetrahedron, 2}()
  - cv = CellScalarValues(qr, ip_function)
  + ip_function = Lagrange{RefTriangle, 2}()
  + ip_geometry = Lagrange{RefTriangle, 1}()
  + cv = CellValues(qr, ip_function, ip_geometry)
  ```

- **FacetValues**: replace usage of `FaceScalarValues` *and* `FaceVectorValues` with
  `FacetValues`. For vector valued problems the interpolation passed to `FacetValues` should
  be vectorized to a `VectorizedInterpolation` (see above). The input quadrature rule should
  be a `FacetQuadratureRule` instead of a `QuadratureRule`.

  Examples:
  ```diff
  # FacetValues for a scalar problem with triangle elements
  - qr = QuadratureRule{1, RefTetrahedron}(quadrature_order)
  - ip = Lagrange{2, RefTetrahedron, 1}()
  - cv = FaceScalarValues(qr, ip)
  + qr = FacetQuadratureRule{RefTriangle}(quadrature_order)
  + ip = Lagrange{RefTriangle, 1}()
  + cv = FacetValues(qr, ip)

  # FaceValues for a vector problem with hexahedronal elements
  - qr = QuadratureRule{2, RefCube}(quadrature_order)
  - ip = Lagrange{3, RefCube, 1}()
  - cv = FaceVectorValues(qr, ip)
  + qr = FacetQuadratureRule{RefHexahedron}(quadrature_order)
  + ip = Lagrange{RefHexahedron, 1}() ^ 3
  + cv = FacetValues(qr, ip)
  ```

- **DofHandler construction**: it is now required to pass the interpolation explicitly when
  adding new fields using `add!` (previously it was optional, defaulting to the default
  interpolation of the elements in the grid). For vector-valued fields the interpolation
  should be vectorized, instead of passing the number of components to `add!` as an integer.

  Examples:
  ```diff
  dh = DofHandler(grid) # grid with triangles

  # Vector field :u
  - add!(dh, :u, 2)
  + add!(dh, :u, Lagrange{RefTriangle, 1}()^2)

  # Scalar field :p
  - add!(dh, :u, 1)
  + add!(dh, :u, Lagrange{RefTriangle, 1}())
  ```

- **Boundary conditions**: The entity enclosing a cell was previously called `face`, but is now
  denoted a `facet`. When applying boundary conditions, rename  `getfaceset` to `getfacetset` and
  `addfaceset!` is now `addfacetset!`. These sets are now described by `FacetIndex` instead of `FaceIndex`.
  When looping over the `facets` of a cell, change `nfaces` to `nfacets`.

  Examples:
  ```diff
  # Dirichlet boundary conditions
  - addfaceset!(grid, "dbc", x -> x[1] ≈ 1.0)
  + addfacetset!(grid, "dbc", x -> x[1] ≈ 1.0)

  - dbc = Dirichlet(:u, getfaceset(grid, "dbc"), Returns(0.0))
  + dbc = Dirichlet(:u, getfacetset(grid, "dbc"), Returns(0.0))

  # Neumann boundary conditions
  - for facet in 1:nfaces(cell)
  -     if (cellid(cell), facet) ∈ getfaceset(grid, "Neumann Boundary")
  + for facet in 1:nfacets(cell)
  +     if (cellid(cell), facet) ∈ getfacetset(grid, "Neumann Boundary")
            # ...
  ```

- **VTK Export**: The VTK export has been changed [#692].
  ```diff
  - vtk_grid(name, dh) do vtk
  -     vtk_point_data(vtk, dh, a)
  -     vtk_point_data(vtk, nodal_data, "my node data")
  -     vtk_point_data(vtk, proj, projected_data, "my projected data")
  -     vtk_cell_data(vtk, proj, projected_data, "my projected data")
  + VTKGridFile(name, dh) do vtk
  +     write_solution(vtk, dh, a)
  +     write_node_data(vtk, nodal_data, "my node data")
  +     write_projection(vtk, proj, projected_data, "my projected data")
  +     write_cell_data(vtk, cell_data, "my projected data")
  end
  ```
  When using a `paraview_collection` collection for e.g. multiple timesteps
  the `VTKGridFile` object can be used instead of the previous type returned
  from `vtk_grid`.

- **Sparsity pattern and global matrix construction**: since there is now explicit support
  for working with the sparsity pattern before instantiating a matrix the function
  `create_sparsity_pattern` has been removed. To recover the old functionality that return a
  sparse matrix from the DofHandler directly use `allocate_matrix` instead.

  Examples:
  ```diff
  # Create sparse matrix from DofHandler
  - K = create_sparsity_pattern(dh)
  + K = allocate_matrix(dh)

  # Create condensed sparse matrix from DofHandler + ConstraintHandler
  - K = create_sparsity_pattern(dh, ch)
  + K = allocate_matrix(dh, ch)
  ```

### Added

- `InterfaceValues` for computing jumps and averages over interfaces. ([#743])

- `InterfaceIterator` and `InterfaceCache` for iterating over interfaces. ([#747])

- `FacetQuadratureRule` implementation for `RefPrism` and `RefPyramid`. ([#779])

- The `DofHandler` now support selectively adding fields on sub-domains (rather than the
  full domain). This new functionality is included with the new `SubDofHandler` struct,
  which, as the name suggest, is a `DofHandler` for a subdomain. ([#624], [#667], [#735])

- New reference shape structs `RefLine`, `RefTriangle`, `RefQuadrilateral`,
  `RefTetrahedron`, `RefHexahedron`, and `RefPrism` have been added. These encode the
  reference dimension, and will thus replace the old reference shapes for which it was
  necessary to always pair with an explicit dimension (i.e. `RefLine` replaces `(RefCube,
  1)`, `RefTriangle` replaces `(RefTetrahedron, 2)`, etc.). For writing "dimension
  independent code" it is possible to use `Ferrite.RefHypercube{dim}` and
  `Ferrite.RefSimplex{dim}`. ([#679])

- New methods for adding entitysets that are located on the boundary of the grid:
  `addboundaryfacetset!` and `addboundaryvertexset!`. These work
  similar to `addfacetset!` and `addvertexset!`, but filters out all
  instances not on the boundary (this can be used to avoid accidental inclusion of internal
  entities in sets used for boundary conditions, for example). ([#606])

- New interpolation `VectorizedInterpolation` which vectorizes scalar interpolations for
  vector-valued problems. A `VectorizedInterpolation` is created from a (scalar)
  interpolation `ip` using either `ip ^ dim` or `VectorizedInterpolation{dim}(ip)`. For
  convenience, the method `VectorizedInterpolation(ip)` vectorizes the interpolation to the
  reference dimension of the interpolation. ([#694], [#736])

- New (scalar) interpolation `Lagrange{RefQuadrilateral, 3}()`, i.e. third order Lagrange
  interpolation for 2D quadrilaterals. ([#701], [#731])

- `CellValues` now support embedded elements. Specifically you can now embed elements with
  reference dimension 1 into spatial dimension 2 or 3, and elements with reference dimension
  2 in to spatial dimension 3. ([#651])

- `CellValues` now support (vector) interpolations with dimension different from the spatial
  dimension. ([#651])

- `FacetQuadratureRule` have been added and should be used for `FacetValues`. A
  `FacetQuadratureRule` for integration of the facets of e.g. a triangle can be constructed by
  `FacetQuadratureRule{RefTriangle}(order)` (similar to how `QuadratureRule` is constructed).
  ([#716])

- New functions `Ferrite.reference_shape_value(::Interpolation, ξ::Vec, i::Int)` and
  `Ferrite.reference_shape_gradient(::Interpolation, ξ::Vec, i::Int)` for evaluating the
  value/gradient of the `i`th shape function of an interpolation in local reference
  coordinate `ξ`. These methods are public but not exported. (Note that these methods return
  the value/gradient wrt. the reference coordinate `ξ`, whereas the corresponding methods
  for `CellValues` etc return the value/gradient wrt the spatial coordinate `x`.) ([#721])

- `FacetIterator` and `FacetCache` have been added. These work similarly to `CellIterator` and
  `CellCache` but are used to iterate over (boundary) face sets instead. These simplify
  boundary integrals in general, and in particular Neumann boundary conditions are more
  convenient to implement now that you can loop directly over the face set instead of
  checking all faces of a cell inside the element routine. ([#495])

- The `ConstraintHandler` now support adding Dirichlet boundary conditions on discontinuous
  interpolations. ([#729])

- `collect_periodic_faces` now have a keyword argument `tol` that can be used to relax the
  default tolerance when necessary. ([#749])

- VTK export now work with `QuadraticHexahedron` elements. ([#714])

- The function `bounding_box(::AbstractGrid)` has been added. It computes the bounding box for
  a given grid (based on its node coordinates), and returns the minimum and maximum vertices
  of the bounding box. ([#880])

- Support for working with sparsity patterns has been added. This means that Ferrite exposes
  the intermediate "state" between the DofHandler and the instantiated matrix as the new
  struct `SparsityPattern`. This make it possible to insert custom equations or couplings in
  the pattern before instantiating the matrix. The function `create_sparsity_pattern` have
  been removed. The new function `allocate_matrix` is instead used to instantiate the
  matrix. Refer to the documentation for more details. ([#888])

  **To upgrade**: if you want to recover the old functionality and don't need to work with
  the pattern, replace any usage of `create_sparsity_pattern` with `allocate_matrix`.

- A new function, `geometric_interpolation`, is exported, which gives the geometric interpolation
  for each cell type. This is equivalent to the deprecated `Ferrite.default_interpolation` function.
  ([#953])

- CellValues and FacetValues can now store and map second order gradients (Hessians). The number
  of gradients computed in CellValues/FacetValues is specified using the keyword arguments
  `update_gradients::Bool` (default true) and `update_hessians::Bool` (default false) in the
  constructors, i.e. `CellValues(...; update_hessians=true)`. ([#953])

- `L2Projector` supports projecting on grids with mixed celltypes. ([#949])

### Changed

- It is now possible to create sparsity patterns with interface couplings, see the new
  function `add_interface_entries!` and the rework of sparsity pattern construction.
  ([#710])

- The `AbstractCell` interface has been reworked. This change should not affect user code,
  but may in some cases be relevant for code parsing external mesh files. In particular, the
  generic `Cell` struct have been removed in favor of concrete cell implementations (`Line`,
  `Triangle`, ...). ([#679], [#712])

  **To upgrade** replace any usage of `Cell{...}(...)` with calls to the concrete
  implementations.

- The default geometric mapping in `CellValues` and `FacetValues` have changed. The new
  default is to always use `Lagrange{refshape, 1}()`, i.e. linear Lagrange polynomials, for
  the geometric interpolation. Previously, the function interpolation was (re) used also for
  the geometry interpolation. ([#695])

  **To upgrade**, if you relied on the previous default, simply pass the function
  interpolation also as the third argument (the geometric interpolation).

- All interpolations are now categorized as either scalar or vector interpolations. All
  (previously) existing interpolations are scalar. (Scalar) interpolations must now be
  explicitly vectorized, using the new `VectorizedInterpolation`, when used for vector
  problems. (Previously implicit vectorization happened in the `CellValues` constructor, and
  when adding fields to the `DofHandler`). ([#694])

- It is now required to explicitly pass the interpolation to the `DofHandler` when adding a
  new field using `add!`. For vector fields the interpolation should be vectorized, instead
  of passing number of components as an integer. ([#694])

  **To upgrade** don't pass the dimension as an integer, and pass the interpolation
  explicitly. See more details in [Upgrading code from Ferrite 0.3 to
  1.0](#upgrading-code-from-ferrite-03-to-10).

- `Interpolation`s should now be constructed using the new reference shapes. Since the new
  reference shapes encode the reference dimension the first type parameter of interpolations
  have been removed. ([#711])
  **To upgrade** replace e.g. `Lagrange{1, RefCube, 1}()` with `Lagrange{RefLine, 1}()`, and
  `Lagrange{2, RefTetrahedron, 1}()` with `Lagrange{RefTriangle, 1}()`, etc.

- `QuadratureRule`s should now be constructed using the new reference shapes. Since the new
  reference shapes encode the reference dimension the first type parameter of
  `QuadratureRule` have been removed. ([#711], [#716])
  **To upgrade** replace e.g. `QuadratureRule{1, RefCube}(order)` with
  `QuadratureRule{RefLine}(order)`, and `QuadratureRule{2, RefTetrahedron}(1)` with
  `Lagrange{RefTriangle}(order)`, etc.

- `CellScalarValues` and `CellVectorValues` have been merged into `CellValues`,
  `FaceScalarValues` and `FaceVectorValues` have been merged into `FacetValues`, and
  `PointScalarValues` and `PointVectorValues` have been merged into `PointValues`. The
  differentiation between scalar and vector have thus been moved to the interpolation (see
  above). Note that previously `CellValues`, `FaceValues`, and `PointValues` where abstract
  types, but they are now concrete implementations with *different type parameters*, except
  `FaceValues` which is now `FacetValues` ([#708])
  **To upgrade**, for scalar problems, it is enough to replace `CellScalarValues` with
  `CellValues`, `FaceScalarValues` with `FacetValues` and `PointScalarValues` with
  `PointValues`, respectively. For vector problems, make sure to vectorize the interpolation
  (see above) and then replace `CellVectorValues` with `CellValues`, `FaceVectorValues` with
  `FacetValues`, and `PointVectorValues` with `PointValues`.

- The quadrature rule passed to `FacetValues` should now be of type `FacetQuadratureRule`
  rather than of type `QuadratureRule`. ([#716])
  **To upgrade** replace the quadrature rule passed to `FacetValues` with a
  `FacetQuadratureRule`.

- Checking if a face `(ele_id, local_face_id) ∈ faceset` has been previously implemented
  by type piracy. In order to be invariant to the underlying `Set` datatype as well as
  omitting type piracy, ([#835]) implemented `isequal` and `hash` for `BoundaryIndex` datatypes.

- **VTK export**: Ferrite no longer extends `WriteVTK.vtk_grid` and associated functions,
  instead the new type `VTKGridFile` should be used instead. New methods exists for writing to
  a `VTKGridFile`, e.g. `write_solution`, `write_cell_data`, `write_node_data`, and `write_projection`.
  See [#692].

- **Definitions**: Previously, `face` and `edge` referred to codimension 1 relative reference shape.
  In Ferrite v1, `volume`, `face`, `edge`, and `vertex` refer to 3, 2, 1, and 0 dimensional entities,
  and `facet` replaces the old definition of `face`. No direct replacement for `edges` exits.
  See [#914] and [#914].
  The main implications of this change are
  * `FaceIndex` -> `FacetIndex` (`FaceIndex` still exists, but has a different meaning)
  * `FaceValues` -> `FacetValues`
  * `nfaces` -> `nfacets` (`nfaces` is now an internal method with different meaning)
  * `addfaceset!` -> `addfacetset`
  * `getfaceset` -> `getfacetset`

  Furthermore, subtypes of `Interpolation` should now define `vertexdof_indices`, `edgedof_indices`,
  `facedof_indices`, `volumedof_indices` (and similar) according to these definitions.

- `Ferrite.getdim` has been changed into `Ferrite.getrefdim` for getting the dimension of the reference shape
  and `Ferrite.getspatialdim` to get the spatial dimension (of the grid). ([#943])

- `Ferrite.getfielddim(::AbstractDofHandler, args...)` has been renamed to `Ferrite.n_components`.
  ([#943])

- The constructor for `ExclusiveTopology` only accept an `AbstractGrid` as input,
  removing the alternative of providing a `Vector{<:AbstractCell}`, as knowing the
  spatial dimension is required for correct code paths.
  Furthermore, it uses a new internal data structure, `ArrayOfVectorViews`, to store the neighborhood
  information more efficiently The datatype for the neighborhood has thus changed to a view of a vector,
  instead of the now removed `EntityNeighborhood` container. This also applies to `vertex_star_stencils`.
  ([#974]).

- `project(::L2Projector, data, qr_rhs)` now expects data to be indexed by the cellid, as opposed to
  the index in the vector of cellids passed to the `L2Projector`. The data may be passed as an
  `AbstractDict{Int, <:AbstractVector}`, as an alternative to `AbstractArray{<:AbstractVector}`.
  ([#949])

### Deprecated

- The rarely (if ever) used methods of `function_value`, `function_gradient`,
  `function_divergence`, and `function_curl` taking *vectorized dof values* as in put have
  been deprecated. ([#698])

- The function `reshape_to_nodes` have been deprecated in favor of `evaluate_at_grid_nodes`.
  ([#703])

- `start_assemble([n::Int])` has been deprecated in favor of calling `Ferrite.COOAssembler()` directly ([#916], [#1058]).

- `start_assemble(f, K)` have been deprecated in favor of the "canonical" `start_assemble(K,
  f)`. ([#707])

- `assemble!(assembler, dofs, fe, Ke)` have been deprecated in favor of the "canonical"
  `assemble!(assembler, dofs, Ke, fe)`. ([#1059])

- `end_assemble` have been deprecated in favor of `finish_assemble`. ([#754])

- `get_point_values` have been deprecated in favor of `evaluate_at_points`. ([#754])

- `transform!` have been deprecated in favor of `transform_coordinates!`. ([#754])

- `Ferrite.default_interpolation` has been deprecated in favor of `geometric_interpolation`. ([#953])

### Removed

- `MixedDofHandler` + `FieldHandler` have been removed in favor of `DofHandler` +
  `SubDofHandler`. Note that the syntax has changed, and note that `SubDofHandler` is much
  more capable compared to `FieldHandler`. Previously it was often required to pass both the
  `MixedDofHandler` and the `FieldHandler` to e.g. the assembly routine, but now it is
  enough to pass the `SubDofHandler` since it can be used for e.g. DoF queries etc.
  ([#624], [#667], [#735])

- Some old methods to construct the `L2Projector` have been removed after being deprecated
  for several releases. ([#697])

- The option `project_to_nodes` have been removed from `project(::L2Projector, ...)`. The
  returned values are now always ordered according to the projectors internal `DofHandler`.
  ([#699])

- The function `compute_vertex_values` have been removed. ([#700])

- The names `getweights`, `getpoints`, `getcellsets`, `getnodesets`, `getfacesets`,
  `getedgesets`, and `getvertexsets` have been removed from the list of exported names. (For
  now you can still use them by prefixing `Ferrite.`, e.g. `Ferrite.getweights`.) ([#754])

- The `onboundary` function (and the associated `boundary_matrix` property of the `Grid`
  datastructure) have been removed ([#924]). Instead of first checking
  `onboundary` and then check whether a facet belong to a specific facetset, check the
  facetset directly. For example:
  ```diff
  - if onboundary(cell, local_face_id) && (cell_id, local_face_id) in getfacesets(grid, "traction_boundary")
  + if (cell_id, local_face_id) in getfacesets(grid, "traction_boundary")
       # integrate the "traction_boundary" boundary
    end
  ```

### Fixed

- Benchmarks now work with master branch. ([#751], [#855])

- Topology construction have been generalized to, in particular, fix construction for 1D and
  for wedge elements. ([#641], [#670], [#684])

### Other improvements

- Documentation:
  - The documentation is now structured according to the Diataxis framework. There is now
    also clear separation between tutorials (for teaching) and code gallery (for showing
    off). ([#737], [#756])
  - New section in the developer documentation that describes the (new) reference shapes and
    their numbering scheme. ([#688])

- Performance:
  - `Ferrite.transform!(grid, f)` (for transforming the node coordinates in the `grid`
    according to a function `f`) is now faster and allocates less. ([#675])
  - Slight performance improvement in construction of `PointEvalHandler` (faster reverse
    coordinate lookup). ([#719])
  - Various performance improvements to topology construction. ([#753], [#759])

- Internal improvements:
  - The dof distribution interface have been updated to support higher order elements
    (future work). ([#627], [#732], [#733])
  - The `AbstractGrid` and `AbstractDofHandler` interfaces are now used more consistently
    internally. This will help with the implementation of distributed grids and DofHandlers.
    ([#655])
  - VTK export now uses the (geometric) interpolation directly when evaluating the finite
    element field instead of trying to work backwards how DoFs map to nodes. ([#703])
  - Improved bounds checking in `assemble!`. ([#706])
  - Internal methods `Ferrite.value` and `Ferrite.derivative` for computing the
    value/gradient of *all* shape functions have been removed. ([#720])
  - `Ferrite.create_incidence_matrix` now work with any `AbstractGrid` (not just `Grid`).
    ([#726])

## [v0.3.14] - 2023-04-03
### Added
 - Support reordering dofs of a `MixedDofHandler` by the built-in orderings `FieldWise` and
   `ComponentWise`. This includes support for reordering dofs of fields on subdomains.
   ([#645])
 - Support specifying the coupling between fields in a `MixedDofHandler` when creating the
   sparsity pattern. ([#650])
 - Support Metis dof reordering with coupling information for `MixedDofHandler`. ([#650])
 - Pretty printing for `MixedDofHandler` and `L2Projector`. ([#465])
### Other improvements
 - The `MixedDofHandler` have gone through a performance review (see [#629]) and
   now performs the same as `DofHandler`. This was part of the push to merge the two DoF
   handlers. Since `MixedDofHandler` is strictly more flexible, and now equally performant,
   it will replace `DofHandler` in the next breaking release. ([#637], [#639], [#642],
   [#643], [#656], [#660])
### Internal changes
Changes listed here should not affect regular usage, but listed here in case you have been
poking into Ferrite internals:
 - `Ferrite.ndim(dh, fieldname)` has been removed, use `Ferrite.getfielddim(dh, fieldname)`
   instead. ([#658])
 - `Ferrite.nfields(dh)` has been removed, use `length(Ferrite.getfieldnames(dh))` instead.
   ([#444], [#653])
 - `getfielddims(::FieldHandler)` and `getfieldinterpolations(::FieldHandler)` have been
   removed ([#647], [#659])

## [v0.3.13] - 2023-03-23
### Added
 - Support for classical trilinear and triquadratic wedge elements. ([#581])
 - Symmetric quadrature rules up to order 10 for prismatic elements. ([#581])
 - Finer granulation of dof distribution, allowing to distribute different amounts of dofs
   per entity. ([#581])
### Fixed
 - Dof distribution for embedded elements. ([#581])
 - Improve numerical accuracy in shape function evaluation for the
   `Lagrange{2,Tetrahedron,(3|4|5)}` interpolations. ([#582],
   [#633])
### Other improvements
 - Documentation:
    - New "Developer documentation" section in the manual for documenting Ferrite.jl
      internals and developer tools. ([#611])
    - Fix a bug in constraint computation in Stoke's flow example. ([#614])
 - Performance:
    - Benchmarking infrastructure to help tracking performance changes. ([#388])
    - Performance improvements for various accessor functions for `MixedDofHandler`.
      ([#621])
### Internal changes
 - To clarify the dof management `vertices(ip)`, `edges(ip)` and `faces(ip)` has been
   deprecated in favor of `vertexdof_indices(ip)`, `edgedof_indices(ip)` and
   `facedof_indices(ip)`. ([#581])
 - Duplicate grid representation has been removed from the `MixedDofHandler`. ([#577])

## [v0.3.12] - 2023-02-28
### Added
 - Added a basic `show` method for assemblers. ([#598])
### Fixed
 - Fix an issue in constraint application of `Symmetric`-wrapped sparse matrices (i.e.
   obtained from `create_symmatric_sparsity_pattern`). In particular, `apply!(K::Symmetric,
   f, ch)` would incorrectly modify `f` if any of the constraints were inhomogeneous.
   ([#592])
 - Properly disable the Metis extension on Julia 1.9 instead of causing precompilation
   errors. ([#588])
 - Fix adding Dirichlet boundary conditions on nodes when using MixedDofHandler.
   ([#593], [#594])
 - Fix accidentally slow implementation of `show` for `Grid`s. ([#599])
 - Fixes to topology functionality. ([#453], [#518],
   [#455])
 - Fix grid coloring for cell sets with 0 or 1 cells. ([#600])
### Other improvements
 - Documentation improvements:
    - Simplications and clarifications to hyperelasticity example. ([#591])
    - Remove duplicate docstring entry for `vtk_point_data`. ([#602])
    - Update documentation about initial conditions. ([#601], [#604])

## [v0.3.11] - 2023-01-17
### Added
 - [Metis.jl](https://github.com/JuliaSparse/Metis.jl) extension for fill-reducing DoF
   permutation. This uses Julias new package extension mechanism (requires Julia 1.10) to
   support a new DoF renumbering order `DofOrder.Ext{Metis}()` that can be passed to
   `renumber!` to renumber DoFs using the Metis.jl library. ([#393], [#549])
 - [BlockArrays.jl](https://github.com/JuliaArrays/BlockArrays.jl) extension for creating a
   globally blocked system matrix. `create_sparsity_pattern(BlockMatrix, dh, ch; kwargs...)`
   return a matrix that is blocked by field (requires DoFs to be (re)numbered by field, i.e.
   `renumber!(dh, DofOrder.FieldWise())`). For custom blocking it is possible to pass an
   uninitialized `BlockMatrix` with the correct block sizes (see `BlockArrays.jl` docs).
   This functionality is useful for e.g. special solvers where individual blocks need to be
   extracted. Requires Julia version 1.9 or above. ([#567])
 - New function `apply_analytical!` for setting the values of the degrees of freedom for a
   specific field according to a spatial function `f(x)`. ([#532])
 - New cache struct `CellCache` to be used when iterating over the cells in a grid or DoF
   handler. `CellCache` caches nodes, coordinates, and DoFs, for the cell. The cache `cc`
   can be re-initialized for a new cell index `ci` by calling `reinit!(cc, ci)`. This can be
   used as an alternative to `CellIterator` when more control over which element to loop
   over is needed. See documentation for `CellCache` for more information. ([#546])
 - It is now possible to create the sparsity pattern without constrained entries (they will
   be zeroed out later anyway) by passing `keep_constrained=false` to
   `create_sparsity_pattern`. This naturally only works together with local condensation of
   constraints since there won't be space allocated in the global matrix for the full (i.e.
   "non-condensed") element matrix. Creating the matrix without constrained entries reduces
   the memory footprint, but unless a significant amount of DoFs are constrained (e.g. high
   mesh resolution at a boundary) the savings are negligible. ([#539])
### Changed
 - `ConstraintHandler`: `update!` is now called implicitly in `close!`. This was easy to
   miss, and somewhat of a strange requirement when solving problems without time stepping.
   ([#459])
 - The function for computing the inhomogeneity in a `Dirichlet` constraint can now be
   specified as either `f(x)` or `f(x, t)`, where `x` is the spatial coordinate and `t` the
   time. ([#459])
 - The elements of a `CellIterator` are now `CellCache` instead of the iterator itself,
   which was confusing in some cases. This change does not affect typical user code.
   ([#546])
### Deprecated
 - Adding fields to a DoF handler with `push!(dh, ...)` has been deprecated in favor of
   `add!(dh, ...)`. This is to make it consistent with how constraints are added to a
   constraint handler. ([#578])
### Fixed
 - Fix `shape_value` for the linear, discontinuous Lagrange interpolation. ([#553])
 - Fix `reference_coordinate` dispatch for discontinuous Lagrange interpolations. ([#559])
 - Fix `show(::Grid)` for custom cell types. ([#570])
 - Fix `apply_zero!(Δa, ch)` when using inhomogeneous affine constraints ([#575])
### Other improvements
 - Internal changes defining a new global matrix/vector "interface". These changes make it
   easy to enable more array types (e.g. `BlockMatrix` support added in this release) and
   solvers in the future. ([#562], [#571])
 - Performance improvements:
    - Reduced time and memory allocations for global sparse matrix creation (Julia >= 1.10).
      ([#563])
 - Documentation improvements:
    - Added an overview of the Examples section. ([#531])
    - Added an example showing topology optimization. ([#531])
    - Various typo fixes. ([#574])
    - Fix broken links. ([#583])

## [v0.3.10] - 2022-12-11
### Added
 - New functions `apply_local!` and `apply_assemble!` for applying constraints locally on
   the element level before assembling to the global system. ([#528])
 - New functionality to renumber DoFs by fields or by components. This is useful when you
   need the global matrix to be blocked. ([#378], [#545])
 - Functionality to renumber DoFs in DofHandler and ConstraintHandler simultaneously:
   `renumber!(dh::DofHandler, ch::ConstraintHandler, order)`. Previously renumbering had to
   be done *before* creating the ConstraintHandler since otherwise DoF numbers would be
   inconsistent. However, this was inconvenient in cases where the constraints impact the
   new DoF order permutation. ([#542])
 - The coupling between fields can now be specified when creating the global matrix with
   `create_sparsity_pattern` by passing a `Matrix{Bool}`. For example, in a problem with
   unknowns `(u, p)` and corresponding test functions `(v, q)`, if there is no coupling
   between `p` and `q` it is unnecessary to allocate entries in the global matrix
   corresponding to these DoFs. This can now be communicated to `create_sparsity_pattern` by
   passing the coupling matrix `[true true; true false]` in the keyword argument `coupling`.
   ([#544])
### Changed
 - Runtime and allocations for application of boundary conditions in `apply!` and
   `apply_zero!` have been improved. As a result, the `strategy` keyword argument is
   obsolete and thus ignored. ([#489])
 - The internal representation of `Dirichlet` boundary conditions and `AffineConstraint`s in
   the `ConstraintHandler` have been unified. As a result, conflicting constraints on DoFs
   are handled more consistently: the constraint added last to the `ConstraintHandler` now
   always override any previous constraints. Conflicting constraints could previously cause
   problems when a DoF where prescribed by both `Dirichlet` and `AffineConstraint`. ([#529])
 - Entries in local matrix/vector are now ignored in the assembly procedure. This allows,
   for example, using a dense local matrix `[a b; c d]` even if no entries exist in the
   global matrix for the `d` block, i.e. in `[A B; C D]` the `D` block is zero, and these
   global entries might not exist in the sparse matrix. (Such sparsity patterns can now be
   created by `create_sparsity_pattern`, see [#544].) ([#543])
### Fixed
 - Fix affine constraints with prescribed DoFs in the right-hand-side. In particular, DoFs
   that are prescribed by just an inhomogeneity are now handled correctly, and nested affine
   constraints now give an error instead of silently giving the wrong result.
   ([#530], [#535])
 - Fixed internal inconsistency in edge ordering for 2nd order RefTetrahedron and RefCube.
   ([#520], [#523])
### Other improvements
 - Performance improvements:
    - Reduced time and memory allocations in DoF distribution for `MixedDofHandler`.
      ([#533])
    - Reduced time and memory allocations reductions in `getcoordinates!`. ([#536])
    - Reduced time and memory allocations in affine constraint condensation.
      ([#537], [#541], [#550])
 - Documentation improvements:
    - Use `:static` scheduling for threaded `for`-loop ([#534])
    - Remove use of `@inbounds` ([#547])
 - Unification of `create_sparsity_pattern` methods to remove code duplication between
   `DofHandler` and `MixedDofHandler`. ([#538], [#540])

## [v0.3.9] - 2022-10-19
### Added
 - New higher order function interpolations for triangles (`Lagrange{2,RefTetrahedron,3}`,
   `Lagrange{2,RefTetrahedron,4}`, and `Lagrange{2,RefTetrahedron,5}`). ([#482], [#512])
 - New Gaussian quadrature formula for triangles up to order 15. ([#514])
 - Add debug mode for working with Ferrite internals. ([#524])
### Changed
 - The default components to constrain in `Dirichlet` and `PeriodicDirichlet` have changed
   from component 1 to all components of the field. For scalar problems this has no effect.
   ([#506], [#509])

## [v0.3.8] - 2022-10-05
### Added
 - Ferrite.jl now has a logo! ([#464])
 - New keyword argument `search_nneighbors::Int` in `PointEvalHandler` for specifying how
   many neighboring elements to consider in the kNN search. The default is still 3 (usually
   sufficient). ([#466])
 - The IJV-assembler now support assembling non-square matrices. ([#471])
 - Periodic boundary conditions have been reworked and generalized. It now supports
   arbitrary relations between the mirror and image boundaries (e.g. not only translations
   in x/y/z direction). ([#478], [#481], [#496], [#501])
### Fixed
 - Fix `PointEvalHandler` when the first point is missing. ([#466])
 - Fix the ordering of nodes on the face for `(Quadratic)Tetrahedron` cells. ([#475])
### Other improvements
 - Many improvements to the documentation. ([#467], [#473], [#487], [#494], [#500])
 - Improved error messages in `reinit!` when number of geometric base functions and number
   of element coordinates mismatch. ([#469])
 - Remove some unnecessary function parametrizations. ([#503])
 - Remove some unnecessary allocations in grid coloring. ([#505])
 - More efficient way of creating the sparsity pattern when using `AffineConstraints` and/or
   `PeriodicDirichlet`. ([#436])

## [v0.3.7] - 2022-07-05
### Fixed
- Fix tests for newer version of WriteVTK (no functional change). ([#462])
### Other improvements
 - Various improvements to the heat equation example and the hyperelasticity example in the
   documentation. ([#460], [#461])

## [v0.3.6] - 2022-06-30
### Fixed
- Fix a bug with `L2Projection` of mixed grid. ([#456])
### Other improvements
 - Expanded manual section of Dirichlet BCs. ([#458])

## [v0.3.5] - 2022-05-30
### Added
- Functionality for querying information about the grid topology (e.g. neighboring cells,
  boundaries, ...). ([#363])
### Fixed
- Fix application of boundary conditions when combining RHSData and affine constraints.
  ([#431])

## [v0.3.4] - 2022-02-25
### Added
- Affine (linear) constraints between degrees-of-freedom. ([#401])
- Periodic Dirichlet boundary conditions. ([#418])
- Evaluation of arbitrary quantities in FE space. ([#425])
### Changed
- Interpolation(s) and the quadrature rule are now stored as part of the `CellValues`
  structs (`cv.func_interp`, `cv.geo_interp`, and `cv.qr`). ([#428])

## [v0.3.3] - 2022-02-04
### Changed
- Verify user input in various functions to eliminate possible out-of-bounds accesses.
  ([#407], [#411])

## [v0.3.2] - 2022-01-18
### Added
- Support for new interpolation types: `DiscontinuousLagrange`, `BubbleEnrichedLagrange`,
  and `CrouzeixRaviart`. ([#352], [#392])
### Changed
- Julia version 1.0 is no longer supported for Ferrite versions >= 0.3.2.
  Use Julia version >= 1.6. ([#385])
- Quadrature data for L2 projection can now be given as a matrix of size "number of
  elements" x "number of quadrature points per element". ([#386])
- Projected values from L2 projection can now be exported directly to VTK. ([#390])
- Grid coloring can now act on a subset of cells. ([#402])
- Various functions related to cell values now use traits to make it easier to extend and
  reuse functionality in external code. ([#404])
### Fixed
- Exporting tensors to VTK now use correct names for the components. ([#406])


<!-- Links generated by Changelog.jl -->

[v0.3.2]: https://github.com/Ferrite-FEM/Ferrite.jl/releases/tag/v0.3.2
[v0.3.3]: https://github.com/Ferrite-FEM/Ferrite.jl/releases/tag/v0.3.3
[v0.3.4]: https://github.com/Ferrite-FEM/Ferrite.jl/releases/tag/v0.3.4
[v0.3.5]: https://github.com/Ferrite-FEM/Ferrite.jl/releases/tag/v0.3.5
[v0.3.6]: https://github.com/Ferrite-FEM/Ferrite.jl/releases/tag/v0.3.6
[v0.3.7]: https://github.com/Ferrite-FEM/Ferrite.jl/releases/tag/v0.3.7
[v0.3.8]: https://github.com/Ferrite-FEM/Ferrite.jl/releases/tag/v0.3.8
[v0.3.9]: https://github.com/Ferrite-FEM/Ferrite.jl/releases/tag/v0.3.9
[v0.3.10]: https://github.com/Ferrite-FEM/Ferrite.jl/releases/tag/v0.3.10
[v0.3.11]: https://github.com/Ferrite-FEM/Ferrite.jl/releases/tag/v0.3.11
[v0.3.12]: https://github.com/Ferrite-FEM/Ferrite.jl/releases/tag/v0.3.12
[v0.3.13]: https://github.com/Ferrite-FEM/Ferrite.jl/releases/tag/v0.3.13
[v0.3.14]: https://github.com/Ferrite-FEM/Ferrite.jl/releases/tag/v0.3.14
[v1.0.0]: https://github.com/Ferrite-FEM/Ferrite.jl/releases/tag/v1.0.0
[#352]: https://github.com/Ferrite-FEM/Ferrite.jl/issues/352
[#363]: https://github.com/Ferrite-FEM/Ferrite.jl/issues/363
[#378]: https://github.com/Ferrite-FEM/Ferrite.jl/issues/378
[#385]: https://github.com/Ferrite-FEM/Ferrite.jl/issues/385
[#386]: https://github.com/Ferrite-FEM/Ferrite.jl/issues/386
[#388]: https://github.com/Ferrite-FEM/Ferrite.jl/issues/388
[#390]: https://github.com/Ferrite-FEM/Ferrite.jl/issues/390
[#392]: https://github.com/Ferrite-FEM/Ferrite.jl/issues/392
[#393]: https://github.com/Ferrite-FEM/Ferrite.jl/issues/393
[#401]: https://github.com/Ferrite-FEM/Ferrite.jl/issues/401
[#402]: https://github.com/Ferrite-FEM/Ferrite.jl/issues/402
[#404]: https://github.com/Ferrite-FEM/Ferrite.jl/issues/404
[#406]: https://github.com/Ferrite-FEM/Ferrite.jl/issues/406
[#407]: https://github.com/Ferrite-FEM/Ferrite.jl/issues/407
[#411]: https://github.com/Ferrite-FEM/Ferrite.jl/issues/411
[#418]: https://github.com/Ferrite-FEM/Ferrite.jl/issues/418
[#425]: https://github.com/Ferrite-FEM/Ferrite.jl/issues/425
[#428]: https://github.com/Ferrite-FEM/Ferrite.jl/issues/428
[#431]: https://github.com/Ferrite-FEM/Ferrite.jl/issues/431
[#436]: https://github.com/Ferrite-FEM/Ferrite.jl/issues/436
[#444]: https://github.com/Ferrite-FEM/Ferrite.jl/issues/444
[#453]: https://github.com/Ferrite-FEM/Ferrite.jl/issues/453
[#455]: https://github.com/Ferrite-FEM/Ferrite.jl/issues/455
[#456]: https://github.com/Ferrite-FEM/Ferrite.jl/issues/456
[#458]: https://github.com/Ferrite-FEM/Ferrite.jl/issues/458
[#459]: https://github.com/Ferrite-FEM/Ferrite.jl/issues/459
[#460]: https://github.com/Ferrite-FEM/Ferrite.jl/issues/460
[#461]: https://github.com/Ferrite-FEM/Ferrite.jl/issues/461
[#462]: https://github.com/Ferrite-FEM/Ferrite.jl/issues/462
[#464]: https://github.com/Ferrite-FEM/Ferrite.jl/issues/464
[#465]: https://github.com/Ferrite-FEM/Ferrite.jl/issues/465
[#466]: https://github.com/Ferrite-FEM/Ferrite.jl/issues/466
[#467]: https://github.com/Ferrite-FEM/Ferrite.jl/issues/467
[#469]: https://github.com/Ferrite-FEM/Ferrite.jl/issues/469
[#471]: https://github.com/Ferrite-FEM/Ferrite.jl/issues/471
[#473]: https://github.com/Ferrite-FEM/Ferrite.jl/issues/473
[#475]: https://github.com/Ferrite-FEM/Ferrite.jl/issues/475
[#478]: https://github.com/Ferrite-FEM/Ferrite.jl/issues/478
[#481]: https://github.com/Ferrite-FEM/Ferrite.jl/issues/481
[#482]: https://github.com/Ferrite-FEM/Ferrite.jl/issues/482
[#487]: https://github.com/Ferrite-FEM/Ferrite.jl/issues/487
[#489]: https://github.com/Ferrite-FEM/Ferrite.jl/issues/489
[#494]: https://github.com/Ferrite-FEM/Ferrite.jl/issues/494
[#495]: https://github.com/Ferrite-FEM/Ferrite.jl/issues/495
[#496]: https://github.com/Ferrite-FEM/Ferrite.jl/issues/496
[#500]: https://github.com/Ferrite-FEM/Ferrite.jl/issues/500
[#501]: https://github.com/Ferrite-FEM/Ferrite.jl/issues/501
[#503]: https://github.com/Ferrite-FEM/Ferrite.jl/issues/503
[#505]: https://github.com/Ferrite-FEM/Ferrite.jl/issues/505
[#506]: https://github.com/Ferrite-FEM/Ferrite.jl/issues/506
[#509]: https://github.com/Ferrite-FEM/Ferrite.jl/issues/509
[#512]: https://github.com/Ferrite-FEM/Ferrite.jl/issues/512
[#514]: https://github.com/Ferrite-FEM/Ferrite.jl/issues/514
[#518]: https://github.com/Ferrite-FEM/Ferrite.jl/issues/518
[#520]: https://github.com/Ferrite-FEM/Ferrite.jl/issues/520
[#523]: https://github.com/Ferrite-FEM/Ferrite.jl/issues/523
[#524]: https://github.com/Ferrite-FEM/Ferrite.jl/issues/524
[#528]: https://github.com/Ferrite-FEM/Ferrite.jl/issues/528
[#529]: https://github.com/Ferrite-FEM/Ferrite.jl/issues/529
[#530]: https://github.com/Ferrite-FEM/Ferrite.jl/issues/530
[#531]: https://github.com/Ferrite-FEM/Ferrite.jl/issues/531
[#532]: https://github.com/Ferrite-FEM/Ferrite.jl/issues/532
[#533]: https://github.com/Ferrite-FEM/Ferrite.jl/issues/533
[#534]: https://github.com/Ferrite-FEM/Ferrite.jl/issues/534
[#535]: https://github.com/Ferrite-FEM/Ferrite.jl/issues/535
[#536]: https://github.com/Ferrite-FEM/Ferrite.jl/issues/536
[#537]: https://github.com/Ferrite-FEM/Ferrite.jl/issues/537
[#538]: https://github.com/Ferrite-FEM/Ferrite.jl/issues/538
[#539]: https://github.com/Ferrite-FEM/Ferrite.jl/issues/539
[#540]: https://github.com/Ferrite-FEM/Ferrite.jl/issues/540
[#541]: https://github.com/Ferrite-FEM/Ferrite.jl/issues/541
[#542]: https://github.com/Ferrite-FEM/Ferrite.jl/issues/542
[#543]: https://github.com/Ferrite-FEM/Ferrite.jl/issues/543
[#544]: https://github.com/Ferrite-FEM/Ferrite.jl/issues/544
[#545]: https://github.com/Ferrite-FEM/Ferrite.jl/issues/545
[#546]: https://github.com/Ferrite-FEM/Ferrite.jl/issues/546
[#547]: https://github.com/Ferrite-FEM/Ferrite.jl/issues/547
[#549]: https://github.com/Ferrite-FEM/Ferrite.jl/issues/549
[#550]: https://github.com/Ferrite-FEM/Ferrite.jl/issues/550
[#553]: https://github.com/Ferrite-FEM/Ferrite.jl/issues/553
[#559]: https://github.com/Ferrite-FEM/Ferrite.jl/issues/559
[#562]: https://github.com/Ferrite-FEM/Ferrite.jl/issues/562
[#563]: https://github.com/Ferrite-FEM/Ferrite.jl/issues/563
[#567]: https://github.com/Ferrite-FEM/Ferrite.jl/issues/567
[#570]: https://github.com/Ferrite-FEM/Ferrite.jl/issues/570
[#571]: https://github.com/Ferrite-FEM/Ferrite.jl/issues/571
[#574]: https://github.com/Ferrite-FEM/Ferrite.jl/issues/574
[#575]: https://github.com/Ferrite-FEM/Ferrite.jl/issues/575
[#577]: https://github.com/Ferrite-FEM/Ferrite.jl/issues/577
[#578]: https://github.com/Ferrite-FEM/Ferrite.jl/issues/578
[#581]: https://github.com/Ferrite-FEM/Ferrite.jl/issues/581
[#582]: https://github.com/Ferrite-FEM/Ferrite.jl/issues/582
[#583]: https://github.com/Ferrite-FEM/Ferrite.jl/issues/583
[#588]: https://github.com/Ferrite-FEM/Ferrite.jl/issues/588
[#591]: https://github.com/Ferrite-FEM/Ferrite.jl/issues/591
[#592]: https://github.com/Ferrite-FEM/Ferrite.jl/issues/592
[#593]: https://github.com/Ferrite-FEM/Ferrite.jl/issues/593
[#594]: https://github.com/Ferrite-FEM/Ferrite.jl/issues/594
[#598]: https://github.com/Ferrite-FEM/Ferrite.jl/issues/598
[#599]: https://github.com/Ferrite-FEM/Ferrite.jl/issues/599
[#600]: https://github.com/Ferrite-FEM/Ferrite.jl/issues/600
[#601]: https://github.com/Ferrite-FEM/Ferrite.jl/issues/601
[#602]: https://github.com/Ferrite-FEM/Ferrite.jl/issues/602
[#604]: https://github.com/Ferrite-FEM/Ferrite.jl/issues/604
[#606]: https://github.com/Ferrite-FEM/Ferrite.jl/issues/606
[#611]: https://github.com/Ferrite-FEM/Ferrite.jl/issues/611
[#614]: https://github.com/Ferrite-FEM/Ferrite.jl/issues/614
[#621]: https://github.com/Ferrite-FEM/Ferrite.jl/issues/621
[#624]: https://github.com/Ferrite-FEM/Ferrite.jl/issues/624
[#627]: https://github.com/Ferrite-FEM/Ferrite.jl/issues/627
[#629]: https://github.com/Ferrite-FEM/Ferrite.jl/issues/629
[#633]: https://github.com/Ferrite-FEM/Ferrite.jl/issues/633
[#637]: https://github.com/Ferrite-FEM/Ferrite.jl/issues/637
[#639]: https://github.com/Ferrite-FEM/Ferrite.jl/issues/639
[#641]: https://github.com/Ferrite-FEM/Ferrite.jl/issues/641
[#642]: https://github.com/Ferrite-FEM/Ferrite.jl/issues/642
[#643]: https://github.com/Ferrite-FEM/Ferrite.jl/issues/643
[#645]: https://github.com/Ferrite-FEM/Ferrite.jl/issues/645
[#647]: https://github.com/Ferrite-FEM/Ferrite.jl/issues/647
[#650]: https://github.com/Ferrite-FEM/Ferrite.jl/issues/650
[#651]: https://github.com/Ferrite-FEM/Ferrite.jl/issues/651
[#653]: https://github.com/Ferrite-FEM/Ferrite.jl/issues/653
[#655]: https://github.com/Ferrite-FEM/Ferrite.jl/issues/655
[#656]: https://github.com/Ferrite-FEM/Ferrite.jl/issues/656
[#658]: https://github.com/Ferrite-FEM/Ferrite.jl/issues/658
[#659]: https://github.com/Ferrite-FEM/Ferrite.jl/issues/659
[#660]: https://github.com/Ferrite-FEM/Ferrite.jl/issues/660
[#667]: https://github.com/Ferrite-FEM/Ferrite.jl/issues/667
[#670]: https://github.com/Ferrite-FEM/Ferrite.jl/issues/670
[#675]: https://github.com/Ferrite-FEM/Ferrite.jl/issues/675
[#679]: https://github.com/Ferrite-FEM/Ferrite.jl/issues/679
[#684]: https://github.com/Ferrite-FEM/Ferrite.jl/issues/684
[#688]: https://github.com/Ferrite-FEM/Ferrite.jl/issues/688
[#692]: https://github.com/Ferrite-FEM/Ferrite.jl/issues/692
[#694]: https://github.com/Ferrite-FEM/Ferrite.jl/issues/694
[#695]: https://github.com/Ferrite-FEM/Ferrite.jl/issues/695
[#697]: https://github.com/Ferrite-FEM/Ferrite.jl/issues/697
[#698]: https://github.com/Ferrite-FEM/Ferrite.jl/issues/698
[#699]: https://github.com/Ferrite-FEM/Ferrite.jl/issues/699
[#700]: https://github.com/Ferrite-FEM/Ferrite.jl/issues/700
[#701]: https://github.com/Ferrite-FEM/Ferrite.jl/issues/701
[#703]: https://github.com/Ferrite-FEM/Ferrite.jl/issues/703
[#706]: https://github.com/Ferrite-FEM/Ferrite.jl/issues/706
[#707]: https://github.com/Ferrite-FEM/Ferrite.jl/issues/707
[#708]: https://github.com/Ferrite-FEM/Ferrite.jl/issues/708
[#710]: https://github.com/Ferrite-FEM/Ferrite.jl/issues/710
[#711]: https://github.com/Ferrite-FEM/Ferrite.jl/issues/711
[#712]: https://github.com/Ferrite-FEM/Ferrite.jl/issues/712
[#714]: https://github.com/Ferrite-FEM/Ferrite.jl/issues/714
[#716]: https://github.com/Ferrite-FEM/Ferrite.jl/issues/716
[#719]: https://github.com/Ferrite-FEM/Ferrite.jl/issues/719
[#720]: https://github.com/Ferrite-FEM/Ferrite.jl/issues/720
[#721]: https://github.com/Ferrite-FEM/Ferrite.jl/issues/721
[#726]: https://github.com/Ferrite-FEM/Ferrite.jl/issues/726
[#729]: https://github.com/Ferrite-FEM/Ferrite.jl/issues/729
[#731]: https://github.com/Ferrite-FEM/Ferrite.jl/issues/731
[#732]: https://github.com/Ferrite-FEM/Ferrite.jl/issues/732
[#733]: https://github.com/Ferrite-FEM/Ferrite.jl/issues/733
[#735]: https://github.com/Ferrite-FEM/Ferrite.jl/issues/735
[#736]: https://github.com/Ferrite-FEM/Ferrite.jl/issues/736
[#737]: https://github.com/Ferrite-FEM/Ferrite.jl/issues/737
[#743]: https://github.com/Ferrite-FEM/Ferrite.jl/issues/743
[#747]: https://github.com/Ferrite-FEM/Ferrite.jl/issues/747
[#749]: https://github.com/Ferrite-FEM/Ferrite.jl/issues/749
[#751]: https://github.com/Ferrite-FEM/Ferrite.jl/issues/751
[#753]: https://github.com/Ferrite-FEM/Ferrite.jl/issues/753
[#754]: https://github.com/Ferrite-FEM/Ferrite.jl/issues/754
[#756]: https://github.com/Ferrite-FEM/Ferrite.jl/issues/756
[#759]: https://github.com/Ferrite-FEM/Ferrite.jl/issues/759
[#779]: https://github.com/Ferrite-FEM/Ferrite.jl/issues/779
[#835]: https://github.com/Ferrite-FEM/Ferrite.jl/issues/835
[#855]: https://github.com/Ferrite-FEM/Ferrite.jl/issues/855
[#880]: https://github.com/Ferrite-FEM/Ferrite.jl/issues/880
[#888]: https://github.com/Ferrite-FEM/Ferrite.jl/issues/888
[#914]: https://github.com/Ferrite-FEM/Ferrite.jl/issues/914
[#916]: https://github.com/Ferrite-FEM/Ferrite.jl/issues/916
[#924]: https://github.com/Ferrite-FEM/Ferrite.jl/issues/924
[#943]: https://github.com/Ferrite-FEM/Ferrite.jl/issues/943
[#949]: https://github.com/Ferrite-FEM/Ferrite.jl/issues/949
[#953]: https://github.com/Ferrite-FEM/Ferrite.jl/issues/953
[#974]: https://github.com/Ferrite-FEM/Ferrite.jl/issues/974
[#1058]: https://github.com/Ferrite-FEM/Ferrite.jl/issues/1058
[#1059]: https://github.com/Ferrite-FEM/Ferrite.jl/issues/1059
[#1083]: https://github.com/Ferrite-FEM/Ferrite.jl/issues/1083
[#1096]: https://github.com/Ferrite-FEM/Ferrite.jl/issues/1096
