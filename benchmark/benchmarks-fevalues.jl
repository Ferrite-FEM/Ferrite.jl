#----------------------------------------------------------------------#
# FEValues: construction, reinit! and function evaluation
#----------------------------------------------------------------------#
# A single `reinit!` (or quadrature-point loop) is well below the noise floor, so all
# per-cell operations are swept over a batch of cells, see helper.jl. The batch is large
# because the cheapest cases (linear Lagrange in 2D) run at ~15 ns per cell.
SUITE["fevalues"] = BenchmarkGroup()

const FEVALUES_BATCH = 512

# Construction precomputes the reference values. A batch of 8 lifts the scalar case (~1.6 us
# per construction) safely over the floor.
SUITE["fevalues"]["construction"] = BenchmarkGroup()
let g = SUITE["fevalues"]["construction"]
    csweep = FerriteBenchmarkHelpers.construction_sweep
    qr2 = QuadratureRule{RefHexahedron}(2)
    ip_geo = Lagrange{RefHexahedron, 1}()
    g["CellValues Lagrange{2} (Hexahedron, 8 constructions)"] = @benchmarkable $csweep($qr2, $(Lagrange{RefHexahedron, 2}()), $ip_geo, 8) evals = 1
    g["CellValues Lagrange{2}^3 (Hexahedron, 8 constructions)"] = @benchmarkable $csweep($qr2, $(Lagrange{RefHexahedron, 2}()^3), $ip_geo, 8) evals = 1
end

# reinit! sweeps: one case per structurally different mapping/dispatch path.
#  - linear scalar on quadrilaterals: the cheapest, most common 2D path
#  - quadratic scalar on tetrahedra: 3D simplex
#  - linear vector field on hexahedra: 3D tensor product with a vectorized interpolation
#  - Raviart-Thomas on triangles: non-identity (Piola) mapping
#  - MultiFieldCellValues: shared geometry mapping for several fields
SUITE["fevalues"]["reinit!"] = BenchmarkGroup()
let g = SUITE["fevalues"]["reinit!"]
    sweep = FerriteBenchmarkHelpers.reinit_sweep!
    batchname = "$(FEVALUES_BATCH) cells"

    quadgrid = generate_grid(Quadrilateral, (24, 24))
    quadbatch = FerriteBenchmarkHelpers.cell_coordinate_batch(quadgrid, FEVALUES_BATCH)
    tetgrid = generate_grid(Tetrahedron, (5, 5, 5))
    tetbatch = FerriteBenchmarkHelpers.cell_coordinate_batch(tetgrid, FEVALUES_BATCH)
    hexgrid = generate_grid(Hexahedron, (8, 8, 8))
    hexbatch = FerriteBenchmarkHelpers.cell_coordinate_batch(hexgrid, FEVALUES_BATCH)
    trigrid = generate_grid(Triangle, (16, 16))
    tribatch = FerriteBenchmarkHelpers.cell_coordinate_batch(trigrid, FEVALUES_BATCH)

    cv = CellValues(QuadratureRule{RefQuadrilateral}(2), Lagrange{RefQuadrilateral, 1}())
    g["CellValues Lagrange{1} (Quadrilateral, $batchname)"] = @benchmarkable $sweep($cv, $quadbatch) evals = 1

    cv = CellValues(QuadratureRule{RefTetrahedron}(2), Lagrange{RefTetrahedron, 2}())
    g["CellValues Lagrange{2} (Tetrahedron, $batchname)"] = @benchmarkable $sweep($cv, $tetbatch) evals = 1

    cv = CellValues(QuadratureRule{RefHexahedron}(2), Lagrange{RefHexahedron, 1}()^3)
    g["CellValues Lagrange{1}^3 (Hexahedron, $batchname)"] = @benchmarkable $sweep($cv, $hexbatch) evals = 1

    cv = CellValues(QuadratureRule{RefTriangle}(2), RaviartThomas{RefTriangle, 1}())
    # The Piola mappings need the cell for the edge orientations, hence the extra argument.
    tricells = [getcells(trigrid, mod1(i, getncells(trigrid))) for i in 1:FEVALUES_BATCH]
    g["CellValues RaviartThomas{1} (Triangle, $batchname)"] = @benchmarkable $sweep($cv, $tricells, $tribatch) evals = 1

    cv = CellValues(QuadratureRule{RefTetrahedron}(2), Nedelec{RefTetrahedron, 1}())
    tetcells = [getcells(tetgrid, mod1(i, getncells(tetgrid))) for i in 1:FEVALUES_BATCH]
    g["CellValues Nedelec{1} (Tetrahedron, $batchname)"] = @benchmarkable $sweep($cv, $tetcells, $tetbatch) evals = 1

    # Second derivatives roughly double the mapping work; a separate code path in reinit!.
    cv = CellValues(QuadratureRule{RefQuadrilateral}(2), Lagrange{RefQuadrilateral, 2}(); update_hessians = true)
    g["CellValues Lagrange{2} with hessians (Quadrilateral, $batchname)"] = @benchmarkable $sweep($cv, $quadbatch) evals = 1

    cmv = MultiFieldCellValues(
        QuadratureRule{RefQuadrilateral}(2),
        (u = Lagrange{RefQuadrilateral, 2}()^2, p = Lagrange{RefQuadrilateral, 1}()),
    )
    g["MultiFieldCellValues u^2+p (Quadrilateral, $batchname)"] = @benchmarkable $sweep($cmv, $quadbatch) evals = 1

    fv = FacetValues(FacetQuadratureRule{RefQuadrilateral}(2), Lagrange{RefQuadrilateral, 2}())
    g["FacetValues Lagrange{2} (Quadrilateral, $batchname)"] = @benchmarkable $sweep($fv, $quadbatch, 4) evals = 1

    fv = FacetValues(FacetQuadratureRule{RefHexahedron}(2), Lagrange{RefHexahedron, 1}()^3)
    g["FacetValues Lagrange{1}^3 (Hexahedron, $batchname)"] = @benchmarkable $sweep($fv, $hexbatch, 6) evals = 1
end

# Solution interpolation at quadrature points (function_value + function_gradient), the
# inner loop of residual evaluations in nonlinear problems.
SUITE["fevalues"]["function values"] = BenchmarkGroup()
let g = SUITE["fevalues"]["function values"]
    nreps = 512
    sweep = FerriteBenchmarkHelpers.function_values_sweep

    quadgrid = generate_grid(Quadrilateral, (2, 2))
    cv = CellValues(QuadratureRule{RefQuadrilateral}(2), Lagrange{RefQuadrilateral, 2}())
    reinit!(cv, getcoordinates(quadgrid, 1))
    ue_batch = [rand(getnbasefunctions(cv)) for _ in 1:nreps]
    g["scalar Lagrange{2} (Quadrilateral, $nreps cells)"] = @benchmarkable $sweep($cv, $ue_batch) evals = 1

    hexgrid = generate_grid(Hexahedron, (2, 2, 2))
    cv = CellValues(QuadratureRule{RefHexahedron}(2), Lagrange{RefHexahedron, 1}()^3)
    reinit!(cv, getcoordinates(hexgrid, 1))
    ue_batch = [rand(getnbasefunctions(cv)) for _ in 1:nreps]
    g["vector Lagrange{1}^3 (Hexahedron, $nreps cells)"] = @benchmarkable $sweep($cv, $ue_batch) evals = 1

    # Mapping quadrature points to spatial coordinates, e.g. for evaluating source terms.
    hexgrid_sc = generate_grid(Hexahedron, (8, 8, 8))
    coord_batch = FerriteBenchmarkHelpers.cell_coordinate_batch(hexgrid_sc, nreps)
    cv = CellValues(QuadratureRule{RefHexahedron}(2), Lagrange{RefHexahedron, 1}())
    g["spatial_coordinate (Hexahedron, $nreps cells)"] = @benchmarkable FerriteBenchmarkHelpers.spatial_coordinate_sweep($cv, $coord_batch) evals = 1
end
