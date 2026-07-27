#----------------------------------------------------------------------#
# Assembly: element kernels, global assembly loops and sparse scatter
#----------------------------------------------------------------------#
using SparseArrays: SparseMatrixCSC
using SparseMatricesCSR: SparseMatrixCSR
using LinearAlgebra: Symmetric

SUITE["assembly"] = BenchmarkGroup()

# Element kernels (reinit! + Ke fill) over a batch of cells: the per-cell work of an
# assembly loop without iterator and scatter. One kernel per shape-function query:
# shape_value (mass), shape_gradient (Laplace) and shape_symmetric_gradient (elasticity).
SUITE["assembly"]["element kernels"] = BenchmarkGroup()
let g = SUITE["assembly"]["element kernels"]
    nbatch = 64
    sweep! = FerriteBenchmarkHelpers.element_sweep!

    quadgrid = generate_grid(Quadrilateral, (8, 8))
    quadbatch = FerriteBenchmarkHelpers.cell_coordinate_batch(quadgrid, nbatch)
    cv = CellValues(QuadratureRule{RefQuadrilateral}(3), Lagrange{RefQuadrilateral, 2}())
    Ke = zeros(getnbasefunctions(cv), getnbasefunctions(cv))
    g["mass Lagrange{2} (Quadrilateral, $nbatch cells)"] = @benchmarkable(
        $sweep!($Ke, $cv, $quadbatch, $(FerriteBenchmarkHelpers.mass_kernel!)), evals = 1
    )

    tetgrid = generate_grid(Tetrahedron, (4, 4, 4))
    tetbatch = FerriteBenchmarkHelpers.cell_coordinate_batch(tetgrid, nbatch)
    cv = CellValues(QuadratureRule{RefTetrahedron}(2), Lagrange{RefTetrahedron, 2}())
    Ke = zeros(getnbasefunctions(cv), getnbasefunctions(cv))
    g["Laplace Lagrange{2} (Tetrahedron, $nbatch cells)"] = @benchmarkable(
        $sweep!($Ke, $cv, $tetbatch, $(FerriteBenchmarkHelpers.laplace_kernel!)), evals = 1
    )

    hexgrid = generate_grid(Hexahedron, (4, 4, 4))
    hexbatch = FerriteBenchmarkHelpers.cell_coordinate_batch(hexgrid, nbatch)
    cv = CellValues(QuadratureRule{RefHexahedron}(2), Lagrange{RefHexahedron, 1}()^3)
    Ke = zeros(getnbasefunctions(cv), getnbasefunctions(cv))
    C = FerriteBenchmarkHelpers.elasticity_tensor(3)
    g["elasticity Lagrange{1}^3 (Hexahedron, $nbatch cells)"] = @benchmarkable(
        $sweep!($Ke, $cv, $hexbatch, $(FerriteBenchmarkHelpers.elasticity_kernel!), $C), evals = 1
    )

    # The divergence coupling block of a mixed u-p formulation, via MultiFieldCellValues.
    cmv = MultiFieldCellValues(
        QuadratureRule{RefQuadrilateral}(2),
        (u = Lagrange{RefQuadrilateral, 2}()^2, p = Lagrange{RefQuadrilateral, 1}()),
    )
    Ke = zeros(getnbasefunctions(cmv.u), getnbasefunctions(cmv.p))
    g["mixed divergence u-p (Quadrilateral, $nbatch cells)"] = @benchmarkable(
        $sweep!($Ke, $cmv, $quadbatch, $(FerriteBenchmarkHelpers.mixed_divergence_kernel!)), evals = 1
    )
end

# Full global assembly loops: CellIterator, reinit!, kernel and scatter, i.e. what a user's
# `assemble_global` looks like in the tutorials. `start_assemble` zeroes the matrix, so the
# same matrix can be reused across samples.
SUITE["assembly"]["global loop"] = BenchmarkGroup()
let g = SUITE["assembly"]["global loop"]
    assemble! = FerriteBenchmarkHelpers.assemble_global!

    quadgrid = generate_grid(Quadrilateral, (40, 40))
    dh = DofHandler(quadgrid)
    add!(dh, :u, Lagrange{RefQuadrilateral, 2}())
    close!(dh)
    cv = CellValues(QuadratureRule{RefQuadrilateral}(3), Lagrange{RefQuadrilateral, 2}())
    K = allocate_matrix(dh)
    f = zeros(ndofs(dh))
    g["Laplace Lagrange{2} (Quadrilateral 40×40)"] = @benchmarkable(
        $assemble!($K, $f, $dh, $cv, $(FerriteBenchmarkHelpers.laplace_kernel!)),
        evals = 1, seconds = 1.0,
    )

    hexgrid = generate_grid(Hexahedron, (8, 8, 8))
    dh = DofHandler(hexgrid)
    add!(dh, :u, Lagrange{RefHexahedron, 1}()^3)
    close!(dh)
    cv = CellValues(QuadratureRule{RefHexahedron}(2), Lagrange{RefHexahedron, 1}()^3)
    K = allocate_matrix(dh)
    f = zeros(ndofs(dh))
    C = FerriteBenchmarkHelpers.elasticity_tensor(3)
    g["elasticity Lagrange{1}^3 (Hexahedron 8×8×8)"] = @benchmarkable(
        $assemble!($K, $f, $dh, $cv, $(FerriteBenchmarkHelpers.elasticity_kernel!), $C),
        evals = 1, seconds = 1.0,
    )
end

# Scatter-only loops with a precomputed element matrix, isolating the sparse `assemble!`
# cost for each supported matrix type.
SUITE["assembly"]["scatter"] = BenchmarkGroup()
let g = SUITE["assembly"]["scatter"]
    scatter! = FerriteBenchmarkHelpers.assemble_scatter!

    grid = generate_grid(Quadrilateral, (40, 40))
    dh = DofHandler(grid)
    add!(dh, :u, Lagrange{RefQuadrilateral, 2}())
    close!(dh)
    dofs_batch = FerriteBenchmarkHelpers.celldofs_batch(dh, getncells(grid))
    n = ndofs_per_cell(dh)
    Ke = rand(n, n)
    Ke = Ke + Ke' # assembling into Symmetric requires a symmetric Ke

    K = allocate_matrix(SparseMatrixCSC{Float64, Int}, dh)
    g["SparseMatrixCSC (Quadrilateral 40×40)"] = @benchmarkable $scatter!($K, $dofs_batch, $Ke) evals = 1

    Ksym = allocate_matrix(Symmetric{Float64, SparseMatrixCSC{Float64, Int}}, dh)
    g["Symmetric CSC (Quadrilateral 40×40)"] = @benchmarkable $scatter!($Ksym, $dofs_batch, $Ke) evals = 1

    Kcsr = allocate_matrix(SparseMatrixCSR{1, Float64, Int}, dh)
    g["SparseMatrixCSR (Quadrilateral 40×40)"] = @benchmarkable $scatter!($Kcsr, $dofs_batch, $Ke) evals = 1
end

# Boundary load assembly: FacetIterator, FacetValues reinit! and vector scatter.
SUITE["assembly"]["facet loop"] = BenchmarkGroup()
let g = SUITE["assembly"]["facet loop"]
    grid = generate_grid(Hexahedron, (10, 10, 10))
    dh = DofHandler(grid)
    add!(dh, :u, Lagrange{RefHexahedron, 1}()^3)
    close!(dh)
    fv = FacetValues(FacetQuadratureRule{RefHexahedron}(2), Lagrange{RefHexahedron, 1}()^3)
    facetset = getfacetset(grid, "right")
    f = zeros(ndofs(dh))
    g["Neumann load Lagrange{1}^3 (Hexahedron 10×10×10, right)"] = @benchmarkable(
        FerriteBenchmarkHelpers.assemble_facet_load!($f, $dh, $fv, $facetset), evals = 1
    )
end

# DG interface assembly: InterfaceIterator, InterfaceValues reinit! and scatter of the
# cross-element interface matrix.
SUITE["assembly"]["interface loop"] = BenchmarkGroup()
let g = SUITE["assembly"]["interface loop"]
    grid = generate_grid(Quadrilateral, (16, 16))
    topology = ExclusiveTopology(grid)
    ip = DiscontinuousLagrange{RefQuadrilateral, 1}()
    dh = DofHandler(grid)
    add!(dh, :u, ip)
    close!(dh)
    iv = InterfaceValues(FacetQuadratureRule{RefQuadrilateral}(2), ip)
    K = allocate_matrix(dh; topology, interface_coupling = trues(1, 1))
    g["interior penalty DiscontinuousLagrange{1} (Quadrilateral 16×16)"] = @benchmarkable(
        FerriteBenchmarkHelpers.assemble_interfaces!($K, $dh, $iv, $topology), evals = 1
    )
end
