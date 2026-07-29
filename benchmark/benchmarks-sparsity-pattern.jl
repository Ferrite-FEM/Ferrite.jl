#----------------------------------------------------------------------#
# Sparsity pattern construction and matrix allocation benchmarks
#----------------------------------------------------------------------#
using LinearAlgebra: Symmetric
using SparseArrays: SparseMatrixCSC
using SparseMatricesCSR: SparseMatrixCSR

SUITE["sparsity-pattern"] = BenchmarkGroup()
SPARSITY_PATTERN_SUITE = SUITE["sparsity-pattern"]

# A single representative DofHandler with two fields of different order and dimension, and one
# ConstraintHandler with a Dirichlet condition (prescribed dofs only) and a periodic condition
# (affine constraints, which add entries to the pattern).
const SP_GRID = generate_grid(Hexahedron, (8, 8, 8))
const SP_DH = let dh = DofHandler(SP_GRID)
    add!(dh, :u, Lagrange{RefHexahedron, 2}())
    add!(dh, :v, Lagrange{RefHexahedron, 1}()^3)
    close!(dh)
end
const SP_CH = let ch = ConstraintHandler(SP_DH)
    add!(ch, Dirichlet(:u, getfacetset(SP_GRID, "top"), Returns(0.0)))
    add!(ch, PeriodicDirichlet(:u, collect_periodic_facets(SP_GRID, "left", "right")))
    close!(ch)
end

build_pattern(args...; kwargs...) = add_sparsity_entries!(init_sparsity_pattern(SP_DH), SP_DH, args...; kwargs...)

# Force the per-entry construction path by seeding one diagonal entry before
# add_sparsity_entries! (from-scratch construction may take a fast path; the diagonal is
# always part of the pattern, so the result is identical).
function build_pattern_per_entry()
    sp = init_sparsity_pattern(SP_DH)
    Ferrite.add_entry!(sp, 1, 1)
    return add_sparsity_entries!(sp, SP_DH)
end

# For the interface (topology) benchmarks: a fully discontinuous two-field discretization on
# the same grid. Symmetric and asymmetric interface_coupling are benchmarked separately since
# they can take structurally different construction paths.
const SP_TOPOLOGY = ExclusiveTopology(SP_GRID)
const SP_DH_DG = let dh = DofHandler(SP_GRID)
    add!(dh, :u, DiscontinuousLagrange{RefHexahedron, 1}()^3)
    add!(dh, :p, DiscontinuousLagrange{RefHexahedron, 1}())
    close!(dh)
end
function build_pattern_dg(; kwargs...)
    return add_sparsity_entries!(init_sparsity_pattern(SP_DH_DG), SP_DH_DG; topology = SP_TOPOLOGY, kwargs...)
end

# Pattern construction, i.e. everything except the final matrix allocation.
SPARSITY_PATTERN_SUITE["pattern"] = BenchmarkGroup()
let SP = SPARSITY_PATTERN_SUITE["pattern"]
    SP["cells"] = @benchmarkable build_pattern() evals = 1 seconds = 1.0
    SP["cells+constraints"] = @benchmarkable build_pattern($SP_CH) evals = 1 seconds = 1.0
    # Full coupling: the same pattern as "cells" through the coupling code path
    SP["cells, coupling"] = @benchmarkable build_pattern(; coupling = $(trues(2, 2))) evals = 1 seconds = 1.0
    # Non-full coupling between the two fields
    SP["cells, coupling (non-full)"] = @benchmarkable build_pattern(; coupling = $([true true; false true])) evals = 1 seconds = 1.0
    # Constrained rows/columns eliminated during construction
    SP["cells, keep_constrained=false"] = @benchmarkable build_pattern($SP_CH; keep_constrained = false) evals = 1 seconds = 1.0
    # The generic per-entry construction (the path taken when entries already exist)
    SP["cells, per-entry"] = @benchmarkable build_pattern_per_entry() evals = 1 seconds = 1.0
    # The affine-constraint pass in isolation (depends on the already-built pattern)
    SP["constraints-onto-cells"] = @benchmarkable(
        add_constraint_entries!(sp, $SP_CH), setup = (sp = build_pattern()), evals = 1, seconds = 1.0
    )
    # Interface entries (e.g. DG): symmetric interface_coupling
    SP["cells+interfaces (DG)"] = @benchmarkable build_pattern_dg(; interface_coupling = $(trues(2, 2))) evals = 1 seconds = 1.0
    # Asymmetric interface_coupling (couples in one direction only)
    SP["cells+interfaces (DG), coupling"] = @benchmarkable build_pattern_dg(; interface_coupling = $([true true; false true])) evals = 1 seconds = 1.0
end

# Matrix allocation from an already constructed pattern. `setup` builds a fresh pattern for
# every sample and `evals = 1` makes sure it is not reused between evaluations, since
# allocating a matrix may mutate the pattern.
SPARSITY_PATTERN_SUITE["matrix-from-pattern"] = BenchmarkGroup()
let SP = SPARSITY_PATTERN_SUITE["matrix-from-pattern"]
    for (name, MatrixType) in (
            "SparseMatrixCSC" => SparseMatrixCSC{Float64, Int},
            "SparseMatrixCSR" => SparseMatrixCSR{1, Float64, Int},
            "Symmetric" => Symmetric{Float64, SparseMatrixCSC{Float64, Int}},
        )
        SP[name] = @benchmarkable(allocate_matrix($MatrixType, sp), setup = (sp = build_pattern()), evals = 1, seconds = 1.0)
    end
end

# The full user facing path: DofHandler (+ ConstraintHandler) to matrix. Every matrix type in
# "matrix-from-pattern" above is repeated here, since the split between building the pattern and
# allocating the matrix is an internal one: work can move across that boundary without the end
# to end cost changing, and only these numbers say what a user actually pays.
SPARSITY_PATTERN_SUITE["matrix-from-dofhandler"] = BenchmarkGroup()
let SP = SPARSITY_PATTERN_SUITE["matrix-from-dofhandler"]
    for (name, MatrixType) in (
            "SparseMatrixCSC" => SparseMatrixCSC{Float64, Int},
            "SparseMatrixCSR" => SparseMatrixCSR{1, Float64, Int},
            "Symmetric" => Symmetric{Float64, SparseMatrixCSC{Float64, Int}},
        )
        SP[name] = @benchmarkable allocate_matrix($MatrixType, $SP_DH) evals = 1 seconds = 1.0
    end
    SP["SparseMatrixCSC, constraints"] = @benchmarkable allocate_matrix(SparseMatrixCSC{Float64, Int}, $SP_DH, $SP_CH) evals = 1 seconds = 1.0
end

# Building a pattern manually. Entries are added out of order to exercise insertion in the
# middle of a row, and with `dense_cols > 0` every row grows past its initial `nnz_per_row`.
function build_pattern_by_entry(rows, cols, nnz_per_row, dense_cols)
    sp = SparsityPattern(rows, cols; nnz_per_row = nnz_per_row)
    stride = cld(cols, nnz_per_row)
    for row in 1:rows
        for k in 0:(nnz_per_row - 1)
            Ferrite.add_entry!(sp, row, mod1(row + k * stride, cols))
        end
        for col in (cols - dense_cols + 1):cols
            Ferrite.add_entry!(sp, row, col)
        end
    end
    return sp
end

SPARSITY_PATTERN_SUITE["add_entry!"] = BenchmarkGroup()
let SP = SPARSITY_PATTERN_SUITE["add_entry!"]
    SP["within-reservation"] = @benchmarkable build_pattern_by_entry(10_000, 10_000, 8, 0) evals = 1
    SP["outgrowing-reservation"] = @benchmarkable build_pattern_by_entry(10_000, 10_000, 8, 24) evals = 1
end
