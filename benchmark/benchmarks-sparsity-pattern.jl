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

# Pattern construction, i.e. everything except the final matrix allocation.
SPARSITY_PATTERN_SUITE["pattern"] = BenchmarkGroup()
let SP = SPARSITY_PATTERN_SUITE["pattern"]
    SP["cells"] = @benchmarkable build_pattern()
    SP["cells+constraints"] = @benchmarkable build_pattern($SP_CH)
    # Full coupling, i.e. the same pattern as "cells" but constructed entry by entry
    SP["cells, coupling"] = @benchmarkable build_pattern(; coupling = $(trues(2, 2)))
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
        SP[name] = @benchmarkable(allocate_matrix($MatrixType, sp), setup = (sp = build_pattern()), evals = 1)
    end
end

# The full user facing path: DofHandler (+ ConstraintHandler) to matrix.
SPARSITY_PATTERN_SUITE["matrix-from-dofhandler"] = BenchmarkGroup()
let SP = SPARSITY_PATTERN_SUITE["matrix-from-dofhandler"]
    SP["SparseMatrixCSC"] = @benchmarkable allocate_matrix(SparseMatrixCSC{Float64, Int}, $SP_DH)
    SP["SparseMatrixCSC, constraints"] = @benchmarkable allocate_matrix(SparseMatrixCSC{Float64, Int}, $SP_DH, $SP_CH)
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
    SP["within-reservation"] = @benchmarkable build_pattern_by_entry(10_000, 10_000, 8, 0)
    SP["outgrowing-reservation"] = @benchmarkable build_pattern_by_entry(10_000, 10_000, 8, 24)
end
