# Benchmark for the performance changes made in this work.
#
# Two changes:
#  (1) src/assembler.jl `_assemble_inner!` (+ CSR ext): sequential `sortedrowdofs`/
#      `sortedcoldofs` reads instead of permutation gathers, and deferred `Ke` load.
#  (2) src/Dofs/sparsity_pattern.jl `_can_use_fastsp`: route `allocate_matrix(dh, ch)`
#      with only Dirichlet (no affine) constraints through the fast pattern builder.
#
# Run:  julia --project=. -O3 benchmark/bench_perf_changes.jl
#
# To get BEFORE numbers, `git stash` the src changes and run again.

using Ferrite, SparseArrays, BenchmarkTools, LinearAlgebra

function setup(CT, RS, ord, dims; vec = false)
    grid = generate_grid(CT, dims)
    ip = vec ? Lagrange{RS, ord}()^Ferrite.getrefdim(CT) : Lagrange{RS, ord}()
    qr = QuadratureRule{RS}(2 * ord)
    cv = CellValues(qr, ip)
    dh = DofHandler(grid); add!(dh, :u, ip); close!(dh)
    ch = ConstraintHandler(dh)
    ∂Ω = union((getfacetset(grid, s) for s in ("left", "right"))...)
    add!(ch, Dirichlet(:u, ∂Ω, (x, t) -> vec ? zeros(Ferrite.getrefdim(CT)) : 0.0)); close!(ch)
    return grid, dh, cv, ch
end

# scatter-only assembly (fixed element matrix) — isolates the global insert cost
function scatter!(K, f, dh, Ke, fe)
    asm = start_assemble(K, f)
    for cell in CellIterator(dh)
        assemble!(asm, celldofs(cell), Ke, fe)
    end
    return K
end

const CASES = (
    ("3D Hex Q2 scalar", Hexahedron, RefHexahedron, 2, (10, 10, 10), false),
    ("3D Tet P1 vector", Tetrahedron, RefTetrahedron, 1, (15, 15, 15), true),
    ("2D Quad Q1 scalar", Quadrilateral, RefQuadrilateral, 1, (100, 100), false),
)

println("="^64)
println("(1) Scatter assemble!  -- src/assembler.jl _assemble_inner!")
println("="^64)
for (nm, CT, RS, ord, dims, vec) in CASES
    grid, dh, cv, _ = setup(CT, RS, ord, dims; vec = vec)
    n = getnbasefunctions(cv)
    K = allocate_matrix(dh); f = zeros(ndofs(dh))
    Ke = rand(n, n); fe = rand(n)
    print("  [$nm] $(ndofs(dh)) dofs, $n/cell: ")
    @btime scatter!($K, $f, $dh, $Ke, $fe)
end

println("\n" * "="^64)
println("(2) allocate_matrix(dh, ch)  -- pure Dirichlet fast path")
println("="^64)
for (nm, CT, RS, ord, dims, vec) in CASES
    _, dh, _, ch = setup(CT, RS, ord, dims; vec = vec)
    print("  [$nm] $(ndofs(dh)) dofs: ")
    @btime allocate_matrix($dh, $ch)
end
println("\nDONE")
