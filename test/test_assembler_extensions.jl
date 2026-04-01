using Ferrite
import SparseMatricesCSR: SparseMatrixCSR, sparsecsr
using SparseArrays, LinearAlgebra

@testset "SparseMatricesCSR extension" begin

    @testset "apply!(::SparseMatrixCSR,...)" begin
        # Specifically this test that values below the diagonal of K2::Symmetric aren't touched
        # and that the missing values are instead taken from above the diagonal.
        grid = generate_grid(Line, (2,))
        dh = DofHandler(grid)
        add!(dh, :u, Lagrange{RefLine, 1}())
        close!(dh)
        ch = ConstraintHandler(dh)
        add!(ch, Dirichlet(:u, getfacetset(grid, "left"), x -> 1))
        close!(ch)
        K0 = sparse(rand(3, 3))
        K0 = K0' * K0
        K1 = SparseMatrixCSR(transpose(copy(K0)))
        K2 = Symmetric(SparseMatrixCSR(transpose(copy(K0))))
        @test K0 == K1
        @test K1 == K2
        sol = [1.0, 2.0, 3.0]
        f0 = K0 * sol
        f1 = K1 * sol
        f2 = K2 * sol
        apply!(K0, f0, ch)
        apply!(K1, f1, ch)
        apply!(K2, f2, ch)
        @test K0 == K1
        @test K1 == K2
        @test f0 == f1
        @test f1 == f2
        # Error for affine constraints
        ch = ConstraintHandler(dh)
        add!(ch, AffineConstraint(1, [3 => 1.0], 1.0))
        close!(ch)
        @test_throws ErrorException("condensation of ::SparseMatrixCSR{1, Float64, Int64} matrix not supported") apply!(K2, f2, ch)
    end

    @testset "assembly integration" begin
        # Setup simple problem
        grid = generate_grid(Line, (2,))
        dh = DofHandler(grid)
        add!(dh, :u, Lagrange{RefLine, 1}())
        close!(dh)

        # Check if the matrix is correctly allocated
        K = allocate_matrix(SparseMatrixCSR, dh)
        I = [1, 1, 2, 2, 2, 3, 3]
        J = [1, 2, 1, 2, 3, 2, 3]
        V = zeros(7)
        K_manual = sparsecsr(I, J, V)
        @test K == K_manual
        @test K.rowptr == K_manual.rowptr
        @test K.colval == K_manual.colval
        f = zeros(3)

        # Check that including the ch doesnot mess up the pattern
        ch = ConstraintHandler(dh)
        add!(ch, Dirichlet(:u, getfacetset(grid, "left"), (x, t) -> 1))
        close!(ch)
        K_ch = allocate_matrix(SparseMatrixCSR, dh, ch)
        @test K == K_ch
        @test K.rowptr == K_ch.rowptr
        @test K.colval == K_ch.colval

        # Check if assembly works
        assembler = start_assemble(K, f)
        ke = [-1.0 1.0; 2.0 -1.0]
        fe = [1.0, 2.0]
        assemble!(assembler, [1, 2], ke, fe)
        assemble!(assembler, [3, 2], ke, fe)
        I = [1, 1, 2, 2, 2, 3, 3]
        J = [1, 2, 1, 2, 3, 2, 3]
        V = [-1.0, 1.0, 2.0, -2.0, 2.0, 1.0, -1.0]
        @test K ≈ sparsecsr(I, J, V)
        @test f ≈ [1.0, 4.0, 1.0]

        # Check if constraint handler integration works
        apply!(K, f, ch)
        I = [1, 1, 2, 2, 2, 3, 3]
        J = [1, 2, 1, 2, 3, 2, 3]
        V = [4 / 3, 0.0, 0.0, -2.0, 2.0, 1.0, -1.0]
        @test K ≈ sparsecsr(I, J, V)
        @test f ≈ [4 / 3, 2.0, 1.0]

        # CSRAssembler: assemble with different row and col dofs
        I = [1, 1, 4, 4, 6, 6]
        J = [1, 3, 1, 3, 1, 3]
        V = zeros(length(I))
        K = sparsecsr(I, J, V)
        f = zeros(6)
        assembler = start_assemble(K, f)
        rdofs = [1, 4, 6]
        cdofs = [1, 3]
        Ke = rand(length(rdofs), length(cdofs))
        fe = rand(length(rdofs))
        assemble!(assembler, rdofs, cdofs, Ke, fe)
        assemble!(assembler, rdofs, cdofs, Ke, fe)
        @test_throws ArgumentError assemble!(assembler, rdofs, Ke, fe) # Not in sparsity pattern
        @test all(K[rdofs, cdofs] .== 2Ke)
        @test all(f[rdofs] .== 2fe)

        # CSRAssembler: Assemble rectangular part in quadratic matrix
        K = SparseMatrixCSR{1}(6, 6, K.rowptr, K.colval, K.nzval)
        assembler = start_assemble(K, f)
        rdofs = [1, 4, 6]
        cdofs = [1, 3]
        Ke = rand(length(rdofs), length(cdofs))
        fe = rand(length(rdofs))
        assemble!(assembler, rdofs, cdofs, Ke, fe)
        assemble!(assembler, rdofs, cdofs, Ke, fe)
        @test_throws ArgumentError assemble!(assembler, rdofs, Ke, fe) # Not in sparsity pattern
        @test all(K[rdofs, cdofs] .== 2Ke)
        @test all(f[rdofs] .== 2fe)

        # Check if coupling works
        grid = generate_grid(Quadrilateral, (2, 2))
        ip = Lagrange{RefQuadrilateral, 1}()
        dh = DofHandler(grid)
        add!(dh, :u, ip)
        add!(dh, :v, ip)
        close!(dh)

        Ke_zeros = zeros(ndofs_per_cell(dh, 1), ndofs_per_cell(dh, 1))
        Ke_rand = rand(ndofs_per_cell(dh, 1), ndofs_per_cell(dh, 1))
        dofs = celldofs(dh, 1)

        for c1 in [true, false], c2 in [true, false], c3 in [true, false], c4 in [true, false]
            coupling = [c1; c2;; c3; c4]
            K = allocate_matrix(SparseMatrixCSR, dh; coupling)
            a = start_assemble(K)
            assemble!(a, dofs, Ke_zeros)
            if all(coupling)
                assemble!(a, dofs, Ke_rand)
                @test Ke_rand ≈ K[dofs, dofs]
            else
                @test_throws ErrorException assemble!(a, dofs, Ke_rand)
            end
        end
    end

end
