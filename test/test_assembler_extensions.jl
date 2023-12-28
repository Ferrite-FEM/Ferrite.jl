import SparseMatricesCSR: SparseMatrixCSR
using SparseArrays, LinearAlgebra

@testset "apply!(::SparseMatrixCSR,...)" begin
    # Specifically this test that values below the diagonal of K2::Symmetric aren't touched
    # and that the missing values are instead taken from above the diagonal.
    grid = generate_grid(Line, (2,))
    dh = DofHandler(grid)
    add!(dh, :u, Lagrange{RefLine,1}())
    close!(dh)
    ch = ConstraintHandler(dh)
    add!(ch, Dirichlet(:u, getfaceset(grid, "left"), x -> 1))
    close!(ch)
    K0 = sparse(rand(3, 3))
    K0 = K0'*K0
    K1 = SparseMatrixCSR(transpose(copy(K0)))
    K2 = Symmetric(SparseMatrixCSR(transpose(copy(K0))))
    @test K0 == K1
    @test K1 == K2
    sol = [1.0, 2.0, 3.0]
    f0 = K0*sol
    f1 = K1*sol
    f2 = K2*sol
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
