import SparseMatricesCSR: SparseMatrixCSR, sparsecsr
using SparseArrays, LinearAlgebra

@testset "SparseMatricesCSR extension" begin

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

@testset "assembly integration" begin
    grid = generate_grid(Line, (2,))
    dh = DofHandler(grid)
    add!(dh, :u, Lagrange{RefLine, 1}())
    close!(dh)

    K = create_sparsity_pattern(SparseMatrixCSR, dh)
    I = [1,1,2,2,2,3,3]
    J = [1,2,1,2,3,2,3]
    V = zeros(7)
    @test K == sparsecsr(I,J,V)
    f = zeros(3)

    ch = ConstraintHandler(dh)
    add!(ch, Dirichlet(:u, getfaceset(grid, "left"), (x, t) -> 1))
    close!(ch)
    @test K == create_sparsity_pattern(SparseMatrixCSR, dh, ch)

    assembler = start_assemble(K, f)
    ke = [-1.0 1.0; 2.0 -1.0]
    fe = [1.0,2.0]
    assemble!(assembler, [1,2], ke,fe)
    assemble!(assembler, [3,2], ke,fe)

    I = [1,1,2,2,2,3,3]
    J = [1,2,1,2,3,2,3]
    V = [-1.0,1.0,2.0,-2.0,2.0,1.0,-1.0]
    @test K ≈ sparsecsr(I,J,V)
    @test f ≈ [1.0,4.0,1.0]

    apply!(K,f,ch)

    I = [1,1,2,2,2,3,3]
    J = [1,2,1,2,3,2,3]
    V = [4/3,0.0,0.0,-2.0,2.0,1.0,-1.0]
    @test K ≈ sparsecsr(I,J,V)
    @test f ≈ [4/3,2.0,1.0]


    grid = generate_grid(Quadrilateral, (2,2))
    ip = Lagrange{RefQuadrilateral,1}()
    dh = DofHandler(grid)
    add!(dh, :u, ip)
    add!(dh, :v, ip)
    close!(dh)

    Ke_zeros = zeros(ndofs_per_cell(dh,1), ndofs_per_cell(dh,1))
    Ke_rand = rand(ndofs_per_cell(dh,1), ndofs_per_cell(dh,1))
    dofs = celldofs(dh,1)

    for c1 ∈ [true, false], c2 ∈ [true, false], c3 ∈ [true, false], c4 ∈ [true, false]
        coupling = [c1; c2;; c3; c4]
        K = create_sparsity_pattern(SparseMatrixCSR, dh; coupling)
        a = start_assemble(K)
        assemble!(a, dofs, Ke_zeros)
        if all(coupling)
            assemble!(a, dofs, Ke_rand)
            @test Ke_rand ≈ K[dofs,dofs]
        else
            @test_throws ErrorException assemble!(a, dofs, Ke_rand)
        end
    end
end

end
