@testset "assemble" begin
    dofs = [1, 3, 5, 7]

    # residual
    ge = rand(4)
    g = zeros(8)
    assemble!(g, dofs, ge)
    @test g[1] == ge[1]
    @test g[3] == ge[2]
    @test g[5] == ge[3]
    @test g[7] == ge[4]

    # stiffness
    a = start_assemble()
    Ke = rand(4, 4)
    assemble!(a, dofs, Ke)
    K = end_assemble(a)
    @test K[1,1] == Ke[1,1]
    @test K[1,5] == Ke[1,3]
    @test K[5,1] == Ke[3,1]

    # assemble with different row and col dofs
    rdofs = [1,4,6]
    cdofs = [1,7]
    a = start_assemble()
    Ke = rand(length(rdofs), length(cdofs))
    assemble!(a, rdofs, cdofs, Ke)
    K = end_assemble(a)
    @test (K[rdofs,cdofs] .== Ke) |> all

    # SparseMatrix assembler
    K = spzeros(10, 10)
    f = zeros(10)
    ke = [rand(4, 4), rand(4, 4)]
    fe = [rand(4), rand(4)]
    dofs = [[1, 5, 3, 7], [10, 8, 2, 5]]
    for i in 1:2
        K[dofs[i], dofs[i]] += ke[i]
        f[dofs[i]] += fe[i]
    end

    Kc = copy(K)
    fc = copy(f)

    assembler = start_assemble(Kc)
    @test all(iszero, Kc.nzval) # start_assemble zeroes
    for i in 1:2
        assemble!(assembler, dofs[i], ke[i])
    end
    @test Kc ≈ K

    assembler = start_assemble(Kc, fc)
    @test all(iszero, Kc.nzval)
    @test all(iszero, fc)
    for i in 1:2
        assemble!(assembler, dofs[i], ke[i], fe[i])
    end
    @test Kc ≈ K
    @test fc ≈ f

    # No zero filling
    assembler = start_assemble(Kc, fc; fillzero=false)
    @test Kc ≈ K
    @test fc ≈ f
    for i in 1:2
        assemble!(assembler, dofs[i], ke[i], fe[i])
    end
    @test Kc ≈ 2K
    @test fc ≈ 2f

    # Error paths
    assembler = start_assemble(Kc, fc)
    @test_throws BoundsError assemble!(assembler, [11, 1, 2, 3], rand(4, 4))
    @test_throws BoundsError assemble!(assembler, [11, 1, 2, 3], rand(4, 4), rand(4))
    @test_throws AssertionError assemble!(assembler, [11, 1, 2], rand(4, 4))
    @test_throws AssertionError assemble!(assembler, [11, 1, 2, 3], rand(4, 4), rand(3))
end
