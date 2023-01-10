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
    K, f = finish_assemble(a)
    @test isempty(f)
    @test K[1,1] == Ke[1,1]
    @test K[1,5] == Ke[1,3]
    @test K[5,1] == Ke[3,1]

    # matrix and vector
    a = start_assemble(zeros(maximum(dofs)))
    fe = rand(4)
    assemble!(a, dofs, Ke, fe)
    K1, f = finish_assemble(a)
    @test K == K1
    @test f[[1, 3, 5, 7]] == fe
    @test iszero(f[[2, 4, 6]])

    # assemble with different row and col dofs
    rdofs = [1,4,6]
    cdofs = [1,7]
    a = start_assemble()
    Ke = rand(length(rdofs), length(cdofs))
    assemble!(a, rdofs, cdofs, Ke)
    K, _ = finish_assemble(a)
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

struct IgnoreMeIfZero
    x::Float64
end
Base.iszero(x::IgnoreMeIfZero) = iszero(x.x)
function Base.:+(y::Float64, x::IgnoreMeIfZero)
    @test !iszero(x.x)
    return y + x.x
end

@testset "assemble! ignoring zeros" begin
    store_dofs    = [1, 5, 2, 8]
    assemble_dofs = [1, 5, 4, 8]
    I = repeat(store_dofs; outer=4)
    J = repeat(store_dofs; inner=4)
    V = zeros(length(I))
    K = sparse(I, J, V)
    D = zeros(size(K))

    # Standard assembler
    a = start_assemble(K)
    ke = rand(4,4); ke[3, :] .= 0; ke[:, 3] .= 0; ke[2,2] = 0
    assemble!(a, assemble_dofs, IgnoreMeIfZero.(ke))
    D[assemble_dofs, assemble_dofs] += ke
    @test K == D

    # Symmetric assembler
    S = Symmetric(K)
    assembler = start_assemble(S)
    fill!(D, 0)
    kes = [(ke[i, j] + ke[j, i]) / 2 for i in 1:4, j in 1:4]
    D[assemble_dofs, assemble_dofs] += kes
    kes[2, 1] = 42 # To check we don't touch elements below diagonal
    assemble!(assembler, assemble_dofs, IgnoreMeIfZero.(kes))
    @test S == D

    # Error paths
    K = spdiagm(0 => zeros(2))
    a = start_assemble(K)
    as = start_assemble(Symmetric(K))
    errr = ErrorException("some row indices were not found")
    ## Errors below diagonal
    @test_throws errr assemble!(a, [1, 2], [1.0 0.0; 3.0 4.0])
    @test_throws errr assemble!(a, [2, 1], [1.0 2.0; 0.0 4.0])
    ## Errors above diagonal
    @test_throws errr assemble!(a, [1, 2], [1.0 2.0; 0.0 4.0])
    @test_throws errr assemble!(as, [1, 2], [1.0 2.0; 0.0 4.0])
    @test_throws errr assemble!(a, [2, 1], [1.0 0.0; 3.0 4.0])
    @test_throws errr assemble!(as, [2, 1], [1.0 0.0; 3.0 4.0])
end
