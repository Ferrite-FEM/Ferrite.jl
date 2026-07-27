using Ferrite, SparseArrays
import LinearAlgebra: Symmetric

@testset "assemble" begin
    dofs = [1, 3, 5, 7]
    maxd = maximum(dofs)

    # Vector assembly
    ge = rand(4)
    g = zeros(8)
    assemble!(g, dofs, ge)
    @test g[dofs] == ge

    # COOAssembler: matrix only, inferred size
    a = Ferrite.COOAssembler()
    Ke = rand(4, 4)
    assemble!(a, dofs, Ke)
    K, f = finish_assemble(a)
    @test K[dofs, dofs] == Ke
    @test size(K) == (maxd, maxd)
    @test isempty(f)

    # COOAssembler: matrix only, given size
    a = Ferrite.COOAssembler(10, 10)
    assemble!(a, dofs, Ke)
    K, f = finish_assemble(a)
    @test K[dofs, dofs] == Ke
    @test size(K) == (10, 10)
    @test isempty(f)

    # COOAssembler: matrix and vector, inferred size
    a = Ferrite.COOAssembler()
    assemble!(a, dofs, Ke, ge)
    K, f = finish_assemble(a)
    @test K[dofs, dofs] == Ke
    @test f[dofs] == ge
    @test size(K) == (maxd, maxd)
    @test length(f) == maxd

    # COOAssembler: matrix and vector, given size
    a = Ferrite.COOAssembler(10, 10)
    assemble!(a, dofs, Ke, ge)
    K, f = finish_assemble(a)
    @test K[dofs, dofs] == Ke
    @test f[dofs] == ge
    @test size(K) == (10, 10)
    @test length(f) == 10

    # COOAssembler: assemble with different row and col dofs
    rdofs = [1, 4, 6]
    cdofs = [1, 7]
    a = Ferrite.COOAssembler()
    Ke = rand(length(rdofs), length(cdofs))
    assemble!(a, rdofs, cdofs, Ke)
    K, _ = finish_assemble(a)
    @test all(K[rdofs, cdofs] .== Ke)

    # CSCAssembler: assemble with different row and col dofs
    I = [1, 1, 4, 4, 6, 6]
    J = [1, 3, 1, 3, 1, 3]
    for T in (Float32, Float64)
        V = zeros(T, length(I))
        K = sparse(I, J, V)
        f = zeros(T, 6)
        assembler = start_assemble(K, f)
        @test isa(assembler, Ferrite.AbstractAssembler{T})
        rdofs = [1, 4, 6]
        cdofs = [1, 3]
        Ke = rand(T, length(rdofs), length(cdofs))
        fe = rand(T, length(rdofs))
        assemble!(assembler, rdofs, cdofs, Ke, fe)
        assemble!(assembler, rdofs, cdofs, Ke, fe)
        @test_throws ArgumentError assemble!(assembler, rdofs, Ke, fe) # Not in sparsity pattern
        @test all(K[rdofs, cdofs] .== 2Ke)
        @test all(f[rdofs] .== 2fe)

        # CSCAssembler: Assemble rectangular part in quadratic matrix
        K = SparseMatrixCSC(6, 6, [K.colptr..., 7, 7, 7], K.rowval, K.nzval)
        assembler = start_assemble(K, f)
        rdofs = [1, 4, 6]
        cdofs = [1, 3]
        Ke = rand(T, length(rdofs), length(cdofs))
        fe = rand(T, length(rdofs))
        assemble!(assembler, rdofs, cdofs, Ke, fe)
        assemble!(assembler, rdofs, cdofs, Ke, fe)
        @test_throws ArgumentError assemble!(assembler, rdofs, Ke, fe) # Not in sparsity pattern
        @test all(K[rdofs, cdofs] .== 2Ke)
        @test all(f[rdofs] .== 2fe)

        # SparseMatrix assembler
        K = spzeros(T, 10, 10)
        f = zeros(T, 10)
        ke = [rand(T, 4, 4), rand(T, 4, 4)]
        fe = [rand(T, 4), rand(T, 4)]
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
        assembler = start_assemble(Kc, fc; fillzero = false)
        @test Kc ≈ K
        @test fc ≈ f
        for i in 1:2
            assemble!(assembler, dofs[i], ke[i], fe[i])
        end
        @test Kc ≈ 2K
        @test fc ≈ 2f
    end

    # Error paths
    K = sparse(I, J, zeros(length(I)))
    assembler = start_assemble(K, zeros(size(K, 1)))
    @test_throws BoundsError assemble!(assembler, [11, 1, 2, 3], rand(4, 4))
    @test_throws BoundsError assemble!(assembler, [11, 1, 2, 3], rand(4, 4), rand(4))
    @test_throws BoundsError assemble!(assembler, [11, 1, 2], rand(4, 4))
    @test_throws BoundsError assemble!(assembler, [11, 1, 2, 3], rand(4, 4), rand(3))
end

@testset "Base.show for assemblers" begin
    A = sparse(rand(10, 10))
    S = Symmetric(A)
    b = rand(10)
    @test occursin(
        r"for assembling into:\n - 10×10 SparseMatrix",
        sprint(show, MIME"text/plain"(), start_assemble(A)),
    )
    @test occursin(
        r"for assembling into:\n - 10×10 SparseMatrix.*\n - 10-element Vector",
        sprint(show, MIME"text/plain"(), start_assemble(A, b)),
    )
    @test occursin(
        r"for assembling into:\n - 10×10 Symmetric.*SparseMatrix",
        sprint(show, MIME"text/plain"(), start_assemble(S)),
    )
    @test occursin(
        r"for assembling into:\n - 10×10 Symmetric.*SparseMatrix.*\n - 10-element Vector",
        sprint(show, MIME"text/plain"(), start_assemble(S, b)),
    )
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
    store_dofs = [1, 5, 2, 8]
    assemble_dofs = [1, 5, 4, 8]
    I = repeat(store_dofs; outer = 4)
    J = repeat(store_dofs; inner = 4)
    V = zeros(length(I))
    K = sparse(I, J, V)
    D = zeros(size(K))

    # Standard assembler
    a = start_assemble(K)
    ke = rand(4, 4); ke[3, :] .= 0; ke[:, 3] .= 0; ke[2, 2] = 0
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
    errr(i, j) = try
        Ferrite._missing_sparsity_pattern_error(i, j)
    catch e
        e
    end
    ## Errors below diagonal
    @test_throws errr(2, 1) assemble!(a, [1, 2], [1.0 0.0; 3.0 4.0])
    @test_throws errr(2, 1) assemble!(a, [2, 1], [1.0 2.0; 0.0 4.0])
    ## Errors above diagonal
    @test_throws errr(1, 2) assemble!(a, [1, 2], [1.0 2.0; 0.0 4.0])
    @test_throws errr(1, 2) assemble!(as, [1, 2], [1.0 2.0; 0.0 4.0])
    @test_throws errr(1, 2) assemble!(a, [2, 1], [1.0 0.0; 3.0 4.0])
    @test_throws errr(1, 2) assemble!(as, [2, 1], [1.0 0.0; 3.0 4.0])
end

@testset "assemble! into long columns" begin
    # Cells 1:ncells each own `nl` consecutive dofs; the first `nshared` of them
    # additionally share the dof `g`, so column `g` holds many more entries than any
    # element has dofs. Such a column is scattered into with a binary search instead of a
    # merge walk over the whole column, and when *every* cell shares `g` the column is
    # dense and is indexed directly. All paths must agree with a plain COO assembly.
    nl, ncells = 8, 200
    g = nl + ncells
    for (nshared, dense) in ((ncells ÷ 2, false), (ncells, true))
        cells = [vcat(collect(c:(c + nl - 1)), c <= nshared ? [g] : Int[]) for c in 1:ncells]
        I, J = Int[], Int[]
        for dofs in cells, j in dofs, i in dofs
            push!(I, i); push!(J, j)
        end
        K = sparse(I, J, zeros(length(I)), g, g)
        # The two fast paths in `_assemble_inner!` are keyed on the column length
        @test (length(nzrange(K, g)) == size(K, 1)) == dense
        dense || @test length(nzrange(K, g)) > Ferrite.SPARSE_COLUMN_SEARCH_RATIO * (nl + 1)

        Kref = copy(K)
        a = start_assemble(K)
        for (c, dofs) in pairs(cells)
            m = length(dofs)
            ke = reshape(collect(1.0:(m^2)), m, m) .+ c
            ke .= ke .+ ke'
            assemble!(a, dofs, ke)
            for (jj, j) in pairs(dofs), (ii, i) in pairs(dofs)
                Kref[i, j] += ke[ii, jj]
            end
        end
        @test K == Kref

        # Symmetric assembler (upper triangle only) must agree as well
        Ks = sparse(I, J, zeros(length(I)), g, g)
        as = start_assemble(Symmetric(Ks))
        for (c, dofs) in pairs(cells)
            m = length(dofs)
            ke = reshape(collect(1.0:(m^2)), m, m) .+ c
            ke .= ke .+ ke'
            assemble!(as, dofs, ke)
        end
        @test triu(Ks) == triu(Kref)
    end

    # A missing entry in a long column must still be reported
    cells = [vcat(collect(c:(c + nl - 1)), [g]) for c in 1:(ncells ÷ 2)]
    I, J = Int[], Int[]
    for dofs in cells, j in dofs, i in dofs
        push!(I, i); push!(J, j)
    end
    K = sparse(I, J, zeros(length(I)), g, g)
    a = start_assemble(K)
    orphan = g - 1 # owned only by the last cells, which do not share `g`
    @test K[orphan, g] == 0 && !(orphan in rowvals(K)[nzrange(K, g)])
    ke = zeros(2, 2); ke[1, 2] = 1.0
    @test_throws ErrorException assemble!(a, [orphan, g], ke)
end
