using Ferrite, BlockArrays, SparseArrays, LinearAlgebra, Test
import SparseMatricesCSR: SparseMatrixCSR

# The constraint application interface is implemented per block type, so everything below is
# run for both of the sparse formats Ferrite ships with.
const BLOCK_TYPES = ("CSC" => SparseMatrixCSC{Float64, Int}, "CSR" => SparseMatrixCSR{1, Float64, Int})

@testset "BlockArrays.jl extension ($blockname blocks)" for (blockname, BT) in BLOCK_TYPES
    grid = generate_grid(Triangle, (10, 10))

    dh = DofHandler(grid)
    ip = Lagrange{RefTriangle, 1}()
    add!(dh, :u, ip^2)
    add!(dh, :p, ip)
    close!(dh)
    renumber!(dh, DofOrder.FieldWise())
    nd = ndofs(dh) ÷ 3

    ch = ConstraintHandler(dh)
    periodic_faces = collect_periodic_facets(grid, "top", "bottom")
    add!(ch, PeriodicDirichlet(:u, periodic_faces))
    add!(ch, Dirichlet(:u, union(getfacetset(grid, "left"), getfacetset(grid, "top")), (x, t) -> [0, 0]))
    add!(ch, Dirichlet(:p, getfacetset(grid, "left"), (x, t) -> 0))
    close!(ch)
    update!(ch, 0)

    K = allocate_matrix(dh, ch)
    f = zeros(axes(K, 1))
    # TODO: allocate_matrix(BlockMatrix, ...) should work and default to field blocking
    bsp = BlockSparsityPattern([2nd, 1nd])
    add_sparsity_entries!(bsp, dh, ch)
    KB = allocate_matrix(BlockMatrix{Float64, Matrix{BT}}, bsp)
    @test KB isa BlockMatrix
    @test blocksize(KB) == (2, 2)
    @test size(KB[Block(1), Block(1)]) == (2nd, 2nd)
    @test size(KB[Block(2), Block(1)]) == (1nd, 2nd)
    @test size(KB[Block(1), Block(2)]) == (2nd, 1nd)
    @test size(KB[Block(2), Block(2)]) == (1nd, 1nd)
    fB = similar(KB, axes(KB, 1))

    # Test the pattern
    fill!(K.nzval, 1)
    foreach(x -> fill!(x.nzval, 1), blocks(KB))
    @test K == KB

    # Zeroing out in start_assemble
    assembler = start_assemble(K, f; atomic = false)
    @test iszero(K)
    @test iszero(f)
    block_assembler = start_assemble(KB, fB)
    @test isa(block_assembler, Ferrite.AbstractAssembler{Float64})
    @test iszero(KB)
    @test iszero(fB)

    # Assembly procedure
    npc = ndofs_per_cell(dh)
    for cc in CellIterator(dh)
        ke = rand(npc, npc)
        fe = rand(npc)
        dofs = celldofs(cc)
        # Standard assemble
        assemble!(assembler, dofs, ke, fe)
        assemble!(block_assembler, dofs, ke, fe)
        # Assemble with local condensation of constraints
        let ke = copy(ke), fe = copy(fe)
            apply_assemble!(assembler, ch, dofs, ke, fe)
        end
        let ke = copy(ke), fe = copy(fe)
            apply_assemble!(block_assembler, ch, dofs, ke, fe)
        end
    end
    @test K ≈ KB
    @test f ≈ fB

    # Global application of BC, including the non-local affine (periodic) constraints
    fdofs = Ferrite.free_dofs(ch)
    pdofs = ch.prescribed_dofs
    let K = copy(K), f = copy(f), KB = copy(KB), fB = copy(fB)
        apply!(K, f, ch)
        apply!(KB, fB, ch)
        @test K ≈ KB
        @test f ≈ fB
        # The free block is the condensed system, the prescribed block is diagonal
        @test KB[fdofs, fdofs] ≈ K[fdofs, fdofs]
        @test isdiag(KB[pdofs, pdofs])
        @test K \ f ≈ KB \ fB
    end
    let K = copy(K), f = copy(f), KB = copy(KB), fB = copy(fB)
        apply_zero!(K, f, ch)
        apply_zero!(KB, fB, ch)
        @test K ≈ KB
        @test f ≈ fB
    end
    # Matrix only, i.e. with an empty rhs
    let K = copy(K), KB = copy(KB)
        apply!(K, ch)
        apply!(KB, ch)
        @test K ≈ KB
    end

    # Custom blocking
    perm = invperm([ch.free_dofs; ch.prescribed_dofs])
    renumber!(dh, ch, perm)
    nfree = length(ch.free_dofs)
    npres = length(ch.prescribed_dofs)
    K = allocate_matrix(dh, ch)
    block_sizes = [nfree, npres]
    bsp = BlockSparsityPattern(block_sizes)
    add_sparsity_entries!(bsp, dh, ch)
    KB = allocate_matrix(BlockMatrix{Float64, Matrix{BT}}, bsp)
    @test blocksize(KB) == (2, 2)
    @test size(KB[Block(1), Block(1)]) == (nfree, nfree)
    @test size(KB[Block(2), Block(1)]) == (npres, nfree)
    @test size(KB[Block(1), Block(2)]) == (nfree, npres)
    @test size(KB[Block(2), Block(2)]) == (npres, npres)
    # Test the pattern
    fill!(K.nzval, 1)
    foreach(x -> fill!(x.nzval, 1), blocks(KB))
    @test K == KB
end

@testset "BlockAssembler with atomic accumulation ($blockname blocks)" for (blockname, BT) in BLOCK_TYPES
    grid = generate_grid(Triangle, (10, 10))
    dh = DofHandler(grid)
    ip = Lagrange{RefTriangle, 1}()
    add!(dh, :u, ip^2)
    add!(dh, :p, ip)
    close!(dh)
    renumber!(dh, DofOrder.FieldWise())
    nd = ndofs(dh) ÷ 3

    # Periodic constraints couple dofs across elements, so `apply_assemble!` below has to
    # write into the global matrix and vector from `Ferrite._condense_local!`
    ch = ConstraintHandler(dh)
    add!(ch, PeriodicDirichlet(:u, collect_periodic_facets(grid, "top", "bottom")))
    add!(ch, Dirichlet(:p, getfacetset(grid, "left"), (x, t) -> 0))
    close!(ch)
    update!(ch, 0)

    bsp = BlockSparsityPattern([2nd, 1nd])
    add_sparsity_entries!(bsp, dh, ch)
    allocate_block_matrix() = allocate_matrix(BlockMatrix{Float64, Matrix{BT}}, bsp)

    # Deterministic fake element contributions computed from the dofs
    element_matrix(dofs) = [sin(i * j / 100) for i in dofs, j in dofs]
    element_vector(dofs) = [cos(i) for i in dofs]

    function doassemble!(assembler; condense = false)
        for cc in CellIterator(dh)
            dofs = celldofs(cc)
            ke = element_matrix(dofs)
            fe = element_vector(dofs)
            if condense
                apply_assemble!(assembler, ch, dofs, ke, fe)
            else
                assemble!(assembler, dofs, ke, fe)
            end
        end
        return
    end

    # Sequential atomic assembly gives bitwise identical results, both for plain assembly
    # and for assembly with local condensation of constraints
    for condense in (false, true)
        K = allocate_block_matrix()
        f = similar(K, axes(K, 1))
        Ka = allocate_block_matrix()
        fa = similar(Ka, axes(Ka, 1))
        doassemble!(start_assemble(K, f); condense)
        doassemble!(start_assemble(Ka, fa; atomic = true); condense)
        @test K == Ka
        @test f == fa
    end

    # A rhs which is not blocked like the matrix takes the fallback path in `assemble!`,
    # which must be atomic too
    K = allocate_block_matrix()
    f = zeros(ndofs(dh))
    Ka = allocate_block_matrix()
    fa = zeros(ndofs(dh))
    doassemble!(start_assemble(K, f))
    doassemble!(start_assemble(Ka, fa; atomic = true))
    @test K == Ka
    @test f == fa

    # Concurrent assembly: shared K and f, but one assembler per task and no coloring
    Kc = allocate_block_matrix()
    fc = similar(Kc, axes(Kc, 1))
    _ = start_assemble(Kc, fc) # zero out
    @sync for chunk in Iterators.partition(1:getncells(grid), cld(getncells(grid), 4))
        Threads.@spawn begin
            asm = start_assemble(Kc, fc; fillzero = false, atomic = true)
            for cellidx in chunk
                dofs = celldofs(dh, cellidx)
                apply_assemble!(asm, ch, dofs, element_matrix(dofs), element_vector(dofs))
            end
        end
    end
    Ks = allocate_block_matrix()
    fs = similar(Ks, axes(Ks, 1))
    doassemble!(start_assemble(Ks, fs; atomic = true); condense = true)
    # Equal up to the (task dependent) summation order
    @test Kc ≈ Ks rtol = 1.0e-14
    @test fc ≈ fs rtol = 1.0e-14

    # Atomic accumulation is only supported for real/complex Float16/Float32/Float64
    Ki = BlockArray(zeros(Int, 4, 4), [2, 2], [2, 2])
    fi = BlockArray(zeros(Int, 4), [2, 2])
    @test_throws ArgumentError start_assemble(Ki, fi; atomic = true)

    # Literal `atomic` values propagate to the type parameter, i.e. `@inferred` holds and
    # the flag is a compile time constant
    KB = allocate_block_matrix()
    fB = similar(KB, axes(KB, 1))
    @test Ferrite._is_atomic(@inferred ((K, f) -> start_assemble(K, f; atomic = true))(KB, fB))
    @test !Ferrite._is_atomic(@inferred ((K, f) -> start_assemble(K, f; atomic = false))(KB, fB))
    @test !Ferrite._is_atomic(@inferred ((K, f) -> start_assemble(K, f))(KB, fB))
end

@testset "BlockAssembler with dense blocks" begin
    # Dense blocks route through the generic matrix addindex! fallback
    KB = BlockArray(zeros(4, 4), [2, 2], [2, 2])
    fB = BlockArray(zeros(4), [2, 2])
    assembler = start_assemble(KB, fB)
    dofs = [1, 3]
    ke = [1.0 2.0; 3.0 4.0]
    fe = [5.0, 6.0]
    assemble!(assembler, dofs, ke, fe)
    D = zeros(4, 4)
    D[dofs, dofs] = ke
    g = zeros(4)
    g[dofs] = fe
    @test KB == D
    @test fB == g
    # Atomic assembly is not supported for dense blocks
    atomic_assembler = start_assemble(KB, fB; atomic = true)
    @test_throws ErrorException assemble!(atomic_assembler, dofs, ke, fe)
end
