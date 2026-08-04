using Ferrite, BlockArrays, SparseArrays, Test

@testset "BlockArrays.jl extension" begin
    grid = generate_grid(Triangle, (10, 10))

    dh = DofHandler(grid)
    ip = Lagrange{RefTriangle, 1}()
    add!(dh, :u, ip^2)
    add!(dh, :p, ip)
    add!(dh, :λ, SystemVariable{3}())
    close!(dh)
    renumber!(dh, DofOrder.FieldWise())

    # Number of nodal DOFs for a scalar field (subtracting the 3 global DOFs of :λ)
    nd = (ndofs(dh) - 3) ÷ 3

    ch = ConstraintHandler(dh)
    periodic_faces = collect_periodic_facets(grid, "top", "bottom")
    add!(ch, PeriodicDirichlet(:u, periodic_faces))
    add!(ch, Dirichlet(:u, union(getfacetset(grid, "left"), getfacetset(grid, "top")), (x, t) -> [0, 0]))
    add!(ch, Dirichlet(:p, getfacetset(grid, "left"), (x, t) -> 0))
    close!(ch)
    update!(ch, 0)

    # 1. Field-wise 3x3 block matrix allocation (:u, :p, :λ)
    sp = init_sparsity_pattern(dh)
    add_sparsity_entries!(sp, dh, ch)
    add_system_variable_entries!(sp, dh, 1:getncells(grid); cell_fields=[:u, :p], system_variable=:λ)
    K = allocate_matrix(sp)
    f = zeros(axes(K, 1))

    bsp = BlockSparsityPattern([2nd, 1nd, 3])
    add_sparsity_entries!(bsp, dh, ch)
    add_system_variable_entries!(bsp, dh, 1:getncells(grid); cell_fields=[:u, :p], system_variable=:λ)
    KB = allocate_matrix(BlockMatrix, bsp)

    @test KB isa BlockMatrix
    @test blocksize(KB) == (3, 3)

    # Row 1 (:u)
    @test size(KB[Block(1), Block(1)]) == (2nd, 2nd)
    @test size(KB[Block(1), Block(2)]) == (2nd, 1nd)
    @test size(KB[Block(1), Block(3)]) == (2nd, 3)
    # Row 2 (:p)
    @test size(KB[Block(2), Block(1)]) == (1nd, 2nd)
    @test size(KB[Block(2), Block(2)]) == (1nd, 1nd)
    @test size(KB[Block(2), Block(3)]) == (1nd, 3)
    # Row 3 (:λ)
    @test size(KB[Block(3), Block(1)]) == (3, 2nd)
    @test size(KB[Block(3), Block(2)]) == (3, 1nd)
    @test size(KB[Block(3), Block(3)]) == (3, 3)

    fB = similar(KB, axes(KB, 1))

    # Test the pattern
    fill!(K.nzval, 1)
    foreach(x -> fill!(x.nzval, 1), blocks(KB))
    @test K == KB

    # Zeroing out in start_assemble
    assembler = start_assemble(K, f)
    @test iszero(K)
    @test iszero(f)
    block_assembler = start_assemble(KB, fB)
    @test isa(block_assembler, Ferrite.AbstractAssembler{Float64})
    @test iszero(KB)
    @test iszero(fB)

    # Assembly procedure (cell DOFs + system variable DOFs)
    λ_dofs = system_variable_dofs(dh, :λ)
    npc = ndofs_per_cell(dh)
    n_ext = npc
    nλ = length(λ_dofs)

    for cc in CellIterator(dh)
        ke = rand(n_ext, n_ext)
        fe = rand(n_ext)
        
        # Standard assemble
        assemble!(assembler, dofs, ke, fe)
        assemble!(block_assembler, dofs, ke, fe)
        
        #TODO: rectangular block array assembler does not implemented
        # ke_uλ = rand(n_ext, nλ)
        # assemble!(assembler, dofs, λ_dofs, ke_uλ)
        # assemble!(block_assembler, dofs, λ_dofs, ke_uλ)

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

    # Global application of BC not supported yet
    @test_throws ErrorException apply!(KB, fB, ch)

    # 2. Custom blocking (Free vs Prescribed DOFs)
    perm = invperm([ch.free_dofs; ch.prescribed_dofs])
    renumber!(dh, ch, perm)
    nfree = length(ch.free_dofs)
    npres = length(ch.prescribed_dofs)

    sp = init_sparsity_pattern(dh)
    add_sparsity_entries!(sp, dh, ch)
    add_system_variable_entries!(sp, dh, 1:getncells(grid); cell_fields=[:u, :p], system_variable=:λ)
    K = allocate_matrix(sp)

    block_sizes = [nfree, npres]
    bsp = BlockSparsityPattern(block_sizes)
    add_sparsity_entries!(bsp, dh, ch)
    add_system_variable_entries!(bsp, dh, 1:getncells(grid); cell_fields=[:u, :p], system_variable=:λ)
    KB = allocate_matrix(BlockMatrix, bsp)

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