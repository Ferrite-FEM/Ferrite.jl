using Ferrite
using Test
using LinearAlgebra
using SparseArrays
using SparseMatricesCSR
using BlockArrays
using ForwardDiff
using OrderedCollections
using Random: randperm

# Structural (pattern) membership test for CSC matrices
hasentry(K::SparseMatrixCSC, i::Int, j::Int) = i in view(rowvals(K), nzrange(K, j))

@testset "AlgebraicVariable declarations" begin
    # Component counts and canonical (Tensors.jl storage) order
    @test AlgebraicVariable().components == (1,)
    @test AlgebraicVariable{Vec{3}}().components == (1, 2, 3)
    @test AlgebraicVariable{Tensor{2, 2}}().components == ((1, 1), (2, 1), (1, 2), (2, 2))
    @test AlgebraicVariable{SymmetricTensor{2, 2}}().components == ((1, 1), (2, 1), (2, 2))
    @test AlgebraicVariable{SymmetricTensor{2, 3}}().components ==
        ((1, 1), (2, 1), (3, 1), (2, 2), (3, 2), (3, 3))
    # Unsupported shapes
    @test_throws ErrorException AlgebraicVariable{Float64}()
    @test_throws ErrorException AlgebraicVariable{Vec{3, Float64}}() # coefficient type must not be fixed
    @test_throws ErrorException AlgebraicVariable{SymmetricTensor{2, 2, Float64}}()
    # Unsupported dimensions (Tensors.jl supports 1:3; dim 0 would own no dofs)
    @test_throws ErrorException AlgebraicVariable{Vec{0}}()
    @test_throws ErrorException AlgebraicVariable{Vec{4}}()
    @test_throws ErrorException AlgebraicVariable{Tensor{2, 4}}()
    # Fourth order tensors
    @test Ferrite.n_algebraic_dofs(AlgebraicVariable{Tensor{4, 2}}()) == 16
    @test Ferrite.n_algebraic_dofs(AlgebraicVariable{SymmetricTensor{4, 2}}()) == 9
    @test Ferrite.n_algebraic_dofs(AlgebraicVariable{SymmetricTensor{4, 3}}()) == 36
end

@testset "fourth order value reconstruction and basis" begin
    grid = generate_grid(Triangle, (1, 1))
    dh = DofHandler(grid)
    add!(dh, :u, Lagrange{RefTriangle, 1}())
    A4var = AlgebraicVariable{Tensor{4, 2}}()
    Cvar = AlgebraicVariable{SymmetricTensor{4, 2}}()
    add!(dh, :A4, A4var)
    add!(dh, :C, Cvar)
    close!(dh)
    for (name, var) in ((:A4, A4var), (:C, Cvar))
        comps = var.components
        dofs = algebraic_dofs(dh, name)
        a = zeros(ndofs(dh))
        a[dofs] .= 1:length(dofs)
        val = algebraic_value(dh, a, name)
        # canonical order matches the reconstructed value's natural indexing
        for (k, c) in pairs(comps)
            @test val[c...] == k
        end
        # the value is the sum of coefficients times basis directions
        av = AlgebraicValues(var)
        @test val == sum(a[dofs[k]] * algebraic_basis_value(av, k) for k in 1:length(dofs))
    end
    # minor symmetry of the reconstructed SymmetricTensor{4}
    a = zeros(ndofs(dh))
    a[algebraic_dofs(dh, :C)] .= 1:9
    C = algebraic_value(dh, a, :C)
    @test C[1, 2, 1, 1] == C[2, 1, 1, 1]
    @test C[1, 1, 1, 2] == C[1, 1, 2, 1]
end

@testset "purely algebraic DofHandler" begin
    grid = generate_grid(Triangle, (2, 2))
    dh = DofHandler(grid)
    add!(dh, :z, AlgebraicVariable{Vec{2}}())
    add!(dh, :p0, AlgebraicVariable())
    close!(dh)
    @test ndofs(dh) == 3
    # Matrix allocation without and with a descriptor
    K0 = allocate_matrix(dh)
    @test size(K0) == (3, 3)
    for d in 1:3
        @test hasentry(K0, d, d)
    end
    cpl = AlgebraicCoupling(; algebraic_coupling = ((:z, :z), (:z, :p0), (:p0, :p0)))
    K = allocate_matrix(dh; algebraic_couplings = (cpl,))
    f = zeros(3)
    assembler = start_assemble(K, f)
    dofs = vcat(algebraic_dofs(dh, :z), algebraic_dofs(dh, :p0))
    Ke = [2.0 0.0 1.0; 0.0 2.0 0.0; 1.0 0.0 2.0]
    assemble!(assembler, dofs, Ke, ones(3))
    K, f = finish_assemble(assembler)
    @test K[dofs, dofs] ≈ Ke
    # Cell/facet descriptors with empty entity sets contribute nothing (in particular on a
    # DofHandler without any SubDofHandlers)
    empty_cpl = CellCoupling(Int[]; algebraic_coupling = ((:z, :p0),))
    empty_fpl = FacetCoupling(Set{FacetIndex}(); algebraic_coupling = ((:z, :p0),))
    K_empty = allocate_matrix(dh; algebraic_couplings = (empty_cpl, empty_fpl))
    @test nnz(K_empty) == nnz(K0)
    # No dangling "Fields:" header when there are no spatial fields
    dhstr = sprint(show, MIME"text/plain"(), dh)
    @test !occursin("Fields:", dhstr)
    @test occursin("Algebraic variables:", dhstr)
end

@testset "descriptor immutability and algebraic_couplings forms" begin
    grid = generate_grid(Quadrilateral, (3, 3))
    dh = DofHandler(grid)
    add!(dh, :u, Lagrange{RefQuadrilateral, 1}())
    add!(dh, :p0, AlgebraicVariable())
    close!(dh)
    @test Ferrite.has_algebraic_variables(dh)
    # The descriptor owns its entity set: mutating the input set afterwards has no effect
    myset = OrderedCollections.OrderedSet{FacetIndex}(getfacetset(grid, "right"))
    cpl = FacetCoupling(myset; algebraic_coupling = ((:u, :p0),))
    n_before = length(Ferrite.entities(cpl))
    push!(myset, first(getfacetset(grid, "left")))
    @test length(Ferrite.entities(cpl)) == n_before
    # Accessors return copies
    ents = Ferrite.entities(cpl)
    push!(ents, first(getfacetset(grid, "left")))
    @test length(Ferrite.entities(cpl)) == n_before
    @test Ferrite.fields(cpl) == (:u, :p0)
    # Empty collections work, including the documented default passed explicitly
    K_empty = allocate_matrix(dh; algebraic_couplings = Ferrite.AbstractCoupling[])
    @test size(K_empty) == (ndofs(dh), ndofs(dh))
    @test nnz(allocate_matrix(dh; algebraic_couplings = ())) == nnz(allocate_matrix(dh))
    # Any iterable of descriptors works (e.g. a generator)
    cpls = [cpl]
    K1 = allocate_matrix(dh; algebraic_couplings = (c for c in cpls))
    K2 = allocate_matrix(dh; algebraic_couplings = cpls)
    @test nnz(K1) == nnz(K2)
    # Junk still errors descriptively, both non-descriptor elements and non-iterables
    @test_throws ErrorException allocate_matrix(dh; algebraic_couplings = (1, 2))
    @test_throws ErrorException allocate_matrix(dh; algebraic_couplings = nothing)
end

@testset "algebraic_coupling pair specification" begin
    grid = generate_grid(Quadrilateral, (3, 3))
    dh = DofHandler(grid)
    add!(dh, :u, Lagrange{RefQuadrilateral, 1}()^2)
    add!(dh, :p, Lagrange{RefQuadrilateral, 1}())
    add!(dh, :λ, AlgebraicVariable())
    add!(dh, :z, AlgebraicVariable{Vec{2}}())
    close!(dh)
    λdof = only(algebraic_dofs(dh, :λ))
    zdofs = algebraic_dofs(dh, :z)
    cells = 1:getncells(grid)
    pattern(K) = (K.colptr, rowvals(K))
    # A symmetric tuple couples both directions
    K_pair = allocate_matrix(dh; algebraic_couplings = CellCoupling(cells; algebraic_coupling = ((:p, :λ),)))
    # `=>` couples one way only (test => trial)
    K_dir = allocate_matrix(dh; algebraic_couplings = CellCoupling(cells; algebraic_coupling = (:p => :λ,)))
    pdof = first(celldofs(dh, 1)[dof_range(dh, :p)])
    @test hasentry(K_dir, pdof, λdof) && !hasentry(K_dir, λdof, pdof)
    # A bare entry is accepted as a one-entry specification
    K_bare_pair = allocate_matrix(dh; algebraic_couplings = CellCoupling(cells; algebraic_coupling = :p => :λ))
    @test pattern(K_bare_pair) == pattern(K_dir)
    K_bare_tuple = allocate_matrix(dh; algebraic_couplings = CellCoupling(cells; algebraic_coupling = (:p, :λ)))
    @test pattern(K_bare_tuple) == pattern(K_pair)
    # Participating variables are derived from the entries, in order of first appearance
    cpl = CellCoupling(cells; algebraic_coupling = (:λ => :p, (:z, :p), :z => :z))
    @test Ferrite.fields(cpl) == (:λ, :p, :z)
    # Symmetric self entry of a multi-dof variable gives its dense block
    K_self = allocate_matrix(dh; algebraic_couplings = AlgebraicCoupling(; algebraic_coupling = ((:z, :z),)))
    for i in zdofs, j in zdofs
        @test hasentry(K_self, i, j)
    end
    # The same directed coupling from a pair and a tuple collides
    @test_throws ErrorException CellCoupling(cells; algebraic_coupling = (:p => :λ, (:p, :λ)))
    @test_throws ErrorException CellCoupling(cells; algebraic_coupling = (:p => :λ, :p => :λ))
    # Every entry must involve an algebraic variable (names are resolved when the
    # descriptor is used against a DofHandler)
    @test_throws ErrorException allocate_matrix(dh; algebraic_couplings = CellCoupling(cells; algebraic_coupling = (:u => :p,)))
    @test_throws ErrorException allocate_matrix(dh; algebraic_couplings = FacetCoupling(getfacetset(grid, "right"); algebraic_coupling = ((:u, :p), (:u, :λ))))
    # The internal Boolean matrix is not a user-facing input form
    @test_throws ErrorException CellCoupling(cells; algebraic_coupling = Bool[0 1; 1 0])
    # Uninterpretable specifications error descriptively
    @test_throws ErrorException CellCoupling(cells; algebraic_coupling = ())
    @test_throws ErrorException CellCoupling(cells; algebraic_coupling = ((:p, :λ, :z),))
    @test_throws ErrorException CellCoupling(cells; algebraic_coupling = (:p,))
    @test_throws ErrorException CellCoupling(cells; algebraic_coupling = :p)
    # An unknown variable name errors when the descriptor is used
    @test_throws ErrorException allocate_matrix(dh; algebraic_couplings = CellCoupling(cells; algebraic_coupling = (:p => :nope,)))
end

@testset "DofHandler integration" begin
    grid = generate_grid(Triangle, (2, 2))
    dh = DofHandler(grid)
    add!(dh, :u, Lagrange{RefTriangle, 1}()^2)
    # add! returns the handler
    @test add!(dh, :p0, AlgebraicVariable()) === dh
    @test add!(dh, :z, AlgebraicVariable{Vec{3}}()) === dh
    add!(dh, :σ̄, AlgebraicVariable{SymmetricTensor{2, 2}}())
    # Name conflicts in both declaration orders
    @test_throws ErrorException add!(dh, :u, AlgebraicVariable()) # spatial exists
    @test_throws ErrorException add!(dh, :p0, AlgebraicVariable()) # algebraic exists
    @test_throws ErrorException add!(dh, :p0, Lagrange{RefTriangle, 1}()) # algebraic exists
    close!(dh)
    # Adding after close!
    @test_throws ErrorException add!(dh, :q, AlgebraicVariable())
    # Deterministic initial numbering: spatial dofs first, then algebraic in declaration
    # order and canonical component order
    nspatial = 2 * getnnodes(grid)
    @test ndofs(dh) == nspatial + 1 + 3 + 3
    @test algebraic_dofs(dh, :p0) == [nspatial + 1]
    @test algebraic_dofs(dh, :z) == collect(nspatial .+ (2:4))
    @test algebraic_dofs(dh, :σ̄) == collect(nspatial .+ (5:7))
    # celldofs remain purely spatial
    @test maximum(maximum, (celldofs(dh, i) for i in 1:getncells(grid))) == nspatial
    @test ndofs_per_cell(dh) == 6
    @test Ferrite.getfieldnames(dh) == [:u]
    # algebraic_dofs returns a copy
    d1 = algebraic_dofs(dh, :p0)
    d1[1] = -1
    @test algebraic_dofs(dh, :p0) == [nspatial + 1]
    # Unknown-name and wrong-kind errors
    @test_throws ErrorException algebraic_dofs(dh, :u)
    @test_throws ErrorException Ferrite.find_field(dh, :p0)
    @test_throws ErrorException dof_range(dh, :p0)
end

@testset "algebraic_value and AlgebraicValues" begin
    grid = generate_grid(Triangle, (2, 2))
    dh = DofHandler(grid)
    add!(dh, :u, Lagrange{RefTriangle, 1}())
    p0var = AlgebraicVariable()
    zvar = AlgebraicVariable{Vec{3}}()
    Avar = AlgebraicVariable{Tensor{2, 2}}()
    σ̄var = AlgebraicVariable{SymmetricTensor{2, 2}}()
    add!(dh, :p0, p0var)
    add!(dh, :z, zvar)
    add!(dh, :A, Avar)
    add!(dh, :σ̄, σ̄var)
    close!(dh)
    for T in (Float64, Float32)
        a = zeros(T, ndofs(dh))
        a[algebraic_dofs(dh, :p0)] .= 3
        a[algebraic_dofs(dh, :z)] .= [1, 2, 4]
        a[algebraic_dofs(dh, :A)] .= [1, 2, 3, 4]
        a[algebraic_dofs(dh, :σ̄)] .= [5, 7, 6]
        p0 = algebraic_value(dh, a, :p0)
        @test p0 === T(3)
        z = algebraic_value(dh, a, :z)
        @test z isa Vec{3, T}
        @test z == Vec{3, T}((1, 2, 4))
        A = algebraic_value(dh, a, :A)
        @test A isa Tensor{2, 2, T}
        @test A == Tensor{2, 2, T}((1, 2, 3, 4)) # column major, matching Tensors storage
        σ̄ = algebraic_value(dh, a, :σ̄)
        @test σ̄ isa SymmetricTensor{2, 2, T}
        @test σ̄ == SymmetricTensor{2, 2, T}((5, 7, 6)) # (1,1), (2,1), (2,2)
        @test σ̄[1, 2] == σ̄[2, 1] == 7
    end
    # Wrong-kind and length errors
    a = zeros(ndofs(dh))
    @test_throws ErrorException algebraic_value(dh, a, :u)
    @test_throws ErrorException algebraic_value(dh, a, :nope)
    @test_throws ErrorException algebraic_value(dh, zeros(3), :p0)
    # DofHandler-free reconstruction from an extracted coefficient slice, in canonical
    # component order, matching the global-vector method
    for T in (Float64, Float32)
        a = zeros(T, ndofs(dh))
        a[algebraic_dofs(dh, :z)] .= [1, 2, 4]
        a[algebraic_dofs(dh, :σ̄)] .= [5, 7, 6]
        for (name, var) in ((:p0, p0var), (:z, zvar), (:A, Avar), (:σ̄, σ̄var))
            av = AlgebraicValues(T, var)
            dofvals = a[algebraic_dofs(dh, name)]
            @test algebraic_value(av, dofvals) === algebraic_value(dh, a, name)
            @test algebraic_value(av, view(a, algebraic_dofs(dh, name))) === algebraic_value(dh, a, name)
        end
    end
    @test algebraic_value(AlgebraicValues(AlgebraicVariable()), [4.0]) === 4.0
    # The scalar type follows the input, so dual numbers pass through (AD inside a kernel)
    let av = AlgebraicValues(AlgebraicVariable{SymmetricTensor{2, 2}}())
        d1 = ForwardDiff.Dual(5.0, 1.0, 0.0, 0.0)
        d2 = ForwardDiff.Dual(7.0, 0.0, 1.0, 0.0)
        d3 = ForwardDiff.Dual(6.0, 0.0, 0.0, 1.0)
        σ̄d = algebraic_value(av, [d1, d2, d3])
        @test σ̄d isa SymmetricTensor{2, 2, typeof(d1)}
        @test ForwardDiff.value(σ̄d[1, 1]) == 5.0
        @test ForwardDiff.value(σ̄d[2, 2]) == 6.0
        @test ForwardDiff.value(σ̄d[2, 1]) == 7.0
        # ∂σ̄/∂coefficient recovers the basis directions
        for k in 1:3
            E = algebraic_basis_value(av, k)
            for i in 1:2, j in 1:2
                @test ForwardDiff.partials(σ̄d[i, j], k) == E[i, j]
            end
        end
    end
    # Length mismatch (e.g. passing the full local vector instead of the variable's slice)
    zv = AlgebraicValues(zvar)
    @test_throws ErrorException algebraic_value(zv, zeros(2))
    # Basis directions map dof index -> canonical component
    @test algebraic_basis_value(zv, 1) == Vec{3}((1.0, 0.0, 0.0))
    @test algebraic_basis_value(zv, 3) == Vec{3}((0.0, 0.0, 1.0))
    @test algebraic_basis_value(AlgebraicValues(Float32, zvar), 2) isa Vec{3, Float32}
    @test_throws ErrorException algebraic_basis_value(zv, 4)
    @test_throws ErrorException algebraic_basis_value(zv, 0)
    @test algebraic_basis_value(AlgebraicValues(p0var), 1) === 1.0
    σ̄full = AlgebraicValues(AlgebraicVariable{SymmetricTensor{2, 2}}())
    E2 = algebraic_basis_value(σ̄full, 2)
    @test E2 == SymmetricTensor{2, 2}((0.0, 1.0, 0.0)) # ones in (2,1) and (1,2)
    @test E2[1, 2] == E2[2, 1] == 1.0
    @test algebraic_basis_value(AlgebraicValues(Avar), 3) == Tensor{2, 2}((0.0, 0.0, 1.0, 0.0))
    # The basis is the derivative of the reconstructed value w.r.t. the coefficient
    ac = algebraic_dofs(dh, :σ̄)
    σ̄v = AlgebraicValues(σ̄var)
    for i in 1:3
        ei = zeros(ndofs(dh)); ei[ac[i]] = 1.0
        @test algebraic_value(dh, ei, :σ̄) == algebraic_basis_value(σ̄v, i)
    end
    let av = @inferred AlgebraicValues(AlgebraicVariable{SymmetricTensor{2, 2}}())
        @test getnbasefunctions(av) == 3
        @test (@inferred algebraic_basis_value(av, 2)) == SymmetricTensor{2, 2}((0.0, 1.0, 0.0))
        @test (@inferred algebraic_value(av, [1.0, 2.0, 3.0])) == SymmetricTensor{2, 2}((1.0, 2.0, 3.0))
        ae = zeros(10); ae[8:10] .= [1.0, 2.0, 3.0]
        @test (@inferred algebraic_value(av, ae, 8:10)) == SymmetricTensor{2, 2}((1.0, 2.0, 3.0))
        @test occursin("AlgebraicValues", sprint(show, MIME"text/plain"(), av))
    end
end

@testset "renumbering" begin
    grid = generate_grid(Triangle, (3, 3))
    dh = DofHandler(grid)
    add!(dh, :u, Lagrange{RefTriangle, 1}()^2)
    add!(dh, :p, Lagrange{RefTriangle, 1}())
    add!(dh, :p0, AlgebraicVariable())
    add!(dh, :σ̄, AlgebraicVariable{SymmetricTensor{2, 2}}())
    close!(dh)
    n = ndofs(dh)
    # Arbitrary permutation and inverse round trip; value reconstruction follows
    perm = randperm(n)
    renumber!(dh, perm)
    a = zeros(n)
    a[algebraic_dofs(dh, :σ̄)] .= [1.0, 3.0, 2.0]
    @test algebraic_value(dh, a, :σ̄) == SymmetricTensor{2, 2}((1.0, 3.0, 2.0))
    renumber!(dh, invperm(perm))
    @test algebraic_dofs(dh, :p0) == [n - 3]
    @test algebraic_dofs(dh, :σ̄) == [n - 2, n - 1, n]
    # Field-wise: every algebraic variable is one block
    renumber!(dh, DofOrder.FieldWise())
    @test algebraic_dofs(dh, :p0) == [n - 3]
    @test algebraic_dofs(dh, :σ̄) == [n - 2, n - 1, n]
    # Custom block targets over the combined variable ordering (u, p, p0, σ̄)
    renumber!(dh, DofOrder.FieldWise([2, 2, 1, 1]))
    @test algebraic_dofs(dh, :p0) == [1]
    @test algebraic_dofs(dh, :σ̄) == [2, 3, 4]
    @test_throws ErrorException renumber!(dh, DofOrder.FieldWise([1, 2, 1])) # wrong length
    # Component-wise: components (ux, uy, p, p0, σ̄₁₁, σ̄₂₁, σ̄₂₂)
    renumber!(dh, DofOrder.ComponentWise())
    @test algebraic_dofs(dh, :σ̄) == [n - 2, n - 1, n]
    renumber!(dh, DofOrder.ComponentWise([7, 6, 5, 4, 3, 2, 1]))
    @test algebraic_dofs(dh, :p0) == [4]
    @test algebraic_dofs(dh, :σ̄) == [3, 2, 1]
    @test_throws ErrorException renumber!(dh, DofOrder.ComponentWise(collect(1:6))) # wrong length
    # Unchanged behavior without algebraic variables
    dh2 = DofHandler(grid)
    add!(dh2, :u, Lagrange{RefTriangle, 1}()^2)
    add!(dh2, :p, Lagrange{RefTriangle, 1}())
    close!(dh2)
    renumber!(dh2, DofOrder.FieldWise())
    @test Set(reduce(vcat, [celldofs(dh2, i) for i in 1:getncells(grid)])) == Set(1:ndofs(dh2))
    renumber!(dh2, DofOrder.ComponentWise())
end

@testset "cell-coupling sparsity" begin
    grid = generate_grid(Quadrilateral, (3, 3))
    dh = DofHandler(grid)
    add!(dh, :u, Lagrange{RefQuadrilateral, 1}()^2)
    add!(dh, :p, Lagrange{RefQuadrilateral, 1}())
    add!(dh, :λ, AlgebraicVariable())
    close!(dh)
    λdof = only(algebraic_dofs(dh, :λ))
    subset = [1, 5]
    subset_dofs(name) = unique!(sort!(reduce(vcat, [celldofs(dh, i)[dof_range(dh, name)] for i in subset])))
    other_dofs(name) = setdiff(
        unique!(sort!(reduce(vcat, [celldofs(dh, i)[dof_range(dh, name)] for i in setdiff(1:getncells(grid), subset)]))),
        subset_dofs(name),
    )
    # Asymmetric coupling on a cell subset, only field :p couples
    cpl = CellCoupling(subset; algebraic_coupling = (:p => :λ,))
    K = allocate_matrix(dh; algebraic_couplings = cpl) # single descriptor, no tuple
    for d in subset_dofs(:p)
        @test hasentry(K, d, λdof)   # test p, trial λ
        @test !hasentry(K, λdof, d)  # asymmetric: no test λ, trial p
    end
    for d in other_dofs(:p)
        @test !hasentry(K, d, λdof)
    end
    # Unselected field :u gets no entries
    for d in subset_dofs(:u)
        @test !hasentry(K, d, λdof) && !hasentry(K, λdof, d)
    end
    # Diagonal is always present, even without self-coupling descriptors
    @test hasentry(K, λdof, λdof)
    # Multiple SubDofHandlers: field :p only on the first half
    dh3 = DofHandler(grid)
    sdh1 = SubDofHandler(dh3, Set(1:4))
    add!(sdh1, :u, Lagrange{RefQuadrilateral, 1}()^2)
    add!(sdh1, :p, Lagrange{RefQuadrilateral, 1}())
    sdh2 = SubDofHandler(dh3, Set(5:9))
    add!(sdh2, :u, Lagrange{RefQuadrilateral, 1}()^2)
    add!(dh3, :λ, AlgebraicVariable())
    close!(dh3)
    λ3 = only(algebraic_dofs(dh3, :λ))
    cpl3 = CellCoupling(1:getncells(grid); algebraic_coupling = ((:u, :λ),))
    K3 = allocate_matrix(dh3; algebraic_couplings = (cpl3,))
    for cell in (2, 7), d in celldofs(dh3, cell)[dof_range(dh3.subdofhandlers[dh3.cell_to_subdofhandler[cell]], :u)]
        @test hasentry(K3, d, λ3)
    end
    # Clear failure when a selected cell lacks a named field
    cpl_bad = CellCoupling(1:getncells(grid); algebraic_coupling = ((:p, :λ),))
    @test_throws ErrorException allocate_matrix(dh3; algebraic_couplings = (cpl_bad,))
    # But restricting to cells that carry the field works
    allocate_matrix(dh3; algebraic_couplings = CellCoupling(1:4; algebraic_coupling = ((:p, :λ),)))
    # A descriptor is DofHandler-independent and can be reused with several DofHandlers
    K_reuse = allocate_matrix(dh; algebraic_couplings = (cpl3,))
    @test hasentry(K_reuse, first(celldofs(dh, 1)[dof_range(dh, :u)]), λdof)
end

@testset "facet-coupling sparsity" begin
    grid = generate_grid(Quadrilateral, (3, 3))
    dh = DofHandler(grid)
    add!(dh, :u, Lagrange{RefQuadrilateral, 2}()^2) # quadratic: interior (non-trace) dofs exist
    add!(dh, :p0, AlgebraicVariable())
    close!(dh)
    p0dof = only(algebraic_dofs(dh, :p0))
    right = getfacetset(grid, "right")
    left = getfacetset(grid, "left")
    cpl = FacetCoupling(right; algebraic_coupling = ((:u, :p0),))
    K = allocate_matrix(dh; algebraic_couplings = (cpl,))
    adjacent = Set(first(f) for f in right)
    for cellid in 1:getncells(grid)
        dofs = celldofs(dh, cellid)
        if cellid in adjacent
            # All dofs of the adjacent cell couple, including non-trace (interior) dofs
            for d in dofs
                @test hasentry(K, d, p0dof) && hasentry(K, p0dof, d)
            end
        end
    end
    # Interior/left cells remain uncoupled (only dofs not shared with adjacent cells)
    adjacent_dofs = Set(reduce(vcat, [celldofs(dh, c) for c in adjacent]))
    for f in left
        for d in setdiff(celldofs(dh, first(f)), adjacent_dofs)
            @test !hasentry(K, d, p0dof) && !hasentry(K, p0dof, d)
        end
    end
    # Asymmetric facet mask
    cpl_asym = FacetCoupling(right; algebraic_coupling = (:u => :p0,))
    Ka = allocate_matrix(dh; algebraic_couplings = (cpl_asym,))
    d = first(celldofs(dh, first(first(right))))
    @test hasentry(Ka, d, p0dof) && !hasentry(Ka, p0dof, d)
    # Multiple boundaries with separate descriptors; overlapping descriptors -> union
    cpl_left = FacetCoupling(left; algebraic_coupling = ((:u, :p0),))
    K2 = allocate_matrix(dh; algebraic_couplings = (vol = cpl, vol2 = cpl, lft = cpl_left)) # duplicate on purpose
    for f in union(right, left), d in celldofs(dh, first(f))
        @test hasentry(K2, d, p0dof)
    end
    # Vector of descriptors also works
    K3 = allocate_matrix(dh; algebraic_couplings = [cpl, cpl_left])
    @test nnz(K3) == nnz(K2)
end

@testset "algebraic-only sparsity" begin
    grid = generate_grid(Triangle, (2, 2))
    dh = DofHandler(grid)
    add!(dh, :u, Lagrange{RefTriangle, 1}())
    add!(dh, :z, AlgebraicVariable{Vec{3}}())
    add!(dh, :w, AlgebraicVariable{Vec{2}}())
    close!(dh)
    zdofs = algebraic_dofs(dh, :z)
    wdofs = algebraic_dofs(dh, :w)
    # Expected diagonal-only storage when no coupling is declared
    K0 = allocate_matrix(dh)
    for (i, d) in pairs(zdofs)
        @test hasentry(K0, d, d)
        @test !hasentry(K0, d, zdofs[mod1(i + 1, 3)])
    end
    # Dense self-coupling of one variable
    K1 = allocate_matrix(dh; algebraic_couplings = AlgebraicCoupling(; algebraic_coupling = ((:z, :z),)))
    for i in zdofs, j in zdofs
        @test hasentry(K1, i, j)
    end
    @test !hasentry(K1, zdofs[1], wdofs[1])
    # Asymmetric cross-coupling between two variables
    K2 = allocate_matrix(dh; algebraic_couplings = AlgebraicCoupling(; algebraic_coupling = (:z => :w,)))
    for i in zdofs, j in wdofs
        @test hasentry(K2, i, j)
        @test !hasentry(K2, j, i)
    end
    # AlgebraicCoupling rejects spatial fields when used
    @test_throws ErrorException allocate_matrix(dh; algebraic_couplings = AlgebraicCoupling(; algebraic_coupling = ((:u, :z),)))
end

@testset "augmented local dofs and assembly" begin
    grid = generate_grid(Quadrilateral, (3, 3))
    dh = DofHandler(grid)
    add!(dh, :u, Lagrange{RefQuadrilateral, 1}()^2)
    add!(dh, :p, Lagrange{RefQuadrilateral, 1}())
    add!(dh, :p0, AlgebraicVariable())
    add!(dh, :z, AlgebraicVariable{Vec{2}}())
    close!(dh)
    right = getfacetset(grid, "right")
    ccpl = CellCoupling(1:getncells(grid); algebraic_coupling = ((:p, :z), (:p, :p0), (:z, :z), (:z, :p0), (:p0, :p0)))
    fcpl = FacetCoupling(right; algebraic_coupling = ((:u, :p0),))
    acpl = AlgebraicCoupling(; algebraic_coupling = ((:p0, :p0), (:p0, :z), (:z, :z)))

    # The augmented local dof vector is plain user code: cell dofs first, then the
    # algebraic dofs in an order chosen by the assembly loop
    n = ndofs_per_cell(dh)
    zdofs = algebraic_dofs(dh, :z)
    p0dofs = algebraic_dofs(dh, :p0)
    nl = n + length(zdofs) + length(p0dofs)
    dofs = Vector{Int}(undef, nl)
    dofs[(n + 1):(n + 2)] .= zdofs
    dofs[n + 3] = only(p0dofs)
    range_u = dof_range(dh, :u)

    # Assembly equivalence with direct global insertion, and residual assembly
    couplings = (c = ccpl, f = fcpl, a = acpl)
    K = allocate_matrix(dh; algebraic_couplings = couplings)
    f = zeros(ndofs(dh))
    Kref = zeros(ndofs(dh), ndofs(dh))
    fref = zeros(ndofs(dh))
    assembler = start_assemble(K, f)
    for cc in CellIterator(dh)
        copyto!(dofs, celldofs(cc))
        Ke = reshape(float.(1:(nl * nl)), nl, nl) .+ cellid(cc)
        fe = float.(1:nl)
        # keep u-rows/cols zero: mask them out like a kernel would
        Ke[range_u, :] .= 0; Ke[:, range_u] .= 0; fe[range_u] .= 0
        assemble!(assembler, dofs, Ke, fe)
        Kref[dofs, dofs] .+= Ke
        fref[dofs] .+= fe
    end
    # An algebraic-only term uses the global algebraic dofs directly
    adofs = vcat(p0dofs, zdofs)
    Ke0 = [1.0 2.0 3.0; 4.0 5.0 6.0; 7.0 8.0 9.0]
    fe0 = [1.0, 2.0, 3.0]
    assemble!(assembler, adofs, Ke0, fe0)
    Kref[adofs, adofs] .+= Ke0
    fref[adofs] .+= fe0
    K, f = finish_assemble(assembler)
    @test Matrix(K) ≈ Kref
    @test f ≈ fref

    # Forgetting algebraic_couplings= gives the existing missing-sparsity-entry error
    Kmiss = allocate_matrix(dh)
    amiss = start_assemble(Kmiss)
    err = try
        assemble!(amiss, adofs, ones(3, 3))
        nothing
    catch e
        e
    end
    @test err isa ErrorException
    @test occursin("missing in the sparsity pattern", err.msg)
end

@testset "apply_assemble! and affine constraints" begin
    grid = generate_grid(Quadrilateral, (3, 3))
    dh = DofHandler(grid)
    add!(dh, :u, Lagrange{RefQuadrilateral, 1}())
    add!(dh, :p0, AlgebraicVariable())
    close!(dh)
    p0dof = only(algebraic_dofs(dh, :p0))
    cpl = CellCoupling(1:getncells(grid); algebraic_coupling = ((:u, :p0), (:p0, :p0)))
    ch = ConstraintHandler(dh)
    add!(ch, Dirichlet(:u, getfacetset(grid, "left"), (x, t) -> 0.0))
    # An AffineConstraint whose master is an algebraic dof
    free_u_dof = celldofs(dh, 3)[2]
    add!(ch, AffineConstraint(free_u_dof, [p0dof => 2.0], 1.0))
    # A regular constraint between two spatial dofs: guards that the elimination
    # incorporates the algebraic row/column of the involved dofs correctly
    u_slave = celldofs(dh, 9)[2]
    u_master = celldofs(dh, 9)[3]
    add!(ch, AffineConstraint(u_slave, [u_master => 0.5], 0.25))
    close!(ch)
    K = allocate_matrix(dh, ch; algebraic_couplings = (cpl,))
    f = zeros(ndofs(dh))
    assembler = start_assemble(K, f)
    nl = ndofs_per_cell(dh) + 1
    dofs = Vector{Int}(undef, nl)
    dofs[nl] = p0dof
    for cc in CellIterator(dh)
        copyto!(dofs, celldofs(cc))
        Ke = Matrix(2.0 * I, nl, nl) .+ 0.5
        fe = ones(nl)
        apply_assemble!(assembler, ch, dofs, Ke, fe)
    end
    K, f = finish_assemble(assembler)
    # The system is fully condensed by apply_assemble! -- no global apply!(K, f, ch)
    a = K \ f
    apply!(a, ch)
    @test a[free_u_dof] ≈ 2.0 * a[p0dof] + 1.0
    @test a[u_slave] ≈ 0.5 * a[u_master] + 0.25
    # Reference path: plain assembly with global elimination must give the same solution
    Kref = allocate_matrix(dh, ch; algebraic_couplings = (cpl,))
    fref = zeros(ndofs(dh))
    aref = start_assemble(Kref, fref)
    for cc in CellIterator(dh)
        copyto!(dofs, celldofs(cc))
        Ke = Matrix(2.0 * I, nl, nl) .+ 0.5
        fe = ones(nl)
        assemble!(aref, dofs, Ke, fe)
    end
    Kref, fref = finish_assemble(aref)
    apply!(Kref, fref, ch)
    bref = Kref \ fref
    apply!(bref, ch)
    @test bref ≈ a
end

@testset "threaded atomic assembly" begin
    grid = generate_grid(Quadrilateral, (8, 8))
    dh = DofHandler(grid)
    add!(dh, :u, Lagrange{RefQuadrilateral, 1}())
    add!(dh, :p0, AlgebraicVariable())
    close!(dh)
    cpl = CellCoupling(1:getncells(grid); algebraic_coupling = ((:u, :p0), (:p0, :p0)))
    ncells = getncells(grid)
    nl = ndofs_per_cell(dh) + 1
    function element!(Ke, fe, cellid)
        Ke .= 1.0 .+ cellid / ncells
        fe .= cellid
        return
    end
    function assemble_all!(K, f, atomic::Bool)
        chunks = collect(Iterators.partition(1:ncells, max(1, ncells ÷ 4)))
        tasks = map(chunks) do chunk
            Threads.@spawn begin
                # Each task owns its assembler, buffers, and iterator
                assembler = start_assemble(K, f; fillzero = false, atomic = atomic)
                dofs = Vector{Int}(undef, nl)
                dofs[nl] = only(algebraic_dofs(dh, :p0))
                for cc in CellIterator(dh, chunk)
                    copyto!(dofs, celldofs(cc))
                    Ke = zeros(nl, nl)
                    fe = zeros(nl)
                    element!(Ke, fe, cellid(cc))
                    assemble!(assembler, dofs, Ke, fe)
                end
            end
        end
        foreach(wait, tasks)
        return K, f
    end
    # CSC
    Kser = allocate_matrix(dh; algebraic_couplings = (cpl,)); fser = zeros(ndofs(dh))
    aser = start_assemble(Kser, fser)
    dofs = Vector{Int}(undef, nl)
    dofs[nl] = only(algebraic_dofs(dh, :p0))
    for cc in CellIterator(dh)
        copyto!(dofs, celldofs(cc))
        Ke = zeros(nl, nl); fe = zeros(nl)
        element!(Ke, fe, cellid(cc))
        assemble!(aser, dofs, Ke, fe)
    end
    finish_assemble(aser)
    Kthr = allocate_matrix(dh; algebraic_couplings = (cpl,)); fthr = zeros(ndofs(dh))
    assemble_all!(Kthr, fthr, true)
    @test Kthr ≈ Kser
    @test fthr ≈ fser
    # CSR (extension)
    Kcsr = allocate_matrix(SparseMatrixCSR, dh; algebraic_couplings = (cpl,)); fcsr = zeros(ndofs(dh))
    assemble_all!(Kcsr, fcsr, true)
    @test Matrix(Kcsr) ≈ Matrix(Kser)
    @test fcsr ≈ fser
    # Documented existing limitation: unsupported eltype
    Kint = sparse([1, 2], [1, 2], Rational{Int}[1, 1])
    @test_throws ArgumentError start_assemble(Kint; atomic = true)
    # BlockArrays assembler (extension)
    renumber!(dh, DofOrder.FieldWise())
    sp = BlockSparsityPattern([ndofs(dh) - 1, 1])
    cpl2 = CellCoupling(1:ncells; algebraic_coupling = ((:u, :p0), (:p0, :p0)))
    add_sparsity_entries!(sp, dh; algebraic_couplings = (cpl2,))
    # Serial reference in the renumbered ordering (the tail is re-queried after renumber!)
    Kser2 = allocate_matrix(dh; algebraic_couplings = (cpl2,)); fser2 = zeros(ndofs(dh))
    aser2 = start_assemble(Kser2, fser2)
    dofs[nl] = only(algebraic_dofs(dh, :p0))
    for cc in CellIterator(dh)
        copyto!(dofs, celldofs(cc))
        Ke = zeros(nl, nl); fe = zeros(nl)
        element!(Ke, fe, cellid(cc))
        assemble!(aser2, dofs, Ke, fe)
    end
    finish_assemble(aser2)
    Kblock = allocate_matrix(BlockMatrix, sp)
    fblock = BlockVector(zeros(ndofs(dh)), [ndofs(dh) - 1, 1])
    start_assemble(Kblock, fblock) # zero K and f once; the tasks use fillzero = false
    assemble_all!(Kblock, fblock, true)
    @test Matrix(Kblock) ≈ Matrix(Kser2)
    @test Vector(fblock) ≈ fser2
end

@testset "spatial-only API guard rails" begin
    grid = generate_grid(Quadrilateral, (3, 3))
    dh = DofHandler(grid)
    add!(dh, :u, Lagrange{RefQuadrilateral, 1}()^2)
    add!(dh, :p, Lagrange{RefQuadrilateral, 1}())
    add!(dh, :p0, AlgebraicVariable())
    close!(dh)
    a = zeros(ndofs(dh))
    ch = ConstraintHandler(dh)
    left = getfacetset(grid, "left")
    @test_throws ErrorException add!(ch, Dirichlet(:p0, left, (x, t) -> 0.0))
    @test_throws ErrorException add!(ch, ProjectedDirichlet(:p0, left, (x, t) -> 0.0))
    @test_throws ErrorException add!(ch, PeriodicDirichlet(:p0, collect_periodic_facets(grid, "left", "right")))
    @test_throws ErrorException apply_analytical!(a, dh, :p0, x -> 0.0)
    @test_throws ErrorException evaluate_at_grid_nodes(dh, a, :p0)
    ph = PointEvalHandler(grid, [Vec((0.1, 0.1))])
    @test_throws ErrorException evaluate_at_points(ph, dh, a, :p0)
    @test_throws ErrorException Ferrite.dof_range(dh, :p0)
    # Everything keeps working for spatial names in the mixed handler
    add!(ch, Dirichlet(:u, left, (x, t) -> zero(Vec{2})))
    close!(ch); update!(ch, 0.0)
    apply_analytical!(a, dh, :p, x -> 1.0)
    @test all(evaluate_at_grid_nodes(dh, a, :p) .≈ 1.0)
    @test evaluate_at_points(ph, dh, a, :p) ≈ [1.0]
    # Whole-solution VTK output writes the spatial fields and omits algebraic variables
    mktempdir() do dir
        VTKGridFile(joinpath(dir, "algvar"), grid) do vtk
            @test write_solution(vtk, dh, a) === vtk
        end
    end
end

# The stress-driven homogenization formulation (#1396/#1422): the macroscopic stress σ̄ is
# an unknown symmetric tensor acting as the Lagrange multiplier that enforces the
# prescribed average strain ε̄ on the RVE. For a homogeneous material the exact solution is
# u = ε̄ ⋅ x and σ̄ = C : ε̄.
@testset "stress-driven homogenization example" begin
    grid = generate_grid(Quadrilateral, (4, 4))
    σ̄var = AlgebraicVariable{SymmetricTensor{2, 2}}()
    av = AlgebraicValues(σ̄var)
    nσ = getnbasefunctions(av)
    dh = DofHandler(grid)
    ip = Lagrange{RefQuadrilateral, 1}()^2
    add!(dh, :u, ip)
    add!(dh, :σ̄, σ̄var)
    close!(dh)

    # Isotropic plane-strain elasticity
    E, ν = 200.0e3, 0.3
    G = E / (2 * (1 + ν))
    Λ = E * ν / ((1 + ν) * (1 - 2ν))
    δ(i, j) = i == j ? 1.0 : 0.0
    C = SymmetricTensor{4, 2}((i, j, k, l) -> Λ * δ(i, j) * δ(k, l) + G * (δ(i, k) * δ(j, l) + δ(i, l) * δ(j, k)))
    ε̄ = SymmetricTensor{2, 2}((1.0e-3, 0.5e-3, -0.7e-3))

    # Remove rigid body modes with the exact solution values (they are consistent with the
    # average-strain constraint)
    ch = ConstraintHandler(dh)
    addvertexset!(grid, "origin", x -> x ≈ Vec((-1.0, -1.0)))
    addvertexset!(grid, "xaxis", x -> x ≈ Vec((1.0, -1.0)))
    add!(ch, Dirichlet(:u, getvertexset(grid, "origin"), x -> ε̄ ⋅ x))
    add!(ch, Dirichlet(:u, getvertexset(grid, "xaxis"), x -> (ε̄ ⋅ x)[2:2], [2]))
    close!(ch); update!(ch, 0.0)

    descriptor = CellCoupling(1:getncells(grid); algebraic_coupling = ((:u, :σ̄),))
    K = allocate_matrix(dh, ch; algebraic_couplings = (rve = descriptor,))
    f = zeros(ndofs(dh))

    qr = QuadratureRule{RefQuadrilateral}(2)
    cv = CellValues(qr, ip)
    nbase = getnbasefunctions(cv)
    nlocal = nbase + nσ
    Ke = zeros(nlocal, nlocal) # hoisted out of the assembly loop
    fe = zeros(nlocal)
    function element_residual!(re, ae, cv, av, range_u, range_σ, C, ε̄)
        fill!(re, 0)
        σ̄e = algebraic_value(av, ae, range_σ)
        for qp in 1:getnquadpoints(cv)
            dΩ = getdetJdV(cv, qp)
            εe = function_symmetric_gradient(cv, qp, ae, range_u)
            for (i, I) in pairs(range_u)
                εi = shape_symmetric_gradient(cv, qp, i)
                re[I] += (εi ⊡ C ⊡ εe - εi ⊡ σ̄e) * dΩ
            end
            for (k, S) in pairs(range_σ)
                Ek = algebraic_basis_value(av, k)
                re[S] += (Ek ⊡ ε̄ - Ek ⊡ εe) * dΩ
            end
        end
        return re
    end
    assembler = start_assemble(K, f)
    dofs = Vector{Int}(undef, nlocal)
    dofs[(nbase + 1):end] .= algebraic_dofs(dh, :σ̄)
    range_u = dof_range(dh, :u)
    range_σ = (nbase + 1):nlocal
    for cc in CellIterator(dh)
        copyto!(dofs, celldofs(cc))
        fill!(Ke, 0); fill!(fe, 0)
        reinit!(cv, cc)
        for qp in 1:getnquadpoints(cv)
            dΩ = getdetJdV(cv, qp)
            for (i, I) in pairs(range_u)
                εi = shape_symmetric_gradient(cv, qp, i)
                for (j, J) in pairs(range_u)
                    εj = shape_symmetric_gradient(cv, qp, j)
                    Ke[I, J] += (εi ⊡ C ⊡ εj) * dΩ
                end
                for (k, S) in pairs(range_σ)
                    Ek = algebraic_basis_value(av, k)
                    # test u / trial σ̄ and (by symmetry) test σ̄ / trial u
                    Ke[I, S] -= (εi ⊡ Ek) * dΩ
                    Ke[S, I] -= (Ek ⊡ εi) * dΩ
                end
            end
            for (k, S) in pairs(range_σ)
                fe[S] -= (algebraic_basis_value(av, k) ⊡ ε̄) * dΩ
            end
        end
        if cellid(cc) == 1
            ae = rand(nlocal)
            Ke_ad = ForwardDiff.jacobian(
                (r, x) -> element_residual!(r, x, cv, av, range_u, range_σ, C, ε̄),
                zeros(nlocal), ae
            )
            @test Ke_ad ≈ Ke
        end
        apply_assemble!(assembler, ch, dofs, Ke, fe)
    end
    K, f = finish_assemble(assembler)
    a = K \ f
    apply!(a, ch)

    σ̄ = algebraic_value(dh, a, :σ̄)
    @test σ̄ ≈ C ⊡ ε̄  rtol = 1.0e-10
    # The displacement field is the homogeneous solution u = ε̄ ⋅ x
    u_nodes = evaluate_at_grid_nodes(dh, a, :u)
    for (n, node) in enumerate(getnodes(grid))
        x = get_node_coordinate(node)
        @test u_nodes[n] ≈ ε̄ ⋅ x  atol = 1.0e-12 * norm(ε̄)
    end
end

# Minimal boundary use case: Poisson problem with pure Neumann boundary conditions, made
# solvable by the boundary mean-value constraint ∫_Γ u dΓ = 0 enforced with a scalar
# algebraic multiplier λ coupled to u only on the boundary Γ.
@testset "boundary mean-value constraint example" begin
    grid = generate_grid(Triangle, (8, 8))
    dh = DofHandler(grid)
    ip = Lagrange{RefTriangle, 1}()
    add!(dh, :u, ip)
    add!(dh, :λ, AlgebraicVariable())
    close!(dh)
    λdof = only(algebraic_dofs(dh, :λ))
    boundary = union((getfacetset(grid, name) for name in ("left", "right", "top", "bottom"))...)
    descriptor = FacetCoupling(boundary; algebraic_coupling = ((:u, :λ),))
    K = allocate_matrix(dh; algebraic_couplings = (mean_u = descriptor,))
    f = zeros(ndofs(dh))

    qr = QuadratureRule{RefTriangle}(2)
    cv = CellValues(qr, ip)
    fqr = FacetQuadratureRule{RefTriangle}(2)
    fv = FacetValues(fqr, ip)
    assembler = start_assemble(K, f)
    # Bulk: ∫ ∇δu ⋅ ∇u dΩ = ∫ δu f dΩ with f = x₁ (zero mean not required thanks to λ)
    nbase = getnbasefunctions(cv)
    Ke = zeros(nbase, nbase)
    fe = zeros(nbase)
    for cc in CellIterator(dh)
        fill!(Ke, 0); fill!(fe, 0)
        reinit!(cv, cc)
        for qp in 1:getnquadpoints(cv)
            dΩ = getdetJdV(cv, qp)
            x = spatial_coordinate(cv, qp, getcoordinates(cc))
            for i in 1:nbase
                fe[i] += shape_value(cv, qp, i) * x[1] * dΩ
                for j in 1:nbase
                    Ke[i, j] += (shape_gradient(cv, qp, i) ⋅ shape_gradient(cv, qp, j)) * dΩ
                end
            end
        end
        assemble!(assembler, celldofs(cc), Ke, fe)
    end
    # Boundary term: λ couples to the trace of u
    nlocal = nbase + 1
    Keb = zeros(nlocal, nlocal)
    feb = zeros(nlocal)
    bdofs = Vector{Int}(undef, nlocal)
    bdofs[nlocal] = λdof
    range_u = dof_range(dh, :u)
    λlocal = nlocal # local index of the multiplier
    for fc in FacetIterator(dh, boundary)
        copyto!(bdofs, celldofs(fc))
        fill!(Keb, 0); fill!(feb, 0)
        reinit!(fv, fc)
        for qp in 1:getnquadpoints(fv)
            dΓ = getdetJdV(fv, qp)
            for (i, I) in pairs(range_u)
                v = shape_value(fv, qp, i)
                Keb[I, λlocal] += v * dΓ
                Keb[λlocal, I] += v * dΓ
            end
        end
        assemble!(assembler, bdofs, Keb, feb)
    end
    K, f = finish_assemble(assembler)
    a = K \ f
    # The mean-value constraint is satisfied
    mean_u = 0.0
    for fc in FacetIterator(dh, boundary)
        reinit!(fv, fc)
        dofs = celldofs(fc)
        for qp in 1:getnquadpoints(fv)
            u_qp = function_value(fv, qp, a[dofs])
            mean_u += u_qp * getdetJdV(fv, qp)
        end
    end
    @test abs(mean_u) < 1.0e-10
    @test abs(a[λdof]) < 1.0e-8 # ∫ f dΩ = 0 for f = x₁, so the multiplier vanishes
end

# Small coupled FE--0D system with a dense algebraic K_pp block
@testset "algebraic K_pp block example" begin
    grid = generate_grid(Line, (4,))
    dh = DofHandler(grid)
    add!(dh, :u, Lagrange{RefLine, 1}())
    add!(dh, :p, AlgebraicVariable{Vec{2}}())
    close!(dh)
    pdofs = algebraic_dofs(dh, :p)
    zd = AlgebraicCoupling(; algebraic_coupling = ((:p, :p),))
    K = allocate_matrix(dh; algebraic_couplings = (zd,))
    f = zeros(ndofs(dh))
    assembler = start_assemble(K, f)
    for cc in CellIterator(dh)
        n = ndofs_per_cell(dh)
        assemble!(assembler, celldofs(cc), Matrix(1.0 * I, n, n), zeros(n))
    end
    Kpp = [2.0 1.0; 1.0 3.0]
    fp = [1.0, 2.0]
    assemble!(assembler, pdofs, Kpp, fp)
    K, f = finish_assemble(assembler)
    a = K \ f
    @test a[pdofs] ≈ Kpp \ fp
end
