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
    # Full and active component counts
    @test active_components(AlgebraicVariable()) == (1,)
    @test active_components(AlgebraicVariable{Vec{3}}()) == (1, 2, 3)
    @test active_components(AlgebraicVariable{Tensor{2, 2}}()) == ((1, 1), (2, 1), (1, 2), (2, 2))
    @test active_components(AlgebraicVariable{SymmetricTensor{2, 2}}()) == ((1, 1), (2, 1), (2, 2))
    @test active_components(AlgebraicVariable{SymmetricTensor{2, 3}}()) ==
        ((1, 1), (2, 1), (3, 1), (2, 2), (3, 2), (3, 3))
    # Partial selections are canonicalized independent of input order
    @test active_components(AlgebraicVariable{Vec{3}}(active_components = (3, 1))) == (1, 3)
    @test active_components(AlgebraicVariable{SymmetricTensor{2, 2}}(active_components = ((2, 2), (1, 1)))) ==
        ((1, 1), (2, 2))
    # Symmetry-equivalent indices are normalized to the same component
    @test active_components(AlgebraicVariable{SymmetricTensor{2, 2}}(active_components = ((1, 2),))) == ((2, 1),)
    @test active_components(AlgebraicVariable(active_components = (1,))) == (1,)
    # Invalid selections
    @test_throws ErrorException AlgebraicVariable{Vec{3}}(active_components = ()) # empty
    @test_throws ErrorException AlgebraicVariable{Vec{3}}(active_components = (0,)) # out of bounds
    @test_throws ErrorException AlgebraicVariable{Vec{3}}(active_components = (4,)) # out of bounds
    @test_throws ErrorException AlgebraicVariable{Vec{3}}(active_components = (1, 1)) # duplicate
    @test_throws ErrorException AlgebraicVariable{Vec{3}}(active_components = ((1, 1),)) # wrong rank
    @test_throws ErrorException AlgebraicVariable{Tensor{2, 2}}(active_components = (1,)) # wrong rank
    @test_throws ErrorException AlgebraicVariable{Tensor{2, 2}}(active_components = ((3, 1),)) # out of bounds
    @test_throws ErrorException AlgebraicVariable{SymmetricTensor{2, 2}}(active_components = ((1, 2), (2, 1))) # symmetry-equivalent duplicate
    @test_throws ErrorException AlgebraicVariable(active_components = (2,))
    # Unsupported shapes
    @test_throws ErrorException AlgebraicVariable{Float64}()
    @test_throws ErrorException AlgebraicVariable{Vec{3, Float64}}() # coefficient type must not be fixed
    @test_throws ErrorException AlgebraicVariable{SymmetricTensor{2, 2, Float64}}()
    # Unsupported dimensions (Tensors.jl supports 1:3; dim 0 would own no dofs)
    @test_throws ErrorException AlgebraicVariable{Vec{0}}()
    @test_throws ErrorException AlgebraicVariable{Vec{4}}()
    @test_throws ErrorException AlgebraicVariable{Tensor{2, 4}}()
    # Fourth order tensors
    @test length(active_components(AlgebraicVariable{Tensor{4, 2}}())) == 16
    @test length(active_components(AlgebraicVariable{SymmetricTensor{4, 2}}())) == 9
    @test length(active_components(AlgebraicVariable{SymmetricTensor{4, 3}}())) == 36
    @test active_components(AlgebraicVariable{SymmetricTensor{4, 2}}(active_components = ((1, 2, 2, 1),))) ==
        ((2, 1, 2, 1),) # both index pairs normalized by minor symmetry
    @test_throws ErrorException AlgebraicVariable{Tensor{4, 2}}(active_components = ((1, 2),)) # wrong rank
    @test_throws ErrorException AlgebraicVariable{SymmetricTensor{4, 2}}(active_components = ((1, 2, 1, 1), (2, 1, 1, 1))) # symmetry-equivalent duplicate
end

@testset "fourth order value reconstruction and basis" begin
    grid = generate_grid(Triangle, (1, 1))
    dh = DofHandler(grid)
    add!(dh, :u, Lagrange{RefTriangle, 1}())
    add!(dh, :A4, AlgebraicVariable{Tensor{4, 2}}())
    add!(dh, :C, AlgebraicVariable{SymmetricTensor{4, 2}}())
    close!(dh)
    for name in (:A4, :C)
        var = algebraic_variable(dh, name)
        comps = active_components(var)
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
    # No SubDofHandlers: no cell dofs, with and without an explicit cell index
    @test ndofs_per_cell(dh) == 0
    @test ndofs_per_cell(dh, 1) == 0
    # Matrix allocation without and with a descriptor
    K0 = allocate_matrix(dh)
    @test size(K0) == (3, 3)
    for d in 1:3
        @test hasentry(K0, d, d)
    end
    cpl = AlgebraicCoupling(dh; fields = (:z, :p0), algebraic_coupling = trues(2, 2))
    K = allocate_matrix(dh; algebraic_couplings = (cpl,))
    f = zeros(3)
    assembler = start_assemble(K, f)
    layout = local_dofs(cpl)
    Ke = [2.0 0.0 1.0; 0.0 2.0 0.0; 1.0 0.0 2.0]
    assemble!(assembler, layout, Ke, ones(3))
    K, f = finish_assemble(assembler)
    @test K[collect(layout), collect(layout)] ≈ Ke
end

@testset "descriptor immutability and algebraic_couplings forms" begin
    grid = generate_grid(Quadrilateral, (3, 3))
    dh = DofHandler(grid)
    add!(dh, :u, Lagrange{RefQuadrilateral, 1}())
    add!(dh, :p0, AlgebraicVariable())
    close!(dh)
    # getalgebraicvariablenames returns a copy
    names = getalgebraicvariablenames(dh)
    push!(names, :junk)
    @test getalgebraicvariablenames(dh) == [:p0]
    @test Ferrite.has_algebraic_variables(dh)
    # The descriptor owns its entity set: mutating the input set afterwards has no effect
    myset = OrderedCollections.OrderedSet{FacetIndex}(getfacetset(grid, "right"))
    cpl = FacetCoupling(dh, myset; fields = (:u, :p0), algebraic_coupling = trues(2, 2))
    n_before = length(Ferrite.entities(cpl))
    push!(myset, first(getfacetset(grid, "left")))
    @test length(Ferrite.entities(cpl)) == n_before
    # Accessors return copies
    ents = Ferrite.entities(cpl)
    push!(ents, first(getfacetset(grid, "left")))
    @test length(Ferrite.entities(cpl)) == n_before
    @test Ferrite.fields(cpl) == (:u, :p0)
    cm = Ferrite.coupling_matrix(cpl)
    cm[1, 1] = false
    @test Ferrite.coupling_matrix(cpl)[1, 1]
    # Empty (supported) collections keep the fast allocation path available
    @test Ferrite._no_algebraic_couplings(())
    @test Ferrite._no_algebraic_couplings(NamedTuple())
    @test Ferrite._no_algebraic_couplings(Ferrite.AbstractCoupling[])
    @test !Ferrite._no_algebraic_couplings((cpl,))
    K_empty = allocate_matrix(dh; algebraic_couplings = Ferrite.AbstractCoupling[])
    @test size(K_empty) == (ndofs(dh), ndofs(dh))
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
    # A symmetric tuple entry is equivalent to the field-level Bool matrix form
    K_pair = allocate_matrix(dh; algebraic_couplings = CellCoupling(dh, cells; algebraic_coupling = ((:p, :λ),)))
    K_mat = allocate_matrix(dh; algebraic_couplings = CellCoupling(dh, cells; fields = (:p, :λ), algebraic_coupling = Bool[0 1; 1 0]))
    @test pattern(K_pair) == pattern(K_mat)
    # `=>` couples one way only (test => trial)
    K_dir = allocate_matrix(dh; algebraic_couplings = CellCoupling(dh, cells; algebraic_coupling = (:p => :λ,)))
    pdof = first(celldofs(dh, 1)[dof_range(dh, :p)])
    @test hasentry(K_dir, pdof, λdof) && !hasentry(K_dir, λdof, pdof)
    # A bare entry is accepted as a one-entry specification
    K_bare_pair = allocate_matrix(dh; algebraic_couplings = CellCoupling(dh, cells; algebraic_coupling = :p => :λ))
    @test pattern(K_bare_pair) == pattern(K_dir)
    K_bare_tuple = allocate_matrix(dh; algebraic_couplings = CellCoupling(dh, cells; algebraic_coupling = (:p, :λ)))
    @test pattern(K_bare_tuple) == pattern(K_pair)
    # Participating variables are derived from the entries, in order of first appearance
    cpl = CellCoupling(dh, cells; algebraic_coupling = (:λ => :p, (:z, :p), :z => :z))
    @test Ferrite.fields(cpl) == (:λ, :p, :z)
    @test Ferrite.coupling_matrix(cpl) == Bool[0 1 0; 0 0 1; 0 1 1]
    # Symmetric self entry of a multi-dof variable gives its dense block
    K_self = allocate_matrix(dh; algebraic_couplings = AlgebraicCoupling(dh; algebraic_coupling = ((:z, :z),)))
    for i in zdofs, j in zdofs
        @test hasentry(K_self, i, j)
    end
    # The same directed coupling from a pair and a tuple collides
    @test_throws ErrorException CellCoupling(dh, cells; algebraic_coupling = (:p => :λ, (:p, :λ)))
    @test_throws ErrorException CellCoupling(dh, cells; algebraic_coupling = (:p => :λ, :p => :λ))
    # Every entry must involve an algebraic variable
    @test_throws ErrorException CellCoupling(dh, cells; algebraic_coupling = (:u => :p,))
    @test_throws ErrorException FacetCoupling(dh, getfacetset(grid, "right"); algebraic_coupling = ((:u, :p), (:u, :λ)))
    # `fields` only goes together with the matrix form, and the matrix form requires it
    @test_throws ErrorException CellCoupling(dh, cells; fields = (:p, :λ), algebraic_coupling = (:p => :λ,))
    @test_throws ErrorException CellCoupling(dh, cells; algebraic_coupling = Bool[0 1; 1 0])
    # Uninterpretable specifications error descriptively
    @test_throws ErrorException CellCoupling(dh, cells; algebraic_coupling = ())
    @test_throws ErrorException CellCoupling(dh, cells; algebraic_coupling = ((:p, :λ, :z),))
    @test_throws ErrorException CellCoupling(dh, cells; algebraic_coupling = (:p,))
    @test_throws ErrorException CellCoupling(dh, cells; algebraic_coupling = :p)
    @test_throws ErrorException CellCoupling(dh, cells; algebraic_coupling = (:p => :nope,))
end

@testset "DofHandler integration" begin
    grid = generate_grid(Triangle, (2, 2))
    dh = DofHandler(grid)
    add!(dh, :u, Lagrange{RefTriangle, 1}()^2)
    # add! returns the handler
    @test add!(dh, :p0, AlgebraicVariable()) === dh
    @test add!(dh, :z, AlgebraicVariable{Vec{3}}()) === dh
    add!(dh, :σ̄, AlgebraicVariable{SymmetricTensor{2, 2}}(active_components = ((1, 1), (2, 2))))
    # Name conflicts in both declaration orders
    @test_throws ErrorException add!(dh, :u, AlgebraicVariable()) # spatial exists
    @test_throws ErrorException add!(dh, :p0, AlgebraicVariable()) # algebraic exists
    @test_throws ErrorException add!(dh, :p0, Lagrange{RefTriangle, 1}()) # algebraic exists
    close!(dh)
    # Adding after close!
    @test_throws ErrorException add!(dh, :q, AlgebraicVariable())
    # Deterministic initial numbering: spatial dofs first, then algebraic in declaration
    # order and active-component order
    nspatial = 2 * getnnodes(grid)
    @test ndofs(dh) == nspatial + 1 + 3 + 2
    @test algebraic_dofs(dh, :p0) == [nspatial + 1]
    @test algebraic_dofs(dh, :z) == collect(nspatial .+ (2:4))
    @test algebraic_dofs(dh, :σ̄) == collect(nspatial .+ (5:6))
    # celldofs remain purely spatial
    @test maximum(maximum, (celldofs(dh, i) for i in 1:getncells(grid))) == nspatial
    @test ndofs_per_cell(dh) == 6
    # Introspection
    @test getalgebraicvariablenames(dh) == [:p0, :z, :σ̄]
    @test Ferrite.getfieldnames(dh) == [:u]
    @test algebraic_variable(dh, :z) isa AlgebraicVariable{Vec{3}}
    # algebraic_dofs returns a copy
    d1 = algebraic_dofs(dh, :p0)
    d1[1] = -1
    @test algebraic_dofs(dh, :p0) == [nspatial + 1]
    # Unknown-name and wrong-kind errors
    @test_throws ErrorException algebraic_variable(dh, :nope)
    @test_throws ErrorException algebraic_variable(dh, :u)
    @test_throws ErrorException algebraic_dofs(dh, :u)
    @test_throws ErrorException Ferrite.find_field(dh, :p0)
    @test_throws ErrorException dof_range(dh, :p0)
end

@testset "algebraic_value and AlgebraicValues" begin
    grid = generate_grid(Triangle, (2, 2))
    dh = DofHandler(grid)
    add!(dh, :u, Lagrange{RefTriangle, 1}())
    add!(dh, :p0, AlgebraicVariable())
    add!(dh, :z, AlgebraicVariable{Vec{3}}(active_components = (1, 3)))
    add!(dh, :A, AlgebraicVariable{Tensor{2, 2}}())
    add!(dh, :σ̄, AlgebraicVariable{SymmetricTensor{2, 2}}(active_components = ((1, 1), (2, 2))))
    close!(dh)
    for T in (Float64, Float32)
        a = zeros(T, ndofs(dh))
        a[algebraic_dofs(dh, :p0)] .= 3
        a[algebraic_dofs(dh, :z)] .= [1, 2]
        a[algebraic_dofs(dh, :A)] .= [1, 2, 3, 4]
        a[algebraic_dofs(dh, :σ̄)] .= [5, 6]
        p0 = algebraic_value(dh, a, :p0)
        @test p0 === T(3)
        z = algebraic_value(dh, a, :z)
        @test z isa Vec{3, T}
        @test z == Vec{3, T}((1, 0, 2)) # typed zero in inactive component
        A = algebraic_value(dh, a, :A)
        @test A isa Tensor{2, 2, T}
        @test A == Tensor{2, 2, T}((1, 2, 3, 4)) # column major, matching Tensors storage
        σ̄ = algebraic_value(dh, a, :σ̄)
        @test σ̄ isa SymmetricTensor{2, 2, T}
        @test σ̄ == SymmetricTensor{2, 2, T}((5, 0, 6))
        # Explicit addition of prescribed inactive components is user code
        σ_prescribed = SymmetricTensor{2, 2, T}((0, 7, 0))
        @test (σ_prescribed + σ̄)[2, 1] == 7
    end
    # Wrong-kind and length errors
    a = zeros(ndofs(dh))
    @test_throws ErrorException algebraic_value(dh, a, :u)
    @test_throws ErrorException algebraic_value(dh, a, :nope)
    @test_throws ErrorException algebraic_value(dh, zeros(3), :p0)
    # DofHandler-free reconstruction through an AlgebraicValues, from an extracted
    # coefficient slice in active-component order, matching the global-vector method
    for T in (Float64, Float32)
        a = zeros(T, ndofs(dh))
        a[algebraic_dofs(dh, :z)] .= [1, 2]
        a[algebraic_dofs(dh, :σ̄)] .= [5, 6]
        for name in (:p0, :z, :A, :σ̄)
            av = AlgebraicValues(algebraic_variable(dh, name))
            gdofs = algebraic_dofs(dh, name)
            @test algebraic_value(av, a[gdofs]) === algebraic_value(dh, a, name)
            @test algebraic_value(av, view(a, gdofs)) === algebraic_value(dh, a, name)
            # global vector together with the global dof numbers
            @test algebraic_value(av, a, gdofs) === algebraic_value(dh, a, name)
        end
    end
    @test algebraic_value(AlgebraicValues(AlgebraicVariable()), [4.0]) === 4.0
    # The scalar type follows the input, so dual numbers pass through (AD inside a kernel)
    let av = AlgebraicValues(AlgebraicVariable{SymmetricTensor{2, 2}}(active_components = ((1, 1), (2, 2))))
        d1 = ForwardDiff.Dual(5.0, 1.0, 0.0)
        d2 = ForwardDiff.Dual(6.0, 0.0, 1.0)
        σ̄d = algebraic_value(av, [d1, d2])
        @test σ̄d isa SymmetricTensor{2, 2, typeof(d1)}
        @test ForwardDiff.value(σ̄d[1, 1]) == 5.0
        @test ForwardDiff.value(σ̄d[2, 2]) == 6.0
        @test ForwardDiff.value(σ̄d[2, 1]) == 0.0
        # ∂σ̄/∂coefficient recovers the basis directions
        for k in 1:2
            E = algebraic_basis_value(av, k)
            for i in 1:2, j in 1:2
                @test ForwardDiff.partials(σ̄d[i, j], k) == E[i, j]
            end
        end
    end
    # Length mismatch (e.g. passing the full local vector instead of the variable's slice)
    @test_throws ErrorException algebraic_value(AlgebraicValues(algebraic_variable(dh, :z)), zeros(3))
    # Basis directions map active dof index -> declared component
    zvar = algebraic_variable(dh, :z)
    zv = AlgebraicValues(zvar)
    @test algebraic_basis_value(zv, 1) == Vec{3}((1.0, 0.0, 0.0))
    @test algebraic_basis_value(zv, 2) == Vec{3}((0.0, 0.0, 1.0))
    @test algebraic_basis_value(AlgebraicValues(Float32, zvar), 2) isa Vec{3, Float32}
    @test_throws ErrorException algebraic_basis_value(zv, 3)
    @test_throws ErrorException algebraic_basis_value(zv, 0)
    @test algebraic_basis_value(AlgebraicValues(algebraic_variable(dh, :p0)), 1) === 1.0
    σ̄full = AlgebraicValues(AlgebraicVariable{SymmetricTensor{2, 2}}())
    E2 = algebraic_basis_value(σ̄full, 2)
    @test E2 == SymmetricTensor{2, 2}((0.0, 1.0, 0.0)) # ones in (2,1) and (1,2)
    @test E2[1, 2] == E2[2, 1] == 1.0
    @test algebraic_basis_value(AlgebraicValues(algebraic_variable(dh, :A)), 3) == Tensor{2, 2}((0.0, 0.0, 1.0, 0.0))
    # The basis is the derivative of the reconstructed value w.r.t. the coefficient
    ac = algebraic_dofs(dh, :σ̄)
    σ̄v = AlgebraicValues(algebraic_variable(dh, :σ̄))
    for i in 1:2
        ei = zeros(ndofs(dh)); ei[ac[i]] = 1.0
        @test algebraic_value(dh, ei, :σ̄) == algebraic_basis_value(σ̄v, i)
    end
    # AlgebraicValues queries are type stable (in contrast to the registry lookup)
    let var = AlgebraicVariable{SymmetricTensor{2, 2}}()
        av = @inferred AlgebraicValues(var)
        @test ndofs(av) == 3
        @test active_components(av) == active_components(var)
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
    add!(dh, :σ̄, AlgebraicVariable{SymmetricTensor{2, 2}}(active_components = ((1, 1), (2, 2))))
    close!(dh)
    n = ndofs(dh)
    # Arbitrary permutation and inverse round trip; value reconstruction follows
    perm = randperm(n)
    renumber!(dh, perm)
    a = zeros(n)
    a[algebraic_dofs(dh, :σ̄)] .= [1.0, 2.0]
    @test algebraic_value(dh, a, :σ̄) == SymmetricTensor{2, 2}((1.0, 0.0, 2.0))
    renumber!(dh, invperm(perm))
    @test algebraic_dofs(dh, :p0) == [n - 2]
    @test algebraic_dofs(dh, :σ̄) == [n - 1, n]
    # Field-wise: every algebraic variable is one block
    renumber!(dh, DofOrder.FieldWise())
    @test algebraic_dofs(dh, :p0) == [n - 2]
    @test algebraic_dofs(dh, :σ̄) == [n - 1, n]
    # Custom block targets over the combined variable ordering (u, p, p0, σ̄)
    renumber!(dh, DofOrder.FieldWise([2, 2, 1, 1]))
    @test algebraic_dofs(dh, :p0) == [1]
    @test algebraic_dofs(dh, :σ̄) == [2, 3]
    @test_throws ErrorException renumber!(dh, DofOrder.FieldWise([1, 2, 1])) # wrong length
    # Component-wise: components (ux, uy, p, p0, σ̄₁₁, σ̄₂₂)
    renumber!(dh, DofOrder.ComponentWise())
    @test algebraic_dofs(dh, :σ̄) == [n - 1, n]
    renumber!(dh, DofOrder.ComponentWise([6, 5, 4, 3, 2, 1]))
    @test algebraic_dofs(dh, :p0) == [3]
    @test algebraic_dofs(dh, :σ̄) == [2, 1]
    @test_throws ErrorException renumber!(dh, DofOrder.ComponentWise(collect(1:5))) # wrong length
    # Public variable-name discovery for building custom targets
    @test length(Ferrite.getfieldnames(dh)) + length(getalgebraicvariablenames(dh)) == 4
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
    cpl = CellCoupling(dh, subset; fields = (:p, :λ), algebraic_coupling = Bool[0 1; 0 0])
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
    # Component-level coupling counts active algebraic components
    dh2 = DofHandler(grid)
    add!(dh2, :u, Lagrange{RefQuadrilateral, 1}()^2)
    add!(dh2, :z, AlgebraicVariable{Vec{3}}(active_components = (1, 3)))
    close!(dh2)
    zdofs = algebraic_dofs(dh2, :z)
    # (ux, uy, z1, z3): couple only uy <-> z3
    cpl2 = CellCoupling(dh2, 1:getncells(grid); fields = (:u, :z), algebraic_coupling = Bool[0 0 0 0; 0 0 0 1; 0 0 0 0; 0 1 0 0])
    K2 = allocate_matrix(dh2; algebraic_couplings = (cpl2,))
    uy = celldofs(dh2, 1)[dof_range(dh2, :u)][2:2:end]
    ux = celldofs(dh2, 1)[dof_range(dh2, :u)][1:2:end]
    for d in uy
        @test hasentry(K2, d, zdofs[2]) && hasentry(K2, zdofs[2], d)
        @test !hasentry(K2, d, zdofs[1]) && !hasentry(K2, zdofs[1], d)
    end
    for d in ux
        @test !hasentry(K2, d, zdofs[2])
    end
    # A matrix sized like neither variant errors
    @test_throws ErrorException CellCoupling(dh2, 1:2; fields = (:u, :z), algebraic_coupling = trues(3, 3))

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
    cpl3 = CellCoupling(dh3, 1:getncells(grid); fields = (:u, :λ), algebraic_coupling = Bool[0 1; 1 0])
    K3 = allocate_matrix(dh3; algebraic_couplings = (cpl3,))
    for cell in (2, 7), d in celldofs(dh3, cell)[dof_range(dh3.subdofhandlers[dh3.cell_to_subdofhandler[cell]], :u)]
        @test hasentry(K3, d, λ3)
    end
    # Clear failure when a selected cell lacks a named field
    @test_throws ErrorException CellCoupling(dh3, 1:getncells(grid); fields = (:p, :λ), algebraic_coupling = Bool[0 1; 1 0])
    # But restricting to cells that carry the field works
    CellCoupling(dh3, 1:4; fields = (:p, :λ), algebraic_coupling = Bool[0 1; 1 0])
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
    cpl = FacetCoupling(dh, right; fields = (:u, :p0), algebraic_coupling = Bool[0 1; 1 0])
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
    cpl_asym = FacetCoupling(dh, right; fields = (:u, :p0), algebraic_coupling = Bool[0 1; 0 0])
    Ka = allocate_matrix(dh; algebraic_couplings = (cpl_asym,))
    d = first(celldofs(dh, first(first(right))))
    @test hasentry(Ka, d, p0dof) && !hasentry(Ka, p0dof, d)
    # Multiple boundaries with separate descriptors; overlapping descriptors -> union
    cpl_left = FacetCoupling(dh, left; fields = (:u, :p0), algebraic_coupling = Bool[0 1; 1 0])
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
    K1 = allocate_matrix(dh; algebraic_couplings = AlgebraicCoupling(dh; fields = (:z,), algebraic_coupling = trues(1, 1)))
    for i in zdofs, j in zdofs
        @test hasentry(K1, i, j)
    end
    @test !hasentry(K1, zdofs[1], wdofs[1])
    # Asymmetric cross-coupling between two variables
    K2 = allocate_matrix(dh; algebraic_couplings = AlgebraicCoupling(dh; fields = (:z, :w), algebraic_coupling = Bool[0 1; 0 0]))
    for i in zdofs, j in wdofs
        @test hasentry(K2, i, j)
        @test !hasentry(K2, j, i)
    end
    # AlgebraicCoupling rejects spatial fields
    @test_throws ErrorException AlgebraicCoupling(dh; fields = (:u, :z), algebraic_coupling = trues(2, 2))
end

@testset "local layouts and assembly" begin
    grid = generate_grid(Quadrilateral, (3, 3))
    dh = DofHandler(grid)
    add!(dh, :u, Lagrange{RefQuadrilateral, 1}()^2)
    add!(dh, :p, Lagrange{RefQuadrilateral, 1}())
    add!(dh, :p0, AlgebraicVariable())
    add!(dh, :z, AlgebraicVariable{Vec{2}}())
    close!(dh)
    right = getfacetset(grid, "right")
    ccpl = CellCoupling(dh, 1:getncells(grid); fields = (:p, :z, :p0), algebraic_coupling = trues(3, 3))
    fcpl = FacetCoupling(dh, right; fields = (:u, :p0), algebraic_coupling = Bool[0 1; 1 0])
    acpl = AlgebraicCoupling(dh; fields = (:p0, :z), algebraic_coupling = trues(2, 2))

    # Deterministic appended ordering: celldofs, then algebraic in descriptor field order
    for cc in CellIterator(dh, [4])
        layout = local_dofs(cc, ccpl)
        @test layout isa LocalDofLayout
        @test collect(layout) == vcat(celldofs(cc), algebraic_dofs(dh, :z), algebraic_dofs(dh, :p0))
        @test layout[dof_range(layout, :u)] == celldofs(cc)[dof_range(dh, :u)]
        @test layout[dof_range(layout, :p)] == celldofs(cc)[dof_range(dh, :p)]
        @test layout[dof_range(layout, :z)] == algebraic_dofs(dh, :z)
        @test layout[dof_range(layout, :p0)] == algebraic_dofs(dh, :p0)
        @test_throws ErrorException dof_range(layout, :nope)
        # Read-only
        @test_throws Exception layout[1] = 1
        # Wrong entity/descriptor combinations (typed API: no such method)
        @test_throws MethodError local_dofs(cc, fcpl)
    end
    # Layout is unaffected by iterator advancement (owns its dof vector)
    layouts = LocalDofLayout[]
    for cc in CellIterator(dh, [1, 2])
        push!(layouts, local_dofs(cc, ccpl))
    end
    @test collect(layouts[1]) == vcat(celldofs(dh, 1), algebraic_dofs(dh, :z), algebraic_dofs(dh, :p0))
    @test collect(layouts[2]) == vcat(celldofs(dh, 2), algebraic_dofs(dh, :z), algebraic_dofs(dh, :p0))
    # In-place reuse with local_dofs!
    reused = LocalDofLayout()
    @test isempty(reused)
    for cc in CellIterator(dh, [1, 5])
        ret = local_dofs!(reused, cc, ccpl)
        @test ret === reused
        fresh = local_dofs(cc, ccpl)
        @test collect(reused) == collect(fresh)
        @test dof_range(reused, :p0) == dof_range(fresh, :p0)
    end
    for fc in FacetIterator(dh, right)
        local_dofs!(reused, fc, fcpl)
        @test collect(reused) == collect(local_dofs(fc, fcpl))
        @test dof_range(reused, :p0) == dof_range(local_dofs(fc, fcpl), :p0)
    end
    # Reuse does not allocate once warmed
    function sum_first_dofs!(layout, cc, cpl, cells)
        s = 0
        for cellid in cells
            reinit!(cc, cellid)
            local_dofs!(layout, cc, cpl)
            s += layout[1]
        end
        return s
    end
    let cc = CellCache(dh, UpdateFlags(nodes = false, coords = false, dofs = true))
        sum_first_dofs!(reused, cc, ccpl, 1:4)
        @test (@allocated sum_first_dofs!(reused, cc, ccpl, 1:4)) == 0
    end
    # Reuse across SubDofHandlers with different local layouts swaps the metadata
    let gridm = generate_grid(Quadrilateral, (2, 1))
        dhm = DofHandler(gridm)
        sdh1 = SubDofHandler(dhm, Set([1]))
        add!(sdh1, :u, Lagrange{RefQuadrilateral, 1}())
        add!(sdh1, :q, Lagrange{RefQuadrilateral, 1}())
        sdh2 = SubDofHandler(dhm, Set([2]))
        add!(sdh2, :u, Lagrange{RefQuadrilateral, 1}())
        add!(dhm, :λ, AlgebraicVariable())
        close!(dhm)
        cplm = CellCoupling(dhm, 1:2; fields = (:u, :λ), algebraic_coupling = trues(2, 2))
        lay = LocalDofLayout()
        for cc in CellIterator(dhm)
            local_dofs!(lay, cc, cplm)
            fresh = local_dofs(cc, cplm)
            @test collect(lay) == collect(fresh)
            @test dof_range(lay, :u) == dof_range(fresh, :u)
            @test dof_range(lay, :λ) == dof_range(fresh, :λ)
        end
        # After visiting cell 2 the metadata reflects the second SubDofHandler (no :q)
        @test length(lay) == ndofs_per_cell(dhm, 2) + 1
        @test_throws ErrorException dof_range(lay, :q)
    end
    # Entity must belong to the descriptor set
    sub = CellCoupling(dh, [1]; fields = (:p, :p0), algebraic_coupling = trues(2, 2))
    for cc in CellIterator(dh, [2])
        @test_throws ErrorException local_dofs(cc, sub)
    end
    for fc in FacetIterator(dh, getfacetset(grid, "left"))
        @test_throws ErrorException local_dofs(fc, fcpl)
    end
    # Mismatched handler
    dhx = DofHandler(grid)
    add!(dhx, :p, Lagrange{RefQuadrilateral, 1}())
    add!(dhx, :p0, AlgebraicVariable())
    close!(dhx)
    cplx = CellCoupling(dhx, [1]; fields = (:p, :p0), algebraic_coupling = trues(2, 2))
    for cc in CellIterator(dh, [1])
        @test_throws ErrorException local_dofs(cc, cplx)
    end
    # Algebraic layout
    lay = local_dofs(acpl)
    @test collect(lay) == vcat(algebraic_dofs(dh, :p0), algebraic_dofs(dh, :z))
    @test dof_range(lay, :p0) == 1:1
    @test dof_range(lay, :z) == 2:3

    # Assembly equivalence with direct global insertion, and residual assembly
    couplings = (c = ccpl, f = fcpl, a = acpl)
    K = allocate_matrix(dh; algebraic_couplings = couplings)
    f = zeros(ndofs(dh))
    Kref = zeros(ndofs(dh), ndofs(dh))
    fref = zeros(ndofs(dh))
    assembler = start_assemble(K, f)
    for cc in CellIterator(dh)
        layout = local_dofs(cc, ccpl)
        nl = length(layout)
        Ke = reshape(float.(1:(nl * nl)), nl, nl) .+ cellid(cc)
        fe = float.(1:nl)
        # keep u-rows/cols zero: mask them out like a kernel would
        ru = dof_range(layout, :u)
        Ke[ru, :] .= 0; Ke[:, ru] .= 0; fe[ru] .= 0
        assemble!(assembler, layout, Ke, fe)
        Kref[collect(layout), collect(layout)] .+= Ke
        fref[collect(layout)] .+= fe
    end
    lay0 = local_dofs(acpl)
    Ke0 = [1.0 2.0 3.0; 4.0 5.0 6.0; 7.0 8.0 9.0]
    fe0 = [1.0, 2.0, 3.0]
    assemble!(assembler, lay0, Ke0, fe0)
    Kref[collect(lay0), collect(lay0)] .+= Ke0
    fref[collect(lay0)] .+= fe0
    K, f = finish_assemble(assembler)
    @test Matrix(K) ≈ Kref
    @test f ≈ fref

    # Correctness after global renumbering
    renumber!(dh, DofOrder.FieldWise())
    ccpl2 = CellCoupling(dh, 1:getncells(grid); fields = (:p, :z, :p0), algebraic_coupling = trues(3, 3))
    for cc in CellIterator(dh, [4])
        layout = local_dofs(cc, ccpl2)
        @test layout[dof_range(layout, :z)] == algebraic_dofs(dh, :z)
        @test layout[dof_range(layout, :p0)] == algebraic_dofs(dh, :p0)
    end

    # Forgetting algebraic_couplings= gives the existing missing-sparsity-entry error
    Kmiss = allocate_matrix(dh)
    amiss = start_assemble(Kmiss)
    lmiss = local_dofs(AlgebraicCoupling(dh; fields = (:p0, :z), algebraic_coupling = trues(2, 2)))
    err = try
        assemble!(amiss, lmiss, ones(3, 3))
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
    cpl = CellCoupling(dh, 1:getncells(grid); fields = (:u, :p0), algebraic_coupling = trues(2, 2))
    ch = ConstraintHandler(dh)
    add!(ch, Dirichlet(:u, getfacetset(grid, "left"), (x, t) -> 0.0))
    # An AffineConstraint whose master is an algebraic dof
    free_u_dof = celldofs(dh, 3)[2]
    add!(ch, AffineConstraint(free_u_dof, [p0dof => 2.0], 1.0))
    close!(ch)
    K = allocate_matrix(dh, ch; algebraic_couplings = (cpl,))
    f = zeros(ndofs(dh))
    assembler = start_assemble(K, f)
    for cc in CellIterator(dh)
        layout = local_dofs(cc, cpl)
        nl = length(layout)
        Ke = Matrix(2.0 * I, nl, nl) .+ 0.5
        fe = ones(nl)
        apply_assemble!(assembler, ch, layout, Ke, fe)
    end
    K, f = finish_assemble(assembler)
    apply!(K, f, ch)
    a = K \ f
    apply!(a, ch)
    @test a[free_u_dof] ≈ 2.0 * a[p0dof] + 1.0
end

# With keep_constrained = false, coupling entries touching an affine-constrained dof must
# be distributed to the constraint's master dofs at insertion: eliminated entries are
# never stored, so the later constraint expansion cannot discover them, yet condensed
# assembly writes to the master entries (here the master is in a cell outside the
# coupling set, so nothing else provides them).
@testset "keep_constrained = false distributes coupling entries to masters" begin
    grid = generate_grid(Quadrilateral, (2, 1)) # two cells sharing one edge
    dh = DofHandler(grid)
    add!(dh, :u, Lagrange{RefQuadrilateral, 1}())
    add!(dh, :λ, AlgebraicVariable())
    close!(dh)
    cd1, cd2 = celldofs(dh, 1), celldofs(dh, 2)
    slave, prescribed = setdiff(cd1, cd2)  # dofs only in cell 1
    master = first(setdiff(cd2, cd1))      # dof only in cell 2
    λdof = only(algebraic_dofs(dh, :λ))
    ch = ConstraintHandler(dh)
    add!(ch, AffineConstraint(slave, [master => 2.0], 0.0))
    add!(ch, AffineConstraint(prescribed, Pair{Int, Float64}[], 1.0)) # no masters
    close!(ch); update!(ch, 0.0)
    coupling = CellCoupling(dh, [1]; algebraic_coupling = ((:u, :λ),))
    K = allocate_matrix(dh, ch; keep_constrained = false, algebraic_couplings = (coupling,))
    # The slave's coupling entries are distributed to its master ...
    @test hasentry(K, master, λdof) && hasentry(K, λdof, master)
    # ... while entries of the slave itself and of masterless prescribed dofs are dropped
    @test !hasentry(K, slave, λdof) && !hasentry(K, λdof, slave)
    @test !hasentry(K, prescribed, λdof) && !hasentry(K, λdof, prescribed)
    # Condensed assembly of the coupling term writes exactly the distributed entries;
    # the result on the free dofs matches the keep_constrained = true assembly
    function assemble_coupling!(K, dh, ch, coupling)
        f = zeros(ndofs(dh))
        assembler = start_assemble(K, f)
        qr = QuadratureRule{RefQuadrilateral}(2)
        cv = CellValues(qr, Lagrange{RefQuadrilateral, 1}())
        for cc in CellIterator(dh, [1])
            reinit!(cv, cc)
            layout = local_dofs(cc, coupling)
            n = length(layout)
            Ke = zeros(n, n); fe = zeros(n)
            ru = dof_range(layout, :u)
            iλ = only(dof_range(layout, :λ))
            for qp in 1:getnquadpoints(cv)
                dΩ = getdetJdV(cv, qp)
                for (i, I) in pairs(ru)
                    v = shape_value(cv, qp, i) * dΩ
                    Ke[I, iλ] += v
                    Ke[iλ, I] += v
                end
            end
            apply_assemble!(assembler, ch, layout, Ke, fe)
        end
        return finish_assemble(assembler)
    end
    Kc, fc = assemble_coupling!(K, dh, ch, coupling)
    Kfull = allocate_matrix(dh, ch; keep_constrained = true, algebraic_couplings = (coupling,))
    Kf, ff = assemble_coupling!(Kfull, dh, ch, coupling)
    free = setdiff(1:ndofs(dh), [slave, prescribed])
    @test Matrix(Kc)[free, free] ≈ Matrix(Kf)[free, free]
    @test fc[free] ≈ ff[free]
end

@testset "threaded atomic assembly" begin
    grid = generate_grid(Quadrilateral, (8, 8))
    dh = DofHandler(grid)
    add!(dh, :u, Lagrange{RefQuadrilateral, 1}())
    add!(dh, :p0, AlgebraicVariable())
    close!(dh)
    cpl = CellCoupling(dh, 1:getncells(grid); fields = (:u, :p0), algebraic_coupling = trues(2, 2))
    ncells = getncells(grid)
    function element!(Ke, fe, cellid)
        Ke .= 1.0 .+ cellid / ncells
        fe .= cellid
        return
    end
    function assemble_all!(K, f, atomic::Bool, coupling = cpl)
        chunks = collect(Iterators.partition(1:ncells, max(1, ncells ÷ 4)))
        tasks = map(chunks) do chunk
            Threads.@spawn begin
                # Each task owns its assembler, buffers, and iterator
                assembler = start_assemble(K, f; fillzero = false, atomic = atomic)
                for cc in CellIterator(dh, chunk)
                    layout = local_dofs(cc, coupling)
                    nl = length(layout)
                    Ke = zeros(nl, nl)
                    fe = zeros(nl)
                    element!(Ke, fe, cellid(cc))
                    assemble!(assembler, layout, Ke, fe)
                end
            end
        end
        foreach(wait, tasks)
        return K, f
    end
    # CSC
    Kser = allocate_matrix(dh; algebraic_couplings = (cpl,)); fser = zeros(ndofs(dh))
    aser = start_assemble(Kser, fser)
    for cc in CellIterator(dh)
        layout = local_dofs(cc, cpl)
        nl = length(layout)
        Ke = zeros(nl, nl); fe = zeros(nl)
        element!(Ke, fe, cellid(cc))
        assemble!(aser, layout, Ke, fe)
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
    cpl2 = CellCoupling(dh, 1:ncells; fields = (:u, :p0), algebraic_coupling = trues(2, 2))
    add_sparsity_entries!(sp, dh; algebraic_couplings = (cpl2,))
    # Serial reference in the renumbered ordering
    Kser2 = allocate_matrix(dh; algebraic_couplings = (cpl2,)); fser2 = zeros(ndofs(dh))
    aser2 = start_assemble(Kser2, fser2)
    for cc in CellIterator(dh)
        layout = local_dofs(cc, cpl2)
        nl = length(layout)
        Ke = zeros(nl, nl); fe = zeros(nl)
        element!(Ke, fe, cellid(cc))
        assemble!(aser2, layout, Ke, fe)
    end
    finish_assemble(aser2)
    Kblock = allocate_matrix(BlockMatrix, sp)
    fblock = BlockVector(zeros(ndofs(dh)), [ndofs(dh) - 1, 1])
    start_assemble(Kblock, fblock) # zero K and f once; the tasks use fillzero = false
    assemble_all!(Kblock, fblock, true, cpl2)
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
    dh = DofHandler(grid)
    ip = Lagrange{RefQuadrilateral, 1}()^2
    add!(dh, :u, ip)
    add!(dh, :σ̄, σ̄var)
    close!(dh)
    av = AlgebraicValues(σ̄var)

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

    descriptor = CellCoupling(dh, 1:getncells(grid); algebraic_coupling = ((:u, :σ̄),))
    K = allocate_matrix(dh, ch; algebraic_couplings = (rve = descriptor,))
    f = zeros(ndofs(dh))

    qr = QuadratureRule{RefQuadrilateral}(2)
    cv = CellValues(qr, ip)
    nbase = getnbasefunctions(cv)
    nlocal = nbase + ndofs(av)
    Ke = zeros(nlocal, nlocal) # hoisted out of the assembly loop
    fe = zeros(nlocal)
    # Element residual evaluated from the gathered augmented local unknowns `ae`: the
    # value and the basis directions are queried inside the routine, so differentiating
    # with respect to `ae` yields the full augmented element matrix (used as an AD
    # cross-check of the hand-assembled bilinear form below)
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
    for cc in CellIterator(dh)
        layout = local_dofs(cc, descriptor)
        range_u = dof_range(layout, :u)
        range_σ = dof_range(layout, :σ̄)
        @assert length(layout) == nlocal
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
                    # test u / trial σ̄ and (by symmetry) test σ̄ / trial u
                    Ek = algebraic_basis_value(av, k)
                    Ke[I, S] -= (εi ⊡ Ek) * dΩ
                    Ke[S, I] -= (Ek ⊡ εi) * dΩ
                end
            end
            for (k, S) in pairs(range_σ)
                fe[S] -= (algebraic_basis_value(av, k) ⊡ ε̄) * dΩ
            end
        end
        if cellid(cc) == 1
            ae = rand(nlocal) # the form is linear, so the Jacobian is independent of ae
            Ke_ad = ForwardDiff.jacobian(
                (r, x) -> element_residual!(r, x, cv, av, range_u, range_σ, C, ε̄),
                zeros(nlocal), ae
            )
            @test Ke_ad ≈ Ke
        end
        apply_assemble!(assembler, ch, layout, Ke, fe)
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
    descriptor = FacetCoupling(dh, boundary; algebraic_coupling = ((:u, :λ),))
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
    for fc in FacetIterator(dh, boundary)
        layout = local_dofs(fc, descriptor)
        range_u = dof_range(layout, :u)
        range_λ = dof_range(layout, :λ)
        fill!(Keb, 0); fill!(feb, 0)
        reinit!(fv, fc)
        for qp in 1:getnquadpoints(fv)
            dΓ = getdetJdV(fv, qp)
            for (i, I) in pairs(range_u)
                v = shape_value(fv, qp, i)
                Keb[I, only(range_λ)] += v * dΓ
                Keb[only(range_λ), I] += v * dΓ
            end
        end
        assemble!(assembler, layout, Keb, feb)
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
    zd = AlgebraicCoupling(dh; fields = (:p,), algebraic_coupling = trues(1, 1))
    K = allocate_matrix(dh; algebraic_couplings = (zd,))
    f = zeros(ndofs(dh))
    assembler = start_assemble(K, f)
    for cc in CellIterator(dh)
        n = ndofs_per_cell(dh)
        assemble!(assembler, celldofs(cc), Matrix(1.0 * I, n, n), zeros(n))
    end
    layout = local_dofs(zd)
    Kpp = [2.0 1.0; 1.0 3.0]
    fp = [1.0, 2.0]
    assemble!(assembler, layout, Kpp, fp)
    K, f = finish_assemble(assembler)
    a = K \ f
    @test a[pdofs] ≈ Kpp \ fp
end
