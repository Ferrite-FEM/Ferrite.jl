using Ferrite, Test, SparseArrays, Random
using SparseMatricesCSR: SparseMatrixCSR
using LinearAlgebra

# Minimal implementation of a custom sparsity pattern
struct TestPattern <: Ferrite.AbstractSparsityPattern
    nrowscols::Tuple{Int, Int}
    data::Vector{Vector{Int}}
    function TestPattern(m::Int, n::Int)
        return new((m, n), Vector{Int}[Int[] for _ in 1:m])
    end
end
Ferrite.getnrows(tp::TestPattern) = tp.nrowscols[1]
Ferrite.getncols(tp::TestPattern) = tp.nrowscols[2]
function Ferrite.add_entry!(tp::TestPattern, row::Int, col::Int)
    if !(1 <= row <= tp.nrowscols[1] && 1 <= col <= tp.nrowscols[2])
        error("out of bounds")
    end
    r = tp.data[row]
    k = searchsortedfirst(r, col)
    if k == lastindex(r) + 1 || r[k] != col
        insert!(r, k, col)
    end
    return
end
Ferrite.eachrow(tp::TestPattern) = tp.data
Ferrite.eachrow(tp::TestPattern, r::Int) = tp.data[r]

function compare_patterns(p1, px...)
    @test all(p -> Ferrite.getnrows(p1) == Ferrite.getnrows(p), px)
    @test all(p -> Ferrite.getncols(p1) == Ferrite.getncols(p), px)
    for rs in zip(Ferrite.eachrow.((p1, px...))...)
        for cs in zip(rs...)
            @test all(c -> cs[1] == c, cs)
        end
    end
    return
end

# Compare the storage of SparseMatrixCSR
function compare_matrices_csr(A1, Ax...)
    @assert A1 isa SparseMatrixCSR
    @assert length(Ax) > 0
    @assert all(A -> A isa SparseMatrixCSR, Ax)
    @test all(A -> size(A1) == size(A), Ax)
    @test all(A -> A1.rowptr == A.rowptr, Ax)
    @test all(A -> A1.colval == A.colval, Ax)
    return
end

# Compare the storage of SparseMatrixCSC
function compare_matrices(A1, Ax...)
    @assert A1 isa SparseMatrixCSC
    @assert length(Ax) > 0
    @assert all(A -> A isa SparseMatrixCSC, Ax)
    @test all(A -> size(A1) == size(A), Ax)
    @test all(A -> A1.colptr == A.colptr, Ax)
    @test all(A -> A1.rowval == A.rowval, Ax)
    return
end

function is_stored(dsp::SparsityPattern, i, j)
    return findfirst(k -> k == j, Ferrite.eachrow(dsp, i)) !== nothing
end

@testset "SparsityPattern" begin

    # Ferrite.add_entry!
    for (m, n) in ((5, 5), (3, 5), (5, 3))
        dsp = SparsityPattern(m, n)
        @test Ferrite.getnrows(dsp) == m
        @test Ferrite.getncols(dsp) == n
        for r in randperm(m), c in randperm(n)
            @test !is_stored(dsp, r, c)
            Ferrite.add_entry!(dsp, r, c)
            @test is_stored(dsp, r, c)
        end
        A = allocate_matrix(dsp)
        fill!(A.nzval, 1)
        @test A == ones(m, n)
        # Error paths
        @test_throws BoundsError Ferrite.add_entry!(dsp, 0, 1)
        @test_throws BoundsError Ferrite.add_entry!(dsp, 1, 0)
        @test_throws BoundsError Ferrite.add_entry!(dsp, m + 1, 1)
        @test_throws BoundsError Ferrite.add_entry!(dsp, 1, n + 1)
    end

    function testdhch()
        local grid, dh, ch
        grid = generate_grid(Quadrilateral, (2, 1))
        dh = DofHandler(grid)
        add!(dh, :v, Lagrange{RefQuadrilateral, 1}()^2)
        add!(dh, :s, Lagrange{RefQuadrilateral, 1}())
        close!(dh)
        ch = ConstraintHandler(dh)
        add!(ch, Dirichlet(:v, getfacetset(grid, "left"), (x, t) -> 0, [2]))
        add!(ch, Dirichlet(:s, getfacetset(grid, "left"), (x, t) -> 0))
        add!(ch, AffineConstraint(15, [1 => 0.5, 7 => 0.5], 0.0))
        close!(ch)
        return dh, ch
    end

    dh, ch = testdhch()

    # Mismatching size

    # Test show method
    dsp = SparsityPattern(ndofs(dh), ndofs(dh))
    str = sprint(show, "text/plain", dsp)
    @test contains(str, "$(ndofs(dh))×$(ndofs(dh))")
    @test contains(str, r" - Sparsity: 100.0% \(0 stored entries\)$"m)
    @test contains(str, r" - Entries per row \(min, max, avg\): 0, 0, 0.0$"m)
    @test contains(str, r" - Memory estimate: .* used, .* allocated$"m)
    add_sparsity_entries!(dsp, dh)
    str = sprint(show, "text/plain", dsp)
    @test contains(str, "$(ndofs(dh))×$(ndofs(dh))")
    @test contains(str, r" - Sparsity: .*% \(252 stored entries\)$"m)
    @test contains(str, r" - Entries per row \(min, max, avg\): 12, 18, 14\.0$"m)
    @test contains(str, r" - Memory estimate: .* used, .* allocated$"m)

    # Test all the possible entrypoints using SparsityPattern with a DofHandler
    compare_matrices(
        # Reference matrix from COO representation
        let I = Int[], J = Int[]
            for c in CellIterator(dh)
                for row in c.dofs, col in c.dofs
                    push!(I, row)
                    push!(J, col)
                end
            end
            sparse(I, J, zeros(Float64, length(I)), ndofs(dh), ndofs(dh))
        end,
        let
            A = allocate_matrix(dh)
            @test A isa SparseMatrixCSC{Float64, Int}
            A
        end,
        let
            A = allocate_matrix(SparseMatrixCSC{Float32, Int}, dh)
            @test A isa SparseMatrixCSC{Float32, Int}
            A
        end,
        let
            dsp = init_sparsity_pattern(dh)
            add_sparsity_entries!(dsp, dh)
            @test dsp isa SparsityPattern
            A = allocate_matrix(dsp)
            @test A isa SparseMatrixCSC{Float64, Int}
            A
        end,
        let
            dsp = init_sparsity_pattern(dh)
            add_sparsity_entries!(dsp, dh)
            A = allocate_matrix(SparseMatrixCSC{Float32, Int32}, dsp)
            @test A isa SparseMatrixCSC{Float32, Int32}
            A
        end,
        let
            dsp = SparsityPattern(ndofs(dh), ndofs(dh); nnz_per_row = 5)
            allocate_matrix(add_sparsity_entries!(dsp, dh))
        end,
    )

    # Test entrypoints with a DofHandler + ConstraintHandler
    compare_matrices(
        let
            A = allocate_matrix(dh, ch)
            @test A isa SparseMatrixCSC{Float64, Int}
            A
        end,
        let
            A = allocate_matrix(SparseMatrixCSC{Float32, Int}, dh, ch)
            @test A isa SparseMatrixCSC{Float32, Int}
            A
        end,
        let
            dsp = init_sparsity_pattern(dh)
            add_sparsity_entries!(dsp, dh, ch)
            @test dsp isa SparsityPattern
            A = allocate_matrix(dsp)
            @test A isa SparseMatrixCSC{Float64, Int}
            A
        end,
        let
            dsp = init_sparsity_pattern(dh)
            add_sparsity_entries!(dsp, dh, ch)
            A = allocate_matrix(SparseMatrixCSC{Float32, Int32}, dsp)
            @test A isa SparseMatrixCSC{Float32, Int32}
            A
        end,
        let
            dsp = SparsityPattern(ndofs(dh), ndofs(dh))
            allocate_matrix(add_sparsity_entries!(dsp, dh, ch))
        end,
    )

    # Test entrypoints with a DofHandler + coupling + remove constrained
    kwargs = (; coupling = [true true; false true], keep_constrained = false)
    compare_matrices(
        let
            A = allocate_matrix(dh, ch; kwargs...)
            @test A isa SparseMatrixCSC{Float64, Int}
            A
        end,
        let
            A = allocate_matrix(SparseMatrixCSC{Float32, Int}, dh, ch; kwargs...)
            @test A isa SparseMatrixCSC{Float32, Int}
            A
        end,
        let
            dsp = init_sparsity_pattern(dh)
            add_sparsity_entries!(dsp, dh, ch; kwargs...)
            @test dsp isa SparsityPattern
            A = allocate_matrix(dsp)
            @test A isa SparseMatrixCSC{Float64, Int}
            A
        end,
        let
            dsp = init_sparsity_pattern(dh)
            add_sparsity_entries!(dsp, dh, ch; kwargs...)
            A = allocate_matrix(SparseMatrixCSC{Float32, Int32}, dsp)
            @test A isa SparseMatrixCSC{Float32, Int32}
            A
        end,
        let
            dsp = SparsityPattern(ndofs(dh), ndofs(dh))
            allocate_matrix(add_sparsity_entries!(dsp, dh, ch; kwargs...))
        end,
    )

end

@testset "Sparsity pattern generics" begin

    # Test setup
    grid = generate_grid(Hexahedron, (5, 5, 5))
    dh = DofHandler(grid)
    add!(dh, :u, Lagrange{RefHexahedron, 2}()^3)
    add!(dh, :p, Lagrange{RefHexahedron, 1}())
    close!(dh)
    ch = ConstraintHandler(dh)
    add!(ch, Dirichlet(:p, union(getfacetset.(Ref(grid), ("left", "right", "top", "bottom", "front", "back"))...), x -> 0))
    add!(ch, PeriodicDirichlet(:u, collect_periodic_facets(grid)))
    close!(ch)

    function make_patterns(dh)
        nd = ndofs(dh)
        tp = TestPattern(nd, nd)
        sp = SparsityPattern(nd, nd)
        bp = BlockSparsityPattern([nd ÷ 2, nd - nd ÷ 2])
        return tp, sp, bp
    end

    # DofHandler
    ps = make_patterns(dh)
    for p in ps
        add_sparsity_entries!(p, dh)
    end
    compare_patterns(ps...)
    # The SparseMatrixCSR instantiation works for any pattern whose eachrow iterates sorted
    # column indices; equal patterns give identical storage. This covers both row copy
    # branches: AbstractVector rows (TestPattern, SparsityPattern) and the iteration
    # fallback (BlockSparsityPattern's lazy rows).
    compare_matrices_csr((allocate_matrix(SparseMatrixCSR{1, Float64, Int}, p) for p in ps)...)

    # DofHandler + ConstraintHandler
    ps = make_patterns(dh)
    for p in ps
        add_sparsity_entries!(p, dh, ch)
    end
    compare_patterns(ps...)

    # DofHandler + ConstraintHandler later
    ps = make_patterns(dh)
    for p in ps
        add_sparsity_entries!(p, dh)
        add_constraint_entries!(p, ch)
    end
    compare_patterns(ps...)

    # Individual pieces
    ps = make_patterns(dh)
    for p in ps
        add_cell_entries!(p, dh)
        add_constraint_entries!(p, ch)
    end
    compare_patterns(ps...)

    # Ignore constrained dofs
    ps = make_patterns(dh)
    for p in ps
        add_sparsity_entries!(p, dh, ch; keep_constrained = false)
        # Test that prescribed dofs only have diagonal entry
        for row in ch.prescribed_dofs
            r = Ferrite.eachrow(p, row)
            col, state = iterate(r)
            @test col == row
            @test iterate(r, state) === nothing
        end
    end
    compare_patterns(ps...)

    # Coupling
    ps = make_patterns(dh)
    for p in ps
        add_sparsity_entries!(p, dh, ch; coupling = [true true; false true])
    end
    compare_patterns(ps...)
    ps = make_patterns(dh)
    for p in ps
        coupling = ones(Bool, 4, 4)
        coupling[4, 1:3] .= false
        add_sparsity_entries!(p, dh, ch; coupling = coupling)
    end
    compare_patterns(ps...)
    ps = make_patterns(dh)
    for p in ps
        coupling = ones(Bool, ndofs_per_cell(dh), ndofs_per_cell(dh))
        coupling[1:2:(ndofs_per_cell(dh) ÷ 2), :] .= false
        add_sparsity_entries!(p, dh, ch; coupling = coupling)
    end
    compare_patterns(ps...)

    # Error paths
    dh_open = DofHandler(grid)
    for p in (SparsityPattern(2, 2), TestPattern(2, 2), BlockSparsityPattern([1, 1]))
        @test_throws ErrorException("the DofHandler must be closed") add_sparsity_entries!(p, dh_open)
    end
    for p in (SparsityPattern(ndofs(dh), 2), TestPattern(ndofs(dh), 2), BlockSparsityPattern([2, 2]))
        @test_throws ErrorException add_sparsity_entries!(p, dh)
    end
    for p in (SparsityPattern(2, ndofs(dh)), TestPattern(2, ndofs(dh)))
        @test_throws ErrorException add_sparsity_entries!(p, dh)
    end
    patterns = (
        SparsityPattern(ndofs(dh), ndofs(dh)),
        TestPattern(ndofs(dh), ndofs(dh)),
        BlockSparsityPattern([ndofs(dh) ÷ 2, ndofs(dh) - ndofs(dh) ÷ 2]),
    )
    for p in patterns
        @test_throws ErrorException add_sparsity_entries!(p, dh; keep_constrained = false)
    end
    ch_open = ConstraintHandler(dh)
    for p in patterns
        @test_throws ErrorException add_sparsity_entries!(p, dh, ch_open; keep_constrained = false)
    end
    ch_bad = ConstraintHandler(close!(DofHandler(grid)))
    for p in patterns
        @test_throws ErrorException add_sparsity_entries!(p, dh, ch_bad; keep_constrained = false)
    end
    # interface_coupling: the topology is constructed automatically when not passed
    sp_ic = add_sparsity_entries!(SparsityPattern(ndofs(dh), ndofs(dh)), dh; interface_coupling = trues(2, 2))
    sp_ic_topo = add_sparsity_entries!(SparsityPattern(ndofs(dh), ndofs(dh)), dh; interface_coupling = trues(2, 2), topology = ExclusiveTopology(grid))
    compare_patterns(sp_ic, sp_ic_topo)
end

@testset "SparsityPattern fast path" begin
    function fsp_test_create_dh(CT)
        RS = getrefshape(CT)
        dim = Ferrite.getrefdim(RS)
        grid = generate_grid(CT, ntuple(_ -> 5, dim))
        dh = DofHandler(grid)
        add!(dh, :a, Lagrange{RS, 1}())
        add!(dh, :b, Lagrange{RS, 2}()^dim)
        return close!(dh)
    end
    # The internal fast (count-presize marker fill) path is taken for any fresh pattern. Force
    # the generic per-entry path for comparison by seeding one diagonal entry first (the
    # diagonal is always part of the pattern, so the built patterns must be identical).
    function fsp_test_build_generic(dh, args...; kwargs...)
        sp = init_sparsity_pattern(dh)
        Ferrite.add_entry!(sp, 1, 1)
        return add_sparsity_entries!(sp, dh, args...; kwargs...)
    end
    for CT in (Line, Quadrilateral, Tetrahedron)
        dh = fsp_test_create_dh(CT)
        sp = add_sparsity_entries!(init_sparsity_pattern(dh), dh)
        sp_gen = fsp_test_build_generic(dh)
        K1 = allocate_matrix(sp)
        K2 = allocate_matrix(sp_gen)
        K3 = allocate_matrix(dh)
        compare_matrices(K1, K2) # For CSC only
        compare_matrices(K1, K3)

        K1_csr = allocate_matrix(SparseMatrixCSR, sp)
        K2_csr = allocate_matrix(SparseMatrixCSR, sp_gen)
        K3_csr = allocate_matrix(SparseMatrixCSR, dh)
        K1_csr.nzval .= 1:length(K1_csr.nzval)
        K2_csr.nzval .= 1:length(K2_csr.nzval)
        K3_csr.nzval .= 1:length(K3_csr.nzval)
        @test K1_csr == K2_csr
        @test K1_csr == K3_csr
    end
    # Coupling and keep_constrained are handled by the fast path (masked/filtered counting);
    # compare against the generic path for the same arguments.
    for CT in (Quadrilateral, Tetrahedron)
        dh = fsp_test_create_dh(CT)
        ch = ConstraintHandler(dh)
        add!(ch, Dirichlet(:a, getfacetset(dh.grid, "left"), x -> 0))
        close!(ch)
        for kwargs in (
                (; coupling = [true true; false true]),
                (; coupling = [true false; false true]),
                (; keep_constrained = false),
                (; coupling = [true true; false true], keep_constrained = false),
            )
            sp_fast = add_sparsity_entries!(init_sparsity_pattern(dh), dh, ch; kwargs...)
            sp_gen = fsp_test_build_generic(dh, ch; kwargs...)
            compare_matrices(allocate_matrix(sp_fast), allocate_matrix(sp_gen))
        end
    end
    # Interface entries (topology) are reserved up front, filtered by interface_coupling. For a
    # single-subdofhandler discretization the reservation is exact and doubles as the fill
    # itself, i.e. no row relocations and no layered interface insertion. (Multiple
    # subdofhandlers still take the layered path: the fast fill's expanded masks are square
    # per-subdofhandler and do not apply to a neighbor with a different dof layout;
    # rectangular per-(sdh, sdh) masks in the walk would lift that, as a follow-up.)
    let
        grid = generate_grid(Quadrilateral, (5, 5))
        dh = DofHandler(grid)
        add!(dh, :a, DiscontinuousLagrange{RefQuadrilateral, 1}())
        add!(dh, :b, DiscontinuousLagrange{RefQuadrilateral, 1}()^2)
        close!(dh)
        topo = ExclusiveTopology(grid)
        for ic in (trues(2, 2), [false false; false true], [true true; false true])
            sp = add_sparsity_entries!(init_sparsity_pattern(dh), dh; topology = topo, interface_coupling = ic)
            sp_gen = fsp_test_build_generic(dh; topology = topo, interface_coupling = ic)
            compare_matrices(allocate_matrix(sp), allocate_matrix(sp_gen))
            # The reservation is exact for a fully discontinuous single-sdh discretization,
            # asymmetric masks included
            @test sum(r -> r.nmax, sp.buffer.indices) == sum(length, Ferrite.eachrow(sp))
        end
        # Mixed continuous/discontinuous fields: with the self-sufficient interface semantics
        # the enumeration is exact for any continuity (single subdofhandler), so the direct
        # fill and exact reservation apply here too.
        dh2 = DofHandler(grid)
        add!(dh2, :a, Lagrange{RefQuadrilateral, 1}())
        add!(dh2, :b, DiscontinuousLagrange{RefQuadrilateral, 1}()^2)
        close!(dh2)
        for ic in (trues(2, 2), [false false; false true])
            sp = add_sparsity_entries!(init_sparsity_pattern(dh2), dh2; topology = topo, interface_coupling = ic)
            sp_gen = fsp_test_build_generic(dh2; topology = topo, interface_coupling = ic)
            compare_matrices(allocate_matrix(sp), allocate_matrix(sp_gen))
            @test sum(r -> r.nmax, sp.buffer.indices) == sum(length, Ferrite.eachrow(sp))
        end
        # Same-side interface blocks: with a restricted cell coupling the interface pass also
        # provides masked pairs within each interface cell, so the own-cell candidates in the
        # fast fill must pass on the interface mask too.
        for (cc, ic) in (
                ([true false; false true], [false true; false false]),
                ([true false; false true], trues(2, 2)),
            )
            sp = add_sparsity_entries!(init_sparsity_pattern(dh2), dh2; coupling = cc, topology = topo, interface_coupling = ic)
            sp_gen = fsp_test_build_generic(dh2; coupling = cc, topology = topo, interface_coupling = ic)
            compare_matrices(allocate_matrix(sp), allocate_matrix(sp_gen))
            @test sum(r -> r.nmax, sp.buffer.indices) == sum(length, Ferrite.eachrow(sp))
        end
        # Multiple subdofhandlers: the direct interface fill must NOT trigger (the square
        # coupling masks do not apply to cross-subdofhandler neighbors in the walk) and the
        # layered path must match the generic one.
        dh3 = DofHandler(grid)
        sdh_a = SubDofHandler(dh3, Set(1:12))
        add!(sdh_a, :a, DiscontinuousLagrange{RefQuadrilateral, 1}())
        sdh_b = SubDofHandler(dh3, Set(13:25))
        add!(sdh_b, :a, DiscontinuousLagrange{RefQuadrilateral, 1}())
        close!(dh3)
        let ic = trues(1, 1)
            sp = add_sparsity_entries!(init_sparsity_pattern(dh3), dh3; topology = topo, interface_coupling = ic)
            sp_gen = fsp_test_build_generic(dh3; topology = topo, interface_coupling = ic)
            compare_matrices(allocate_matrix(sp), allocate_matrix(sp_gen))
        end
    end
    # Patterns allocated larger than ndofs(dh) work with the fast path (the extra rows and
    # columns get diagonal-only entries)
    let
        dh = fsp_test_create_dh(Quadrilateral)
        n = ndofs(dh) + 3
        sp_big = add_sparsity_entries!(SparsityPattern(n, n), dh)
        @test Ferrite.getnrows(sp_big) == n
        sp_big_gen = SparsityPattern(n, n)
        Ferrite.add_entry!(sp_big_gen, 1, 1) # forces the generic branch
        add_sparsity_entries!(sp_big_gen, dh)
        compare_matrices(allocate_matrix(sp_big), allocate_matrix(sp_big_gen))
    end
    # Test different number types (Int32, Float32)
    for Tv in (Float32, Float64)
        for Ti in (Int32, Int64)
            dh = fsp_test_create_dh(Quadrilateral)
            MatrixType = SparseMatrixCSC{Float64, Ti}
            K1 = allocate_matrix(MatrixType, dh)
            @test isa(K1, MatrixType)
            compare_matrices(K1, allocate_matrix(dh))

            MatrixTypeCSR = SparseMatrixCSR{1, Tv, Ti}
            K1_csr = allocate_matrix(MatrixTypeCSR, dh)
            @test isa(K1_csr, MatrixTypeCSR)
        end
    end

    # Test logic for Symmetric matrices works (will use SparsityPattern)
    dh = fsp_test_create_dh(Triangle)
    for MatrixType in (
            Symmetric{Float64, SparseMatrixCSC{Float64, Int}},
            # Symmetric{Float64, SparseMatrixCSR{1, Float64, Int}} # Does not work, see #1311
        )
        K1 = allocate_matrix(MatrixType, dh)
        @test isa(K1, MatrixType)
    end
    # Symmetric shares the specialized raw-read path, which must handle unsorted rows;
    # compare against the generic AbstractSparsityPattern method (which sorts) on the same
    # pattern. Order matters: the specialized call must run first, while rows are unsorted.
    let dh = fsp_test_create_dh(Quadrilateral)
        sp = add_sparsity_entries!(init_sparsity_pattern(dh), dh)
        K = allocate_matrix(Symmetric{Float64, SparseMatrixCSC{Float64, Int}}, sp)
        S = Base.invoke(
            Ferrite._allocate_matrix,
            Tuple{Type{SparseMatrixCSC{Float64, Int}}, Ferrite.AbstractSparsityPattern, Bool},
            SparseMatrixCSC{Float64, Int}, sp, true
        )
        @test K.data.colptr == S.colptr
        @test K.data.rowval == S.rowval
    end
end
