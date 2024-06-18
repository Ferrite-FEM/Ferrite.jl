using Ferrite, Test, SparseArrays, Random

# Minimal implementation of a custom sparsity pattern
struct TestPattern <: Ferrite.AbstractSparsityPattern
    nrowscols::Tuple{Int, Int}
    data::Vector{Vector{Int}}
    function TestPattern(m::Int, n::Int)
        return new((m, n), Vector{Int}[Int[] for _ in 1:m])
    end
end
Ferrite.n_rows(tp::TestPattern) = tp.nrowscols[1]
Ferrite.n_cols(tp::TestPattern) = tp.nrowscols[2]
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
    @test all(p -> Ferrite.n_rows(p1) == Ferrite.n_rows(p), px)
    @test all(p -> Ferrite.n_cols(p1) == Ferrite.n_cols(p), px)
    for rs in zip(Ferrite.eachrow.((p1, px...,))...)
        for cs in zip(rs...)
            @test all(c -> cs[1] == c, cs)
        end
    end
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
    return findfirst(k -> k == j, dsp.rows[i]) !== nothing
end

@testset "SparsityPattern" begin

    # Ferrite.add_entry!
    for (m, n) in ((5, 5), (3, 5), (5, 3))
        dsp = SparsityPattern(m, n)
        @test Ferrite.n_rows(dsp) == m
        @test Ferrite.n_cols(dsp) == n
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
        @test_throws BoundsError Ferrite.add_entry!(dsp, m+1, 1)
        @test_throws BoundsError Ferrite.add_entry!(dsp, 1, n+1)
    end

    function testdhch()
        local grid, dh, ch
        grid = generate_grid(Quadrilateral, (2, 1))
        dh = DofHandler(grid)
        add!(dh, :v, Lagrange{RefQuadrilateral,1}()^2)
        add!(dh, :s, Lagrange{RefQuadrilateral,1}())
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
    create_sparsity_pattern!(dsp, dh)
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
            create_sparsity_pattern!(dsp, dh)
            @test dsp isa SparsityPattern
            A = allocate_matrix(dsp)
            @test A isa SparseMatrixCSC{Float64, Int}
            A
        end,
        let
            dsp = init_sparsity_pattern(dh)
            create_sparsity_pattern!(dsp, dh)
            A = allocate_matrix(SparseMatrixCSC{Float32, Int32}, dsp)
            @test A isa SparseMatrixCSC{Float32, Int32}
            A
        end,
        let
            dsp = SparsityPattern(ndofs(dh), ndofs(dh); nnz_per_row = 5)
            allocate_matrix(create_sparsity_pattern!(dsp, dh))
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
            create_sparsity_pattern!(dsp, dh, ch)
            @test dsp isa SparsityPattern
            A = allocate_matrix(dsp)
            @test A isa SparseMatrixCSC{Float64, Int}
            A
        end,
        let
            dsp = init_sparsity_pattern(dh)
            create_sparsity_pattern!(dsp, dh, ch)
            A = allocate_matrix(SparseMatrixCSC{Float32, Int32}, dsp)
            @test A isa SparseMatrixCSC{Float32, Int32}
            A
        end,
        let
            dsp = SparsityPattern(ndofs(dh), ndofs(dh))
            allocate_matrix(create_sparsity_pattern!(dsp, dh, ch))
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
            create_sparsity_pattern!(dsp, dh, ch; kwargs...)
            @test dsp isa SparsityPattern
            A = allocate_matrix(dsp)
            @test A isa SparseMatrixCSC{Float64, Int}
            A
        end,
        let
            dsp = init_sparsity_pattern(dh)
            create_sparsity_pattern!(dsp, dh, ch; kwargs...)
            A = allocate_matrix(SparseMatrixCSC{Float32, Int32}, dsp)
            @test A isa SparseMatrixCSC{Float32, Int32}
            A
        end,
        let
            dsp = SparsityPattern(ndofs(dh), ndofs(dh))
            allocate_matrix(create_sparsity_pattern!(dsp, dh, ch; kwargs...))
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
    add!(ch, Dirichlet(:p, union(getfacetset.(Ref(grid), ("left", "right", "top", "bottom", "front", "back"),)...), x -> 0))
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
        create_sparsity_pattern!(p, dh)
    end
    compare_patterns(ps...)

    # DofHandler + ConstraintHandler
    ps = make_patterns(dh)
    for p in ps
        create_sparsity_pattern!(p, dh, ch)
    end
    compare_patterns(ps...)

    # DofHandler + ConstraintHandler later
    ps = make_patterns(dh)
    for p in ps
        create_sparsity_pattern!(p, dh)
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
        create_sparsity_pattern!(p, dh, ch; keep_constrained=false)
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
        create_sparsity_pattern!(p, dh, ch; coupling = [true true; false true])
    end
    compare_patterns(ps...)
    ps = make_patterns(dh)
    for p in ps
        coupling = ones(Bool, 4, 4)
        coupling[4, 1:3] .= false
        create_sparsity_pattern!(p, dh, ch; coupling = coupling)
    end
    compare_patterns(ps...)
    ps = make_patterns(dh)
    for p in ps
        coupling = ones(Bool, ndofs_per_cell(dh), ndofs_per_cell(dh))
        coupling[1:2:(ndofs_per_cell(dh) ÷ 2), :] .= false
        create_sparsity_pattern!(p, dh, ch; coupling = coupling)
    end
    compare_patterns(ps...)

    # Error paths
    dh_open = DofHandler(grid)
    for p in (SparsityPattern(2, 2), TestPattern(2, 2), BlockSparsityPattern([1, 1]))
        @test_throws ErrorException("the DofHandler must be closed") create_sparsity_pattern!(p, dh_open)
    end
    for p in (SparsityPattern(ndofs(dh), 2), TestPattern(ndofs(dh), 2), BlockSparsityPattern([2, 2]))
        @test_throws ErrorException create_sparsity_pattern!(p, dh)
    end
    for p in (SparsityPattern(2, ndofs(dh)), TestPattern(2, ndofs(dh)))
        @test_throws ErrorException create_sparsity_pattern!(p, dh)
    end
    patterns = (
        SparsityPattern(ndofs(dh), ndofs(dh)),
        TestPattern(ndofs(dh), ndofs(dh)),
        BlockSparsityPattern([ndofs(dh) ÷ 2, ndofs(dh) - ndofs(dh) ÷ 2]),
    )
    for p in patterns
        @test_throws ErrorException create_sparsity_pattern!(p, dh; keep_constrained=false)
    end
    ch_open = ConstraintHandler(dh)
    for p in patterns
        @test_throws ErrorException create_sparsity_pattern!(p, dh, ch_open; keep_constrained=false)
    end
    ch_bad = ConstraintHandler(close!(DofHandler(grid)))
    for p in patterns
        @test_throws ErrorException create_sparsity_pattern!(p, dh, ch_bad; keep_constrained=false)
    end
end
