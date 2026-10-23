# Tests for interface assembly with dofs shared between the neighboring cells (#1433):
# the stacked interface dof layout, interface_dof_range, condense_duplicates!,
# and assembly of interface terms for continuous fields.
using Ferrite
using Test
using LinearAlgebra
using SparseArrays
import ForwardDiff

# The duplication map T (unique -> stacked coefficients, u_stacked = T * u_unique) built
# independently of the library, from the dof vector alone (first-occurrence order)
function duplication_map(dofs::AbstractVector{Int})
    udofs = unique(dofs)
    T = zeros(length(dofs), length(udofs))
    for (i, d) in pairs(dofs)
        T[i, findfirst(==(d), udofs)] = 1.0
    end
    return udofs, T
end

# Dense reference for `assemble!` semantics: K[dofs, dofs] += Ke with summation over
# repeated indices
function dense_scatter!(K::Matrix, dofs, Ke, f = nothing, fe = nothing)
    for (j, J) in pairs(dofs), (i, I) in pairs(dofs)
        K[I, J] += Ke[i, j]
    end
    if f !== nothing
        for (i, I) in pairs(dofs)
            f[I] += fe[i]
        end
    end
    return
end

@testset "Stacked interface dof layout" begin
    function check_layout(ip, grid; shares::Bool)
        dh = DofHandler(grid)
        add!(dh, :u, ip)
        close!(dh)
        topology = ExclusiveTopology(grid)
        ic = first(InterfaceIterator(dh, topology))
        sd = interfacedofs(ic)
        @test sd == [celldofs(ic.a); celldofs(ic.b)]
        @test ndofs_per_interface(ic) == length(sd)
        @test (length(unique(sd)) < length(sd)) == shares
        return ic
    end
    grid_quad = generate_grid(Quadrilateral, (2, 1))
    grid_tri = generate_grid(Triangle, (2, 1))
    # H1 shares facet (and vertex) dofs; L2 with cell-interior dofs shares nothing
    check_layout(Lagrange{RefQuadrilateral, 2}(), grid_quad; shares = true)
    check_layout(DiscontinuousLagrange{RefQuadrilateral, 1}(), grid_quad; shares = false)
    # Nonconforming interpolations are classified L2Conformity but place dofs on facets,
    # which are shared: dof sharing follows dof placement, not conformity
    check_layout(CrouzeixRaviart{RefTriangle, 1}(), grid_tri; shares = true)
    @test Ferrite.conformity(CrouzeixRaviart{RefTriangle, 1}()) isa Ferrite.L2Conformity
    check_layout(RannacherTurek{RefQuadrilateral, 1}(), grid_quad; shares = true)
    @test Ferrite.conformity(RannacherTurek{RefQuadrilateral, 1}()) isa Ferrite.L2Conformity
    # H(div) and H(curl) share facet/edge dofs; vector-valued H1 shares too
    check_layout(RaviartThomas{RefTriangle, 1}(), grid_tri; shares = true)
    check_layout(Nedelec{RefTriangle, 1}(), grid_tri; shares = true)
    check_layout(Lagrange{RefQuadrilateral, 1}()^2, grid_quad; shares = true)

    # ndofs_per_interface varies across interfaces when the cells belong to different
    # SubDofHandlers (the same-subdomain interface of the larger handler is the biggest)
    grid3 = generate_grid(Quadrilateral, (3, 1))
    topo3 = ExclusiveTopology(grid3)
    dh4 = DofHandler(grid3)
    big = SubDofHandler(dh4, Set([1, 2])) # cells 1-2: large sdh with an internal interface
    add!(big, :u, Lagrange{RefQuadrilateral, 2}())
    add!(big, :p, Lagrange{RefQuadrilateral, 1}())
    small = SubDofHandler(dh4, Set(3)) # cell 3: strictly smaller sdh
    add!(small, :u, Lagrange{RefQuadrilateral, 2}())
    close!(dh4)
    @test ndofs_per_cell(big) > ndofs_per_cell(small)
    sizes = Int[]
    for ic4 in InterfaceIterator(dh4, topo3)
        @test ndofs_per_interface(ic4) == length(interfacedofs(ic4))
        push!(sizes, ndofs_per_interface(ic4))
    end
    @test length(sizes) == 2 && length(unique(sizes)) == 2
end

@testset "interface_dof_range and InterfaceDofRange" begin
    # Single SubDofHandler, mixed continuous/discontinuous
    grid = generate_grid(Quadrilateral, (2, 1))
    topology = ExclusiveTopology(grid)
    dh = DofHandler(grid)
    add!(dh, :u, Lagrange{RefQuadrilateral, 1}())
    add!(dh, :P, DiscontinuousLagrange{RefQuadrilateral, 1}())
    close!(dh)
    ic = first(InterfaceIterator(dh, topology))
    sdh = dh.subdofhandlers[1]
    ncdofs = ndofs_per_cell(sdh)
    for field in (:u, :P)
        r = interface_dof_range(ic, field)
        @test r isa Ferrite.InterfaceDofRange
        @test r.here == dof_range(sdh, field)
        @test r.there == dof_range(sdh, field) .+ ncdofs
        # AbstractVector behavior: length, getindex, iteration, pairs
        @test length(r) == length(r.here) + length(r.there)
        @test collect(r) == [collect(r.here); collect(r.there)]
        @test [i for (i, I) in pairs(r)] == 1:length(r)
        @test [I for (i, I) in pairs(r)] == collect(r)
        @test_throws BoundsError r[length(r) + 1]
        # The values index the stacked layout
        @test interfacedofs(ic)[r] == [celldofs(ic.a)[dof_range(sdh, field)]; celldofs(ic.b)[dof_range(sdh, field)]]
        # Single SubDofHandler: the range can be computed once from the DofHandler,
        # outside the interface loop
        rdh = interface_dof_range(dh, field)
        @test rdh isa Ferrite.InterfaceDofRange
        @test rdh.here == r.here && rdh.there == r.there
    end
    # ... and so can the stacked size
    @test ndofs_per_interface(dh) == ndofs_per_interface(ic) == 2 * ndofs_per_cell(dh)
    # function_* evaluation with an InterfaceDofRange equals the two-range methods
    iv_u = InterfaceValues(FacetQuadratureRule{RefQuadrilateral}(2), Lagrange{RefQuadrilateral, 1}())
    reinit!(iv_u, ic)
    a = rand(ndofs(dh))
    ue = a[interfacedofs(ic)]
    ru = interface_dof_range(ic, :u)
    for qp in 1:getnquadpoints(iv_u)
        # Reference: evaluate each side directly through the FacetValues with the
        # corresponding block of the stacked coefficient vector
        v_here = function_value(iv_u.here, qp, ue, ru.here)
        v_there = function_value(iv_u.there, qp, ue, ru.there)
        @test function_value(iv_u, qp, ue, ru; here = true) == v_here
        @test function_value(iv_u, qp, ue, ru; here = false) == v_there
        @test function_value_jump(iv_u, qp, ue, ru) == v_there - v_here
        @test function_value_average(iv_u, qp, ue, ru) == (v_here + v_there) / 2
        g_here = function_gradient(iv_u.here, qp, ue, ru.here)
        g_there = function_gradient(iv_u.there, qp, ue, ru.there)
        @test function_gradient_jump(iv_u, qp, ue, ru) == g_there - g_here
        @test function_gradient_average(iv_u, qp, ue, ru) == (g_here + g_there) / 2
    end

    # Multiple SubDofHandlers with different field orders on the two sides
    dh2 = DofHandler(grid)
    ip = Lagrange{RefQuadrilateral, 1}()
    sdh1 = SubDofHandler(dh2, Set(1))
    add!(sdh1, :u, ip)
    add!(sdh1, :p, ip)
    add!(sdh1, :q, ip) # only on this subdomain
    sdh2 = SubDofHandler(dh2, Set(2))
    add!(sdh2, :p, ip) # note: different field order than sdh1
    add!(sdh2, :u, ip)
    close!(dh2)
    ic2 = first(InterfaceIterator(dh2, topology))
    sdh_a = dh2.subdofhandlers[dh2.cell_to_subdofhandler[cellid(ic2.a)]]
    sdh_b = dh2.subdofhandlers[dh2.cell_to_subdofhandler[cellid(ic2.b)]]
    for field in (:u, :p)
        r = interface_dof_range(ic2, field)
        @test r.here == dof_range(sdh_a, field)
        @test r.there == dof_range(sdh_b, field) .+ ndofs_per_cell(sdh_a)
        @test interfacedofs(ic2)[r] ==
            [celldofs(ic2.a)[dof_range(sdh_a, field)]; celldofs(ic2.b)[dof_range(sdh_b, field)]]
    end
    # A field present on only one side is rejected
    @test_throws ErrorException interface_dof_range(ic2, :q)
    err = try
        interface_dof_range(ic2, :q); nothing
    catch e
        e
    end
    @test occursin("only one side", err.msg)
    # Unknown fields are rejected too
    @test_throws ErrorException interface_dof_range(ic2, :nonexistent)
    # The DofHandler methods require a single SubDofHandler (the range/size is not
    # constant across interfaces otherwise) and point at the InterfaceCache methods
    err2 = try
        interface_dof_range(dh2, :u); nothing
    catch e
        e
    end
    @test err2 isa ErrorException
    @test occursin("InterfaceCache", err2.msg)
    err3 = try
        ndofs_per_interface(dh2); nothing
    catch e
        e
    end
    @test err3 isa ErrorException
    @test occursin("InterfaceCache", err3.msg)
    # Grid-constructed caches have no dof information
    ic3 = InterfaceCache(grid)
    reinit!(ic3, FacetIndex(1, 2), FacetIndex(2, 4))
    @test isempty(interfacedofs(ic3))
    @test ndofs_per_interface(ic3) == 0
    @test_throws ErrorException interface_dof_range(ic3, :u)

    # SubDofHandler-constructed cache: dof information resolves through sdh.dh. The cache
    # keeps the existing domain restriction: both cells must belong to the sub-handler.
    grid3 = generate_grid(Quadrilateral, (3, 1))
    dh3 = DofHandler(grid3)
    sdh31 = SubDofHandler(dh3, Set([1, 2]))
    add!(sdh31, :u, ip)
    sdh32 = SubDofHandler(dh3, Set(3))
    add!(sdh32, :u, ip)
    close!(dh3)
    ics = InterfaceCache(sdh31)
    reinit!(ics, FacetIndex(1, 2), FacetIndex(2, 4)) # interface within sdh31's cellset
    @test interfacedofs(ics) == [celldofs(dh3, 1); celldofs(dh3, 2)]
    r = interface_dof_range(ics, :u)
    @test r.here == dof_range(sdh31, :u)
    @test r.there == dof_range(sdh31, :u) .+ ndofs_per_cell(sdh31)

    # DofHandler not covering all cells: reject dof_range for interfaces into the
    # uncovered subdomain
    dh5 = DofHandler(grid3)
    sdh5 = SubDofHandler(dh5, Set(1)) # cells 2, 3 uncovered
    add!(sdh5, :u, ip)
    close!(dh5)
    ic5 = InterfaceCache(dh5)
    reinit!(ic5, FacetIndex(1, 2), FacetIndex(2, 4))
    err = try
        interface_dof_range(ic5, :u); nothing
    catch e
        e
    end
    @test err isa ErrorException
    @test occursin("not defined on both cells", err.msg)
end

@testset "condense_duplicates!" begin
    # Mixed continuous/discontinuous DofHandler: interfaces with shared u dofs, exercised
    # across two SubDofHandlers of different sizes (the buffer must grow and shrink)
    grid = generate_grid(Quadrilateral, (3, 1))
    topology = ExclusiveTopology(grid)
    dh = DofHandler(grid)
    big = SubDofHandler(dh, Set([1, 2]))
    add!(big, :u, Lagrange{RefQuadrilateral, 2}())
    add!(big, :P, DiscontinuousLagrange{RefQuadrilateral, 1}())
    small = SubDofHandler(dh, Set(3))
    add!(small, :u, Lagrange{RefQuadrilateral, 2}())
    close!(dh)
    nd = ndofs(dh)
    K = allocate_matrix(dh; topology = topology, interface_coupling = trues(2, 2))
    f = zeros(nd)
    buf = CondensationBuffer{Float64}(2 * maximum(ndofs_per_cell, dh.subdofhandlers))
    # Congruence transform against the dense reference of the stacked scatter, for
    # arbitrary nonsymmetric input (two passes: sizes grow and shrink between interfaces)
    for atomic in (false, true)
        assembler = start_assemble(K, f; atomic)
        Kref = zeros(nd, nd)
        fref = zeros(nd)
        for pass in 1:2, ic in InterfaceIterator(dh, topology)
            ns = ndofs_per_interface(ic)
            Ke = rand(ns, ns)
            fe = rand(ns)
            udofs, Kc, fc = condense_duplicates!(buf, interfacedofs(ic), Ke, fe)
            udofs_ref, T = duplication_map(interfacedofs(ic))
            @test udofs == udofs_ref # unique dofs in first-occurrence order
            @test length(udofs) < ns # u is continuous: dofs are indeed repeated
            @test Matrix(Kc) ≈ T' * Ke * T
            @test collect(fc) ≈ T' * fe
            assemble!(assembler, udofs, Kc, fc)
            dense_scatter!(Kref, interfacedofs(ic), Ke, fref, fe)
        end
        @test Matrix(K) ≈ Kref
        @test f ≈ fref
    end

    ic1 = first(InterfaceIterator(dh, topology))
    n1 = ndofs_per_interface(ic1)
    # Matrix-only and vector-only methods
    let Ke = rand(n1, n1), fe = rand(n1)
        _, T = duplication_map(interfacedofs(ic1))
        udofs2, Kc2 = condense_duplicates!(buf, interfacedofs(ic1), Ke)
        @test Matrix(Kc2) ≈ T' * Ke * T
        udofs3, fc3 = condense_duplicates!(buf, interfacedofs(ic1), fe)
        @test collect(fc3) ≈ T' * fe
        @test udofs2 == udofs3
        # Symmetric stacked input gives symmetric condensed output (congruence transform)
        Ks = Ke + Ke'
        _, Kcs = condense_duplicates!(buf, interfacedofs(ic1), Ks)
        @test Matrix(Kcs) ≈ Matrix(Kcs)'
    end
    # Dimension checks
    @test_throws DimensionMismatch condense_duplicates!(buf, interfacedofs(ic1), rand(n1 + 1, n1 + 1))
    @test_throws DimensionMismatch condense_duplicates!(buf, interfacedofs(ic1), rand(n1, n1), rand(n1 + 1))
    @test_throws DimensionMismatch condense_duplicates!(buf, interfacedofs(ic1), rand(n1 + 1))
    # Aliasing: views of the buffer's storage (e.g. outputs of a previous call) must not
    # be passed back as input, even when the sizes happen to match
    @test_throws ArgumentError condense_duplicates!(buf, interfacedofs(ic1), view(buf.Ke, 1:n1, 1:n1))
    @test_throws ArgumentError condense_duplicates!(buf, interfacedofs(ic1), rand(n1, n1), view(buf.fe, 1:n1))
    @test_throws ArgumentError condense_duplicates!(buf, interfacedofs(ic1), view(buf.fe, 1:n1))

    # No shared dofs: the map is the identity and the condensation is a copy into the
    # buffer (not a pass-through of the inputs, so that the return types do not depend on
    # the values of the input data)
    dhdg = DofHandler(grid)
    add!(dhdg, :u, DiscontinuousLagrange{RefQuadrilateral, 1}())
    close!(dhdg)
    icdg = first(InterfaceIterator(dhdg, topology))
    let Ke = rand(8, 8), fe = rand(8)
        udofs, Kc, fc = condense_duplicates!(buf, interfacedofs(icdg), Ke, fe)
        @test udofs == interfacedofs(icdg)
        @test Kc == Ke && Kc !== Ke
        @test fc == fe && fc !== fe
    end
    # Type stability: the same return types for the shared and the no-sharing case, also
    # when the input is a view (whose type would leak into the return under pass-through)
    let
        Kview = view(zeros(n1 + 5, n1 + 5), 1:n1, 1:n1)
        fview = view(zeros(n1 + 5), 1:n1)
        T_shared = typeof(@inferred condense_duplicates!(buf, interfacedofs(ic1), Kview, fview))
        Kviewdg = view(zeros(20, 20), 1:8, 1:8)
        fviewdg = view(zeros(20), 1:8)
        T_dg = typeof(@inferred condense_duplicates!(buf, interfacedofs(icdg), Kviewdg, fviewdg))
        @test T_shared === T_dg
        @test isconcretetype(T_shared)
        Tv = typeof(@inferred condense_duplicates!(buf, interfacedofs(ic1), fview))
        @test isconcretetype(Tv)
    end
    # An undersized buffer grows as needed
    let smallbuf = CondensationBuffer{Float64}(), Ke = rand(n1, n1)
        _, T = duplication_map(interfacedofs(ic1))
        _, Kc = condense_duplicates!(smallbuf, interfacedofs(ic1), Ke)
        @test Matrix(Kc) ≈ T' * Ke * T
    end

    # The raw stacked scatter is rejected deterministically (independent of the storage
    # pattern) with an error pointing at condense_duplicates!
    err = try
        assemble!(start_assemble(K, f), interfacedofs(ic1), rand(n1, n1))
        nothing
    catch e
        e
    end
    @test err isa ArgumentError
    @test occursin("repeated entries", err.msg)
    @test occursin("condense_duplicates!", err.msg)
    # ... also for the symmetric assembler and for rectangular row/column dofs
    dh1 = DofHandler(generate_grid(Line, (4,)))
    add!(dh1, :u, Lagrange{RefLine, 1}())
    close!(dh1)
    topo1 = ExclusiveTopology(dh1.grid)
    Ksym = Symmetric(allocate_matrix(dh1; topology = topo1, interface_coupling = trues(1, 1)))
    ic4 = first(InterfaceIterator(dh1, topo1))
    @test interfacedofs(ic4)[2] == interfacedofs(ic4)[3] # P1 on Line: dofs [d0, D, D, d2]
    @test_throws ArgumentError assemble!(start_assemble(Ksym), interfacedofs(ic4), ones(4, 4))
    let Kr = spzeros(5, 5)
        @test_throws ArgumentError assemble!(start_assemble(Kr), [3, 1, 3], [2, 4, 1], zeros(3, 3))
        @test_throws ArgumentError assemble!(start_assemble(Kr), [3, 1, 4], [2, 4, 2], zeros(3, 3))
    end

    # COO and dense-vector assembly sum duplicates natively: raw stacked assembly must
    # equal the explicit congruence transform
    buf1 = CondensationBuffer{Float64}(ndofs_per_interface(dh1))
    K1 = allocate_matrix(dh1; topology = topo1, interface_coupling = trues(1, 1))
    f1 = zeros(ndofs(dh1))
    assembler = start_assemble(K1, f1)
    Kref = zeros(ndofs(dh1), ndofs(dh1))
    fref = zeros(ndofs(dh1))
    coo = Ferrite.COOAssembler()
    fraw = zeros(ndofs(dh1))
    for ic in InterfaceIterator(dh1, topo1)
        # #1433 MWE (4 cells to reach the sparse merge walk; 2 cells only exercises the
        # dense-column fast path): condensed all-ones gives the congruence transform
        udofs, Kc, fc = condense_duplicates!(buf1, interfacedofs(ic), ones(4, 4), ones(4))
        assemble!(assembler, udofs, Kc, fc)
        dense_scatter!(Kref, interfacedofs(ic), ones(4, 4), fref, ones(4))
        assemble!(coo, interfacedofs(ic), ones(4, 4))
        assemble!(fraw, interfacedofs(ic), ones(4))
    end
    @test Matrix(K1) ≈ Kref
    @test f1 ≈ fref
    Kcoo, _ = finish_assemble(coo)
    @test Matrix(Kcoo) ≈ Kref
    @test fraw ≈ fref
    # TᵀKeT of all-ones has 4 at the shared position (storage semantics -- a kernel that
    # *means* K[D, D] == 1 must weight the two copies accordingly, see the weak-form test)
    _, Kc1 = condense_duplicates!(buf1, interfacedofs(ic4), ones(4, 4))
    @test Matrix(Kc1) == [1.0 2.0 1.0; 2.0 4.0 2.0; 1.0 2.0 1.0]

    # Symmetric assembler: condense the full stacked matrix of the symmetric operator,
    # then assemble the condensed (duplicate-free) system as usual
    for atomic in (false, true)
        Ks = Symmetric(allocate_matrix(dh1; topology = topo1, interface_coupling = trues(1, 1)))
        assembler_s = start_assemble(Ks; atomic)
        Krefs = zeros(ndofs(dh1), ndofs(dh1))
        for ic in InterfaceIterator(dh1, topo1)
            n = ndofs_per_interface(ic)
            Kfull = Matrix(Symmetric(rand(n, n))) # the symmetric operator, fully filled
            udofs, Kc = condense_duplicates!(buf1, interfacedofs(ic), Kfull)
            assemble!(assembler_s, udofs, Kc)
            dense_scatter!(Krefs, interfacedofs(ic), Kfull)
        end
        @test Matrix(Ks) ≈ Krefs
    end
end

@testset "Interface term only for the discontinuous field next to a continuous field" begin
    # Common mixed case: continuous u and discontinuous P where only P has interface
    # terms (u appears only in cell integrals). The interface_coupling mask then only
    # needs the P-P block: the u rows/columns of the stacked local matrix are structural
    # zeros (the kernel never writes them), their condensed values are exact zeros, and
    # the assembler's zero-skip tolerates the missing pattern entries.
    grid = generate_grid(Quadrilateral, (3, 1))
    topology = ExclusiveTopology(grid)
    dh = DofHandler(grid)
    add!(dh, :u, Lagrange{RefQuadrilateral, 1}())
    add!(dh, :P, DiscontinuousLagrange{RefQuadrilateral, 1}())
    close!(dh)
    iv_P = InterfaceValues(FacetQuadratureRule{RefQuadrilateral}(2), DiscontinuousLagrange{RefQuadrilateral, 1}())
    μ = 2.0
    function P_penalty_kernel!(Ke, ic)
        rP = interface_dof_range(ic, :P)
        for qp in 1:getnquadpoints(iv_P)
            dΓ = getdetJdV(iv_P, qp)
            for (i, I) in pairs(rP)
                δP_jump = shape_value_jump(iv_P, qp, i)
                for (j, J) in pairs(rP)
                    Ke[I, J] += μ * δP_jump * shape_value_jump(iv_P, qp, j) * dΓ
                end
            end
        end
        return
    end
    cell_coupling = [true false; false true]
    interface_coupling = [false false; false true]
    K = allocate_matrix(dh; coupling = cell_coupling, topology = topology, interface_coupling = interface_coupling)
    assembler = start_assemble(K)
    buf = CondensationBuffer{Float64}(ndofs_per_interface(dh))
    Kref = zeros(ndofs(dh), ndofs(dh))
    for ic in InterfaceIterator(dh, topology)
        reinit!(iv_P, ic)
        n = ndofs_per_interface(ic)
        Ke = zeros(n, n)
        P_penalty_kernel!(Ke, ic)
        # The u dofs are shared even though the form never touches them: condensation is
        # still required (the stacked dof vector contains duplicates), and the condensed
        # u rows/columns are structural zeros that the assembler's zero-skip tolerates
        # although the pattern has no interface entries for u
        udofs, Kc = condense_duplicates!(buf, interfacedofs(ic), Ke)
        @test length(udofs) < n
        upos = unique(indexin(interfacedofs(ic)[interface_dof_range(ic, :u)], udofs))
        @test all(iszero, Kc[upos, :])
        @test all(iszero, Kc[:, upos])
        assemble!(assembler, udofs, Kc)
        dense_scatter!(Kref, interfacedofs(ic), Ke)
    end
    @test Matrix(K) ≈ Kref
    # Negative control: the P-only mask is load-bearing. A form that couples P to the
    # continuous u produces nonzero condensed values at positions this pattern does not
    # have, and assembly reports the missing entry.
    iv_u = InterfaceValues(FacetQuadratureRule{RefQuadrilateral}(2), Lagrange{RefQuadrilateral, 1}())
    assembler2 = start_assemble(K)
    ic = first(InterfaceIterator(dh, topology))
    reinit!(iv_P, ic)
    reinit!(iv_u, ic)
    n = ndofs_per_interface(ic)
    Ke = zeros(n, n)
    P_penalty_kernel!(Ke, ic)
    rP = interface_dof_range(ic, :P)
    ru = interface_dof_range(ic, :u)
    for qp in 1:getnquadpoints(iv_P)
        dΓ = getdetJdV(iv_P, qp)
        for (i, I) in pairs(rP), (j, J) in pairs(ru)
            Ke[I, J] += shape_value_jump(iv_P, qp, i) * shape_value_average(iv_u, qp, j) * dΓ
        end
    end
    udofs, Kc = condense_duplicates!(buf, interfacedofs(ic), Ke)
    err = try
        assemble!(assembler2, udofs, Kc)
        nothing
    catch e
        e
    end
    @test err isa ErrorException
    @test occursin("missing in the sparsity pattern", err.msg)
end

@testset "Mixed continuous u / discontinuous P interface form" begin
    # Assemble B((δP, δu), (P, u)) = ∫ μ [[δP]][[P]] + [[δP]]{{u}} + [[P]]{{δu}} dΓ and
    # verify vᵀ K w against direct evaluation of the form with function_value_jump/average.
    # For the continuous field u the average weighting is what makes the duplicated basis
    # correct: each copy of a shared dof gets weight 1/2 (summing to the single-valued
    # trace); raw side values summed over both copies double-count.
    grid = generate_grid(Quadrilateral, (2, 2))
    topology = ExclusiveTopology(grid)
    ip_u = Lagrange{RefQuadrilateral, 1}()
    ip_P = DiscontinuousLagrange{RefQuadrilateral, 1}()
    dh = DofHandler(grid)
    add!(dh, :u, ip_u)
    add!(dh, :P, ip_P)
    close!(dh)
    qr_facet = FacetQuadratureRule{RefQuadrilateral}(2)
    iv_u = InterfaceValues(qr_facet, ip_u)
    iv_P = InterfaceValues(qr_facet, ip_P)
    μ = 3.0

    # NOTE: all locals are deliberately named distinctly from the enclosing testset
    # variables: this is a closure, and assigning to a name that also exists in the
    # enclosing scope would rebind the *shared* variable.
    function assemble_interface_matrix(kernel!::F) where {F}
        Kmat = allocate_matrix(dh; topology = topology, interface_coupling = trues(2, 2))
        asm = start_assemble(Kmat)
        cbuf = CondensationBuffer{Float64}(ndofs_per_interface(dh))
        Kstore = zeros(ndofs_per_interface(dh), ndofs_per_interface(dh))
        for ic_ in InterfaceIterator(dh, topology)
            reinit!(iv_u, ic_)
            reinit!(iv_P, ic_)
            n_ = ndofs_per_interface(ic_)
            Ke_ = @view Kstore[1:n_, 1:n_]
            fill!(Ke_, 0)
            kernel!(Ke_, ic_)
            udofs_, Kc_ = condense_duplicates!(cbuf, interfacedofs(ic_), Ke_)
            assemble!(asm, udofs_, Kc_)
        end
        return Kmat
    end

    # Correct kernel: average (trace) weighting for the continuous field
    function correct_kernel!(Ke, ic)
        rP = interface_dof_range(ic, :P)
        ru = interface_dof_range(ic, :u)
        for qp in 1:getnquadpoints(iv_P)
            dΓ = getdetJdV(iv_P, qp)
            for (i, I) in pairs(rP)
                δP_jump = shape_value_jump(iv_P, qp, i)
                for (j, J) in pairs(rP)
                    Ke[I, J] += μ * δP_jump * shape_value_jump(iv_P, qp, j) * dΓ
                end
                for (j, J) in pairs(ru)
                    uj = shape_value_average(iv_u, qp, j)
                    Ke[I, J] += δP_jump * uj * dΓ
                    Ke[J, I] += δP_jump * uj * dΓ
                end
            end
        end
        return
    end
    # Wrong kernel: sums the raw side values of the continuous field (weight 2 for the
    # copies of a shared dof). This assembles without error — condensation preserves the
    # (wrong) local form, it does not repair it.
    function wrong_kernel!(Ke, ic)
        rP = interface_dof_range(ic, :P)
        ru = interface_dof_range(ic, :u)
        for qp in 1:getnquadpoints(iv_P)
            dΓ = getdetJdV(iv_P, qp)
            for (i, I) in pairs(rP)
                δP_jump = shape_value_jump(iv_P, qp, i)
                for (j, J) in pairs(rP)
                    Ke[I, J] += μ * δP_jump * shape_value_jump(iv_P, qp, j) * dΓ
                end
                for (j, J) in pairs(ru)
                    uj = shape_value(iv_u, qp, j; here = true) + shape_value(iv_u, qp, j; here = false)
                    Ke[I, J] += δP_jump * uj * dΓ
                    Ke[J, I] += δP_jump * uj * dΓ
                end
            end
        end
        return
    end

    K = assemble_interface_matrix(correct_kernel!)
    Kwrong = assemble_interface_matrix(wrong_kernel!)

    # Direct evaluation of B(v, w) by quadrature over the interfaces
    function direct_form(v, w)
        B = 0.0
        for ic_ in InterfaceIterator(dh, topology)
            reinit!(iv_u, ic_)
            reinit!(iv_P, ic_)
            rP = interface_dof_range(ic_, :P)
            ru = interface_dof_range(ic_, :u)
            ve = v[interfacedofs(ic_)]
            we = w[interfacedofs(ic_)]
            for qp in 1:getnquadpoints(iv_P)
                dΓ = getdetJdV(iv_P, qp)
                vP_jump = function_value_jump(iv_P, qp, ve, rP)
                wP_jump = function_value_jump(iv_P, qp, we, rP)
                # u is continuous: the average IS the single-valued value on the interface
                wu = function_value_average(iv_u, qp, we, ru)
                vu = function_value_average(iv_u, qp, ve, ru)
                B += (μ * vP_jump * wP_jump + vP_jump * wu + wP_jump * vu) * dΓ
            end
        end
        return B
    end

    v = rand(ndofs(dh))
    w = rand(ndofs(dh))
    @test dot(v, K, w) ≈ direct_form(v, w)
    # The wrong kernel gives a different operator (double-counted u coupling)
    @test !(dot(v, Kwrong, w) ≈ direct_form(v, w))

    # Constrained assembly: apply_assemble! on the condensed system must be equivalent
    # to apply_local! + assemble! (the condensed dofs are unique and the matrix square,
    # so the existing constraint machinery applies unchanged)
    ic = first(InterfaceIterator(dh, topology))
    reinit!(iv_u, ic)
    reinit!(iv_P, ic)
    sd = interfacedofs(ic)
    shared_u = [d for d in sd if count(==(d), sd) > 1]
    ch = ConstraintHandler(dh)
    add!(ch, AffineConstraint(shared_u[1], [shared_u[end] => 0.5], 1.0))
    close!(ch)
    n = ndofs_per_interface(ic)
    Ke = rand(n, n)
    fe = rand(n)
    buf = CondensationBuffer{Float64}(ndofs_per_interface(dh))
    udofs, Kc, fc = condense_duplicates!(buf, interfacedofs(ic), Ke, fe)
    Kc_copy = Matrix(Kc)
    fc_copy = collect(fc)
    K1 = allocate_matrix(dh, ch; topology = topology, interface_coupling = trues(2, 2))
    f1 = zeros(ndofs(dh))
    K2 = copy(K1)
    f2 = zeros(ndofs(dh))
    asm1 = start_assemble(K1, f1)
    apply_assemble!(asm1, ch, udofs, Kc, fc) # note: destructive on the buffer contents
    asm2 = start_assemble(K2, f2)
    apply_local!(Kc_copy, fc_copy, udofs, ch)
    assemble!(asm2, udofs, Kc_copy, fc_copy)
    @test Matrix(K1) ≈ Matrix(K2)
    @test f1 ≈ f2
end

@testset "interface_dof_range in the assembly loop across SubDofHandlers" begin
    # The two cells of an interface may belong to different SubDofHandlers with
    # different field orders; interface_dof_range(ic, field) then places the field's dofs
    # correctly in the stacked local matrix on both sides. Assemble the mixed form from
    # the previous testset over a grid where one subdomain has the field order swapped,
    # with the kernel indexing exclusively through dof_range, and verify the operator
    # against direct evaluation of the form by quadrature.
    grid = generate_grid(Quadrilateral, (3, 1))
    topology = ExclusiveTopology(grid)
    ip_u = Lagrange{RefQuadrilateral, 1}()
    ip_P = DiscontinuousLagrange{RefQuadrilateral, 1}()
    dh = DofHandler(grid)
    sdh1 = SubDofHandler(dh, Set([1, 2]))
    add!(sdh1, :u, ip_u)
    add!(sdh1, :P, ip_P)
    sdh2 = SubDofHandler(dh, Set(3))
    add!(sdh2, :P, ip_P) # field order swapped relative to sdh1
    add!(sdh2, :u, ip_u)
    close!(dh)
    qr_facet = FacetQuadratureRule{RefQuadrilateral}(2)
    iv_u = InterfaceValues(qr_facet, ip_u)
    iv_P = InterfaceValues(qr_facet, ip_P)
    μ = 3.0
    # The cross-subdomain interface really has different field layouts on the two sides
    ic23 = InterfaceCache(dh)
    reinit!(ic23, FacetIndex(2, 2), FacetIndex(3, 4))
    @test interface_dof_range(ic23, :u).here != interface_dof_range(ic23, :u).there .- ndofs_per_cell(sdh1)
    K = allocate_matrix(dh; topology = topology, interface_coupling = trues(2, 2))
    asm = start_assemble(K)
    buf = CondensationBuffer{Float64}() # grows on demand (sizes vary across SubDofHandlers)
    nif = 0
    for ic in InterfaceIterator(dh, topology)
        nif += 1
        reinit!(iv_u, ic)
        reinit!(iv_P, ic)
        n = ndofs_per_interface(ic)
        Ke = zeros(n, n)
        rP = interface_dof_range(ic, :P)
        ru = interface_dof_range(ic, :u)
        for qp in 1:getnquadpoints(iv_P)
            dΓ = getdetJdV(iv_P, qp)
            for (i, I) in pairs(rP)
                δP_jump = shape_value_jump(iv_P, qp, i)
                for (j, J) in pairs(rP)
                    Ke[I, J] += μ * δP_jump * shape_value_jump(iv_P, qp, j) * dΓ
                end
                for (j, J) in pairs(ru)
                    uj = shape_value_average(iv_u, qp, j)
                    Ke[I, J] += δP_jump * uj * dΓ
                    Ke[J, I] += δP_jump * uj * dΓ
                end
            end
        end
        udofs, Kc = condense_duplicates!(buf, interfacedofs(ic), Ke)
        assemble!(asm, udofs, Kc)
    end
    @test nif == 2 # one interface within sdh1, one across sdh1/sdh2
    # Direct evaluation of the form by quadrature over the interfaces
    v = rand(ndofs(dh))
    w = rand(ndofs(dh))
    B = 0.0
    for ic in InterfaceIterator(dh, topology)
        reinit!(iv_u, ic)
        reinit!(iv_P, ic)
        rP = interface_dof_range(ic, :P)
        ru = interface_dof_range(ic, :u)
        ve = v[interfacedofs(ic)]
        we = w[interfacedofs(ic)]
        for qp in 1:getnquadpoints(iv_P)
            dΓ = getdetJdV(iv_P, qp)
            vP_jump = function_value_jump(iv_P, qp, ve, rP)
            wP_jump = function_value_jump(iv_P, qp, we, rP)
            wu = function_value_average(iv_u, qp, we, ru)
            vu = function_value_average(iv_u, qp, ve, ru)
            B += (μ * vP_jump * wP_jump + vP_jump * wu + wP_jump * vu) * dΓ
        end
    end
    @test abs(B) > 0 # nondegenerate operator
    @test dot(v, K, w) ≈ B
end

@testset "AD: stacked residual -> assembled Jacobian" begin
    # Differentiating a stacked-coefficient interface residual gives a stacked Jacobian
    # with the shared dofs appearing twice; condensing it with condense_duplicates!
    # yields Tᵀ Js T -- the correct total derivative with respect to the unique dofs by
    # the chain rule through the gather u_s = T u_u.
    grid = generate_grid(Quadrilateral, (2, 1))
    topology = ExclusiveTopology(grid)
    ip_u = Lagrange{RefQuadrilateral, 1}()
    dh = DofHandler(grid)
    add!(dh, :u, ip_u)
    close!(dh)
    qr_facet = FacetQuadratureRule{RefQuadrilateral}(2)
    iv_u = InterfaceValues(qr_facet, ip_u)
    ic = first(InterfaceIterator(dh, topology))
    reinit!(iv_u, ic)
    ru = interface_dof_range(ic, :u)
    # A (nonlinear) interface residual in the stacked layout: r_i = ∫ {{δu}}_i {{u}}^2 dΓ
    function stacked_residual(ue)
        re = zeros(eltype(ue), length(ue))
        for qp in 1:getnquadpoints(iv_u)
            dΓ = getdetJdV(iv_u, qp)
            uval = function_value_average(iv_u, qp, ue, ru)
            for (i, I) in pairs(ru)
                re[I] += shape_value_average(iv_u, qp, i) * uval^2 * dΓ
            end
        end
        return re
    end
    a = rand(ndofs(dh))
    sd = interfacedofs(ic)
    ns = ndofs_per_interface(ic)
    udofs, T = duplication_map(sd)
    nu = length(udofs)
    # Differentiate the stacked residual at the consistent point (both copies of a shared
    # dof gathered from the same global value)
    ue_stacked = a[sd]
    Js = ForwardDiff.jacobian(stacked_residual, ue_stacked)
    fs = stacked_residual(ue_stacked)
    # The condensed Jacobian equals differentiation with respect to the unique
    # coefficients through the gather
    unique_residual(uuc) = T' * stacked_residual(T * uuc)
    uu = a[udofs]
    Ju = ForwardDiff.jacobian(unique_residual, uu)
    @test T' * Js * T ≈ Ju
    # Independent arbiter (both routes above go through ForwardDiff): central finite
    # differences. In particular this rules out double counting of the shared dofs, which
    # appear twice in the stacked input: the total derivative with respect to a unique dof
    # is the *sum* of the two copies' partials, a double count would show up as a factor 2
    # in the shared-dof rows/columns.
    let h = 1.0e-6, Jfd = zeros(nu, nu)
        for j in 1:nu
            up = copy(uu); up[j] += h
            um = copy(uu); um[j] -= h
            Jfd[:, j] = (unique_residual(up) - unique_residual(um)) / (2h)
        end
        @test isapprox(Ju, Jfd; rtol = 1.0e-5)
        for j in 1:nu
            count(==(udofs[j]), sd) > 1 || continue
            @test norm(Ju[:, j]) ≈ norm(Jfd[:, j]) rtol = 1.0e-5 # ratio 1, not 2
        end
    end
    # Condensing the stacked (Js, fs) with a (Dual-free) buffer and assembling deposits
    # exactly (Tᵀ Js T, Tᵀ fs) at the unique dofs
    buf = CondensationBuffer{Float64}(ns)
    udofs_c, Jc, fc = condense_duplicates!(buf, interfacedofs(ic), Js, fs)
    @test udofs_c == udofs
    @test Matrix(Jc) ≈ Ju
    K = allocate_matrix(dh; topology = topology, interface_coupling = trues(1, 1))
    f = zeros(ndofs(dh))
    assembler = start_assemble(K, f)
    assemble!(assembler, udofs_c, Jc, fc)
    @test Matrix(K)[udofs, udofs] ≈ Ju
    @test f[udofs] ≈ T' * fs

    # A Dual-typed buffer works too (the buffer eltype is the user's choice, matching the
    # local matrix eltype). Condensation is linear, so it acts on values and partials alike.
    DualT = ForwardDiff.Dual{Nothing, Float64, 1}
    Jdual = [DualT(Js[i, j], ForwardDiff.Partials((2 * Js[i, j],))) for i in 1:ns, j in 1:ns]
    bufdual = CondensationBuffer{DualT}(ns)
    _, JcD = condense_duplicates!(bufdual, interfacedofs(ic), Jdual)
    @test ForwardDiff.value.(Matrix(JcD)) ≈ Ju
    @test map(d -> ForwardDiff.partials(d)[1], Matrix(JcD)) ≈ 2 .* Ju

    # Alternative AD route (the recommended one is to condense the stacked Jacobian, as
    # above): the residual function itself condenses through the vector-only method with a
    # dual-typed buffer and is differentiated with respect to the *unique* coefficients
    # (gathering the stacked coefficients through the duplication map). This exercises
    # dual numbers through the vector condensation path; the result must equal Tᵀ Js T.
    function condensed_residual(uuc)
        ue2 = T * uuc                                             # gather: u_s = T * u_u
        re = stacked_residual(ue2)                                # dual-valued stacked residual
        bufD = CondensationBuffer{eltype(re)}(length(re))    # dual-typed buffer
        _, fcD = condense_duplicates!(bufD, interfacedofs(ic), re)                # fc = Tᵀ re
        return copy(fcD)                                          # copy out of the buffer
    end
    @test condensed_residual(uu) ≈ T' * fs
    @test ForwardDiff.jacobian(condensed_residual, uu) ≈ Ju
end
