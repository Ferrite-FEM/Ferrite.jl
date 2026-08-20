# Tests for interface assembly with dofs shared between the neighboring cells (#1433):
# the InterfaceCache stacked/unique dof map, dof_range(::InterfaceCache),
# condense_interface!, and assembly of interface terms for continuous fields.
using Ferrite
using Test
using LinearAlgebra
using SparseArrays
import ForwardDiff

# The stacked -> unique map as an explicit matrix (T in the design doc: the map from
# unique to stacked coefficients, u_stacked = T * u_unique)
function duplication_map(ic::InterfaceCache)
    ns = nstacked_interface_dofs(ic)
    nu = nunique_interface_dofs(ic)
    T = zeros(ns, nu)
    for i in 1:ns
        T[i, ic.stacked_to_unique[i]] = 1.0
    end
    return T
end

@testset "InterfaceCache stacked/unique dof map" begin
    function check_map(ip, grid)
        dh = DofHandler(grid)
        add!(dh, :u, ip)
        close!(dh)
        topology = ExclusiveTopology(grid)
        ic = first(InterfaceIterator(dh, topology))
        sd = interfacedofs(ic)
        ud = unique_interfacedofs(ic)
        n_a = length(celldofs(ic.a))
        # Stacked layout is the concatenation
        @test sd == [celldofs(ic.a); celldofs(ic.b)]
        # Unique dofs in first-occurrence order
        @test ud == unique(sd)
        # Sizes
        @test nstacked_interface_dofs(ic) == length(sd)
        @test nunique_interface_dofs(ic) == length(ud)
        @test nstacked_interface_dofs(ic) <= max_nstacked_interface_dofs(dh)
        # The map reproduces the stacked dofs and has an identity prefix for the here side
        @test [ud[ic.stacked_to_unique[i]] for i in eachindex(sd)] == sd
        @test ic.stacked_to_unique[1:n_a] == 1:n_a
        # is_shared agrees with the map
        for i in eachindex(sd)
            @test is_shared(ic, i) == (count(==(sd[i]), sd) > 1)
        end
        return ic
    end
    grid_quad = generate_grid(Quadrilateral, (2, 1))
    grid_tri = generate_grid(Triangle, (2, 1))
    # H1: shares facet (and vertex) dofs
    ic = check_map(Lagrange{RefQuadrilateral, 2}(), grid_quad)
    @test ic.any_shared
    # L2 (cell-interior dofs): nothing shared
    ic = check_map(DiscontinuousLagrange{RefQuadrilateral, 1}(), grid_quad)
    @test !ic.any_shared
    @test unique_interfacedofs(ic) == interfacedofs(ic)
    # Nonconforming interpolations are classified L2Conformity but place dofs on facets,
    # which are shared: dof sharing follows dof placement, not conformity, so these must
    # be detected as sharing (this is the reason the map is built from actual global dof
    # equality and not gated on `conformity`)
    ic = check_map(CrouzeixRaviart{RefTriangle, 1}(), grid_tri)
    @test Ferrite.conformity(CrouzeixRaviart{RefTriangle, 1}()) isa Ferrite.L2Conformity
    @test ic.any_shared
    ic = check_map(RannacherTurek{RefQuadrilateral, 1}(), grid_quad)
    @test Ferrite.conformity(RannacherTurek{RefQuadrilateral, 1}()) isa Ferrite.L2Conformity
    @test ic.any_shared
    # H(div) and H(curl): share facet/edge dofs
    ic = check_map(RaviartThomas{RefTriangle, 1}(), grid_tri)
    @test ic.any_shared
    ic = check_map(Nedelec{RefTriangle, 1}(), grid_tri)
    @test ic.any_shared
    # Vector-valued H1
    ic = check_map(Lagrange{RefQuadrilateral, 1}()^2, grid_quad)
    @test ic.any_shared
end

@testset "dof_range(::InterfaceCache) and InterfaceDofRange" begin
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
        r = dof_range(ic, field)
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
    end
    # function_* evaluation with an InterfaceDofRange equals the two-range methods
    iv_u = InterfaceValues(FacetQuadratureRule{RefQuadrilateral}(2), Lagrange{RefQuadrilateral, 1}())
    reinit!(iv_u, ic)
    a = rand(ndofs(dh))
    ue = a[interfacedofs(ic)]
    ru = dof_range(ic, :u)
    for qp in 1:getnquadpoints(iv_u)
        @test function_value(iv_u, qp, ue, ru; here = true) ==
            function_value(iv_u, qp, ue, ru.here, ru.there; here = true)
        @test function_value_jump(iv_u, qp, ue, ru) ==
            function_value_jump(iv_u, qp, ue, ru.here, ru.there)
        @test function_value_average(iv_u, qp, ue, ru) ==
            function_value_average(iv_u, qp, ue, ru.here, ru.there)
        @test function_gradient_jump(iv_u, qp, ue, ru) ==
            function_gradient_jump(iv_u, qp, ue, ru.here, ru.there)
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
    dh2_sdh_a = dh2.subdofhandlers[ic2.sdh_index_a]
    dh2_sdh_b = dh2.subdofhandlers[ic2.sdh_index_b]
    for field in (:u, :p)
        r = dof_range(ic2, field)
        @test r.here == dof_range(dh2_sdh_a, field)
        @test r.there == dof_range(dh2_sdh_b, field) .+ ndofs_per_cell(dh2_sdh_a)
        @test interfacedofs(ic2)[r] ==
            [celldofs(ic2.a)[dof_range(dh2_sdh_a, field)]; celldofs(ic2.b)[dof_range(dh2_sdh_b, field)]]
    end
    # A field present on only one side is rejected
    @test_throws ErrorException dof_range(ic2, :q)
    err = try dof_range(ic2, :q); nothing; catch e; e; end
    @test occursin("only one side", err.msg)
    # Unknown fields are rejected too
    @test_throws ErrorException dof_range(ic2, :nonexistent)
    # Grid-constructed caches have no dof information
    ic3 = InterfaceCache(grid)
    reinit!(ic3, FacetIndex(1, 2), FacetIndex(2, 4))
    @test_throws ErrorException dof_range(ic3, :u)
end

@testset "condense_interface!" begin
    grid = generate_grid(Quadrilateral, (2, 1))
    topology = ExclusiveTopology(grid)
    dh = DofHandler(grid)
    add!(dh, :u, Lagrange{RefQuadrilateral, 2}())
    add!(dh, :P, DiscontinuousLagrange{RefQuadrilateral, 1}())
    close!(dh)
    buf = InterfaceAssemblyBuffer{Float64}(max_nstacked_interface_dofs(dh))
    for ic in InterfaceIterator(dh, topology)
        ns = nstacked_interface_dofs(ic)
        nu = nunique_interface_dofs(ic)
        @test nu < ns # u is continuous
        # Congruence transform for an arbitrary nonsymmetric matrix/vector
        Ke = rand(ns, ns)
        fe = rand(ns)
        udofs, Kc, fc = condense_interface!(buf, ic, Ke, fe)
        T = duplication_map(ic)
        @test udofs == unique_interfacedofs(ic)
        @test size(Kc) == (nu, nu)
        @test Matrix(Kc) ≈ T' * Ke * T
        @test collect(fc) ≈ T' * fe
        # Matrix-only method
        udofs2, Kc2 = condense_interface!(buf, ic, Ke)
        @test Matrix(Kc2) ≈ T' * Ke * T
        # Symmetric stacked input gives symmetric condensed output (congruence transform)
        Ks = Ke + Ke'
        _, Kcs = condense_interface!(buf, ic, Ks)
        @test Matrix(Kcs) ≈ Matrix(Kcs)'
        # Dimension checks
        @test_throws DimensionMismatch condense_interface!(buf, ic, rand(ns + 1, ns + 1))
        @test_throws DimensionMismatch condense_interface!(buf, ic, Ke, rand(ns + 1))
    end
    # Pass-through (no copy) when nothing is shared
    dhdg = DofHandler(grid)
    add!(dhdg, :u, DiscontinuousLagrange{RefQuadrilateral, 1}())
    close!(dhdg)
    icdg = first(InterfaceIterator(dhdg, topology))
    Ke = rand(8, 8)
    fe = rand(8)
    udofs, Kc, fc = condense_interface!(buf, icdg, Ke, fe)
    @test Kc === Ke
    @test fc === fe
    @test udofs === interfacedofs(icdg)
    # An undersized buffer grows as needed
    smallbuf = InterfaceAssemblyBuffer{Float64}()
    dhh1 = DofHandler(grid)
    add!(dhh1, :u, Lagrange{RefQuadrilateral, 1}())
    close!(dhh1)
    ich1 = first(InterfaceIterator(dhh1, topology))
    T = duplication_map(ich1)
    Ke = rand(8, 8)
    _, Kc = condense_interface!(smallbuf, ich1, Ke)
    @test Matrix(Kc) ≈ T' * Ke * T
end

@testset "#1433 regression: assembly with a continuous field across interfaces" begin
    # NOTE: with only two cells (three dofs) every column of K is fully dense and the
    # dense-column fast path in the assembler happens to accumulate duplicated dofs
    # correctly. Four cells make the interface columns sparse enough to hit the merge
    # walk, which is where the raw duplicated scatter fails.
    grid = generate_grid(Line, (4,))
    dh = DofHandler(grid)
    add!(dh, :u, Lagrange{RefLine, 1}())
    close!(dh)
    topology = ExclusiveTopology(grid)
    K = allocate_matrix(dh; topology = topology, interface_coupling = trues(1, 1))
    f = zeros(ndofs(dh))
    buf = InterfaceAssemblyBuffer{Float64}(max_nstacked_interface_dofs(dh))
    # Reference: dense accumulation over the stacked scatter (the congruence transform)
    Kref = zeros(ndofs(dh), ndofs(dh))
    fref = zeros(ndofs(dh))
    assembler = start_assemble(K, f)
    for ic in InterfaceIterator(dh, topology)
        Ke = ones(4, 4)
        fe = ones(4)
        udofs, Kc, fc = condense_interface!(buf, ic, Ke, fe)
        assemble!(assembler, udofs, Kc, fc)
        sd = interfacedofs(ic)
        for j in 1:4, i in 1:4
            Kref[sd[i], sd[j]] += Ke[i, j]
        end
        for i in 1:4
            fref[sd[i]] += fe[i]
        end
    end
    @test Matrix(K) ≈ Kref
    @test f ≈ fref
    # In particular the shared-dof diagonal receives the sum of the 2x2 duplicated block:
    # TᵀKeT of all-ones has 4 at the shared position (storage semantics — a kernel that
    # *means* K[D, D] == 1 must weight the two copies accordingly, see the weak-form test)
    ic = first(InterfaceIterator(dh, topology))
    @test interfacedofs(ic)[2] == interfacedofs(ic)[3] # P1 on Line: dofs [d0, D, D, d2]
    _, Kc1 = condense_interface!(buf, ic, ones(4, 4))
    @test Matrix(Kc1) == [1.0 2.0 1.0; 2.0 4.0 2.0; 1.0 2.0 1.0]
    # The raw duplicated scatter still errors on the merge path, now with a diagnostic
    # that mentions the duplicated dofs and points to condense_interface!
    assembler2 = start_assemble(K, f)
    err = try
        assemble!(assembler2, interfacedofs(ic), ones(4, 4))
        nothing
    catch e
        e
    end
    @test err isa ErrorException
    @test occursin("duplicated", err.msg)
    @test occursin("condense_interface!", err.msg)
    # COOAssembler sums duplicates natively: raw stacked assembly must equal the
    # explicit congruence transform
    coo = Ferrite.COOAssembler()
    for ic in InterfaceIterator(dh, topology)
        assemble!(coo, interfacedofs(ic), ones(4, 4))
    end
    Kcoo, _ = finish_assemble(coo)
    @test Matrix(Kcoo) ≈ Kref
end

@testset "Interface term only for the discontinuous field next to a continuous field" begin
    # Common mixed case: continuous u and discontinuous P where only P has interface
    # terms (u appears only in cell integrals). The interface_coupling mask then only
    # needs the P-P block, since that is the only block the form produces. The stacked
    # system still contains the (shared) u dofs, but their condensed rows/columns are
    # *structural* zeros — the kernel never writes them, so this does not rely on
    # floating point cancellation — and the assembler tolerates missing pattern entries
    # for exact zeros.
    grid = generate_grid(Quadrilateral, (3, 1)) # enough cells to leave the dense-column path
    topology = ExclusiveTopology(grid)
    ip_u = Lagrange{RefQuadrilateral, 1}()
    ip_P = DiscontinuousLagrange{RefQuadrilateral, 1}()
    dh = DofHandler(grid)
    add!(dh, :u, ip_u)
    add!(dh, :P, ip_P)
    close!(dh)
    iv_P = InterfaceValues(FacetQuadratureRule{RefQuadrilateral}(2), ip_P)
    μ = 2.0
    function P_penalty_kernel!(Ke, ic)
        rP = dof_range(ic, :P)
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
    # Field order (:u, :P): interface entries only for the P-P block, and no u-P cell
    # coupling either (the imagined cell integrals couple u-u and P-P only)
    cell_coupling = [true false; false true]
    interface_coupling = [false false; false true]
    K = allocate_matrix(dh; coupling = cell_coupling, topology = topology, interface_coupling = interface_coupling)
    assembler = start_assemble(K)
    buf = InterfaceAssemblyBuffer{Float64}(max_nstacked_interface_dofs(dh))
    Kdense = zeros(ndofs(dh), ndofs(dh))
    n_interfaces = 0
    for ic in InterfaceIterator(dh, topology)
        n_interfaces += 1
        reinit!(iv_P, ic)
        n = nstacked_interface_dofs(ic)
        Ke = zeros(n, n)
        P_penalty_kernel!(Ke, ic)
        # The u dofs are shared even though the form never touches them: condensation is
        # still required (the stacked scatter vector contains duplicates)
        @test nunique_interface_dofs(ic) < nstacked_interface_dofs(ic)
        udofs, Kc = condense_interface!(buf, ic, Ke)
        @test length(udofs) < n
        # The condensed u rows/columns are structural zeros
        upos = unique(ic.stacked_to_unique[dof_range(ic, :u)])
        @test all(iszero, Kc[upos, :])
        @test all(iszero, Kc[:, upos])
        # Assembles cleanly although the pattern has no interface entries for u
        assemble!(assembler, udofs, Kc)
        sd = interfacedofs(ic)
        for j in 1:n, i in 1:n
            Kdense[sd[i], sd[j]] += Ke[i, j]
        end
    end
    @test n_interfaces == 2
    @test Matrix(K) ≈ Kdense
    # Negative control: the P-only mask is load-bearing. A form that couples P to the
    # continuous u writes nonzero values into positions this pattern does not have, and
    # assembly of the condensed (duplicate-free) system reports the missing entry.
    iv_u = InterfaceValues(FacetQuadratureRule{RefQuadrilateral}(2), ip_u)
    assembler2 = start_assemble(K)
    ic = first(InterfaceIterator(dh, topology))
    reinit!(iv_P, ic)
    reinit!(iv_u, ic)
    n = nstacked_interface_dofs(ic)
    Ke = zeros(n, n)
    P_penalty_kernel!(Ke, ic)
    rP = dof_range(ic, :P)
    ru = dof_range(ic, :u)
    for qp in 1:getnquadpoints(iv_P)
        dΓ = getdetJdV(iv_P, qp)
        for (i, I) in pairs(rP), (j, J) in pairs(ru)
            Ke[I, J] += shape_value_jump(iv_P, qp, i) * shape_value_average(iv_u, qp, j) * dΓ
        end
    end
    udofs, Kc = condense_interface!(buf, ic, Ke)
    err = try
        assemble!(assembler2, udofs, Kc)
        nothing
    catch e
        e
    end
    @test err isa ErrorException
    @test occursin("missing in the sparsity pattern", err.msg)
    # The condensed dof vector is duplicate-free, so this is a genuine missing entry and
    # the duplicate diagnostic must NOT fire
    @test !occursin("duplicated", err.msg)
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
    # enclosing scope would rebind the *shared* variable (e.g. a local `K = ...` here
    # would clobber the testset's `K` on the second call).
    function assemble_interface_matrix(kernel!::F) where {F}
        Kmat = allocate_matrix(dh; topology = topology, interface_coupling = trues(2, 2))
        asm = start_assemble(Kmat)
        cbuf = InterfaceAssemblyBuffer{Float64}(max_nstacked_interface_dofs(dh))
        Kstore = zeros(max_nstacked_interface_dofs(dh), max_nstacked_interface_dofs(dh))
        for ic_ in InterfaceIterator(dh, topology)
            reinit!(iv_u, ic_)
            reinit!(iv_P, ic_)
            n_ = nstacked_interface_dofs(ic_)
            Ke_ = @view Kstore[1:n_, 1:n_]
            fill!(Ke_, 0)
            kernel!(Ke_, ic_)
            udofs_, Kc_ = condense_interface!(cbuf, ic_, Ke_)
            assemble!(asm, udofs_, Kc_)
        end
        return Kmat
    end

    # Correct kernel: average (trace) weighting for the continuous field
    function correct_kernel!(Ke, ic)
        rP = dof_range(ic, :P)
        ru = dof_range(ic, :u)
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
        rP = dof_range(ic, :P)
        ru = dof_range(ic, :u)
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
        for ic in InterfaceIterator(dh, topology)
            reinit!(iv_u, ic)
            reinit!(iv_P, ic)
            rP = dof_range(ic, :P)
            ru = dof_range(ic, :u)
            ve = v[interfacedofs(ic)]
            we = w[interfacedofs(ic)]
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
    shared_u = [d for (i, d) in pairs(interfacedofs(ic)) if is_shared(ic, i)]
    ch = ConstraintHandler(dh)
    add!(ch, AffineConstraint(shared_u[1], [shared_u[2] => 0.5], 1.0))
    close!(ch)
    n = nstacked_interface_dofs(ic)
    Ke = rand(n, n)
    fe = rand(n)
    buf = InterfaceAssemblyBuffer{Float64}(max_nstacked_interface_dofs(dh))
    udofs, Kc, fc = condense_interface!(buf, ic, Ke, fe)
    Kc_copy = Matrix(Kc)
    fc_copy = collect(fc)
    K1 = allocate_matrix(dh, ch; topology = topology, interface_coupling = trues(2, 2))
    f1 = zeros(ndofs(dh))
    K2 = copy(K1)
    f2 = zeros(ndofs(dh))
    asm1 = start_assemble(K1, f1)
    apply_assemble!(asm1, ch, udofs, Kc, fc)
    asm2 = start_assemble(K2, f2)
    apply_local!(Kc_copy, fc_copy, udofs, ch)
    assemble!(asm2, udofs, Kc_copy, fc_copy)
    @test Matrix(K1) ≈ Matrix(K2)
    @test f1 ≈ f2
end

@testset "Lazy map, construction modes, aliasing, sizing bound" begin
    # Lazy unique-map construction: reinit! must not build the map, so loops that never
    # use it (raw DG assembly, sparsity construction) do not pay for it
    grid = generate_grid(Quadrilateral, (2, 1))
    topology = ExclusiveTopology(grid)
    dh = DofHandler(grid)
    add!(dh, :u, Lagrange{RefQuadrilateral, 1}())
    close!(dh)
    for ic in InterfaceIterator(dh, topology)
        @test !ic.unique_map_valid # not built by reinit!
    end
    ic = first(InterfaceIterator(dh, topology))
    @test !ic.unique_map_valid
    udofs = unique_interfacedofs(ic) # first accessor builds it
    @test ic.unique_map_valid
    # Identity rule: condense returns unique_interfacedofs(ic) (the same object), on both
    # the condensing and the pass-through path
    buf = InterfaceAssemblyBuffer{Float64}(max_nstacked_interface_dofs(dh))
    n = nstacked_interface_dofs(ic)
    ud2, Kc = condense_interface!(buf, ic, rand(n, n))
    @test ud2 === udofs === unique_interfacedofs(ic)
    dhdg = DofHandler(grid)
    add!(dhdg, :u, DiscontinuousLagrange{RefQuadrilateral, 1}())
    close!(dhdg)
    icdg = first(InterfaceIterator(dhdg, topology))
    Ke = rand(8, 8)
    uddg, Kdg = condense_interface!(buf, icdg, Ke)
    @test uddg === unique_interfacedofs(icdg) === interfacedofs(icdg)
    @test Kdg === Ke
    # Aliasing: views into the buffer's storage (e.g. outputs of a previous call) must not
    # be passed back as input, even when the sizes happen to match
    bad_K = reshape(view(buf.Kc, 1:(n * n)), n, n)
    @test_throws ArgumentError condense_interface!(buf, ic, bad_K)
    bad_f = view(buf.fc, 1:n)
    @test_throws ArgumentError condense_interface!(buf, ic, rand(n, n), bad_f)
    # (undersized previous outputs are caught by the dimension check instead)
    @test_throws DimensionMismatch condense_interface!(buf, ic, Kc)

    # Grid-only cache/iterator: no dof information, empty maps, targeted errors
    icg = first(InterfaceIterator(grid, topology))
    @test isempty(interfacedofs(icg))
    @test nstacked_interface_dofs(icg) == 0
    @test nunique_interface_dofs(icg) == 0
    @test_throws ErrorException dof_range(icg, :u)

    # SubDofHandler-constructed cache: dof information resolves through sdh.dh. The cache
    # keeps the existing domain restriction: both cells must belong to the sub-handler.
    grid2x1 = generate_grid(Quadrilateral, (3, 1))
    dh2 = DofHandler(grid2x1)
    ip = Lagrange{RefQuadrilateral, 1}()
    sdh1 = SubDofHandler(dh2, Set([1, 2]))
    add!(sdh1, :u, ip)
    sdh2 = SubDofHandler(dh2, Set(3))
    add!(sdh2, :u, ip)
    close!(dh2)
    ics = InterfaceCache(sdh1)
    reinit!(ics, FacetIndex(1, 2), FacetIndex(2, 4)) # interface within sdh1's cellset
    @test interfacedofs(ics) == [celldofs(dh2, 1); celldofs(dh2, 2)]
    r = dof_range(ics, :u)
    @test r.here == dof_range(sdh1, :u)
    @test r.there == dof_range(sdh1, :u) .+ ndofs_per_cell(sdh1)
    @test interfacedofs(ics)[r] == [celldofs(dh2, 1); celldofs(dh2, 2)]

    # DofHandler not covering all cells: reject dof_range for interfaces into the
    # uncovered subdomain
    dh3 = DofHandler(grid)
    sdh3 = SubDofHandler(dh3, Set(1)) # cell 2 uncovered
    add!(sdh3, :u, ip)
    close!(dh3)
    ic3 = InterfaceCache(dh3)
    reinit!(ic3, FacetIndex(1, 2), FacetIndex(2, 4))
    @test ic3.sdh_index_b == 0
    err = try dof_range(ic3, :u); nothing; catch e; e; end
    @test err isa ErrorException
    @test occursin("not defined on both cells", err.msg)

    # max_nstacked_interface_dofs must bound same-subdomain interfaces of the largest
    # SubDofHandler (2 * maximum, not the sum of the two largest distinct handlers)
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
    bound = max_nstacked_interface_dofs(dh4)
    # 2 * maximum: the big-big interface (cells 1-2) needs more than
    # ndofs_per_cell(big) + ndofs_per_cell(small) (the "sum of the two largest distinct
    # handlers" would undersize it)
    @test bound == 2 * ndofs_per_cell(big)
    @test bound > ndofs_per_cell(big) + ndofs_per_cell(small)
    buf4 = InterfaceAssemblyBuffer{Float64}() # undersized on purpose: grows on demand
    sizes = Int[]
    for pass in 1:2 # two passes: buffer reused with both increasing and decreasing sizes
        for ic4 in InterfaceIterator(dh4, topo3)
            ns = nstacked_interface_dofs(ic4)
            push!(sizes, ns)
            @test ns <= bound
            # Buffer reuse across differently-sized interfaces: the active region is
            # re-zeroed, the result matches the explicit congruence transform every time
            Ke4 = rand(ns, ns)
            fe4 = rand(ns)
            _, Kc4, fc4 = condense_interface!(buf4, ic4, Ke4, fe4)
            T = duplication_map(ic4)
            @test Matrix(Kc4) ≈ T' * Ke4 * T
            @test collect(fc4) ≈ T' * fe4
        end
    end
    @test length(sizes) == 4 && length(unique(sizes)) == 2 # two interfaces, two sizes
end

@testset "AD: stacked residual -> condensed Jacobian" begin
    # Differentiating a stacked-coefficient interface residual gives a stacked Jacobian,
    # which the (user-typed) buffer condenses like any other stacked local matrix.
    # Equivalence: condensing the stacked Jacobian equals differentiating with respect to
    # the unique coefficients through the gather u_stacked = u_unique[stacked_to_unique]
    # (i.e. u_s = T * u_u, so J_u = Tᵀ J_s T by the chain rule).
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
    ru = dof_range(ic, :u)
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
    ns = nstacked_interface_dofs(ic)
    nu = nunique_interface_dofs(ic)
    T = duplication_map(ic)
    # Stacked route: AD in stacked coefficients, then condense with a Dual-free buffer
    ue_stacked = a[interfacedofs(ic)]
    Js = ForwardDiff.jacobian(stacked_residual, ue_stacked)
    fs = stacked_residual(ue_stacked)
    buf = InterfaceAssemblyBuffer{Float64}(ns)
    udofs, Jc, fc = condense_interface!(buf, ic, Js, fs)
    # Unique route: AD in unique coefficients through the gather
    uu = a[unique_interfacedofs(ic)]
    unique_residual(uuc) = T' * stacked_residual(uuc[ic.stacked_to_unique])
    Ju = ForwardDiff.jacobian(unique_residual, uu)
    @test Matrix(Jc) ≈ Ju
    @test collect(fc) ≈ unique_residual(uu)
    # A Dual-typed buffer works too (the buffer eltype is the user's choice, matching the
    # local matrix eltype). Condensation is linear, so it acts on values and partials alike.
    DualT = ForwardDiff.Dual{Nothing, Float64, 1}
    Jdual = [DualT(Js[i, j], ForwardDiff.Partials((2 * Js[i, j],))) for i in 1:ns, j in 1:ns]
    bufdual = InterfaceAssemblyBuffer{DualT}(ns)
    _, JcD = condense_interface!(bufdual, ic, Jdual)
    @test ForwardDiff.value.(Matrix(JcD)) ≈ Ju
    @test map(d -> ForwardDiff.partials(d)[1], Matrix(JcD)) ≈ 2 .* Ju
end
