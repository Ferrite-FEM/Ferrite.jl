# Tests for the stacked interface dof layout, interface_dof_range /
# InterfaceDofRange, and ndofs_per_interface (Ferrite-FEM/Ferrite.jl#1433).
using Ferrite
using Test
using LinearAlgebra
using SparseArrays

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

@testset "interface_dof_range in the assembly loop across SubDofHandlers" begin
    # The two cells of an interface may belong to different SubDofHandlers with
    # different field orders; interface_dof_range(ic, field) then places the field's dofs
    # correctly in the stacked local matrix on both sides. Assemble a mixed interface
    # form over a grid where one subdomain has the field order swapped, with the kernel
    # indexing exclusively through interface_dof_range, and verify the operator against
    # direct evaluation of the form by quadrature. (Both fields are discontinuous, so no
    # dofs are shared between the cells and the stacked system assembles directly.)
    grid = generate_grid(Quadrilateral, (3, 1))
    topology = ExclusiveTopology(grid)
    ip_u = DiscontinuousLagrange{RefQuadrilateral, 1}()
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
        @test allunique(interfacedofs(ic)) # discontinuous fields: nothing shared
        assemble!(asm, interfacedofs(ic), Ke)
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
