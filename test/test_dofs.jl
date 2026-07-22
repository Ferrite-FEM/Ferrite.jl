# Imports for parallel (isolated) test execution:
using LinearAlgebra
using SparseArrays
import Metis

@testset "DofHandler construction" begin
    grid = generate_grid(Quadrilateral, (2, 1))
    dh = DofHandler(grid)
    # incompatible refshape (#638)
    @test_throws ErrorException add!(dh, :u, Lagrange{RefTriangle, 1}())
    @test_throws ErrorException add!(dh, :u, Lagrange{RefTetrahedron, 1}())
    # field already exists
    add!(dh, :u, Lagrange{RefQuadrilateral, 1}())
    @test_throws ErrorException add!(dh, :u, Lagrange{RefQuadrilateral, 1}())

    # Invalid SubDofHandler construction
    dh = DofHandler(grid)
    sdh1 = Ferrite.SubDofHandler(dh, Set(1))
    # Subdomains not disjoint
    @test_throws ErrorException Ferrite.SubDofHandler(dh, Set(1:getncells(grid)))
    # add field to DofHandler that has subdomains
    @test_throws ErrorException add!(dh, :u, Lagrange{RefQuadrilateral, 1}())

    # inconsistent field across several SubDofHandlers
    dh = DofHandler(grid)
    sdh1 = Ferrite.SubDofHandler(dh, Set(1))
    sdh2 = Ferrite.SubDofHandler(dh, Set(2))
    add!(sdh1, :u, Lagrange{RefQuadrilateral, 1}())
    # different number of components in different sdh
    @test_throws ErrorException add!(sdh2, :u, Lagrange{RefQuadrilateral, 1}()^2)
    # different interpolation order in different sdh
    @test_logs (:warn,) add!(sdh2, :u, Lagrange{RefQuadrilateral, 2}())
end


# misc dofhandler unit tests
@testset "dofs" begin

    # set up a test DofHandler
    grid = generate_grid(Triangle, (10, 10))
    dh = DofHandler(grid)
    add!(dh, :u, Lagrange{RefTriangle, 2}()^2)
    add!(dh, :p, Lagrange{RefTriangle, 1}())
    close!(dh)

    # dof_range
    @test (@inferred dof_range(dh, :u)) == 1:12
    @test (@inferred dof_range(dh, :p)) == 13:15
    # dof_range for SubDofHandler
    ip = Lagrange{RefTriangle, 1}()
    dh = DofHandler(grid)
    sdh = SubDofHandler(dh, Set(1:getncells(grid)))
    add!(sdh, :u, ip^2)
    add!(sdh, :c, ip)

    @test dof_range(sdh, Ferrite.find_field(sdh, :u)) == 1:6
    @test dof_range(sdh, Ferrite.find_field(sdh, :c)) == 7:9
end # testset

@testset "Dofs for Line2" begin

    nodes = [Node{2, Float64}(Vec(0.0, 0.0)), Node{2, Float64}(Vec(1.0, 1.0)), Node{2, Float64}(Vec(2.0, 0.0))]
    cells = [Line((1, 2)), Line((2, 3))]
    grid = Grid(cells, nodes)

    #2d line with 1st order 1d interpolation
    dh = DofHandler(grid)
    add!(dh, :x, Lagrange{RefLine, 1}()^2)
    close!(dh)

    @test celldofs(dh, 1) == [1, 2, 3, 4]
    @test celldofs(dh, 2) == [3, 4, 5, 6]

    #2d line with 2nd order 1d interpolation
    dh = DofHandler(grid)
    add!(dh, :x, Lagrange{RefLine, 2}()^2)
    close!(dh)

    @test celldofs(dh, 1) == [1, 2, 3, 4, 5, 6]
    @test celldofs(dh, 2) == [3, 4, 7, 8, 9, 10]

    #3d line with 2nd order 1d interpolation
    dh = DofHandler(grid)
    add!(dh, :u, Lagrange{RefLine, 2}()^3)
    add!(dh, :θ, Lagrange{RefLine, 2}()^3)
    close!(dh)

    @test celldofs(dh, 1) == collect(1:18)
    @test celldofs(dh, 2) == [
        4, 5, 6, 19, 20, 21, 22, 23, 24,    # u
        13, 14, 15, 25, 26, 27, 28, 29, 30, # θ
    ]
end

@testset "Dofs for quad in 3d (shell)" begin

    nodes = [
        Node{3, Float64}(Vec(0.0, 0.0, 0.0)), Node{3, Float64}(Vec(1.0, 0.0, 0.0)),
        Node{3, Float64}(Vec(1.0, 1.0, 0.0)), Node{3, Float64}(Vec(0.0, 1.0, 0.0)),
        Node{3, Float64}(Vec(2.0, 0.0, 0.0)), Node{3, Float64}(Vec(2.0, 2.0, 0.0)),
    ]

    cells = [Quadrilateral((1, 2, 3, 4)), Quadrilateral((2, 5, 6, 3))]
    grid = Grid(cells, nodes)

    #3d quad with 1st order 2d interpolation
    dh = DofHandler(grid)
    add!(dh, :u, Lagrange{RefQuadrilateral, 1}()^3)
    add!(dh, :θ, Lagrange{RefQuadrilateral, 1}()^3)
    close!(dh)

    @test celldofs(dh, 1) == collect(1:24)
    @test celldofs(dh, 2) == [
        4, 5, 6, 25, 26, 27, 28, 29, 30, 7, 8, 9,       # u
        16, 17, 18, 31, 32, 33, 34, 35, 36, 19, 20, 21, # θ
    ]

    #3d quads with two quadratic interpolations fields
    #Only 1 dim per field for simplicity...
    dh = DofHandler(grid)
    add!(dh, :u, Lagrange{RefQuadrilateral, 2}())
    add!(dh, :θ, Lagrange{RefQuadrilateral, 2}())
    close!(dh)

    @test celldofs(dh, 1) == collect(1:18)
    @test celldofs(dh, 2) == [2, 19, 20, 3, 21, 22, 23, 6, 24, 11, 25, 26, 12, 27, 28, 29, 15, 30]

    # test evaluate_at_grid_nodes
    ## DofHandler
    mesh = generate_grid(Quadrilateral, (1, 1))
    dh = DofHandler(mesh)
    add!(dh, :v, Lagrange{RefQuadrilateral, 1}()^2)
    add!(dh, :s, Lagrange{RefQuadrilateral, 1}())
    close!(dh)

    u = Float64[1.1, 1.2, 2.1, 2.2, 4.1, 4.2, 3.1, 3.2, 1.3, 2.3, 4.3, 3.3]
    u2 = Float32[1.1, 1.2, 2.1, 2.2, 4.1, 4.2, 3.1, 3.2, 1.3, 2.3, 4.3, 3.3]
    uv = @view u[1:end]
    # :s on solution
    s_nodes = evaluate_at_grid_nodes(dh, u, :s)
    @test s_nodes ≈ [i + 0.3 for i in 1:4]
    @test eltype(s_nodes) == Float64
    @test eltype(evaluate_at_grid_nodes(dh, u2, :s)) == Float32
    # :s on a view into solution
    sv_nodes = evaluate_at_grid_nodes(dh, uv, :s)
    @test sv_nodes ≈ [i + 0.3 for i in 1:4]
    # :v on solution
    v_nodes = evaluate_at_grid_nodes(dh, u, :v)
    @test v_nodes ≈ [Vec{2, Float64}(i -> j + i / 10) for j in 1:4]
    @test eltype(v_nodes) == Vec{2, Float64}
    @test eltype(evaluate_at_grid_nodes(dh, u2, :v)) == Vec{2, Float32}
    # :v on a view into solution
    vv_nodes = evaluate_at_grid_nodes(dh, uv, :v)
    @test vv_nodes ≈ [Vec{2, Float64}(i -> j + i / 10) for j in 1:4]

end

@testset "renumber!" begin
    function dhmdhch()
        local dh, mdh, ch
        grid = generate_grid(Triangle, (10, 10))
        dh = DofHandler(grid)
        add!(dh, :u, Lagrange{RefTriangle, 1}())
        close!(dh)
        # subdomains
        mdh = DofHandler(grid)
        sdh1 = SubDofHandler(mdh, Set(1:(getncells(grid) ÷ 2)))
        add!(sdh1, :u, Lagrange{RefTriangle, 1}())
        sdh2 = SubDofHandler(mdh, Set((getncells(grid) ÷ 2 + 1):getncells(grid)))
        add!(sdh2, :u, Lagrange{RefTriangle, 1}())
        close!(mdh)
        ch = ConstraintHandler(dh)
        add!(ch, Dirichlet(:u, getfacetset(grid, "left"), (x, t) -> 0))
        add!(ch, Dirichlet(:u, getfacetset(grid, "right"), (x, t) -> 2))
        face_map = collect_periodic_facets(grid, "bottom", "top")
        add!(ch, PeriodicDirichlet(:u, face_map))
        close!(ch)
        update!(ch, 0)
        return dh, mdh, ch
    end
    dh, mdh, ch = dhmdhch()

    perm = randperm(ndofs(dh))
    iperm = invperm(perm)

    # Roundtrip tests
    original_dofs = copy(dh.cell_dofs)
    renumber!(dh, perm)
    renumber!(dh, iperm)
    @test original_dofs == dh.cell_dofs
    original_dofs_mdh = copy(mdh.cell_dofs)
    renumber!(mdh, perm)
    renumber!(mdh, iperm)
    @test original_dofs_mdh == mdh.cell_dofs
    original_prescribed = copy(ch.prescribed_dofs)
    original_inhomogeneities = copy(ch.inhomogeneities)
    original_affine_inhomogeneities = copy(ch.affine_inhomogeneities)
    original_dofcoefficients = [c === nothing ? c : copy(c) for c in ch.dofcoefficients]
    renumber!(dh, ch, perm)
    renumber!(dh, ch, iperm)
    @test original_dofs == dh.cell_dofs
    @test original_prescribed == ch.prescribed_dofs
    @test original_inhomogeneities == ch.inhomogeneities
    @test original_affine_inhomogeneities == ch.affine_inhomogeneities
    @test original_dofcoefficients == ch.dofcoefficients

    # Integration tests
    K = allocate_matrix(dh, ch)
    f = zeros(ndofs(dh))
    a = start_assemble(K, f)
    dhp, _, chp = dhmdhch()
    renumber!(dhp, chp, perm)
    Kp = allocate_matrix(dhp, chp)
    fp = zeros(ndofs(dhp))
    ap = start_assemble(Kp, fp)
    for cellid in 1:getncells(dh.grid)
        ke = Float64[3 -1 -2; -1 4 -1; -2 -1 5] * cellid
        fe = Float64[1, 2, 3] * cellid
        assemble!(a, celldofs(dh, cellid), ke, fe)
        assemble!(ap, celldofs(dhp, cellid), ke, fe)
    end
    apply!(K, f, ch)
    apply!(Kp, fp, chp)
    u = K \ f
    up = Kp \ fp
    @test norm(u) ≈ norm(up) ≈ 15.47826706793882
    @test u ≈ up[perm]
    @test u[iperm] ≈ up


    ###################################
    # Renumbering by field/components #
    ###################################

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
        add!(ch, AffineConstraint(13, [15 => 0.5, 16 => 0.5], 0.0))
        close!(ch)
        return dh, ch
    end

    # Original numbering
    dho, cho = testdhch()
    #        :v                :s
    #  7,8───5,6──15,16  12────11────18
    #   │  1  │  2  │     │  1  │  2  │
    #  1,2───3,4──13,14   9────10────17
    @test celldofs(dho, 1) == 1:12
    @test celldofs(dho, 2) == [3, 4, 13, 14, 15, 16, 5, 6, 10, 17, 18, 11]
    @test cho.prescribed_dofs == [2, 8, 9, 12, 13]

    # By field
    dh, ch = testdhch()
    renumber!(dh, ch, DofOrder.FieldWise())
    #        :v                :s
    #  7,8───5,6──11,12  16────15────18
    #   │  1  │  2  │     │  1  │  2  │
    #  1,2───3,4───9,10  13────14────17
    @test celldofs(dh, 1) == [1, 2, 3, 4, 5, 6, 7, 8, 13, 14, 15, 16]
    @test celldofs(dh, 2) == [3, 4, 9, 10, 11, 12, 5, 6, 14, 17, 18, 15]
    @test ch.prescribed_dofs == sort!([2, 8, 13, 16, 9])
    for el in 1:2, r in [dof_range(dh, :v), dof_range(dh, :s)]
        # Test stability within each block: i < j -> p(i) < p(j), i > j -> p(i) > p(j)
        @test sign.(diff(celldofs(dh, el)[r])) == sign.(diff(celldofs(dho, el)[r]))
    end

    # By field, reordered
    dh, ch = testdhch()
    renumber!(dh, ch, DofOrder.FieldWise([2, 1]))
    #        :v                :s
    # 13,14─11,12─17,18   4─────3─────6
    #   │  1  │  2  │     │  1  │  2  │
    #  7,8───9,10─15,16   1─────2─────5
    @test celldofs(dh, 1) == [7, 8, 9, 10, 11, 12, 13, 14, 1, 2, 3, 4]
    @test celldofs(dh, 2) == [9, 10, 15, 16, 17, 18, 11, 12, 2, 5, 6, 3]
    @test ch.prescribed_dofs == sort!([8, 14, 1, 4, 15])
    for el in 1:2, r in [dof_range(dh, :v), dof_range(dh, :s)]
        @test sign.(diff(celldofs(dh, el)[r])) == sign.(diff(celldofs(dho, el)[r]))
    end

    # By component
    dh, ch = testdhch()
    renumber!(dh, ch, DofOrder.ComponentWise())
    #        :v                :s
    #  4,10──3,9───6,12  16────15────18
    #   │  1  │  2  │     │  1  │  2  │
    #  1,7───2,8───5,11  13────14────17
    @test celldofs(dh, 1) == [1, 7, 2, 8, 3, 9, 4, 10, 13, 14, 15, 16]
    @test celldofs(dh, 2) == [2, 8, 5, 11, 6, 12, 3, 9, 14, 17, 18, 15]
    @test ch.prescribed_dofs == sort!([7, 10, 13, 16, 5])
    for el in 1:2, r in [dof_range(dh, :v)[1:2:end], dof_range(dh, :v)[2:2:end], dof_range(dh, :s)]
        @test sign.(diff(celldofs(dh, el)[r])) == sign.(diff(celldofs(dho, el)[r]))
    end

    # By component, reordered
    dh, ch = testdhch()
    renumber!(dh, ch, DofOrder.ComponentWise([3, 1, 2]))
    #        :v                :s
    # 16,4──15,3──18,6   10─────9────12
    #   │  1  │  2  │     │  1  │  2  │
    # 13,1──14,2──17,5    7─────8────11
    @test celldofs(dh, 1) == [13, 1, 14, 2, 15, 3, 16, 4, 7, 8, 9, 10]
    @test celldofs(dh, 2) == [14, 2, 17, 5, 18, 6, 15, 3, 8, 11, 12, 9]
    @test ch.prescribed_dofs == sort!([1, 4, 7, 10, 17])
    for el in 1:2, r in [dof_range(dh, :v)[1:2:end], dof_range(dh, :v)[2:2:end], dof_range(dh, :s)]
        @test sign.(diff(celldofs(dh, el)[r])) == sign.(diff(celldofs(dho, el)[r]))
    end

    #######################################
    # Field on subdomain #
    #######################################

    function test_dhch_subdomain()
        local grid, dh, ch
        grid = generate_grid(Quadrilateral, (2, 1))
        ip = Lagrange{RefQuadrilateral, 1}()
        dh = DofHandler(grid)
        sdh1 = SubDofHandler(dh, Set(1))
        add!(sdh1, :v, ip^2)
        add!(sdh1, :s, ip)
        sdh2 = SubDofHandler(dh, Set(2))
        add!(sdh2, :v, ip^2)
        close!(dh)
        ch = ConstraintHandler(dh)
        add!(ch, Dirichlet(:v, getfacetset(grid, "left"), (x, t) -> 0, [2]))
        add!(ch, Dirichlet(:s, getfacetset(grid, "left"), (x, t) -> 0))
        add!(ch, AffineConstraint(13, [15 => 0.5, 16 => 0.5], 0.0))
        close!(ch)
        return dh, ch
    end

    # Original numbering
    dho, cho = test_dhch_subdomain()
    #        :v                :s
    #  7,8───5,6──15,16  12────11────
    #   │  1  │  2  │     │  1  │  2  │
    #  1,2───3,4──13,14   9────10────
    @test celldofs(dho, 1) == 1:12
    @test celldofs(dho, 2) == [3, 4, 13, 14, 15, 16, 5, 6]
    @test cho.prescribed_dofs == [2, 8, 9, 12, 13]

    # By field
    dh, ch = test_dhch_subdomain()
    renumber!(dh, ch, DofOrder.FieldWise())
    #        :v                :s
    #  7,8───5,6──11,12  16────15────
    #   │  1  │  2  │     │  1  │  2  │
    #  1,2───3,4───9,10  13────14────
    @test celldofs(dh, 1) == [1, 2, 3, 4, 5, 6, 7, 8, 13, 14, 15, 16]
    @test celldofs(dh, 2) == [3, 4, 9, 10, 11, 12, 5, 6]
    @test ch.prescribed_dofs == sort!([2, 8, 13, 16, 9])
    for r in [dof_range(dh.subdofhandlers[1], :v), dof_range(dh.subdofhandlers[1], :s)]
        # Test stability within each block: i < j -> p(i) < p(j), i > j -> p(i) > p(j)
        @test sign.(diff(celldofs(dh, 1)[r])) == sign.(diff(celldofs(dho, 1)[r]))
    end
    r = dof_range(dh.subdofhandlers[2], :v)
    @test sign.(diff(celldofs(dh, 2)[r])) == sign.(diff(celldofs(dho, 2)[r]))

    # By field, reordered
    dh, ch = test_dhch_subdomain()
    renumber!(dh, ch, DofOrder.FieldWise([2, 1]))
    #        :v                :s
    # 11,12──9,10─15,16   4─────3─────
    #   │  1  │  2  │     │  1  │  2  │
    #  5,6───7,8──13,14   1─────2─────
    @test celldofs(dh, 1) == [5, 6, 7, 8, 9, 10, 11, 12, 1, 2, 3, 4]
    @test celldofs(dh, 2) == [7, 8, 13, 14, 15, 16, 9, 10]
    @test ch.prescribed_dofs == sort!([6, 12, 1, 4, 13])
    for r in [dof_range(dh.subdofhandlers[1], :v), dof_range(dh.subdofhandlers[1], :s)]
        # Test stability within each block: i < j -> p(i) < p(j), i > j -> p(i) > p(j)
        @test sign.(diff(celldofs(dh, 1)[r])) == sign.(diff(celldofs(dho, 1)[r]))
    end
    r = dof_range(dh.subdofhandlers[2], :v)
    @test sign.(diff(celldofs(dh, 2)[r])) == sign.(diff(celldofs(dho, 2)[r]))

    # By component
    dh, ch = test_dhch_subdomain()
    renumber!(dh, ch, DofOrder.ComponentWise())
    #        :v                :s
    #  4,10──3,9───6,12  16────15────
    #   │  1  │  2  │     │  1  │  2  │
    #  1,7───2,8───5,11  13────14────
    @test celldofs(dh, 1) == [1, 7, 2, 8, 3, 9, 4, 10, 13, 14, 15, 16]
    @test celldofs(dh, 2) == [2, 8, 5, 11, 6, 12, 3, 9]
    @test ch.prescribed_dofs == sort!([7, 10, 13, 16, 5])
    dof_range_v1 = dof_range(dh.subdofhandlers[1], :v)
    dof_range_s1 = dof_range(dh.subdofhandlers[1], :s)
    for r in [dof_range_v1[1:2:end], dof_range_v1[2:2:end], dof_range_s1]
        # Test stability within each block: i < j -> p(i) < p(j), i > j -> p(i) > p(j)
        @test sign.(diff(celldofs(dh, 1)[r])) == sign.(diff(celldofs(dho, 1)[r]))
    end
    dof_range_v2 = dof_range(dh.subdofhandlers[2], :v)
    for r in [dof_range_v2[1:2:end], dof_range_v2[2:2:end]]
        @test sign.(diff(celldofs(dh, 2)[r])) == sign.(diff(celldofs(dho, 2)[r]))
    end

    # By component, reordered
    dh, ch = test_dhch_subdomain()
    renumber!(dh, ch, DofOrder.ComponentWise([3, 1, 2]))
    #        :v                :s
    # 14,4──13,3──16,6   10─────9────
    #   │  1  │  2  │     │  1  │  2  │
    # 11,1──12,2──15,5    7─────8────
    @test celldofs(dh, 1) == [11, 1, 12, 2, 13, 3, 14, 4, 7, 8, 9, 10]
    @test celldofs(dh, 2) == [12, 2, 15, 5, 16, 6, 13, 3]
    @test ch.prescribed_dofs == sort!([1, 4, 7, 10, 15])
    dof_range_v1 = dof_range(dh.subdofhandlers[1], :v)
    dof_range_s1 = dof_range(dh.subdofhandlers[1], :s)
    for r in [dof_range_v1[1:2:end], dof_range_v1[2:2:end], dof_range_s1]
        # Test stability within each block: i < j -> p(i) < p(j), i > j -> p(i) > p(j)
        @test sign.(diff(celldofs(dh, 1)[r])) == sign.(diff(celldofs(dho, 1)[r]))
    end
    dof_range_v2 = dof_range(dh.subdofhandlers[2], :v)
    for r in [dof_range_v2[1:2:end], dof_range_v2[2:2:end]]
        @test sign.(diff(celldofs(dh, 2)[r])) == sign.(diff(celldofs(dho, 2)[r]))
    end

    # Metis ordering
    # TODO: Should probably test that the new order result in less fill-in
    dh, ch = testdhch()
    renumber!(dh, DofOrder.Ext{Metis}())
    @test_throws ErrorException renumber!(dh, ch, DofOrder.Ext{Metis}())
    renumber!(dh, DofOrder.Ext{Metis}(coupling = [true true; true false]))
    @test_throws ErrorException renumber!(dh, ch, DofOrder.Ext{Metis}(coupling = [true true; true false]))
end

@testset "dof coupling" begin
    grid = generate_grid(Quadrilateral, (1, 1))
    dh = DofHandler(grid)
    add!(dh, :u, Lagrange{RefQuadrilateral, 1}()^2)
    add!(dh, :p, Lagrange{RefQuadrilateral, 1}())
    close!(dh)
    ch = ConstraintHandler(dh)
    close!(ch)
    udofs = vdofs = dof_range(dh, :u)
    u1dofs = v1dofs = udofs[1:2:end]
    u2dofs = v2dofs = udofs[2:2:end]
    pdofs = qdofs = dof_range(dh, :p)

    function is_stored(A, i, j)
        A = A isa Symmetric ? A.data : A
        for m in nzrange(A, j)
            A.rowval[m] == i && return true
        end
        return false
    end
    function is_stored(sparsity_pattern::SparsityPattern, i, j)
        return findfirst(k -> k == j, sparsity_pattern.rows[i]) !== nothing
    end

    # Full coupling (default)
    sparsity_pattern = init_sparsity_pattern(dh)
    add_sparsity_entries!(sparsity_pattern, dh)
    K = allocate_matrix(sparsity_pattern)
    @test eltype(K) == Float64
    for j in 1:ndofs(dh), i in 1:ndofs(dh)
        @test is_stored(sparsity_pattern, i, j)
        @test is_stored(K, i, j)
    end

    # Field coupling
    coupling = [
        # u    p
        true true  # v
        true false # q
    ]
    sparsity_pattern = init_sparsity_pattern(dh)
    add_sparsity_entries!(sparsity_pattern, dh; coupling = coupling)
    K = allocate_matrix(sparsity_pattern)
    # Kch = allocate_matrix(dh, ch; coupling=coupling)
    # @test K.rowval == Kch.rowval
    # @test K.colptr == Kch.colptr
    # KS = create_symmetric_sparsity_pattern(dh; coupling=coupling)
    # KSch = create_symmetric_sparsity_pattern(dh, ch; coupling=coupling)
    # @test KS.data.rowval == KSch.data.rowval
    # @test KS.data.colptr == KSch.data.colptr
    for j in udofs, i in Iterators.flatten((vdofs, qdofs))
        @test is_stored(sparsity_pattern, i, j)
        @test is_stored(K, i, j)
        # @test is_stored(KS, i, j) == (i <= j)
    end
    for j in pdofs, i in vdofs
        @test is_stored(sparsity_pattern, i, j)
        @test is_stored(K, i, j)
        # @test is_stored(KS, i, j)
    end
    for j in pdofs, i in qdofs
        @test is_stored(sparsity_pattern, i, j) == (i == j)
        @test is_stored(K, i, j) == (i == j)
        # @test is_stored(KS, i, j) == (i == j)
    end

    # Component coupling
    coupling = [
        # u1   u2    p
        true  true  false # v1
        true  false true  # v2
        false true  true  # q
    ]
    sparsity_pattern = init_sparsity_pattern(dh)
    add_sparsity_entries!(sparsity_pattern, dh; coupling = coupling)
    K = allocate_matrix(sparsity_pattern)
    # KS = create_symmetric_sparsity_pattern(dh; coupling=coupling)
    for j in u1dofs, i in vdofs
        @test is_stored(sparsity_pattern, i, j)
        @test is_stored(K, i, j)
        # @test is_stored(KS, i, j) == (i <= j)
    end
    for j in u1dofs, i in qdofs
        @test !is_stored(sparsity_pattern, i, j)
        @test !is_stored(K, i, j)
        # @test !is_stored(KS, i, j)
    end
    for j in u2dofs, i in Iterators.flatten((v1dofs, qdofs))
        @test is_stored(sparsity_pattern, i, j)
        @test is_stored(K, i, j)
        # @test is_stored(KS, i, j) == (i <= j)
    end
    for j in u2dofs, i in v2dofs
        @test is_stored(sparsity_pattern, i, j) == (i == j)
        @test is_stored(K, i, j) == (i == j)
        # @test is_stored(KS, i, j) == (i == j)
    end
    for j in pdofs, i in v1dofs
        @test !is_stored(sparsity_pattern, i, j)
        @test !is_stored(K, i, j)
        # @test !is_stored(KS, i, j)
    end
    for j in pdofs, i in Iterators.flatten((v2dofs, qdofs))
        @test is_stored(sparsity_pattern, i, j)
        @test is_stored(K, i, j)
        # @test is_stored(KS, i, j) == (i <= j)
    end

    # Error paths
    @test_throws ErrorException("coupling not square") allocate_matrix(dh; coupling = [true true])
    # @test_throws ErrorException("coupling not symmetric") create_symmetric_sparsity_pattern(dh; coupling=[true true; false true])
    # @test_throws ErrorException("could not create coupling") create_symmetric_sparsity_pattern(dh; coupling=falses(100, 100))

    # Test coupling with subdomains
    grid = generate_grid(Quadrilateral, (1, 2))
    dh = DofHandler(grid)
    sdh1 = SubDofHandler(dh, Set(1))
    add!(sdh1, :u, Lagrange{RefQuadrilateral, 1}()^2)
    add!(sdh1, :p, Lagrange{RefQuadrilateral, 1}())
    sdh2 = SubDofHandler(dh, Set(2))
    add!(sdh2, :u, Lagrange{RefQuadrilateral, 1}()^2)
    close!(dh)

    sparsity_pattern = init_sparsity_pattern(dh)
    add_sparsity_entries!(sparsity_pattern, dh; coupling = [true true; true false])
    K = allocate_matrix(sparsity_pattern)
    KS = Symmetric(allocate_matrix(dh; #= symmetric=true, =# coupling = [true true; true false]))
    # Subdomain 1: u and p
    udofs = celldofs(dh, 1)[dof_range(sdh1, :u)]
    pdofs = celldofs(dh, 1)[dof_range(sdh1, :p)]
    for j in udofs, i in Iterators.flatten((udofs, pdofs))
        @test is_stored(sparsity_pattern, i, j)
        @test is_stored(K, i, j)
        # @test is_stored(KS, i, j) == (i <= j)
    end
    for j in pdofs, i in udofs
        @test is_stored(sparsity_pattern, i, j)
        @test is_stored(K, i, j)
        # @test is_stored(KS, i, j)
    end
    for j in pdofs, i in pdofs
        @test is_stored(sparsity_pattern, i, j) == (i == j)
        @test is_stored(K, i, j) == (i == j)
        # @test is_stored(KS, i, j) == (i == j)
    end
    # Subdomain 2: u
    udofs = celldofs(dh, 2)[dof_range(sdh2, :u)]
    for j in udofs, i in udofs
        @test is_stored(sparsity_pattern, i, j)
        @test is_stored(K, i, j)
        # @test is_stored(KS, i, j) == (i <= j)
    end
end

@testset "dof cross-coupling" begin
    couplings = [
        # Field couplings
        # reshape.(Iterators.product(fill([true, false], 9)...) |> collect |> vec .|> collect, Ref((3,3))),
        [
            true  true  true
            true  true  true
            true  true  true
        ],
        [
            true   false  false
            false  true  false
            false  false  true
        ],
        [
            true   true  false
            true  true  true
            false  true  true
        ],

        # Component coupling
        [
            true    true    true    true
            true    true    true    true
            true    true    true    true
            true    true    true    true
        ],
        [
            true     false    false    false
            false    true     false    false
            false    false    true     false
            false    false    false    true
        ],
        [
            true    true    true    false
            true    true    true    true
            true    true    true    true
            false    true    true    true
        ],
    ]
    function is_stored(A, i, j)
        A = A isa Symmetric ? A.data : A
        for m in nzrange(A, j)
            A.rowval[m] == i && return true
        end
        return false
    end
    function _check_dofs(K, dh, sdh, cell_idx, coupling, coupling_idx, vdim, neighbors, is_cross_element)
        for field1_idx in eachindex(sdh.field_names)
            i_dofs = dof_range(sdh, field1_idx)
            ip1 = sdh.field_interpolations[field1_idx]
            vdim[1] = typeof(ip1) <: VectorizedInterpolation && size(coupling)[1] == 4 ? Ferrite.get_n_copies(ip1) : 1
            for dim1 in 1:vdim[1]
                for cell2_idx in neighbors
                    sdh2 = dh.subdofhandlers[dh.cell_to_subdofhandler[cell2_idx]]
                    coupling_idx[2] = 1
                    for field2_idx in eachindex(sdh2.field_names)
                        j_dofs = dof_range(sdh2, field2_idx)
                        ip2 = sdh2.field_interpolations[field2_idx]
                        vdim[2] = typeof(ip2) <: VectorizedInterpolation && size(coupling)[1] == 4 ? Ferrite.get_n_copies(ip2) : 1
                        for dim2 in 1:vdim[2]
                            i_dofs_v = i_dofs[dim1:vdim[1]:end]
                            j_dofs_v = j_dofs[dim2:vdim[2]:end]
                            for i_idx in i_dofs_v, j_idx in j_dofs_v
                                i = celldofs(dh, cell_idx)[i_idx]
                                j = celldofs(dh, cell2_idx)[j_idx]
                                is_cross_element && (i ∈ celldofs(dh, cell2_idx) || j ∈ celldofs(dh, cell_idx)) && continue
                                @test is_stored(K, i, j) == coupling[coupling_idx...]
                            end
                            coupling_idx[2] += 1
                        end
                    end
                end
                coupling_idx[1] += 1
            end
        end
    end
    function check_coupling(dh, topology, K, coupling, interface_coupling)
        for cell_idx in eachindex(getcells(dh.grid))
            sdh = dh.subdofhandlers[dh.cell_to_subdofhandler[cell_idx]]
            coupling_idx = [1, 1]
            interface_coupling_idx = [1, 1]
            vdim = [1, 1]
            # test inner coupling
            _check_dofs(K, dh, sdh, cell_idx, coupling, coupling_idx, vdim, [cell_idx], false)
            # test cross-element coupling
            neighborhood = Ferrite.get_facet_facet_neighborhood(topology, grid)
            neighbors = [neighborhood[cell_idx, i] for i in 1:size(neighborhood, 2)]
            _check_dofs(K, dh, sdh, cell_idx, interface_coupling, interface_coupling_idx, vdim, [i[1][1] for i in neighbors[.!isempty.(neighbors)]], true)
        end
    end
    grid = generate_grid(Quadrilateral, (2, 2))
    topology = ExclusiveTopology(grid)
    dh = DofHandler(grid)
    add!(dh, :u, DiscontinuousLagrange{RefQuadrilateral, 1}()^2)
    add!(dh, :p, DiscontinuousLagrange{RefQuadrilateral, 1}())
    add!(dh, :w, Lagrange{RefQuadrilateral, 1}())
    close!(dh)
    for coupling in couplings, interface_coupling in couplings
        K = allocate_matrix(dh; coupling = coupling, topology = topology, interface_coupling = interface_coupling)
        all(coupling) && @test K == allocate_matrix(dh, topology = topology, interface_coupling = interface_coupling)
        check_coupling(dh, topology, K, coupling, interface_coupling)
    end

    # Error paths
    @test_throws ErrorException("coupling not square") allocate_matrix(dh; coupling = [true true])
    # @test_throws ErrorException("coupling not symmetric") allocate_matrix(dh; coupling=[true true; false true])
    @test_throws ErrorException("could not create coupling") allocate_matrix(dh; coupling = falses(100, 100))

    # Test coupling with subdomains
    # Note: `check_coupling` works for this case only because the second domain has dofs from the first domain in order. Otherwise tests like in continuous ip are required.
    grid = generate_grid(Quadrilateral, (2, 1))
    topology = ExclusiveTopology(grid)

    dh = DofHandler(grid)
    sdh1 = SubDofHandler(dh, Set(1))
    add!(sdh1, :u, DiscontinuousLagrange{RefQuadrilateral, 1}()^2)
    add!(sdh1, :y, DiscontinuousLagrange{RefQuadrilateral, 1}())
    add!(sdh1, :p, Lagrange{RefQuadrilateral, 1}())
    sdh2 = SubDofHandler(dh, Set(2))
    add!(sdh2, :u, DiscontinuousLagrange{RefQuadrilateral, 1}()^2)
    close!(dh)

    for coupling in couplings, interface_coupling in couplings
        K = allocate_matrix(dh; coupling = coupling, topology = topology, interface_coupling = interface_coupling)
        all(coupling) && @test K == allocate_matrix(dh, topology = topology, interface_coupling = interface_coupling)
        check_coupling(dh, topology, K, coupling, interface_coupling)
    end

    # Testing Crouzeix-Raviart coupling
    grid = generate_grid(Triangle, (2, 1))
    topology = ExclusiveTopology(grid)
    dh = DofHandler(grid)
    add!(dh, :u, CrouzeixRaviart{RefTriangle, 1}())
    close!(dh)
    coupling = trues(3, 3)
    K = allocate_matrix(dh; coupling = coupling, topology = topology, interface_coupling = coupling)
    K_cont = allocate_matrix(dh; coupling = coupling, topology = topology, interface_coupling = falses(3, 3))
    K_default = allocate_matrix(dh)
    @test K == K_cont == K_default
end


@testset "shell on solid face" begin

    # Node numbering:
    # 3 ____ 4  4
    # |      |  |
    # |      |  | (Beam attached to facet)
    # 1 ____ 2  2

    dim = 2
    grid = generate_grid(Quadrilateral, (1, 1))
    line1 = Line((2, 4))
    grid = Grid([grid.cells[1], line1], grid.nodes)

    order = 2
    ip_solid = Lagrange{RefQuadrilateral, order}() #^dim
    ip_shell = Lagrange{RefLine, order}()

    dh = DofHandler(grid)
    sdh_solid = SubDofHandler(dh, Set(1))
    add!(sdh_solid, :u, ip_solid)
    sdh_shell = SubDofHandler(dh, Set(2))
    add!(sdh_shell, :u, ip_shell)
    close!(dh)

    dofsquad = zeros(Int, ndofs_per_cell(dh, 1))
    dofsbeam = zeros(Int, ndofs_per_cell(dh, 2))

    celldofs!(dofsquad, dh, 1)
    celldofs!(dofsbeam, dh, 2)
    @test dofsbeam == [2, 3, 6]

    # Node numbering:
    #            5--------7
    #           /        /|
    #          /        / |
    #         6--------8  |
    #         |        |  3   <-- Shell attached on face (4, 3, 7, 8)
    #         |        | /
    #         |        |/
    #         2--------4

    dim = 2
    grid = generate_grid(Hexahedron, (1, 1, 1))
    shell = Quadrilateral((4, 3, 7, 8))
    grid = Grid([grid.cells[1], shell], grid.nodes)

    order = 2
    ip_solid = Lagrange{RefHexahedron, order}() #^dim
    ip_shell = Lagrange{RefQuadrilateral, order}()

    dh = DofHandler(grid)
    sdh_solid = SubDofHandler(dh, Set(1))
    add!(sdh_solid, :u, ip_solid)
    sdh_shell = SubDofHandler(dh, Set(2))
    add!(sdh_shell, :u, ip_shell)
    Ferrite.close!(dh)

    dofsolid = zeros(Int, ndofs_per_cell(dh, 1))
    dofsshell = zeros(Int, ndofs_per_cell(dh, 2))

    celldofs!(dofsolid, dh, 1)
    celldofs!(dofsshell, dh, 2)

    #Would be nice to have this utility:
    #facedofs!(dofs, dh, FaceIndex(1,4))

    #Shared node dofs
    @test dofsshell[1:4] == [3, 4, 8, 7]
    #Shared edge dofs
    @test dofsshell[5:8] == [11, 20, 15, 19]
    #Shared face dof
    @test dofsshell[9] == 24
end

# Test-local nodal interpolations with multiple interior dofs per edge and face, mimicking
# the dof layout of cubic/quartic Lagrange interpolations. Only what the dof distribution
# needs is implemented (dof index tables and reference coordinates; no shape functions), so
# that Ferrite.permute_and_push! can be tested independently of the interpolations shipped
# by Ferrite.
struct LatticeTestInterpolation{shape, order} <: Ferrite.ScalarInterpolation{shape, order} end
Ferrite.adjust_dofs_during_distribution(::LatticeTestInterpolation) = true
Ferrite.interior_facedofs_on_lattice(::LatticeTestInterpolation) = true

Ferrite.getnbasefunctions(::LatticeTestInterpolation{RefTetrahedron, order}) where {order} = (order + 1) * (order + 2) * (order + 3) ÷ 6
Ferrite.vertexdof_indices(::LatticeTestInterpolation{RefTetrahedron}) = ((1,), (2,), (3,), (4,))
Ferrite.edgedof_interior_indices(::LatticeTestInterpolation{RefTetrahedron, 3}) = ((5, 6), (7, 8), (9, 10), (11, 12), (13, 14), (15, 16))
Ferrite.facedof_interior_indices(::LatticeTestInterpolation{RefTetrahedron, 3}) = ((17,), (18,), (19,), (20,))
Ferrite.edgedof_interior_indices(::LatticeTestInterpolation{RefTetrahedron, 4}) = ((5, 6, 7), (8, 9, 10), (11, 12, 13), (14, 15, 16), (17, 18, 19), (20, 21, 22))
Ferrite.facedof_interior_indices(::LatticeTestInterpolation{RefTetrahedron, 4}) = ((23, 24, 25), (26, 27, 28), (29, 30, 31), (32, 33, 34))
Ferrite.volumedof_interior_indices(::LatticeTestInterpolation{RefTetrahedron, 4}) = (35,)

# Barycentric multi-indices α (with |α| = order) for the nodes, in local dof order: vertex
# dofs, then edge interior dofs (following the local edge direction), then face interior
# dofs (in the lattice enumeration assumed by Ferrite.permute_and_push!), and finally
# volume interior dofs. The node corresponding to α is located at ∑ₜ αₜ xₜ / order, with xₜ
# the reference vertex coordinates.
function lattice_test_tet_multiindices(order::Int)
    tet_edges = ((1, 2), (2, 3), (3, 1), (1, 4), (2, 4), (3, 4))
    tet_faces = ((1, 3, 2), (1, 2, 4), (2, 3, 4), (1, 4, 3))
    αs = NTuple{4, Int}[]
    for v in 1:4 # vertex nodes
        push!(αs, ntuple(t -> t == v ? order : 0, 4))
    end
    for (a, b) in tet_edges # edge interior nodes, from vertex a towards vertex b
        for k in 1:(order - 1)
            push!(αs, ntuple(t -> t == a ? order - k : (t == b ? k : 0), 4))
        end
    end
    q = order - 3 # order of the face interior lattices
    for (a, b, c) in tet_faces # face interior nodes
        for t2 in 0:q, t1 in 0:(q - t2)
            t3 = q - t1 - t2
            push!(αs, ntuple(t -> t == a ? t1 + 1 : (t == b ? t2 + 1 : (t == c ? t3 + 1 : 0)), 4))
        end
    end
    for s3 in 0:(order - 4), s2 in 0:(order - 4 - s3), s1 in 0:(order - 4 - s3 - s2) # volume interior nodes
        push!(αs, (s1 + 1, s2 + 1, s3 + 1, order - 3 - s1 - s2 - s3))
    end
    return αs
end

function Ferrite.reference_coordinates(::LatticeTestInterpolation{RefTetrahedron, order}) where {order}
    return [Vec{3, Float64}((α[2], α[3], α[4]) ./ order) for α in lattice_test_tet_multiindices(order)]
end

Ferrite.getnbasefunctions(::LatticeTestInterpolation{RefHexahedron, 3}) = 64
Ferrite.vertexdof_indices(::LatticeTestInterpolation{RefHexahedron}) = ((1,), (2,), (3,), (4,), (5,), (6,), (7,), (8,))
Ferrite.edgedof_interior_indices(::LatticeTestInterpolation{RefHexahedron, 3}) = (
    (9, 10), (11, 12), (13, 14), (15, 16), (17, 18), (19, 20),
    (21, 22), (23, 24), (25, 26), (27, 28), (29, 30), (31, 32),
)
Ferrite.facedof_interior_indices(::LatticeTestInterpolation{RefHexahedron, 3}) = (
    (33, 34, 35, 36), (37, 38, 39, 40), (41, 42, 43, 44),
    (45, 46, 47, 48), (49, 50, 51, 52), (53, 54, 55, 56),
)
Ferrite.volumedof_interior_indices(::LatticeTestInterpolation{RefHexahedron, 3}) = (57, 58, 59, 60, 61, 62, 63, 64)

# Tensor-product multi-indices (a, b, c) ∈ (1:4)³ for the 64 nodes of the 4×4×4 lattice, in
# the same local dof order as above. The node for (a, b, c) is located at (x_a, x_b, x_c)
# with x the equispaced 1D nodes (-1, -1/3, 1/3, 1).
function lattice_test_hex3_multiindices()
    vertex_idx = (
        (1, 1, 1), (4, 1, 1), (4, 4, 1), (1, 4, 1),
        (1, 1, 4), (4, 1, 4), (4, 4, 4), (1, 4, 4),
    )
    hex_edges = (
        (1, 2), (2, 3), (3, 4), (4, 1), (5, 6), (6, 7),
        (7, 8), (8, 5), (1, 5), (2, 6), (3, 7), (4, 8),
    )
    hex_faces = (
        (1, 4, 3, 2), (1, 2, 6, 5), (2, 3, 7, 6),
        (3, 4, 8, 7), (1, 5, 8, 4), (5, 6, 7, 8),
    )
    αs = NTuple{3, Int}[]
    for v in 1:8 # vertex nodes
        push!(αs, vertex_idx[v])
    end
    for (a, b) in hex_edges # edge interior nodes, from vertex a towards vertex b
        ia, ib = vertex_idx[a], vertex_idx[b]
        for k in 1:2
            push!(αs, ntuple(t -> ia[t] + (k * (ib[t] - ia[t])) ÷ 3, 3))
        end
    end
    for (a, b, _, d) in hex_faces # face interior nodes, i (a→b) fastest, j (a→d) slowest
        ia, ib, id = vertex_idx[a], vertex_idx[b], vertex_idx[d]
        for j in 0:1, i in 0:1
            push!(αs, ntuple(t -> ia[t] + ((i + 1) * (ib[t] - ia[t]) + (j + 1) * (id[t] - ia[t])) ÷ 3, 3))
        end
    end
    for c in (2, 3), b in (2, 3), a in (2, 3) # volume interior nodes
        push!(αs, (a, b, c))
    end
    return αs
end

function Ferrite.reference_coordinates(::LatticeTestInterpolation{RefHexahedron, 3})
    m = (-3, -1, 1, 3) # 1D nodes scaled by 3
    return [Vec{3, Float64}((m[α[1]] / 3, m[α[2]] / 3, m[α[3]] / 3)) for α in lattice_test_hex3_multiindices()]
end

@testset "dof distribution on a mixed-dimensional shared face" begin
    # A 2D cell can share its interior face dofs with a face of a 3D cell through
    # SubDofHandlers. Both cells must associate every shared global dof with the same
    # physical location, regardless of the 2D cell orientation or which SubDofHandler is
    # visited first.
    solid_grid = generate_grid(Hexahedron, (1, 1, 1))
    solid = solid_grid.cells[1]
    nodes = solid_grid.nodes
    face = (4, 3, 7, 8)
    shell_orientations = (
        face,
        (face[2], face[3], face[4], face[1]),
        (face[3], face[4], face[1], face[2]),
        (face[4], face[1], face[2], face[3]),
        reverse(face),
        (face[3], face[2], face[1], face[4]),
        (face[2], face[1], face[4], face[3]),
        (face[1], face[4], face[3], face[2]),
    )

    ip_solid = LatticeTestInterpolation{RefHexahedron, 3}()
    ip_shell = Lagrange{RefQuadrilateral, 3}()
    ipg_solid = Lagrange{RefHexahedron, 1}()
    ipg_shell = Lagrange{RefQuadrilateral, 1}()
    for shell_nodes in shell_orientations, shell_first in (false, true)
        grid = Grid([solid, Quadrilateral(shell_nodes)], nodes)
        dh = DofHandler(grid)
        if shell_first
            sdh_shell = SubDofHandler(dh, Set(2))
            add!(sdh_shell, :u, ip_shell)
            sdh_solid = SubDofHandler(dh, Set(1))
            add!(sdh_solid, :u, ip_solid)
        else
            sdh_solid = SubDofHandler(dh, Set(1))
            add!(sdh_solid, :u, ip_solid)
            sdh_shell = SubDofHandler(dh, Set(2))
            add!(sdh_shell, :u, ip_shell)
        end
        close!(dh)

        dof_location = Dict{Int, Vec{3, Float64}}()
        nclash = 0
        for (cellnr, ip, ipg) in (
                (1, ip_solid, ipg_solid),
                (2, ip_shell, ipg_shell),
            )
            x = getcoordinates(grid, cellnr)
            for (dof, ξ) in zip(celldofs(dh, cellnr), Ferrite.reference_coordinates(ip))
                xdof = sum(Ferrite.reference_shape_value(ipg, ξ, k) * x[k] for k in eachindex(x))
                loc = get!(dof_location, dof, xdof)
                isapprox(loc, xdof; atol = 1.0e-12) || (nclash += 1)
            end
        end
        @test nclash == 0
        shared = intersect(Set(celldofs(dh, 1)), Set(celldofs(dh, 2)))
        @test length(shared) == 16 # 4 vertex + 4 * 2 edge + 4 face dofs
    end
end

@testset "canonical facedof index helpers" begin
    # Brute-force geometric checks of the helpers used by Ferrite.permute_and_push! to
    # adjust face dofs to the orientation of the local face: the local lattice point must
    # map to the index of the canonical lattice point at the same location.

    # Triangular faces: all 6 orientations of a face spanned by nodes (1, 2, 3). The
    # position (scaled by q + 3) of the lattice point with sub-multi-index t relative to
    # the face fc is given by the barycentric weights (t .+ 1) on the face vertices.
    tri_positions = (Vec((0, 0)), Vec((1, 0)), Vec((0, 1)))
    tri_point(t, fc) = (t[1] + 1) * tri_positions[fc[1]] + (t[2] + 1) * tri_positions[fc[2]] + (t[3] + 1) * tri_positions[fc[3]]
    tri_faces = ((1, 2, 3), (2, 3, 1), (3, 1, 2), (1, 3, 2), (3, 2, 1), (2, 1, 3))
    for q in 0:3, local_face in tri_faces
        orientation = Ferrite.SurfaceOrientationInfo(local_face)
        # Canonical enumeration of the lattice points for the sorted face (1, 2, 3)
        canonical_points = [tri_point((t1, t2, q - t1 - t2), (1, 2, 3)) for t2 in 0:q for t1 in 0:(q - t2)]
        cidxs = Int[]
        for t2 in 0:q, t1 in 0:(q - t2)
            x = tri_point((t1, t2, q - t1 - t2), local_face)
            cidx = Ferrite._canonical_facedof_index_triangle(t1, t2, q, orientation)
            push!(cidxs, cidx)
            @test canonical_points[cidx] == x
        end
        @test sort(cidxs) == 1:length(canonical_points) # bijection
    end

    # Quadrilateral faces: all 8 orientations of a face spanned by nodes (1, 2, 3, 4). The
    # position (scaled by (m + 1)²) of the interior lattice point (i, j) relative to the
    # face fc follows from bilinear interpolation of the corners.
    quad_positions = (Vec((0, 0)), Vec((1, 0)), Vec((1, 1)), Vec((0, 1)))
    function quad_point(i, j, m, fc)
        u, v, w = i + 1, j + 1, m + 1
        return (w - u) * (w - v) * quad_positions[fc[1]] + u * (w - v) * quad_positions[fc[2]] +
            u * v * quad_positions[fc[3]] + (w - u) * v * quad_positions[fc[4]]
    end
    quad_faces = (
        (1, 2, 3, 4), (2, 3, 4, 1), (3, 4, 1, 2), (4, 1, 2, 3), # rotations
        (1, 4, 3, 2), (4, 3, 2, 1), (3, 2, 1, 4), (2, 1, 4, 3), # reversed rotations
    )
    for m in 1:3, local_face in quad_faces
        orientation = Ferrite.SurfaceOrientationInfo(local_face)
        canonical_points = [quad_point(i, j, m, (1, 2, 3, 4)) for j in 0:(m - 1) for i in 0:(m - 1)]
        cidxs = Int[]
        for j in 0:(m - 1), i in 0:(m - 1)
            x = quad_point(i, j, m, local_face)
            cidx = Ferrite._canonical_facedof_index_quadrilateral(i, j, m, orientation)
            push!(cidxs, cidx)
            @test canonical_points[cidx] == x
        end
        @test sort(cidxs) == 1:length(canonical_points) # bijection
    end
end

@testset "interior_facedofs_on_lattice opt-in" begin
    # Permuting multiple dofs on a shared 3D face requires opting in to the lattice
    # assumption; interpolations that have not opted in must error rather than silently
    # produce a wrong (non-lattice) permutation.
    @test Ferrite.interior_facedofs_on_lattice(LatticeTestInterpolation{RefTetrahedron, 4}())
    @test Ferrite.interior_facedofs_on_lattice(LatticeTestInterpolation{RefTetrahedron, 4}()^3)
    @test Ferrite.interior_facedofs_on_lattice(Lagrange{RefQuadrilateral, 3}())
    @test !Ferrite.interior_facedofs_on_lattice(Nedelec{RefTetrahedron, 1}()) # default

    orientation = Ferrite.SurfaceOrientationInfo((2, 3, 1)) # a rotated triangular face
    dofs = 1:1:3 # three interior face dofs, n_copies = 1
    # rdim = 3, adjust = true, multiple dofs, not on lattice => error
    @test_throws ErrorException Ferrite.permute_and_push!(Int[], dofs, orientation, true, false, 3, 3)
    # On a lattice it permutes the three dofs without error
    cell_dofs = Int[]
    Ferrite.permute_and_push!(cell_dofs, dofs, orientation, true, true, 3, 3)
    @test length(cell_dofs) == 3
    # A lattice interpolation on a 2D cell uses the same canonical face ordering so it can
    # share these dofs with a 3D face.
    cell_dofs_2d = Int[]
    Ferrite.permute_and_push!(cell_dofs_2d, dofs, orientation, true, true, 3, 2)
    @test cell_dofs_2d == cell_dofs
    # Non-lattice face dofs on a 2D cell retain their local ordering.
    cell_dofs_2d_nonlattice = Int[]
    Ferrite.permute_and_push!(cell_dofs_2d_nonlattice, dofs, orientation, true, false, 3, 2)
    @test cell_dofs_2d_nonlattice == collect(dofs)
end

@testset "dof distribution on shared faces" begin
    # Two cells sharing an entity must associate the same global dof with the same location
    # on the entity, regardless of the relative orientation of the cells. For the cubic
    # lattice interpolation this requires adjusting multiple dofs per edge, and for the
    # quartic one additionally multiple dofs per face, see Ferrite.permute_and_push!.
    all_permutations(t::NTuple{4, Int}) = [
        (t[i], t[j], t[k], t[l]) for i in 1:4 for j in 1:4 for k in 1:4 for l in 1:4
            if length(unique((i, j, k, l))) == 4
    ]
    nodes = Node.(
        [
            Vec((0.0, 0.0, 0.0)), Vec((1.0, 0.0, 0.0)), Vec((0.0, 1.0, 0.0)),
            Vec((0.0, 0.0, 1.0)), Vec((1.0, 1.0, 1.0)),
        ]
    )
    ipg = Lagrange{RefTetrahedron, 1}() # geometric interpolation of Tetrahedron
    for (ip, nshared) in (
            (Lagrange{RefTetrahedron, 2}(), 6),  # 3 vertex + 3 * 1 edge dofs
            (LatticeTestInterpolation{RefTetrahedron, 3}(), 10), # 3 vertex + 3 * 2 edge + 1 face dofs
            (LatticeTestInterpolation{RefTetrahedron, 4}(), 15), # 3 vertex + 3 * 3 edge + 3 face dofs
            (LatticeTestInterpolation{RefTetrahedron, 4}()^2, 30),
        )
        base_ip = ip isa VectorizedInterpolation ? ip.ip : ip
        n_copies = ip isa VectorizedInterpolation ? Ferrite.get_n_copies(ip) : 1
        ξs = Ferrite.reference_coordinates(base_ip)
        # Loop over all orderings of the cell vertices for both cells. The cells share the
        # face spanned by nodes (2, 3, 4).
        for tet1 in all_permutations((1, 2, 3, 4)), tet2 in all_permutations((2, 3, 4, 5))
            grid = Grid([Tetrahedron(tet1), Tetrahedron(tet2)], nodes)
            dh = close!(add!(DofHandler(grid), :u, ip))
            # Compute the location of each global dof from each cell and check consistency
            dof_location = Dict{Int, Tuple{Vec{3, Float64}, Int}}()
            nclash = 0
            for cellnr in 1:2
                x = getcoordinates(grid, cellnr)
                cdofs = celldofs(dh, cellnr)
                for (i, ξ) in pairs(ξs)
                    xdof = sum(Ferrite.reference_shape_value(ipg, ξ, k) * x[k] for k in 1:length(x))
                    for c in 1:n_copies
                        dof = cdofs[(i - 1) * n_copies + c]
                        loc = get!(dof_location, dof, (xdof, c))
                        if !(isapprox(loc[1], xdof; atol = 1.0e-12) && loc[2] == c)
                            nclash += 1
                        end
                    end
                end
            end
            @test nclash == 0
            # Check that the expected number of dofs are shared between the cells
            shared = intersect(Set(celldofs(dh, 1)), Set(celldofs(dh, 2)))
            @test length(shared) == nshared
            @test ndofs(dh) == 2 * getnbasefunctions(ip) - nshared
        end
    end
end

@testset "dof distribution on shared faces (hexahedron)" begin
    # Two hexahedra sharing a quadrilateral face must associate the same global dof with the
    # same location on the face for any relative orientation of the cells. For the tricubic
    # lattice interpolation the shared face carries multiple interior dofs, exercising the
    # quadrilateral branch of Ferrite.permute_and_push!.

    # Centered coordinates of the 8 hex corners in Ferrite (reference) node ordering.
    corner = (
        (-1, -1, -1), (1, -1, -1), (1, 1, -1), (-1, 1, -1),
        (-1, -1, 1), (1, -1, 1), (1, 1, 1), (-1, 1, 1),
    )
    col_perms = ((1, 2, 3), (1, 3, 2), (2, 1, 3), (2, 3, 1), (3, 1, 2), (3, 2, 1))
    matvec(R, v) = ntuple(r -> sum(R[r][c] * v[c] for c in 1:3), 3)
    det3(R) =
        R[1][1] * (R[2][2] * R[3][3] - R[2][3] * R[3][2]) -
        R[1][2] * (R[2][1] * R[3][3] - R[2][3] * R[3][1]) +
        R[1][3] * (R[2][1] * R[3][2] - R[2][2] * R[3][1])
    # All 24 proper rotations of the cube, as permutations of the corner slots that keep the
    # (positively oriented) hexahedron valid.
    rotations = NTuple{8, Int}[]
    for cols in col_perms, s1 in (-1, 1), s2 in (-1, 1), s3 in (-1, 1)
        s = (s1, s2, s3)
        R = ntuple(r -> ntuple(c -> (c == cols[r] ? s[r] : 0), 3), 3)
        det3(R) == 1 || continue
        push!(rotations, ntuple(j -> findfirst(==(matvec(R, corner[j])), corner), 8))
    end
    @test length(rotations) == 24

    # Two stacked unit cubes: the bottom cube (nodes 1-8) and the top cube (nodes 5-8 shared
    # with the bottom cube, plus new nodes 9-12), sharing the z = 1 face.
    nodes = Node.(
        [
            Vec((0.0, 0.0, 0.0)), Vec((1.0, 0.0, 0.0)), Vec((1.0, 1.0, 0.0)), Vec((0.0, 1.0, 0.0)),
            Vec((0.0, 0.0, 1.0)), Vec((1.0, 0.0, 1.0)), Vec((1.0, 1.0, 1.0)), Vec((0.0, 1.0, 1.0)),
            Vec((0.0, 0.0, 2.0)), Vec((1.0, 0.0, 2.0)), Vec((1.0, 1.0, 2.0)), Vec((0.0, 1.0, 2.0)),
        ]
    )
    bottom = (1, 2, 3, 4, 5, 6, 7, 8)
    top = (5, 6, 7, 8, 9, 10, 11, 12)
    ipg = Lagrange{RefHexahedron, 1}() # geometric interpolation of Hexahedron
    for (ip, nshared) in (
            (Lagrange{RefHexahedron, 2}(), 9),  # 4 vertex + 4 * 1 edge + 1 face dofs
            (LatticeTestInterpolation{RefHexahedron, 3}(), 16), # 4 vertex + 4 * 2 edge + 4 face dofs
            (LatticeTestInterpolation{RefHexahedron, 3}()^2, 32),
        )
        base_ip = ip isa VectorizedInterpolation ? ip.ip : ip
        n_copies = ip isa VectorizedInterpolation ? Ferrite.get_n_copies(ip) : 1
        ξs = Ferrite.reference_coordinates(base_ip)
        # Loop over all rotations of both cells (i.e. all relative orientations of the shared
        # face).
        for rot1 in rotations, rot2 in rotations
            h1 = Hexahedron(ntuple(k -> bottom[rot1[k]], 8))
            h2 = Hexahedron(ntuple(k -> top[rot2[k]], 8))
            grid = Grid([h1, h2], nodes)
            dh = close!(add!(DofHandler(grid), :u, ip))
            # Compute the location of each global dof from each cell and check consistency
            dof_location = Dict{Int, Tuple{Vec{3, Float64}, Int}}()
            nclash = 0
            for cellnr in 1:2
                x = getcoordinates(grid, cellnr)
                cdofs = celldofs(dh, cellnr)
                for (i, ξ) in pairs(ξs)
                    xdof = sum(Ferrite.reference_shape_value(ipg, ξ, k) * x[k] for k in 1:length(x))
                    for c in 1:n_copies
                        dof = cdofs[(i - 1) * n_copies + c]
                        loc = get!(dof_location, dof, (xdof, c))
                        if !(isapprox(loc[1], xdof; atol = 1.0e-12) && loc[2] == c)
                            nclash += 1
                        end
                    end
                end
            end
            @test nclash == 0
            # Check that the expected number of dofs are shared between the cells
            shared = intersect(Set(celldofs(dh, 1)), Set(celldofs(dh, 2)))
            @test length(shared) == nshared
            @test ndofs(dh) == 2 * getnbasefunctions(ip) - nshared
        end
    end
end
