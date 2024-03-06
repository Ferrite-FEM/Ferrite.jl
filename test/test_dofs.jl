@testset "DofHandler construction" begin
    grid = generate_grid(Quadrilateral, (2,1))
    dh = DofHandler(grid)
    # incompatible refshape (#638)
    @test_throws ErrorException add!(dh, :u, Lagrange{RefTriangle, 1}())
    @test_throws ErrorException add!(dh, :u, Lagrange{RefTetrahedron, 1}())
    # field already exists
    add!(dh, :u, Lagrange{RefQuadrilateral, 1}())
    @test_throws ErrorException add!(dh, :u, Lagrange{RefQuadrilateral, 1}())

    # Invalid SubDofHandler construction
    dh = DofHandler(grid)
    sdh1 = Ferrite.SubDofHandler(dh, Set(1,))
    # Subdomains not disjoint
    @test_throws ErrorException Ferrite.SubDofHandler(dh, Set(1:getncells(grid)))
    # add field to DofHandler that has subdomains
    @test_throws ErrorException add!(dh, :u, Lagrange{RefQuadrilateral, 1}())

    # inconsistent field across several SubDofHandlers
    dh = DofHandler(grid)
    sdh1 = Ferrite.SubDofHandler(dh, Set(1,))
    sdh2 = Ferrite.SubDofHandler(dh, Set(2,))
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
add!(dh, :u, Lagrange{RefTriangle,2}()^2)
add!(dh, :p, Lagrange{RefTriangle,1}())
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

nodes = [Node{2,Float64}(Vec(0.0,0.0)), Node{2,Float64}(Vec(1.0,1.0)), Node{2,Float64}(Vec(2.0,0.0))]
cells = [Line((1,2)), Line((2,3))]
grid = Grid(cells,nodes)

#2d line with 1st order 1d interpolation
dh = DofHandler(grid)
add!(dh, :x, Lagrange{RefLine,1}()^2)
close!(dh)

@test celldofs(dh,1) == [1,2,3,4]
@test celldofs(dh,2) == [3,4,5,6]

#2d line with 2nd order 1d interpolation
dh = DofHandler(grid)
add!(dh, :x, Lagrange{RefLine,2}()^2)
close!(dh)

@test celldofs(dh,1) == [1,2,3,4,5,6]
@test celldofs(dh,2) == [3,4,7,8,9,10]

#3d line with 2nd order 1d interpolation
dh = DofHandler(grid)
add!(dh, :u, Lagrange{RefLine,2}()^3)
add!(dh, :θ, Lagrange{RefLine,2}()^3)
close!(dh)

@test celldofs(dh,1) == collect(1:18)
@test celldofs(dh,2) == [4,5,6,19,20,21,22,23,24,    # u
                         13,14,15,25,26,27,28,29,30] # θ
end

@testset "Dofs for quad in 3d (shell)" begin

nodes = [Node{3,Float64}(Vec(0.0,0.0,0.0)), Node{3,Float64}(Vec(1.0,0.0,0.0)), 
            Node{3,Float64}(Vec(1.0,1.0,0.0)), Node{3,Float64}(Vec(0.0,1.0,0.0)),
            Node{3,Float64}(Vec(2.0,0.0,0.0)), Node{3,Float64}(Vec(2.0,2.0,0.0))]

cells = [Quadrilateral((1,2,3,4)), Quadrilateral((2,5,6,3))]
grid = Grid(cells,nodes)

#3d quad with 1st order 2d interpolation
dh = DofHandler(grid)
add!(dh, :u, Lagrange{RefQuadrilateral,1}()^3)
add!(dh, :θ, Lagrange{RefQuadrilateral,1}()^3)
close!(dh)

@test celldofs(dh,1) == collect(1:24)
@test celldofs(dh,2) == [4,5,6,25,26,27,28,29,30,7,8,9, # u
                         16,17,18,31,32,33,34,35,36,19,20,21]# θ

#3d quads with two quadratic interpolations fields
#Only 1 dim per field for simplicity...
dh = DofHandler(grid)
add!(dh, :u, Lagrange{RefQuadrilateral,2}())
add!(dh, :θ, Lagrange{RefQuadrilateral,2}())
close!(dh)

@test celldofs(dh,1) == collect(1:18)
@test celldofs(dh,2) == [2, 19, 20, 3, 21, 22, 23, 6, 24, 11, 25, 26, 12, 27, 28, 29, 15, 30]

# test evaluate_at_grid_nodes
## DofHandler
mesh = generate_grid(Quadrilateral, (1,1))
dh = DofHandler(mesh)
add!(dh, :v, Lagrange{RefQuadrilateral,1}()^2)
add!(dh, :s, Lagrange{RefQuadrilateral,1}())
close!(dh)

u = [1.1, 1.2, 2.1, 2.2, 4.1, 4.2, 3.1, 3.2, 1.3, 2.3, 4.3, 3.3]

s_nodes = evaluate_at_grid_nodes(dh, u, :s)
@test s_nodes ≈ [i+0.3 for i=1:4]
v_nodes = evaluate_at_grid_nodes(dh, u, :v)
@test v_nodes ≈ [Vec{2,Float64}(i -> j+i/10) for j = 1:4]
end

@testset "renumber!" begin
    function dhmdhch()
        local dh, mdh, ch
        grid = generate_grid(Triangle, (10, 10))
        dh = DofHandler(grid)
        add!(dh, :u, Lagrange{RefTriangle,1}())
        close!(dh)
        # subdomains
        mdh = DofHandler(grid)
        sdh1 = SubDofHandler(mdh, Set(1:getncells(grid)÷2))
        add!(sdh1, :u, Lagrange{RefTriangle,1}())
        sdh2 = SubDofHandler(mdh, Set((getncells(grid)÷2+1):getncells(grid)))
        add!(sdh2, :u, Lagrange{RefTriangle,1}())
        close!(mdh)
        ch = ConstraintHandler(dh)
        add!(ch, Dirichlet(:u, getfaceset(grid, "left"), (x, t) -> 0))
        add!(ch, Dirichlet(:u, getfaceset(grid, "right"), (x, t) -> 2))
        face_map = collect_periodic_faces(grid, "bottom", "top")
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
    K = create_matrix(dh, ch)
    f = zeros(ndofs(dh))
    a = start_assemble(K, f)
    dhp, _, chp = dhmdhch()
    renumber!(dhp, chp, perm)
    Kp = create_matrix(dhp, chp)
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
        add!(dh, :v, Lagrange{RefQuadrilateral,1}()^2)
        add!(dh, :s, Lagrange{RefQuadrilateral,1}())
        close!(dh)
        ch = ConstraintHandler(dh)
        add!(ch, Dirichlet(:v, getfaceset(grid, "left"), (x, t) -> 0, [2]))
        add!(ch, Dirichlet(:s, getfaceset(grid, "left"), (x, t) -> 0))
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
        ip = Lagrange{RefQuadrilateral,1}()
        dh = DofHandler(grid)
        sdh1 = SubDofHandler(dh, Set(1))
        add!(sdh1, :v, ip^2)
        add!(sdh1, :s, ip)
        sdh2 = SubDofHandler(dh, Set(2))
        add!(sdh2, :v, ip^2)
        close!(dh)
        ch = ConstraintHandler(dh)
        add!(ch, Dirichlet(:v, getfaceset(grid, "left"), (x, t) -> 0, [2]))
        add!(ch, Dirichlet(:s, getfaceset(grid, "left"), (x, t) -> 0))
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
    @test celldofs(dh, 2) == [12, 2, 15, 5, 16, 6, 13, 3, ]
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
    if HAS_EXTENSIONS && MODULE_CAN_BE_TYPE_PARAMETER
        # TODO: Should probably test that the new order result in less fill-in
        dh, ch = testdhch()
        renumber!(dh, DofOrder.Ext{Metis}())
        @test_throws ErrorException renumber!(dh, ch, DofOrder.Ext{Metis}())
        renumber!(dh, DofOrder.Ext{Metis}(coupling=[true true; true false]))
        @test_throws ErrorException renumber!(dh, ch, DofOrder.Ext{Metis}(coupling=[true true; true false]))
    end
end

@testset "dof coupling" begin
    grid = generate_grid(Quadrilateral, (1, 1))
    dh = DofHandler(grid)
    add!(dh, :u, Lagrange{RefQuadrilateral,1}()^2)
    add!(dh, :p, Lagrange{RefQuadrilateral,1}())
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
    sparsity_pattern = create_sparsity_pattern(dh)
    K = create_matrix(sparsity_pattern)
    @test eltype(K) == Float64
    for j in 1:ndofs(dh), i in 1:ndofs(dh)
        @test is_stored(sparsity_pattern, i, j)
        @test is_stored(K, i, j)
    end

    # Field coupling
    coupling = [
    #   u    p
        true true  # v
        true false # q
    ]
    sparsity_pattern = create_sparsity_pattern(dh; coupling=coupling)
    K = create_matrix(sparsity_pattern)
    # Kch = create_matrix(dh, ch; coupling=coupling)
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
    #   u1    u2    p
        true  true  false # v1
        true  false true  # v2
        false true  true  # q
    ]
    sparsity_pattern = create_sparsity_pattern(dh; coupling=coupling)
    K = create_matrix(sparsity_pattern)
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
    @test_throws ErrorException("coupling not square") create_matrix(dh; coupling=[true true])
    # @test_throws ErrorException("coupling not symmetric") create_symmetric_sparsity_pattern(dh; coupling=[true true; false true])
    # @test_throws ErrorException("could not create coupling") create_symmetric_sparsity_pattern(dh; coupling=falses(100, 100))

    # Test coupling with subdomains
    grid = generate_grid(Quadrilateral, (1, 2))
    dh = DofHandler(grid)
    sdh1 = SubDofHandler(dh, Set(1))
    add!(sdh1, :u, Lagrange{RefQuadrilateral,1}()^2)
    add!(sdh1, :p, Lagrange{RefQuadrilateral,1}())
    sdh2 = SubDofHandler(dh, Set(2))
    add!(sdh2, :u, Lagrange{RefQuadrilateral,1}()^2)
    close!(dh)

    sparsity_pattern = create_sparsity_pattern(dh; coupling = [true true; true false])
    K = create_matrix(sparsity_pattern)
    KS = Symmetric(create_matrix(dh; #= symmetric=true, =# coupling = [true true; true false]))
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
                        # is_cross_element && !all(Ferrite.is_discontinuous.([ip1, ip2])) && continue
                        for  dim2 in 1:vdim[2]
                            i_dofs_v = i_dofs[dim1:vdim[1]:end]
                            j_dofs_v = j_dofs[dim2:vdim[2]:end]
                            for i_idx in i_dofs_v, j_idx in j_dofs_v
                                i = celldofs(dh,cell_idx)[i_idx]
                                j = celldofs(dh,cell2_idx)[j_idx]
                                is_cross_element && (i ∈ celldofs(dh,cell2_idx) || j ∈ celldofs(dh,cell_idx)) && continue
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
    function check_coupling(dh, topology, K, coupling, cross_coupling)
        for cell_idx in eachindex(getcells(dh.grid))
            sdh = dh.subdofhandlers[dh.cell_to_subdofhandler[cell_idx]]
            coupling_idx = [1,1]
            cross_coupling_idx = [1,1]
            vdim = [1,1]
            # test inner coupling
            _check_dofs(K, dh, sdh, cell_idx, coupling, coupling_idx, vdim, [cell_idx], false)
            # test cross-element coupling
            neighborhood = Ferrite.getdim(dh.grid.cells[1]) > 1 ? topology.face_face_neighbor : topology.vertex_vertex_neighbor
            neighbors = neighborhood[cell_idx, :]
            _check_dofs(K, dh, sdh, cell_idx, cross_coupling, cross_coupling_idx, vdim, [i[1][1] for i in  neighbors[.!isempty.(neighbors)]], true)
        end
    end
    grid = generate_grid(Quadrilateral, (2, 2))
    topology = ExclusiveTopology(grid)
    dh = DofHandler(grid)
    add!(dh, :u, DiscontinuousLagrange{RefQuadrilateral,1}()^2)
    add!(dh, :p, DiscontinuousLagrange{RefQuadrilateral,1}())
    add!(dh, :w, Lagrange{RefQuadrilateral,1}())
    close!(dh)
    for coupling in couplings, cross_coupling in couplings
        K = create_matrix(dh; coupling=coupling, topology = topology, cross_coupling = cross_coupling)
        all(coupling) && @test K == create_matrix(dh, topology = topology, cross_coupling = cross_coupling) 
        check_coupling(dh, topology, K, coupling, cross_coupling)
    end

    # Error paths
    @test_throws ErrorException("coupling not square") create_matrix(dh; coupling=[true true])
    # @test_throws ErrorException("coupling not symmetric") create_matrix(dh; coupling=[true true; false true])
    @test_throws ErrorException("could not create coupling") create_matrix(dh; coupling=falses(100, 100))
 
    # Test coupling with subdomains
    # Note: `check_coupling` works for this case only because the second domain has dofs from the first domain in order. Otherwise tests like in continuous ip are required.
    grid = generate_grid(Quadrilateral, (2, 1))
    topology = ExclusiveTopology(grid)

    dh = DofHandler(grid)
    sdh1 = SubDofHandler(dh, Set(1))
    add!(sdh1, :u, DiscontinuousLagrange{RefQuadrilateral,1}()^2)
    add!(sdh1, :y, DiscontinuousLagrange{RefQuadrilateral,1}())
    add!(sdh1, :p, Lagrange{RefQuadrilateral,1}())
    sdh2 = SubDofHandler(dh, Set(2))
    add!(sdh2, :u, DiscontinuousLagrange{RefQuadrilateral,1}()^2)
    close!(dh)

    for coupling in couplings, cross_coupling in couplings
        K = create_matrix(dh; coupling=coupling, topology = topology, cross_coupling = cross_coupling)
        all(coupling) && @test K == create_matrix(dh, topology = topology, cross_coupling = cross_coupling)
        check_coupling(dh, topology, K, coupling, cross_coupling)
    end

    # Testing Crouzeix-Raviart coupling
    grid = generate_grid(Triangle, (2, 1))
    topology = ExclusiveTopology(grid)
    dh = DofHandler(grid)
    add!(dh, :u, CrouzeixRaviart{RefTriangle,1}())
    close!(dh)
    coupling = trues(3,3)
    K = create_matrix(dh; coupling=coupling, topology = topology, cross_coupling = coupling)
    K_cont = create_matrix(dh; coupling=coupling, topology = topology, cross_coupling = falses(3,3))
    K_default = create_matrix(dh)
    @test K == K_cont == K_default
end
