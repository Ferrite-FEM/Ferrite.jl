# misc dofhandler unit tests
@testset "dofs" begin

# set up a test DofHandler
grid = generate_grid(Triangle, (10, 10))
dh = DofHandler(grid)
add!(dh, :u, 2, Lagrange{2,RefTetrahedron,2}())
add!(dh, :p, 1, Lagrange{2,RefTetrahedron,1}())
close!(dh)

# dof_range
@test (@inferred dof_range(dh, :u)) == 1:12
@test (@inferred dof_range(dh, :p)) == 13:15
# dof_range for FieldHandler (use with MixedDofHandler)
ip = Lagrange{2, RefTetrahedron, 1}()
field_u = Field(:u, ip, 2)
field_c = Field(:c, ip, 1)
fh = FieldHandler([field_u, field_c], Set(1:getncells(grid)))
@test dof_range(fh, :u) == 1:6
@test dof_range(fh, :c) == 7:9

end # testset

@testset "Dofs for Line2" begin

nodes = [Node{2,Float64}(Vec(0.0,0.0)), Node{2,Float64}(Vec(1.0,1.0)), Node{2,Float64}(Vec(2.0,0.0))]
cells = [Line2D((1,2)), Line2D((2,3))]
grid = Grid(cells,nodes)

#2d line with 1st order 1d interpolation
dh = DofHandler(grid)
add!(dh, :x, 2)
close!(dh)

@test celldofs(dh,1) == [1,2,3,4]
@test celldofs(dh,2) == [3,4,5,6]

#2d line with 2nd order 1d interpolation
dh = DofHandler(grid)
add!(dh, :x, 2, Lagrange{1,RefCube,2}())
close!(dh)

@test celldofs(dh,1) == [1,2,3,4,5,6]
@test celldofs(dh,2) == [3,4,7,8,9,10]

#3d line with 2nd order 1d interpolation
dh = DofHandler(grid)
add!(dh, :u, 3, Lagrange{1,RefCube,2}())
add!(dh, :θ, 3, Lagrange{1,RefCube,2}())
close!(dh)

@test celldofs(dh,1) == collect(1:18)
@test celldofs(dh,2) == [4,5,6,19,20,21,22,23,24,    # u
                         13,14,15,25,26,27,28,29,30] # θ
end

@testset "Dofs for quad in 3d (shell)" begin

nodes = [Node{3,Float64}(Vec(0.0,0.0,0.0)), Node{3,Float64}(Vec(1.0,0.0,0.0)), 
            Node{3,Float64}(Vec(1.0,1.0,0.0)), Node{3,Float64}(Vec(0.0,1.0,0.0)),
            Node{3,Float64}(Vec(2.0,0.0,0.0)), Node{3,Float64}(Vec(2.0,2.0,0.0))]

cells = [Quadrilateral3D((1,2,3,4)), Quadrilateral3D((2,5,6,3))]
grid = Grid(cells,nodes)

#3d quad with 1st order 2d interpolation
dh = DofHandler(grid)
add!(dh, :u, 3, Lagrange{2,RefCube,1}())
add!(dh, :θ, 3, Lagrange{2,RefCube,1}())
close!(dh)

@test celldofs(dh,1) == collect(1:24)
@test celldofs(dh,2) == [4,5,6,25,26,27,28,29,30,7,8,9, # u
                         16,17,18,31,32,33,34,35,36,19,20,21]# θ

#3d quads with two quadratic interpolations fields
#Only 1 dim per field for simplicity...
dh = DofHandler(grid)
add!(dh, :u, 1, Lagrange{2,RefCube,2}())
add!(dh, :θ, 1, Lagrange{2,RefCube,2}())
close!(dh)

@test celldofs(dh,1) == collect(1:18)
@test celldofs(dh,2) == [2, 19, 20, 3, 21, 22, 23, 6, 24, 11, 25, 26, 12, 27, 28, 29, 15, 30]

# test reshape_to_nodes
## DofHandler
mesh = generate_grid(Quadrilateral, (1,1))
dh = DofHandler(mesh)
add!(dh, :v, 2)
add!(dh, :s, 1)
close!(dh)

u = [1.1, 1.2, 2.1, 2.2, 4.1, 4.2, 3.1, 3.2, 1.3, 2.3, 4.3, 3.3]

s_nodes = reshape_to_nodes(dh, u, :s)
@test s_nodes ≈ [i+0.3 for i=1:4]'
v_nodes = reshape_to_nodes(dh, u, :v)
@test v_nodes ≈ [i==3 ? 0.0 : j+i/10 for i=1:3, j=1:4]
end

@testset "renumber!" begin
    function dhmdhch()
        local dh, mdh, ch
        grid = generate_grid(Triangle, (10, 10))
        dh = DofHandler(grid)
        add!(dh, :u, 1)
        close!(dh)
        mdh = MixedDofHandler(grid)
        add!(mdh, FieldHandler([Field(:u, Lagrange{2,RefTetrahedron,1}(), 1)], Set(1:getncells(grid)÷2)))
        add!(mdh, FieldHandler([Field(:u, Lagrange{2,RefTetrahedron,1}(), 1)], Set((getncells(grid)÷2+1):getncells(grid))))
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
    original_dofs_mdh = copy(mdh.cell_dofs.values)
    renumber!(mdh, perm)
    renumber!(mdh, iperm)
    @test original_dofs_mdh == mdh.cell_dofs.values
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
    K = create_sparsity_pattern(dh, ch)
    f = zeros(ndofs(dh))
    a = start_assemble(K, f)
    dhp, _, chp = dhmdhch()
    renumber!(dhp, chp, perm)
    Kp = create_sparsity_pattern(dhp, chp)
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
        add!(dh, :v, 2)
        add!(dh, :s, 1)
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
    add!(dh, :u, 2)
    add!(dh, :p, 1)
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

    # Full coupling (default)
    K = create_sparsity_pattern(dh)
    for j in 1:ndofs(dh), i in 1:ndofs(dh)
        @test is_stored(K, i, j)
    end

    # Field coupling
    coupling = [
    #   u    p
        true true  # v
        true false # q
    ]
    K = create_sparsity_pattern(dh; coupling=coupling)
    Kch = create_sparsity_pattern(dh, ch; coupling=coupling)
    @test K.rowval == Kch.rowval
    @test K.colptr == Kch.colptr
    KS = create_symmetric_sparsity_pattern(dh; coupling=coupling)
    KSch = create_symmetric_sparsity_pattern(dh, ch; coupling=coupling)
    @test KS.data.rowval == KSch.data.rowval
    @test KS.data.colptr == KSch.data.colptr
    for j in udofs, i in Iterators.flatten((vdofs, qdofs))
        @test is_stored(K, i, j)
        @test is_stored(KS, i, j) == (i <= j)
    end
    for j in pdofs, i in vdofs
        @test is_stored(K, i, j)
        @test is_stored(KS, i, j)
    end
    for j in pdofs, i in qdofs
        @test is_stored(K, i, j) == (i == j)
        @test is_stored(KS, i, j) == (i == j)
    end

    # Component coupling
    coupling = [
    #   u1    u2    p
        true  true  false # v1
        true  false true  # v2
        false true  true  # q
    ]
    K = create_sparsity_pattern(dh; coupling=coupling)
    KS = create_symmetric_sparsity_pattern(dh; coupling=coupling)
    for j in u1dofs, i in vdofs
        @test is_stored(K, i, j)
        @test is_stored(KS, i, j) == (i <= j)
    end
    for j in u1dofs, i in qdofs
        @test !is_stored(K, i, j)
        @test !is_stored(KS, i, j)
    end
    for j in u2dofs, i in Iterators.flatten((v1dofs, qdofs))
        @test is_stored(K, i, j)
        @test is_stored(KS, i, j) == (i <= j)
    end
    for j in u2dofs, i in v2dofs
        @test is_stored(K, i, j) == (i == j)
        @test is_stored(KS, i, j) == (i == j)
    end
    for j in pdofs, i in v1dofs
        @test !is_stored(K, i, j)
        @test !is_stored(KS, i, j)
    end
    for j in pdofs, i in Iterators.flatten((v2dofs, qdofs))
        @test is_stored(K, i, j)
        @test is_stored(KS, i, j) == (i <= j)
    end

    # Error paths
    @test_throws ErrorException("coupling not square") create_sparsity_pattern(dh; coupling=[true true])
    @test_throws ErrorException("coupling not symmetric") create_symmetric_sparsity_pattern(dh; coupling=[true true; false true])
    @test_throws ErrorException("could not create coupling") create_symmetric_sparsity_pattern(dh; coupling=falses(100, 100))
end
