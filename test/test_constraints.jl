# misc constraint tests
@testset "node bc" begin
    grid = generate_grid(Triangle, (1, 1))
    addnodeset!(grid, "nodeset", x-> x[2] == -1 || x[1] == -1)
    dh = DofHandler(grid)
    push!(dh, :u, 2)
    push!(dh, :p, 1)
    close!(dh)
    ch = ConstraintHandler(dh)
    dbc1 = Dirichlet(:u, getnodeset(grid, "nodeset"), (x,t) -> x, [1, 2])
    dbc2 = Dirichlet(:p, getnodeset(grid, "nodeset"), (x,t) -> 0, 1)
    add!(ch, dbc1)
    add!(ch, dbc2)
    close!(ch)
    update!(ch)

    @test ch.prescribed_dofs == collect(1:9)
    @test ch.inhomogeneities == [-1, -1, 1, -1, -1, 1, 0, 0, 0]

    ## test node bc with mixed dof handler
   dim = 2
   mesh = generate_grid(QuadraticQuadrilateral, (2,1))
   addcellset!(mesh, "set1", Set(1))
   addcellset!(mesh, "set2", Set(2))
   addnodeset!(mesh, "bottom", Set(1:5))

   dh  = MixedDofHandler(mesh)

   ip_quadratic = Lagrange{dim, RefCube, 2}()
   ip_linear = Lagrange{dim, RefCube, 1}()
   field_u = Field(:u, ip_quadratic, dim)
   field_c = Field(:c, ip_linear, 1)
   push!(dh, FieldHandler([field_u, field_c], getcellset(mesh, "set1")))
   push!(dh, FieldHandler([field_u], getcellset(mesh, "set2")))

   close!(dh)

   ch = ConstraintHandler(dh)
   add!(ch, dh.fieldhandlers[1], Dirichlet(:u, getnodeset(mesh, "bottom"), (x,t)->1.0, 1))
   add!(ch, dh.fieldhandlers[2], Dirichlet(:u, getnodeset(mesh, "bottom"), (x,t)->1.0, 1))
   add!(ch, dh.fieldhandlers[1], Dirichlet(:c, getnodeset(mesh, "bottom"), (x,t)->2.0, 1))
   close!(ch)
   update!(ch)

   @test ch.prescribed_dofs == [1,3,9,19,20,23,27]
   @test ch.inhomogeneities == [1.0, 1.0, 1.0, 2.0, 2.0, 1.0, 1.0]

   ## MixedDofHandler: let first FieldHandler not have all fields
   dim = 2
   mesh = generate_grid(Quadrilateral, (2,1))
   addcellset!(mesh, "set1", Set(1))
   addcellset!(mesh, "set2", Set(2))

   ip = Lagrange{dim, RefCube, 1}()
   field_u = Field(:u, ip, dim)
   field_c = Field(:c, ip, 1)

   dh = MixedDofHandler(mesh)
   push!(dh, FieldHandler([field_u], getcellset(mesh, "set1")))
   push!(dh, FieldHandler([field_u, field_c], getcellset(mesh, "set2")))
   close!(dh)

   ch = ConstraintHandler(dh)
   add!(ch, dh.fieldhandlers[1], Dirichlet(:u, getfaceset(mesh, "bottom"), (x,t)->1.0, 1))
   add!(ch, dh.fieldhandlers[2], Dirichlet(:u, getfaceset(mesh, "bottom"), (x,t)->1.0, 1))
   add!(ch, dh.fieldhandlers[2], Dirichlet(:c, getfaceset(mesh, "bottom"), (x,t)->2.0, 1))
   close!(ch)
   update!(ch)

   @test ch.prescribed_dofs == [1,3,9,13,14]
   @test ch.inhomogeneities == [1.0, 1.0, 1.0, 2.0, 2.0]
end

@testset "edge bc" begin
    grid = generate_grid(Hexahedron, (1, 1, 1))
    addedgeset!(grid, "edge", x-> x[1] ≈ -1.0 && x[3] ≈ -1.0)

    dh = DofHandler(grid)
    push!(dh, :u, 3)
    push!(dh, :p, 1)
    close!(dh)

    ch = ConstraintHandler(dh)
    dbc1 = Dirichlet(:u, getedgeset(grid, "edge"), (x,t) -> x, [1, 2, 3])
    add!(ch, dbc1)
    close!(ch)
    update!(ch)

    @test ch.prescribed_dofs == [1,2,3,10,11,12]
    @test ch.inhomogeneities == [-1.0, -1.0, -1.0, -1.0, 1.0, -1.0]


    #Shell mesh edge bcs
    nodes = [Node{3,Float64}(Vec(0.0,0.0,0.0)), Node{3,Float64}(Vec(1.0,0.0,0.0)), 
             Node{3,Float64}(Vec(1.0,1.0,0.0)), Node{3,Float64}(Vec(0.0,1.0,0.0)),
             Node{3,Float64}(Vec(2.0,0.0,0.0)), Node{3,Float64}(Vec(2.0,2.0,0.0))]

    cells = [Quadrilateral3D((1,2,3,4)), Quadrilateral3D((2,5,6,3))]
    grid = Grid(cells,nodes)

    #3d quad with 1st order 2d interpolation
    dh = DofHandler(grid)
    push!(dh, :u, 1, Lagrange{2,RefCube,2}())
    push!(dh, :θ, 1, Lagrange{2,RefCube,2}())
    close!(dh)

    addedgeset!(grid, "edge", x -> x[2] ≈ 0.0) #bottom edge
    ch = ConstraintHandler(dh)
    dbc1 = Dirichlet(:θ, getedgeset(grid, "edge"), (x,t) -> (0.0,), [1])
    add!(ch, dbc1)
    close!(ch)
    update!(ch)

    @test ch.prescribed_dofs == [10, 11, 14, 25, 27]
end

@testset "affine constraints" begin

    grid = generate_grid(Line, (10,))
    dh = DofHandler(grid)
    push!(dh, :u, 1)
    close!(dh)

    test_acs = [
        # Simple homogeneous constraint
        [AffineConstraint(4, [(7 => 1.0)], 0.0)],
        # Two dofs and inhomogeneity
        [AffineConstraint(2, [(5 => 1.0), (6 =>2.0)], 1.0)],
        # Two linear constraints
        [AffineConstraint(2, [9=>1.0], 0.0),
         AffineConstraint(3, [9=>1.0], 0.0)],
        #
        [AffineConstraint(2, [7=>3.0, 8=>1.0], -1.0),
         AffineConstraint(4, [9=>-1.0], 2.0)]
    ]

    for acs in test_acs
        ch = ConstraintHandler(dh)
        add!(ch, Dirichlet(:u, getfaceset(grid, "left"), (x,t)->0.0))
        for lc in acs
            add!(ch, lc)
        end
        close!(ch)
        update!(ch, 0.0)
        C, g = Ferrite.create_constraint_matrix(ch)
        
        # Assemble
        K = create_sparsity_pattern(dh, ch)
        f = zeros(ndofs(dh)); f[end] = 1.0
        for cell in CellIterator(dh)
            K[celldofs(cell), celldofs(cell)] += 2.0 * [1 -1; -1 1]
        end
        copies = (K = copy(K), f1 = copy(f), f2 = copy(f))

        # Solve by actually condensing the matrix
        ff  = C' * (f - K * g)
        KK = C' * K * C
        _aa = KK \ ff
        aa = C * _aa + g

        # Solving by modifying K inplace
        apply!(K, f, ch)
        a = K \ f
        apply!(a, ch)

        # Solve with extracted RHS data
        rhs = get_rhs_data(ch, copies.K)
        apply!(copies.K, ch)
        apply_rhs!(rhs, copies.f1, ch)
        a_rhs1 = apply!(copies.K \ copies.f1, ch)
        apply_rhs!(rhs, copies.f2, ch)
        a_rhs2 = apply!(copies.K \ copies.f2, ch)

        @test a ≈ aa ≈ a_rhs1 ≈ a_rhs2
    end

end

# Rotate pi/2 around z
function rotpio2(v)
    v3 = Vec{3}(i -> i <= length(v) ? v[i] : 0.0)
    z = Vec{3}((0.0, 0.0, 1.0))
    rv = Tensors.rotate(v3, z, pi/2)
    return typeof(v)(i -> rv[i])
end

@testset "periodic bc: collect_periodic_faces" begin
    # 1D (TODO: Broken)
    # grid = generate_grid(Line, (2,))
    # face_map = collect_periodic_faces(grid)


    # 2D simple grid
    #       3       3
    #   ┌───────┬───────┐
    #   │       │       │
    # 4 │   3   │   4   │ 2
    #   │       │       │
    #   ├───────┼───────┤
    #   │       │       │
    # 4 │   1   │   2   │ 2
    #   │       │       │
    #   └───────┴───────┘
    #       1       1
    grid = generate_grid(Quadrilateral, (2, 2))
    correct_map = Dict{FaceIndex,FaceIndex}(
        FaceIndex(1, 1) => FaceIndex(3, 3),
        FaceIndex(2, 1) => FaceIndex(4, 3),
        FaceIndex(1, 4) => FaceIndex(2, 2),
        FaceIndex(3, 4) => FaceIndex(4, 2),
    )

    # Brute force path with no boundary info
    face_map = collect_periodic_faces(grid)
    @test face_map == correct_map

    # Brute force path with boundary info
    face_map = collect_periodic_faces(grid,
        union(
            getfaceset(grid, "left"),
            getfaceset(grid, "bottom"),
        ),
        union(
            getfaceset(grid, "right"),
            getfaceset(grid, "top"),
        )
    )
    @test face_map == correct_map

    # Brute force, keeping the mirror/image ordering
    face_map = collect_periodic_faces(grid,
        union(
            getfaceset(grid, "right"),
            getfaceset(grid, "top"),
        ),
        union(
            getfaceset(grid, "left"),
            getfaceset(grid, "bottom"),
        )
    )
    @test face_map == Dict(values(correct_map) .=> keys(correct_map))

    # Known pairs with transformation
    face_map = collect_periodic_faces(grid, "left", "right", x -> x + Vec{2}((1.0, 0.0)))
    collect_periodic_faces!(face_map, grid, "bottom", "top", x -> x + Vec{2}((0.0, 1.0)))
    @test face_map == correct_map

    # More advanced transformation by rotation
    face_map = collect_periodic_faces(grid, "left", "bottom", rotpio2)
    collect_periodic_faces!(face_map, grid, "right", "top", rotpio2)
    @test face_map == Dict{FaceIndex,FaceIndex}(
        FaceIndex(3, 4) => FaceIndex(1, 1),
        FaceIndex(1, 4) => FaceIndex(2, 1),
        FaceIndex(2, 2) => FaceIndex(4, 3),
        FaceIndex(4, 2) => FaceIndex(3, 3),
    )

    # 3D simple grid (TODO: better than just smoke tests...)
    grid = generate_grid(Hexahedron, (2, 2, 2))
    face_map = collect_periodic_faces(grid)
    @test length(face_map) == 12
end # testset

@testset "periodic bc: dof mapping" begin
    grid = generate_grid(Quadrilateral, (2, 2))
    face_map = collect_periodic_faces(grid, "left", "right")
    collect_periodic_faces!(face_map, grid, "bottom", "top")
    function get_dof_map(ch)
        m = Dict{Int,Int}()
        for ac in ch.acs
            mdof = ac.constrained_dof
            @test ac.b == 0
            @test length(ac.entries) == 1
            idof, weight = first(ac.entries)
            @test weight == 1
            m[mdof] = idof
        end
        return m
    end

    #             Distributed dofs
    #       Scalar                Vector
    #  8───────7───────9   15,16───13,14───17,18
    #  │       │       │     │       │       │
    #  │       │       │     │       │       │
    #  │       │       │     │       │       │
    #  4───────3───────6    7,8─────5,6────11,12
    #  │       │       │     │       │       │
    #  │       │       │     │       │       │
    #  │       │       │     │       │       │
    #  1───────2───────5    1,2─────3,4─────9,10

    # Scalar
    dh = DofHandler(grid)
    push!(dh, :s, 1)
    close!(dh)
    ch = ConstraintHandler(dh)
    pbc = PeriodicDirichlet(:s, face_map)
    add!(ch, pbc)
    dof_map = get_dof_map(ch)
    @test dof_map == Dict{Int,Int}(
        1 => 9,
        2 => 7,
        5 => 9,
        4 => 6,
        8 => 9,
    )

    # Vector
    dh = DofHandler(grid)
    push!(dh, :v, 2)
    close!(dh)
    ch = ConstraintHandler(dh)
    pbc = PeriodicDirichlet(:v, face_map, [1, 2])
    add!(ch, pbc)
    dof_map = get_dof_map(ch)
    @test dof_map == Dict{Int,Int}(
        1 => 17, 2 => 18,
        3 => 13, 4 => 14,
        9 => 17, 10 => 18,
        7 => 11, 8 => 12,
        15 => 17, 16 => 18,
    )
    ## Just component #2
    ch = ConstraintHandler(dh)
    pbc = PeriodicDirichlet(:v, face_map, 2)
    add!(ch, pbc)
    dof_map = get_dof_map(ch)
    @test dof_map == Dict{Int,Int}(
        2 => 18,
        4 => 14,
        10 => 18,
        8 => 12,
        16 => 18,
    )

    # Rotation (TODO: No support for rotating the dofs yet!)
    face_map_rot = collect_periodic_faces(grid, "left", "bottom", rotpio2)
    collect_periodic_faces!(face_map_rot, grid, "right", "top", rotpio2)
    ch = ConstraintHandler(dh)
    pbc = PeriodicDirichlet(:v, face_map_rot, [1, 2])
    add!(ch, pbc)
    dof_map = get_dof_map(ch)
    @test_broken dof_map == Dict{Int,Int}(
        # 15 => 15, # 15 -> 1 -> 9 -> 17 -> 15
        # 16 => 16, # 16 -> 2 -> 10 -> 18 -> 16
        7 => 3,
        8 => 4,
        1 => 15, # 1 -> 9 -> 17 -> 15
        2 => 16, # 2 -> 10 -> 18 -> 16
        9 => 17,
        10 => 18,
        11 => 13,
        12 => 14,
        17 => 15,
        18 => 16,
    )
end # testset
