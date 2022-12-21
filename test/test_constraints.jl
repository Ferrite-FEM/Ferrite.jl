# misc constraint tests

@testset "constructors and error checking" begin
    grid = generate_grid(Triangle, (2, 2))
    Γ = getfaceset(grid, "left")
    face_map = collect_periodic_faces(grid, "left", "right")
    dh = DofHandler(grid)
    push!(dh, :s, 1)
    push!(dh, :v, 2)
    close!(dh)
    ch = ConstraintHandler(dh)

    # Dirichlet
    @test_throws ErrorException("components are empty: $(Int)[]") Dirichlet(:u, Γ, (x, t) -> 0, Int[])
    @test_throws ErrorException("components not sorted: [2, 1]") Dirichlet(:u, Γ, (x, t) -> 0, Int[2, 1])
    @test_throws ErrorException("components not unique: [2, 2]") Dirichlet(:u, Γ, (x, t) -> 0, Int[2, 2])
    ## Scalar
    dbc = Dirichlet(:s, Γ, (x, t) -> 0)
    add!(ch, dbc)
    @test dbc.components == [1]
    dbc = Dirichlet(:s, Γ, (x, t) -> 0, [1])
    add!(ch, dbc)
    @test dbc.components == [1]
    dbc = Dirichlet(:s, Γ, (x, t) -> 0, [1, 2])
    @test_throws ErrorException("components [1, 2] not within range of field :s (1 dimension(s))") add!(ch, dbc)
    dbc = Dirichlet(:p, Γ, (x, t) -> 0)
    @test_throws ErrorException("could not find field :p in DofHandler (existing fields: [:s, :v])") add!(ch, dbc)
    ## Vector
    dbc = Dirichlet(:v, Γ, (x, t) -> 0)
    add!(ch, dbc)
    @test dbc.components == [1, 2]
    dbc = Dirichlet(:v, Γ, (x, t) -> 0, [1])
    add!(ch, dbc)
    @test dbc.components == [1]
    dbc = Dirichlet(:v, Γ, (x, t) -> 0, [2, 3])
    @test_throws ErrorException("components [2, 3] not within range of field :v (2 dimension(s))") add!(ch, dbc)

    # PeriodicDirichlet
    @test_throws ErrorException("components are empty: $(Int)[]") PeriodicDirichlet(:u, face_map, Int[])
    @test_throws ErrorException("components not sorted: [2, 1]") PeriodicDirichlet(:u, face_map, Int[2, 1])
    @test_throws ErrorException("components not unique: [2, 2]") PeriodicDirichlet(:u, face_map, Int[2, 2])
    ## Scalar
    pdbc = PeriodicDirichlet(:s, face_map)
    add!(ConstraintHandler(dh), pdbc)
    @test pdbc.components == [1]
    pdbc = PeriodicDirichlet(:s, face_map, (x,t) -> 0)
    add!(ConstraintHandler(dh), pdbc)
    @test pdbc.components == [1]
    pdbc = PeriodicDirichlet(:s, face_map, [1])
    add!(ConstraintHandler(dh), pdbc)
    @test pdbc.components == [1]
    pdbc = PeriodicDirichlet(:s, face_map, [1, 2])
    @test_throws ErrorException("components [1, 2] not within range of field :s (1 dimension(s))") add!(ConstraintHandler(dh), pdbc)
    pdbc = PeriodicDirichlet(:p, face_map)
    @test_throws ErrorException("could not find field :p in DofHandler (existing fields: [:s, :v])") add!(ConstraintHandler(dh), pdbc)
    ## Vector
    pdbc = PeriodicDirichlet(:v, face_map)
    add!(ConstraintHandler(dh), pdbc)
    @test pdbc.components == [1, 2]
    pdbc = PeriodicDirichlet(:v, face_map, (x, t) -> 0*x)
    add!(ConstraintHandler(dh), pdbc)
    @test pdbc.components == [1, 2]
    pdbc = PeriodicDirichlet(:v, face_map, rand(2,2))
    add!(ConstraintHandler(dh), pdbc)
    @test pdbc.components == [1, 2]
    pdbc = PeriodicDirichlet(:v, face_map, rand(1,1))
    @test_throws ErrorException("size of rotation matrix does not match the number of components") add!(ConstraintHandler(dh), pdbc)
    pdbc = PeriodicDirichlet(:v, face_map, (x, t) -> 0, [1])
    add!(ConstraintHandler(dh), pdbc)
    @test pdbc.components == [1]
    pdbc = PeriodicDirichlet(:v, face_map, (x, t) -> 0, [2, 3])
    @test_throws ErrorException("components [2, 3] not within range of field :v (2 dimension(s))") add!(ConstraintHandler(dh), pdbc)
    pdbc = PeriodicDirichlet(:v, face_map, rand(2, 2), [2, 3])
    @test_throws ErrorException("components [2, 3] not within range of field :v (2 dimension(s))") add!(ConstraintHandler(dh), pdbc)
end

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
        Kl = create_sparsity_pattern(dh, ch)
        fl = copy(f)
        assembler = start_assemble(Kl, fl)
        for cell in CellIterator(dh)
            local_matrix = 2.0 * [1 -1; -1 1]
            local_vector = zeros(2)
            K[celldofs(cell), celldofs(cell)] += local_matrix
            # Assemble with local condensation
            apply_assemble!(assembler, ch, celldofs(cell), local_matrix, local_vector)
        end
        fl[end] += 1
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

        # Solve with matrix assembled by local condensation
        al = Kl \ fl
        apply!(al, ch)

        # Solve with extracted RHS data
        rhs = get_rhs_data(ch, copies.K)
        apply!(copies.K, ch)
        apply_rhs!(rhs, copies.f1, ch)
        a_rhs1 = apply!(copies.K \ copies.f1, ch)
        apply_rhs!(rhs, copies.f2, ch)
        a_rhs2 = apply!(copies.K \ copies.f2, ch)

        @test a ≈ aa ≈ al ≈ a_rhs1 ≈ a_rhs2
    end

end

# Rotate -pi/2 around dir
function rotpio2(v, dir=3)
    v3 = Vec{3}(i -> i <= length(v) ? v[i] : 0.0)
    z = Vec{3}(i -> i == dir ? 1.0 : 0.0)
    rv = Tensors.rotate(v3, z, -pi/2)
    return typeof(v)(i -> rv[i])
end

@testset "periodic bc: collect_periodic_faces" begin
    # 1D (TODO: Broken)
    # grid = generate_grid(Line, (2,))
    # face_map = collect_periodic_faces(grid)


    # 2D quad grid
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
    for grid  in (generate_grid(Quadrilateral, (2, 2)), generate_grid(QuadraticQuadrilateral, (2, 2)))
        correct_map = [
            PeriodicFacePair(FaceIndex(1, 1), FaceIndex(3, 3), 0x00, true),
            PeriodicFacePair(FaceIndex(2, 1), FaceIndex(4, 3), 0x00, true),
            PeriodicFacePair(FaceIndex(1, 4), FaceIndex(2, 2), 0x00, true),
            PeriodicFacePair(FaceIndex(3, 4), FaceIndex(4, 2), 0x00, true),
        ]

        # Brute force path with no boundary info
        face_map = collect_periodic_faces(grid)
        @test issetequal(face_map, correct_map)

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
        @test issetequal(face_map, correct_map)

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
        @test issetequal(face_map, map(x -> PeriodicFacePair(x.image, x.mirror, x.rotation, x.mirrored), correct_map))

        # Known pairs with transformation
        face_map = collect_periodic_faces(grid, "left", "right", x -> x - Vec{2}((2.0, 0.0)))
        collect_periodic_faces!(face_map, grid, "bottom", "top", x -> x - Vec{2}((0.0, 2.0)))
        @test issetequal(face_map, correct_map)

        # More advanced transformation by rotation
        face_map = collect_periodic_faces(grid, "left", "bottom", rotpio2)
        collect_periodic_faces!(face_map, grid, "right", "top", rotpio2)
        @test issetequal(face_map, [
            PeriodicFacePair(FaceIndex(3, 4), FaceIndex(1, 1), 0x00, false),
            PeriodicFacePair(FaceIndex(1, 4), FaceIndex(2, 1), 0x00, false),
            PeriodicFacePair(FaceIndex(2, 2), FaceIndex(4, 3), 0x00, false),
            PeriodicFacePair(FaceIndex(4, 2), FaceIndex(3, 3), 0x00, false),
        ])

        # Rotate and translate
        face_map = collect_periodic_faces(grid, "bottom", "left", x -> rotpio2(x) - Vec{2}((0.0, 2.0)))
        @test issetequal(face_map, [
            PeriodicFacePair(FaceIndex(1, 1), FaceIndex(1, 4), 0x00, true),
            PeriodicFacePair(FaceIndex(2, 1), FaceIndex(3, 4), 0x00, true),
        ])
    end

    ####################################################################

    # 2D tri grid
    #       2       2
    #   ┌───────┬───────┐
    #   │ \   6 │ \   8 │
    # 3 │   \   │   \   │ 1
    #   │ 5   \ │ 7   \ │
    #   ├───────┼───────┤
    #   │ \   2 │ \   4 │
    # 3 │   \   │   \   │ 1
    #   │ 1   \ │3    \ │
    #   └───────┴───────┘
    #       1       1

    for grid in (generate_grid(Triangle, (2, 2)), generate_grid(QuadraticTriangle, (2, 2)))

        correct_map = [
            PeriodicFacePair(FaceIndex(1, 1), FaceIndex(6, 2), 0x00, true),
            PeriodicFacePair(FaceIndex(3, 1), FaceIndex(8, 2), 0x00, true),
            PeriodicFacePair(FaceIndex(1, 3), FaceIndex(4, 1), 0x00, true),
            PeriodicFacePair(FaceIndex(5, 3), FaceIndex(8, 1), 0x00, true),
        ]

        # Brute force path with no boundary info
        face_map = collect_periodic_faces(grid)
        @test issetequal(face_map, correct_map)

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
        @test issetequal(face_map, correct_map)

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
        @test issetequal(face_map, map(x -> PeriodicFacePair(x.image, x.mirror, x.rotation, x.mirrored), correct_map))

        # Known pairs with transformation
        face_map = collect_periodic_faces(grid, "left", "right", x -> x - Vec{2}((2.0, 0.0)))
        collect_periodic_faces!(face_map, grid, "bottom", "top", x -> x - Vec{2}((0.0, 2.0)))
        @test issetequal(face_map, correct_map)

        # More advanced transformation by rotation
        face_map = collect_periodic_faces(grid, "left", "bottom", rotpio2)
        collect_periodic_faces!(face_map, grid, "right", "top", rotpio2)
        @test issetequal(face_map, [
            PeriodicFacePair(FaceIndex(5, 3), FaceIndex(1, 1), 0x00, false),
            PeriodicFacePair(FaceIndex(1, 3), FaceIndex(3, 1), 0x00, false),
            PeriodicFacePair(FaceIndex(4, 1), FaceIndex(8, 2), 0x00, false),
            PeriodicFacePair(FaceIndex(8, 1), FaceIndex(6, 2), 0x00, false),
        ])

        # Rotate and translate
        face_map = collect_periodic_faces(grid, "bottom", "left", x -> rotpio2(x) - Vec{2}((0.0, 2.0)))
        @test issetequal(face_map, [
            PeriodicFacePair(FaceIndex(1, 1), FaceIndex(1, 3), 0x00, true),
            PeriodicFacePair(FaceIndex(3, 1), FaceIndex(5, 3), 0x00, true),
        ])
    end

    ####################################################################

    # 3D hex grids
    grid = generate_grid(Hexahedron, (1, 1, 1))
    face_map = collect_periodic_faces(grid)
    @test issetequal(face_map, [
        PeriodicFacePair(FaceIndex(1, 1), FaceIndex(1, 6), 0x00, true),
        PeriodicFacePair(FaceIndex(1, 2), FaceIndex(1, 4), 0x03, true),
        PeriodicFacePair(FaceIndex(1, 5), FaceIndex(1, 3), 0x00, true),
    ])

    grid = generate_grid(Hexahedron, (2, 2, 2))
    face_map = collect_periodic_faces(grid, "left", "right", x -> x - Vec{3}((2.0, 0.0, 0.0)))
    collect_periodic_faces!(face_map, grid, "bottom", "top")
    collect_periodic_faces!(face_map, grid, "front", "back")
    @test issetequal(face_map, [
        PeriodicFacePair(FaceIndex(1, 5), FaceIndex(2, 3), 0x00, true),
        PeriodicFacePair(FaceIndex(3, 5), FaceIndex(4, 3), 0x00, true),
        PeriodicFacePair(FaceIndex(5, 5), FaceIndex(6, 3), 0x00, true),
        PeriodicFacePair(FaceIndex(7, 5), FaceIndex(8, 3), 0x00, true),
        PeriodicFacePair(FaceIndex(1, 1), FaceIndex(5, 6), 0x00, true),
        PeriodicFacePair(FaceIndex(2, 1), FaceIndex(6, 6), 0x00, true),
        PeriodicFacePair(FaceIndex(3, 1), FaceIndex(7, 6), 0x00, true),
        PeriodicFacePair(FaceIndex(4, 1), FaceIndex(8, 6), 0x00, true),
        PeriodicFacePair(FaceIndex(1, 2), FaceIndex(3, 4), 0x03, true),
        PeriodicFacePair(FaceIndex(2, 2), FaceIndex(4, 4), 0x03, true),
        PeriodicFacePair(FaceIndex(5, 2), FaceIndex(7, 4), 0x03, true),
        PeriodicFacePair(FaceIndex(6, 2), FaceIndex(8, 4), 0x03, true),
    ])

    # Rotation
    grid = generate_grid(Hexahedron, (2, 2, 2))
    face_map = collect_periodic_faces(grid, "left", "front", rotpio2)
    @test issetequal(face_map, [
        PeriodicFacePair(FaceIndex(1, 5), FaceIndex(2, 2), 0x03, false),
        PeriodicFacePair(FaceIndex(3, 5), FaceIndex(1, 2), 0x03, false),
        PeriodicFacePair(FaceIndex(5, 5), FaceIndex(6, 2), 0x03, false),
        PeriodicFacePair(FaceIndex(7, 5), FaceIndex(5, 2), 0x03, false),
    ])

    # Rotation and translation
    grid = generate_grid(Hexahedron, (2, 2, 2))
    face_map = collect_periodic_faces(grid, "front", "left", x -> rotpio2(x) - Vec{3}((0.0, 2.0, 0.0)))
    @test issetequal(face_map, [
        PeriodicFacePair(FaceIndex(1, 2), FaceIndex(1, 5), 0x00, true),
        PeriodicFacePair(FaceIndex(2, 2), FaceIndex(3, 5), 0x00, true),
        PeriodicFacePair(FaceIndex(5, 2), FaceIndex(5, 5), 0x00, true),
        PeriodicFacePair(FaceIndex(6, 2), FaceIndex(7, 5), 0x00, true),
    ])

    ####################################################################

    # 3D tetra grid
    grid = generate_grid(Tetrahedron, (1, 1, 1))
    face_map = collect_periodic_faces(grid)
    @test issetequal(face_map, [
        PeriodicFacePair(FaceIndex(1, 4), FaceIndex(4, 1), 0x00, true)
        PeriodicFacePair(FaceIndex(2, 2), FaceIndex(6, 1), 0x00, true)
        PeriodicFacePair(FaceIndex(2, 1), FaceIndex(3, 3), 0x02, true)
        PeriodicFacePair(FaceIndex(5, 1), FaceIndex(4, 3), 0x02, true)
        PeriodicFacePair(FaceIndex(1, 1), FaceIndex(5, 3), 0x00, true)
        PeriodicFacePair(FaceIndex(3, 1), FaceIndex(6, 3), 0x00, true)
    ])

    grid = generate_grid(Tetrahedron, (2, 2, 2))
    face_map = collect_periodic_faces(grid, "left", "right", x -> x - Vec{3}((2.0, 0.0, 0.0)))
    collect_periodic_faces!(face_map, grid, "bottom", "top")
    collect_periodic_faces!(face_map, grid, "front", "back")
    @test issetequal(face_map, [
        PeriodicFacePair(FaceIndex(1, 4), FaceIndex(10, 1), 0x00, true)
        PeriodicFacePair(FaceIndex(2, 2), FaceIndex(12, 1), 0x00, true)
        PeriodicFacePair(FaceIndex(13, 4), FaceIndex(22, 1), 0x00, true)
        PeriodicFacePair(FaceIndex(14, 2), FaceIndex(24, 1), 0x00, true)
        PeriodicFacePair(FaceIndex(25, 4), FaceIndex(34, 1), 0x00, true)
        PeriodicFacePair(FaceIndex(26, 2), FaceIndex(36, 1), 0x00, true)
        PeriodicFacePair(FaceIndex(37, 4), FaceIndex(46, 1), 0x00, true)
        PeriodicFacePair(FaceIndex(38, 2), FaceIndex(48, 1), 0x00, true)
        PeriodicFacePair(FaceIndex(2, 1), FaceIndex(15, 3), 0x02, true)
        PeriodicFacePair(FaceIndex(5, 1), FaceIndex(16, 3), 0x02, true)
        PeriodicFacePair(FaceIndex(8, 1), FaceIndex(21, 3), 0x02, true)
        PeriodicFacePair(FaceIndex(11, 1), FaceIndex(22, 3), 0x02, true)
        PeriodicFacePair(FaceIndex(26, 1), FaceIndex(39, 3), 0x02, true)
        PeriodicFacePair(FaceIndex(29, 1), FaceIndex(40, 3), 0x02, true)
        PeriodicFacePair(FaceIndex(32, 1), FaceIndex(45, 3), 0x02, true)
        PeriodicFacePair(FaceIndex(35, 1), FaceIndex(46, 3), 0x02, true)
        PeriodicFacePair(FaceIndex(1, 1), FaceIndex(29, 3), 0x00, true)
        PeriodicFacePair(FaceIndex(3, 1), FaceIndex(30, 3), 0x00, true)
        PeriodicFacePair(FaceIndex(7, 1), FaceIndex(35, 3), 0x00, true)
        PeriodicFacePair(FaceIndex(9, 1), FaceIndex(36, 3), 0x00, true)
        PeriodicFacePair(FaceIndex(13, 1), FaceIndex(41, 3), 0x00, true)
        PeriodicFacePair(FaceIndex(15, 1), FaceIndex(42, 3), 0x00, true)
        PeriodicFacePair(FaceIndex(19, 1), FaceIndex(47, 3), 0x00, true)
        PeriodicFacePair(FaceIndex(21, 1), FaceIndex(48, 3), 0x00, true)
    ])

    # Rotation
    grid = generate_grid(Tetrahedron, (1, 1, 1))
    face_map = collect_periodic_faces(grid, "left", "front", rotpio2)
    @test issetequal(face_map, [
        PeriodicFacePair(FaceIndex(1, 4), FaceIndex(2, 1), 0x02, false)
        PeriodicFacePair(FaceIndex(2, 2), FaceIndex(5, 1), 0x00, false)
    ])

    # Rotation and translation
    grid = generate_grid(Tetrahedron, (1, 1, 1))
    face_map = collect_periodic_faces(grid, "front", "left", x -> rotpio2(rotate(x, Vec{3}((1., 0., 0.)), 3pi/2)) - Vec{3}((0.0, 2.0, 0.0)))
    @test issetequal(face_map, [
        PeriodicFacePair(FaceIndex(2, 1), FaceIndex(1, 4), 0x01, true)
        PeriodicFacePair(FaceIndex(5, 1), FaceIndex(2, 2), 0x01, true)
    ])
end # testset

@testset "periodic bc: dof mapping" begin
    grid = generate_grid(Quadrilateral, (2, 2))

    function get_dof_map(ch)
        m = Dict{Int,Any}()
        for (mdof,b,entries) in zip(ch.prescribed_dofs, ch.inhomogeneities, ch.dofcoefficients)
            if entries !== nothing
                @test b == 0
                if length(entries) == 1
                    idof, weight = first(entries)
                    @test weight == 1
                    m[mdof] = idof
                else
                    m[mdof] = entries
                end
            end
        end
        return m
    end
    function compare_by_dbc(dh, pdbc, dbc1, dbc2)
        ch = ConstraintHandler(dh)
        add!(ch, pdbc)
        close!(ch)
        ch1 = ConstraintHandler(dh)
        add!(ch1, dbc1)
        close!(ch1)
        ch2 = ConstraintHandler(dh)
        add!(ch2, dbc2)
        close!(ch2)
        dof_map = get_dof_map(ch)
        @test issetequal(keys(dof_map), ch1.prescribed_dofs)
        @test issetequal(values(dof_map), ch2.prescribed_dofs)
    end


    #             Distributed dofs
    #       Scalar                Vector
    #  8───────7───────9   15,16───13,14───17,18
    #  │       │       │     │       │       │
    #  │   3   │   4   │     │       │       │
    #  │       │       │     │       │       │
    #  4───────3───────6    7,8─────5,6────11,12
    #  │       │       │     │       │       │
    #  │   1   │   2   │     │       │       │
    #  │       │       │     │       │       │
    #  1───────2───────5    1,2─────3,4─────9,10

    # Scalar
    dh = DofHandler(grid)
    push!(dh, :s, 1)
    close!(dh)
    ch = ConstraintHandler(dh)
    face_map = collect_periodic_faces(grid, "left", "right")
    collect_periodic_faces!(face_map, grid, "bottom", "top")
    pbc = PeriodicDirichlet(:s, face_map)
    add!(ch, pbc)
    @test get_dof_map(ch) == Dict{Int,Int}(
        1 => 9,
        2 => 7,
        5 => 9,
        4 => 6,
        8 => 9,
    )

    # Rotation
    ch = ConstraintHandler(dh)
    face_map = collect_periodic_faces(grid, "left", "bottom", rotpio2)
    pbc = PeriodicDirichlet(:s, face_map)
    add!(ch, pbc)
    @test get_dof_map(ch) == Dict{Int,Int}(
        8 => 5, # 8 -> 1 -> 5
        4 => 2,
        1 => 5,
    )

    # Rotation and translation
    ch = ConstraintHandler(dh)
    face_map = collect_periodic_faces(grid, "bottom", "left", x -> rotpio2(x) - Vec{2}((0.0, 2.0)))
    pbc = PeriodicDirichlet(:s, face_map)
    add!(ch, pbc)
    @test get_dof_map(ch) == Dict{Int,Int}(
        # 1 => 1,
        2 => 4,
        5 => 8,
    )

    # Vector
    dh = DofHandler(grid)
    push!(dh, :v, 2)
    close!(dh)
    ch = ConstraintHandler(dh)
    face_map = collect_periodic_faces(grid, "left", "right")
    collect_periodic_faces!(face_map, grid, "bottom", "top")
    pbc = PeriodicDirichlet(:v, face_map, [1, 2])
    add!(ch, pbc)
    @test get_dof_map(ch) == Dict{Int,Int}(
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
    @test get_dof_map(ch) == Dict{Int,Int}(
        2 => 18,
        4 => 14,
        10 => 18,
        8 => 12,
        16 => 18,
    )

    # Rotation without dof rotation
    face_map = collect_periodic_faces(grid, "left", "bottom", rotpio2)
    ch = ConstraintHandler(dh)
    pbc = PeriodicDirichlet(:v, face_map, [1, 2])
    add!(ch, pbc)
    @test get_dof_map(ch) == Dict{Int,Int}(
        15 => 9, # 15 -> 1 -> 9
        16 => 10, # 16 -> 2 -> 10
        7 => 3,
        8 => 4,
        1 => 9,
        2 => 10,
    )

    # Rotation with dof rotation
    face_map = collect_periodic_faces(grid, "left", "bottom", rotpio2)
    ch = ConstraintHandler(dh)
    pbc = PeriodicDirichlet(:v, face_map, rotation_tensor(-π/2), [1, 2])
    add!(ch, pbc)
    dof_map = get_dof_map(ch)
    correct_dof_map = Dict{Int,Any}(
        15 => [9 => 0, 10 => 1], # 15 -> 2 -> 10
        16 => [9 => -1, 10 => 0], # 16 -> -1 -> -9
        7 => [3 => 0, 4 => 1],
        8 => [3 => -1, 4 => 0],
        1 => [9 => 0, 10 => 1],
        2 => [9 => -1, 10 => 0],
    )
    @test length(dof_map) == length(correct_dof_map)
    for (k, v) in dof_map
        cmap = correct_dof_map[k]
        @test first.(v) == first.(cmap)
        @test last.(v) ≈ last.(cmap)
    end

    # Rotation and translation
    face_map = collect_periodic_faces(grid, "bottom", "left", x -> rotpio2(x) - Vec{2}((0.0, 2.0)))
    ch = ConstraintHandler(dh)
    pbc = PeriodicDirichlet(:v, face_map, [1, 2])
    add!(ch, pbc)
    @test get_dof_map(ch) == Dict{Int,Int}(
        # 1 => 1,
        # 2 => 2,
        3 => 7,
        4 => 8,
        9 => 15,
        10 => 16,
    )

    # Quadratic interpolation
    # 17───19──16──24──22
    #  │       │       │
    # 20   21  18  25  23
    #  │       │       │
    #  4───7───3───14──11
    #  │       │       │
    #  8   9   6   15  13
    #  │       │       │
    #  1───5───2───12──10
    dh = DofHandler(grid)
    push!(dh, :s, 1, Lagrange{2,RefCube,2}())
    close!(dh)
    ch = ConstraintHandler(dh)
    face_map = collect_periodic_faces(grid, "left", "right")
    collect_periodic_faces!(face_map, grid, "bottom", "top")
    pbc = PeriodicDirichlet(:s, face_map)
    add!(ch, pbc)
    @test get_dof_map(ch) == Dict{Int,Int}(
        1 => 22, # 1 -> 10/17 -> 22
        8 => 13,
        4 => 11,
        20 => 23,
        17 => 22,
        5 => 19,
        2 => 16,
        12 => 24,
        10 => 22,
    )

    # Rotation
    ch = ConstraintHandler(dh)
    face_map = collect_periodic_faces(grid, "left", "bottom", rotpio2)
    pbc = PeriodicDirichlet(:s, face_map)
    add!(ch, pbc)
    @test get_dof_map(ch) == Dict{Int,Int}(
        17 => 10, # 17 -> 1 -> 10
        20 => 5,
        4 => 2,
        8 => 12,
        1 => 10,
    )

    # Rotation and translation
    ch = ConstraintHandler(dh)
    face_map = collect_periodic_faces(grid, "bottom", "left", x -> rotpio2(x) - Vec{2}((0.0, 2.0)))
    pbc = PeriodicDirichlet(:s, face_map)
    add!(ch, pbc)
    @test get_dof_map(ch) == Dict{Int,Int}(
        # 1 => 1,
        5 => 8,
        2 => 4,
        12 => 20,
        10 => 17,
    )

    # Face rotation with dof rotation
    # 33,34──37,38──31,32──47,48──43,44
    #   │             │             │
    #   │             │             │
    # 39,40  41,42  35,36  49,50  45,46
    #   │             │             │
    #   │             │             │
    #  7,8───13,14───5,6───27,28──21,22
    #   │             │             │
    #   │             │             │
    # 15,16  17,18  11,12  29,30  25,26
    #   │             │             │
    #   │             │             │
    #  1,2────9,10───3,4───23,24──19,20
    dh = DofHandler(grid)
    push!(dh, :v, 2, Lagrange{2,RefCube,2}())
    close!(dh)
    ch = ConstraintHandler(dh)
    face_map = collect_periodic_faces(grid, "left", "bottom", rotpio2)
    pbc = PeriodicDirichlet(:v, face_map, rotation_tensor(-π/2), [1, 2])
    add!(ch, pbc)
    close!(ch)
    dof_map = get_dof_map(ch)
    correct_dof_map = Dict{Int,Any}(
        33 => [19 => 0, 20 => 1], # 33 -> 2 -> 20
        34 => [19 => -1, 20 => 0], # 34 -> -1 -> -19
        39 => [9 => 0, 10 => 1],
        40 => [9 => -1, 10 => 0],
        7 => [3 => 0, 4 => 1],
        8 => [3 => -1, 4 => 0],
        15 => [23 => 0, 24 => 1],
        16 => [23 => -1, 24 => 0],
        1 => [19 => 0, 20 => 1],
        2 => [19 => -1, 20 => 0],
    )
    @test length(dof_map) == length(correct_dof_map)
    for (k, v) in dof_map
        cmap = correct_dof_map[k]
        @test first.(v) == first.(cmap)
        @test last.(v) ≈ last.(cmap)
    end

    # 3D hex scalar/vector
    grid = generate_grid(Hexahedron, (1, 1, 1))
    face_map = collect_periodic_faces(grid)
    dh = DofHandler(grid)
    push!(dh, :s, 1)
    push!(dh, :v, 2)
    close!(dh)

    ch = ConstraintHandler(dh)
    pbc = PeriodicDirichlet(:s, face_map)
    add!(ch, pbc)
    pbc = PeriodicDirichlet(:v, face_map, [1, 2])
    add!(ch, pbc)
    @test get_dof_map(ch) == Dict{Int,Int}(
        1 => 7, 2 => 7,
        3 => 7, 4 => 7,
        5 => 7, 6 => 7,
        8 => 7,
        9 => 21, 11 => 21,
        13 => 21, 15 => 21,
        17 => 21, 19 => 21,
        23 => 21,
        10 => 22, 12 => 22,
        14 => 22, 16 => 22,
        18 => 22, 20 => 22,
        24 => 22,
    )

    # Rotation
    ch = ConstraintHandler(dh)
    face_map = collect_periodic_faces(grid, "left", "front", rotpio2)
    pbc = PeriodicDirichlet(:s, face_map)
    add!(ch, pbc)
    @test get_dof_map(ch) == Dict{Int,Int}(
        1 => 2,
        4 => 2, # 4 -> 1 -> 2
        5 => 6,
        8 => 6, # 8 -> 5 -> 6
    )

    ch = ConstraintHandler(dh)
    face_map = collect_periodic_faces(grid, "front", "left", x -> rotpio2(x) - Vec{3}((0.0, 2.0, 0.0)))
    pbc = PeriodicDirichlet(:s, face_map)
    add!(ch, pbc)
    @test get_dof_map(ch) == Dict{Int,Int}(
        # 1 => 1,
        # 5 => 5,
        2 => 4,
        6 => 8,
    )

    # Quadratic interpolation
    grid = generate_grid(Hexahedron, (5, 5, 5))
    dh = DofHandler(grid)
    push!(dh, :s, 1, Lagrange{3,RefCube,2}())
    push!(dh, :v, 2, Lagrange{3,RefCube,2}())
    close!(dh)

    compare_by_dbc(
        dh,
        PeriodicDirichlet(:s, collect_periodic_faces(grid, "left", "right")),
        Dirichlet(:s, getfaceset(grid, "left"), (x, t) -> 0.),
        Dirichlet(:s, getfaceset(grid, "right"), (x, t) -> 0.),
    )

    compare_by_dbc(
        dh,
        PeriodicDirichlet(:v, collect_periodic_faces(grid, "left", "right"), [1, 2]),
        Dirichlet(:v, getfaceset(grid, "left"), (x, t) -> [0., 0.], [1, 2]),
        Dirichlet(:v, getfaceset(grid, "right"), (x, t) -> [0., 0.], [1, 2]),
    )

    compare_by_dbc(
        dh,
        PeriodicDirichlet(:v, collect_periodic_faces(grid, "left", "right"), [2]),
        Dirichlet(:v, getfaceset(grid, "left"), (x, t) -> 0., [2]),
        Dirichlet(:v, getfaceset(grid, "right"), (x, t) -> 0., [2]),
    )

    # 3D tetra scalar
    grid = generate_grid(Tetrahedron, (1, 1, 1))
    dh = DofHandler(grid)
    push!(dh, :s, 1)
    close!(dh)
    face_map = collect_periodic_faces(grid)
    ch = ConstraintHandler(dh)
    pbc = PeriodicDirichlet(:s, face_map)
    add!(ch, pbc)
    close!(ch)
    @test get_dof_map(ch) == Dict{Int,Int}(
        1 => 7, 2 => 7, 3 => 7, 4 => 7, 5 => 7, 6 => 7, 8 => 7,
    )

    # 3D tetra vector
    grid = generate_grid(Tetrahedron, (2, 1, 1))
    dh = DofHandler(grid)
    push!(dh, :v, 2)
    close!(dh)
    face_map = collect_periodic_faces(grid, "left", "right")
    ch = ConstraintHandler(dh)
    pbc = PeriodicDirichlet(:v, face_map, [1, 2])
    add!(ch, pbc)
    close!(ch)
    @test get_dof_map(ch) == Dict{Int,Int}(
        1 => 17,
        2 => 18,
        9 => 23,
        10 => 24,
        7 => 21,
        8 => 22,
        5 => 19,
        6 => 20,
    )

    # 3D hex vector with dof rotation
    grid = generate_grid(Hexahedron, (1, 1, 1))
    dh = DofHandler(grid)
    push!(dh, :v, 3)
    close!(dh)
    rot = rotation_tensor(Vec{3}((0., 1., 0.)), π/2)
    face_map = collect_periodic_faces(grid, "left", "bottom", x -> rot ⋅ x)
    ch = ConstraintHandler(dh)
    pbc = PeriodicDirichlet(:v, face_map, rot, [1, 2, 3])
    add!(ch, pbc)
    close!(ch)
    dof_map = get_dof_map(ch)
    correct_dof_map = Dict{Int,Any}(
        1 => [4 => 0, 5 => 0, 6 => 1],
        2 => [4 => 0, 5 => 1, 6 => 0],
        3 => [4 => -1, 5 => 0, 6 => 0],
        10 => [7 => 0, 8 => 0, 9 => 1],
        11 => [7 => 0, 8 => 1, 9 => 0],
        12 => [7 => -1, 8 => 0, 9 => 0],
        13 => [4 => 0, 5 => 0, 6 => 1],
        14 => [4 => 0, 5 => 1, 6 => 0],
        15 => [4 => -1, 5 => 0, 6 => 0],
        22 => [7 => 0, 8 => 0, 9 => 1],
        23 => [7 => 0, 8 => 1, 9 => 0],
        24 => [7 => -1, 8 => 0, 9 => 0],
    )
    @test length(dof_map) == length(correct_dof_map)
    for (k, v) in dof_map
        cmap = correct_dof_map[k]
        @test first.(v) == first.(cmap)
        @test last.(v) ≈ last.(cmap)
    end

    for (D, CT, IT) in (
        (2, Quadrilateral, Lagrange{2,RefCube,1}()),
        (2, Quadrilateral, Lagrange{2,RefCube,2}()),
        (2, Triangle, Lagrange{2,RefTetrahedron,1}()),
        (2, Triangle, Lagrange{2,RefTetrahedron,2}()),
        (3, Hexahedron, Lagrange{3,RefCube,1}()),
        (3, Hexahedron, Lagrange{3,RefCube,2}()),
        (3, Tetrahedron, Lagrange{3,RefTetrahedron,1}()),
        (3, Tetrahedron, Lagrange{3,RefTetrahedron,2}()),
    )
        grid = generate_grid(CT, ntuple(i -> 5, D))
        dh = DofHandler(grid)
        push!(dh, :s, 1, IT)
        push!(dh, :v, D, IT)
        close!(dh)

        # Scalar
        compare_by_dbc(
            dh,
            PeriodicDirichlet(:s, collect_periodic_faces(grid, "left", "right")),
            Dirichlet(:s, getfaceset(grid, "left"), (x,t) -> 0),
            Dirichlet(:s, getfaceset(grid, "right"), (x,t) -> 0),
        )
        compare_by_dbc(
            dh,
            PeriodicDirichlet(:s, collect_periodic_faces(grid, "right", "left")),
            Dirichlet(:s, getfaceset(grid, "right"), (x,t) -> 0),
            Dirichlet(:s, getfaceset(grid, "left"), (x,t) -> 0),
        )
        compare_by_dbc(
            dh,
            PeriodicDirichlet(:s, collect_periodic_faces(grid, "bottom", "top")),
            Dirichlet(:s, getfaceset(grid, "bottom"), (x,t) -> 0),
            Dirichlet(:s, getfaceset(grid, "top"), (x,t) -> 0),
        )
        compare_by_dbc(
            dh,
            PeriodicDirichlet(:s, collect_periodic_faces(grid, "top", "bottom")),
            Dirichlet(:s, getfaceset(grid, "top"), (x,t) -> 0),
            Dirichlet(:s, getfaceset(grid, "bottom"), (x,t) -> 0),
        )
        if D == 3
            compare_by_dbc(
                dh,
                PeriodicDirichlet(:s, collect_periodic_faces(grid, "front", "back")),
                Dirichlet(:s, getfaceset(grid, "front"), (x,t) -> 0),
                Dirichlet(:s, getfaceset(grid, "back"), (x,t) -> 0),
            )
            compare_by_dbc(
                dh,
                PeriodicDirichlet(:s, collect_periodic_faces(grid, "back", "front")),
                Dirichlet(:s, getfaceset(grid, "back"), (x,t) -> 0),
                Dirichlet(:s, getfaceset(grid, "front"), (x,t) -> 0),
            )
        end

        # Vector
        compare_by_dbc(
            dh,
            PeriodicDirichlet(:v, collect_periodic_faces(grid, "left", "right"), collect(1:D)),
            Dirichlet(:v, getfaceset(grid, "left"), (x,t) -> fill(0., D), collect(1:D)),
            Dirichlet(:v, getfaceset(grid, "right"), (x,t) -> fill(0., D), collect(1:D)),
        )
        compare_by_dbc(
            dh,
            PeriodicDirichlet(:v, collect_periodic_faces(grid, "right", "left"), [D-1]),
            Dirichlet(:v, getfaceset(grid, "right"), (x,t) -> 0, [D-1]),
            Dirichlet(:v, getfaceset(grid, "left"), (x,t) -> 0, [D-1]),
        )
        compare_by_dbc(
            dh,
            PeriodicDirichlet(:v, collect_periodic_faces(grid, "bottom", "top"), [1, 2]),
            Dirichlet(:v, getfaceset(grid, "bottom"), (x,t) -> [0., 0.], [1, 2]),
            Dirichlet(:v, getfaceset(grid, "top"), (x,t) -> [0., 0.], [1, 2]),
        )
        compare_by_dbc(
            dh,
            PeriodicDirichlet(:v, collect_periodic_faces(grid, "top", "bottom"), [D]),
            Dirichlet(:v, getfaceset(grid, "top"), (x,t) -> 0, [D]),
            Dirichlet(:v, getfaceset(grid, "bottom"), (x,t) -> 0, [D]),
        )
        if D == 3
            compare_by_dbc(
                dh,
                PeriodicDirichlet(:v, collect_periodic_faces(grid, "front", "back"), 1:D),
                Dirichlet(:v, getfaceset(grid, "front"), (x,t) -> fill(0., D), 1:D),
                Dirichlet(:v, getfaceset(grid, "back"), (x,t) -> fill(0., D), 1:D),
            )
            compare_by_dbc(
                dh,
                PeriodicDirichlet(:v, collect_periodic_faces(grid, "back", "front"), D),
                Dirichlet(:v, getfaceset(grid, "back"), (x,t) -> 0, D),
                Dirichlet(:v, getfaceset(grid, "front"), (x,t) -> 0, D),
            )
        end
    end

end # testset

@testset "Affine constraints with master dofs that are prescribed" begin
    grid = generate_grid(Quadrilateral, (2, 2))
    dh = DofHandler(grid); push!(dh, :u, 1); close!(dh)

    #  8───7───9
    #  │   │   │
    #  4───3───6
    #  │   │   │
    #  1───2───5

    ke = [ 1.0   -0.25  -0.5   -0.25
          -0.25   1.0   -0.25  -0.5
          -0.5   -0.25   1.0   -0.25
          -0.25  -0.5   -0.25   1.0 ]
    fe = rand(4)

@testset "affine constraints before/after Dirichlet" begin
    # Construct two ConstraintHandler's which should result in the same end result.

    ## Ordering of constraints for first ConstraintHandler:
    ##  1. DBC left: u1 = u4 = u8 = 0
    ##  2. DBC right: u5 = u6 = u9 = 1
    ##  3. Periodic bottom/top: u1 = u8, u2 = u7, u5 = u9
    ## meaning that u1 = 0 and u5 = 1 are overwritten by 3 and we end up with
    ##  u1 = u8 = 0
    ##  u2 = u7
    ##  u4 = 0
    ##  u5 = u9 = 1
    ##  u6 = 1
    ##  u8 = 0
    ##  u9 = 1
    ## where the inhomogeneity of u1 and u5 have to be resolved at runtime.
    ch1 = ConstraintHandler(dh)
    add!(ch1, Dirichlet(:u, getfaceset(grid, "left"), (x, t) -> 0))
    add!(ch1, Dirichlet(:u, getfaceset(grid, "right"), (x, t) -> 1))
    add!(ch1, PeriodicDirichlet(:u, collect_periodic_faces(grid, "bottom", "top")))
    close!(ch1)
    update!(ch1, 0)

    ## Ordering of constraints for second ConstraintHandler:
    ##  1. Periodic bottom/top: u1 = u8, u2 = u7, u5 = u9
    ##  2. DBC left: u1 = u4 = u8 = 0
    ##  3. DBC right: u5 = u6 = u9 = 1
    ## meaning that u1 = u8 and u5 = u9 are overwritten by 2 and 3 and we end up with
    ##  u1 = 0
    ##  u2 = u7
    ##  u4 = 0
    ##  u5 = 1
    ##  u6 = 1
    ##  u8 = 0
    ##  u9 = 1
    ch2 = ConstraintHandler(dh)
    add!(ch2, PeriodicDirichlet(:u, collect_periodic_faces(grid, "bottom", "top")))
    add!(ch2, Dirichlet(:u, getfaceset(grid, "left"), (x, t) -> 0))
    add!(ch2, Dirichlet(:u, getfaceset(grid, "right"), (x, t) -> 1))
    close!(ch2)
    update!(ch2, 0)

    K1 = create_sparsity_pattern(dh, ch1)
    f1 = zeros(ndofs(dh))
    a1 = start_assemble(K1, f1)
    K2 = create_sparsity_pattern(dh, ch2)
    f2 = zeros(ndofs(dh))
    a2 = start_assemble(K2, f2)

    for cell in CellIterator(dh)
        assemble!(a1, celldofs(cell), ke, fe)
        assemble!(a2, celldofs(cell), ke, fe)
    end

    # Equivalent assembly
    @test K1 == K2
    @test f1 == f2

    # Equivalence after apply!
    apply!(K1, f1, ch1)
    apply!(K2, f2, ch2)
    @test K1 == K2
    @test f1 == f2
    @test apply!(K1 \ f1, ch1) ≈ apply!(K2 \ f2, ch2)
end # subtestset

@testset "time dependence" begin
    ## Pure Dirichlet
    ch1 = ConstraintHandler(dh)
    add!(ch1, Dirichlet(:u, getfaceset(grid, "top"), (x, t) -> 3.0t + 2.0))
    add!(ch1, Dirichlet(:u, getfaceset(grid, "bottom"), (x, t) -> 1.5t + 1.0))
    add!(ch1, Dirichlet(:u, getfaceset(grid, "left"), (x, t) -> 1.0t))
    add!(ch1, Dirichlet(:u, getfaceset(grid, "right"), (x, t) -> 2.0t))
    close!(ch1)
    ## Dirichlet with corresponding AffineConstraint on dof 2 and 7
    ch2 = ConstraintHandler(dh)
    add!(ch2, AffineConstraint(7, [8 => 1.0, 9 => 1.0], 2.0))
    add!(ch2, AffineConstraint(2, [1 => 0.5, 5 => 0.5], 1.0))
    add!(ch2, Dirichlet(:u, getfaceset(grid, "left"), (x, t) -> 1.0t))
    add!(ch2, Dirichlet(:u, getfaceset(grid, "right"), (x, t) -> 2.0t))
    close!(ch2)

    K1 = create_sparsity_pattern(dh, ch1)
    f1 = zeros(ndofs(dh))
    K2 = create_sparsity_pattern(dh, ch2)
    f2 = zeros(ndofs(dh))

    for t in (1.0, 2.0)
        update!(ch1, t)
        update!(ch2, t)
        a1 = start_assemble(K1, f1)
        a2 = start_assemble(K2, f2)
        for cell in CellIterator(dh)
            assemble!(a1, celldofs(cell), ke, fe)
            assemble!(a2, celldofs(cell), ke, fe)
        end
        @test K1 == K2
        @test f1 == f2
        apply!(K1, f1, ch1)
        apply!(K2, f2, ch2)
        @test K1 == K2
        @test f1 == f2
        @test K1 \ f1 ≈ K2 \ f2
    end

end # subtestset


@testset "error paths" begin
    ch = ConstraintHandler(dh)
    add!(ch, AffineConstraint(1, [2 => 1.0], 0.0))
    add!(ch, AffineConstraint(2, [3 => 1.0], 0.0))
    @test_throws ErrorException("nested affine constraints currently not supported") close!(ch)
end # subtestset

end # testset

@testset "local application of bc" begin
    grid = generate_grid(Quadrilateral, (5,5))
    dh = DofHandler(grid)
    push!(dh, :u, 1)
    close!(dh)
    # Dirichlet BC
    ch_dbc = ConstraintHandler(dh)
    add!(ch_dbc, Dirichlet(:u, getfaceset(grid, "left"), (x, t) -> 0))
    add!(ch_dbc, Dirichlet(:u, getfaceset(grid, "right"), (x, t) -> 1))
    close!(ch_dbc)
    update!(ch_dbc, 0)
    # Dirichlet BC as affine constraints
    ch_ac = ConstraintHandler(dh)
    for (dof, value) in zip(ch_dbc.prescribed_dofs, ch_dbc.inhomogeneities)
        add!(ch_ac, AffineConstraint(dof, Pair{Int,Float64}[], value))
    end
    close!(ch_ac)
    update!(ch_ac, 0)
    # Periodic constraints (non-local couplings) #
    ch_p = ConstraintHandler(dh)
    # TODO: Order matters, but probably shouldn't, see Ferrite-FEM/Ferrite.jl#530
    face_map = collect_periodic_faces(grid, getfaceset(grid, "bottom"), getfaceset(grid, "top"))
    add!(ch_p, PeriodicDirichlet(:u, face_map))
    add!(ch_p, Dirichlet(:u, getfaceset(grid, "left"), (x, t) -> 0))
    add!(ch_p, Dirichlet(:u, getfaceset(grid, "right"), (x, t) -> 1))
    close!(ch_p)
    update!(ch_p, 0)

    function element!(ke, fe, cv, k, b)
        fill!(ke, 0)
        fill!(fe, 0)
        for qp in 1:getnquadpoints(cv)
            dΩ = getdetJdV(cv, qp)
            for i in 1:getnbasefunctions(cv)
                ϕi = shape_value(cv, qp, i)
                fe[i] += ( ϕi * b ) * dΩ
                for j in 1:getnbasefunctions(cv)
                    ∇ϕi = shape_gradient(cv, qp, i)
                    ∇ϕj = shape_gradient(cv, qp, j)
                    ke[i,j] += ( ∇ϕi ⋅ (k * ∇ϕj) ) * dΩ
                end
            end
        end
    end

    for azero in (nothing, false, true)

        S = create_sparsity_pattern(dh)
        f = zeros(ndofs(dh))

        K_dbc_standard = copy(S)
        f_dbc_standard = copy(f)
        assembler_dbc_standard = start_assemble(K_dbc_standard, f_dbc_standard)
        K_dbc_ch = copy(S)
        f_dbc_ch = copy(f)
        assembler_dbc_ch = start_assemble(K_dbc_ch, f_dbc_ch)
        K_dbc_local = copy(S)
        f_dbc_local = copy(f)
        assembler_dbc_local = start_assemble(K_dbc_local, f_dbc_local)

        S = create_sparsity_pattern(dh, ch_ac)

        K_ac_standard = copy(S)
        f_ac_standard = copy(f)
        assembler_ac_standard = start_assemble(K_ac_standard, f_ac_standard)
        K_ac_ch = copy(S)
        f_ac_ch = copy(f)
        assembler_ac_ch = start_assemble(K_ac_ch, f_ac_ch)
        K_ac_local = copy(S)
        f_ac_local = copy(f)
        assembler_ac_local = start_assemble(K_ac_local, f_ac_local)

        S = create_sparsity_pattern(dh, ch_p)

        K_p_standard = copy(S)
        f_p_standard = copy(f)
        assembler_p_standard = start_assemble(K_p_standard, f_p_standard)
        K_p_ch = copy(S)
        f_p_ch = copy(f)
        assembler_p_ch = start_assemble(K_p_ch, f_p_ch)
        # K_p_local = copy(S)
        # f_p_local = copy(f)
        # assembler_p_local = start_assemble(K_p_local, f_p_local)

        ke = zeros(ndofs_per_cell(dh), ndofs_per_cell(dh))
        fe = zeros(ndofs_per_cell(dh))
        cv = CellScalarValues(QuadratureRule{2,RefCube}(2), Lagrange{2,RefCube,1}())

        for cell in CellIterator(dh)
            reinit!(cv, cell)
            global_dofs = celldofs(cell)
            element!(ke, fe, cv, cellid(cell), 1/cellid(cell))
            # Standard application
            assemble!(assembler_dbc_standard, global_dofs, ke, fe)
            assemble!(assembler_ac_standard, global_dofs, ke, fe)
            assemble!(assembler_p_standard, global_dofs, ke, fe)
            # Assemble with ConstraintHandler
            let apply_f! = azero === nothing ? apply_assemble! :
                (args...) -> apply_assemble!(args...; apply_zero=azero)
                let ke = copy(ke), fe = copy(fe)
                    apply_f!(assembler_dbc_ch, ch_dbc, global_dofs, ke, fe)
                end
                let ke = copy(ke), fe = copy(fe)
                    apply_f!(assembler_ac_ch, ch_ac, global_dofs, ke, fe)
                end
                let ke = copy(ke), fe = copy(fe)
                    apply_f!(assembler_p_ch, ch_p, global_dofs, ke, fe)
                end
            end
            # Assemble after apply_local!
            let apply_f! = azero === nothing ? apply_local! :
                (args...) -> apply_local!(args...; apply_zero=azero)
                let ke = copy(ke), fe = copy(fe)
                    apply_f!(ke, fe, global_dofs, ch_dbc)
                    assemble!(assembler_dbc_local, global_dofs, ke, fe)
                end
                let ke = copy(ke), fe = copy(fe)
                    apply_f!(ke, fe, global_dofs, ch_ac)
                    assemble!(assembler_ac_local, global_dofs, ke, fe)
                end
                let ke = copy(ke), fe = copy(fe)
                    if cellid(cell) in first.(getfaceset(grid, "bottom"))
                        # Throws for all cells on the image boundary
                        @test_throws ErrorException apply_f!(ke, fe, global_dofs, ch_p)
                    else
                        # Should not throw otherwise
                        apply_f!(ke, fe, global_dofs, ch_p)
                    end
                    # assemble!(assembler_p_local, ch_p, global_dofs, ke, fe)
                end
            end
        end

        # apply! for the standard ones
        let apply_f! = azero === true ? apply_zero! : apply!
            apply_f!(K_dbc_standard, f_dbc_standard, ch_dbc)
            apply_f!(K_ac_standard, f_ac_standard, ch_ac)
            apply_f!(K_p_standard, f_p_standard, ch_p)
        end

        fdofs = free_dofs(ch_dbc)
        fdofs′ = free_dofs(ch_ac)
        @test fdofs == fdofs′

        # Everything should be identical now for free entries
        @test K_dbc_standard[fdofs, fdofs] ≈ K_dbc_ch[fdofs, fdofs] ≈ K_dbc_local[fdofs, fdofs] ≈
              K_ac_standard[fdofs, fdofs] ≈ K_ac_ch[fdofs, fdofs] ≈ K_ac_local[fdofs, fdofs]
        @test f_dbc_standard[fdofs] ≈ f_dbc_ch[fdofs] ≈ f_dbc_local[fdofs] ≈
              f_ac_standard[fdofs] ≈ f_ac_ch[fdofs] ≈ f_ac_local[fdofs]
        fdofs_p = free_dofs(ch_p)
        @test K_p_standard[fdofs_p, fdofs_p] ≈ K_p_ch[fdofs_p, fdofs_p]
        @test f_p_standard[fdofs_p] ≈ f_p_ch[fdofs_p]

        # For prescribed dofs the matrices will be diagonal, but different
        pdofs = ch_dbc.prescribed_dofs
        @test isdiag(K_dbc_standard[pdofs, pdofs])
        @test isdiag(K_dbc_ch[pdofs, pdofs])
        @test isdiag(K_dbc_local[pdofs, pdofs])
        @test isdiag(K_ac_standard[pdofs, pdofs])
        @test isdiag(K_ac_ch[pdofs, pdofs])
        @test isdiag(K_ac_local[pdofs, pdofs])
        pdofs_p = ch_p.prescribed_dofs
        @test isdiag(K_p_standard[pdofs_p, pdofs_p])
        @test isdiag(K_p_ch[pdofs_p, pdofs_p])

        # No coupling between constraints (no post-solve apply! needed) so this should
        # compute the inhomogeneities correctly
        scale = azero === true ? 0 : 1
        @test f_dbc_standard[pdofs] ./ diag(K_dbc_standard[pdofs, pdofs]) ≈ scale * ch_dbc.inhomogeneities
        @test f_dbc_ch[pdofs] ./ diag(K_dbc_ch[pdofs, pdofs]) ≈ scale * ch_dbc.inhomogeneities
        @test f_dbc_local[pdofs] ./ diag(K_dbc_local[pdofs, pdofs]) ≈ scale * ch_dbc.inhomogeneities
        @test f_ac_standard[pdofs] ./ diag(K_ac_standard[pdofs, pdofs]) ≈ scale * ch_ac.inhomogeneities
        @test f_ac_ch[pdofs] ./ diag(K_ac_ch[pdofs, pdofs]) ≈ scale * ch_ac.inhomogeneities
        @test f_ac_local[pdofs] ./ diag(K_ac_local[pdofs, pdofs]) ≈ scale * ch_ac.inhomogeneities

        # Test the solutions
        u_dbc = K_dbc_standard \ f_dbc_standard
        u_p = (azero === true ? apply_zero! : apply!)(K_p_standard \ f_p_standard, ch_p)
        if azero === true
            @test norm(u_dbc) ≈ 0.06401424182205259
            @test norm(u_p) ≈ 0.02672553850330505
        else
            @test norm(u_dbc) ≈ 3.8249286998373586
            @test norm(u_p) ≈ 3.7828270430540893
        end
        # vtk_grid("local_application_azero_$(azero)", grid) do vtk
        #     vtk_point_data(vtk, dh, u_dbc, "_dbc")
        #     vtk_point_data(vtk, dh, u_p, "_p")
        # end
        @test K_dbc_standard \ f_dbc_standard ≈ K_dbc_ch \ f_dbc_ch ≈ K_dbc_local \ f_dbc_local ≈
              K_ac_standard \ f_ac_standard ≈ K_ac_ch \ f_ac_ch ≈ K_ac_local \ f_ac_local
        let apply_f! = azero === true ? apply_zero! : apply!
            @test apply_f!(K_p_standard \ f_p_standard, ch_p) ≈ apply_f!(K_p_ch \ f_p_ch, ch_p)
        end
    end
end # testset

@testset "Sparsity pattern without constrained dofs" begin
    grid = generate_grid(Triangle, (5, 5))
    dh = DofHandler(grid)
    push!(dh, :u, 1)
    close!(dh)
    ch = ConstraintHandler(dh)
    add!(ch, Dirichlet(:u, getfaceset(grid, "left"), (x, t) -> 0))
    close!(ch)
    Kfull = create_sparsity_pattern(dh, ch)
    K = create_sparsity_pattern(dh, ch; keep_constrained=false)
    # Pattern tests
    nonzero_edges = Set(
        (i, j) for d in 1:getncells(grid)
               for (i, j) in Iterators.product(celldofs(dh, d), celldofs(dh, d))
    )
    zero_edges = setdiff(Set(Iterators.product(1:ndofs(dh), 1:ndofs(dh))), nonzero_edges)
    for (i, j) in zero_edges
        @test Base.isstored(K, i, j) == Base.isstored(Kfull, i, j) == false
    end
    for (i, j) in nonzero_edges
        if i != j && (haskey(ch.dofmapping, i) || haskey(ch.dofmapping, j))
            @test !Base.isstored(K, i, j)
        else
            @test Base.isstored(K, i, j)
        end
        @test Base.isstored(Kfull, i, j)
    end
    # Assembly
    afull = start_assemble(Kfull)
    a = start_assemble(K)
    Kc = copy(K)
    ac = start_assemble(Kc, zeros(ndofs(dh)))
    for cell in CellIterator(dh)
        ke = rand(ndofs_per_cell(dh), ndofs_per_cell(dh))
        assemble!(afull, celldofs(cell), ke)
        kec = copy(ke)
        apply_local!(kec, rand(ndofs_per_cell(dh)), celldofs(cell), ch)
        assemble!(a, celldofs(cell), kec)
        apply_assemble!(ac, ch, celldofs(cell), ke, rand(ndofs_per_cell(dh)))
    end
    apply!(Kfull, ch)
    fd = free_dofs(ch)
    pd = ch.prescribed_dofs
    @test K ≈ Kc
    @test K[fd, fd] ≈ Kc[fd, fd] ≈ Kfull[fd, fd]
    for d in ch.prescribed_dofs
        K[d, d] = Kc[d, d] = Kfull[d, d] = 1
    end
    @test K ≈ Kc ≈ Kfull
end # testset
