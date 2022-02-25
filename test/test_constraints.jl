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

@testset "edge bc mixed grid" begin
    # Test mesh
    # 6---7---8---9---10    
    # |   |2 /|   |   |     :u 1:5
    # | 1 | / | 4 | 5 |     :v 1:2
    # |   |/ 3|   |   |     
    # 1---2---3---4---5
    #
    # |---A---|---B---|
    #     |---C---|
    #
    # Dofs per node (note, depends on push! order below) (:u, [:v])
    # node nr     1        2      3      4      5      6      7      8       9     10
    nodedofs = [(1, 5), (2, 6), (11,), (12,), (14,), (4,8), (3,7), (9,10), (13,), (15,)]
    
    # Create Grid based on above drawing
    nodes = [Node{2,Float64}(Vec{2,Float64}((i, j))) for j in 0:1 for i in 0:4]
    quadcells = [Quadrilateral((i, i+1, i+6, i+5)) for i in [1, 3, 4]]
    tricells = [Triangle((2,8,7)), Triangle((2,3,8))]
    cells = [quadcells[1], tricells..., quadcells[2:end]...]
    cellsets = Dict("onlyuQ" => Set(4:5), "onlyuT" => Set(3:3),
                    "uandvQ" => Set(1:1), "uandvT" => Set(2:2))
    facesets = Dict("A" => Set((FaceIndex(1,1), FaceIndex(3,1))),
                    "B" => Set((FaceIndex(4,1), FaceIndex(5,1))),
                    "C" => Set((FaceIndex(3,1), FaceIndex(4,1))))
    grid = Grid(cells, nodes, cellsets=cellsets, facesets=facesets)

    # Create MixedDofHandler based on grid
    dim = Ferrite.getdim(grid)  # 2
    ip_quad = Lagrange{dim,RefCube,1}()
    ip_tria = Lagrange{dim,RefTetrahedron,1}()
    dh = MixedDofHandler(grid)
    field_uT = Field(:u, ip_tria, 1)
    field_uQ = Field(:u, ip_quad, 1)
    field_vT = Field(:v, ip_tria, 1)
    field_vQ = Field(:v, ip_quad, 1)

    # Order important for test to ensure consistent dof ordering
    push!(dh, FieldHandler([field_uQ, field_vQ], getcellset(grid, "uandvQ")))
    push!(dh, FieldHandler([field_uT, field_vT], getcellset(grid, "uandvT")))
    push!(dh, FieldHandler([field_uT], getcellset(grid, "onlyuT")))
    push!(dh, FieldHandler([field_uQ], getcellset(grid, "onlyuQ")))
    close!(dh)

    # Add constraints 
    ch = ConstraintHandler(dh)
    dA_u = Dirichlet(:u, getfaceset(grid, "A"), (x,t) -> 1.0)
    dA_v = Dirichlet(:v, getfaceset(grid, "A"), (x,t) -> 2.0)
    dB_u = Dirichlet(:u, getfaceset(grid, "B"), (x,t) -> 3.0)  # Note, overwrites dA_u on node 3 
    dB_v = Dirichlet(:v, getfaceset(grid, "B"), (x,t) -> 4.0)  # :v not on cells with "B"-faces
    dC_v = Dirichlet(:v, getfaceset(grid, "C"), (x,t) -> 5.0)  # :v not on cells with "C"-faces
    
    @test_logs min_level=Logging.Warn add!(ch, dA_u)    # No warning should be issued
    @test_logs min_level=Logging.Warn add!(ch, dA_v)    # No warning should be issued
    @test_logs min_level=Logging.Warn add!(ch, dB_u)    # No warning should be issued
    @test_logs (:warn,) add!(ch, dB_v)  # Warn about :v not in cells connected with dB_v's faceset
    @test_logs (:warn,) add!(ch, dC_v)  # Warn about :v not in cells connected with dC_v's faceset
    close!(ch)
    
    # The full bottom part of the mesh has been prescribed
    @test sort(ch.prescribed_dofs) == sort([nd[i] for nd in nodedofs[1:5] for i in 1:length(nd)])

    # Test that the correct dofs have been prescribed
    update!(ch, 0.0)
    #                 nodes       N1,  N2,  N1,  N2,  N3,  N4,  N5
    #                 field       :u,  :u,  :v,  :v,  :u,  :u,  :u
    #                   dof        1,   2,   5,   6,  11,  12,  14
    @test ch.inhomogeneities == [1.0, 1.0, 2.0, 2.0, 3.0, 3.0, 3.0]
    # Note that dB_u overwrite dA_u @ N3, hence the value 3.0 there
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

        # Solve by actually condensing the matrix
        ff  = C' * (f - K * g)
        KK = C' * K * C
        _aa = KK \ ff
        aa = C * _aa + g

        # Solving by modifying K inplace
        apply!(K, f, ch)
        a = K \ f
        apply!(a, ch)

        @test a ≈ aa
    end

end
