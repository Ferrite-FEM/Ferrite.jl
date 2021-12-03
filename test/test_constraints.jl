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

@testset "linear constraints" begin

    grid = generate_grid(Line, (10,))
    dh = DofHandler(grid)
    push!(dh, :u, 1)
    close!(dh)

    test_lcs = [
        #Simple homogeneous constraint
        [LinearConstraint(4, [(7 => 1.0)], 0.0)],
        #Two dofs and inhomogeneity
        [LinearConstraint(2, [(5 => 1.0), (6 =>2.0)], 1.0)],
        #Two linear constraints
        [LinearConstraint(2, [9=>1.0], 0.0),
         LinearConstraint(3, [9=>1.0], 0.0)],
        #
        [LinearConstraint(2, [7=>3.0, 8=>1.0], -1.0),
         LinearConstraint(4, [9=>-1.0], 2.0)]
    ]

    for lcs in test_lcs
        ch = ConstraintHandler(dh)
        add!(ch, Dirichlet(:u, getfaceset(grid, "left"), (x,t)->0.0))
        for lc in lcs
            add!(ch, lc)
        end
        close!(ch)
        update!(ch, 0.0)
        C, g = Ferrite.create_constraint_matrix(ch)
        
        #Assemble
        K = create_sparsity_pattern(dh, ch)
        f = zeros(ndofs(dh)); f[end] = 1.0
        for cell in CellIterator(dh)
            K[celldofs(cell),celldofs(cell)] += 2.0*[1 -1; -1 1]
        end

        #Solve by actually condensing matrix
        ff  = C'*(f - K*g)
        KK = C'*K*C;
        _aa = KK\ff
        aa = C*_aa + g

        #Solving by modifying K inplace
        apply!(K, f, ch)
        a = K\f
        apply!(a,ch)

        @test all(a .≈ aa)
    end

end