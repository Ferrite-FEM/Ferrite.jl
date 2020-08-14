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
    @test ch.values == [-1, -1, 1, -1, -1, 1, 0, 0, 0]

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
    @test ch.values == [1.0, 1.0, 1.0, 2.0, 2.0, 1.0, 1.0]
end
