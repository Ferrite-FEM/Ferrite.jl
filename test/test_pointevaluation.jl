function scalar_field()
    # isoparametric approximation
    mesh = generate_grid(QuadraticQuadrilateral, (20, 20))
    f(x) = x[1]^2
    nodal_vals = [f(p.x) for p in mesh.nodes]

    ip_f = Lagrange{2,RefCube,2}() # function interpolation
    ip_g = Lagrange{2,RefCube,2}() # geometry interpolation

    # compute values in quadrature points
    qr = QuadratureRule{2, RefCube}(3) # exactly approximate quadratic field
    cv = CellScalarValues(qr, ip_f, ip_g)
    qp_vals = [Vector{Float64}(undef, getnquadpoints(cv)) for i=1:getncells(mesh)]
    for cellid in eachindex(mesh.cells)
        xe = getcoordinates(mesh, cellid)
        reinit!(cv, xe)
        for qp in 1:getnquadpoints(cv)
            qp_vals[cellid][qp] = f(spatial_coordinate(cv, qp, xe))
        end
    end

    # do a L2Projection for getting values in dofs
    projector = L2Projector(ip_f, mesh)
    dof_vals = project(projector, qp_vals, qr; project_to_nodes=false)

    # points where we want to retrieve field values
    points = [Vec((x, 0.52)) for x in range(0.0; stop=1.0, length=100)]

    # set up PointEvalHandler and retrieve values
    ph = PointEvalHandler(projector.dh, points)
    vals = get_point_values(ph, dof_vals, projector)
    @test f.(points) ≈ vals

    # alternatively retrieve vals from nodal values
    vals = get_point_values(ph, nodal_vals)
    @test f.(points) ≈ vals
end

function vector_field()
    ## vector field
    # isoparametric approximation
    mesh = generate_grid(QuadraticQuadrilateral, (20, 20))
    f(x) = Vec((x[1]^2, x[1]))
    nodal_vals = [f(p.x) for p in mesh.nodes]

    ip_f = Lagrange{2,RefCube,2}() # function interpolation
    ip_g = Lagrange{2,RefCube,2}() # geometry interpolation

    # compute values in quadrature points
    qr = QuadratureRule{2, RefCube}(3) # exactly approximate quadratic field
    cv = CellScalarValues(qr, ip_f, ip_g)
    qp_vals = [Vector{Vec{2,Float64}}(undef, getnquadpoints(cv)) for i=1:getncells(mesh)]
    for cellid in eachindex(mesh.cells)
        xe = getcoordinates(mesh, cellid)
        reinit!(cv, xe)
        for qp in 1:getnquadpoints(cv)
            qp_vals[cellid][qp] = f(spatial_coordinate(cv, qp, xe))
        end
    end

    # do a L2Projection for getting values in dofs
    projector = L2Projector(ip_f, mesh)
    dof_vals = project(projector, qp_vals, qr; project_to_nodes=false)

    # points where we want to retrieve field values
    points = [Vec((x, 0.52)) for x in range(0.0; stop=1.0, length=100)]

    # set up PointEvalHandler and retrieve values
    ph = PointEvalHandler(projector.dh, points)
    vals = get_point_values(ph, dof_vals, projector)
    @test f.(points) ≈ vals

    # alternatively retrieve vals from nodal values
    vals = get_point_values(ph, nodal_vals)
    @test f.(points) ≈ vals
end

function superparametric()
    # superparametric approximation
    mesh = generate_grid(Quadrilateral, (20, 20))
    f(x) = x*x[1]
    ip_f = Lagrange{2,RefCube,2}() # function interpolation
    ip_g = Lagrange{2,RefCube,1}() # geometry interpolation

    # compute values in quadrature points
    qr = QuadratureRule{2, RefCube}(3) # exactly approximate quadratic field
    cv = CellScalarValues(qr, ip_f, ip_g)
    qp_vals = [Vector{Vec{2,Float64}}(undef, getnquadpoints(cv)) for i=1:getncells(mesh)]
    for cellid in eachindex(mesh.cells)
        xe = getcoordinates(mesh, cellid)
        reinit!(cv, xe)
        for qp in 1:getnquadpoints(cv)
            qp_vals[cellid][qp] = f(spatial_coordinate(cv, qp, xe))
        end
    end

    # do a L2Projection for getting values in dofs
    projector = L2Projector(ip_f, mesh)
    dof_vals = project(projector, qp_vals, qr; project_to_nodes=false)

    # points where we want to retrieve field values
    points = [Vec((x, 0.52)) for x in range(0.0; stop=1.0, length=100)]

    # set up PointEvalHandler and retrieve values
    ph = PointEvalHandler(projector.dh, points)
    vals = get_point_values(ph, dof_vals, projector)

    # can recover a quadratic field by a quadratic approximation
    @test f.(points) ≈ vals
end

function dofhandler()
    mesh = generate_grid(Quadrilateral, (2,2))
    dof_vals = [1., 2., 5., 4., 3., 6., 8., 7., 9.]
    points = [node.x for node in mesh.nodes] # same as nodes

    dh = DofHandler(mesh)
    push!(dh, :s, 1) # a scalar field
    close!(dh)

    ph = PointEvalHandler(dh, points)
    vals = get_point_values(ph, dof_vals, :s)
    @test vals ≈ 1.0:9.0

    vals = get_point_values(ph, collect(1.0:9.0))
    @test vals ≈ 1.0:9.0
end

function mixed_grid()
    ## Mixed grid where not all cells have the same fields 

    # 5_______6
    # |\      | 
    # |   \   |
    # 3______\4
    # |       |
    # |       |
    # 1_______2 

    nodes = [Node((0.0, 0.0)),
            Node((1.0, 0.0)),
            Node((0.0, 1.0)),
            Node((1.0, 1.0)),
            Node((0.0, 2.0)),
            Node((1.0, 2.0))]

    cells = Ferrite.AbstractCell[Quadrilateral((1,2,4,3)),
            Triangle((3,4,6)),
            Triangle((3,6,5))]

    mesh = Grid(cells, nodes)
    addcellset!(mesh, "quads", Set{Int}((1,)))
    addcellset!(mesh, "tris", Set{Int}((2, 3)))

    ip_quad = Lagrange{2,RefCube,1}()
    ip_tri = Lagrange{2,RefTetrahedron,1}()

    f(x) = x[1]

    # compute values in quadrature points for quad
    qr = QuadratureRule{2, RefCube}(2)
    cv = CellScalarValues(qr, ip_quad)
    qp_vals_quads = [Vector{Float64}(undef, getnquadpoints(cv)) for cell in getcellset(mesh, "quads")]
    for (local_cellid, global_cellid) in enumerate(getcellset(mesh, "quads"))
        xe = getcoordinates(mesh, global_cellid)
        reinit!(cv, xe)
        for qp in 1:getnquadpoints(cv)
            qp_vals_quads[local_cellid][qp] = f(spatial_coordinate(cv, qp, xe))
        end
    end

    # construct projector 
    projector = L2Projector(ip_quad, mesh; set=getcellset(mesh, "quads"))

    points = [Vec((x, 2x)) for x in range(0.0; stop=1.0, length=10)]

    # first alternative: L2Projection to dofs
    dof_vals = project(projector, qp_vals_quads, qr; project_to_nodes = false)
    ph = PointEvalHandler(projector.dh, points)
    vals = get_point_values(ph, dof_vals, projector)
    @test vals[1:5] ≈ f.(points[1:5])
    @test all(isnan.(vals[6:end]))
    # second alternative: L2Projection to nodes
    nodal_vals = project(projector, qp_vals_quads, qr; project_to_nodes = true)
    vals = get_point_values(ph, nodal_vals)
    @test vals[1:5] ≈ f.(points[1:5])
    @test all(isnan.(vals[6:end]))


    # second alternative: assume a vector field :v
    dh = MixedDofHandler(mesh)
    field = Field(:v, ip_quad, 2)
    fh_quad = FieldHandler([field], getcellset(mesh, "quads"))
    push!(dh, fh_quad)
    field = Field(:v, ip_tri, 2) 
    fh_tri = FieldHandler([field], getcellset(mesh, "tris"))
    push!(dh, fh_tri)
    close!(dh)

    dof_vals = [1., 1., 2., 2., 4., 4., 3., 3., 6., 6., 5., 5.]
    points = [node.x for node in mesh.nodes]
    ph = PointEvalHandler(dh, points)
    vals = get_point_values(ph, dof_vals, :v)
    @test vals == [Vec((i, i)) for i=1.0:6.0]
end

@testset "PointEvalHandler" begin
    scalar_field()
    vector_field()
    dofhandler()
    superparametric()
    mixed_grid()
end