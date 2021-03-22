
# Tests a L2-projection of integration point values (to nodal values),
# determined from the function y = 1 + x[1]^2 + (2x[2])^2
function test_projection(order)
    if order == 1
        grid = generate_grid(Quadrilateral, (1, 1), Vec((0.,0.)), Vec((1.,1.)))
    elseif order == 2
        # grid = generate_grid(QuadraticQuadrilateral, (1, 1), Vec((0.,0.)), Vec((1.,1.)))
        grid = generate_grid(Quadrilateral, (1, 1), Vec((0.,0.)), Vec((1.,1.)))
    end

    dim = 2
    ip = Lagrange{dim, RefCube, order}()
    ip_geom = Lagrange{dim, RefCube, 1}()
    qr = QuadratureRule{dim, RefCube}(order+1)
    cv = CellScalarValues(qr, ip, ip_geom)

    # Create node values for the cell
    f(x) = Tensor{1,1,Float64}((1 + x[1]^2 + (2x[2])^2, ))
    xe = getcoordinates(grid, 1)
    # analytical values
    qp_values = [[f(spatial_coordinate(cv, qp, xe)) for qp in 1:getnquadpoints(cv)]]

    # Now recover the nodal values using a L2 projection.
    proj = L2Projector(ip, grid; geom_ip=ip_geom)
    point_vars = project(proj, qp_values, qr)
    ## Old API with fe values as first arg
    proj = @test_deprecated L2Projector(cv, ip, grid)
    point_vars_2 = @test_deprecated project(qp_values, proj)
    ## Old API with qr as first arg
    proj = @test_deprecated L2Projector(qr, ip, grid)
    point_vars_3 = @test_deprecated project(qp_values, proj)

    @test point_vars ≈ point_vars_2 ≈ point_vars_3

    if order == 1
        # A linear approximation can not recover a quadratic solution,
        # so projected values will be different from the analytical ones
        ae = [Vec{1}((0.1666666666666664,)), Vec{1}((1.166666666666667,)),
              Vec{1}((4.166666666666666,)),  Vec{1}((5.166666666666667,))]
        @test point_vars[1:4] ≈ ae
    elseif order == 2
        # For a quadratic approximation the analytical solution is recovered
        ae = compute_vertex_values(grid, f)
        @test point_vars[1:4] ≈ ae
    end
end

# Test a mixed grid, where only a subset of the cells contains a field
function test_projection_mixedgrid()
    # generate a mesh with 1 quadrilateral and 2 triangular elements
    dim = 2
    nodes = Node{dim, Float64}[]
    push!(nodes, Node((0.0, 0.0)))
    push!(nodes, Node((1.0, 0.0)))
    push!(nodes, Node((2.0, 0.0)))
    push!(nodes, Node((0.0, 1.0)))
    push!(nodes, Node((1.0, 1.0)))
    push!(nodes, Node((2.0, 1.0)))

    cells = Ferrite.AbstractCell[]
    push!(cells, Quadrilateral((1,2,5,4)))
    push!(cells, Triangle((2,3,6)))
    push!(cells, Triangle((2,6,5)))

    mesh = Grid(cells, nodes)

    order = 2
    ip = Lagrange{dim, RefCube, order}()
    ip_geom = Lagrange{dim, RefCube, 1}()
    qr = QuadratureRule{dim, RefCube}(order+1)
    cv = CellScalarValues(qr, ip, ip_geom)

    # Create node values for the 1st cell
    # use a SymmetricTensor here for testing the symmetric version of project
    f(x) = SymmetricTensor{2,2,Float64}((1 + x[1]^2, 2x[2]^2, x[1]*x[2]))
    xe = getcoordinates(mesh, 1)
    ae = compute_vertex_values(mesh, f)
    # analytical values
    qp_values = [[f(spatial_coordinate(cv, qp, xe)) for qp in 1:getnquadpoints(cv)]]

    # Now recover the nodal values using a L2 projection.
    # Assume f would only exist on the first cell, we project it to the nodes of the
    # 1st cell while ignoring the rest of the domain. NaNs should be stored in all
    # nodes that do not belong to the 1st cell
    proj = L2Projector(ip, mesh; geom_ip=ip_geom, set=1:1)
    point_vars = project(proj, qp_values, qr)
    ## Old API with fe values as first arg
    proj = @test_deprecated L2Projector(cv, ip, mesh, 1:1)
    point_vars_2 = @test_deprecated project(qp_values, proj)
    ## Old API with qr as first arg
    proj = @test_deprecated L2Projector(qr, ip, mesh, 1:1)
    point_vars_3 = @test_deprecated project(qp_values, proj)

    # In the nodes of the 1st cell we should recover the field
    for node in mesh.cells[1].nodes
        @test ae[node] ≈ point_vars[node] ≈ point_vars_2[node] ≈ point_vars_3[node]
    end

    # in all other nodes we should have NaNs
    for node in setdiff(1:getnnodes(mesh), mesh.cells[1].nodes)
        for d1 = 1:dim, d2 = 1:dim
             @test isnan(point_vars[node][d1, d2])
             @test isnan(point_vars_2[node][d1, d2])
             @test isnan(point_vars_3[node][d1, d2])
         end
    end
end

@testset "Test L2-Projection" begin
    test_projection(1)
    test_projection(2)
    test_projection_mixedgrid()
end
