

function test_projection(order)
    if order == 1
        grid = Ferrite.generate_grid(Quadrilateral, (1, 1), Vec((0.,0.)), Vec((1.,1.)))
    elseif order == 2
        # grid = Ferrite.generate_grid(QuadraticQuadrilateral, (1, 1), Vec((0.,0.)), Vec((1.,1.)))
        grid = Ferrite.generate_grid(Quadrilateral, (1, 1), Vec((0.,0.)), Vec((1.,1.)))
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

    # Now recover the nodal values using a L2 projection. Since f is quadratic and the interpolation as well, we should recover the exact nodal values
    projector = L2Projector(cv, ip, grid)

    point_vars = Ferrite.project(qp_values, projector)

    ae = compute_vertex_values(grid, f)
    # The projection gives the values in node order -> reorder ae
    # @test point_vars[1:4] ≈ [ae[1], ae[2], ae[4], ae[3]]
    # return point_vars[1:4], [ae[1], ae[2], ae[4], ae[3]]
    return point_vars[1:4], [ae[1], ae[2], ae[3], ae[4]]
end

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
    projector = L2Projector(cv, ip, mesh, 1:1)
    point_vars = Ferrite.project(qp_values, projector)

    # In the nodes of the 1st cell we should recover the field
    for node in mesh.cells[1].nodes
        @test ae[node] ≈ point_vars[node]
    end

    # in all other nodes we should have NaNs
    for node in setdiff(1:getnnodes(mesh), mesh.cells[1].nodes)
        for d1 = 1:dim, d2 = 1:dim
             @test isnan(point_vars[node][d1, d2])
         end
    end
end

function test_projection_newprojector(order)
    if order == 1
        grid = Ferrite.generate_grid(Quadrilateral, (1, 1), Vec((0.,0.)), Vec((1.,1.)))
    elseif order == 2
        # grid = Ferrite.generate_grid(QuadraticQuadrilateral, (1, 1), Vec((0.,0.)), Vec((1.,1.)))
        grid = Ferrite.generate_grid(Quadrilateral, (1, 1), Vec((0.,0.)), Vec((1.,1.)))
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

    # Now recover the nodal values using a L2 projection. Since f is quadratic and the interpolation as well, we should recover the exact nodal values
    projector = L2Projector(qr, ip, grid)

    point_vars = Ferrite.project(qp_values, projector)

    ae = compute_vertex_values(grid, f)
    # The projection gives the values in node order -> reorder ae
    # @test point_vars[1:4] ≈ [ae[1], ae[2], ae[4], ae[3]]
    # return point_vars[1:4], [ae[1], ae[2], ae[4], ae[3]]
    return point_vars[1:4], [ae[1], ae[2], ae[3], ae[4]]
end

@testset "Test L2-Projection" begin
    # Tests a L2-projection of integration point values (to nodal values), determined from the function y = 1 + x[1]^2 + (2x[2])^2

    # A linear approximation can not recover a quadratic solution, so projected values will be different from the analytical ones
    projected_vars, analytical_vars = test_projection(1)
    @test projected_vars ≉  analytical_vars

    # For a quadratic approximation the analytical solution is recovered
    projected_vars, analytical_vars = test_projection(2)
    @test projected_vars ≈ analytical_vars

    # Test a mixed grid, where only a subset of the cells contains a field
    test_projection_mixedgrid()

end
