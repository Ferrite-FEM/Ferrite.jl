

function test_projection(order)
    if order == 1
        grid = JuAFEM.generate_grid(Quadrilateral, (1, 1), Vec((0.,0.)), Vec((1.,1.)))
    elseif order == 2
        # grid = JuAFEM.generate_grid(QuadraticQuadrilateral, (1, 1), Vec((0.,0.)), Vec((1.,1.)))
        grid = JuAFEM.generate_grid(Quadrilateral, (1, 1), Vec((0.,0.)), Vec((1.,1.)))
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

    point_vars = JuAFEM.project(qp_values, projector)

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


end
