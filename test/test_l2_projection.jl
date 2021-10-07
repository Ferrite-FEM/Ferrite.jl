
# Tests a L2-projection of integration point values (to nodal values),
# determined from the function y = 1 + x[1]^2 + (2x[2])^2
function test_projection(order, refshape)
    element = refshape == RefCube ? Quadrilateral : Triangle
    if order == 1
        grid = generate_grid(element, (1, 1), Vec((0.,0.)), Vec((1.,1.)))
    elseif order == 2
        # grid = generate_grid(QuadraticQuadrilateral, (1, 1), Vec((0.,0.)), Vec((1.,1.)))
        grid = generate_grid(element, (1, 1), Vec((0.,0.)), Vec((1.,1.)))
    end

    dim = 2
    ip = Lagrange{dim, refshape, order}()
    ip_geom = Lagrange{dim, refshape, 1}()
    qr = Ferrite._mass_qr(ip)
    cv = CellScalarValues(qr, ip, ip_geom)

    # Create node values for the cell
    f(x) = 1 + x[1]^2 + (2x[2])^2
    # Nodal approximations for this simple grid when using linear interpolation
    f_approx(i) = refshape == RefCube ?
        [0.1666666666666664, 1.166666666666667, 4.166666666666666, 5.166666666666667][i] :
        [0.444444444444465, 1.0277777777778005, 4.027777777777753, 5.444444444444435][i]

    # analytical values
    function analytical(f)
        qp_values = []
        for cell in CellIterator(grid)
            reinit!(cv, cell)
            r = [f(spatial_coordinate(cv, qp, getcoordinates(cell))) for qp in 1:getnquadpoints(cv)]
            push!(qp_values, r)
        end
        return identity.(qp_values) # Tighten the type
    end

    qp_values = analytical(f)

    # Now recover the nodal values using a L2 projection.
    proj = L2Projector(ip, grid; geom_ip=ip_geom)
    point_vars = project(proj, qp_values, qr)
    ## Old API with fe values as first arg
    proj2 = @test_deprecated L2Projector(cv, ip, grid)
    point_vars_2 = @test_deprecated project(qp_values, proj2)
    ## Old API with qr as first arg
    proj3 = @test_deprecated L2Projector(qr, ip, grid)
    point_vars_3 = @test_deprecated project(qp_values, proj3)

    @test point_vars ≈ point_vars_2 ≈ point_vars_3

    if order == 1
        # A linear approximation can not recover a quadratic solution,
        # so projected values will be different from the analytical ones
        ae = [f_approx(i) for i in 1:4]
    elseif order == 2
        # For a quadratic approximation the analytical solution is recovered
        ae = compute_vertex_values(grid, f)
    end
    @test point_vars[1:4] ≈ ae

    # Vec
    f_vector(x) = Vec{1,Float64}((f(x),))
    qp_values = analytical(f_vector)
    point_vars = project(proj, qp_values, qr)
    if order == 1
        ae = [Vec{1,Float64}((f_approx(j),)) for j in 1:4]
    elseif order == 2
        ae = compute_vertex_values(grid, f_vector)
    end
    @test point_vars[1:4] ≈ ae

    # Tensor
    f_tensor(x) = Tensor{2,2,Float64}((f(x),2*f(x),3*f(x),4*f(x)))
    qp_values = analytical(f_tensor)
    point_vars = project(proj, qp_values, qr)
    if order == 1
        ae = [Tensor{2,2,Float64}((f_approx(i),2*f_approx(i),3*f_approx(i),4*f_approx(i))) for i in 1:4]
    elseif order == 2
        ae = compute_vertex_values(grid, f_tensor)
    end
    @test point_vars[1:4] ≈ ae

    # SymmetricTensor
    f_stensor(x) = SymmetricTensor{2,2,Float64}((f(x),2*f(x),3*f(x)))
    qp_values = analytical(f_stensor)
    point_vars = project(proj, qp_values, qr)
    if order == 1
        ae = [SymmetricTensor{2,2,Float64}((f_approx(i),2*f_approx(i),3*f_approx(i))) for i in 1:4]
    elseif order == 2
        ae = compute_vertex_values(grid, f_stensor)
    end
    @test point_vars[1:4] ≈ ae

    # Test error-path with bad qr
    if refshape == RefTetrahedron && order == 2
        bad_order = 2
    else
        bad_order = 1
    end
    @test_throws LinearAlgebra.PosDefException L2Projector(ip, grid; qr_lhs=QuadratureRule{dim,refshape}(bad_order), geom_ip=ip_geom)
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

function test_node_reordering()
    grid = generate_grid(Quadrilateral, (1, 1), Vec((0.,0.)), Vec((2.,2.)))
    dim = 2
    ip = Lagrange{dim, RefCube, 2}()
    ip_geo = Lagrange{dim, RefCube,1}()
    qr = QuadratureRule{dim, RefCube}(3)
    cv = CellScalarValues(qr, ip, ip_geo)

    f(x) = x[1]+x[2]

    qp_values = [[f(spatial_coordinate(cv, qp, getcoordinates(cell))) for qp in 1:getnquadpoints(cv)] for cell in CellIterator(grid)]

    projector = L2Projector(ip, grid)
    projected_vals_nodes = project(projector, qp_values, qr)
    projected_vals_dofs = project(projector, qp_values, qr; project_to_nodes=false)

    tol = 1e-12
    @test all(projected_vals_nodes - [0.0, 2.0, 2.0, 4.0] .< tol)
    @test all(projected_vals_dofs - [0., 2., 4., 2., 1., 3., 3., 1., 2.] .< tol)
end

@testset "Test L2-Projection" begin
    test_projection(1, RefCube)
    test_projection(1, RefTetrahedron)
    test_projection(2, RefCube)
    test_projection(2, RefTetrahedron)
    test_projection_mixedgrid()
end


dh = MixedDofHandler(grid)
field = Field(:_, ip, 2)

fh = FieldHandler([field], Set(1:1))
push!(dh, fh)
_, vertex_dict, edge_dict, face_dict = Ferrite.__close!(dh)