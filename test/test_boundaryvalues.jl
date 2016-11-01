@testset "BoundaryCellValues" begin
for (function_space, quad_rule) in  ((Lagrange{1, RefCube, 1}(), QuadratureRule{0, RefCube}(2)),
                                     (Lagrange{1, RefCube, 2}(), QuadratureRule{0, RefCube}(2)),
                                     (Lagrange{2, RefCube, 1}(), QuadratureRule{1, RefCube}(2)),
                                     (Lagrange{2, RefCube, 2}(), QuadratureRule{1, RefCube}(2)),
                                     (Lagrange{2, RefTetrahedron, 1}(), QuadratureRule{1, RefTetrahedron}(2)),
                                     (Lagrange{2, RefTetrahedron, 2}(), QuadratureRule{1, RefTetrahedron}(2)),
                                     (Lagrange{3, RefCube, 1}(), QuadratureRule{2, RefCube}(2)),
                                     (Serendipity{2, RefCube, 2}(), QuadratureRule{1, RefCube}(2)),
                                     (Lagrange{3, RefTetrahedron, 1}(), QuadratureRule{2, RefTetrahedron}(2)))

    bv = BoundaryScalarValues(quad_rule, function_space)
    ndim = getdim(function_space)
    n_basefuncs = getnbasefunctions(function_space)

    x = valid_coordinates(function_space)
    boundary_nodes, cell_nodes = topology_test_nodes(function_space)
    for boundary in 1:JuAFEM.getnboundaries(function_space)
        reinit!(bv, x, boundary)
        boundary_quad_rule = getquadrule(bv)
        @test JuAFEM.getcurrentboundary(bv) == boundary

        # We test this by applying a given deformation gradient on all the nodes.
        # Since this is a linear deformation we should get back the exact values
        # from the interpolation.
        u = Vec{ndim, Float64}[zero(Tensor{1,ndim}) for i in 1:n_basefuncs]
        u_scal = zeros(n_basefuncs)
        H = rand(Tensor{2, ndim})
        V = rand(Tensor{1, ndim})
        for i in 1:n_basefuncs
            u[i] = H ⋅ x[i]
            u_scal[i] = V ⋅ x[i]
        end

        for i in 1:length(getpoints(boundary_quad_rule))
            @test function_gradient(bv, i, u) ≈ H
            @test function_symmetric_gradient(bv, i, u) ≈ 0.5(H + H')
            @test function_divergence(bv, i, u) ≈ trace(H)
            @test function_gradient(bv, i, u_scal) ≈ V
            function_value(bv, i, u_scal)
            function_value(bv, i, u)
        end

        # Test of volume
        vol = 0.0
        for i in 1:getnquadpoints(bv)
            vol += getdetJdV(bv,i)
        end
        x_boundary = x[[JuAFEM.getboundarylist(function_space)[boundary]...]]
        @test vol ≈ calculate_volume(JuAFEM.getlowerdim(function_space), x_boundary)

        # Test of utility functions
        @test getfunctionspace(bv) == function_space
        @test getgeometricspace(bv) == function_space

        # Test quadrature rule after reinit! with ref. coords
        x = reference_coordinates(function_space)
        reinit!(bv, x, boundary)
        vol = 0.0
        for i in 1:getnquadpoints(bv)
            vol += getdetJdV(bv, i)
        end
        @test vol ≈ reference_volume(function_space, boundary)

        # Test spatial coordinate (after reinit with ref.coords we should get back the quad_points)
        for (i, qp_x) in enumerate(getpoints(boundary_quad_rule))
            @test spatial_coordinate(bv, i, x) ≈ qp_x
        end

    end

    # Test boundary number calculation
    boundary_nodes, cell_nodes = topology_test_nodes(function_space)
    for boundary in 1:JuAFEM.getnboundaries(function_space)
        @test getboundarynumber(boundary_nodes[boundary], cell_nodes, function_space) == boundary
    end
    @test_throws ArgumentError getboundarynumber(boundary_nodes[JuAFEM.getnboundaries(function_space)+1], cell_nodes, function_space)


end

end # of testset
