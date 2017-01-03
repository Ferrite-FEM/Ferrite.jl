@testset "BoundaryValues" begin
for (func_interpol, quad_rule) in  ((Lagrange{1, RefCube, 1}(), QuadratureRule{0, RefCube}(2)),
                                    (Lagrange{1, RefCube, 2}(), QuadratureRule{0, RefCube}(2)),
                                    (Lagrange{2, RefCube, 1}(), QuadratureRule{1, RefCube}(2)),
                                    (Lagrange{2, RefCube, 2}(), QuadratureRule{1, RefCube}(2)),
                                    (Lagrange{2, RefTetrahedron, 1}(), QuadratureRule{1, RefTetrahedron}(2)),
                                    (Lagrange{2, RefTetrahedron, 2}(), QuadratureRule{1, RefTetrahedron}(2)),
                                    (Lagrange{3, RefCube, 1}(), QuadratureRule{2, RefCube}(2)),
                                    (Serendipity{2, RefCube, 2}(), QuadratureRule{1, RefCube}(2)),
                                    (Lagrange{3, RefTetrahedron, 1}(), QuadratureRule{2, RefTetrahedron}(2)))

    for fe_valtype in (BoundaryScalarValues, BoundaryVectorValues)
        bv = fe_valtype(quad_rule, func_interpol)
        ndim = getdim(func_interpol)
        n_basefuncs = getnbasefunctions(func_interpol)

        fe_valtype == BoundaryScalarValues && @test getnbasefunctions(bv) == n_basefuncs
        fe_valtype == BoundaryVectorValues && @test getnbasefunctions(bv) == n_basefuncs * getdim(func_interpol)

        x = valid_coordinates(func_interpol)
        boundary_nodes, cell_nodes = topology_test_nodes(func_interpol)
        for boundary in 1:JuAFEM.getnboundaries(func_interpol)
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
            u_vector = reinterpret(Float64, u, (n_basefuncs*ndim,))

            for i in 1:length(getpoints(boundary_quad_rule))
                @test function_gradient(bv, i, u) ≈ H
                @test function_symmetric_gradient(bv, i, u) ≈ 0.5(H + H')
                @test function_divergence(bv, i, u) ≈ trace(H)
                function_value(bv, i, u)
                if isa(bv, BoundaryScalarValues)
                    @test function_gradient(bv, i, u_scal) ≈ V
                    function_value(bv, i, u_scal)
                elseif isa(bv, BoundaryVectorValues)
                    @test function_gradient(bv, i, u_vector) ≈ function_gradient(bv, i, u) ≈ H
                    @test function_value(bv, i, u_vector) ≈ function_value(bv, i, u)
                    @test function_divergence(bv, i, u_vector) ≈ function_divergence(bv, i, u) ≈ trace(H)
                end
            end

            # Test of volume
            vol = 0.0
            for i in 1:getnquadpoints(bv)
                vol += getdetJdV(bv,i)
            end
            x_boundary = x[[JuAFEM.getboundarylist(func_interpol)[boundary]...]]
            @test vol ≈ calculate_volume(JuAFEM.getlowerdim(func_interpol), x_boundary)

            # Test of utility functions
            @test getfunctioninterpolation(bv) == func_interpol
            @test getgeometryinterpolation(bv) == func_interpol

            # Test quadrature rule after reinit! with ref. coords
            x = reference_coordinates(func_interpol)
            reinit!(bv, x, boundary)
            vol = 0.0
            for i in 1:getnquadpoints(bv)
                vol += getdetJdV(bv, i)
            end
            @test vol ≈ reference_volume(func_interpol, boundary)

            # Test spatial coordinate (after reinit with ref.coords we should get back the quad_points)
            for (i, qp_x) in enumerate(getpoints(boundary_quad_rule))
                @test spatial_coordinate(bv, i, x) ≈ qp_x
            end

        end

        # Test boundary number calculation
        boundary_nodes, cell_nodes = topology_test_nodes(func_interpol)
        for boundary in 1:JuAFEM.getnboundaries(func_interpol)
            @test getboundarynumber(boundary_nodes[boundary], cell_nodes, func_interpol) == boundary
        end
        @test_throws ArgumentError getboundarynumber(boundary_nodes[JuAFEM.getnboundaries(func_interpol)+1], cell_nodes, func_interpol)
    end
end

end # of testset
