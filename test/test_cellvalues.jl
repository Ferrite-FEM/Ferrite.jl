@testset "CellValues" begin
for (func_interpol, quad_rule) in  (
                                    (Lagrange{1, RefCube, 1}(), QuadratureRule{1, RefCube}(2)),
                                    (Lagrange{1, RefCube, 2}(), QuadratureRule{1, RefCube}(2)),
                                    (Lagrange{2, RefCube, 1}(), QuadratureRule{2, RefCube}(2)),
                                    (Lagrange{2, RefCube, 2}(), QuadratureRule{2, RefCube}(2)),
                                    (Lagrange{2, RefTetrahedron, 1}(), QuadratureRule{2, RefTetrahedron}(2)),
                                    (Lagrange{2, RefTetrahedron, 2}(), QuadratureRule{2, RefTetrahedron}(2)),
                                    (Lagrange{3, RefCube, 1}(), QuadratureRule{3, RefCube}(2)),
                                    (Serendipity{2, RefCube, 2}(), QuadratureRule{2, RefCube}(2)),
                                    (Lagrange{3, RefTetrahedron, 1}(), QuadratureRule{3, RefTetrahedron}(2)),
                                    (Lagrange{3, RefTetrahedron, 2}(), QuadratureRule{3, RefTetrahedron}(2))
                                   )

    for fe_valtype in (CellScalarValues, CellVectorValues)
        cv = fe_valtype(quad_rule, func_interpol)
        ndim = JuAFEM.getdim(func_interpol)
        n_basefuncs = getnbasefunctions(func_interpol)

        fe_valtype == CellScalarValues && @test getnbasefunctions(cv) == n_basefuncs
        fe_valtype == CellVectorValues && @test getnbasefunctions(cv) == n_basefuncs * JuAFEM.getdim(func_interpol)

        x, n = valid_coordinates_and_normals(func_interpol)
        reinit!(cv, x)

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

        for i in 1:length(getpoints(quad_rule))
            @test function_gradient(cv, i, u) ≈ H
            @test function_symmetric_gradient(cv, i, u) ≈ 0.5(H + H')
            @test function_divergence(cv, i, u) ≈ trace(H)
            function_value(cv, i, u)
            if isa(cv, CellScalarValues)
                @test function_gradient(cv, i, u_scal) ≈ V
                function_value(cv, i, u_scal)
            elseif isa(cv, CellVectorValues)
                @test function_gradient(cv, i, u_vector) ≈ function_gradient(cv, i, u) ≈ H
                @test function_value(cv, i, u_vector) ≈ function_value(cv, i, u)
                @test function_divergence(cv, i, u_vector) ≈ function_divergence(cv, i, u) ≈ trace(H)
            end
        end

        # Test of volume
        vol = 0.0
        for i in 1:getnquadpoints(cv)
            vol += getdetJdV(cv,i)
        end
        @test vol ≈ calculate_volume(func_interpol, x)

        # Test quadrature rule after reinit! with ref. coords
        x = reference_coordinates(func_interpol)
        reinit!(cv, x)
        vol = 0.0
        for i in 1:getnquadpoints(cv)
            vol += getdetJdV(cv,i)
        end
        @test vol ≈ reference_volume(func_interpol)

        # Test spatial coordinate (after reinit with ref.coords we should get back the quad_points)
        for (i, qp_x) in enumerate(getpoints(quad_rule))
            @test spatial_coordinate(cv, i, x) ≈ qp_x
        end

        # test copy
        cvc = copy(cv)
        for fname in fieldnames(cv)
            @test typeof(cv) == typeof(cvc)
            @test pointer(getfield(cv, fname)) != pointer(getfield(cvc, fname))
            @test getfield(cv, fname) == getfield(cvc, fname)
        end
    end
end

end # of testset
