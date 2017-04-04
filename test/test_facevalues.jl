@testset "FaceValues" begin
for (func_interpol, quad_rule) in  (
                                    (Lagrange{1, RefCube, 1}(), QuadratureRule{0, RefCube}(2)),
                                    (Lagrange{1, RefCube, 2}(), QuadratureRule{0, RefCube}(2)),
                                    (Lagrange{2, RefCube, 1}(), QuadratureRule{1, RefCube}(2)),
                                    (Lagrange{2, RefCube, 2}(), QuadratureRule{1, RefCube}(2)),
                                    (Lagrange{2, RefTetrahedron, 1}(), QuadratureRule{1, RefTetrahedron}(2)),
                                    (Lagrange{2, RefTetrahedron, 2}(), QuadratureRule{1, RefTetrahedron}(2)),
                                    (Lagrange{3, RefCube, 1}(), QuadratureRule{2, RefCube}(2)),
                                    (Serendipity{2, RefCube, 2}(), QuadratureRule{1, RefCube}(2)),
                                    (Lagrange{3, RefTetrahedron, 1}(), QuadratureRule{2, RefTetrahedron}(2)),
                                    (Lagrange{3, RefTetrahedron, 2}(), QuadratureRule{2, RefTetrahedron}(2))
                                   )

    for fe_valtype in (FaceScalarValues, FaceVectorValues)
        fv = fe_valtype(quad_rule, func_interpol)
        ndim = getdim(func_interpol)
        n_basefuncs = getnbasefunctions(func_interpol)

        fe_valtype == FaceScalarValues && @test getnbasefunctions(fv) == n_basefuncs
        fe_valtype == FaceVectorValues && @test getnbasefunctions(fv) == n_basefuncs * getdim(func_interpol)

        xs, n = valid_coordinates_and_normals(func_interpol)
        face_nodes, cell_nodes = topology_test_nodes(func_interpol)
        for face in 1:JuAFEM.getnfaces(func_interpol)
            reinit!(fv, xs, face)
            @test JuAFEM.getcurrentface(fv) == face

            # We test this by applying a given deformation gradient on all the nodes.
            # Since this is a linear deformation we should get back the exact values
            # from the interpolation.
            u = Vec{ndim, Float64}[zero(Tensor{1,ndim}) for i in 1:n_basefuncs]
            u_scal = zeros(n_basefuncs)
            H = rand(Tensor{2, ndim})
            V = rand(Tensor{1, ndim})
            for i in 1:n_basefuncs
                u[i] = H ⋅ xs[i]
                u_scal[i] = V ⋅ xs[i]
            end
            u_vector = reinterpret(Float64, u, (n_basefuncs*ndim,))

            for i in 1:length(getnquadpoints(fv))
                @test getnormal(fv, i) ≈ n[face]
                @test function_gradient(fv, i, u) ≈ H
                @test function_symmetric_gradient(fv, i, u) ≈ 0.5(H + H')
                @test function_divergence(fv, i, u) ≈ trace(H)
                function_value(fv, i, u)
                if isa(fv, FaceScalarValues)
                    @test function_gradient(fv, i, u_scal) ≈ V
                    function_value(fv, i, u_scal)
                elseif isa(fv, FaceVectorValues)
                    @test function_gradient(fv, i, u_vector) ≈ function_gradient(fv, i, u) ≈ H
                    @test function_value(fv, i, u_vector) ≈ function_value(fv, i, u)
                    @test function_divergence(fv, i, u_vector) ≈ function_divergence(fv, i, u) ≈ trace(H)
                end
            end

            # Test of volume
            vol = 0.0
            for i in 1:getnquadpoints(fv)
                vol += getdetJdV(fv,i)
            end
            x_face = xs[[JuAFEM.getfacelist(func_interpol)[face]...]]
            @test vol ≈ calculate_volume(JuAFEM.getlowerdim(func_interpol), x_face)

            # Test quadrature rule after reinit! with ref. coords
            x = reference_coordinates(func_interpol)
            reinit!(fv, x, face)
            vol = 0.0
            for i in 1:getnquadpoints(fv)
                vol += getdetJdV(fv, i)
            end
            @test vol ≈ reference_volume(func_interpol, face)

            # Test spatial coordinate (after reinit with ref.coords we should get back the quad_points)
            # TODO: Renable somehow after quad rule is no longer stored in FaceValues
            #for (i, qp_x) in enumerate(getpoints(quad_rule))
            #    @test spatial_coordinate(fv, i, x) ≈ qp_x
            #end

        end

        # Test face number calculation
        face_nodes, cell_nodes = topology_test_nodes(func_interpol)
        for face in 1:JuAFEM.getnfaces(func_interpol)
            @test getfacenumber(face_nodes[face], cell_nodes, func_interpol) == face
        end
        @test_throws ArgumentError getfacenumber(face_nodes[JuAFEM.getnfaces(func_interpol)+1], cell_nodes, func_interpol)
    end
end

end # of testset
