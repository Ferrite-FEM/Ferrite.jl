@testset "FaceValues" begin
for (scalar_interpol, quad_rule) in (
                                    (Lagrange{1, RefCube, 1}(), QuadratureRule{0, RefCube}(2)),
                                    (Lagrange{1, RefCube, 2}(), QuadratureRule{0, RefCube}(2)),
                                    (Lagrange{2, RefCube, 1}(), QuadratureRule{1, RefCube}(2)),
                                    (Lagrange{2, RefCube, 2}(), QuadratureRule{1, RefCube}(2)),
                                    (Lagrange{2, RefTetrahedron, 1}(), QuadratureRule{1, RefTetrahedron}(2)),
                                    (Lagrange{2, RefTetrahedron, 2}(), QuadratureRule{1, RefTetrahedron}(2)),
                                    (Lagrange{3, RefCube, 1}(), QuadratureRule{2, RefCube}(2)),
                                    (Serendipity{2, RefCube, 2}(), QuadratureRule{1, RefCube}(2)),
                                    (Lagrange{3, RefTetrahedron, 1}(), QuadratureRule{2, RefTetrahedron}(2)),
                                    (Lagrange{3, RefTetrahedron, 2}(), QuadratureRule{2, RefTetrahedron}(2)),
                                   )

    for func_interpol in (scalar_interpol, VectorizedInterpolation(scalar_interpol))
        geom_interpol = scalar_interpol # Tests below assume this
        n_basefunc_base = getnbasefunctions(scalar_interpol)
        fv = FaceValues(quad_rule, func_interpol, geom_interpol)
        ndim = Ferrite.getdim(func_interpol)
        n_basefuncs = getnbasefunctions(func_interpol)

        @test getnbasefunctions(fv) == n_basefuncs

        xs, n = valid_coordinates_and_normals(func_interpol)
        for face in 1:Ferrite.nfaces(func_interpol)
            reinit!(fv, xs, face)
            @test Ferrite.getcurrentface(fv) == face

            # We test this by applying a given deformation gradient on all the nodes.
            # Since this is a linear deformation we should get back the exact values
            # from the interpolation.
            u = Vec{ndim, Float64}[zero(Tensor{1,ndim}) for i in 1:n_basefunc_base]
            u_scal = zeros(n_basefunc_base)
            H = rand(Tensor{2, ndim})
            V = rand(Tensor{1, ndim})
            for i in 1:n_basefunc_base
                u[i] = H ⋅ xs[i]
                u_scal[i] = V ⋅ xs[i]
            end
            u_vector = reinterpret(Float64, u)

            for i in 1:length(getnquadpoints(fv))
                @test getnormal(fv, i) ≈ n[face]
                if func_interpol isa Ferrite.ScalarInterpolation
                    @test function_gradient(fv, i, u) ≈ H
                    @test function_symmetric_gradient(fv, i, u) ≈ 0.5(H + H')
                    @test function_divergence(fv, i, u_scal) ≈ sum(V)
                    @test function_divergence(fv, i, u) ≈ tr(H)
                    @test function_gradient(fv, i, u_scal) ≈ V
                    ndim == 3 && @test function_curl(fv, i, u) ≈ Ferrite.curl_from_gradient(H)
                    function_value(fv, i, u)
                    function_value(fv, i, u_scal)
                else # func_interpol isa Ferrite.VectorInterpolation
                    @test function_gradient(fv, i, u_vector) ≈ H
                    @test (@test_deprecated function_gradient(fv, i, u)) ≈ H
                    @test function_symmetric_gradient(fv, i, u_vector) ≈ 0.5(H + H')
                    @test (@test_deprecated function_symmetric_gradient(fv, i, u)) ≈ 0.5(H + H')
                    @test function_divergence(fv, i, u_vector) ≈ tr(H)
                    @test (@test_deprecated function_divergence(fv, i, u)) ≈ tr(H)
                    if ndim == 3
                        @test function_curl(fv, i, u_vector) ≈ Ferrite.curl_from_gradient(H)
                        @test (@test_deprecated function_curl(fv, i, u)) ≈ Ferrite.curl_from_gradient(H)
                    end
                    @test function_value(fv, i, u_vector) ≈ (@test_deprecated function_value(fv, i, u))
                end
            end

            # Test of volume
            vol = 0.0
            for i in 1:getnquadpoints(fv)
                vol += getdetJdV(fv,i)
            end
            let ip_base = func_interpol isa VectorizedInterpolation ? func_interpol.ip : func_interpol
                x_face = xs[[Ferrite.facedof_indices(ip_base)[face]...]]
                @test vol ≈ calculate_volume(Ferrite.getlowerdim(ip_base), x_face)
            end

            # Test quadrature rule after reinit! with ref. coords
            x = Ferrite.reference_coordinates(func_interpol)
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

        # test copy
        fvc = copy(fv)
        @test typeof(fv) == typeof(fvc)
        for fname in fieldnames(typeof(fv))
            v = getfield(fv, fname)
            v isa Ferrite.ScalarWrapper && continue
            vc = getfield(fvc, fname)
            if hasmethod(pointer, Tuple{typeof(v)})
                @test pointer(v) != pointer(vc)
            end
            @test v == vc
        end
    end
end

end # of testset
