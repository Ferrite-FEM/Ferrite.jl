@testset "CellValues" begin
for (func_interpol, quad_rule) in  (
                                    (Lagrange{1, RefCube, 1}(), QuadratureRule{1, RefCube}(2)),
                                    (Lagrange{1, RefCube, 2}(), QuadratureRule{1, RefCube}(2)),
                                    (Lagrange{2, RefCube, 1}(), QuadratureRule{2, RefCube}(2)),
                                    (Lagrange{2, RefCube, 2}(), QuadratureRule{2, RefCube}(2)),
                                    (Lagrange{2, RefTetrahedron, 1}(), QuadratureRule{2, RefTetrahedron}(2)),
                                    (Lagrange{2, RefTetrahedron, 2}(), QuadratureRule{2, RefTetrahedron}(2)),
                                    (Lagrange{2, RefTetrahedron, 3}(), QuadratureRule{2, RefTetrahedron}(2)),
                                    (Lagrange{2, RefTetrahedron, 4}(), QuadratureRule{2, RefTetrahedron}(2)),
                                    (Lagrange{2, RefTetrahedron, 5}(), QuadratureRule{2, RefTetrahedron}(2)),
                                    (Lagrange{3, RefCube, 1}(), QuadratureRule{3, RefCube}(2)),
                                    (Serendipity{2, RefCube, 2}(), QuadratureRule{2, RefCube}(2)),
                                    (Lagrange{3, RefTetrahedron, 1}(), QuadratureRule{3, RefTetrahedron}(2)),
                                    (Lagrange{3, RefTetrahedron, 2}(), QuadratureRule{3, RefTetrahedron}(2))
                                   )

    for fe_valtype in (CellScalarValues, CellVectorValues)
        geom_interpol = func_interpol # Tests below assume this
        n_basefunc_base = getnbasefunctions(func_interpol)
        if fe_valtype == CellVectorValues
            func_interpol = VectorizedInterpolation(func_interpol)
        end
        cv = fe_valtype(quad_rule, func_interpol, geom_interpol)
        ndim = Ferrite.getdim(func_interpol)
        n_basefuncs = getnbasefunctions(func_interpol)

        @test getnbasefunctions(cv) == n_basefuncs

        x, n = valid_coordinates_and_normals(func_interpol)
        reinit!(cv, x)

        # We test this by applying a given deformation gradient on all the nodes.
        # Since this is a linear deformation we should get back the exact values
        # from the interpolation.
        u = Vec{ndim, Float64}[zero(Tensor{1,ndim}) for i in 1:n_basefunc_base]
        u_scal = zeros(n_basefunc_base)
        H = rand(Tensor{2, ndim})
        V = rand(Tensor{1, ndim})
        for i in 1:n_basefunc_base
            u[i] = H ⋅ x[i]
            u_scal[i] = V ⋅ x[i]
        end
        u_vector = reinterpret(Float64, u)

        for i in 1:length(getpoints(quad_rule))
            if isa(cv, CellScalarValues)
                @test function_gradient(cv, i, u) ≈ H
                @test function_symmetric_gradient(cv, i, u) ≈ 0.5(H + H')
                @test function_divergence(cv, i, u) ≈ tr(H)
                @test function_gradient(cv, i, u_scal) ≈ V
                ndim == 3 && @test function_curl(cv, i, u) ≈ Ferrite.curl_from_gradient(H)
                function_value(cv, i, u)
                function_value(cv, i, u_scal)
            elseif isa(cv, CellVectorValues)
                @test function_gradient(cv, i, u_vector)  ≈ H
                @test (@test_deprecated function_gradient(cv, i, u)) ≈ H
                @test function_symmetric_gradient(cv, i, u_vector) ≈ 0.5(H + H')
                @test (@test_deprecated function_symmetric_gradient(cv, i, u)) ≈ 0.5(H + H')
                @test function_divergence(cv, i, u_vector) ≈ tr(H)
                @test (@test_deprecated function_divergence(cv, i, u)) ≈ tr(H)
                if ndim == 3
                    @test function_curl(cv, i, u_vector) ≈ Ferrite.curl_from_gradient(H)
                    @test (@test_deprecated function_curl(cv, i, u)) ≈ Ferrite.curl_from_gradient(H)
                end
                function_value(cv, i, u_vector) ≈ (@test_deprecated function_value(cv, i, u))
            end
        end

        # Test of volume
        vol = 0.0
        for i in 1:getnquadpoints(cv)
            vol += getdetJdV(cv,i)
        end
        @test vol ≈ calculate_volume(func_interpol, x)

        # Test quadrature rule after reinit! with ref. coords
        x = Ferrite.reference_coordinates(func_interpol)
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
        @test typeof(cv) == typeof(cvc)
        for fname in fieldnames(typeof(cv))
            v = getfield(cv, fname)
            vc = getfield(cvc, fname)
            if hasmethod(pointer, Tuple{typeof(v)})
                @test pointer(getfield(cv, fname)) != pointer(getfield(cvc, fname))
            end
            @test v == vc
        end
    end
end

@testset "#265: error message for incompatible geometric interpolation" begin
    dim = 1
    deg = 1
    grid = generate_grid(Line, (2,))
    ip_fe = Lagrange{dim, RefCube, deg}()
    dh = DofHandler(grid)
    add!(dh, :u, ip_fe)
    close!(dh);
    cell = first(CellIterator(dh))
    ip_geo = Lagrange{dim, RefCube, 2}()
    qr = QuadratureRule{dim, RefCube}(deg+1)
    cv = CellScalarValues(qr, ip_fe, ip_geo)
    res = @test_throws ArgumentError reinit!(cv, cell)
    @test occursin("265", res.value.msg)
    ip_geo = Lagrange{dim, RefCube, 1}()
    cv = CellScalarValues(qr, ip_fe, ip_geo)
    reinit!(cv, cell)
end

@testset "error paths in function_* and reinit!" begin
    dim = 2
    qp = 1
    ip = Lagrange{dim,RefTetrahedron,1}()
    qr = QuadratureRule{dim,RefTetrahedron}(1)
    qr_f = QuadratureRule{1,RefTetrahedron}(1)
    csv = CellScalarValues(qr, ip)
    cvv = CellVectorValues(qr, VectorizedInterpolation(ip))
    fsv = FaceScalarValues(qr_f, ip)
    fvv = FaceVectorValues(qr_f, VectorizedInterpolation(ip))
    x, n = valid_coordinates_and_normals(ip)
    reinit!(csv, x)
    reinit!(cvv, x)
    reinit!(fsv, x, 1)
    reinit!(fvv, x, 1)

    # Wrong number of coordinates
    xx = [x; x]
    @test_throws ArgumentError reinit!(csv, xx)
    @test_throws ArgumentError reinit!(cvv, xx)
    @test_throws ArgumentError reinit!(fsv, xx, 1)
    @test_throws ArgumentError reinit!(fvv, xx, 1)

    @test_throws ArgumentError spatial_coordinate(csv, qp, xx)
    @test_throws ArgumentError spatial_coordinate(cvv, qp, xx)
    @test_throws ArgumentError spatial_coordinate(fsv, qp, xx)
    @test_throws ArgumentError spatial_coordinate(fvv, qp, xx)

    # Wrong number of (local) dofs
    # Scalar values, scalar dofs
    ue = rand(getnbasefunctions(csv) + 1)
    @test_throws ArgumentError function_value(csv, qp, ue)
    @test_throws ArgumentError function_gradient(csv, qp, ue)
    # Vector values, scalar dofs
    ue = rand(getnbasefunctions(cvv) + 1)
    @test_throws ArgumentError function_value(cvv, qp, ue)
    @test_throws ArgumentError function_gradient(cvv, qp, ue)
    @test_throws ArgumentError function_divergence(cvv, qp, ue)
    # Scalar values, vector dofs
    ue = [rand(Vec{dim}) for _ in 1:(getnbasefunctions(csv) + 1)]
    @test_throws ArgumentError function_value(csv, qp, ue)
    @test_throws ArgumentError function_gradient(csv, qp, ue)
    @test_throws ArgumentError function_divergence(csv, qp, ue)
end

end # of testset
