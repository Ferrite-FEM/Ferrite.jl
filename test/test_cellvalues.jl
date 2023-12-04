@testset "CellValues" begin
@testset "ip=$scalar_interpol quad_rule=$(typeof(quad_rule))" for (scalar_interpol, quad_rule) in  (
                                    (Lagrange{RefLine, 1}(), QuadratureRule{RefLine}(2)),
                                    (Lagrange{RefLine, 2}(), QuadratureRule{RefLine}(2)),
                                    (Lagrange{RefQuadrilateral, 1}(), QuadratureRule{RefQuadrilateral}(2)),
                                    (Lagrange{RefQuadrilateral, 2}(), QuadratureRule{RefQuadrilateral}(2)),
                                    (Lagrange{RefTriangle, 1}(), QuadratureRule{RefTriangle}(2)),
                                    (Lagrange{RefTriangle, 2}(), QuadratureRule{RefTriangle}(2)),
                                    (Lagrange{RefTriangle, 3}(), QuadratureRule{RefTriangle}(2)),
                                    (Lagrange{RefTriangle, 4}(), QuadratureRule{RefTriangle}(2)),
                                    (Lagrange{RefTriangle, 5}(), QuadratureRule{RefTriangle}(2)),
                                    (Lagrange{RefHexahedron, 1}(), QuadratureRule{RefHexahedron}(2)),
                                    (Serendipity{RefQuadrilateral, 2}(), QuadratureRule{RefQuadrilateral}(2)),
                                    (Lagrange{RefTriangle, 1}(), QuadratureRule{RefTriangle}(2)),
                                    (Lagrange{RefTetrahedron, 2}(), QuadratureRule{RefTetrahedron}(2)),
                                    (Lagrange{RefPrism, 2}(), QuadratureRule{RefPrism}(2)),
                                    (Lagrange{RefPyramid, 2}(), QuadratureRule{RefPyramid}(2)),
                                   )

    for func_interpol in (scalar_interpol, VectorizedInterpolation(scalar_interpol))
        geom_interpol = scalar_interpol # Tests below assume this
        n_basefunc_base = getnbasefunctions(scalar_interpol)
        cv = CellValues(quad_rule, func_interpol, geom_interpol)
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

        for i in 1:getnquadpoints(cv)
            if func_interpol isa Ferrite.ScalarInterpolation
                @test function_gradient(cv, i, u) ≈ H
                @test function_symmetric_gradient(cv, i, u) ≈ 0.5(H + H')
                @test function_divergence(cv, i, u_scal) ≈ sum(V)
                @test function_divergence(cv, i, u) ≈ tr(H)
                @test function_gradient(cv, i, u_scal) ≈ V
                ndim == 3 && @test function_curl(cv, i, u) ≈ Ferrite.curl_from_gradient(H)
                function_value(cv, i, u)
                function_value(cv, i, u_scal)
            else# func_interpol isa Ferrite.VectorInterpolation
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
                @test function_value(cv, i, u_vector) ≈ (@test_deprecated function_value(cv, i, u))
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
        for (i, qp_x) in pairs(Ferrite.getpoints(quad_rule))
            @test spatial_coordinate(cv, i, x) ≈ qp_x
        end

        @testset "copy(::CellValues)" begin
            cvc = copy(cv)
            @test typeof(cv) == typeof(cvc)

            # Test that all mutable types in FunctionValues and GeometryMapping have been copied
            for key in (:fun_values, :geo_mapping)
                val = getfield(cv, key)
                valc = getfield(cvc, key)
                for fname in fieldnames(typeof(val))
                    v = getfield(val, fname)
                    vc = getfield(valc, fname)
                    isbits(v) || @test v !== vc
                    @test v == vc
                end
            end
            # Test that qr and detJdV is copied as expected. 
            # Note that qr remain aliased, as defined by `copy(qr)=qr`, see quadrature.jl.
            for fname in (:qr, :detJdV)
                v = getfield(cv, fname)
                vc = getfield(cvc, fname)
                fname === :qr || @test v !== vc
                @test v == vc
            end
        end
    end
end

@testset "#265: error message for incompatible geometric interpolation" begin
    dim = 1
    deg = 1
    grid = generate_grid(Line, (2,))
    ip_fe = Lagrange{RefLine, deg}()
    dh = DofHandler(grid)
    add!(dh, :u, ip_fe)
    close!(dh);
    cell = first(CellIterator(dh))
    ip_geo = Lagrange{RefLine, 2}()
    qr = QuadratureRule{RefLine}(deg+1)
    cv = CellValues(qr, ip_fe, ip_geo)
    res = @test_throws ArgumentError reinit!(cv, cell)
    @test occursin("265", res.value.msg)
    ip_geo = Lagrange{RefLine, 1}()
    cv = CellValues(qr, ip_fe, ip_geo)
    reinit!(cv, cell)
end

@testset "error paths in function_* and reinit!" begin
    dim = 2
    qp = 1
    ip = Lagrange{RefTriangle,1}()
    qr = QuadratureRule{RefTriangle}(1)
    qr_f = FaceQuadratureRule{RefTriangle}(1)
    csv = CellValues(qr, ip)
    cvv = CellValues(qr, VectorizedInterpolation(ip))
    csv_embedded = CellValues(qr, ip, ip^3)
    fsv = FaceValues(qr_f, ip)
    fvv = FaceValues(qr_f, VectorizedInterpolation(ip))
    fsv_embedded = FaceValues(qr_f, ip, ip^3)
    
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

    # Wrong dimension of coordinates 
    @test_throws ArgumentError reinit!(csv_embedded, x)
    @test_throws ArgumentError reinit!(fsv_embedded, x, 1)

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

@testset "Embedded elements" begin
    @testset "Scalar/vector on curves (vdim = $vdim)" for vdim in (0, 1, 2, 3)
        ip_base = Lagrange{RefLine,1}()
        ip = vdim > 0 ? ip_base^vdim : ip_base
        ue = 2 * rand(getnbasefunctions(ip))
        qr = QuadratureRule{RefLine}(1)
        # Reference values
        csv1 = CellValues(qr, ip)
        reinit!(csv1, [Vec((0.0,)), Vec((1.0,))])

        ## sdim = 2, Consistency with 1D
        csv2 = CellValues(qr, ip, ip_base^2)
        reinit!(csv2, [Vec((0.0, 0.0)), Vec((1.0, 0.0))])
        # Test spatial interpolation
        @test spatial_coordinate(csv2, 1, [Vec((0.0, 0.0)), Vec((1.0, 0.0))]) == Vec{2}((0.5, 0.0))
        # Test volume
        @test getdetJdV(csv1, 1) == getdetJdV(csv2, 1)
        # Test flip
        @test shape_value(csv1, 1, 1) == shape_value(csv2, 1, 1)
        @test shape_value(csv1, 1, 2) == shape_value(csv2, 1, 2)
        # Test evals
        @test function_value(csv1, 1, ue) == function_value(csv2, 1, ue)
        if vdim == 0
            @test function_gradient(csv1, 1, ue)[1] == function_gradient(csv2, 1, ue)[1]
            @test 0.0 == function_gradient(csv2, 1, ue)[2]
        else
            @test function_gradient(csv1, 1, ue)[:, 1] == function_gradient(csv2, 1, ue)[:, 1]
            @test                          zeros(vdim) == function_gradient(csv2, 1, ue)[:, 2]
        end

        ## sdim = 3, Consistency with 1D
        csv3 = CellValues(qr, ip, ip_base^3)
        reinit!(csv3, [Vec((0.0, 0.0, 0.0)), Vec((1.0, 0.0, 0.0))])
        # Test spatial interpolation
        @test spatial_coordinate(csv3, 1, [Vec((0.0, 0.0, 0.0)), Vec((1.0, 0.0, 0.0))]) == Vec{3}((0.5, 0.0, 0.0))
        # Test volume
        @test getdetJdV(csv1, 1) == getdetJdV(csv3, 1)
        # Test flip
        @test shape_value(csv1, 1, 1) == shape_value(csv3, 1, 1)
        @test shape_value(csv1, 1, 2) == shape_value(csv3, 1, 2)
        # Test evals
        @test function_value(csv1, 1, ue) == function_value(csv3, 1, ue)
        if vdim == 0
            @test function_gradient(csv1, 1, ue)[1] == function_gradient(csv3, 1, ue)[1]
            @test 0.0 == function_gradient(csv3, 1, ue)[2]
            @test 0.0 == function_gradient(csv3, 1, ue)[3]
        else
            @test function_gradient(csv1, 1, ue)[:, 1] == function_gradient(csv3, 1, ue)[:, 1]
            @test zeros(vdim, 2)                       == function_gradient(csv3, 1, ue)[:, 2:3]
        end

        ## sdim = 3, Consistency in 2D
        reinit!(csv2, [Vec((-1.0, 2.0)), Vec((3.0, -4.0))])
        reinit!(csv3, [Vec((-1.0, 2.0, 0.0)), Vec((3.0, -4.0, 0.0))])
        # Test spatial interpolation
        @test spatial_coordinate(csv2, 1, [Vec((-1.0, 2.0)), Vec((3.0, -4.0))]) == Vec{2}((1.0, -1.0))
        @test spatial_coordinate(csv3, 1, [Vec((-1.0, 2.0, 0.0)), Vec((3.0, -4.0, 0.0))]) == Vec{3}((1.0, -1.0, 0.0))
        # Test volume
        @test getdetJdV(csv2, 1) == getdetJdV(csv3, 1)
        # Test evals
        @test function_value(csv2, 1, ue) == function_value(csv3, 1, ue)
        if vdim == 0
            @test function_gradient(csv2, 1, ue)[1:2] == function_gradient(csv3, 1, ue)[1:2]
            @test                                 0.0 == function_gradient(csv3, 1, ue)[3]
        else
            @test function_gradient(csv2, 1, ue)[:, 1:2] == function_gradient(csv3, 1, ue)[:, 1:2]
            @test                            zeros(vdim) == function_gradient(csv3, 1, ue)[:, 3]
        end
        ## Change plane
        reinit!(csv3, [Vec((-1.0, 0.0, 2.0)), Vec((3.0, 0.0, -4.0))])
        # Test spatial interpolation
        @test spatial_coordinate(csv3, 1, [Vec((-1.0, 0.0, 2.0)), Vec((3.0, 0.0, -4.0))]) == Vec{3}((1.0, 0.0, -1.0))
        # Test volume
        @test getdetJdV(csv2, 1) == getdetJdV(csv3, 1)
        # Test evals
        @test function_value(csv2, 1, ue) == function_value(csv3, 1, ue)
        if vdim == 0
            @test function_gradient(csv2, 1, ue)[1] == function_gradient(csv3, 1, ue)[1]
            @test                               0.0 == function_gradient(csv3, 1, ue)[2]
            @test function_gradient(csv2, 1, ue)[2] == function_gradient(csv3, 1, ue)[3]
        else
            @test function_gradient(csv2, 1, ue)[:, 1] == function_gradient(csv3, 1, ue)[:, 1]
            @test                          zeros(vdim) == function_gradient(csv3, 1, ue)[:, 2]
            @test function_gradient(csv2, 1, ue)[:, 2] == function_gradient(csv3, 1, ue)[:, 3]
        end
    end

    @testset "Scalar/vector on surface (vdim = $vdim)" for vdim in (0, 1, 2, 3)
        ip_base = Lagrange{RefQuadrilateral,1}()
        ip = vdim > 0 ? ip_base^vdim : ip_base
        ue = rand(getnbasefunctions(ip))
        qr = QuadratureRule{RefQuadrilateral}(1)
        csv2 = CellValues(qr, ip)
        csv3 = CellValues(qr, ip, ip_base^3)
        reinit!(csv2, [Vec((-1.0,-1.0)), Vec((1.0,-1.0)), Vec((1.0,1.0)), Vec((-1.0,1.0))])
        reinit!(csv3, [Vec((-1.0,-1.0,0.0)), Vec((1.0,-1.0,0.0)), Vec((1.0,1.0,0.0)), Vec((-1.0,1.0,0.0))])
        # Test spatial interpolation
        @test spatial_coordinate(csv2, 1, [Vec((-1.0,-1.0)), Vec((1.0,-1.0)), Vec((1.0,1.0)), Vec((-1.0,1.0))]) == Vec{2}((0.0, 0.0))
        @test spatial_coordinate(csv3, 1, [Vec((-1.0,-1.0,0.0)), Vec((1.0,-1.0,0.0)), Vec((1.0,1.0,0.0)), Vec((-1.0,1.0,0.0))]) == Vec{3}((0.0, 0.0, 0.0))
        # Test volume
        @test getdetJdV(csv2, 1) == getdetJdV(csv3, 1)
        # Test evals
        @test function_value(csv2, 1, ue) == function_value(csv3, 1, ue)
        if vdim == 0
            @test function_gradient(csv2, 1, ue)[1:2] == function_gradient(csv3, 1, ue)[1:2]
            @test                                 0.0 == function_gradient(csv3, 1, ue)[3]
        else
            @test function_gradient(csv2, 1, ue)[:, 1:2] == function_gradient(csv3, 1, ue)[:, 1:2]
            @test                            zeros(vdim) == function_gradient(csv3, 1, ue)[:, 3]
        end
    end
end

@testset "CellValues constructor entry points" begin
    qr = QuadratureRule{RefTriangle}(1)
    
    for fun_ip in (Lagrange{RefTriangle, 1}(), Lagrange{RefTriangle, 2}()^2)
        value_type(T) = fun_ip isa ScalarInterpolation ? T : Vec{2, T}
        grad_type(T) = fun_ip isa ScalarInterpolation ? Vec{2, T} : Tensor{2, 2, T, 4}
        # Quadrature + scalar function
        cv = CellValues(qr, fun_ip)
        @test Ferrite.shape_value_type(cv) == value_type(Float64)
        @test Ferrite.shape_gradient_type(cv) == grad_type(Float64)
        @test Ferrite.get_geometric_interpolation(cv) == Lagrange{RefTriangle, 1}()
        # Numeric type + quadrature + scalar function
        cv = CellValues(Float32, qr, fun_ip)
        @test Ferrite.shape_value_type(cv) == value_type(Float32)
        @test Ferrite.shape_gradient_type(cv) == grad_type(Float32)
        @test Ferrite.get_geometric_interpolation(cv) == Lagrange{RefTriangle, 1}()
        for geo_ip in (Lagrange{RefTriangle, 2}(), Lagrange{RefTriangle, 2}()^2)
            scalar_ip(ip) = ip isa VectorizedInterpolation ? ip.ip : ip
            # Quadrature + scalar function + geo
            cv = CellValues(qr, fun_ip, geo_ip)
            @test Ferrite.shape_value_type(cv) == value_type(Float64)
            @test Ferrite.shape_gradient_type(cv) == grad_type(Float64)
            @test Ferrite.get_geometric_interpolation(cv) == scalar_ip(geo_ip)
            # Numeric type + quadrature + scalar function + scalar geo
            cv = CellValues(Float32, qr, fun_ip, geo_ip)
            @test Ferrite.shape_value_type(cv) == value_type(Float32)
            @test Ferrite.shape_gradient_type(cv) == grad_type(Float32)
            @test Ferrite.get_geometric_interpolation(cv) == scalar_ip(geo_ip)
        end
    end
end

@testset "show" begin
    cv_quad = CellValues(QuadratureRule{RefQuadrilateral}(2), Lagrange{RefQuadrilateral,2}()^2)
    showstring = sprint(show, MIME"text/plain"(), cv_quad)
    @test startswith(showstring, "CellValues(vdim=2, rdim=2, and sdim=2): 4 quadrature points")
    @test contains(showstring, "Function interpolation: Lagrange{RefQuadrilateral, 2}()^2")

    cv_wedge = CellValues(QuadratureRule{RefPrism}(2), Lagrange{RefPrism,2}())
    showstring = sprint(show, MIME"text/plain"(), cv_wedge)
    @test startswith(showstring, "CellValues(scalar, rdim=3, and sdim=3): 5 quadrature points")
    @test contains(showstring, "Function interpolation: Lagrange{RefPrism, 2}()")

    pv = PointValues(cv_wedge)
    pv_showstring = sprint(show, MIME"text/plain"(), pv)
    @test startswith(pv_showstring, "PointValues containing")
    @test contains(pv_showstring, "Function interpolation: Lagrange{RefPrism, 2}()")
end

@testset "CustomCellValues" begin
    
    @testset "SimpleCellValues" begin
        include(joinpath(@__DIR__, "../docs/src/topics/SimpleCellValues_literate.jl"))
    end
    
    @testset "TestCustomCellValues" begin
    
        struct TestCustomCellValues{CV<:CellValues} <: Ferrite.AbstractValues
            cv::CV
        end
        # Check that the list in devdocs/FEValues.md is true
        # If changes are made that makes the following tests fails,
        # the devdocs should be updated accordingly.
        for op = (:shape_value, :shape_gradient, :getnquadpoints, :getnbasefunctions, :geometric_value, :getngeobasefunctions)
            @eval Ferrite.$op(cv::TestCustomCellValues, args...; kwargs...) = Ferrite.$op(cv.cv, args...; kwargs...)
        end
        ip = Lagrange{RefQuadrilateral,1}()^2
        qr = QuadratureRule{RefQuadrilateral}(2)
        cv = CellValues(qr, ip)
        grid = generate_grid(Quadrilateral, (1,1))
        x = getcoordinates(grid, 1)
        cell = getcells(grid, 1)
        reinit!(cv, cell, x)
        ae = rand(getnbasefunctions(cv))
        q_point = rand(1:getnquadpoints(cv))
        cv_custom = TestCustomCellValues(cv)
        for fun in (function_value, function_gradient, 
                        function_divergence, function_symmetric_gradient, function_curl)
            @test fun(cv_custom, q_point, ae) == fun(cv, q_point, ae)
        end
        @test spatial_coordinate(cv_custom, q_point, x) == spatial_coordinate(cv, q_point, x)
    end
end

end # of testset
