# Test that all values in the struct are equal,
# but that bits-types are not aliased to eachother.
function test_equal_but_unaliased(a::T, b::T) where T
    for fname in fieldnames(T)
        a_val = getfield(a, fname)
        b_val = getfield(b, fname)
        isbits(a_val) || @test a_val !== b_val
        @test a_val == b_val
    end
end
    
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
        cv = @inferred CellValues(quad_rule, func_interpol, geom_interpol)
        ndim = Ferrite.getdim(func_interpol)
        n_basefuncs = getnbasefunctions(func_interpol)

        @test getnbasefunctions(cv) == n_basefuncs

        x, n = valid_coordinates_and_normals(func_interpol)
        reinit!(cv, x)
        @test_call reinit!(cv, x)

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

            test_equal_but_unaliased(cv.fun_values, cvc.fun_values)
            test_equal_but_unaliased(cv.geo_mapping, cvc.geo_mapping)
            # qr remain aliased, as defined by `copy(qr)=qr`, see quadrature.jl.
            @test cvc.qr === cv.qr
            # While detJdV is copied
            @test cvc.detJdV !== cv.detJdV
            @test cvc.detJdV == cv.detJdV
        end
    end
end

@testset "CellMultiValues" begin
    # Here we test that CellMultiValues give the same output as CellValues, 
    # as that output is thoroughly tested above
    ipu = Lagrange{RefQuadrilateral,2}()^2
    ipp = Lagrange{RefQuadrilateral,1}()
    ipT = ipp
    qr = QuadratureRule{RefQuadrilateral}(2)
    cvu = CellValues(qr, ipu)
    cvp = CellValues(qr, ipp)
    cmv = CellMultiValues(qr, (u = ipu, p = ipp, T = ipT))
    cmv_u = CellMultiValues(qr, (u = ipu,)) # Case with a single interpolation 
    cmv3 = CellMultiValues(qr, (u = ipu, T = Lagrange{RefQuadrilateral,2}(), p = ipp)) # Case with 3 unique IPs
    
    @test cmv[:p] === cmv[:T] # Correct aliasing for identical interpolations
    # Correctly inferred geometric interpolation:
    @test Ferrite.geometric_interpolation(cmv) == Ferrite.geometric_interpolation(cvu)
    # Expected function interpolation
    @test Ferrite.function_interpolation(cmv[:u]) == ipu
    @test Ferrite.function_interpolation(cmv[:p]) == ipp

    # Correct number outputs 
    @test getnquadpoints(cmv) == getnquadpoints(cvu)
    @test getnbasefunctions(cmv[:u]) == getnbasefunctions(cvu)
    @test getnbasefunctions(cmv[:p]) == getnbasefunctions(cvp)

    # Reinitialization
    ref_coords = Ferrite.reference_coordinates(Ferrite.geometric_interpolation(cmv))
    x = map(xref -> xref + rand(typeof(xref))/5, ref_coords) # Random pertubation 
    reinit!.((cvu, cvp, cmv, cmv_u, cmv3), (x,))

    @test_call reinit!(cmv, x) # JET testing (e.g. type stability)
    @test_call reinit!(cmv_u, x) # JET testing (e.g. type stability)
    @test_call reinit!(cmv3, x) # JET testing (e.g. type stability)

    # Test type-stable access by hard-coded key (relies on constant propagation)
    _getufield(x) = x[:u]
    @inferred _getufield(cmv)
    @inferred _getufield(cmv3)
    
    # Test output values when used in an element routine
    ue = rand(getnbasefunctions(cmv[:u]) + getnbasefunctions(cmv[:p]))
    dru = 1:getnbasefunctions(cmv[:u])
    drp = (1:getnbasefunctions(cmv[:p])) .+ getnbasefunctions(cmv[:u])
    for q_point in 1:getnquadpoints(cmv)
        for cmv_test in (cmv, cmv_u, cmv3)    
            @test getdetJdV(cvu, q_point) ≈ getdetJdV(cmv, q_point)
            @test spatial_coordinate(cvu, q_point, x) ≈ spatial_coordinate(cmv, q_point, x)
        end

        for (cv, fv, dr) in (
                (cvu, cmv[:u], dru), 
                (cvu, cmv_u[:u], dru),
                (cvu, cmv3[:u], dru),
                (cvp, cmv[:p], drp),
                (cvp, cmv3[:p], drp),
                )
            value = function_value(cv, q_point, ue, dr)
            gradient = function_gradient(cv, q_point, ue, dr)
            @test function_value(fv, q_point, ue[dr]) ≈ value
            @test function_value(fv, q_point, ue, dr) ≈ value
            @test function_gradient(fv, q_point, ue[dr]) ≈ gradient
            @test function_gradient(fv, q_point, ue, dr) ≈ gradient
            if value isa Vec 
                @test function_symmetric_gradient(cmv[:u], q_point, ue, dr) ≈ symmetric(gradient)
                @test function_divergence(cmv[:u], q_point, ue, dr) ≈ tr(gradient)
            end
            for i in 1:getnbasefunctions(fv)
                Ni = shape_value(cv, q_point, i)
                ∇Ni = shape_gradient(cv, q_point, i)
                @test shape_value(fv, q_point, i) ≈ Ni
                @test shape_gradient(fv, q_point, i) ≈ ∇Ni
                if Ni isa Vec
                    @test shape_symmetric_gradient(fv, q_point, i) ≈ symmetric(∇Ni)
                    @test shape_divergence(fv, q_point, i) ≈ tr(∇Ni)
                end
            end
        end
    end
    @testset "copy(::CellMultiValues)" begin
        cmv_copy = @inferred copy(cmv)
        @test cmv_copy isa typeof(cmv)

        # Test that all mutable types in FunctionValues and GeometryMapping have been copied
        for (fv, fvc) in zip(cmv.fun_values_tuple, cmv_copy.fun_values_tuple)
            test_equal_but_unaliased(fv, fvc)
        end
        test_equal_but_unaliased(cmv.geo_mapping, cmv_copy.geo_mapping)

        # Test that aliasing is preserved between equal interpolations 
        @test cmv_copy[:p] === cmv_copy[:T]

        # qr remain aliased, as defined by `copy(qr)=qr`, see quadrature.jl.
        @test cmv_copy.qr === cmv.qr
        # While detJdV is copied
        @test cmv_copy.detJdV !== cmv.detJdV
        @test cmv_copy.detJdV == cmv.detJdV
        
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
    cmv = CellMultiValues(qr, (s = ip, v = VectorizedInterpolation(ip)))
    fsv = FaceValues(qr_f, ip)
    fvv = FaceValues(qr_f, VectorizedInterpolation(ip))
    fsv_embedded = FaceValues(qr_f, ip, ip^3)
    
    x, n = valid_coordinates_and_normals(ip)
    reinit!(csv, x)
    reinit!(cvv, x)
    reinit!(cmv, x)
    reinit!(fsv, x, 1)
    reinit!(fvv, x, 1)
    
    # Wrong number of coordinates
    xx = [x; x]
    @test_throws ArgumentError reinit!(csv, xx)
    @test_throws ArgumentError reinit!(cvv, xx)
    @test_throws ArgumentError reinit!(cmv, xx)
    @test_throws ArgumentError reinit!(fsv, xx, 1)
    @test_throws ArgumentError reinit!(fvv, xx, 1)

    @test_throws ArgumentError spatial_coordinate(csv, qp, xx)
    @test_throws ArgumentError spatial_coordinate(cvv, qp, xx)
    @test_throws ArgumentError spatial_coordinate(cmv, qp, xx)
    @test_throws ArgumentError spatial_coordinate(fsv, qp, xx)
    @test_throws ArgumentError spatial_coordinate(fvv, qp, xx)

    # Wrong dimension of coordinates 
    @test_throws ArgumentError reinit!(csv_embedded, x)
    @test_throws ArgumentError reinit!(fsv_embedded, x, 1)

    # Wrong number of (local) dofs
    # Scalar values
    ue = rand(getnbasefunctions(csv) + 1)
    ue_vec = [rand(Vec{dim}) for _ in 1:(getnbasefunctions(csv) + 1)]
    for test_values in (csv, cmv[:s])
        # Scalar dofs
        @test_throws ArgumentError function_value(test_values, qp, ue)
        @test_throws ArgumentError function_gradient(test_values, qp, ue)
        # Vector dofs
        @test_throws ArgumentError function_value(test_values, qp, ue)
        @test_throws ArgumentError function_gradient(test_values, qp, ue)
        @test_throws ArgumentError function_divergence(test_values, qp, ue)
    end

    # Vector values, scalar dofs
    ue = rand(getnbasefunctions(cvv) + 1)
    for test_values in (cvv, cmv[:v])
        @test_throws ArgumentError function_value(test_values, qp, ue)
        @test_throws ArgumentError function_gradient(test_values, qp, ue)
        @test_throws ArgumentError function_divergence(test_values, qp, ue)
    end
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
        @test_call skip=true reinit!(csv2, [Vec((0.0, 0.0)), Vec((1.0, 0.0))]) # External error in pinv
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
        @test_call skip=true reinit!(csv3, [Vec((0.0, 0.0, 0.0)), Vec((1.0, 0.0, 0.0))]) # External error in pinv
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
        @test_call skip=true reinit!(csv2, [Vec((-1.0,-1.0)), Vec((1.0,-1.0)), Vec((1.0,1.0)), Vec((-1.0,1.0))]) # External error in pinv
        reinit!(csv3, [Vec((-1.0,-1.0,0.0)), Vec((1.0,-1.0,0.0)), Vec((1.0,1.0,0.0)), Vec((-1.0,1.0,0.0))])
        @test_call skip=true reinit!(csv3, [Vec((-1.0,-1.0,0.0)), Vec((1.0,-1.0,0.0)), Vec((1.0,1.0,0.0)), Vec((-1.0,1.0,0.0))]) # External error in pinv
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
        @test Ferrite.geometric_interpolation(cv) == Lagrange{RefTriangle, 1}()
        # Numeric type + quadrature + scalar function
        cv = CellValues(Float32, qr, fun_ip)
        @test Ferrite.shape_value_type(cv) == value_type(Float32)
        @test Ferrite.shape_gradient_type(cv) == grad_type(Float32)
        @test Ferrite.geometric_interpolation(cv) == Lagrange{RefTriangle, 1}()
        for geo_ip in (Lagrange{RefTriangle, 2}(), Lagrange{RefTriangle, 2}()^2)
            scalar_ip(ip) = ip isa VectorizedInterpolation ? ip.ip : ip
            # Quadrature + scalar function + geo
            cv = CellValues(qr, fun_ip, geo_ip)
            @test Ferrite.shape_value_type(cv) == value_type(Float64)
            @test Ferrite.shape_gradient_type(cv) == grad_type(Float64)
            @test Ferrite.geometric_interpolation(cv) == scalar_ip(geo_ip)
            # Numeric type + quadrature + scalar function + scalar geo
            cv = CellValues(Float32, qr, fun_ip, geo_ip)
            @test Ferrite.shape_value_type(cv) == value_type(Float32)
            @test Ferrite.shape_gradient_type(cv) == grad_type(Float32)
            @test Ferrite.geometric_interpolation(cv) == scalar_ip(geo_ip)
        end
        x = Ferrite.reference_coordinates(fun_ip)
        @test_call reinit!(cv, x)
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
