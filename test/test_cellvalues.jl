@testset "CellValues" begin
@testset "ip=$scalar_interpol" for (scalar_interpol, quad_rule) in  (
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
    for func_interpol in (scalar_interpol, VectorizedInterpolation(scalar_interpol)), DiffOrder in 1:2
        (DiffOrder==2 && Ferrite.getorder(func_interpol)==1) && continue #No need to test linear interpolations again
        geom_interpol = scalar_interpol # Tests below assume this
        n_basefunc_base = getnbasefunctions(scalar_interpol)
        update_gradients = true
        update_hessians = (DiffOrder==2 && Ferrite.getorder(func_interpol) > 1)
        cv = CellValues(quad_rule, func_interpol, geom_interpol, Ferrite.ValuesUpdateFlags(func_interpol; update_gradients, update_hessians))
        if update_gradients && !update_hessians # Check correct and type-stable default constructor
            cv_default = @inferred CellValues(quad_rule, func_interpol, geom_interpol)
            @test typeof(cv) === typeof(cv_default)
        end
        rdim = Ferrite.getrefdim(func_interpol)
        n_basefuncs = getnbasefunctions(func_interpol)

        @test getnbasefunctions(cv) == n_basefuncs

        coords, n = valid_coordinates_and_normals(func_interpol)
        reinit!(cv, coords)
        @test_call reinit!(cv, coords)

        # We test this by applying a given deformation gradient on all the nodes.
        # Since this is a linear deformation we should get back the exact values
        # from the interpolation.
        V, G, H = if func_interpol isa Ferrite.ScalarInterpolation
            (rand(), rand(Tensor{1, rdim}), Tensor{2, rdim}((i,j)-> i==j ? rand() : 0.0))
        else
            (rand(Tensor{1, rdim}), rand(Tensor{2, rdim}), Tensor{3, rdim}((i,j,k)-> i==j==k ? rand() : 0.0))
        end

        u_funk(x,V,G,H) = begin
            if update_hessians
                0.5*x⋅H⋅x + G⋅x + V
            else
                G⋅x + V
            end
        end

        _ue = [u_funk(coords[i],V,G,H) for i in 1:n_basefunc_base]
        ue = reinterpret(Float64, _ue)

        for i in 1:getnquadpoints(cv)
            xqp = spatial_coordinate(cv, i, coords)
            Hqp, Gqp, Vqp = Tensors.hessian(x -> u_funk(x,V,G,H), xqp, :all)

            @test function_value(cv, i, ue) ≈ Vqp
            @test function_gradient(cv, i, ue) ≈ Gqp
            if update_hessians
                #Note, the jacobian of the element is constant, which makes the hessian (of the mapping)
                #zero. So this is not the optimal test
                @test Ferrite.function_hessian(cv, i, ue) ≈ Hqp
            end
            if func_interpol isa Ferrite.VectorInterpolation
                @test function_symmetric_gradient(cv, i, ue) ≈ 0.5(Gqp + Gqp')
                @test function_divergence(cv, i, ue) ≈ tr(Gqp)
                rdim == 3 && @test function_curl(cv, i, ue) ≈ Ferrite.curl_from_gradient(Gqp)
            else
                @test function_divergence(cv, i, ue) ≈ sum(Gqp)
            end
        end

        #Test CellValues when input is a ::Vector{<:Vec} (most of which is deprecated)
        ue_vec = [zero(Vec{rdim,Float64}) for i in 1:n_basefunc_base]
        G_vector = rand(Tensor{2, rdim})
        for i in 1:n_basefunc_base
            ue_vec[i] = G_vector ⋅ coords[i]
        end

        for i in 1:getnquadpoints(cv)
            if func_interpol isa Ferrite.ScalarInterpolation
                @test function_gradient(cv, i, ue_vec) ≈ G_vector
            else# func_interpol isa Ferrite.VectorInterpolation
                @test (@test_deprecated function_gradient(cv, i, ue_vec)) ≈ G_vector
                @test (@test_deprecated function_symmetric_gradient(cv, i, ue_vec)) ≈ 0.5(G_vector + G_vector')
                @test (@test_deprecated function_divergence(cv, i, ue_vec)) ≈ tr(G_vector)
                if rdim == 3
                    @test (@test_deprecated function_curl(cv, i, ue_vec)) ≈ Ferrite.curl_from_gradient(G_vector)
                end
                @test_deprecated function_value(cv, i, ue_vec) #no value to test against
            end
        end

        #Check if the non-linear mapping is correct
        #Only do this for one interpolation becuase it relise on AD on "iterative function"
        if scalar_interpol === Lagrange{RefQuadrilateral, 2}()
            coords_nl = [x+rand(x)*0.01 for x in coords] #add some displacement to nodes
            reinit!(cv, coords_nl)

            _ue_nl = [u_funk(coords_nl[i],V,G,H) for i in 1:n_basefunc_base]
            ue_nl = reinterpret(Float64, _ue_nl)

            for i in 1:getnquadpoints(cv)
                xqp = spatial_coordinate(cv, i, coords_nl)
                Hqp, Gqp, Vqp = Tensors.hessian(x -> function_value_from_physical_coord(func_interpol, coords_nl, x, ue_nl), xqp, :all)
                @test function_value(cv, i, ue_nl) ≈ Vqp
                @test function_gradient(cv, i, ue_nl) ≈ Gqp
                if update_hessians
                    @test Ferrite.function_hessian(cv, i, ue_nl) ≈ Hqp
                end
            end
            reinit!(cv, coords) # reinit back to old coords
        end

        # Test of volume
        vol = 0.0
        for i in 1:getnquadpoints(cv)
            vol += getdetJdV(cv,i)
        end
        @test vol ≈ calculate_volume(func_interpol, coords)

        # Test quadrature rule after reinit! with ref. coords
        coords = Ferrite.reference_coordinates(func_interpol)
        reinit!(cv, coords)
        vol = 0.0
        for i in 1:getnquadpoints(cv)
            vol += getdetJdV(cv,i)
        end
        @test vol ≈ reference_volume(func_interpol)

        # Test spatial coordinate (after reinit with ref.coords we should get back the quad_points)
        for (i, qp_x) in pairs(Ferrite.getpoints(quad_rule))
            @test spatial_coordinate(cv, i, coords) ≈ qp_x
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
    qr_f = FacetQuadratureRule{RefTriangle}(1)
    csv = CellValues(qr, ip)
    cvv = CellValues(qr, VectorizedInterpolation(ip))
    csv_embedded = CellValues(qr, ip, ip^3)
    fsv = FacetValues(qr_f, ip)
    fvv = FacetValues(qr_f, VectorizedInterpolation(ip))
    fsv_embedded = FacetValues(qr_f, ip, ip^3)

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

    @testset "CellValues with hessians" begin
        ip = Lagrange{RefQuadrilateral,2}()
        qr = QuadratureRule{RefQuadrilateral}(2)

        cv_vector = CellValues(qr, ip^2, ip^3; update_hessians = true)
        cv_scalar = CellValues(qr, ip, ip^3; update_hessians = true)

        coords = [Vec{3}((x[1], x[2], 0.0)) for x in Ferrite.reference_coordinates(ip)]
        @test_throws ErrorException reinit!(cv_vector, coords) #Not implemented for embedded elements
        @test_throws ErrorException reinit!(cv_scalar, coords)
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
