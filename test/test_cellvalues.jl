@testset "CellValues" begin
    @testset "ip=$scalar_interpol" for (scalar_interpol, quad_rule) in (
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
            (DiffOrder == 2 && Ferrite.getorder(func_interpol) == 1) && continue # No need to test linear interpolations again
            geom_interpol = scalar_interpol # Tests below assume this
            n_basefunc_base = getnbasefunctions(scalar_interpol)
            update_gradients = true
            update_hessians = (DiffOrder == 2 && Ferrite.getorder(func_interpol) > 1)
            cv = CellValues(quad_rule, func_interpol, geom_interpol; update_gradients, update_hessians)
            if update_gradients && !update_hessians # Check correct and type-stable default constructor
                cv_default = @inferred CellValues(quad_rule, func_interpol, geom_interpol)
                @test typeof(cv) === typeof(cv_default)
                @inferred CellValues(quad_rule, func_interpol, geom_interpol; update_gradients = Val(false), update_detJdV = Val(false))
            end
            rdim = Ferrite.getrefdim(func_interpol)
            RefShape = Ferrite.getrefshape(func_interpol)
            n_basefuncs = getnbasefunctions(func_interpol)

            @test getnbasefunctions(cv) == n_basefuncs

            coords, n = valid_coordinates_and_normals(func_interpol)
            reinit!(cv, coords)

            # We test this by applying a given deformation gradient on all the nodes.
            # Since this is a linear deformation we should get back the exact values
            # from the interpolation.
            V, G, H = if func_interpol isa Ferrite.ScalarInterpolation
                (rand(), rand(Tensor{1, rdim}), Tensor{2, rdim}((i, j) -> i == j ? rand() : 0.0))
            else
                (rand(Tensor{1, rdim}), rand(Tensor{2, rdim}), Tensor{3, rdim}((i, j, k) -> i == j == k ? rand() : 0.0))
            end

            function u_funk(x, V, G, H)
                if update_hessians
                    0.5 * x ⋅ H ⋅ x + G ⋅ x + V
                else
                    G ⋅ x + V
                end
            end

            _ue = [u_funk(coords[i], V, G, H) for i in 1:n_basefunc_base]
            ue = reinterpret(Float64, _ue)

            for i in 1:getnquadpoints(cv)
                xqp = spatial_coordinate(cv, i, coords)
                Hqp, Gqp, Vqp = Tensors.hessian(x -> u_funk(x, V, G, H), xqp, :all)

                @test function_value(cv, i, ue) ≈ Vqp
                @test function_gradient(cv, i, ue) ≈ Gqp
                if update_hessians
                    # Note, the jacobian of the element is constant, which makes the hessian (of the mapping)
                    # zero. So this is not the optimal test
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

            # Test CellValues when input is a ::Vector{<:Vec} (most of which is deprecated)
            ue_vec = [zero(Vec{rdim, Float64}) for i in 1:n_basefunc_base]
            G_vector = rand(Tensor{2, rdim})
            for i in 1:n_basefunc_base
                ue_vec[i] = G_vector ⋅ coords[i]
            end

            for i in 1:getnquadpoints(cv)
                if func_interpol isa Ferrite.ScalarInterpolation
                    @test function_gradient(cv, i, ue_vec) ≈ G_vector
                else # func_interpol isa Ferrite.VectorInterpolation
                    @test_throws Ferrite.DeprecationError function_gradient(cv, i, ue_vec)
                    @test_throws Ferrite.DeprecationError function_symmetric_gradient(cv, i, ue_vec)
                    @test_throws Ferrite.DeprecationError function_divergence(cv, i, ue_vec)
                    if rdim == 3
                        @test_throws Ferrite.DeprecationError function_curl(cv, i, ue_vec)
                    end
                    @test_throws Ferrite.DeprecationError function_value(cv, i, ue_vec) # no value to test against
                end
            end

            # Check if the non-linear mapping is correct
            # Only do this for one interpolation becuase it relise on AD on "iterative function"
            if scalar_interpol === Lagrange{RefQuadrilateral, 2}()
                coords_nl = [x + rand(x) * 0.01 for x in coords] # add some displacement to nodes
                reinit!(cv, coords_nl)

                _ue_nl = [u_funk(coords_nl[i], V, G, H) for i in 1:n_basefunc_base]
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
                vol += getdetJdV(cv, i)
            end
            @test vol ≈ calculate_volume(func_interpol, coords)

            # Test quadrature rule after reinit! with ref. coords
            coords = Ferrite.reference_coordinates(func_interpol)
            reinit!(cv, coords)
            vol = 0.0
            for i in 1:getnquadpoints(cv)
                vol += getdetJdV(cv, i)
            end
            @test vol ≈ reference_volume(RefShape)

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

    @testset "GeometryMapping" begin
        grid = generate_grid(Quadrilateral, (1, 1))
        cc = first(CellIterator(grid))

        qr = QuadratureRule{RefQuadrilateral}(1)
        ξ = first(Ferrite.getpoints(qr))
        ip = Lagrange{RefQuadrilateral, 1}()

        cv0 = CellValues(Float64, qr, ip, ip^2; update_detJdV = false, update_gradients = false, update_hessians = false)
        reinit!(cv0, cc)
        @test Ferrite.calculate_mapping(cv0.geo_mapping, 1, cc.coords) == Ferrite.calculate_mapping(ip, ξ, cc.coords, Val(0))

        cv1 = CellValues(Float64, qr, ip, ip^2; update_detJdV = false, update_gradients = true, update_hessians = false)
        reinit!(cv1, cc)
        @test Ferrite.calculate_mapping(cv1.geo_mapping, 1, cc.coords) == Ferrite.calculate_mapping(ip, ξ, cc.coords, Val(1))

        cv2 = CellValues(Float64, qr, ip, ip^2; update_detJdV = false, update_gradients = false, update_hessians = true)
        reinit!(cv2, cc)
        @test Ferrite.calculate_mapping(cv2.geo_mapping, 1, cc.coords) == Ferrite.calculate_mapping(ip, ξ, cc.coords, Val(2))
    end

    @testset "Non-identity mapping gradients" begin
        function test_gradient(ip_fun, cell::CT) where {CT <: Ferrite.AbstractCell}
            ip_geo = geometric_interpolation(CT)
            RefShape = Ferrite.getrefshape(ip_fun)
            x_ref = Ferrite.reference_coordinates(ip_geo)
            # Random cell coordinates, but small pertubation to ensure 1-1 mapping.
            cell_coords = (1 .+ rand(length(x_ref)) / 10) .* x_ref

            function calculate_value_differentiable(ξ::Vec; i = 1)
                T = eltype(ξ)
                qr = QuadratureRule{RefShape}([zero(T)], [ξ])
                gm = Ferrite.GeometryMapping{1}(T, ip_geo, qr)
                mv = Ferrite.calculate_mapping(gm, 1, cell_coords)
                fv0 = Ferrite.FunctionValues{0}(eltype(T), ip_fun, qr, ip_geo^length(ξ))
                Ferrite.apply_mapping!(fv0, 1, mv, cell)
                return shape_value(fv0, 1, i)
            end

            function spatial_coordinate_differentiable(ξ)
                x = zero(ξ)
                for i in 1:getnbasefunctions(ip_geo)
                    x += cell_coords[i] * Ferrite.reference_shape_value(ip_geo, ξ, i)
                end
                return x
            end

            ξ_rand = sample_random_point(RefShape)
            qr = QuadratureRule{RefShape}([NaN], [ξ_rand])
            cv = CellValues(qr, ip_fun, ip_geo)
            reinit!(cv, cell, cell_coords)

            # Test jacobian calculation
            dxdξ = gradient(spatial_coordinate_differentiable, ξ_rand) # Jacobian
            J = Ferrite.getjacobian(Ferrite.calculate_mapping(cv.geo_mapping, 1, cell_coords))
            @test dxdξ ≈ J
            for i in 1:getnbasefunctions(ip_fun)
                dNdξ, N = gradient(z -> calculate_value_differentiable(z; i), ξ_rand, :all)
                # Test that using FunctionValues{0} and FunctionValues{1} gives same mapped value
                @test shape_value(cv, 1, i) ≈ N
                # Test gradient. Hard to differentiate wrt. x, easier to differentiate with ξ,
                # and use the chain-rule to test.
                # Note that N̂(ξ) = reference_shape_value != N(ξ) = shape_value in general.
                #                         dNdx ⋅ dxdξ = dNdξ
                @test shape_gradient(cv, 1, i) ⋅ dxdξ ≈ dNdξ
            end
        end
        test_ips = [
            Lagrange{RefTriangle, 2}(), Lagrange{RefQuadrilateral, 2}(), Lagrange{RefHexahedron, 2}()^3, # Test should also work for identity mapping
            Nedelec{RefTriangle, 1}(), Nedelec{RefTriangle, 2}(),
            RaviartThomas{RefTriangle, 1}(), RaviartThomas{RefTriangle, 2}(), BrezziDouglasMarini{RefTriangle, 1}(),
        ]
        # Makes most sense to test for nonlinear geometries, so we use such cells for testing only
        cell_from_refshape(::Type{RefTriangle}) = QuadraticTriangle((ntuple(identity, 6)))
        cell_from_refshape(::Type{RefQuadrilateral}) = QuadraticQuadrilateral((ntuple(identity, 9)))
        cell_from_refshape(::Type{RefHexahedron}) = QuadraticHexahedron((ntuple(identity, 27)))
        for ip in test_ips
            cell = cell_from_refshape(getrefshape(ip))
            @testset "$ip" begin
                test_gradient(ip, cell)
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
        close!(dh)
        cell = first(CellIterator(dh))
        ip_geo = Lagrange{RefLine, 2}()
        qr = QuadratureRule{RefLine}(deg + 1)
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
        ip = Lagrange{RefTriangle, 1}()
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
            ip_base = Lagrange{RefLine, 1}()
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
                @test zeros(vdim, 2) == function_gradient(csv3, 1, ue)[:, 2:3]
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
            ip_base = Lagrange{RefQuadrilateral, 1}()
            ip = vdim > 0 ? ip_base^vdim : ip_base
            ue = rand(getnbasefunctions(ip))
            qr = QuadratureRule{RefQuadrilateral}(1)
            csv2 = CellValues(qr, ip)
            csv3 = CellValues(qr, ip, ip_base^3)
            reinit!(csv2, [Vec((-1.0, -1.0)), Vec((1.0, -1.0)), Vec((1.0, 1.0)), Vec((-1.0, 1.0))])
            reinit!(csv3, [Vec((-1.0, -1.0, 0.0)), Vec((1.0, -1.0, 0.0)), Vec((1.0, 1.0, 0.0)), Vec((-1.0, 1.0, 0.0))])
            # Test spatial interpolation
            @test spatial_coordinate(csv2, 1, [Vec((-1.0, -1.0)), Vec((1.0, -1.0)), Vec((1.0, 1.0)), Vec((-1.0, 1.0))]) == Vec{2}((0.0, 0.0))
            @test spatial_coordinate(csv3, 1, [Vec((-1.0, -1.0, 0.0)), Vec((1.0, -1.0, 0.0)), Vec((1.0, 1.0, 0.0)), Vec((-1.0, 1.0, 0.0))]) == Vec{3}((0.0, 0.0, 0.0))
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
            ip = Lagrange{RefQuadrilateral, 2}()
            qr = QuadratureRule{RefQuadrilateral}(2)

            cv_vector = CellValues(qr, ip^2, ip^3; update_hessians = true)
            cv_scalar = CellValues(qr, ip, ip^3; update_hessians = true)

            coords = [Vec{3}((2x[1], 0.5x[2], rand())) for x in Ferrite.reference_coordinates(ip)]
            # TODO
            # @test_throws ErrorException reinit!(cv_vector, coords) # Not implemented for embedded elements

            # Test Scalar H
            reinit!(cv_scalar, coords)

            qp = 1
            H = Ferrite.calculate_mapping(cv_scalar.geo_mapping, qp, coords).H

            d2Ndξ2 = cv_scalar.fun_values.d2Ndξ2
            ∂A₁∂₁ = Vec{3}(i -> getindex.(d2Ndξ2[:, qp], 1, 1) ⋅ getindex.(coords, i))
            ∂A₂∂₂ = Vec{3}(i -> getindex.(d2Ndξ2[:, qp], 2, 2) ⋅ getindex.(coords, i))
            ∂A₁∂₂ = Vec{3}(i -> getindex.(d2Ndξ2[:, qp], 1, 2) ⋅ getindex.(coords, i)) # = ∂A₂∂₁
            ∂A₂∂₁ = Vec{3}(i -> getindex.(d2Ndξ2[:, qp], 2, 1) ⋅ getindex.(coords, i))

            @test ∂A₁∂₁ ≈ H[:, 1, 1]
            @test H[:, 2, 2] ≈ ∂A₂∂₂
            @test H[:, 1, 2] ≈ ∂A₁∂₂
            @test H[:, 2, 1] ≈ ∂A₂∂₁

            # Test Mapping
            coords_scaled = [Vec{3}((2x[1], 0.5x[2], 0.0)) for x in Ferrite.reference_coordinates(ip)]
            reinit!(cv_scalar, coords_scaled)

            scale_x = 2.0
            scale_y = 0.5

            coords_ref = [Vec{3}((x[1], x[2], 0.0)) for x in Ferrite.reference_coordinates(ip)]
            cv_ref = CellValues(qr, ip, ip^3; update_hessians = true)
            reinit!(cv_ref, coords_ref)

            @test shape_hessian(cv_scalar, qp, 1)[1, 1] * scale_x^2 ≈ shape_hessian(cv_ref, qp, 1)[1, 1]
            @test shape_hessian(cv_scalar, qp, 1)[2, 2] * scale_y^2 ≈ shape_hessian(cv_ref, qp, 1)[2, 2]
            @test shape_hessian(cv_scalar, qp, 1)[3, 3] ≈ shape_hessian(cv_ref, qp, 1)[3, 3]
            @test shape_hessian(cv_scalar, qp, 1)[1, 2] * scale_x * scale_y ≈ shape_hessian(cv_ref, qp, 1)[1, 2]
            @test shape_hessian(cv_scalar, qp, 1)[2, 1] * scale_x * scale_y ≈ shape_hessian(cv_ref, qp, 1)[2, 1]

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
        end
    end

    @testset "construction errors" begin
        @test_throws ArgumentError CellValues(QuadratureRule{RefTriangle}(1), Lagrange{RefQuadrilateral, 1}())
        @test_throws ArgumentError CellValues(QuadratureRule{RefTriangle}(1), Lagrange{RefTriangle, 1}(), Lagrange{RefQuadrilateral, 1}())
        @test_throws ArgumentError CellValues(QuadratureRule{RefTriangle}(1), Lagrange{RefQuadrilateral, 1}(), Lagrange{RefQuadrilateral, 1}())
        @test_throws ArgumentError CellValues(QuadratureRule{RefTriangle}(1), Lagrange{RefQuadrilateral, 1}(), Lagrange{RefTriangle, 1}())
    end

    @testset "show" begin
        cv_quad = CellValues(QuadratureRule{RefQuadrilateral}(2), Lagrange{RefQuadrilateral, 2}()^2)
        showstring = sprint(show, MIME"text/plain"(), cv_quad)
        @test startswith(showstring, "CellValues(vdim=2, rdim=2, and sdim=2): 4 quadrature points")
        @test contains(showstring, "Function interpolation: Lagrange{RefQuadrilateral, 2}()^2")

        cv_wedge = CellValues(QuadratureRule{RefPrism}(2), Lagrange{RefPrism, 2}())
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

            struct TestCustomCellValues{CV <: CellValues} <: Ferrite.AbstractValues
                cv::CV
            end
            # Check that the list in devdocs/FEValues.md is true
            # If changes are made that makes the following tests fails,
            # the devdocs should be updated accordingly.
            for op in (:shape_value, :shape_gradient, :getnquadpoints, :getnbasefunctions, :geometric_value, :getngeobasefunctions)
                @eval Ferrite.$op(cv::TestCustomCellValues, args...; kwargs...) = Ferrite.$op(cv.cv, args...; kwargs...)
            end
            ip = Lagrange{RefQuadrilateral, 1}()^2
            qr = QuadratureRule{RefQuadrilateral}(2)
            cv = CellValues(qr, ip)
            grid = generate_grid(Quadrilateral, (1, 1))
            x = getcoordinates(grid, 1)
            cell = getcells(grid, 1)
            reinit!(cv, cell, x)
            ae = rand(getnbasefunctions(cv))
            q_point = rand(1:getnquadpoints(cv))
            cv_custom = TestCustomCellValues(cv)
            for fun in (
                    function_value, function_gradient,
                    function_divergence, function_symmetric_gradient, function_curl,
                )
                @test fun(cv_custom, q_point, ae) == fun(cv, q_point, ae)
            end
            @test spatial_coordinate(cv_custom, q_point, x) == spatial_coordinate(cv, q_point, x)
        end
    end

end # of testset
