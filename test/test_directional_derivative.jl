using Ferrite, Test, ForwardDiff

@testset "Directional derivatives" begin
    @testset "$shape, order $order" for shape in (RefLine, RefTriangle, RefQuadrilateral, RefTetrahedron), order in 1:2
        ip = Lagrange{shape, order}()
        dim = Ferrite.getrefdim(ip)
        qr = QuadratureRule{shape}(3)
        coords = Ferrite.reference_coordinates(ip)
        direction = Vec{dim}(i -> (-1.0)^i * (i + 1))
        # A quadratic field has a gradient that varies between quadrature points.
        scalar_field(x) = sum(x) + (order == 2 ? sum(abs2, x) : 0.0)
        scalar_derivative(x) = sum(direction) + (order == 2 ? 2 * x ⋅ direction : 0.0)
        # Include off-diagonal terms to distinguish gradient contraction order.
        vector_field(x) = Vec{dim}(i -> i * scalar_field(x) + x[mod1(i + 1, dim)])
        vector_derivative(x) = Vec{dim}(i -> i * scalar_derivative(x) + direction[mod1(i + 1, dim)])
        scalar_dofs = scalar_field.(coords)
        vector_dofs = vector_field.(coords)
        flat_dofs = collect(reinterpret(Float64, vector_dofs))
        cmv = MultiFieldCellValues(qr, (s = ip, v = ip^dim), ip)
        reinit!(cmv, coords)
        for (field_ip, dofs, expected, fv) in (
                (ip, scalar_dofs, scalar_derivative, cmv.s),
                (ip, vector_dofs, vector_derivative, cmv.s),
                (ip^dim, flat_dofs, vector_derivative, cmv.v),
            )
            cv = CellValues(qr, field_ip, ip)
            reinit!(cv, coords)
            for values in (cv, fv), qp in 1:getnquadpoints(cv)
                x = spatial_coordinate(cv, qp, coords)
                result = @inferred function_directional_derivative(values, qp, dofs, direction)
                @test result ≈ expected(x)
                @test function_directional_derivative(values, qp, dofs, zero(direction)) == zero(result)
                padded = vcat(dofs, dofs)
                @test function_directional_derivative(values, qp, padded, direction, (length(dofs) + 1):length(padded)) ≈ result
                # Noncontiguous DOF selections are supported too.
                selected = collect(1:2:length(padded))
                padded[selected] = dofs
                @test function_directional_derivative(values, qp, padded, direction, selected) ≈ result
                for i in 1:getnbasefunctions(values)
                    @test (@inferred shape_directional_derivative(values, qp, i, direction)) ≈ shape_gradient(values, qp, i) ⋅ direction
                end
            end
            facet_values = FacetValues(FacetQuadratureRule{shape}(3), field_ip, ip)
            for facet in 1:Ferrite.nfacets(ip)
                reinit!(facet_values, coords, facet)
                for qp in 1:getnquadpoints(facet_values)
                    x = spatial_coordinate(facet_values, qp, coords)
                    @test function_directional_derivative(facet_values, qp, dofs, direction) ≈ expected(x)
                end
            end
            pv = Ferrite.PointValues(field_ip, ip)
            ξ = Vec{dim}(_ -> 0.2)
            reinit!(pv, coords, ξ)
            @test function_directional_derivative(pv, dofs, direction) ≈ expected(ξ)
            @test function_directional_derivative(pv, vcat(dofs, dofs), direction, 1:length(dofs)) ≈ expected(ξ)
        end
    end

    @testset "Promotion, AD, and curved quadratic geometry" begin
        ip = Lagrange{RefTriangle, 2}()
        coords = [Vec((x[1] + 0.15 * x[1] * x[2], x[2] + 0.1 * x[1]^2)) for x in Ferrite.reference_coordinates(ip)]
        for T in (Float32, Float64), field_ip in (ip, ip^2)
            cv = CellValues(T, QuadratureRule{RefTriangle}(3), field_ip, ip)
            reinit!(cv, Vec{2, T}.(coords))
            dofs = T.(1:getnbasefunctions(cv))
            direction = Vec{2, T}((2, -3))
            as_vector(x::Number) = [x]
            as_vector(x::Vec) = collect(x)
            for qp in 1:getnquadpoints(cv)
                reference(u, d) = function_gradient(cv, qp, u) ⋅ d
                derivative(u, d) = function_directional_derivative(cv, qp, u, d)
                @test (@inferred derivative(dofs, direction)) ≈ reference(dofs, direction)
                @test eltype(derivative(dofs, direction)) == T
                @test eltype(derivative(Float64.(dofs), direction)) == Float64
                @test eltype(derivative(dofs, Vec{2, Float64}(direction))) == Float64
                @test ForwardDiff.jacobian(u -> as_vector(derivative(u, direction)), dofs) ≈ ForwardDiff.jacobian(u -> as_vector(reference(u, direction)), dofs)
                @test ForwardDiff.jacobian(d -> as_vector(derivative(dofs, Vec{2}(d))), collect(direction)) ≈ ForwardDiff.jacobian(d -> as_vector(reference(dofs, Vec{2}(d))), collect(direction))
            end
            @test_throws ArgumentError function_directional_derivative(cv, 1, dofs[2:end], direction)
            @test_throws BoundsError function_directional_derivative(cv, 1, dofs, direction, 2:(length(dofs) + 1))
            @test_throws ErrorException function_directional_derivative(cv, 0, dofs, direction)
            @test_throws ErrorException function_directional_derivative(cv, getnquadpoints(cv) + 1, dofs, direction)
            @test_throws BoundsError shape_directional_derivative(cv, 1, 0, direction)
        end
    end

    @testset "Embedded elements" begin
        ip = Lagrange{RefLine, 2}()
        coords = [Vec((x[1], 2 * x[1], 0.0)) for x in Ferrite.reference_coordinates(ip)]
        direction = Vec((2.0, -3.0, 1.0))
        for field_ip in (ip, ip^2)
            cv = CellValues(QuadratureRule{RefLine}(3), field_ip, ip^3)
            reinit!(cv, coords)
            dofs = Float64.(1:getnbasefunctions(cv))
            for qp in 1:getnquadpoints(cv)
                @test (@inferred function_directional_derivative(cv, qp, dofs, direction)) ≈ function_gradient(cv, qp, dofs) ⋅ direction
            end
        end
    end

    @testset "Interfaces" begin
        grid = generate_grid(Quadrilateral, (2, 1))
        direction = Vec((2.0, -3.0))
        for ip in (Lagrange{RefQuadrilateral, 2}(), Lagrange{RefQuadrilateral, 2}()^2)
            iv = InterfaceValues(FacetValues(FacetQuadratureRule{RefQuadrilateral}(3), ip))
            reinit!(iv, first(InterfaceIterator(grid)))
            dofs = Float64.(1:getnbasefunctions(iv))
            n_here = getnbasefunctions(iv.here)
            padded = vcat([0.0], dofs, [0.0])
            range_here = 2:(n_here + 1)
            range_there = (n_here + 2):(length(dofs) + 1)
            for here in (true, false), qp in 1:getnquadpoints(iv)
                expected = function_gradient(iv, qp, dofs; here = here) ⋅ direction
                @test function_directional_derivative(iv, qp, dofs, direction; here = here) ≈ expected
                @test function_directional_derivative(iv, qp, padded, direction, range_here, range_there; here = here) ≈ expected
                for i in 1:getnbasefunctions(iv)
                    @test shape_directional_derivative(iv, qp, i, direction; here = here) ≈ shape_gradient(iv, qp, i; here = here) ⋅ direction
                end
            end
            @test_throws BoundsError function_directional_derivative(iv, 1, dofs[2:end], direction; here = true)
            @test_throws ArgumentError function_directional_derivative(iv, 1, padded, direction, 2:n_here, range_there; here = true)
        end
    end
end
