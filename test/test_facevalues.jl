# Imports for parallel (isolated) test execution:
using LinearAlgebra
include(joinpath(@__DIR__, "test_utils.jl"))

# Test that all values in the struct are equal,
# but that bits-types are not aliased to eachother.
function test_equal_but_unaliased(a::T, b::T) where {T}
    for fname in fieldnames(T)
        a_val = getfield(a, fname)
        b_val = getfield(b, fname)
        isbits(a_val) || @test a_val !== b_val
        @test a_val == b_val
    end
    return
end

@testset "FacetValues" begin
    for (scalar_interpol, quad_rule) in (
            (Lagrange{RefLine, 1}(), FacetQuadratureRule{RefLine}(2)),
            (Lagrange{RefLine, 2}(), FacetQuadratureRule{RefLine}(2)),
            (Lagrange{RefQuadrilateral, 1}(), FacetQuadratureRule{RefQuadrilateral}(2)),
            (Lagrange{RefQuadrilateral, 2}(), FacetQuadratureRule{RefQuadrilateral}(2)),
            (Lagrange{RefTriangle, 1}(), FacetQuadratureRule{RefTriangle}(2)),
            (Lagrange{RefTriangle, 2}(), FacetQuadratureRule{RefTriangle}(2)),
            (Lagrange{RefHexahedron, 1}(), FacetQuadratureRule{RefHexahedron}(2)),
            (Serendipity{RefQuadrilateral, 2}(), FacetQuadratureRule{RefQuadrilateral}(2)),
            (Lagrange{RefTetrahedron, 1}(), FacetQuadratureRule{RefTetrahedron}(2)),
            (Lagrange{RefTetrahedron, 2}(), FacetQuadratureRule{RefTetrahedron}(2)),
            (Lagrange{RefPyramid, 2}(), FacetQuadratureRule{RefPyramid}(2)),
            (Lagrange{RefPrism, 2}(), FacetQuadratureRule{RefPrism}(2)),
        )
        for func_interpol in (scalar_interpol, VectorizedInterpolation(scalar_interpol)), DiffOrder in 1:2
            (DiffOrder == 2 && Ferrite.getorder(func_interpol) == 1) && continue # No need to test linear interpolations again
            geom_interpol = scalar_interpol # Tests below assume this
            n_basefunc_base = getnbasefunctions(scalar_interpol)
            update_gradients = true
            update_hessians = (DiffOrder == 2 && Ferrite.getorder(func_interpol) > 1)
            fv = FacetValues(quad_rule, func_interpol, geom_interpol; update_gradients, update_hessians)
            if update_gradients && !update_hessians # Check correct and type-stable default constructor
                fv_default = @inferred FacetValues(quad_rule, func_interpol, geom_interpol)
                @test typeof(fv) === typeof(fv_default)
                @inferred FacetValues(quad_rule, func_interpol, geom_interpol; update_hessians = Val(true))
            end

            rdim = Ferrite.getrefdim(func_interpol)
            RefShape = Ferrite.getrefshape(func_interpol)
            n_basefuncs = getnbasefunctions(func_interpol)

            @test getnbasefunctions(fv) == n_basefuncs

            coords, n = valid_coordinates_and_normals(func_interpol)
            for facet in 1:Ferrite.nfacets(func_interpol)
                reinit!(fv, coords, facet)
                @test Ferrite.getcurrentfacet(fv) == facet

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

                for i in 1:getnquadpoints(fv)
                    xqp = spatial_coordinate(fv, i, coords)
                    Hqp, Gqp, Vqp = Tensors.hessian(x -> u_funk(x, V, G, H), xqp, :all)

                    @test function_value(fv, i, ue) ≈ Vqp
                    @test function_gradient(fv, i, ue) ≈ Gqp
                    if update_hessians
                        # Note, the jacobian of the element is constant, which makes the hessian (of the mapping)
                        # zero. So this is not the optimal test
                        @test Ferrite.function_hessian(fv, i, ue) ≈ Hqp
                    end
                    if func_interpol isa Ferrite.VectorInterpolation
                        @test function_symmetric_gradient(fv, i, ue) ≈ 0.5(Gqp + Gqp')
                        @test function_divergence(fv, i, ue) ≈ tr(Gqp)
                        rdim == 3 && @test function_curl(fv, i, ue) ≈ Ferrite.curl_from_gradient(Gqp)
                    else
                        @test function_divergence(fv, i, ue) ≈ sum(Gqp)
                    end
                end

                # Test CellValues when input is a ::Vector{<:Vec} (most of which is deprecated)
                ue_vec = [zero(Vec{rdim, Float64}) for i in 1:n_basefunc_base]
                G_vector = rand(Tensor{2, rdim})
                for i in 1:n_basefunc_base
                    ue_vec[i] = G_vector ⋅ coords[i]
                end

                for i in 1:getnquadpoints(fv)
                    if func_interpol isa Ferrite.ScalarInterpolation
                        @test function_gradient(fv, i, ue_vec) ≈ G_vector
                    else # func_interpol isa Ferrite.VectorInterpolation
                        @test_throws Ferrite.DeprecationError function_gradient(fv, i, ue_vec)
                        @test_throws Ferrite.DeprecationError function_symmetric_gradient(fv, i, ue_vec)
                        @test_throws Ferrite.DeprecationError function_divergence(fv, i, ue_vec)
                        if rdim == 3
                            @test_throws Ferrite.DeprecationError function_curl(fv, i, ue_vec)
                        end
                        @test_throws Ferrite.DeprecationError function_value(fv, i, ue_vec) # no value to test against
                    end
                end

                # Check if the non-linear mapping is correct
                # Only do this for one interpolation because it relies on AD on "iterative function"
                if scalar_interpol === Lagrange{RefQuadrilateral, 2}()
                    coords_nl = [x + rand(x) * 0.01 for x in coords] # add some displacement to nodes
                    reinit!(fv, coords_nl, facet)

                    _ue_nl = [u_funk(coords_nl[i], V, G, H) for i in 1:n_basefunc_base]
                    ue_nl = reinterpret(Float64, _ue_nl)

                    for i in 1:getnquadpoints(fv)
                        xqp = spatial_coordinate(fv, i, coords_nl)
                        Hqp, Gqp, Vqp = Tensors.hessian(x -> function_value_from_physical_coord(func_interpol, coords_nl, x, ue_nl), xqp, :all)
                        @test function_value(fv, i, ue_nl) ≈ Vqp
                        @test function_gradient(fv, i, ue_nl) ≈ Gqp
                        if update_hessians
                            @test Ferrite.function_hessian(fv, i, ue_nl) ≈ Hqp
                        end
                    end
                    reinit!(fv, coords, facet) # reinit back to old coords
                end


                # Test of volume
                vol = 0.0
                for i in 1:getnquadpoints(fv)
                    vol += getdetJdV(fv, i)
                end
                let ip_base = func_interpol isa VectorizedInterpolation ? func_interpol.ip : func_interpol
                    x_face = coords[[Ferrite.facetdof_indices(ip_base)[facet]...]]
                    @test vol ≈ calculate_facet_area(ip_base, x_face, facet)
                end

                # Test quadrature rule after reinit! with ref. coords
                x = Ferrite.reference_coordinates(func_interpol)
                reinit!(fv, x, facet)
                vol = 0.0
                for i in 1:getnquadpoints(fv)
                    vol += getdetJdV(fv, i)
                end
                @test vol ≈ reference_facet_area(RefShape, facet)

                # Test spatial coordinate (after reinit with ref.coords we should get back the quad_points)
                # # TODO: Renable somehow after quad rule is no longer stored in FacetValues
                # for (i, qp_x) in enumerate(getpoints(quad_rule))
                #     @test spatial_coordinate(fv, i, x) ≈ qp_x
                # end

            end

            @testset "copy(::FacetValues)" begin
                fvc = copy(fv)
                @test typeof(fv) == typeof(fvc)

                # Test that all mutable types in FunctionValues and GeometryMapping have been copied
                for i in eachindex(getfield(fv, :fun_values))
                    for (v, vc) in zip(getfield(fv, :fun_values)[i], getfield(fvc, :fun_values)[i])
                        test_equal_but_unaliased(v, vc)
                    end
                    test_equal_but_unaliased(getfield(fv, :geo_mapping)[i], getfield(fvc, :geo_mapping)[i])
                end
                # Test that fqr, detJdV, and normals, are copied as expected.
                # Note that qr remain aliased, as defined by `copy(qr)=qr`, see quadrature.jl.
                for fname in (:fqr, :detJdV, :normals)
                    v = getfield(fv, fname)
                    vc = getfield(fvc, fname)
                    if fname !== :fqr # Test unaliased
                        @test v !== vc
                    end
                    @test v == vc
                end
            end
        end
    end

    @testset "construction errors" begin
        @test_throws ArgumentError FacetValues(FacetQuadratureRule{RefTriangle}(1), Lagrange{RefQuadrilateral, 1}())
        @test_throws ArgumentError FacetValues(FacetQuadratureRule{RefTriangle}(1), Lagrange{RefTriangle, 1}(), Lagrange{RefQuadrilateral, 1}())
        @test_throws ArgumentError FacetValues(FacetQuadratureRule{RefTriangle}(1), Lagrange{RefQuadrilateral, 1}(), Lagrange{RefQuadrilateral, 1}())
        @test_throws ArgumentError FacetValues(FacetQuadratureRule{RefTriangle}(1), Lagrange{RefQuadrilateral, 1}(), Lagrange{RefTriangle, 1}())
    end

    @testset "show" begin
        # Just smoke test to make sure show doesn't error.
        fv = FacetValues(FacetQuadratureRule{RefQuadrilateral}(2), Lagrange{RefQuadrilateral, 2}())
        showstring = sprint(show, MIME"text/plain"(), fv)
        @test startswith(showstring, "FacetValues(scalar, rdim=2, sdim=2): 2 quadrature points per face")
        @test contains(showstring, "Function interpolation: Lagrange{RefQuadrilateral, 2}()")
        @test contains(showstring, "Geometric interpolation: Lagrange{RefQuadrilateral, 1}()^2")
        fv2 = copy(fv)
        push!(Ferrite.getweights(fv2.fqr.facet_rules[1]), 1)
        showstring = sprint(show, MIME"text/plain"(), fv2)
        @test startswith(showstring, "FacetValues(scalar, rdim=2, sdim=2): (3, 2, 2, 2) quadrature points on each face")
    end

    @testset "Multi-field FacetValues" begin
        # Test that multi-field FacetValues give the same output as single-field FacetValues,
        # as that output is thoroughly tested above
        ipu = Lagrange{RefQuadrilateral, 2}()^2
        ipp = Lagrange{RefQuadrilateral, 1}()
        ipT = ipp

        fqr = FacetQuadratureRule{RefQuadrilateral}(2)
        fvu = FacetValues(fqr, ipu)
        fvp = FacetValues(fqr, ipp)
        fmv = FacetValues(fqr, (u = ipu, p = ipp, T = ipT))

        @test fmv isa Ferrite.MultiFieldFacetValues
        @test !(fvu isa Ferrite.MultiFieldFacetValues)
        @test typeof(FacetValues(fqr, (u = ipu, p = ipp, T = ipT))) === typeof(fmv)
        @test propertynames(fmv) == (:u, :p, :T)

        @test fmv.p === fmv.T # Correct aliasing for identical interpolations
        @test fmv.u !== fmv.p
        @test Ferrite.geometric_interpolation(fmv) == Ferrite.geometric_interpolation(fvu)
        @test Ferrite.function_interpolation(fmv.u) == ipu
        @test Ferrite.function_interpolation(fmv.p) == ipp

        # Test type-stable access by hard-coded key (relies on constant propagation)
        _getufield(x) = x.u
        @inferred _getufield(fmv)

        ref_coords = Ferrite.reference_coordinates(Ferrite.geometric_interpolation(fmv))
        x = map(xref -> xref + rand(typeof(xref)) / 5, ref_coords) # Random perturbation
        ue = rand(getnbasefunctions(fmv.u))
        for facet_nr in 1:Ferrite.nfacets(fmv)
            reinit!.((fvu, fvp, fmv), (x,), facet_nr)
            @test fmv.p === fmv.T # Aliasing holds for each facet
            @test getnquadpoints(fmv) == getnquadpoints(fvu)
            for q_point in 1:getnquadpoints(fmv)
                @test getdetJdV(fmv, q_point) ≈ getdetJdV(fvu, q_point)
                @test getnormal(fmv, q_point) ≈ getnormal(fvu, q_point)
                @test spatial_coordinate(fmv, q_point, x) ≈ spatial_coordinate(fvu, q_point, x)
                for i in 1:getnbasefunctions(fmv.u)
                    @test shape_value(fmv.u, q_point, i) ≈ shape_value(fvu, q_point, i)
                    @test shape_gradient(fmv.u, q_point, i) ≈ shape_gradient(fvu, q_point, i)
                end
                for i in 1:getnbasefunctions(fmv.p)
                    @test shape_value(fmv.p, q_point, i) ≈ shape_value(fvp, q_point, i)
                    @test shape_gradient(fmv.p, q_point, i) ≈ shape_gradient(fvp, q_point, i)
                end
                @test function_value(fmv.u, q_point, ue) ≈ function_value(fvu, q_point, ue)
            end
        end

        @testset "copy(::MultiFieldFacetValues)" begin
            fmv_copy = @inferred copy(fmv)
            @test typeof(fmv_copy) === typeof(fmv)
            @test fmv_copy.p === fmv_copy.T # Aliasing preserved
            for i in eachindex(getfield(fmv, :fun_values))
                for (v, vc) in zip(getfield(fmv, :fun_values)[i], getfield(fmv_copy, :fun_values)[i])
                    test_equal_but_unaliased(v, vc)
                end
            end
        end

        @testset "Error paths specific to multi-field FacetValues" begin
            @test_throws ArgumentError getnbasefunctions(fmv)
            @test_throws "getnbasefunctions" getnbasefunctions(fmv)
            for f in (shape_value, shape_gradient, shape_symmetric_gradient, shape_divergence)
                @test_throws ArgumentError f(fmv, 1, 1)
                @test_throws "$(nameof(f))" f(fmv, 1, 1)
            end
            for f in (function_value, function_gradient, function_symmetric_gradient, function_divergence)
                @test_throws ArgumentError f(fmv, 1, ue)
                @test_throws "$(nameof(f))" f(fmv, 1, ue)
            end
        end

        @testset "show" begin
            showstring = sprint(show, MIME"text/plain"(), fmv)
            @test startswith(showstring, "FacetValues with 2 quadrature points per facet")
            @test contains(showstring, "u: Lagrange{RefQuadrilateral, 2}()^2")
            @test contains(showstring, "p: Lagrange{RefQuadrilateral, 1}()")
        end
    end

end # of testset

@testset "EmbeddedLineFacetValues" begin

    for dim in (2, 3)
        for (order, ct) in zip((1, 2), (Line, QuadraticLine))
            grid = generate_grid(ct, (1,), zero(Vec{dim}), 1.2 * ones(Vec{dim}))
            transform_coordinates!(grid, x -> x + basevec(x)[1] * norm(x)^2) # Make geometry nonlinear
            ip = Lagrange{RefLine, order}()
            ip_geo = Lagrange{RefLine, order}()^dim

            dξ = 1.0e-6
            ξ = Vec{1}.([(-1.0,), (-1.0 + dξ,), (1.0 - dξ,), (1.0,)])
            qr = QuadratureRule{RefLine}(fill(NaN, 4), ξ)

            cv = CellValues(qr, ip, ip_geo)
            cell_coords = Ferrite.getcoordinates(grid, 1)
            reinit!(cv, cell_coords)

            fqr = FacetQuadratureRule{RefLine}(1)
            fv = FacetValues(fqr, ip, ip_geo)

            # Facet 1
            reinit!(fv, cell_coords, 1)
            x1 = spatial_coordinate(cv, 1, cell_coords)
            x2 = spatial_coordinate(cv, 2, cell_coords)
            @assert norm(x1 - spatial_coordinate(fv, 1, cell_coords)) < 1.0e-14 # Handle x ≈ 0
            @test getnormal(fv, 1) ≈ normalize(x1 - x2) atol = 1.0e-6
            @test getdetJdV(fv, 1) ≈ 1

            # Facet 2
            reinit!(fv, cell_coords, 2)
            x3 = spatial_coordinate(cv, 3, cell_coords)
            x4 = spatial_coordinate(cv, 4, cell_coords)
            @assert x4 ≈ spatial_coordinate(fv, 1, cell_coords)
            @test getnormal(fv, 1) ≈ normalize(x4 - x3) atol = 1.0e-6
            @test getdetJdV(fv, 1) ≈ 1
        end
    end

    # Test unknown facet error path as its not yet tested in "test_quadrules.jl"
    @test_throws ArgumentError Ferrite.weighted_normal(zero(MixedTensor2{2, 1}), RefLine, 3)
end
