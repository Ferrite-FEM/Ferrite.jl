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

    for func_interpol in (scalar_interpol, VectorizedInterpolation(scalar_interpol))
        geom_interpol = scalar_interpol # Tests below assume this
        n_basefunc_base = getnbasefunctions(scalar_interpol)
        fv = if VERSION ≥ v"1.9"
            @inferred FacetValues(quad_rule, func_interpol, geom_interpol)
        else # Type unstable on 1.6, but works at least for 1.9 and later. PR882
            FacetValues(quad_rule, func_interpol, geom_interpol)
        end
        ndim = Ferrite.getdim(func_interpol)
        n_basefuncs = getnbasefunctions(func_interpol)

        @test getnbasefunctions(fv) == n_basefuncs

        xs, n = valid_coordinates_and_normals(func_interpol)
        for face in 1:Ferrite.nfacets(func_interpol)
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
            for i in 1:getnquadpoints(fv)
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
                x_face = xs[[Ferrite.facetdof_indices(ip_base)[face]...]]
                @test vol ≈ calculate_facet_area(ip_base, x_face, face)
            end

            # Test quadrature rule after reinit! with ref. coords
            x = Ferrite.reference_coordinates(func_interpol)
            reinit!(fv, x, face)
            vol = 0.0
            for i in 1:getnquadpoints(fv)
                vol += getdetJdV(fv, i)
            end
            @test vol ≈ reference_face_area(func_interpol, face)

            # Test spatial coordinate (after reinit with ref.coords we should get back the quad_points)
            # TODO: Renable somehow after quad rule is no longer stored in FacetValues
            #for (i, qp_x) in enumerate(getpoints(quad_rule))
            #    @test spatial_coordinate(fv, i, x) ≈ qp_x
            #end

        end

        @testset "copy(::FacetValues)" begin
            fvc = copy(fv)
            @test typeof(fv) == typeof(fvc)

            # Test that all mutable types in FunctionValues and GeometryMapping have been copied
            for key in (:fun_values, :geo_mapping)
                for i in eachindex(getfield(fv, key))
                    val = getfield(fv, key)[i]
                    valc = getfield(fvc, key)[i]
                    for fname in fieldnames(typeof(val))
                        v = getfield(val, fname)
                        vc = getfield(valc, fname)
                        isbits(v) || @test v !== vc
                        @test v == vc
                    end
                end
            end
            # Test that qr, detJdV, normals, and current_face are copied as expected. 
            # Note that qr remain aliased, as defined by `copy(qr)=qr`, see quadrature.jl.
            # Make it easy to test scalar wrapper equality
            _mock_isequal(a, b) = a == b
            _mock_isequal(a::T, b::T) where {T<:Ferrite.ScalarWrapper} = a[] == b[]
            for fname in (:fqr, :detJdV, :normals, :current_face)
                v = getfield(fv, fname)
                vc = getfield(fvc, fname)
                if fname !== :fqr # Test unaliased
                    @test v !== vc
                end
                @test _mock_isequal(v, vc)
            end
        end
    end
end

@testset "show" begin
    # Just smoke test to make sure show doesn't error. 
    fv = FacetValues(FacetQuadratureRule{RefQuadrilateral}(2), Lagrange{RefQuadrilateral,2}())
    showstring = sprint(show, MIME"text/plain"(), fv)
    @test startswith(showstring, "FacetValues(scalar, rdim=2, sdim=2): 2 quadrature points per face")
    @test contains(showstring, "Function interpolation: Lagrange{RefQuadrilateral, 2}()")
    @test contains(showstring, "Geometric interpolation: Lagrange{RefQuadrilateral, 1}()^2")
    fv.fqr.face_rules[1] = deepcopy(fv.fqr.face_rules[1])
    push!(Ferrite.getweights(fv.fqr.face_rules[1]), 1)
    showstring = sprint(show, MIME"text/plain"(), fv)
    @test startswith(showstring, "FacetValues(scalar, rdim=2, sdim=2): (3, 2, 2, 2) quadrature points on each face")
end

end # of testset
