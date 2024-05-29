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
        (DiffOrder==2 && Ferrite.getorder(func_interpol)==1) && continue #No need to test linear interpolations again
        geom_interpol = scalar_interpol # Tests below assume this
        n_basefunc_base = getnbasefunctions(scalar_interpol)
        update_gradients = true
        update_hessians = (DiffOrder==2 && Ferrite.getorder(func_interpol) > 1)

        fv = FacetValues(quad_rule, func_interpol, geom_interpol; update_gradients, update_hessians)
        if update_gradients && !update_hessians && VERSION ≥ v"1.9" # Check correct and type-stable default constructor
            fv_default = FacetValues(quad_rule, func_interpol, geom_interpol)
            @test typeof(fv) === typeof(fv_default)
        end

        rdim = Ferrite.getrefdim(func_interpol)
        n_basefuncs = getnbasefunctions(func_interpol)

        @test getnbasefunctions(fv) == n_basefuncs

        coords, n = valid_coordinates_and_normals(func_interpol)
        for face in 1:Ferrite.nfacets(func_interpol)
            reinit!(fv, coords, face)
            @test Ferrite.getcurrentfacet(fv) == face

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

            for i in 1:getnquadpoints(fv)
                xqp = spatial_coordinate(fv, i, coords)
                Hqp, Gqp, Vqp = Tensors.hessian(x -> u_funk(x,V,G,H), xqp, :all)

                @test function_value(fv, i, ue) ≈ Vqp
                @test function_gradient(fv, i, ue) ≈ Gqp
                if update_hessians
                    #Note, the jacobian of the element is constant, which makes the hessian (of the mapping)
                    #zero. So this is not the optimal test
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

            #Test CellValues when input is a ::Vector{<:Vec} (most of which is deprecated)
            ue_vec = [zero(Vec{rdim,Float64}) for i in 1:n_basefunc_base]
            G_vector = rand(Tensor{2, rdim})
            for i in 1:n_basefunc_base
                ue_vec[i] = G_vector ⋅ coords[i]
            end

            for i in 1:getnquadpoints(fv)
                if func_interpol isa Ferrite.ScalarInterpolation
                    @test function_gradient(fv, i, ue_vec) ≈ G_vector
                else# func_interpol isa Ferrite.VectorInterpolation
                    @test (@test_deprecated function_gradient(fv, i, ue_vec)) ≈ G_vector
                    @test (@test_deprecated function_symmetric_gradient(fv, i, ue_vec)) ≈ 0.5(G_vector + G_vector')
                    @test (@test_deprecated function_divergence(fv, i, ue_vec)) ≈ tr(G_vector)
                    if rdim == 3
                        @test (@test_deprecated function_curl(fv, i, ue_vec)) ≈ Ferrite.curl_from_gradient(G_vector)
                    end
                    @test_deprecated function_value(fv, i, ue_vec) #no value to test against
                end
            end

            #Check if the non-linear mapping is correct
            #Only do this for one interpolation becuase it relise on AD on "iterative function"
            if scalar_interpol === Lagrange{RefQuadrilateral, 2}()
                coords_nl = [x+rand(x)*0.01 for x in coords] #add some displacement to nodes
                reinit!(fv, coords_nl, face)

                _ue_nl = [u_funk(coords_nl[i],V,G,H) for i in 1:n_basefunc_base]
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
                reinit!(fv, coords, face) # reinit back to old coords
            end


            # Test of volume
            vol = 0.0
            for i in 1:getnquadpoints(fv)
                vol += getdetJdV(fv,i)
            end
            let ip_base = func_interpol isa VectorizedInterpolation ? func_interpol.ip : func_interpol
                x_face = coords[[Ferrite.facetdof_indices(ip_base)[face]...]]
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
            # Test that qr, detJdV, normals, and current_facet are copied as expected.
            # Note that qr remain aliased, as defined by `copy(qr)=qr`, see quadrature.jl.
            # Make it easy to test scalar wrapper equality
            _mock_isequal(a, b) = a == b
            _mock_isequal(a::T, b::T) where {T<:Ferrite.ScalarWrapper} = a[] == b[]
            for fname in (:fqr, :detJdV, :normals, :current_facet)
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
