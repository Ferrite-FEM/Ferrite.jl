@testset "InterfaceValues" begin
    getcelltypedim(::Type{<:Ferrite.AbstractCell{shape}}) where {dim, shape <: Ferrite.AbstractRefShape{dim}} = dim
    for (cell_shape, scalar_interpol, quad_rule) in (
                                        (Line, Lagrange{RefLine, 1}(), FaceQuadratureRule{RefLine}(2)),
                                        (QuadraticLine, Lagrange{RefLine, 2}(), FaceQuadratureRule{RefLine}(2)),
                                        (Quadrilateral, Lagrange{RefQuadrilateral, 1}(), FaceQuadratureRule{RefQuadrilateral}(2)),
                                        (QuadraticQuadrilateral, Lagrange{RefQuadrilateral, 2}(), FaceQuadratureRule{RefQuadrilateral}(2)),
                                        (Triangle, Lagrange{RefTriangle, 1}(), FaceQuadratureRule{RefTriangle}(2)),
                                        (QuadraticTriangle, Lagrange{RefTriangle, 2}(), FaceQuadratureRule{RefTriangle}(2)),
                                        (Hexahedron, Lagrange{RefHexahedron, 1}(), FaceQuadratureRule{RefHexahedron}(2)),
                                        # (QuadraticQuadrilateral, Serendipity{RefQuadrilateral, 2}(), FaceQuadratureRule{RefQuadrilateral}(2)),
                                        (Tetrahedron, Lagrange{RefTetrahedron, 1}(), FaceQuadratureRule{RefTetrahedron}(2)),
                                        # (QuadraticTetrahedron, Lagrange{RefTetrahedron, 2}(), FaceQuadratureRule{RefTetrahedron}(2)),
                                       )
    dim = getcelltypedim(cell_shape)
    grid = generate_grid(cell_shape, ntuple(i -> i == 1 ? 2 : 1, dim))
    topology = ExclusiveTopology(grid)
    for func_interpol in (scalar_interpol, VectorizedInterpolation(scalar_interpol))
        geom_interpol = scalar_interpol # Tests below assume this
        n_basefunc_base = 2 * getnbasefunctions(scalar_interpol)
        iv = Ferrite.InterfaceValues(grid, quad_rule, func_interpol, geom_interpol)
        ndim = Ferrite.getdim(func_interpol)
        n_basefuncs = 2 * getnbasefunctions(func_interpol)

        @test getnbasefunctions(iv) == n_basefuncs

        for face in topology.face_skeleton
            neighbors = dim > 1 ? topology.face_neighbor[face[1], face[2]] : topology.vertex_neighbor[face[1], face[2]]
            isempty(neighbors) && continue
            other_face = neighbors[1]
            dim == 1 && (other_face = FaceIndex(other_face[1], other_face[2]))
            cell_coords = get_cell_coordinates(grid, face[1])
            other_cell_coords = get_cell_coordinates(grid, other_face[1])
            ##############
            #   reinit!  #
            ##############
            reinit!(iv.face_values, cell_coords, face[2])
            reinit!(iv.face_values_neighbor, other_cell_coords, other_face[2])
            iv.cell_idx[] = face[1]
            iv.cell_idx_neighbor[] = other_face[1]
            ##############
            # end reinit!#
            ##############
            ioi = Ferrite.InterfaceOrientationInfo(grid, face, other_face)
            nqp = getnquadpoints(iv)
            # Should have same quadrature points
            @test nqp == getnquadpoints(iv.face_values) == getnquadpoints(iv.face_values_neighbor)
            for qp in 1:nqp
                other_qp = Ferrite.get_neighbor_quadp(iv, qp)
                # If correctly synced quadrature points coordinates should match
                @test spatial_coordinate(iv, qp, cell_coords) ≈ spatial_coordinate(iv.face_values, qp, cell_coords) ≈
                spatial_coordinate(iv.face_values_neighbor, other_qp, other_cell_coords)
                for i in 1:getnbasefunctions(iv)
                    shapevalue = shape_value(iv, qp, i)
                    shape_avg = shape_value_average(iv, qp, i)
                    shape_jump = shape_value_jump(iv, qp, i)
                    
                    shapegrad = shape_gradient(iv, qp, i)
                    shapegrad_avg = shape_gradient_average(iv, qp, i)
                    shapegrad_jump = shape_gradient_jump(iv, qp, i)
                    normal = getnormal(iv.face_values_neighbor, other_qp)
                    # Test values (May be removed as it mirrors implementation)
                    if i > getnbasefunctions(iv.face_values)
                        @test shapevalue ≈ shape_value(iv.face_values_neighbor, other_qp, i - getnbasefunctions(iv.face_values))
                        @test shapegrad ≈ shape_gradient(iv.face_values_neighbor, other_qp, i - getnbasefunctions(iv.face_values))
                    else
                        normal = getnormal(iv.face_values, qp)
                        @test shapevalue ≈ shape_value(iv.face_values, qp, i)
                        @test shapegrad ≈ shape_gradient(iv.face_values, qp, i)
                    end

                    @test shape_avg ≈ 0.5 * shapevalue
                    @test shape_jump ≈ (shapevalue isa Number ? normal * shapevalue : normal ⋅ shapevalue)
                    @test shapegrad_avg ≈ 0.5 * shapegrad
                    @test shapegrad_jump ≈ shapegrad ⋅ normal

                    # Test dimensions:
                    # Jump of a [scalar -> vector, vector -> scalar, matrix -> vector]
                    if !isempty(size(shapevalue))
                        @test shape_jump isa Number
                    else
                        @test !(shape_jump isa Number)
                    end

                    if isempty(size(shapegrad)) || length(size(shapegrad)) > 1
                        @test !(shapegrad_jump isa Number)
                    else
                        @test shapegrad_jump isa Number
                    end
                    
                end
            end

            # Test function* copied from facevalues tests
            for here in (true, false)
                u = Vec{ndim, Float64}[zero(Tensor{1,ndim}) for i in 1:n_basefunc_base]
                u_scal = zeros(n_basefunc_base)
                H = rand(Tensor{2, ndim})
                V = rand(Tensor{1, ndim})
                for i in 1:n_basefunc_base
                    xs = i <= n_basefunc_base ÷ 2 ? cell_coords : other_cell_coords
                    j = i <= n_basefunc_base ÷ 2 ? i : i - Ferrite.getngeobasefunctions(iv.face_values)
                    u[i] = H ⋅ xs[j]
                    u_scal[i] = V ⋅ xs[j]
                end
                u_vector = reinterpret(Float64, u)
                for i in 1:length(getnquadpoints(iv))
                    if func_interpol isa Ferrite.ScalarInterpolation
                        @test function_gradient(iv, i, u, here = here) ≈ H
                        @test function_symmetric_gradient(iv, i, u, here = here) ≈ 0.5(H + H')
                        @test function_divergence(iv, i, u_scal, here = here) ≈ sum(V)
                        @test function_divergence(iv, i, u, here = here) ≈ tr(H)
                        @test function_gradient(iv, i, u_scal, here = here) ≈ V
                        ndim == 3 && @test function_curl(iv, i, u, here = here) ≈ Ferrite.curl_from_gradient(H)
                        function_value(iv, i, u, here = here)
                        function_value(iv, i, u_scal, here = here)
                    else # func_interpol isa Ferrite.VectorInterpolation
                        @test function_gradient(iv, i, u_vector, here = here) ≈ H
                        @test (@test_deprecated function_gradient(iv, i, u, here = here)) ≈ H
                        @test function_symmetric_gradient(iv, i, u_vector, here = here) ≈ 0.5(H + H')
                        @test (@test_deprecated function_symmetric_gradient(iv, i, u, here = here)) ≈ 0.5(H + H')
                        @test function_divergence(iv, i, u_vector, here = here) ≈ tr(H)
                        @test (@test_deprecated function_divergence(iv, i, u, here = here)) ≈ tr(H)
                        if ndim == 3
                            @test function_curl(iv, i, u_vector, here = here) ≈ Ferrite.curl_from_gradient(H)
                            @test (@test_deprecated function_curl(iv, i, u, here = here)) ≈ Ferrite.curl_from_gradient(H)
                        end
                        @test function_value(iv, i, u_vector, here = here) ≈ (@test_deprecated function_value(iv, i, u, here = here))
                    end
                end
                # Test of volume
                vol = 0.0
                for i in 1:getnquadpoints(iv)
                    vol += getdetJdV(iv, i; here = here)
                end
                here = true
                let ip_base = func_interpol isa VectorizedInterpolation ? func_interpol.ip : func_interpol
                    xs = here ? cell_coords : other_cell_coords
                    x_face = xs[[Ferrite.facedof_indices(ip_base)[here ? face[2] : other_face[2]]...]]
                    @test vol ≈ calculate_face_area(ip_base, x_face, here ? face[2] : other_face[2])
                end

                # Test quadrature rule after reinit! with ref. coords
                x = Ferrite.reference_coordinates(func_interpol)
                reinit!(here ? iv.face_values : iv.face_values_neighbor, x, here ? face[2] : other_face[2])
                vol = 0.0
                for i in 1:getnquadpoints(iv)
                    vol += getdetJdV(iv, i; here = here)
                end
                @test vol ≈ reference_face_area(func_interpol, here ? face[2] : other_face[2])

                # Test spatial coordinate (after reinit with ref.coords we should get back the quad_points)
                # TODO: Renable somehow after quad rule is no longer stored in FaceValues
                #for (i, qp_x) in enumerate(getpoints(quad_rule))
                #    @test spatial_coordinate(fv, i, x) ≈ qp_x
                #end
            end
        end
        # test copy
        ivc = copy(iv)
        @test typeof(iv) == typeof(ivc)
        for fname in fieldnames(typeof(iv))
            v = getfield(iv, fname)
            v isa Ferrite.ScalarWrapper && continue
            vc = getfield(ivc, fname)
            if hasmethod(pointer, Tuple{typeof(v)})
                @test pointer(v) != pointer(vc)
            end
            v isa FaceValues && for subfname in fieldnames(typeof(v))
                subv = getfield(v, subfname)
                subv isa Ferrite.ScalarWrapper && continue
                subvc = getfield(vc, subfname)
                if hasmethod(pointer, Tuple{typeof(subv)})
                    @test pointer(subv) != pointer(subvc)
                end
                # TODO: check this
                # @test subv == subvc this errors for array fields and works in FaceValues test
            end
            v isa FaceValues && continue
            @test v == vc
        end
    end
end

end # of testset
                                