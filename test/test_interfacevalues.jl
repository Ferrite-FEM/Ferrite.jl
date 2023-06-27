@testset "InterfaceValues" begin
    getcelltypedim(::Type{<:Ferrite.AbstractCell{shape}}) where {dim, shape <: Ferrite.AbstractRefShape{dim}} = dim
    for (cell_shape, scalar_interpol, quad_rule) in (
                                        (Line, DiscontinuousLagrange{RefLine, 1}(), FaceQuadratureRule{RefLine}(2)),
                                        (QuadraticLine, DiscontinuousLagrange{RefLine, 2}(), FaceQuadratureRule{RefLine}(2)),
                                        (Quadrilateral, DiscontinuousLagrange{RefQuadrilateral, 1}(), FaceQuadratureRule{RefQuadrilateral}(2)),
                                        (QuadraticQuadrilateral, DiscontinuousLagrange{RefQuadrilateral, 2}(), FaceQuadratureRule{RefQuadrilateral}(2)),
                                        (Triangle, DiscontinuousLagrange{RefTriangle, 1}(), FaceQuadratureRule{RefTriangle}(2)),
                                        (QuadraticTriangle, DiscontinuousLagrange{RefTriangle, 2}(), FaceQuadratureRule{RefTriangle}(2)),
                                        (Hexahedron, DiscontinuousLagrange{RefHexahedron, 1}(), FaceQuadratureRule{RefHexahedron}(2)),
                                        # (QuadraticQuadrilateral, Serendipity{RefQuadrilateral, 2}(), FaceQuadratureRule{RefQuadrilateral}(2)),
                                        (Tetrahedron, DiscontinuousLagrange{RefTetrahedron, 1}(), FaceQuadratureRule{RefTetrahedron}(2)),
                                        # (QuadraticTetrahedron, Lagrange{RefTetrahedron, 2}(), FaceQuadratureRule{RefTetrahedron}(2)),
                                       )
        dim = getcelltypedim(cell_shape)
        grid = generate_grid(cell_shape, ntuple(i -> i == 1 ? 2 : 1, dim))
        topology = ExclusiveTopology(grid)
        for func_interpol in (scalar_interpol,#= VectorizedInterpolation(scalar_interpol)=#)
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
                iv.face_values_neighbor.current_face[] = other_face[2]
                iv.cell_idx[] = face[1]
                iv.cell_idx_neighbor[] = other_face[1]
                ioi = Ferrite.InterfaceOrientationInfo(grid, face, other_face)
                iv.ioi[] = ioi
                getpoints(iv.face_values_neighbor.qr, other_face[2]) .= Vec{dim}.(Ferrite.transform_interface_point.(Ref(iv), getpoints(quad_rule, face[2])))  
                reinit!(iv.face_values_neighbor, other_cell_coords, other_face[2], true)
                ##############
                # end reinit!#
                ##############
                nqp = getnquadpoints(iv)
                # Should have same quadrature points
                @test nqp == getnquadpoints(iv.face_values) == getnquadpoints(iv.face_values_neighbor)
                for qp in 1:nqp
                    # If correctly synced quadrature points coordinates should match
                    @test spatial_coordinate(iv, qp, cell_coords) ≈ spatial_coordinate(iv.face_values, qp, cell_coords) ≈
                    spatial_coordinate(iv.face_values_neighbor, qp, other_cell_coords)
                    for i in 1:getnbasefunctions(iv)
                        shapevalue = shape_value(iv, qp, i)
                        shape_avg = shape_value_average(iv, qp, i)
                        shape_jump = shape_value_jump(iv, qp, i)
                        shape_jump_no_normal = shape_value_jump(iv, qp, i, false)
                        
                        shapegrad = shape_gradient(iv, qp, i)
                        shapegrad_avg = shape_gradient_average(iv, qp, i)
                        shapegrad_jump = shape_gradient_jump(iv, qp, i)
                        shapegrad_jump_no_normal = shape_gradient_jump(iv, qp, i, false)
                        normal = getnormal(iv, qp, false)
                        # Test values (May be removed as it mirrors implementation)
                        if i > getnbasefunctions(iv.face_values)
                            @test shapevalue ≈ shape_value(iv.face_values_neighbor, qp, i - getnbasefunctions(iv.face_values))
                            @test shapegrad ≈ shape_gradient(iv.face_values_neighbor, qp, i - getnbasefunctions(iv.face_values))
                        else
                            normal = getnormal(iv, qp)
                            @test shapevalue ≈ shape_value(iv.face_values, qp, i)
                            @test shapegrad ≈ shape_gradient(iv.face_values, qp, i)
                        end

                        @test shape_avg ≈ 0.5 * shapevalue
                        @test shape_jump ≈ shapevalue * normal ≈ shape_jump_no_normal * getnormal(iv, qp)
                        @test shapegrad_avg ≈ 0.5 * shapegrad
                        @test shapegrad_jump ≈ shapegrad ⋅ normal ≈ shapegrad_jump_no_normal ⋅ getnormal(iv, qp)

                        # Test dimensions:
                        # Jump of a [scalar -> vector, vector -> scalar]
                        if !isempty(size(shapevalue))
                            @test shape_jump isa Number
                        else
                            @test !(shape_jump isa Number)
                        end

                        if isempty(size(shapegrad))
                            @test !(shapegrad_jump isa Number)
                        else
                            @test shapegrad_jump isa Number
                        end
                    end
                end
                @test_throws ErrorException("Invalid base function $(n_basefuncs + 1). Interface has only $(n_basefuncs) base functions") shape_value_jump(iv, 1, n_basefuncs + 1)
                @test_throws ErrorException("Invalid base function $(n_basefuncs + 1). Interface has only $(n_basefuncs) base functions") shape_gradient_average(iv, 1, n_basefuncs + 1)

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
                    for i in 1:getnquadpoints(iv)
                        @test function_gradient(iv, i, u, here = here) ≈ H
                        @test function_symmetric_gradient(iv, i, u, here = here) ≈ 0.5(H + H')
                        @test function_divergence(iv, i, u_scal, here = here) ≈ sum(V)
                        @test function_divergence(iv, i, u, here = here) ≈ tr(H)
                        @test function_gradient(iv, i, u_scal, here = here) ≈ V
                        ndim == 3 && @test function_curl(iv, i, u, here = here) ≈ Ferrite.curl_from_gradient(H)

                        @test function_value_average(iv, i, u_scal) ≈ function_value(iv, i, u_scal, here = here)
                        @test all(function_value_jump(iv, i, u_scal) .<= 30 * eps(Float64))
                        @test function_gradient_average(iv, i, u_scal) ≈ function_gradient(iv, i, u_scal, here = here)
                        @test function_gradient_jump(iv, i, u_scal) <= 30 * eps(Float64)

                        @test function_value_average(iv, i, u) ≈ function_value(iv, i, u, here = here)
                        @test all(function_value_jump(iv, i, u) .<= 30 * eps(Float64))
                        @test function_gradient_average(iv, i, u) ≈ function_gradient(iv, i, u, here = here)
                        @test all(function_gradient_jump(iv, i, u) .<= 30 * eps(Float64))

                    end
                    # Test of volume
                    vol = 0.0
                    for i in 1:getnquadpoints(iv)
                        vol += getdetJdV(iv, i; here = here)
                    end
                    
                    xs = here ? cell_coords : other_cell_coords
                    x_face = xs[[Ferrite.dirichlet_facedof_indices(scalar_interpol)[here ? face[2] : other_face[2]]...]]
                    @test vol ≈ calculate_face_area(scalar_interpol, x_face, here ? face[2] : other_face[2])
                end
            end
            # Test copy
            ivc = copy(iv)
            @test typeof(iv) == typeof(ivc)
            for fname in fieldnames(typeof(iv))
                v = getfield(iv, fname)
                v isa Ferrite.ScalarWrapper && continue
                vc = getfield(ivc, fname)
                if hasmethod(pointer, Tuple{typeof(v)})
                    @test pointer(v) != pointer(vc)
                end
                v isa FaceValues && continue
                @test v == vc
            end
        end
    end
end # of testset
                                