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
        end
    end
end

end # of testset
                                