@testset "ReferenceShape" begin
    for RefShape in (RefLine, RefTriangle, RefQuadrilateral, RefTetrahedron, RefHexahedron, RefPyramid, RefPrism)
        @testset "transform facet points" begin
            # Test both center point and random points on the facet
            ref_coords = Ferrite.reference_coordinates(Lagrange{RefShape, 1}())
            for facet in 1:nfacets(RefShape)
                facet_nodes = collect(Ferrite.reference_facets(RefShape)[facet])
                facet_coords = ref_coords[facet_nodes]
                center_coord = sum(facet_coords) / length(facet_nodes)
                rand_weights = rand(length(facet_nodes))
                rand_coord = sum(rand_weights .* facet_coords) / sum(rand_weights)
                for point in (center_coord, rand_coord)
                    cell_to_facet = Ferrite.element_to_facet_transformation(point, RefShape, facet)
                    facet_to_cell = Ferrite.facet_to_element_transformation(cell_to_facet, RefShape, facet)
                    @test point ≈ facet_to_cell
                end
            end
        end

        @testset "reference_face_edgenrs" begin
            vertices_count = zeros(Int, Ferrite.nvertices(RefShape))
            for (faceedges, facevertices) in zip(
                    Ferrite.reference_face_edgenrs(RefShape),
                    Ferrite.reference_faces(RefShape)
                )
                # For a given face, each vertex should appear
                # exactly twice when counting the containing edges
                fill!(vertices_count, 0)
                for edgenr in faceedges
                    for v in Ferrite.reference_edges(RefShape)[edgenr]
                        vertices_count[v] += 1
                    end
                end
                @test all(i -> vertices_count[i] == 2, facevertices)  # Visited exactly twice
                @test sum(vertices_count) == 2 * length(facevertices) # No other vertices visited
            end
        end
    end
end
