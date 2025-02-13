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
    end
end
