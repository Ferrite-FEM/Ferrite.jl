@testset "Grid" begin

    for (celltype, nels, elsperel) in ((Line, (5,), 1),
                                       (QuadraticLine, (5,), 1),
                                       (Quadrilateral, (5,5), 1),
                                       (QuadraticQuadrilateral, (5,5), 1),
                                       (Hexahedron, (5,5,5), 1),
                                       (Triangle, (5,5), 2),
                                       (QuadraticTriangle, (5,5), 2),
                                       (Tetrahedron, (5,5,5), 5))

        # Create test grid
        grid = generate_grid(celltype, nels)

        # Test grid utility
        @test getncells(grid) == prod(nels)*elsperel
    end

    # VTK types of different Cells.
    for (dim, n, m) in ((1,2,2), (1,3,2), (2,4,4), (2,9,4), (3,8,6), (2,3,3), (2,6,3), (3,4,4), (2,8,4))
        @test getVTKtype(Cell{dim, n, m}).nodes == n
    end

end # of testset
