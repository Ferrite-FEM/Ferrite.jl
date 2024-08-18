@testset "grid boundary" begin
    function _extractboundary(grid::Ferrite.AbstractGrid{3}, topology::ExclusiveTopology, _ftype::Function)
        return union((  _ftype(grid, topology, x -> x[3] ≈ -1.0),
                        _ftype(grid, topology, x -> x[3] ≈  1.0),
                        _ftype(grid, topology, x -> x[1] ≈  1.0),
                        _ftype(grid, topology, x -> x[1] ≈ -1.0),
                        _ftype(grid, topology, x -> x[2] ≈  1.0),
                        _ftype(grid, topology, x -> x[2] ≈ -1.0))...)
    end
    function _extractboundary(grid::Ferrite.AbstractGrid{2}, topology::ExclusiveTopology, _ftype::Function)
        return union((  _ftype(grid, topology, x -> x[1] ≈  1.0),
                        _ftype(grid, topology, x -> x[1] ≈ -1.0),
                        _ftype(grid, topology, x -> x[2] ≈  1.0),
                        _ftype(grid, topology, x -> x[2] ≈ -1.0))...)
    end
    function extractboundary(grid::Ferrite.AbstractGrid{3}, topology::ExclusiveTopology)
        facets   = _extractboundary(grid, topology, Ferrite.create_boundaryfacetset)
        faces    = _extractboundary(grid, topology, Ferrite.create_boundaryfaceset)
        edges    = _extractboundary(grid, topology, Ferrite.create_boundaryedgeset)
        vertices = _extractboundary(grid, topology, Ferrite.create_boundaryvertexset)
        return union(facets, faces, edges, vertices)
    end
    function extractboundary(grid::Ferrite.AbstractGrid{2}, topology::ExclusiveTopology)
        facets   = _extractboundary(grid, topology, Ferrite.create_boundaryfacetset)
        edges    = _extractboundary(grid, topology, Ferrite.create_boundaryedgeset)
        vertices = _extractboundary(grid, topology, Ferrite.create_boundaryvertexset)
        return union(facets, edges, vertices)
    end
    function _extractboundarycheck(grid::Ferrite.AbstractGrid{3}, _ftype::Function)
        return union((
        _ftype(grid, x -> x[3] ≈ -1.0),
        _ftype(grid, x -> x[3] ≈  1.0),
        _ftype(grid, x -> x[1] ≈  1.0),
        _ftype(grid, x -> x[1] ≈ -1.0),
        _ftype(grid, x -> x[2] ≈  1.0),
        _ftype(grid, x -> x[2] ≈ -1.0))...)
    end
    function _extractboundarycheck(grid::Ferrite.AbstractGrid{2}, _ftype::Function)
        return union((
        _ftype(grid, x -> x[1] ≈  1.0),
        _ftype(grid, x -> x[1] ≈ -1.0),
        _ftype(grid, x -> x[2] ≈  1.0),
        _ftype(grid, x -> x[2] ≈ -1.0))...)
    end
    function extractboundarycheck(grid::Ferrite.AbstractGrid{3})
        faces    = _extractboundarycheck(grid, Ferrite.create_faceset)
        facets   = _extractboundarycheck(grid, Ferrite.create_facetset)
        edges    = _extractboundarycheck(grid, Ferrite.create_edgeset)
        vertices = _extractboundarycheck(grid, Ferrite.create_vertexset)
        return union(facets, faces, edges, vertices)
    end
    function extractboundarycheck(grid::Ferrite.AbstractGrid{2})
        facets   = _extractboundarycheck(grid, Ferrite.create_facetset)
        edges    = _extractboundarycheck(grid, Ferrite.create_edgeset)
        vertices = _extractboundarycheck(grid, Ferrite.create_vertexset)
        return union(facets, edges, vertices)
    end
    #=
    @testset "getentities" begin
    #                            (8)
    #                   (7) +-----+-----+(9)
    #                       |\    |\    |
    #                       | \ 6 | \ 8 |
    #                       |  \  |  \  |
    #                       | 5 \ | 7 \ |
    #                       |    \|    \|
    #                   (4) +-----+-----+(6)
    #                       |\    |\    |
    #                       | \ 2 | \ 4 |
    #                       |  \  |  \  |
    #                       | 1 \ | 3 \ |
    #                       |    \|    \|
    #                   (1) +-----+-----+(3)
    #                            (2)
        grid = generate_grid(Triangle, (2, 2));
        topology = ExclusiveTopology(grid);
        for cell in 1:getncells(grid)
            @test Ferrite.getedgevertices(grid, EdgeIndex(cell, 1)) == Set([VertexIndex(cell, 1), VertexIndex(cell, 2)])
            @test Ferrite.getedgevertices(grid, EdgeIndex(cell, 2)) == Set([VertexIndex(cell, 2), VertexIndex(cell, 3)])
            @test Ferrite.getedgevertices(grid, EdgeIndex(cell, 3)) == Set([VertexIndex(cell, 3), VertexIndex(cell, 1)])
        end
    # 3D for getfaceedges and getedgevertices
        grid = generate_grid(Tetrahedron, (2, 2, 2));
        topology = ExclusiveTopology(grid);
        for cell in 1:getncells(grid)
            # getfacevertices
            @test Ferrite.getfacevertices(grid, FaceIndex(cell, 1)) == Set([VertexIndex(cell, 1), VertexIndex(cell, 2), VertexIndex(cell, 3)])
            @test Ferrite.getfacevertices(grid, FaceIndex(cell, 2)) == Set([VertexIndex(cell, 1), VertexIndex(cell, 2), VertexIndex(cell, 4)])
            @test Ferrite.getfacevertices(grid, FaceIndex(cell, 3)) == Set([VertexIndex(cell, 2), VertexIndex(cell, 3), VertexIndex(cell, 4)])
            @test Ferrite.getfacevertices(grid, FaceIndex(cell, 4)) == Set([VertexIndex(cell, 1), VertexIndex(cell, 3), VertexIndex(cell, 4)])
            # getfaceedges
            @test Ferrite.getfaceedges(grid, FaceIndex(cell, 1)) == Set([EdgeIndex(cell, 1), EdgeIndex(cell, 2), EdgeIndex(cell, 3)])
            @test Ferrite.getfaceedges(grid, FaceIndex(cell, 2)) == Set([EdgeIndex(cell, 1), EdgeIndex(cell, 4), EdgeIndex(cell, 5)])
            @test Ferrite.getfaceedges(grid, FaceIndex(cell, 3)) == Set([EdgeIndex(cell, 2), EdgeIndex(cell, 5), EdgeIndex(cell, 6)])
            @test Ferrite.getfaceedges(grid, FaceIndex(cell, 4)) == Set([EdgeIndex(cell, 3), EdgeIndex(cell, 4), EdgeIndex(cell, 6)])
            # getedgevertices
            @test Ferrite.getedgevertices(grid, EdgeIndex(cell, 1)) == Set([VertexIndex(cell, 1), VertexIndex(cell, 2)])
            @test Ferrite.getedgevertices(grid, EdgeIndex(cell, 2)) == Set([VertexIndex(cell, 2), VertexIndex(cell, 3)])
            @test Ferrite.getedgevertices(grid, EdgeIndex(cell, 3)) == Set([VertexIndex(cell, 1), VertexIndex(cell, 3)])
            @test Ferrite.getedgevertices(grid, EdgeIndex(cell, 4)) == Set([VertexIndex(cell, 1), VertexIndex(cell, 4)])
            @test Ferrite.getedgevertices(grid, EdgeIndex(cell, 5)) == Set([VertexIndex(cell, 2), VertexIndex(cell, 4)])
            @test Ferrite.getedgevertices(grid, EdgeIndex(cell, 6)) == Set([VertexIndex(cell, 3), VertexIndex(cell, 4)])
        end
    end
    @testset "filter" begin
    #                            (8)
    #                   (7) +-----+-----+(9)
    #                       |\    |\    |
    #                       | \ 6 | \ 8 |
    #                       |  \  |  \  |
    #                       | 5 \ | 7 \ |
    #                       |    \|    \|
    #                   (4) +-----+-----+(6)
    #                       |\    |\    |
    #                       | \ 2 | \ 4 |
    #                       |  \  |  \  |
    #                       | 1 \ | 3 \ |
    #                       |    \|    \|
    #                   (1) +-----+-----+(3)
    #                            (2)
        grid = generate_grid(Triangle, (2, 2));
        topology = ExclusiveTopology(grid);
        addedgeset!(grid, "all", x->true)
        addvertexset!(grid, "all", x->true)
        directions = ["bottom", "top", "left", "right"]
        conditions = [x->x[2]≈-1, x->x[2]≈1, x->x[1]≈-1, x->x[1]≈1]
        for diridx in 1:4
            addedgeset!(grid, directions[diridx]*"_nall", conditions[diridx];all=false)
            addvertexset!(grid, directions[diridx], conditions[diridx];all=true)
            addvertexset!(grid, directions[diridx]*"_nall", conditions[diridx];all=false)
            #faces
            @test Ferrite.filteredges(grid, grid.edgesets["all"], conditions[diridx];all=true) ==
                grid.edgesets[directions[diridx]]
            @test Ferrite.filteredges(grid, grid.edgesets["all"], conditions[diridx];all=false) ==
                grid.edgesets[directions[diridx]*"_nall"]
            #vertices
            @test Ferrite.filtervertices(grid, grid.vertexsets["all"], conditions[diridx];all=true) ==
                grid.vertexsets[directions[diridx]]
            @test Ferrite.filtervertices(grid, grid.vertexsets["all"], conditions[diridx];all=false) ==
                grid.vertexsets[directions[diridx]*"_nall"]
        end
    end
    @testset "getinstances" begin
    #                            (8)
    #                   (7) +-----+-----+(9)
    #                       |\    |\    |
    #                       | \ 6 | \ 8 |
    #                       |  \  |  \  |
    #                       | 5 \ | 7 \ |
    #                       |    \|    \|
    #                   (4) +-----+-----+(6)
    #                       |\    |\    |
    #                       | \ 2 | \ 4 |
    #                       |  \  |  \  |
    #                       | 1 \ | 3 \ |
    #                       |    \|    \|
    #                   (1) +-----+-----+(3)
    #                            (2)
        grid = generate_grid(Triangle, (2, 2));
        topology = ExclusiveTopology(grid);
        addedgeset!(grid, "all", x->true)
        addvertexset!(grid, "all", x->true)
        @test ∪([Ferrite.getedgeinstances(grid, topology,face) for face in Ferrite.facetskeleton(topology, grid)]...)  == grid.edgesets["all"]
    end
    =#
    @testset "addboundaryset ($cell_type)" for cell_type in [
        # Line, # topology construction error
        # QuadraticLine, # topology construction error

        Triangle,
        QuadraticTriangle,

        Quadrilateral,
        QuadraticQuadrilateral,

        Tetrahedron,
        # QuadraticTetrahedron, # grid construction error

        Hexahedron,
        # QuadraticHexahedron, # grid construction error
        # Wedge, # grid construction error (nfacepoints)

        # SerendipityQuadraticQuadrilateral, # grid construction error
        SerendipityQuadraticHexahedron
    ]
        # Grid tests - Regression test for https://github.com/Ferrite-FEM/Ferrite.jl/discussions/565
        grid = generate_grid(cell_type, ntuple(i->3, Ferrite.getrefdim(cell_type)))
        topology = ExclusiveTopology(grid)
        @test extractboundary(grid, topology) == extractboundarycheck(grid)

        filter_function(x) = x[1] > 0
        addboundaryvertexset!(grid, topology, "test_boundary_vertexset", filter_function)
        @test getvertexset(grid, "test_boundary_vertexset") == Ferrite.create_boundaryvertexset(grid, topology, filter_function)
        addboundaryfacetset!(grid, topology, "test_boundary_facetset", filter_function)
        @test getfacetset(grid, "test_boundary_facetset") == Ferrite.create_boundaryfacetset(grid, topology, filter_function)
    end
end
