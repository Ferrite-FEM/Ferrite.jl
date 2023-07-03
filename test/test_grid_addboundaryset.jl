@testset "grid boundary" begin
    function _extractboundary(grid::Ferrite.AbstractGrid{3}, topology::ExclusiveTopology, _ftype::Function, _set::Dict)
        _ftype(grid, topology, "b_bottom", x -> x[3] ≈ -1.0)
        _ftype(grid, topology, "b_top", x -> x[3] ≈ 1.0)
        _ftype(grid, topology, "b_right", x -> x[1] ≈ 1.0)
        _ftype(grid, topology, "b_left", x -> x[1] ≈ -1.0)
        _ftype(grid, topology, "b_front", x -> x[2] ≈ 1.0)
        _ftype(grid, topology, "b_back", x -> x[2] ≈ -1.0)
        return union(_set["b_bottom"], _set["b_top"],
            _set["b_right"], _set["b_left"],
            _set["b_front"], _set["b_back"])
    end
    function _extractboundary(grid::Ferrite.AbstractGrid{2}, topology::ExclusiveTopology, _ftype::Function, _set::Dict)
        _ftype(grid, topology, "b_bottom", x -> x[2] ≈ -1.0)
        _ftype(grid, topology, "b_top", x -> x[2] ≈ 1.0)
        _ftype(grid, topology, "b_right", x -> x[1] ≈ 1.0)
        _ftype(grid, topology, "b_left", x -> x[1] ≈ -1.0)

        return union(_set["b_bottom"], _set["b_top"],
            _set["b_right"], _set["b_left"])
    end
    function extractboundary(grid::Ferrite.AbstractGrid{3}, topology::ExclusiveTopology)
        faces = _extractboundary(grid, topology, addboundaryfaceset!, grid.facesets)
        edges = _extractboundary(grid, topology, addboundaryedgeset!, grid.edgesets)
        vertices = _extractboundary(grid, topology, addboundaryvertexset!, grid.vertexsets)
        return union(faces, edges, vertices)
    end
    function extractboundary(grid::Ferrite.AbstractGrid{2}, topology::ExclusiveTopology)
        faces = _extractboundary(grid, topology, addboundaryfaceset!, grid.facesets)
        vertices = _extractboundary(grid, topology, addboundaryvertexset!, grid.vertexsets)
        return union(faces, vertices)
    end
    function _extractboundarycheck(grid::Ferrite.AbstractGrid{3}, _ftype::Function, _set::Dict)
        _ftype(grid, "b_bottom_c", x -> x[3] ≈ -1.0)
        _ftype(grid, "b_top_c", x -> x[3] ≈ 1.0)
        _ftype(grid, "b_right_c", x -> x[1] ≈ 1.0)
        _ftype(grid, "b_left_c", x -> x[1] ≈ -1.0)
        _ftype(grid, "b_front_c", x -> x[2] ≈ 1.0)
        _ftype(grid, "b_back_c", x -> x[2] ≈ -1.0)
        return union(_set["b_bottom_c"], _set["b_top_c"],
            _set["b_right_c"], _set["b_left_c"],
            _set["b_front_c"], _set["b_back_c"])
    end
    function _extractboundarycheck(grid::Ferrite.AbstractGrid{2}, _ftype::Function, _set::Dict)
        _ftype(grid, "b_bottom_c", x -> x[2] ≈ -1.0)
        _ftype(grid, "b_top_c", x -> x[2] ≈ 1.0)
        _ftype(grid, "b_right_c", x -> x[1] ≈ 1.0)
        _ftype(grid, "b_left_c", x -> x[1] ≈ -1.0)

        return union(_set["b_bottom_c"], _set["b_top_c"],
            _set["b_right_c"], _set["b_left_c"])
    end
    function extractboundarycheck(grid::Ferrite.AbstractGrid{3})
        faces = _extractboundarycheck(grid, addfaceset!, grid.facesets)
        edges = _extractboundarycheck(grid, addedgeset!, grid.edgesets)
        vertices = _extractboundarycheck(grid, addvertexset!, grid.vertexsets)
        return union(faces, edges, vertices)
    end
    function extractboundarycheck(grid::Ferrite.AbstractGrid{2})
        faces = _extractboundarycheck(grid, addfaceset!, grid.facesets)
        vertices = _extractboundarycheck(grid, addvertexset!, grid.vertexsets)
        return union(faces, vertices)
    end
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
            @test Ferrite.getfacevertices(grid, FaceIndex(cell, 1)) == Set([VertexIndex(cell, 1), VertexIndex(cell, 2)])
            @test Ferrite.getfacevertices(grid, FaceIndex(cell, 2)) == Set([VertexIndex(cell, 2), VertexIndex(cell, 3)])
            @test Ferrite.getfacevertices(grid, FaceIndex(cell, 3)) == Set([VertexIndex(cell, 3), VertexIndex(cell, 1)])
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
        addfaceset!(grid, "all", x->true)
        addvertexset!(grid, "all", x->true)
        directions = ["bottom", "top", "left", "right"]
        conditions = [x->x[2]≈-1, x->x[2]≈1, x->x[1]≈-1, x->x[1]≈1]
        for diridx in 1:4
            addfaceset!(grid, directions[diridx]*"_nall", conditions[diridx];all=false)
            addvertexset!(grid, directions[diridx], conditions[diridx];all=true)
            addvertexset!(grid, directions[diridx]*"_nall", conditions[diridx];all=false)
            #faces
            @test Ferrite.filterfaces(grid, grid.facesets["all"], conditions[diridx];all=true) ==
                grid.facesets[directions[diridx]]
            @test Ferrite.filterfaces(grid, grid.facesets["all"], conditions[diridx];all=false) ==
                grid.facesets[directions[diridx]*"_nall"]
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
        addfaceset!(grid, "all", x->true)
        addvertexset!(grid, "all", x->true)
        @test ∪([Ferrite.getfaceinstances(grid, topology,face) for face in Ferrite.faceskeleton(topology, grid)]...)  == grid.facesets["all"]
    end
    @testset "addboundaryset" for cell_type in [
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
        grid = generate_grid(cell_type, ntuple(i->3, Ferrite.getdim(cell_type)))
        topology = ExclusiveTopology(grid)
        @test extractboundary(grid, topology) == extractboundarycheck(grid)
    end
end
