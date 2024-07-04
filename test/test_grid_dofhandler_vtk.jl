# to test vtk-files
OVERWRITE_CHECKSUMS = false
checksums_file = joinpath(dirname(@__FILE__), "checksums.sha1")
if OVERWRITE_CHECKSUMS
    csio = open(checksums_file, "w")
else
    csio = open(checksums_file, "r")
end

@testset "Grid, DofHandler, vtk" begin
    for (celltype, dim) in ((Line,                   1),
                            (QuadraticLine,          1),
                            (Quadrilateral,          2),
                            (QuadraticQuadrilateral, 2),
                            (Triangle,               2),
                            (QuadraticTriangle,      2),
                            (Hexahedron,             3),
                            (SerendipityQuadraticHexahedron, 3),
                            (Tetrahedron,            3),
                            (Wedge,                  3),
                            (Pyramid,                3))

        # create test grid, do some operations on it and then test
        # the resulting sha1 of the stored vtk file
        # after manually checking the exported vtk
        nels = ntuple(x->5, dim)
        right = Vec{dim, Float64}(ntuple(x->1.5, dim))
        left = -right
        grid = generate_grid(celltype, nels, left, right)

        transform_coordinates!(grid, x-> 2x)

        radius = 2*1.5
        addcellset!(grid, "cell-1", [1,])
        addcellset!(grid, "middle-cells", x -> norm(x) < radius)
        addfacetset!(grid, "middle-facetset", x -> norm(x) < radius)
        addfacetset!(grid, "right-facetset", getfacetset(grid, "right"))
        addnodeset!(grid, "middle-nodes", x -> norm(x) < radius)

        gridfilename = "grid-$(repr(celltype))"
        VTKGridFile(gridfilename, grid) do vtk
            Ferrite.write_cellset(vtk, grid, "cell-1")
            Ferrite.write_cellset(vtk, grid, "middle-cells")
            Ferrite.write_nodeset(vtk, grid, "middle-nodes")
        end

        # test the sha of the file
        sha = bytes2hex(open(SHA.sha1, gridfilename*".vtu"))
        if OVERWRITE_CHECKSUMS
            write(csio, sha, "\n")
        else
            @test sha in split(chomp(readline(csio)))
            rm(gridfilename*".vtu")
        end

        # Create a DofHandler, add some things, write to file and
        # then check the resulting sha
        dofhandler = DofHandler(grid)
        ip = geometric_interpolation(celltype)
        @test ip == geometric_interpolation(getcells(grid, 1)) # Test ::AbstractCell dispatch
        add!(dofhandler, :temperature, ip)
        add!(dofhandler, :displacement, ip^dim)
        close!(dofhandler)
        ch = ConstraintHandler(dofhandler)
        dbc = Dirichlet(:temperature, getfacetset(grid, "right-facetset"), (x,t)->1)
        add!(ch, dbc)
        dbc = Dirichlet(:temperature, getfacetset(grid, "left"), (x,t)->4)
        add!(ch, dbc)
        for d in 1:dim
            dbc = Dirichlet(:displacement, union(getfacetset(grid, "left")), (x,t) -> d, d)
            add!(ch, dbc)
        end
        close!(ch)
        update!(ch, 0.0)
        u = zeros(ndofs(dofhandler))
        apply_analytical!(u, dofhandler, :temperature, x -> 2x[1])
        apply_analytical!(u, dofhandler, :displacement, x -> -2x)
        apply!(u, ch)

        dofhandlerfilename = "dofhandler-$(repr(celltype))"
        VTKGridFile(dofhandlerfilename, grid) do vtk
            Ferrite.write_constraints(vtk, ch)
            write_solution(vtk, dofhandler, u)
        end

        # test the sha of the file
        sha = bytes2hex(open(SHA.sha1, dofhandlerfilename*".vtu"))
        if OVERWRITE_CHECKSUMS
            write(csio, sha, "\n")
        else
            @test sha in split(chomp(readline(csio)))
            rm(dofhandlerfilename*".vtu")
        end

        minv, maxv = Ferrite.bounding_box(grid)
        @test minv ≈ 2left
        @test maxv ≈ 2right

    end

end # of testset

close(csio)

@testset "vtk tensor export" begin
    # open files
    checksums_file_tensors = joinpath(dirname(@__FILE__), "checksums2.sha1")
    if OVERWRITE_CHECKSUMS
        csio = open(checksums_file_tensors, "w")
    else
        csio = open(checksums_file_tensors, "r")
    end

    # 3D grid
    grid = generate_grid(Hexahedron, (1,1,1))

    sym_tensor_data = [SymmetricTensor{2,3}(ntuple(i->i, 6)) for j=1.0:8.0]
    tensor_data = [Tensor{2,3}(ntuple(i->i, 9)) for j=1.0:8.0]
    vector_data = [Vec{3}(ntuple(i->i, 3)) for j=1:8]

    filename_3d = "test_vtk_3d"
    VTKGridFile(filename_3d, grid) do vtk
        write_node_data(vtk, sym_tensor_data, "symmetric tensor")
        write_node_data(vtk, tensor_data, "tensor")
        write_node_data(vtk, vector_data, "vector")
    end

    # 2D grid
    grid = generate_grid(Quadrilateral, (1,1))

    sym_tensor_data = [SymmetricTensor{2,2}(ntuple(i->i, 3)) for j=1.0:4.0]
    tensor_data = [Tensor{2,2}(ntuple(i->i, 4)) for j=1.0:4.0]
    tensor_data_1D = [SymmetricTensor{2,1}(ntuple(i->i, 1)) for j=1.0:4.0]
    vector_data = [Vec{2}(ntuple(i->i, 2)) for j=1:4]

    filename_2d = "test_vtk_2d"
    VTKGridFile(filename_2d, grid) do vtk
        write_node_data(vtk, sym_tensor_data, "symmetric tensor")
        write_node_data(vtk, tensor_data, "tensor")
        write_node_data(vtk, tensor_data_1D, "tensor_1d")
        write_node_data(vtk, vector_data, "vector")
    end

    # test the shas of the files
    files = [filename_3d, filename_2d]
    for filename in files
        sha = bytes2hex(open(SHA.sha1, filename*".vtu"))
        if OVERWRITE_CHECKSUMS
            write(csio, sha, "\n")
        else
            @test sha in split(chomp(readline(csio)))
            rm(filename*".vtu")
        end
    end

    close(csio)
end

@testset "Grid utils" begin

    grid = Ferrite.generate_grid(QuadraticQuadrilateral, (1, 1), Vec((0.,0.)), Vec((1.,1.)))

    addcellset!(grid, "cell_set", [1]);
    node_set = Set(1:getnnodes(grid))
    addnodeset!(grid, "node_set", node_set)

    @test Ferrite.getnodesets(grid) == Dict("node_set" => node_set)

    @test getnodes(grid, [1]) == [getnodes(grid, 1)] # untested

    @test length(getnodes(grid, "node_set")) == 9

    @test collect(get_node_coordinate(getnodes(grid, 5)).data) ≈ [0.5, 0.5]

    @test getcells(grid, "cell_set") == [getcells(grid, 1)]

    # CellIterator on a grid without DofHandler
    grid = generate_grid(Triangle, (4,4))
    n = 0
    ci = CellIterator(grid)
    @test length(ci) == getncells(grid)
    for c in ci
        getcoordinates(c)
        getnodes(c)
        n += cellid(c)
    end
    @test n == div(getncells(grid)*(getncells(grid) + 1), 2)

    # FacetCache
    grid = generate_grid(Triangle, (3,3))
    fc = FacetCache(grid)
    facetindex = first(getfacetset(grid, "left"))
    cell_id, facet_id = facetindex
    reinit!(fc, facetindex)
    # @test Ferrite.faceindex(fc) == faceindex
    @test cellid(fc) == cell_id
    # @test Ferrite.faceid(fc) == face_id
    @test getnodes(fc) == collect(getcells(grid, cell_id).nodes)
    @test getcoordinates(fc) == getcoordinates(grid, cell_id)
    @test length(celldofs(fc)) == 0 # Empty because no DofHandler given

    # FacetIterator, also tests `reinit!(fv::FacetValues, fc::FacetCache)`
    for (dim, celltype) in ((1, Line), (2, Quadrilateral), (3, Hexahedron))
        grid = generate_grid(celltype, ntuple(_ -> 3, dim))
        ip = Lagrange{Ferrite.RefHypercube{dim}, 1}()^dim
        fqr = FacetQuadratureRule{Ferrite.RefHypercube{dim}}(2)
        fv = FacetValues(fqr, ip)
        dh = DofHandler(grid); add!(dh, :u, ip); close!(dh)
        facetset = getfacetset(grid, "right")
        for dh_or_grid in (grid, dh)
            @test first(FacetIterator(dh_or_grid, facetset)) isa FacetCache
            area = 0.0
            for face in FacetIterator(dh_or_grid, facetset)
                reinit!(fv, face)
                for q_point in 1:getnquadpoints(fv)
                    area += getdetJdV(fv, q_point)
                end
            end
            dim == 1 && @test area ≈ 1.0
            dim == 2 && @test area ≈ 2.0
            dim == 3 && @test area ≈ 4.0
        end
    end

    # InterfaceCache
    grid = generate_grid(Quadrilateral, (2,1))
    ic = InterfaceCache(grid)
    reinit!(ic, FaceIndex(1,2), FaceIndex(2,4))
    @test interfacedofs(ic) == Int[] # Empty because no DofHandler given
    ip = DiscontinuousLagrange{RefQuadrilateral, 1}()
    dh = DofHandler(grid); add!(dh, :u, ip); close!(dh)
    ic = InterfaceCache(dh)
    reinit!(ic, FaceIndex(1,2), FaceIndex(2,4))
    @test interfacedofs(ic) == collect(1:8)
    # Mixed Elements
    dim = 2
    nodes = [Node((-1.0, 0.0)), Node((0.0, 0.0)), Node((1.0, 0.0)), Node((-1.0, -1.0)), Node((0.0, 1.0))]
    cells = [
                Quadrilateral((1,2,5,4)),
                Triangle((3,5,2)),
            ]
    grid = Grid(cells, nodes)
    ip1 = DiscontinuousLagrange{RefQuadrilateral, 1}()
    ip2 = DiscontinuousLagrange{RefTriangle, 1}()
    dh = DofHandler(grid);
    sdh1 = SubDofHandler(dh, Set([1])); add!(sdh1, :u, ip1);
    sdh2 = SubDofHandler(dh, Set([2])); add!(sdh2, :u, ip2);
    close!(dh)
    ic = InterfaceCache(dh)
    reinit!(ic, FaceIndex(1,2), FaceIndex(2,3))
    @test interfacedofs(ic) == collect(1:7)
    # Unit test of some utilities
    mixed_grid = Grid([Quadrilateral((1, 2, 3, 4)),Triangle((3, 2, 5))],
                      [Node(coord) for coord in zeros(Vec{2,Float64}, 5)])
    cellset = Set(1:getncells(mixed_grid))
    facetset = Set(FacetIndex(i, 1) for i in 1:getncells(mixed_grid))
    @test_throws ErrorException Ferrite._check_same_celltype(mixed_grid, cellset)
    @test_throws ErrorException Ferrite._check_same_celltype(mixed_grid, facetset)
    std_grid = generate_grid(Quadrilateral, (getncells(mixed_grid),1))
    @test Ferrite._check_same_celltype(std_grid, cellset) === nothing
    @test Ferrite._check_same_celltype(std_grid, facetset) === nothing
end

@testset "Grid sets" begin
    grid = Ferrite.generate_grid(Hexahedron, (1, 1, 1), Vec((0.,0., 0.)), Vec((1.,1.,1.)))

    #Test manual add
    addcellset!(grid, "cell_set", [1]);
    addnodeset!(grid, "node_set", [1])
    addfacetset!(grid, "face_set", [FacetIndex(1,1)])
    addvertexset!(grid, "vert_set", [VertexIndex(1,1)])

    #Test function add
    addfacetset!(grid, "left_face", (x)-> x[1] ≈ 0.0)
    left_lower_edge = Ferrite.create_edgeset(grid, (x)-> x[1] ≈ 0.0 && x[3] ≈ 0.0)
    addvertexset!(grid, "left_corner", (x)-> x[1] ≈ 0.0 && x[2] ≈ 0.0 && x[3] ≈ 0.0)

    @test 1 in Ferrite.getnodeset(grid, "node_set")
    @test FacetIndex(1,5) in getfacetset(grid, "left_face")
    @test EdgeIndex(1,4) in left_lower_edge
    @test VertexIndex(1,1) in getvertexset(grid, "left_corner")

end

@testset "Grid topology" begin
#
#      (1) (2) (3) (4)
#       +---+---+---+
#
    linegrid = generate_grid(Line,(3,))
    linetopo = ExclusiveTopology(linegrid)
    @test linetopo.vertex_vertex_neighbor[1,2] == Ferrite.EntityNeighborhood(VertexIndex(2,1))
    @test getneighborhood(linetopo, linegrid, VertexIndex(1,2)) == [VertexIndex(2,1)]
    @test linetopo.vertex_vertex_neighbor[2,1] == Ferrite.EntityNeighborhood(VertexIndex(1,2))
    @test getneighborhood(linetopo, linegrid, VertexIndex(2,1)) == [VertexIndex(1,2)]
    @test linetopo.vertex_vertex_neighbor[2,2] == Ferrite.EntityNeighborhood(VertexIndex(3,1))
    @test getneighborhood(linetopo, linegrid, VertexIndex(2,2)) == [VertexIndex(3,1)]
    @test linetopo.vertex_vertex_neighbor[3,1] == Ferrite.EntityNeighborhood(VertexIndex(2,2))
    @test getneighborhood(linetopo, linegrid, VertexIndex(3,1)) == [VertexIndex(2,2)]
    @test getneighborhood(linetopo, linegrid, FacetIndex(3,1)) == getneighborhood(linetopo, linegrid, VertexIndex(3,1))

    linefaceskeleton = Ferrite.vertexskeleton(linetopo, linegrid)
    quadlinegrid = generate_grid(QuadraticLine,(3,))
    quadlinetopo = ExclusiveTopology(quadlinegrid)
    quadlinefaceskeleton = Ferrite.facetskeleton(quadlinetopo, quadlinegrid)
    # Test faceskeleton
    @test Set(linefaceskeleton) == Set(quadlinefaceskeleton) == Set([
        VertexIndex(1,1), VertexIndex(1,2), VertexIndex(2,2),VertexIndex(3,2),
    ])

#                           (11)
#                   (10)+-----+-----+(12)
#                       |  5  |  6  |
#                   (7) +-----+-----+(9)
#                       |  3  |  4  |
#                   (4) +-----+-----+(6)
#                       |  1  |  2  |
#                   (1) +-----+-----+(3)
#                            (2)
    quadgrid = generate_grid(Quadrilateral,(2,3))
    topology = ExclusiveTopology(quadgrid)
    faceskeleton = Ferrite.edgeskeleton(topology, quadgrid)
    #test vertex neighbors maps cellid and local vertex id to neighbor id and neighbor local vertex id
    @test topology.vertex_vertex_neighbor[1,3] == Ferrite.EntityNeighborhood(VertexIndex(4,1))
    @test topology.vertex_vertex_neighbor[2,4] == Ferrite.EntityNeighborhood(VertexIndex(3,2))
    @test Set(getneighborhood(topology, quadgrid, VertexIndex(2,4))) == Set([VertexIndex(1,3), VertexIndex(3,2), VertexIndex(4,1)])
    @test topology.vertex_vertex_neighbor[3,3] == Ferrite.EntityNeighborhood(VertexIndex(6,1))
    @test topology.vertex_vertex_neighbor[3,2] == Ferrite.EntityNeighborhood(VertexIndex(2,4))
    @test topology.vertex_vertex_neighbor[4,1] == Ferrite.EntityNeighborhood(VertexIndex(1,3))
    @test topology.vertex_vertex_neighbor[4,4] == Ferrite.EntityNeighborhood(VertexIndex(5,2))
    @test topology.vertex_vertex_neighbor[5,2] == Ferrite.EntityNeighborhood(VertexIndex(4,4))
    @test topology.vertex_vertex_neighbor[6,1] == Ferrite.EntityNeighborhood(VertexIndex(3,3))
    @test isempty(getneighborhood(topology, quadgrid, VertexIndex(2,2)))
    @test length(getneighborhood(topology, quadgrid, VertexIndex(2,4))) == 3
    #test face neighbor maps cellid and local face id to neighbor id and neighbor local face id
    @test topology.edge_edge_neighbor[1,2] == Ferrite.EntityNeighborhood(EdgeIndex(2,4))
    @test topology.edge_edge_neighbor[1,3] == Ferrite.EntityNeighborhood(EdgeIndex(3,1))
    @test topology.edge_edge_neighbor[2,3] == Ferrite.EntityNeighborhood(EdgeIndex(4,1))
    @test topology.edge_edge_neighbor[2,4] == Ferrite.EntityNeighborhood(EdgeIndex(1,2))
    @test topology.edge_edge_neighbor[3,1] == Ferrite.EntityNeighborhood(EdgeIndex(1,3))
    @test topology.edge_edge_neighbor[3,2] == Ferrite.EntityNeighborhood(EdgeIndex(4,4))
    @test topology.edge_edge_neighbor[3,3] == Ferrite.EntityNeighborhood(EdgeIndex(5,1))
    @test topology.edge_edge_neighbor[4,1] == Ferrite.EntityNeighborhood(EdgeIndex(2,3))
    @test topology.edge_edge_neighbor[4,3] == Ferrite.EntityNeighborhood(EdgeIndex(6,1))
    @test topology.edge_edge_neighbor[4,4] == Ferrite.EntityNeighborhood(EdgeIndex(3,2))
    @test topology.edge_edge_neighbor[5,1] == Ferrite.EntityNeighborhood(EdgeIndex(3,3))
    @test topology.edge_edge_neighbor[5,2] == Ferrite.EntityNeighborhood(EdgeIndex(6,4))
    @test topology.edge_edge_neighbor[5,3] == Ferrite.EntityNeighborhood(Ferrite.BoundaryIndex[])
    @test topology.edge_edge_neighbor[5,4] == Ferrite.EntityNeighborhood(Ferrite.BoundaryIndex[])
    @test topology.edge_edge_neighbor[6,1] == Ferrite.EntityNeighborhood(EdgeIndex(4,3))
    @test topology.edge_edge_neighbor[6,2] == Ferrite.EntityNeighborhood(Ferrite.BoundaryIndex[])
    @test topology.edge_edge_neighbor[6,3] == Ferrite.EntityNeighborhood(Ferrite.BoundaryIndex[])
    @test topology.edge_edge_neighbor[6,4] == Ferrite.EntityNeighborhood(EdgeIndex(5,2))

    @test getneighborhood(topology, quadgrid, EdgeIndex(2,4)) == getneighborhood(topology, quadgrid, FacetIndex(2,4))

    quadquadgrid = generate_grid(QuadraticQuadrilateral,(2,3))
    quadtopology = ExclusiveTopology(quadquadgrid)
    quadfaceskeleton = Ferrite.edgeskeleton(quadtopology, quadquadgrid)
    # Test faceskeleton
    @test Set(faceskeleton) == Set(quadfaceskeleton) == Set([
        EdgeIndex(1,1), EdgeIndex(1,2), EdgeIndex(1,3), EdgeIndex(1,4),
        EdgeIndex(2,1), EdgeIndex(2,2), EdgeIndex(2,3),
                        EdgeIndex(3,2), EdgeIndex(3,3), EdgeIndex(3,4),
                        EdgeIndex(4,2), EdgeIndex(4,3),
                        EdgeIndex(5,2), EdgeIndex(5,3), EdgeIndex(5,4),
                        EdgeIndex(6,2), EdgeIndex(6,3),
    ])

#                         (8)
#                (7) +-----+-----+(9)
#                    |  3  |  4  |
#                (4) +-----+-----+(6) bottom view
#                    |  1  |  2  |
#                (1) +-----+-----+(3)
#                         (2)
#                         (15)
#               (16) +-----+-----+(17)
#                    |  3  |  4  |
#               (13) +-----+-----+(15) top view
#                    |  1  |  2  |
#               (10) +-----+-----+(12)
#                        (11)
    hexgrid = generate_grid(Hexahedron,(2,2,1))
    topology = ExclusiveTopology(hexgrid)
    @test topology.edge_edge_neighbor[1,11] == Ferrite.EntityNeighborhood(EdgeIndex(4,9))
    @test Set(getneighborhood(topology,hexgrid,EdgeIndex(1,11),true)) == Set([EdgeIndex(4,9),EdgeIndex(2,12),EdgeIndex(3,10),EdgeIndex(1,11)])
    @test Set(getneighborhood(topology,hexgrid,EdgeIndex(1,11),false)) == Set([EdgeIndex(4,9),EdgeIndex(2,12),EdgeIndex(3,10)])
    @test topology.edge_edge_neighbor[2,12] == Ferrite.EntityNeighborhood(EdgeIndex(3,10))
    @test Set(getneighborhood(topology,hexgrid,EdgeIndex(2,12),true)) == Set([EdgeIndex(3,10),EdgeIndex(1,11),EdgeIndex(4,9),EdgeIndex(2,12)])
    @test Set(getneighborhood(topology,hexgrid,EdgeIndex(2,12),false)) == Set([EdgeIndex(3,10),EdgeIndex(1,11),EdgeIndex(4,9)])
    @test topology.edge_edge_neighbor[3,10] == Ferrite.EntityNeighborhood(EdgeIndex(2,12))
    @test topology.edge_edge_neighbor[4,9] == Ferrite.EntityNeighborhood(EdgeIndex(1,11))
    @test getneighborhood(topology,hexgrid,FaceIndex((1,3))) == [FaceIndex((2,5))]
    @test getneighborhood(topology,hexgrid,FaceIndex((1,4))) == [FaceIndex((3,2))]
    @test getneighborhood(topology,hexgrid,FaceIndex((2,4))) == [FaceIndex((4,2))]
    @test getneighborhood(topology,hexgrid,FaceIndex((2,5))) == [FaceIndex((1,3))]
    @test getneighborhood(topology,hexgrid,FaceIndex((3,2))) == [FaceIndex((1,4))]
    @test getneighborhood(topology,hexgrid,FaceIndex((3,3))) == [FaceIndex((4,5))]
    @test getneighborhood(topology,hexgrid,FaceIndex((4,2))) == [FaceIndex((2,4))]
    @test getneighborhood(topology,hexgrid,FaceIndex((4,5))) == [FaceIndex((3,3))]

    @test getneighborhood(topology, hexgrid, FaceIndex(2,4)) == getneighborhood(topology, hexgrid, FacetIndex(2,4))

    # regression for https://github.com/Ferrite-FEM/Ferrite.jl/issues/518
    serendipitygrid = generate_grid(SerendipityQuadraticHexahedron,(2,2,1))
    stopology = ExclusiveTopology(serendipitygrid)
    @test all(stopology.face_face_neighbor .== topology.face_face_neighbor)
    @test all(stopology.vertex_vertex_neighbor .== topology.vertex_vertex_neighbor)
    # Test faceskeleton
    faceskeleton = Ferrite.faceskeleton(topology, hexgrid)
    sfaceskeleton = Ferrite.faceskeleton(stopology, serendipitygrid)
    @test Set(faceskeleton) == Set(sfaceskeleton) == Set([
        FaceIndex(1,1), FaceIndex(1,2), FaceIndex(1,3), FaceIndex(1,4), FaceIndex(1,5), FaceIndex(1,6),
        FaceIndex(2,1), FaceIndex(2,2), FaceIndex(2,3), FaceIndex(2,4),                 FaceIndex(2,6),
        FaceIndex(3,1),                 FaceIndex(3,3), FaceIndex(3,4), FaceIndex(3,5), FaceIndex(3,6),
        FaceIndex(4,1),                 FaceIndex(4,3), FaceIndex(4,4),                 FaceIndex(4,6),
    ])

#                   +-----+-----+
#                   |\  6 |\  8 |
#                   |  \  |  \  |
#                   |  5 \| 7  \|
#                   +-----+-----+
#                   |\  2 |\  4 |
#                   |  \  |  \  |
#                   |  1 \| 3  \|
#                   +-----+-----+
# test for multiple vertex_neighbors as in e.g. ele 3, local vertex 3 (middle node)
    trigrid = generate_grid(Triangle,(2,2))
    topology = ExclusiveTopology(trigrid)
    @test topology.vertex_vertex_neighbor[3,3] == Ferrite.EntityNeighborhood([VertexIndex(5,2),VertexIndex(6,1),VertexIndex(7,1)])

    quadtrigrid = generate_grid(QuadraticTriangle,(2,2))
    quadtopology = ExclusiveTopology(trigrid)
    # add more regression for https://github.com/Ferrite-FEM/Ferrite.jl/issues/518
    @test all(quadtopology.face_face_neighbor .== topology.face_face_neighbor)
    @test all(quadtopology.vertex_vertex_neighbor .== topology.vertex_vertex_neighbor)
    # Test faceskeleton
    trifaceskeleton = Ferrite.edgeskeleton(topology, trigrid)
    quadtrifaceskeleton = Ferrite.edgeskeleton(quadtopology, quadtrigrid)
    @test Set(trifaceskeleton) == Set(quadtrifaceskeleton) == Set([
        EdgeIndex(1,1), EdgeIndex(1,2), EdgeIndex(1,3),
        EdgeIndex(2,1), EdgeIndex(2,2),
        EdgeIndex(3,1), EdgeIndex(3,2),
        EdgeIndex(4,1), EdgeIndex(4,2),
                        EdgeIndex(5,2), EdgeIndex(5,3),
        EdgeIndex(6,1), EdgeIndex(6,2),
                        EdgeIndex(7,2),
        EdgeIndex(8,1), EdgeIndex(8,2),
    ])
# Test tetrahedron faceskeleton
    tetgrid = generate_grid(Tetrahedron, (1,1,1))
    topology = ExclusiveTopology(tetgrid)
    tetfaceskeleton = Ferrite.faceskeleton(topology, tetgrid)
    @test Set(tetfaceskeleton) == Set([
        FaceIndex(1,1), FaceIndex(1,2), FaceIndex(1,3), FaceIndex(1,4),
        FaceIndex(2,1), FaceIndex(2,2), FaceIndex(2,3),
        FaceIndex(3,1), FaceIndex(3,2), FaceIndex(3,3),
        FaceIndex(4,1), FaceIndex(4,2), FaceIndex(4,3),
        FaceIndex(5,1),                 FaceIndex(5,3), FaceIndex(5,4),
        FaceIndex(6,1),                 FaceIndex(6,3),
    ])
# test mixed grid
    cells = [
        Hexahedron((1, 2, 3, 4, 5, 6, 7, 8)),
        Hexahedron((11, 13, 14, 12, 15, 16, 17, 18)),
        Quadrilateral((2, 9, 10, 3)),
        Quadrilateral((9, 11, 12, 10)),
        ]
    nodes = [Node(coord) for coord in zeros(Vec{3,Float64}, 18)]
    grid = Grid(cells, nodes)
    topology = ExclusiveTopology(grid)

    @test_throws ArgumentError Ferrite.facetskeleton(topology, grid)
    # @test topology.face_face_neighbor[3,4] == Ferrite.EntityNeighborhood(EdgeIndex(1,2))
    # @test topology.edge_edge_neighbor[1,2] == Ferrite.EntityNeighborhood(FaceIndex(3,4))
    # # regression that it doesn't error for boundary faces, see https://github.com/Ferrite-FEM/Ferrite.jl/issues/518
    # @test topology.face_face_neighbor[1,6] == topology.face_face_neighbor[1,1] == zero(Ferrite.EntityNeighborhood{FaceIndex})
    # @test topology.edge_edge_neighbor[1,1] == topology.edge_edge_neighbor[1,3] == zero(Ferrite.EntityNeighborhood{FaceIndex})
    # @test topology.face_face_neighbor[3,1] == topology.face_face_neighbor[3,3] == zero(Ferrite.EntityNeighborhood{FaceIndex})
    # @test topology.face_face_neighbor[4,1] == topology.face_face_neighbor[4,3] == zero(Ferrite.EntityNeighborhood{FaceIndex})

    cells = [
        Quadrilateral((1, 2, 6, 5)),
        Quadrilateral((3, 4, 8, 7)),
        Line((2, 3)),
        Line((6, 7)),
        ]
    nodes = [Node(coord) for coord in zeros(Vec{2,Float64}, 18)]
    grid = Grid(cells, nodes)
    topology = ExclusiveTopology(grid)
    @test_throws ArgumentError Ferrite.facetskeleton(topology, grid)
    @test_throws ArgumentError getneighborhood(topology, grid, FacetIndex(1,1))
    @test_throws ArgumentError Ferrite.get_facet_facet_neighborhood(topology, grid)

#
#                   +-----+-----+-----+
#                   |  7  |  8  |  9  |
#                   +-----+-----+-----+
#                   |  4  |  5  |  6  |
#                   +-----+-----+-----+
#                   |  1  |  2  |  3  |
#                   +-----+-----+-----+
# test application: form level 1 neighborhood patches of elements
    quadgrid = generate_grid(Quadrilateral,(3,3))
    topology = ExclusiveTopology(quadgrid)
    patches = Vector{Int}[Ferrite.getneighborhood(topology, quadgrid, CellIndex(i)) for i in 1:getncells(quadgrid)]

    @test issubset([4,5,2], patches[1]) # neighbor elements of element 1 are 4 5 and 2
    @test issubset([1,4,5,6,3], patches[2])
    @test issubset([2,5,6], patches[3])
    @test issubset([7,8,5,2,1], patches[4])
    @test issubset([1,2,3,4,6,7,8,9], patches[5])
    @test issubset([3,2,5,8,9], patches[6])
    @test issubset([4,5,8], patches[7])
    @test issubset([7,4,5,6,9], patches[8])
    @test issubset([8,5,6], patches[9])

# test star stencils
    stars = Ferrite.vertex_star_stencils(topology, quadgrid)
    @test Set(Ferrite.getstencil(stars, quadgrid, VertexIndex(1,1))) == Set([VertexIndex(1,2), VertexIndex(1,4), VertexIndex(1,1)])
    @test Set(Ferrite.getstencil(stars, quadgrid, VertexIndex(2,1))) == Set([VertexIndex(1,1), VertexIndex(1,3), VertexIndex(2,2), VertexIndex(2,4), VertexIndex(1,2), VertexIndex(2,1)])
    @test Set(Ferrite.getstencil(stars, quadgrid, VertexIndex(5,4))) == Set([VertexIndex(4,2), VertexIndex(4,4), VertexIndex(5,1), VertexIndex(5,3), VertexIndex(7,1), VertexIndex(7,3), VertexIndex(8,2), VertexIndex(8,4), VertexIndex(4,3), VertexIndex(5,4), VertexIndex(7,2), VertexIndex(8,1)])
    @test Set(Ferrite.toglobal(quadgrid, Ferrite.getstencil(stars, quadgrid, VertexIndex(1,1)))) == Set([1,2,5])
    @test Set(Ferrite.toglobal(quadgrid, Ferrite.getstencil(stars, quadgrid, VertexIndex(2,1)))) == Set([2,1,6,3])
    @test Set(Ferrite.toglobal(quadgrid, Ferrite.getstencil(stars, quadgrid, VertexIndex(5,4)))) == Set([10,6,9,11,14])

    face_skeleton = Ferrite.edgeskeleton(topology, quadgrid)
    @test Set(face_skeleton) == Set([EdgeIndex(1,1),EdgeIndex(1,2),EdgeIndex(1,3),EdgeIndex(1,4),
        EdgeIndex(2,1),EdgeIndex(2,2),EdgeIndex(2,3),
        EdgeIndex(3,1),EdgeIndex(3,2),EdgeIndex(3,3),
        EdgeIndex(4,2),EdgeIndex(4,3),EdgeIndex(4,4),
        EdgeIndex(5,2),EdgeIndex(5,3),EdgeIndex(6,2),EdgeIndex(6,3),
        EdgeIndex(7,2),EdgeIndex(7,3),EdgeIndex(7,4),
        EdgeIndex(8,2),EdgeIndex(8,3),EdgeIndex(9,2),EdgeIndex(9,3)])
    @test length(face_skeleton) == 4*3 + 3*4

    quadratic_quadgrid = generate_grid(QuadraticQuadrilateral,(3,3))
    quadgrid_topology = ExclusiveTopology(quadratic_quadgrid)
    #quadface_skeleton = Ferrite.faceskeleton(topology, quadgrid)
    #@test quadface_skeleton == face_skeleton

    # add more regression for https://github.com/Ferrite-FEM/Ferrite.jl/issues/518
    @test all(quadgrid_topology.edge_edge_neighbor .== topology.edge_edge_neighbor)
    @test all(quadgrid_topology.vertex_vertex_neighbor .== topology.vertex_vertex_neighbor)
    quadratic_patches = Vector{Int}[Ferrite.getneighborhood(quadgrid_topology, quadratic_quadgrid, CellIndex(i)) for i in 1:getncells(quadratic_quadgrid)]
    @test all(patches .== quadratic_patches)

#
#                   +-----+-----+-----+
#                   |  7  |  8  |  9  |
#                   +-----+-----+-----+
#                   |  4  |  5  |  6  |
#                   +-----+-----+-----+
#                   |  1  |  2  |  3  |
#                   +-----+-----+-----+
# test application: integrate jump across element boundary 5
    ip = DiscontinuousLagrange{RefQuadrilateral, 1}()^2
    qr_facet = FacetQuadratureRule{RefQuadrilateral}(2)
    iv = InterfaceValues(qr_facet, ip)
    dh = DofHandler(quadgrid)
    add!(dh, :u, ip)
    close!(dh)
    u = [5.0 for _ in 1:4*9*2]
    u[4*4*2+1 : 5*4*2] .= 3.0
    jump_int = 0.
    jump_abs = 0.
    # Test interface Iterator
    for ic in InterfaceIterator(dh)
        any(cellid.([ic.a, ic.b]) .== 5) || continue
        reinit!(iv, ic)
        for q_point in 1:getnquadpoints(iv)
            dΩ = getdetJdV(iv, q_point)
            normal_a = getnormal(iv, q_point)
            u_interface = u[interfacedofs(ic)]
            jump_int += function_value_jump(iv, q_point, u_interface) ⋅ normal_a * dΩ
            jump_abs += abs(function_value_jump(iv, q_point, u_interface) ⋅ normal_a) * dΩ
        end
    end
    @test isapprox(jump_abs, 2/3*2*4,atol=1e-6) # 2*4*0.66666, jump is always 2, 4 sides, length =0.66
    @test isapprox(jump_int, 0.0, atol=1e-6)
end

@testset "grid coloring" begin
    function test_coloring(grid, cellset=1:getncells(grid))
        for alg in (ColoringAlgorithm.Greedy, ColoringAlgorithm.WorkStream)
            color_vectors = create_coloring(grid, cellset; alg=alg)
            @test sum(length, color_vectors, init=0) == length(cellset)
            @test union!(Set{Int}(), color_vectors...)  == Set(cellset)
            conn = Ferrite.create_incidence_matrix(grid, cellset)
            for color in color_vectors, c1 in color, c2 in color
                @test !conn[c1, c2]
            end
        end
    end
    test_coloring(generate_grid(Line, (5,)))
    test_coloring(generate_grid(QuadraticLine, (5,)))
    test_coloring(generate_grid(Triangle, (5, 5)))
    test_coloring(generate_grid(QuadraticTriangle, (5, 5)))
    test_coloring(generate_grid(Quadrilateral, (5, 5)))
    test_coloring(generate_grid(QuadraticQuadrilateral, (5, 5)))
    test_coloring(generate_grid(Tetrahedron, (5, 5, 5)))
    # test_coloring(generate_grid(QuadraticTetrahedron, (5, 5, 5)))
    test_coloring(generate_grid(Hexahedron, (5, 5, 5)))
    # test_coloring(generate_grid(QuadraticHexahedron, (5, 5, 5)))

    # color only a subset
    test_coloring(generate_grid(Line, (5,)), 1:3)
    test_coloring(generate_grid(Triangle, (5, 5)), Set{Int}(1:3^2))
    test_coloring(generate_grid(Quadrilateral, (5, 5)), Set{Int}(1:3^2))
    test_coloring(generate_grid(Tetrahedron, (5, 5, 5)), Set{Int}(1:3^3))
    test_coloring(generate_grid(Hexahedron, (5, 5, 5)), Set{Int}(1:3^3))
    # unconnected subset
    test_coloring(generate_grid(Triangle, (10, 10)), union(Set(1:10), Set(70:80)))

    #Special case with one and zero elements in the sets
    test_coloring(generate_grid(Quadrilateral, (2, 2)), [1])
    test_coloring(generate_grid(Quadrilateral, (2, 2)), [])
end

@testset "High order dof distribution" begin
    # 3-----4
    # | \   |
    # |  \  |
    # |   \ |
    # 1-----2
    grid = generate_grid(Triangle, (1, 1))

    ## Lagrange{RefTriangle,3}
    # Dofs per position per triangle
    # 3      3-14-15-11
    # | \     \      |
    # 9  7     7  16 13
    # |   \     \    |
    # |    \     \   |
    # 8  10 6     6  12
    # |      \     \ |
    # 1-4---5-2      2
    dh = DofHandler(grid)
    add!(dh, :u, Lagrange{RefTriangle,3}())
    close!(dh)
    @test celldofs(dh, 1) == [1, 2, 3, 4, 5, 6, 7, 9, 8, 10]
    @test celldofs(dh, 2) == [2, 11, 3, 12, 13, 15, 14, 7, 6, 16]

    ## Lagrange{RefTriangle,3}
    # First dof per position per triangle
    # 5      5-27-29-21
    # | \     \      |
    # 17 13   13  31 25
    # |   \     \    |
    # |    \     \   |
    # 15 19 11   11  23
    # |      \     \ |
    # 1-7---9-3      3
    dh = DofHandler(grid)
    add!(dh, :u, Lagrange{RefTriangle,3}()^2)
    close!(dh)
    @test celldofs(dh, 1) == [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 17, 18, 15, 16, 19, 20]
    @test celldofs(dh, 2) == [3, 4, 21, 22, 5, 6, 23, 24, 25, 26, 29, 30, 27, 28, 13, 14, 11, 12, 31, 32]
end

@testset "vectorization layer compat" begin
    struct VectorLagrangeTest{shape,order,vdim} <: ScalarInterpolation{shape,order} end
    Ferrite.adjust_dofs_during_distribution(ip::VectorLagrangeTest{<:Any, order}) where {order} = order > 2

    @testset "1d" begin
        grid = generate_grid(Line, (2,))

        Ferrite.vertexdof_indices(::VectorLagrangeTest{RefLine,1,2}) = ((1,2),(3,4))
        dh1 = DofHandler(grid)
        add!(dh1, :u, VectorLagrangeTest{RefLine,1,2}())
        close!(dh1)
        dh2 = DofHandler(grid)
        # TODO: Why was this RefQuadrilateral? Check it is correct to test with RefLine
        add!(dh2, :u, Lagrange{RefLine,1}()^2)
        close!(dh2)
        @test dh1.cell_dofs == dh2.cell_dofs

        Ferrite.vertexdof_indices(::VectorLagrangeTest{RefLine,1,3}) = ((1,2,3),(4,5,6))
        dh1 = DofHandler(grid)
        add!(dh1, :u, VectorLagrangeTest{RefLine,1,3}())
        close!(dh1)
        dh2 = DofHandler(grid)
        # TODO: Why was this RefQuadrilateral? Check it is correct to test with RefLine
        add!(dh2, :u, Lagrange{RefLine,1}()^3)
        close!(dh2)
        @test dh1.cell_dofs == dh2.cell_dofs
    end

    @testset "2d" begin
        grid = generate_grid(Quadrilateral, (2,2))
        Ferrite.vertexdof_indices(::VectorLagrangeTest{RefQuadrilateral,1,2}) = ((1,2),(3,4),(5,6),(7,8))
        dh1 = DofHandler(grid)
        add!(dh1, :u, VectorLagrangeTest{RefQuadrilateral,1,2}())
        close!(dh1)
        dh2 = DofHandler(grid)
        add!(dh2, :u, Lagrange{RefQuadrilateral,1}()^2)
        close!(dh2)
        @test dh1.cell_dofs == dh2.cell_dofs

        Ferrite.vertexdof_indices(::VectorLagrangeTest{RefQuadrilateral,1,3}) = ((1,2,3),(4,5,6),(7,8,9),(10,11,12))
        Ferrite.facedof_indices(::VectorLagrangeTest{RefQuadrilateral,1,3}) = ((1,2,3,4,5,6), (4,5,6,7,8,9), (7,8,9,10,11,12), (10,11,12,1,2,3))
        dh1 = DofHandler(grid)
        add!(dh1, :u, VectorLagrangeTest{RefQuadrilateral,1,3}())
        close!(dh1)
        dh2 = DofHandler(grid)
        add!(dh2, :u, Lagrange{RefQuadrilateral,1}()^3)
        close!(dh2)
        @test dh1.cell_dofs == dh2.cell_dofs
    end

    @testset "paraview_collection" begin
        grid = generate_grid(Triangle, (2,2))
        celldata = rand(getncells(grid))
        pvd = WriteVTK.paraview_collection("collection")
        vtk1 = VTKGridFile("file1", grid)
        write_cell_data(vtk1, celldata, "celldata")
        @assert isopen(vtk1.vtk)
        pvd[0.5] = vtk1
        @test !isopen(vtk1.vtk) # Should be closed when adding it
        vtk2 = VTKGridFile("file2", grid)
        WriteVTK.collection_add_timestep(pvd, vtk2, 1.0)
        @test !isopen(vtk2.vtk) # Should be closed when adding it
    end
end
