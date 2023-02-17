# to test vtk-files
using StableRNGs
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
                            (Cell{3,20,6},           3),
                            (Tetrahedron,            3))

        # create test grid, do some operations on it and then test
        # the resulting sha1 of the stored vtk file
        # after manually checking the exported vtk
        nels = ntuple(x->5, dim)
        right = Vec{dim, Float64}(ntuple(x->1.5, dim))
        left = -right
        grid = generate_grid(celltype, nels, left, right)

        transform!(grid, x-> 2x)

        radius = 2*1.5
        addcellset!(grid, "cell-1", [1,])
        addcellset!(grid, "middle-cells", x -> norm(x) < radius)
        addfaceset!(grid, "middle-faceset", x -> norm(x) < radius)
        addfaceset!(grid, "right-faceset", getfaceset(grid, "right"))
        addnodeset!(grid, "middle-nodes", x -> norm(x) < radius)

        gridfilename = "grid-$(Ferrite.celltypes[celltype])"
        vtk_grid(gridfilename, grid) do vtk
            vtk_cellset(vtk, grid, "cell-1")
            vtk_cellset(vtk, grid, "middle-cells")
            vtk_nodeset(vtk, grid, "middle-nodes")
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
        push!(dofhandler, :temperature, 1)
        push!(dofhandler, :displacement, 3)
        close!(dofhandler)
        ch = ConstraintHandler(dofhandler)
        dbc = Dirichlet(:temperature, union(getfaceset(grid, "left"), getfaceset(grid, "right-faceset")), (x,t)->1)
        add!(ch, dbc)
        dbc = Dirichlet(:temperature, getfaceset(grid, "middle-faceset"), (x,t)->4)
        add!(ch, dbc)
        for d in 1:dim
            dbc = Dirichlet(:displacement, union(getfaceset(grid, "left")), (x,t) -> d, d)
            add!(ch, dbc)
        end
        close!(ch)
        update!(ch, 0.0)
        rng = StableRNG(1234)
        u = rand(rng, ndofs(dofhandler))
        apply!(u, ch)

        dofhandlerfilename = "dofhandler-$(Ferrite.celltypes[celltype])"
        vtk_grid(dofhandlerfilename, dofhandler) do vtk
            vtk_point_data(vtk, ch)
            vtk_point_data(vtk, dofhandler, u)
        end

        # test the sha of the file
        sha = bytes2hex(open(SHA.sha1, dofhandlerfilename*".vtu"))
        if OVERWRITE_CHECKSUMS
            write(csio, sha, "\n")
        else
            @test sha in split(chomp(readline(csio)))
            rm(dofhandlerfilename*".vtu")
        end

    end

end # of testset

close(csio)

@testset "vtk tensor export" begin
    # open files
    checksums_file_tensors = joinpath(dirname(@__FILE__), "checksums2.sha1")
    if OVERWRITE_CHECKSUMS
        csio = open(checksums_file_tensors, "a")
    else
        csio = open(checksums_file_tensors, "r")
    end

    # 3D grid
    grid = generate_grid(Hexahedron, (1,1,1))

    sym_tensor_data = [SymmetricTensor{2,3}(ntuple(i->i, 6)) for j=1.0:8.0]
    tensor_data = [Tensor{2,3}(ntuple(i->i, 9)) for j=1.0:8.0]
    vector_data = [Vec{3}(ntuple(i->i, 3)) for j=1:8]

    filename_3d = "test_vtk_3d"
    vtk_grid(filename_3d, grid) do vtk_file
        vtk_point_data(vtk_file, sym_tensor_data, "symmetric tensor")
        vtk_point_data(vtk_file, tensor_data, "tensor")
        vtk_point_data(vtk_file, vector_data, "vector")
    end

    # 2D grid
    grid = generate_grid(Quadrilateral, (1,1))

    sym_tensor_data = [SymmetricTensor{2,2}(ntuple(i->i, 3)) for j=1.0:4.0]
    tensor_data = [Tensor{2,2}(ntuple(i->i, 4)) for j=1.0:4.0]
    tensor_data_1D = [SymmetricTensor{2,1}(ntuple(i->i, 1)) for j=1.0:4.0]
    vector_data = [Vec{2}(ntuple(i->i, 2)) for j=1:4]

    filename_2d = "test_vtk_2d"
    vtk_grid(filename_2d, grid) do vtk_file
        vtk_point_data(vtk_file, sym_tensor_data, "symmetric tensor")
        vtk_point_data(vtk_file, tensor_data, "tensor")
        vtk_point_data(vtk_file, tensor_data_1D, "tensor_1d")
        vtk_point_data(vtk_file, vector_data, "vector")
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

    @test getnodesets(grid) == Dict("node_set" => node_set)

    @test getnodes(grid, [1]) == [getnodes(grid, 1)] # untested

    @test length(getnodes(grid, "node_set")) == 9

    @test collect(getcoordinates(getnodes(grid, 5)).data) ≈ [0.5, 0.5]

    @test getcells(grid, "cell_set") == [getcells(grid, 1)]

    f(x) = Tensor{1,1,Float64}((1 + x[1]^2 + 2x[2]^2, ))

    values = compute_vertex_values(grid, f)
    @test f([0.0, 0.0]) == values[1]
    @test f([0.5, 0.5]) == values[5]
    @test f([1.0, 1.0]) == values[9]

    @test compute_vertex_values(grid, collect(1:9), f) == values

    # Can we test this in a better way? The set makes the order random.
    @test length(compute_vertex_values(grid, "node_set", f)) == 9

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
end

@testset "Grid sets" begin

    grid = Ferrite.generate_grid(Hexahedron, (1, 1, 1), Vec((0.,0., 0.)), Vec((1.,1.,1.)))

    #Test manual add
    addcellset!(grid, "cell_set", [1]);
    addnodeset!(grid, "node_set", [1])
    addfaceset!(grid, "face_set", [FaceIndex(1,1)])
    addedgeset!(grid, "edge_set", [EdgeIndex(1,1)])
    addvertexset!(grid, "vert_set", [VertexIndex(1,1)])

    #Test function add
    addfaceset!(grid, "left_face", (x)-> x[1] ≈ 0.0)
    addedgeset!(grid, "left_lower_edge", (x)-> x[1] ≈ 0.0 && x[3] ≈ 0.0)
    addvertexset!(grid, "left_corner", (x)-> x[1] ≈ 0.0 && x[2] ≈ 0.0 && x[3] ≈ 0.0)

    @test 1 in Ferrite.getnodeset(grid, "node_set")
    @test FaceIndex(1,5) in getfaceset(grid, "left_face")
    @test EdgeIndex(1,4) in getedgeset(grid, "left_lower_edge")
    @test VertexIndex(1,1) in getvertexset(grid, "left_corner")

end

@testset "Grid topology" begin
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
    #test vertex neighbors maps cellid and local vertex id to neighbor id and neighbor local vertex id
    @test topology.vertex_neighbor[1,3] == Ferrite.EntityNeighborhood(VertexIndex(4,1))
    @test topology.vertex_neighbor[2,4] == Ferrite.EntityNeighborhood(VertexIndex(3,2))
    @test topology.vertex_neighbor[3,3] == Ferrite.EntityNeighborhood(VertexIndex(6,1))
    @test topology.vertex_neighbor[3,2] == Ferrite.EntityNeighborhood(VertexIndex(2,4))
    @test topology.vertex_neighbor[4,1] == Ferrite.EntityNeighborhood(VertexIndex(1,3))
    @test topology.vertex_neighbor[4,4] == Ferrite.EntityNeighborhood(VertexIndex(5,2))
    @test topology.vertex_neighbor[5,2] == Ferrite.EntityNeighborhood(VertexIndex(4,4))
    @test topology.vertex_neighbor[6,1] == Ferrite.EntityNeighborhood(VertexIndex(3,3))
    #test face neighbor maps cellid and local face id to neighbor id and neighbor local face id 
    @test topology.face_neighbor[1,2] == Ferrite.EntityNeighborhood(FaceIndex(2,4))
    @test topology.face_neighbor[1,3] == Ferrite.EntityNeighborhood(FaceIndex(3,1))
    @test topology.face_neighbor[2,3] == Ferrite.EntityNeighborhood(FaceIndex(4,1))
    @test topology.face_neighbor[2,4] == Ferrite.EntityNeighborhood(FaceIndex(1,2))
    @test topology.face_neighbor[3,1] == Ferrite.EntityNeighborhood(FaceIndex(1,3))
    @test topology.face_neighbor[3,2] == Ferrite.EntityNeighborhood(FaceIndex(4,4))
    @test topology.face_neighbor[3,3] == Ferrite.EntityNeighborhood(FaceIndex(5,1))
    @test topology.face_neighbor[4,1] == Ferrite.EntityNeighborhood(FaceIndex(2,3))
    @test topology.face_neighbor[4,3] == Ferrite.EntityNeighborhood(FaceIndex(6,1))
    @test topology.face_neighbor[4,4] == Ferrite.EntityNeighborhood(FaceIndex(3,2))
    @test topology.face_neighbor[5,1] == Ferrite.EntityNeighborhood(FaceIndex(3,3))
    @test topology.face_neighbor[5,2] == Ferrite.EntityNeighborhood(FaceIndex(6,4))
    @test topology.face_neighbor[5,3] == Ferrite.EntityNeighborhood(Ferrite.BoundaryIndex[])
    @test topology.face_neighbor[5,4] == Ferrite.EntityNeighborhood(Ferrite.BoundaryIndex[])
    @test topology.face_neighbor[6,1] == Ferrite.EntityNeighborhood(FaceIndex(4,3))
    @test topology.face_neighbor[6,2] == Ferrite.EntityNeighborhood(Ferrite.BoundaryIndex[])
    @test topology.face_neighbor[6,3] == Ferrite.EntityNeighborhood(Ferrite.BoundaryIndex[])
    @test topology.face_neighbor[6,4] == Ferrite.EntityNeighborhood(FaceIndex(5,2))
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
    @test topology.edge_neighbor[1,11] == Ferrite.EntityNeighborhood(EdgeIndex(4,9))
    @test topology.edge_neighbor[2,12] == Ferrite.EntityNeighborhood(EdgeIndex(3,10))
    @test topology.edge_neighbor[3,10] == Ferrite.EntityNeighborhood(EdgeIndex(2,12))
    @test topology.edge_neighbor[4,9] == Ferrite.EntityNeighborhood(EdgeIndex(1,11))
    @test isempty(topology.vertex_neighbor)
    @test topology.face_neighbor[1,3] == Ferrite.EntityNeighborhood(FaceIndex(2,5))
    @test topology.face_neighbor[1,4] == Ferrite.EntityNeighborhood(FaceIndex(3,2))
    @test topology.face_neighbor[2,4] == Ferrite.EntityNeighborhood(FaceIndex(4,2))
    @test topology.face_neighbor[2,5] == Ferrite.EntityNeighborhood(FaceIndex(1,3))
    @test topology.face_neighbor[3,2] == Ferrite.EntityNeighborhood(FaceIndex(1,4))
    @test topology.face_neighbor[3,3] == Ferrite.EntityNeighborhood(FaceIndex(4,5))
    @test topology.face_neighbor[4,2] == Ferrite.EntityNeighborhood(FaceIndex(2,4))
    @test topology.face_neighbor[4,5] == Ferrite.EntityNeighborhood(FaceIndex(3,3))

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
    @test topology.vertex_neighbor[3,3] == Ferrite.EntityNeighborhood([VertexIndex(5,2),VertexIndex(6,1),VertexIndex(7,1)])

# test mixed grid
    cells = [
        Hexahedron((1, 2, 3, 4, 5, 6, 7, 8)),
        Quadrilateral((3, 2, 9, 10)),
        ]
    nodes = [Node(coord) for coord in zeros(Vec{2,Float64}, 10)]
    grid = Grid(cells, nodes)
    topology = ExclusiveTopology(grid)
    @test isempty(topology.vertex_neighbor)
    @test topology.face_neighbor[2,1] == Ferrite.EntityNeighborhood(EdgeIndex(1,2))
    @test topology.edge_neighbor[1,2] == Ferrite.EntityNeighborhood(FaceIndex(2,1))
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
    
    @test Ferrite.getneighborhood(topology, quadgrid, VertexIndex(1,1)) == [VertexIndex(1,2), VertexIndex(1,4)]
    @test Ferrite.getneighborhood(topology, quadgrid, VertexIndex(2,1)) == [VertexIndex(1,1), VertexIndex(1,3), VertexIndex(2,2), VertexIndex(2,4)]
    @test Ferrite.getneighborhood(topology, quadgrid, VertexIndex(5,4)) == [VertexIndex(4,2), VertexIndex(4,4), VertexIndex(5,1), VertexIndex(5,3), VertexIndex(7,1), VertexIndex(7,3), VertexIndex(8,2), VertexIndex(8,4)]
    @test Ferrite.toglobal(quadgrid, Ferrite.getneighborhood(topology, quadgrid, VertexIndex(1,1))) == [2,5]
    @test Ferrite.toglobal(quadgrid, Ferrite.getneighborhood(topology, quadgrid, VertexIndex(2,1))) == [1,6,3]
    @test Ferrite.toglobal(quadgrid, Ferrite.getneighborhood(topology, quadgrid, VertexIndex(5,4))) == [6,9,11,14]
    
    @test topology.face_skeleton == [FaceIndex(1,1),FaceIndex(1,2),FaceIndex(1,3),FaceIndex(1,4),
                                          FaceIndex(2,1),FaceIndex(2,2),FaceIndex(2,3),
                                          FaceIndex(3,1),FaceIndex(3,2),FaceIndex(3,3),
                                          FaceIndex(4,2),FaceIndex(4,3),FaceIndex(4,4),
                                          FaceIndex(5,2),FaceIndex(5,3),FaceIndex(6,2),FaceIndex(6,3),
                                          FaceIndex(7,2),FaceIndex(7,3),FaceIndex(7,4),
                                          FaceIndex(8,2),FaceIndex(8,3),FaceIndex(9,2),FaceIndex(9,3)]
    @test length(topology.face_skeleton) == 4*3 + 3*4

    quadratic_quadgrid = generate_grid(QuadraticQuadrilateral,(3,3))
    quadgrid_topology = ExclusiveTopology(grid)
    quadgrid_topology.face_skeleton == topology.face_skeleton
#                           
#                   +-----+-----+-----+
#                   |  7  |  8  |  9  |
#                   +-----+-----+-----+
#                   |  4  |  5  |  6  |
#                   +-----+-----+-----+
#                   |  1  |  2  |  3  |
#                   +-----+-----+-----+
# test application: integrate jump across element boundary 5
    function reinit!(fv::FaceValues, cellid::Int, faceid::Int, grid)
        coords = getcoordinates(grid, cellid)
        Ferrite.reinit!(fv, coords, faceid)
    end
    reinit!(fv::FaceValues, faceid::FaceIndex, grid) = reinit!(fv,faceid[1],faceid[2],grid) # wrapper for reinit!(fv,cellid,faceid,grid)
    face_neighbors_ele5 = nonzeros(topology.face_neighbor[5,:]) 
    ip = Lagrange{2, RefCube, 1}()
    qr_face = QuadratureRule{1, RefCube}(2)
    fv_ele = FaceVectorValues(qr_face, ip)
    fv_neighbor = FaceVectorValues(qr_face, ip)
    u_ele5 = [3.0 for _ in 1:8]
    u_neighbors = [5.0 for _ in 1:8]
    jump_int = 0.
    jump_abs = 0.
    for ele_faceid in 1:nfaces(quadgrid.cells[5])
        reinit!(fv_ele, 5, ele_faceid, quadgrid)
        for q_point in 1:getnquadpoints(fv_ele)
            dΩ = getdetJdV(fv_ele, q_point)
            normal_5 = getnormal(fv_ele, q_point)
            u_5_n = function_value(fv_ele, q_point, u_ele5) ⋅ normal_5
            for neighbor_entity in face_neighbors_ele5[ele_faceid].neighbor_info # only one entity can be changed to get rid of the for loop
                reinit!(fv_neighbor, neighbor_entity, quadgrid)
                normal_neighbor = getnormal(fv_neighbor, q_point) 
                u_neighbor = function_value(fv_neighbor, q_point, u_neighbors) ⋅ normal_neighbor
                jump_int += (u_5_n + u_neighbor) * dΩ
                jump_abs += abs(u_5_n + u_neighbor) * dΩ
            end 
        end
    end
    @test isapprox(jump_abs, 2/3*2*4,atol=1e-6) # 2*4*0.66666, jump is always 2, 4 sides, length =0.66
    @test isapprox(jump_int, 0.0, atol=1e-6)
end

@testset "grid coloring" begin
    function test_coloring(grid, cellset=1:getncells(grid))
        for alg in (ColoringAlgorithm.Greedy, ColoringAlgorithm.WorkStream)
            color_vectors = create_coloring(grid, cellset; alg=alg)
            @test sum(length, color_vectors) == length(cellset)
            @test union(Set.(color_vectors)...) == Set(cellset)
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
end

@testset "DoF distribution" begin
    # _________
    # |\      |
    # |  \  2 |
    # | 1  \  |
    # |______\|
    grid = generate_grid(Triangle, (1, 1))

    ## Lagrange{2,RefTetrahedron,3}
    dh = DofHandler(grid)
    push!(dh, :u, 1, Lagrange{2,RefTetrahedron,3}())
    close!(dh)
    @test celldofs(dh, 1) == [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    @test celldofs(dh, 2) == [2, 11, 3, 12, 13, 14, 15, 7, 6, 16]

    ## Lagrange{2,RefTetrahedron,4}
    dh = DofHandler(grid)
    push!(dh, :u, 1, Lagrange{2,RefTetrahedron,4}())
    close!(dh)
    @test celldofs(dh, 1) == [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]
    @test celldofs(dh, 2) == [2, 16, 3, 17, 18, 19, 20, 21, 22, 9, 8, 7, 23, 24, 25]

    ## Lagrange{2,RefTetrahedron,5}
    dh = DofHandler(grid)
    push!(dh, :u, 1, Lagrange{2,RefTetrahedron,5}())
    close!(dh)
    @test celldofs(dh, 1) == [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21]
    @test celldofs(dh, 2) == [2, 22, 3, 23, 24, 25, 26, 27, 28, 29, 30, 11, 10, 9, 8, 31, 32, 33, 34, 35, 36]

    # Consistency check for dof computation.
    grid = generate_grid(Hexahedron, (2, 2, 2))
    dh = DofHandler(grid)
    push!(dh, :u, 3, Lagrange{3,RefCube,2}())
    push!(dh, :v, 1, Lagrange{3,RefCube,2}())
    push!(dh, :w, 3, Lagrange{3,RefCube,1}())
    push!(dh, :x, 3, Lagrange{3,RefCube,2}())
    vertexdicts, edgedicts, facedicts = Ferrite.__close!(dh)
    @test Ferrite.find_field(dh, :u) == 1
    @test Ferrite.find_field(dh, :v) == 2
    @test Ferrite.find_field(dh, :w) == 3
    for (ci,cell) ∈ enumerate(getcells(grid))
        for field_idx ∈ 1:3
            for vi ∈ 1:length(Ferrite.vertices(cell))
                vertex = VertexIndex(ci,vi)
                global_vertex = Ferrite.toglobal(grid, vertex)
                dofs = Ferrite.vertex_dofs(dh, field_idx, vertex)
                if !haskey(vertexdicts[field_idx], global_vertex)
                    @test empty(dofs)
                    @test !Ferrite.has_vertex_dofs(dh, field_idx, vertex)
                else 
                    @test vertexdicts[field_idx][global_vertex] == dofs[1]
                    @test Ferrite.has_vertex_dofs(dh, field_idx, vertex)
                end
            end

            for fi ∈ 1:length(Ferrite.faces(cell))
                face = FaceIndex(ci,fi)
                global_face = Ferrite.toglobal(grid, face)
                dofs = Ferrite.face_dofs(dh, field_idx, face)
                if !haskey(facedicts[field_idx], global_face)
                    @test isempty(dofs)
                    @test !Ferrite.has_face_dofs(dh, field_idx, face)
                else
                    @test facedicts[field_idx][global_face] == dofs[1]
                    @test Ferrite.has_face_dofs(dh, field_idx, face)
                end
            end

            for ei ∈ 1:length(Ferrite.edges(cell))
                edge = EdgeIndex(ci,ei)
                global_edge = Ferrite.toglobal(grid, edge)
                dofs = Ferrite.edge_dofs(dh, field_idx, edge)
                if !haskey(edgedicts[field_idx], global_edge) 
                    @test isempty(dofs) 
                    @test !Ferrite.has_edge_dofs(dh, field_idx, edge)
                else
                    @test edgedicts[field_idx][global_edge][1] == dofs[1]
                    @test Ferrite.has_edge_dofs(dh, field_idx, edge)
                end
            end
        end
    end
end
