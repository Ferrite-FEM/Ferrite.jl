# to test vtk-files
using StableRNGs
OVERWRITE_CHECKSUMS = false
checksums_file = joinpath(dirname(@__FILE__), "checksums.sha1")
checksum_list = read(checksums_file, String)
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

        gridfilename = "grid-$(repr(celltype))"
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
            @test chomp(readline(csio)) == sha
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

        dofhandlerfilename = "dofhandler-$(repr(celltype))"
        vtk_grid(dofhandlerfilename, dofhandler) do vtk
            vtk_point_data(vtk, ch)
            vtk_point_data(vtk, dofhandler, u)
        end

        # test the sha of the file
        sha = bytes2hex(open(SHA.sha1, dofhandlerfilename*".vtu"))
        if OVERWRITE_CHECKSUMS
            write(csio, sha, "\n")
        else
            @test chomp(readline(csio)) == sha
            rm(dofhandlerfilename*".vtu")
        end

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
            @test chomp(readline(csio)) == sha
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

@testset "grid coloring" begin
    function test_coloring(grid, cellset=Set(1:getncells(grid)))
        for alg in (Ferrite.GREEDY, Ferrite.WORKSTREAM)
            color_vectors = create_coloring(grid, cellset; alg=alg)
            @test sum(length, color_vectors) == length(cellset)
            @test union(Set.(color_vectors)...) == cellset
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
    test_coloring(generate_grid(Line, (5,)), Set{Int}(1:3))
    test_coloring(generate_grid(Triangle, (5, 5)), Set{Int}(1:3^2))
    test_coloring(generate_grid(Quadrilateral, (5, 5)), Set{Int}(1:3^2))
    test_coloring(generate_grid(Tetrahedron, (5, 5, 5)), Set{Int}(1:3^3))
    test_coloring(generate_grid(Hexahedron, (5, 5, 5)), Set{Int}(1:3^3))
    # unconnected subset
    test_coloring(generate_grid(Triangle, (10, 10)), union(Set(1:10), Set(70:80)))
end
