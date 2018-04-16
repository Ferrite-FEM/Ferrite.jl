# to test vtk-files
OVERWRITE_CHECKSUMS = false
checksums_file = joinpath(dirname(@__FILE__), "checksums.sha1")
checksum_list = readstring(checksums_file)
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

        gridfilename = "grid-$(JuAFEM.celltypes[celltype])"
        VTK.grid(gridfilename, grid) do vtk
            VTK.cellset(vtk, grid, "cell-1")
            VTK.cellset(vtk, grid, "middle-cells")
            VTK.nodeset(vtk, grid, "middle-nodes")
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
        srand(1234)
        u = rand(ndofs(dofhandler))
        apply!(u, ch)

        dofhandlerfilename = "dofhandler-$(JuAFEM.celltypes[celltype])"
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

if OVERWRITE_CHECKSUMS
    close(csio)
end
