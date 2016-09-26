using SHA
@testset "VTK" begin
    OVERWRITE_CHECKSUMS = false
    checksums_file = "checksums.sha1"
    checksum_list = readstring(checksums_file)
    if OVERWRITE_CHECKSUMS
        csio = open(checksums_file, "w")
    end

    # Set up small mesh
    n1 = Vec{2}((0.0,0.0))
    n2 = Vec{2}((1.0,0.0))
    n3 = Vec{2}((2.0,0.0))
    n4 = Vec{2}((0.0,1.0))
    n5 = Vec{2}((1.0,1.0))
    n6 = Vec{2}((2.0,1.0))
    n7 = Vec{2}((0.0,2.0))
    n8 = Vec{2}((1.0,2.0))
    n9 = Vec{2}((2.0,2.0))
    coords = Vec{2,Float64}[n1,n2,n3,n4,n5,n6,n7,n8,n9]

    #####################
    # Triangle 2-D mesh #
    #####################
    celltype = WriteVTK.VTKCellTypes.VTK_TRIANGLE
    triangle_mesh_edof = [1 2 4
                          2 5 4
                          2 6 5
                          2 3 6
                          4 8 7
                          4 5 8
                          5 6 8
                          6 9 8]'
    vtkfile = vtk_grid("triangles",coords,triangle_mesh_edof,celltype)
    vtk_save(vtkfile)

    sha = bytes2hex(sha1("triangles.vtu"))
    if OVERWRITE_CHECKSUMS
        write(csio, sha*" triangles.vtu\n")
    else
        # Returns 0:-1 if string is not found
        cmp = search(checksum_list, sha)
        @test cmp != 0:-1
    end

    # Test old version as well
    coordmat = reinterpret(Float64,coords,(2,9))
    vtkfile = vtk_grid(triangle_mesh_edof,coordmat,"triangles_old")
    vtk_save(vtkfile)

    sha = bytes2hex(sha1("triangles_old.vtu"))
    if OVERWRITE_CHECKSUMS
        write(csio, sha*" triangles_old.vtu\n")
    else
        # Returns 0:-1 if string is not found
        cmp = search(checksum_list, sha)
        @test cmp != 0:-1
    end

    #################
    # Quad 2-D mesh #
    #################
    celltype = WriteVTK.VTKCellTypes.VTK_QUAD
    quad_mesh_edof = [1 2 5 4
                      2 3 6 5
                      4 5 8 7
                      5 6 9 8]'
    vtkfile = vtk_grid("quads",coords,quad_mesh_edof,celltype)
    vtk_save(vtkfile)

    sha = bytes2hex(sha1("quads.vtu"))
    if OVERWRITE_CHECKSUMS
        write(csio, sha*" quads.vtu\n")
    else
        # Returns 0:-1 if string is not found
        cmp = search(checksum_list, sha)
        @test cmp != 0:-1
    end

    OVERWRITE_CHECKSUMS && close(csio)
end
