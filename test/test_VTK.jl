using SHA
@testset "VTK" begin
    OVERWRITE_CHECKSUMS = false
    checksums_file = joinpath(dirname(@__FILE__), "checksums.sha1")
    checksum_list = readstring(checksums_file)
    if OVERWRITE_CHECKSUMS
        csio = open(checksums_file, "w")
    end

    # Set up small mesh
    n1 = Vec{2}((0.0,0.0)); n2 = Vec{2}((1.0,0.0))
    n3 = Vec{2}((2.0,0.0)); n4 = Vec{2}((0.0,1.0))
    n5 = Vec{2}((1.0,1.0)); n6 = Vec{2}((2.0,1.0))
    n7 = Vec{2}((0.0,2.0)); n8 = Vec{2}((1.0,2.0))
    n9 = Vec{2}((2.0,2.0))
    coords_2D = Vec{2,Float64}[n1,n2,n3,n4,n5,n6,n7,n8,n9]

    triangle_mesh_edof = [1 2 2 2 4 4 5 6
                          2 5 6 3 8 5 6 9
                          4 4 5 6 7 8 8 8]

    quad_mesh_edof = [1 2 4 5
                      2 3 5 6
                      5 6 8 9
                      4 5 7 8]

    n1 = Vec{3}((0.0,0.0,0.0)); n2 = Vec{3}((1.0,0.0,0.0))
    n3 = Vec{3}((2.0,0.0,0.0)); n4 = Vec{3}((0.0,2.0,0.0))
    n5 = Vec{3}((1.0,2.0,0.0)); n6 = Vec{3}((2.0,2.0,0.0))
    n7 = Vec{3}((0.0,0.0,3.0)); n8 = Vec{3}((1.0,0.0,3.0))
    n9 = Vec{3}((2.0,0.0,3.0)); n10 = Vec{3}((0.0,2.0,3.0))
    n11 = Vec{3}((1.0,2.0,3.0)); n12 = Vec{3}((2.0,2.0,3.0))
    coords_3D = Vec{3,Float64}[n1,n2,n3,n4,n5,n6,n7,n8,n9,n10,n11,n12]

    cube_mesh_edof = [1 2 5 4 7 8 11 10
                      2 3 6 5 8 9 12 11]'

    tetra_mesh_edof = [1 2 2 2 4 2 3 3 3 5
                       2 5 4 7 7 3 6 5 8 8
                       4 4 7 8 11 5 5 8 9 12
                       7 11 11 11 10 8 12 12 12 11]

    for (name, coord, edof, ctype) in (("triangles", coords_2D, triangle_mesh_edof, VTKCellTypes.VTK_TRIANGLE),
                                       ("quads", coords_2D, quad_mesh_edof, VTKCellTypes.VTK_QUAD),
                                       ("cubes", coords_3D, cube_mesh_edof, VTKCellTypes.VTK_HEXAHEDRON),
                                       ("tetras", coords_3D, tetra_mesh_edof, VTKCellTypes.VTK_TETRA))

        vtkfile = vtk_grid(name, coord, edof, ctype)
        vtk_save(vtkfile)

        sha = bytes2hex(sha1(name*".vtu"))
        if OVERWRITE_CHECKSUMS
            write(csio, sha*" "*name*".vtu\n")
        else
            # Returns 0:-1 if string is not found
            cmp = search(checksum_list, sha)
            @test cmp != 0:-1
            rm(name*".vtu")
        end

        # Test deprecated method, PR #70
        coordmat = reinterpret(Float64,coord,(size(coord[1],1),length(coord)))

        vtkfile = vtk_grid(edof, coordmat, name*"_old")
        vtk_save(vtkfile)

        sha = bytes2hex(sha1(name*"_old.vtu"))
        if OVERWRITE_CHECKSUMS
            write(csio, sha*" "*name*"_old.vtu\n")
        else
            # Returns 0:-1 if string is not found
            cmp = search(checksum_list, sha)
            @test cmp != 0:-1
            rm(name*"_old.vtu")
        end
    end

    OVERWRITE_CHECKSUMS && close(csio)

    # Test getVTKtype
    for functionspace in (Lagrange{1, RefCube, 1}(),
                          Lagrange{1, RefCube, 2}(),
                          Lagrange{2, RefCube, 1}(),
                          Lagrange{2, RefCube, 2}(),
                          Lagrange{2, RefTetrahedron, 1}(),
                          Lagrange{2, RefTetrahedron, 2}(),
                          Lagrange{3, RefCube, 1}(),
                          Serendipity{2, RefCube, 2}(),
                          Lagrange{3, RefTetrahedron, 1}())

        @test getVTKtype(functionspace).nodes == getnbasefunctions(functionspace)
    end

end # of testset
