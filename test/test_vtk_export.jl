@testset "VTKGridFile" begin #TODO: Move all vtk tests here
    @testset "show(::VTKGridFile)" begin
        mktempdir() do tmp
            grid = generate_grid(Quadrilateral, (2, 2))
            vtk = VTKGridFile(joinpath(tmp, "showfile"), grid)
            showstring_open = sprint(show, MIME"text/plain"(), vtk)
            @test startswith(showstring_open, "VTKGridFile for the open file")
            @test contains(showstring_open, "showfile.vtu")
            close(vtk)
            showstring_closed = sprint(show, MIME"text/plain"(), vtk)
            @test startswith(showstring_closed, "VTKGridFile for the closed file")
            @test contains(showstring_closed, "showfile.vtu")
        end
    end
    @testset "cellcolors" begin
        mktempdir() do tmp
            grid = generate_grid(Quadrilateral, (4, 4))
            colors = create_coloring(grid)
            fname = joinpath(tmp, "colors")
            v = VTKGridFile(fname, grid) do vtk::VTKGridFile
                @test Ferrite.write_cell_colors(vtk, grid, colors) === vtk
            end
            @test v isa VTKGridFile
            @test bytes2hex(open(SHA.sha1, fname * ".vtu")) == "b804d0b064121b672d8e35bcff8446eda361cac3"
        end
    end
    @testset "constraints" begin
        mktempdir() do tmp
            grid = generate_grid(Tetrahedron, (4, 4, 4))
            dh = DofHandler(grid)
            add!(dh, :u, Lagrange{RefTetrahedron, 1}())
            close!(dh)
            ch = ConstraintHandler(dh)
            add!(ch, Dirichlet(:u, getfacetset(grid, "left"), x -> 0.0))
            addnodeset!(grid, "nodeset", x -> x[1] ≈ 1.0)
            add!(ch, Dirichlet(:u, getnodeset(grid, "nodeset"), x -> 0.0))
            close!(ch)
            fname = joinpath(tmp, "constraints")
            v = VTKGridFile(fname, grid) do vtk::VTKGridFile
                @test Ferrite.write_constraints(vtk, ch) === vtk
            end
            @test v isa VTKGridFile
            @test bytes2hex(open(SHA.sha1, fname * ".vtu")) == "31b506bd9729b11992f8bcb79a2191eb65d223bf"
        end
    end
    @testset "write_cellset" begin
        # More tests in `test_grid_dofhandler_vtk.jl`, this just validates writing all sets in the grid
        # which is not tested there, see https://github.com/Ferrite-FEM/Ferrite.jl/pull/948
        mktempdir() do tmp
            grid = generate_grid(Quadrilateral, (2, 2))
            addcellset!(grid, "set1", 1:2)
            addcellset!(grid, "set2", 1:4)
            manual = joinpath(tmp, "manual")
            auto = joinpath(tmp, "auto")
            v = VTKGridFile(manual, grid) do vtk::VTKGridFile
                @test Ferrite.write_cellset(vtk, grid, keys(Ferrite.getcellsets(grid))) === vtk
            end
            @test v isa VTKGridFile
            v = VTKGridFile(auto, grid) do vtk::VTKGridFile
                @test Ferrite.write_cellset(vtk, grid) === vtk
            end
            @test v isa VTKGridFile
            @test bytes2hex(open(SHA.sha1, manual * ".vtu")) == bytes2hex(open(SHA.sha1, auto * ".vtu"))
        end
    end
    @testset "type promotion" begin
        grid = generate_grid(Triangle, (2, 2))
        dh = DofHandler(grid)
        add!(dh, :u, Lagrange{RefTriangle, 1}()^2)
        add!(dh, :p, Lagrange{RefTriangle, 1}())
        close!(dh)
        for T in (Int, Float32, Float64)
            u = collect(T, 1:ndofs(dh))
            for n in (:u, :p)
                data = Ferrite._evaluate_at_grid_nodes(dh, u, n, Val(true))
                @test data isa Matrix{promote_type(T, Float32)}
                @test size(data) == (n === :u ? 3 : 1, getnnodes(grid))
            end
        end
    end
    @testset "write_solution view" begin
        grid = generate_grid(Hexahedron, (5, 5, 5))
        dofhandler = DofHandler(grid)
        ip = geometric_interpolation(Hexahedron)
        add!(dofhandler, :temperature, ip)
        add!(dofhandler, :displacement, ip^3)
        close!(dofhandler)
        u = rand(ndofs(dofhandler))
        dofhandlerfilename = "dofhandler-no-views"
        VTKGridFile(dofhandlerfilename, grid) do vtk::VTKGridFile
            @test write_solution(vtk, dofhandler, u) === vtk
        end
        dofhandler_views_filename = "dofhandler-views"
        VTKGridFile(dofhandler_views_filename, grid) do vtk::VTKGridFile
            @test write_solution(vtk, dofhandler, (@view u[1:end])) === vtk
        end

        # test the sha of the file
        sha = bytes2hex(open(SHA.sha1, dofhandlerfilename * ".vtu"))
        sha_views = bytes2hex(open(SHA.sha1, dofhandler_views_filename * ".vtu"))

        @test sha == sha_views
        rm(dofhandlerfilename * ".vtu")
        rm(dofhandler_views_filename * ".vtu")
    end
end
