@testset "VTKFile" begin #TODO: Move all vtk tests here
    @testset "show(::VTKFile)" begin
        mktempdir() do tmp
            grid = generate_grid(Quadrilateral, (2,2))
            vtk = VTKFile(joinpath(tmp, "showfile"), grid)
            showstring_open = sprint(show, MIME"text/plain"(), vtk)
            @test startswith(showstring_open, "VTKFile for the open file")
            @test contains(showstring_open, "showfile.vtu")
            close(vtk)
            showstring_closed = sprint(show, MIME"text/plain"(), vtk)
            @test startswith(showstring_closed, "VTKFile for the closed file")
            @test contains(showstring_closed, "showfile.vtu")
        end
    end
    @testset "cellcolors" begin
        mktempdir() do tmp
            grid = generate_grid(Quadrilateral, (4, 4))
            colors = create_coloring(grid)
            fname = joinpath(tmp, "colors")
            VTKFile(fname, grid) do vtk
                Ferrite.write_cell_colors(vtk, grid, colors)
            end
            @test bytes2hex(open(SHA.sha1, fname*".vtu")) == "b804d0b064121b672d8e35bcff8446eda361cac3"
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
            VTKFile(fname, grid) do vtk
                Ferrite.write_constraints(vtk, ch)
            end
            @test bytes2hex(open(SHA.sha1, fname*".vtu")) == "31b506bd9729b11992f8bcb79a2191eb65d223bf"
        end
    end
    @testset "write_cellset" begin
        # More tests in `test_grid_dofhandler_vtk.jl`, this just validates writing all sets in the grid
        # which is not tested there, see https://github.com/Ferrite-FEM/Ferrite.jl/pull/948
        mktempdir() do tmp
            grid = generate_grid(Quadrilateral, (2,2))
            addcellset!(grid, "set1", 1:2)
            addcellset!(grid, "set2", 1:4)
            manual = joinpath(tmp, "manual")
            auto = joinpath(tmp, "auto")
            VTKFile(manual, grid) do vtk
                Ferrite.write_cellset(vtk, grid, keys(Ferrite.getcellsets(grid)))
            end
            VTKFile(auto, grid) do vtk
                Ferrite.write_cellset(vtk, grid)
            end
            @test bytes2hex(open(SHA.sha1, manual*".vtu")) == bytes2hex(open(SHA.sha1, auto*".vtu"))
        end
    end
end
