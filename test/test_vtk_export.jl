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
end
