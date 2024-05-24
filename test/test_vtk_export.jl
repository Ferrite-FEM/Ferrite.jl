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

    @testset "discontinuous" begin
        # Produce a u such that the overall shape is f(x, xc) = 2 * (x[1]^2 - x[2]^2) + (xc[1]^2 - xc[2]^2)
        # where xc is the center point of the cell. To avoid floating point issues for the hash,
        # we test that all values are approximately an integer, and round to integers before storing.
        function calculate_u(dh)
            f(z) = z[1]^2 - z[2]^2
            u = zeros(ndofs(dh))
            ip = Ferrite.getfieldinterpolation(dh, (1, 1)) # Only one subdofhandler and one field.
            cv = CellValues(QuadratureRule{RefQuadrilateral}(:lobatto, 2), ip)
            for cell in CellIterator(dh)
                reinit!(cv, cell)
                # Cell center
                xc = sum(getcoordinates(cell)) / getnquadpoints(cv)
                for q_point in 1:getnquadpoints(cv)
                    x = spatial_coordinate(cv, q_point, getcoordinates(cell))
                    for i in 1:getnbasefunctions(cv)
                        δu = shape_value(cv, q_point, i)
                        val = δu * f(x) * 2 + f(xc)
                        intval = round(Int, val)
                        # Ensure output unaffected by floating point errors,
                        # as we will compare vtk output with a hash
                        @assert abs(val - intval) < sqrt(eps())
                        u[celldofs(cell)[i]] += intval
                    end
                end
            end
            return u
        end

        mktempdir() do tmp
            nel = 20 # Dimensions assure integer coordinates at nodes and quad cell centers
            xcorner = nel * ones(Vec{2})
            grid = generate_grid(Quadrilateral, (nel, nel), -xcorner, xcorner)
            # Good to keep for comparison:
            # dh_cont = close!(add!(DofHandler(grid), :u, Lagrange{RefQuadrilateral,1}()))
            # u_cont = calculate_u(dh_cont)
            dh_dg = close!(add!(DofHandler(grid), :u, DiscontinuousLagrange{RefQuadrilateral,1}()))

            u_dg = calculate_u(dh_dg)

            testhash = "aac3484ca583d8c27c925a468436876398c6b0c7"

            fname1 = joinpath(tmp, "discont_kwarg")
            VTKFile(fname1, grid; write_discontinuous=true) do vtk
                write_solution(vtk, dh_dg, u_dg)
            end
            @test bytes2hex(open(SHA.sha1, fname1*".vtu")) == testhash

            fname2 = joinpath(tmp, "discont_auto")
            VTKFile(fname2, dh_dg) do vtk
                write_solution(vtk, dh_dg, u_dg)
            end
            @test bytes2hex(open(SHA.sha1, fname2*".vtu")) == testhash
        end

    end
end
