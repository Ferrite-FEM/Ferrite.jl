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

    @testset "discontinuous" begin
        # First test a continuous case, which should produce a continuous field.
        grid = generate_grid(Tetrahedron, (3, 3, 3))
        ip = DiscontinuousLagrange{RefTetrahedron, 1}()
        dh = close!(add!(DofHandler(grid), :u, ip))
        a = zeros(ndofs(dh))
        apply_analytical!(a, dh, :u, x -> round(Int, sum(y -> y^2, x)))
        mktempdir() do tmp
            fname = joinpath(tmp, "discontinuous_export_of_continuous_field")
            v = VTKGridFile(fname, dh) do vtk::VTKGridFile
                write_solution(vtk, dh, a)
            end
            @test Ferrite.write_discontinuous(v)
            @test bytes2hex(open(SHA.sha1, fname * ".vtu")) == "9c159760c7d5e2c437ba2faed73967bf687aa9f3"
        end

        ip = DiscontinuousLagrange{RefTetrahedron, 1}()
        dh = DofHandler(grid)
        add!(dh, :u, ip)
        add!(dh, :v, Lagrange{RefTetrahedron, 1}())
        close!(dh)
        a = zeros(ndofs(dh))
        apply_analytical!(a, dh, :u, x -> round(Int, sum(y -> y^2, x)))
        apply_analytical!(a, dh, :v, x -> round(Int, sum(y -> y^2, x)))
        nodedata_v = evaluate_at_grid_nodes(dh, a, :v)
        ch = ConstraintHandler(dh)
        add!(ch, Dirichlet(:u, getfacetset(grid, "left"), Returns(1.0)))
        add!(ch, Dirichlet(:v, getfacetset(grid, "right"), Returns(2.0)))
        close!(ch)
        a2 = zeros(ndofs(dh))
        apply!(a2, ch)
        mktempdir() do tmp
            fname = joinpath(tmp, "discontinuous_exports_of_continuous_field")
            v = VTKGridFile(fname, dh) do vtk::VTKGridFile
                write_solution(vtk, dh, a)
                write_solution(vtk, dh, a2, "_bc")
                write_node_data(vtk, nodedata_v, "nodedata_v")
                Ferrite.write_constraints(vtk, ch)
            end
            @test Ferrite.write_discontinuous(v)
            @test bytes2hex(open(SHA.sha1, fname * ".vtu")) == "d665ec0c4d6bb5112614c3f081a7c684f8cb6356"
        end

        # Produce a u such that the overall shape is f(x, xc) = 2 * (x[1]^2 - x[2]^2) - (xc[1]^2 - xc[2]^2)
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
                        val = δu * (f(x) * 2 - f(xc))
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
            dh_dg = close!(add!(DofHandler(grid), :u, DiscontinuousLagrange{RefQuadrilateral, 1}()))

            u_dg = calculate_u(dh_dg)

            testhash = "daf0cbe26ff709705f338526b19881ef5758f16b"

            fname1 = joinpath(tmp, "discont_kwarg")
            VTKGridFile(fname1, grid; write_discontinuous = true) do vtk
                write_solution(vtk, dh_dg, u_dg)
            end
            @test bytes2hex(open(SHA.sha1, fname1 * ".vtu")) == testhash

            fname2 = joinpath(tmp, "discont_auto")
            VTKGridFile(fname2, dh_dg) do vtk
                write_solution(vtk, dh_dg, u_dg)
            end
            @test bytes2hex(open(SHA.sha1, fname2 * ".vtu")) == testhash
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
