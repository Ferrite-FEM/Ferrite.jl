# Imports for parallel (isolated) test execution:
include("vtk_test_utils.jl")

# Cell types defined outside of Ferrite, as in downstream packages, with and without a custom
# VTK node order
struct VTKTestCell <: Ferrite.AbstractCell{RefQuadrilateral}
    nodes::NTuple{4, Int}
end
Ferrite.cell_to_vtkcell(::Type{VTKTestCell}) = Ferrite.VTKCellTypes.VTK_QUAD
Ferrite.nodes_to_vtkorder(cell::VTKTestCell) = [cell.nodes[1], cell.nodes[2], cell.nodes[4], cell.nodes[3]]
struct VTKTestCellDefaultOrder <: Ferrite.AbstractCell{RefQuadrilateral}
    nodes::NTuple{4, Int}
end
Ferrite.cell_to_vtkcell(::Type{VTKTestCellDefaultOrder}) = Ferrite.VTKCellTypes.VTK_QUAD

@testset "VTKGridFile" begin #TODO: Move all vtk tests here
    @testset "VTK cells" begin
        connectivity(c) = collect(Int, c.connectivity)
        # The node numbers of the cell in VTK order (cf. vtk_test_utils.jl)
        expected(cell) = collect(cell.nodes)[vtk_node_order(cell)]
        # A single cell type
        for (CT, n) in ((Quadrilateral, 4), (Pyramid, 5), (QuadraticWedge, 18), (QuadraticHexahedron, 27))
            cells = [CT(ntuple(i -> 100k + i, n)) for k in 1:3]
            cls = Ferrite.create_vtk_cells(cells)
            @test isconcretetype(eltype(cls))
            @test all(c -> c.ctype == Ferrite.cell_to_vtkcell(CT), cls)
            @test map(connectivity, cls) == map(expected, cells)
        end
        # Mixed cell types
        cells = Ferrite.AbstractCell[Quadrilateral((1, 2, 3, 4)), Triangle((2, 5, 3)), Pyramid((1, 2, 3, 4, 5))]
        cls = Ferrite.create_vtk_cells(cells)
        @test isconcretetype(eltype(cls))
        @test map(c -> c.ctype, cls) == map(c -> Ferrite.cell_to_vtkcell(typeof(c)), cells)
        @test map(connectivity, cls) == map(expected, cells)
        # Full grid data, also for discontinuous export, on a mixed grid
        nodes = [Node((0.0, 0.0)), Node((1.0, 0.0)), Node((1.0, 1.0)), Node((0.0, 1.0)), Node((2.0, 0.0))]
        grid = Grid(Union{Quadrilateral, Triangle}[Quadrilateral((1, 2, 3, 4)), Triangle((2, 5, 3))], nodes)
        coords, cls = Ferrite.create_vtk_griddata(grid)
        @test isconcretetype(eltype(cls))
        @test map(connectivity, cls) == [[1, 2, 3, 4], [2, 5, 3]]
        coords, cls, cellnodes, node_mapping = Ferrite.create_discontinuous_vtk_griddata(grid)
        @test isconcretetype(eltype(cls))
        @test map(connectivity, cls) == [[1, 2, 3, 4], [5, 6, 7]]
        @test node_mapping == [1, 2, 3, 4, 2, 5, 3]
        # Discontinuous export of a single cell type with reordered nodes
        grid = generate_grid(Pyramid, (1, 1, 1))
        coords, cls, cellnodes, node_mapping = Ferrite.create_discontinuous_vtk_griddata(grid)
        @test isconcretetype(eltype(cls))
        @test map(connectivity, cls) == [collect(r)[VTK_NODE_ORDER[Pyramid]] for r in cellnodes]
        # The cells can be given as a tuple
        cells = (Quadrilateral((1, 2, 3, 4)), Pyramid((1, 2, 3, 4, 5)))
        @test map(connectivity, Ferrite.create_vtk_cells(cells)) == map(expected, collect(cells))
        # nodes_to_vtkorder gives the VTK node order of a single cell
        @test Ferrite.nodes_to_vtkorder(Pyramid((1, 2, 3, 4, 5))) == [1, 2, 4, 3, 5]
        @test Ferrite.nodes_to_vtkorder(Quadrilateral((1, 2, 3, 4))) == [1, 2, 3, 4]
        # All cell types of Ferrite that can be exported use the non-allocating path
        for m in methods(Ferrite.cell_to_vtkcell)
            m.module === Ferrite || continue
            @test m.sig.parameters[2].parameters[1] <: Ferrite.FerriteCell
        end
        # Cell types defined outside of Ferrite use nodes_to_vtkorder
        cells = Ferrite.AbstractCell[VTKTestCell((1, 2, 3, 4)), VTKTestCellDefaultOrder((1, 2, 3, 4)), Quadrilateral((1, 2, 3, 4))]
        cls = Ferrite.create_vtk_cells(cells)
        @test isconcretetype(eltype(cls))
        @test map(connectivity, cls) == [[1, 2, 4, 3], [1, 2, 3, 4], [1, 2, 3, 4]]
        grid = Grid([VTKTestCell((1, 2, 3, 4))], [Node((0.0, 0.0)), Node((1.0, 0.0)), Node((0.0, 1.0)), Node((1.0, 1.0))])
        coords, cls, cellnodes, node_mapping = Ferrite.create_discontinuous_vtk_griddata(grid)
        @test map(connectivity, cls) == [[1, 2, 4, 3]]
    end

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
            data = read_vtk(fname * ".vtu")
            test_vtk_grid(data, grid)
            @test keys(data.cell_data) == Set(["coloring"])
            @test data.cell_data["coloring"] == [findfirst(c -> cell in c, colors) for cell in 1:getncells(grid)]
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
            data = read_vtk(fname * ".vtu")
            test_vtk_grid(data, grid)
            @test keys(data.point_data) == Set(["u_bc"])
            @test data.point_data["u_bc"] == [abs(x[1]) ≈ 1 for x in get_node_coordinate.(getnodes(grid))]
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
            data = read_vtk(fname * ".vtu")
            point_nodes = test_vtk_grid(data, grid; discontinuous = true)
            xs = get_node_coordinate.(getnodes(grid))[point_nodes]
            @test keys(data.point_data) == Set(["u"])
            @test data.point_data["u"] ≈ [round(Int, sum(y -> y^2, x)) for x in xs]
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
                write_solution(vtk, dh, a2, "_applied")
                write_node_data(vtk, nodedata_v, "nodedata_v")
                Ferrite.write_constraints(vtk, ch)
            end
            @test Ferrite.write_discontinuous(v)
            data = read_vtk(fname * ".vtu")
            point_nodes = test_vtk_grid(data, grid; discontinuous = true)
            xs = get_node_coordinate.(getnodes(grid))[point_nodes]
            on_left = [x[1] ≈ -1 for x in xs]
            on_right = [x[1] ≈ 1 for x in xs]
            @test keys(data.point_data) == Set(["u", "v", "u_applied", "v_applied", "nodedata_v", "u_bc", "v_bc"])
            @test data.point_data["u"] ≈ [round(Int, sum(y -> y^2, x)) for x in xs]
            @test data.point_data["v"] ≈ data.point_data["u"]
            @test data.point_data["nodedata_v"] ≈ data.point_data["u"]
            # The discontinuous field :u is only constrained in the cells with a facet in the set
            u_applied = zeros(length(point_nodes))
            point_offsets = cumsum([0; [Ferrite.nnodes(cell) for cell in getcells(grid)]])
            for (cellid, facetid) in getfacetset(grid, "left")
                facetnodes = Ferrite.facets(getcells(grid, cellid))[facetid]
                for (i, node) in enumerate(Ferrite.get_node_ids(getcells(grid, cellid)))
                    node in facetnodes && (u_applied[point_offsets[cellid] + i] = 1)
                end
            end
            @test data.point_data["u_applied"] == u_applied
            @test data.point_data["v_applied"] == 2 .* on_right
            @test data.point_data["u_bc"] == on_left
            @test data.point_data["v_bc"] == on_right
        end

        # Produce a u such that the overall shape is f(x, xc) = 2 * (x[1]^2 - x[2]^2) - (xc[1]^2 - xc[2]^2)
        # where xc is the center point of the cell.
        f(z) = z[1]^2 - z[2]^2
        function calculate_u(dh)
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
                        u[celldofs(cell)[i]] += δu * (f(x) * 2 - f(xc))
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

            fname1 = joinpath(tmp, "discont_kwarg")
            VTKGridFile(fname1, grid; write_discontinuous = true) do vtk
                write_solution(vtk, dh_dg, u_dg)
            end
            data = read_vtk(fname1 * ".vtu")
            point_nodes = test_vtk_grid(data, grid; discontinuous = true)
            expected_u = Float64[]
            for cell in getcells(grid)
                xs = get_node_coordinate.(getnodes(grid, collect(cell.nodes)))
                xc = sum(xs) / length(xs)
                append!(expected_u, [2 * f(x) - f(xc) for x in xs])
            end
            @test keys(data.point_data) == Set(["u"])
            @test data.point_data["u"] ≈ expected_u

            fname2 = joinpath(tmp, "discont_auto")
            VTKGridFile(fname2, dh_dg) do vtk
                write_solution(vtk, dh_dg, u_dg)
            end
            @test isequal(read_vtk(fname2 * ".vtu"), data)
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
            data = read_vtk(manual * ".vtu")
            test_vtk_grid(data, grid)
            @test keys(data.cell_data) == Set(["set1", "set2"])
            @test data.cell_data["set1"] == [1, 1, 0, 0]
            @test data.cell_data["set2"] == [1, 1, 1, 1]
            @test isequal(read_vtk(auto * ".vtu"), data)
        end
    end
    @testset "quadratic cells without grid generator" begin
        # Parametric coordinates of the VTK cells in [0, 1]^3, as returned by
        # `vtkCell::GetParametricCoords` (vtkQuadraticTetra, vtkTriQuadraticHexahedron and
        # vtkBiQuadraticQuadraticWedge).
        vtk_coords = [
            QuadraticTetrahedron => [
                (0, 0, 0), (1, 0, 0), (0, 1, 0), (0, 0, 1),
                (0.5, 0, 0), (0.5, 0.5, 0), (0, 0.5, 0), (0, 0, 0.5), (0.5, 0, 0.5), (0, 0.5, 0.5),
            ],
            QuadraticHexahedron => [
                (0, 0, 0), (1, 0, 0), (1, 1, 0), (0, 1, 0), (0, 0, 1), (1, 0, 1), (1, 1, 1), (0, 1, 1),
                (0.5, 0, 0), (1, 0.5, 0), (0.5, 1, 0), (0, 0.5, 0), (0.5, 0, 1), (1, 0.5, 1),
                (0.5, 1, 1), (0, 0.5, 1), (0, 0, 0.5), (1, 0, 0.5), (1, 1, 0.5), (0, 1, 0.5),
                (0, 0.5, 0.5), (1, 0.5, 0.5), (0.5, 0, 0.5), (0.5, 1, 0.5), (0.5, 0.5, 0), (0.5, 0.5, 1),
                (0.5, 0.5, 0.5),
            ],
            QuadraticWedge => [
                (0, 0, 0), (1, 0, 0), (0, 1, 0), (0, 0, 1), (1, 0, 1), (0, 1, 1),
                (0.5, 0, 0), (0.5, 0.5, 0), (0, 0.5, 0), (0.5, 0, 1), (0.5, 0.5, 1), (0, 0.5, 1),
                (0, 0, 0.5), (1, 0, 0.5), (0, 1, 0.5), (0.5, 0, 0.5), (0.5, 0.5, 0.5), (0, 0.5, 0.5),
            ],
        ]
        mktempdir() do tmp
            for (C, coords) in vtk_coords
                @testset "$C" begin
                    # A single cell with the nodes placed at the reference coordinates, mapped
                    # to [0, 1]^3, such that each node's position identifies it.
                    ip = geometric_interpolation(C)
                    ξs = Ferrite.reference_coordinates(ip)
                    to_unit(ξ) = getrefshape(ip) === RefHexahedron ? (ξ + ones(ξ)) / 2 : ξ
                    grid = Grid([C(ntuple(identity, length(ξs)))], [Node(to_unit(ξ)) for ξ in ξs])
                    fname = joinpath(tmp, string(C))
                    VTKGridFile(fname, grid) do vtk
                    end
                    data = read_vtk(fname * ".vtu")
                    @test data.celltypes == [Ferrite.cell_to_vtkcell(C).vtk_id]
                    connectivity = only(data.cells)
                    @test length(connectivity) == length(coords)
                    @test data.points[:, connectivity] ≈ reduce(hcat, collect.(coords))
                end
            end
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
        tmp = mktempdir()
        dofhandlerfilename = joinpath(tmp, "dofhandler-no-views")
        VTKGridFile(dofhandlerfilename, grid) do vtk::VTKGridFile
            @test write_solution(vtk, dofhandler, u) === vtk
        end
        dofhandler_views_filename = joinpath(tmp, "dofhandler-views")
        VTKGridFile(dofhandler_views_filename, grid) do vtk::VTKGridFile
            @test write_solution(vtk, dofhandler, (@view u[1:end])) === vtk
        end

        # test that the content of the files is the same
        data = read_vtk(dofhandlerfilename * ".vtu")
        @test keys(data.point_data) == Set(["temperature", "displacement"])
        @test data.point_data["temperature"] ≈ evaluate_at_grid_nodes(dofhandler, u, :temperature)
        @test isequal(read_vtk(dofhandler_views_filename * ".vtu"), data)
    end
    @testset "discontinuous_projection" begin
        mktempdir() do tmp
            grid = generate_grid(Quadrilateral, (2, 2), Vec{2}((0.0, 0.0)), Vec{2}((1.0, 1.0)))
            qr = QuadratureRule{RefQuadrilateral}(2)
            ip = Lagrange{RefQuadrilateral, 1}()^2
            dh = DofHandler(grid)
            add!(dh, :u, ip)
            close!(dh)
            nQP = getnquadpoints(qr)
            proj = L2Projector(ip, grid)
            for T in (SymmetricTensor{2, 2}, Tensor{2, 2})
                qp_quantities = [[zero(T) for _ in 1:nQP] for _ in 1:getncells(grid)]
                for cell in CellIterator(dh)
                    cell_quantities = qp_quantities[cellid(cell)]
                    for qp in 1:nQP
                        cell_quantities[qp] = rand(T)
                    end
                end
                field = project(proj, qp_quantities, qr)
                VTKGridFile(joinpath(tmp, "output"), dh, write_discontinuous = true) do vtk
                    write_projection(vtk, proj, field, "cellid")
                end
            end
        end
    end
    @testset "discontinuous_embedded" begin
        mktempdir() do tmp
            grid = generate_grid(Line, (2,), Vec((0.0, 0.0)), Vec((1.0, 0.5)))

            qr = QuadratureRule{RefLine}(2)
            ip = Lagrange{RefLine, 1}()^2
            dh = DofHandler(grid)
            add!(dh, :u, ip)
            close!(dh)

            u = collect(range(0, 1, ndofs(dh)))
            VTKGridFile(joinpath(tmp, "output"), dh, write_discontinuous = true) do vtk
                write_solution(vtk, dh, u)
            end

            data = read_vtk(joinpath(tmp, "output") * ".vtu")
            test_vtk_grid(data, grid; discontinuous = true)
            # The vector dofs of each cell are ordered by node, 2D vectors are padded with zeros
            expected_u = reduce(hcat, [reshape(u[celldofs(dh, i)], 2, :); 0 0] for i in 1:getncells(grid))
            @test keys(data.point_data) == Set(["u"])
            @test data.point_data["u"] ≈ expected_u
        end
    end
end
