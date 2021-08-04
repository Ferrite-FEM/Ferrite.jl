# Some helper functions
function get_cellset(cell_type, cells)
    return Set(findall(c-> typeof(c) == cell_type, cells))
end

function create_field(;name, field_dim, order, spatial_dim, cellshape)
    interpolation = Lagrange{spatial_dim, cellshape, order}()
    return Field(name, interpolation, field_dim)
end

function get_2d_grid()
    # GIVEN: two cells, a quad and a triangle sharing one face
    cells = [
        Quadrilateral((1, 2, 3, 4)),
        Triangle((3, 2, 5))
        ]
    coords = zeros(Vec{2,Float64}, 5)
    nodes = [Node(coord) for coord in zeros(Vec{2,Float64}, 5)]
    return Grid(cells,nodes)
end

function test_1d_bar_beam()
    # Something like a truss and a Timoshenko beam.
    # two line-cells with 2 dofs/node -> "bars"
    # one line-cell with a mixed field: one vector field and one scalar -> "beam"
    cells = [
        Line((1, 2)),
        Line((2, 3)),
        Line((1, 3)),
        ]
    nodes = [Node(coord) for coord in zeros(Vec{2,Float64}, 3)]
    grid = Grid(cells, nodes)

    field1 = create_field(name=:u, spatial_dim=1, field_dim=2, order=1, cellshape=RefCube)
    field2 = create_field(name=:u, spatial_dim=1, field_dim=2, order=1, cellshape=RefCube)
    field3 = create_field(name=:θ, spatial_dim=1, field_dim=1, order=1, cellshape=RefCube)

    dh = MixedDofHandler(grid);
    push!(dh, FieldHandler([field2, field3], Set(3)));
    push!(dh, FieldHandler([field1], Set((1, 2))));
    close!(dh)
    @test ndofs(dh) == 8
    @test celldofs(dh, 3) == collect(1:6)
    @test celldofs(dh, 1) == [1, 2, 7, 8]
    @test celldofs(dh, 2) == [7, 8, 3, 4]

    #        7,8
    #       /   \
    #     /      \
    #   /_________\
    # 1,2,5       3,4,6
end

function test_2d_scalar()

    grid = get_2d_grid()
    # WHEN: adding a scalar field for each cell and generating dofs
    field1 = create_field(name=:u, spatial_dim=2, field_dim=1, order=1, cellshape=RefCube)
    field2 = create_field(name=:u, spatial_dim=2, field_dim=1, order=1, cellshape=RefTetrahedron)
    dh = MixedDofHandler(grid);
    push!(dh, FieldHandler([field1], Set(1)));
    push!(dh, FieldHandler([field2], Set(2)));
    close!(dh)

    # THEN: we expect 5 dofs and dof 2 and 3 being shared
    @test ndofs(dh) == 5
    @test dh.cell_dofs.values == [1, 2, 3, 4, 3, 2, 5]
    @test celldofs(dh, 1) == [1, 2, 3, 4]
    @test celldofs(dh, 2) == [3, 2, 5]
end

function test_2d_error()

    grid = get_2d_grid()
    # the refshape of the field must be the same as the refshape of the elements it is added to
    field1 = create_field(name=:u, spatial_dim=2, field_dim=1, order=1, cellshape=RefTetrahedron)
    field2 = create_field(name=:u, spatial_dim=2, field_dim=1, order=1, cellshape=RefCube)
    dh = MixedDofHandler(grid);
    @test_throws ErrorException push!(dh, FieldHandler([field1], Set(1)));
    @test_throws ErrorException push!(dh, FieldHandler([field2], Set(2)));
    # all cells within a FieldHandler should be of the same celltype
    @test_throws ErrorException push!(dh, FieldHandler([field1], Set([1,2])));

end

function test_2d_vector()

    grid = get_2d_grid()
    ## vector field
    field1 = create_field(name = :u, spatial_dim=2, field_dim = 2, order = 1, cellshape = RefCube)
    field2 = create_field(name = :u, spatial_dim=2, field_dim = 2, order = 1, cellshape = RefTetrahedron)
    dh = MixedDofHandler(grid);
    push!(dh, FieldHandler([field1], Set(1)));
    push!(dh, FieldHandler([field2], Set(2)));
    close!(dh)

    # THEN: we expect 10 dofs and dof 3-6 being shared
    @test ndofs(dh) == 10
    @test celldofs(dh, 1) == [1, 2, 3, 4, 5, 6, 7, 8]
    @test celldofs(dh, 2) == [5, 6, 3, 4, 9, 10]

end

function test_2d_mixed_1_el()
    grid = get_2d_grid()
    ## mixed field of same order
    field1 = create_field(name = :u, spatial_dim=2, field_dim = 2, order = 1, cellshape = RefCube)
    field2 = create_field(name = :p, spatial_dim=2, field_dim = 1, order = 1, cellshape = RefCube)
    dh = MixedDofHandler(grid);
    push!(dh, FieldHandler([field1, field2], Set(1)));
    close!(dh)

    # THEN: we expect 12 dofs
    @test ndofs(dh) == 12
    @test ndofs_per_cell(dh, 1) == 12
    @test celldofs(dh, 1) == [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
end

function test_2d_mixed_2_el()

    grid = get_2d_grid()
    ## mixed field of same order
    field1_quad = create_field(name = :u, spatial_dim=2, field_dim = 2, order = 1, cellshape = RefCube)
    field2_quad = create_field(name = :p, spatial_dim=2, field_dim = 1, order = 1, cellshape = RefCube)
    field1_tri = create_field(name = :u, spatial_dim=2, field_dim = 2, order = 1, cellshape = RefTetrahedron)
    field2_tri = create_field(name = :p, spatial_dim=2, field_dim = 1, order = 1, cellshape = RefTetrahedron)
    dh = MixedDofHandler(grid);
    push!(dh, FieldHandler([field1_quad, field2_quad], Set(1)));
    push!(dh, FieldHandler([field1_tri, field2_tri], Set(2)));
    close!(dh)

    # THEN: we expect 15 dofs
    @test ndofs(dh) == 15
    @test ndofs_per_cell(dh, 1) == 12
    @test ndofs_per_cell(dh, 2) == 9
    @test celldofs(dh, 1) == [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
    @test celldofs(dh, 2) == [5, 6, 3, 4, 13, 14, 11, 10, 15]
end

function test_face_dofs_2_tri()
    cells = [
        Triangle((1, 2, 3)),
        Triangle((2, 4, 3))
        ]
    nodes = [Node(coord) for coord in zeros(Vec{2,Float64}, 4)]
    grid = Grid(cells, nodes);
    field1 = create_field(name = :u, spatial_dim = 2, field_dim = 2, order = 2, cellshape = RefTetrahedron)
    dh = MixedDofHandler(grid);
    push!(dh, FieldHandler([field1], Set((1, 2))));
    #push!(dh, FieldHandler([field2], Set(2)));
    close!(dh)

    # THEN:
    @test ndofs(dh) == 18
    @test celldofs(dh, 1) == [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
    @test celldofs(dh, 2) == [3, 4, 13, 14, 5, 6, 15, 16, 17, 18, 9, 10]
end

function test_3d_tetrahedrons()
    cells = [
        Tetrahedron((1, 2, 3, 7)),
        Tetrahedron((1, 5, 2, 7)),
        Tetrahedron((2, 4, 3, 7)),
        Tetrahedron((2, 8, 4, 7)),
        Tetrahedron((2, 5, 6, 7)),
        Tetrahedron((2, 6, 8, 7)),
        ]
    nodes = [Node(coord) for coord in zeros(Vec{3,Float64}, 8)]
    grid = Grid(cells, nodes)
    field = create_field(name = :u, spatial_dim=3,  field_dim = 3, order = 2, cellshape = RefTetrahedron)
    dh = MixedDofHandler(grid);
    push!(dh, FieldHandler([field], Set((1, 2, 3, 4, 5, 6))));
    close!(dh)

    # reference using the regular DofHandler
    tet_grid = generate_grid(Tetrahedron, (1, 1,1))
    tet_dh = DofHandler(tet_grid)
    push!(tet_dh, :u, 3, Lagrange{3,RefTetrahedron,2}())
    close!(tet_dh)

    for i in 1:6
        @test celldofs(dh, i) == celldofs(tet_dh, i)
    end
end

function test_face_dofs_quad_tri()
    # quadratic quad and quadratic triangle
    grid = get_2d_grid()
    field1 = create_field(name = :u, spatial_dim = 2, field_dim = 2, order = 2, cellshape = RefCube)
    field2 = create_field(name = :u, spatial_dim = 2, field_dim = 2, order = 2, cellshape = RefTetrahedron)
    dh = MixedDofHandler(grid);
    push!(dh, FieldHandler([field1], Set(1)));
    push!(dh, FieldHandler([field2], Set(2)));
    close!(dh)

    # THEN:
    @test ndofs(dh) == 24
    @test celldofs(dh, 1) == [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18]
    @test celldofs(dh, 2) == [5, 6, 3, 4, 19, 20, 11, 12, 21, 22, 23, 24]
end

function test_serendipity_quad_tri()
    # bi-quadratic quad (Serendipity) and quadratic triangle
    grid = get_2d_grid()
    interpolation = Serendipity{2, RefCube, 2}()
    field1 = Field(:u, interpolation, 2)
    field2 = create_field(name = :u, spatial_dim = 2, field_dim = 2, order = 2, cellshape = RefTetrahedron)
    dh = MixedDofHandler(grid);
    push!(dh, FieldHandler([field1], Set(1)));
    push!(dh, FieldHandler([field2], Set(2)));
    close!(dh)

    # THEN:
    @test ndofs(dh) == 22
    @test celldofs(dh, 1) == [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]
    @test celldofs(dh, 2) == [5, 6, 3, 4, 17, 18, 11, 12, 19, 20, 21, 22]
end

function test_2d_mixed_field_triangles()
    # Mixed field (Taylor-Hood)
    # quadratic vector field + linear scalar field on
    # celltypes: 2 Triangles
    cells = [
        Triangle((1, 2, 3)),
        Triangle((2, 4, 3))
        ]
    nodes = [Node(coord) for coord in zeros(Vec{2,Float64}, 4)]
    grid = Grid(cells, nodes)
    field1 = create_field(name=:u, spatial_dim=2, field_dim=2, order=2, cellshape=RefTetrahedron)
    field2 = create_field(name=:p, spatial_dim=2, field_dim=1, order=1, cellshape=RefTetrahedron)
    dh = MixedDofHandler(grid);
    push!(dh, FieldHandler([field1, field2], Set((1, 2))));
    close!(dh)
    @test ndofs(dh) == 22
    @test celldofs(dh, 1) == collect(1:15)
    @test celldofs(dh, 2) == [3, 4, 16, 17, 5, 6, 18, 19, 20, 21, 9, 10, 14, 22, 15]
end

function test_2d_mixed_field_mixed_celltypes()
    # Mixed field (Taylor-Hood)
    # quadratic vector field + linear scalar field on
    # celltypes: 1 Quadrilateral and 1 Triangle

    grid = get_2d_grid()
    field1 = create_field(name=:u, spatial_dim=2, field_dim=2, order=2, cellshape=RefCube)
    field2 = create_field(name=:p, spatial_dim=2, field_dim=1, order=1, cellshape=RefCube)
    field3 = create_field(name=:u, spatial_dim=2, field_dim=2, order=2, cellshape=RefTetrahedron)
    field4 = create_field(name=:p, spatial_dim=2, field_dim=1, order=1, cellshape=RefTetrahedron)
    dh = MixedDofHandler(grid);
    push!(dh, FieldHandler([field1, field2], Set(1)));
    push!(dh, FieldHandler([field3, field4], Set(2)));
    close!(dh)
    @test ndofs(dh) == 29
    @test celldofs(dh, 1) == collect(1:22)
    @test celldofs(dh, 2) == [5, 6, 3, 4, 23, 24, 11, 12, 25, 26, 27, 28, 21, 20, 29]
end

function test_3d_mixed_field_mixed_celltypes()
    # Mixed celltypes in 3d
    # one hex-cell with a vector field (3 dofs/node)
    # one quad-cell (connected along bottom edge of the hex) with two mixed vector fields (3+3 dofs/node)
    # One field type is common for both cells
    cells = [
        Hexahedron((1, 2, 3, 4, 5, 6, 7, 8)),
        Quadrilateral((3, 2, 9, 10)),
        ]
    nodes = [Node(coord) for coord in zeros(Vec{2,Float64}, 10)]
    grid = Grid(cells, nodes)

    # E.g. 3d continuum el -> 3dofs/node
    field1 = create_field(name=:u, spatial_dim=3, field_dim=3, order=1, cellshape=RefCube)
    # E.g. displacement field + rotation field -> 6 dofs/node
    field2 = create_field(name=:u, spatial_dim=3, field_dim=3, order=1, cellshape=RefCube)
    field3 = create_field(name=:θ, spatial_dim=3, field_dim=3, order=1, cellshape=RefCube)

    dh = MixedDofHandler(grid);
    push!(dh, FieldHandler([field1], Set(1)));
    push!(dh, FieldHandler([field2, field3], Set(2)));
    close!(dh)
    @test ndofs(dh) == 42
    @test celldofs(dh, 1) == collect(1:24)
    @test celldofs(dh, 2) == [7, 8, 9, 4, 5, 6, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42]
end

function test_2_element_heat_eq()
    # Regression test solving the heat equation.
    # The grid consists of two quad elements treated as two different cell types to test the Grid and all other necessary functions

    temp_grid = generate_grid(Quadrilateral, (2, 1))
    grid = Grid(temp_grid.cells, temp_grid.nodes)  # regular grid -> mixed grid
    grid.facesets = temp_grid.facesets;

    # Create two identical fields
    f1 = create_field(name=:u, spatial_dim=2, field_dim=1, order=1, cellshape=RefCube)
    f2 = create_field(name=:u, spatial_dim=2, field_dim=1, order=1, cellshape=RefCube)

    dh = MixedDofHandler(grid);
    push!(dh, FieldHandler([f1], Set(1)));  # first field applies to cell 1
    push!(dh, FieldHandler([f2], Set(2)));  # second field applies to cell 2
    close!(dh)

    # Create two Dirichlet boundary conditions - one for each field.
    ch = ConstraintHandler(dh);
    ∂Ω1 = getfaceset(grid, "left")
    ∂Ω2 = getfaceset(grid, "right")
    dbc1 = Dirichlet(:u, ∂Ω1, (x, t) -> 0)
    dbc2 = Dirichlet(:u, ∂Ω2, (x, t) -> 0)
    add!(ch, dh.fieldhandlers[1], dbc1);
    add!(ch, dh.fieldhandlers[2], dbc2);
    close!(ch)

    function doassemble(cellset, cellvalues, assembler, dh)

        n = ndofs_per_cell(dh, first(cellset))
        n_basefuncs = getnbasefunctions(cellvalues)
        Ke = zeros(n, n)
        fe = zeros(n)
        eldofs = zeros(Int, n)

        for cellnum in cellset
            celldofs!(eldofs, dh, cellnum)
            xe = getcoordinates(dh.grid, cellnum)
            reinit!(cellvalues, xe)
            fill!(Ke, 0)
            fill!(fe, 0)

            for q_point in 1:getnquadpoints(cellvalues)
                dΩ = getdetJdV(cellvalues, q_point)

                for i in 1:n_basefuncs
                    v  = shape_value(cellvalues, q_point, i)
                    ∇v = shape_gradient(cellvalues, q_point, i)
                    fe[i] += v * dΩ
                    for j in 1:n_basefuncs
                        ∇u = shape_gradient(cellvalues, q_point, j)
                        Ke[i, j] += (∇v ⋅ ∇u) * dΩ
                    end
                end
            end
            assemble!(assembler, eldofs, fe, Ke)
        end
    end

    K = create_sparsity_pattern(dh)
    f = zeros(ndofs(dh));
    assembler = start_assemble(K, f);
    # Use the same assemble function since it is the same weak form for both cell-types
    for fh in dh.fieldhandlers
        qr = QuadratureRule{2, RefCube}(2)
        interp = fh.fields[1].interpolation
        cellvalues = CellScalarValues(qr, interp)
        doassemble(fh.cellset, cellvalues, assembler, dh)
    end

    update!(ch, 0.0);
    apply!(K, f, ch)
    u = K \ f;

    # tested against heat_equation.jl (in the examples folder) using 2x1 cells and no
    # dbc on top and bottom boundary
    @test u == [0.0, 0.5, 0.5, 0.0, 0.0, 0.0]

    gridfilename = "mixed_grid"
    addcellset!(grid, "cell-1", [1,])
    addcellset!(grid, "cell-2", [2,])
    vtk_grid(gridfilename, grid) do vtk
        vtk_cellset(vtk, grid, "cell-1")
        vtk_cellset(vtk, grid, "cell-2")
        vtk_point_data(vtk, dh, u)
        # vtk_point_data(vtk, ch)  #FIXME
    end
    sha = bytes2hex(open(SHA.sha1, gridfilename*".vtu"))
    @test sha == "e96732c000b0b385db7444f002461468b60b3b2c"

end


function test_element_order()
    # Check that one can have non-contigous ordering of cells in a Grid
    # Something like this:
    #        ______
    #      /|     |\
    # (1) / | (2) | \ (3)
    #    /__|_____|__\
    cells = [
        Triangle((1, 2, 3)),
        Quadrilateral((2, 4, 5, 3)),
        Triangle((4, 6, 5))
        ]
    nodes = [Node(coord) for coord in zeros(Vec{2,Float64}, 6)]
    grid = Grid(cells, nodes)
    field1_tri = create_field(name=:u, spatial_dim=3, field_dim=2, order=1, cellshape=RefTetrahedron)
    field1_quad = create_field(name=:u, spatial_dim=3, field_dim=2, order=1, cellshape=RefCube)

    dh = MixedDofHandler(grid);
    # Note the jump in cell numbers
    push!(dh, FieldHandler([field1_tri], Set((1, 3))));
    push!(dh, FieldHandler([field1_quad], Set(2)));
    # Dofs are first created for cell 1 and 3, thereafter cell 2
    close!(dh)

    @test ndofs(dh) == 12
    @test celldofs(dh, 1) == collect(1:6)
    @test celldofs(dh, 2) == [3, 4, 7, 8, 11, 12, 5, 6]
    @test celldofs(dh, 3) == [7,8, 9, 10, 11, 12]

end

function test_field_on_subdomain()
    grid = get_2d_grid() # cell 1: quad, cell2: triangle
    dh = MixedDofHandler(grid)

    # assume two fields: a scalar field :s and a vector field :v
    # :v lives on both cells, :s lives only on the triangle
    ip_tri = Lagrange{2,RefTetrahedron,1}()
    ip_quad = Lagrange{2,RefCube,1}()
    v_tri = Field(:v, ip_tri, 2)
    v_quad = Field(:v, ip_quad, 2)
    s = Field(:s, ip_tri, 1)

    push!(dh, FieldHandler([v_quad], Set((1,))))
    push!(dh, FieldHandler([v_tri, s], Set((2,))))

    close!(dh)

    # retrieve field dimensions
    @test Ferrite.getfielddim(dh, :v) == 2
    @test Ferrite.getfielddim(dh, :s) ==1

    # find field in FieldHandler
    @test Ferrite.find_field(dh.fieldhandlers[1], :v) == 1
    @test Ferrite.find_field(dh.fieldhandlers[2], :v) == 1
    @test Ferrite.find_field(dh.fieldhandlers[2], :s) == 2
    @test_throws ErrorException Ferrite.find_field(dh.fieldhandlers[1], :s)
end

function test_reshape_to_nodes()

    # 5_______6
    # |\      | 
    # |   \   |
    # 3______\4
    # |       |
    # |       |
    # 1_______2 

    nodes = [Node((0.0, 0.0)),
    Node((1.0, 0.0)),
    Node((0.0, 1.0)),
    Node((1.0, 1.0)),
    Node((0.0, 2.0)),
    Node((1.0, 2.0))]
    cells = Ferrite.AbstractCell[Quadrilateral((1,2,4,3)),
    Triangle((3,4,6)),
    Triangle((3,6,5))]
    mesh = Grid(cells, nodes)
    addcellset!(mesh, "quads", Set{Int}((1,)))
    addcellset!(mesh, "tris", Set{Int}((2, 3)))

    ip_quad = Lagrange{2,RefCube,1}()
    ip_tri = Lagrange{2,RefTetrahedron,1}()

    dh = MixedDofHandler(mesh)
    field_v_tri = Field(:v, ip_tri, 2) # vector field :v everywhere
    fh_tri = FieldHandler([field_v_tri], getcellset(mesh, "tris"))
    push!(dh, fh_tri)
    field_v_quad = Field(:v, ip_quad, 2)
    field_s_quad = Field(:s, ip_quad, 1) # scalar field :s only on quad
    fh_quad = FieldHandler([field_v_quad, field_s_quad], getcellset(mesh, "quads"))
    push!(dh, fh_quad)
    close!(dh)

    u = collect(1.:16.)

    s_nodes = reshape_to_nodes(dh, u, :s)
    @test s_nodes[1:4] ≈ [13., 14., 16., 15.]
    @test all(isnan.(s_nodes[5:6]))
    v_nodes = reshape_to_nodes(dh, u, :v)
    @test v_nodes ≈ hcat(   [9., 10., 0.],
                    [11., 12., 0.],
                    [1., 2., 0.],
                    [3., 4., 0.],
                    [7., 8., 0.],
                    [5., 6., 0.])
end

function test_mixed_grid_show()
    grid = get_2d_grid()
    str = sprint(show, MIME("text/plain"), grid)
    @test occursin("2 Quadrilateral/Triangle cells", str)
end

@testset "MixedDofHandler" begin

    test_1d_bar_beam();
    test_2d_scalar();
    test_2d_error();
    test_2d_vector();
    test_2d_mixed_1_el();
    test_2d_mixed_2_el();
    test_face_dofs_2_tri();
    test_face_dofs_quad_tri();
    test_serendipity_quad_tri();
    test_3d_tetrahedrons();
    test_2d_mixed_field_triangles();
    test_2d_mixed_field_mixed_celltypes();
    test_3d_mixed_field_mixed_celltypes();
    test_2_element_heat_eq();
    test_element_order();
    test_field_on_subdomain();
    test_reshape_to_nodes()
    test_mixed_grid_show()
end