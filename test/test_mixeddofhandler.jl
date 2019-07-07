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

    return MixedGrid(cells, Node{0, Float64}[])
end

# Tests
function test_1d_bar_beam()
    # Something like a truss and a Timoshenko beam.
    # two line-cells with 2 dofs/node -> "bars"
    # one line-cell with a mixed field: one vector field and one scalar -> "beam"
    cells = [
        Line((1, 2)),
        Line((2, 3)),
        Line((1, 3)),
        ]
    grid = MixedGrid(cells, Node{0, Float64}[])

    field1 = create_field(name=:u, spatial_dim=1, field_dim=2, order=1, cellshape=RefCube)
    field2 = create_field(name=:u, spatial_dim=1, field_dim=2, order=1, cellshape=RefCube)
    field3 = create_field(name=:θ, spatial_dim=1, field_dim=1, order=1, cellshape=RefCube)

    dh = MixedDofHandler(grid);
    push!(dh, FieldHandler([field2, field3], Set(3)));
    push!(dh, FieldHandler([field1], Set((1, 2))));
    close!(dh)
    @test ndofs(dh) == 8
    @test celldofs(dh, 1) == collect(1:6)
    @test celldofs(dh, 2) == [1, 2, 7, 8]
    @test celldofs(dh, 3) == [7, 8, 3, 4]

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
    @test dh.cell_dofs == [1, 2, 3, 4, 3, 2, 5]
    @test celldofs(dh, 1) == [1, 2, 3, 4]
    @test celldofs(dh, 2) == [3, 2, 5]
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
    @test dh.cell_dofs == [celldofs(dh, 1)..., celldofs(dh, 2)...]
end

function test_2d_mixed_1_el()
    grid = get_2d_grid()
    ## mixed field of same order
    field1 = create_field(name = :u, spatial_dim=2, field_dim = 2, order = 1, cellshape = RefCube)
    field2 = create_field(name = :p, spatial_dim=2, field_dim = 1, order = 1, cellshape = RefTetrahedron)
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
    field1 = create_field(name = :u, spatial_dim=2, field_dim = 2, order = 1, cellshape = RefCube)
    field2 = create_field(name = :p, spatial_dim=2, field_dim = 1, order = 1, cellshape = RefTetrahedron)
    dh = MixedDofHandler(grid);
    push!(dh, FieldHandler([field1, field2], Set(1)));
    push!(dh, FieldHandler([field1, field2], Set(2)));
    close!(dh)

    # THEN: we expect 15 dofs
    @test ndofs(dh) == 15
    @test ndofs_per_cell(dh, 1) == 12
    @test ndofs_per_cell(dh, 2) == 9
    @test celldofs(dh, 1) == [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
    @test celldofs(dh, 2) == [5, 6, 3, 4, 13, 14, 11, 10, 15]
end

function test_2d_mixed_2_tri()
    cells = [
        Triangle((1, 2, 3)),
        Triangle((2, 4, 3))
        ]
    grid = MixedGrid(cells, Node{0, Float64}[])
    field1 = create_field(name = :u, spatial_dim=2, field_dim = 2, order = 1, cellshape = RefCube)
    field2 = create_field(name = :p, spatial_dim=2, field_dim = 1, order = 1, cellshape = RefTetrahedron)
    dh = MixedDofHandler(grid);
    push!(dh, FieldHandler([field1, field2], Set(1)));
    push!(dh, FieldHandler([field1, field2], Set(2)));
    close!(dh)
    @test ndofs(dh) == 12
    @test ndofs_per_cell(dh, 1) == 9
    @test ndofs_per_cell(dh, 2) == 9
    @test celldofs(dh, 1) == [1, 2, 3, 4, 5, 6, 7, 8, 9]
    @test celldofs(dh, 2) == [3, 4, 10, 11, 5, 6, 8, 12, 9]

    # reference
    # grid = generate_grid(Triangle, (1, 1))
    # dh = DofHandler(grid)
    # push!(dh, :u, 2, Lagrange{2,RefTetrahedron,1}())
    # push!(dh, :p, 1, Lagrange{2,RefTetrahedron,1}())
    # close!(dh)
    # celldofs(dh, 1)
    # celldofs(dh, 2)
end

function test_face_dofs_2_tri()
    cells = [
        Triangle((1, 2, 3)),
        Triangle((2, 4, 3))
        ]
    grid = MixedGrid(cells, Node{0, Float64}[]);
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

    grid = MixedGrid(cells, Node{0, Float64}[])
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

    grid = MixedGrid(cells, Node{0, Float64}[])
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
    grid = MixedGrid(cells, Node{0, Float64}[])

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



@testset "MixedDofHandler" begin

    test_1d_bar_beam();
    test_2d_scalar();
    test_2d_vector();
    test_2d_mixed_1_el();
    test_2d_mixed_2_el();
    test_2d_mixed_2_tri();
    test_face_dofs_2_tri();
    test_face_dofs_quad_tri();
    test_serendipity_quad_tri();
    test_3d_tetrahedrons();
    test_2d_mixed_field_triangles();
    test_2d_mixed_field_mixed_celltypes();
    test_3d_mixed_field_mixed_celltypes();
end
