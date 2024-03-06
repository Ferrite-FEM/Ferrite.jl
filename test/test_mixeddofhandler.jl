# Some helper functions
function get_cellset(cell_type, cells)
    return Set(findall(c-> typeof(c) == cell_type, cells))
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

    ip = Lagrange{RefLine, 1}()

    dh = DofHandler(grid);
    sdh1 = SubDofHandler(dh, Set(3))
    add!(sdh1, :u, ip^2)
    add!(sdh1, :θ, ip)
    sdh2 = SubDofHandler(dh, Set((1,2)))
    add!(sdh2, :u, ip^2)
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
    dh = DofHandler(grid);
    sdh1 = SubDofHandler(dh, Set(1))
    add!(sdh1, :u, Lagrange{RefQuadrilateral, 1}())
    sdh2 = SubDofHandler(dh, Set(2))
    add!(sdh2, :u, Lagrange{RefTriangle, 1}())
    close!(dh)

    # THEN: we expect 5 dofs and dof 2 and 3 being shared
    @test ndofs(dh) == 5
    @test dh.cell_dofs == [1, 2, 3, 4, 3, 2, 5]
    @test celldofs(dh, 1) == [1, 2, 3, 4]
    @test celldofs(dh, 2) == [3, 2, 5]
end

function test_2d_error()
    grid = get_2d_grid()
    # the refshape of the field must be the same as the refshape of the elements it is added to
    dh = DofHandler(grid);
    # wrong refshape compared to cell
    sdh1 = SubDofHandler(dh, Set(1))
    @test_throws ErrorException add!(sdh1, :u, Lagrange{RefTriangle, 1}());
    sdh2 = SubDofHandler(dh, Set(2))
    @test_throws ErrorException add!(sdh2, :u, Lagrange{RefQuadrilateral, 1}())

    # all cells within a SubDofHandler should be of the same celltype
    @test_throws ErrorException SubDofHandler(dh, Set((1,2)))
end

function test_2d_vector()

    grid = get_2d_grid()
    ## vector field
    dh = DofHandler(grid)
    sdh1 = SubDofHandler(dh, Set(1))
    add!(sdh1, :u, Lagrange{RefQuadrilateral, 1}()^2)
    sdh2 = SubDofHandler(dh, Set(2))
    add!(sdh2, :u, Lagrange{RefTriangle, 1}()^2)
    close!(dh)

    # THEN: we expect 10 dofs and dof 3-6 being shared
    @test ndofs(dh) == 10
    @test celldofs(dh, 1) == [1, 2, 3, 4, 5, 6, 7, 8]
    @test celldofs(dh, 2) == [5, 6, 3, 4, 9, 10]

end

function test_2d_mixed_1_el()
    grid = get_2d_grid()
    ## mixed field of same order
    dh = DofHandler(grid);
    sdh1 = SubDofHandler(dh, Set(1))
    add!(sdh1, :u, Lagrange{RefQuadrilateral, 1}()^2)
    add!(sdh1, :p, Lagrange{RefQuadrilateral, 1}())
    close!(dh)

    # THEN: we expect 12 dofs
    @test ndofs(dh) == 12
    @test ndofs_per_cell(dh, 1) == 12
    @test celldofs(dh, 1) == [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
    
    @test Set(Ferrite.getfieldnames(dh)) == Set(Ferrite.getfieldnames(dh.subdofhandlers[1]))
end

function test_2d_mixed_2_el()

    grid = get_2d_grid()
    ## mixed field of same order
    dh = DofHandler(grid);
    sdh1 = SubDofHandler(dh, Set(1))
    add!(sdh1, :u, Lagrange{RefQuadrilateral, 1}()^2)
    add!(sdh1, :p, Lagrange{RefQuadrilateral, 1}())
    sdh2 = SubDofHandler(dh, Set(2))
    add!(sdh2, :u, Lagrange{RefTriangle, 1}()^2)
    add!(sdh2, :p, Lagrange{RefTriangle, 1}())
    close!(dh)

    # THEN: we expect 15 dofs
    @test ndofs(dh) == 15
    @test ndofs_per_cell(dh, 1) == 12
    @test ndofs_per_cell(dh.subdofhandlers[1]) == 12
    @test ndofs_per_cell(dh, 2) == 9
    @test ndofs_per_cell(dh.subdofhandlers[2]) == 9
    @test_throws ErrorException ndofs_per_cell(dh)
    @test_throws ErrorException Ferrite.nnodes_per_cell(grid)
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
    dh = DofHandler(grid);
    add!(dh, :u, Lagrange{RefTriangle, 2}()^2)
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
    dh = DofHandler(grid);
    add!(dh, :u, Lagrange{RefTetrahedron, 2}()^3)
    close!(dh)

    # reference using the regular DofHandler
    tet_grid = generate_grid(Tetrahedron, (1, 1,1))
    tet_dh = DofHandler(tet_grid)
    add!(tet_dh, :u, Lagrange{RefTetrahedron,2}()^3)
    close!(tet_dh)

    for i in 1:6
        @test celldofs(dh, i) == celldofs(tet_dh, i)
    end
end

function test_face_dofs_quad_tri()
    # quadratic quad and quadratic triangle
    grid = get_2d_grid()
    dh = DofHandler(grid);
    sdh1 = SubDofHandler(dh, Set(1))
    add!(sdh1, :u, Lagrange{RefQuadrilateral, 2}()^2)
    sdh2 = SubDofHandler(dh, Set(2))
    add!(sdh2, :u, Lagrange{RefTriangle, 2}()^2)
    close!(dh)

    # THEN:
    @test ndofs(dh) == 24
    @test celldofs(dh, 1) == [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18]
    @test celldofs(dh, 2) == [5, 6, 3, 4, 19, 20, 11, 12, 21, 22, 23, 24]
end

function test_serendipity_quad_tri()
    # bi-quadratic quad (Serendipity) and quadratic triangle
    grid = get_2d_grid()
    interpolation = Serendipity{RefQuadrilateral, 2}()
    dh = DofHandler(grid);
    sdh1 = SubDofHandler(dh, Set(1))
    add!(sdh1, :u, interpolation^2)
    sdh2 = SubDofHandler(dh, Set(2))
    add!(sdh2, :u, Lagrange{RefTriangle,2}()^2)
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
    dh = DofHandler(grid);
    add!(dh, :u, Lagrange{RefTriangle, 2}()^2)
    add!(dh, :p, Lagrange{RefTriangle, 1}())
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
    dh = DofHandler(grid);
    sdh_quad = SubDofHandler(dh, Set(1))
    add!(sdh_quad, :u, Lagrange{RefQuadrilateral, 2}()^2)
    add!(sdh_quad, :p, Lagrange{RefQuadrilateral, 1}())
    sdh_tri = SubDofHandler(dh, Set(2))
    add!(sdh_tri, :u, Lagrange{RefTriangle, 2}()^2)
    add!(sdh_tri, :p, Lagrange{RefTriangle, 1}())
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
    nodes = [Node(coord) for coord in zeros(Vec{3,Float64}, 10)]
    grid = Grid(cells, nodes)

    dh = DofHandler(grid);
    # E.g. 3d continuum el -> 3dofs/node
    sdh_hex = SubDofHandler(dh, Set(1))
    add!(sdh_hex, :u, Lagrange{RefHexahedron, 1}()^3)
    # E.g. displacement field + rotation field -> 6 dofs/node
    sdh_quad = SubDofHandler(dh, Set(2))
    add!(sdh_quad, :u, Lagrange{RefQuadrilateral, 1}()^3)
    add!(sdh_quad, :θ, Lagrange{RefQuadrilateral, 1}()^3)
    close!(dh)

    @test ndofs(dh) == 42
    @test celldofs(dh, 1) == collect(1:24)
    @test celldofs(dh, 2) == [7, 8, 9, 4, 5, 6, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42]
end

function test_2_element_heat_eq()
    # Regression test solving the heat equation.
    # The grid consists of two quad elements treated as two different cell types to test the Grid and all other necessary functions

    grid = generate_grid(Quadrilateral, (2, 1))

    dh = DofHandler(grid);
    sdh1 = SubDofHandler(dh, Set(1))
    add!(sdh1, :u, Lagrange{RefQuadrilateral, 1}())
    sdh2 = SubDofHandler(dh, Set(2))
    add!(sdh2, :u, Lagrange{RefQuadrilateral, 1}())
    close!(dh)

    # Create two Dirichlet boundary conditions - one for each field.
    ch = ConstraintHandler(dh);
    ∂Ω1 = getfaceset(grid, "left")
    ∂Ω2 = getfaceset(grid, "right")
    dbc1 = Dirichlet(:u, ∂Ω1, (x, t) -> 0)
    dbc2 = Dirichlet(:u, ∂Ω2, (x, t) -> 0)
    add!(ch, dbc1);
    add!(ch, dbc2);
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

    K = create_matrix(dh)
    f = zeros(ndofs(dh));
    assembler = start_assemble(K, f);
    # Use the same assemble function since it is the same weak form for both cell-types
    for sdh in dh.subdofhandlers
        qr = QuadratureRule{RefQuadrilateral}(2)
        interp = sdh.field_interpolations[1]
        cellvalues = CellValues(qr, interp)
        doassemble(sdh.cellset, cellvalues, assembler, dh)
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
    @test sha in ("e96732c000b0b385db7444f002461468b60b3b2c", "7b26edc27b5e59a2f60907374cd5a5790cc37a6a")

end


function test_element_order()
    # Check that one can have non-contiguous ordering of cells in a Grid
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

    dh = DofHandler(grid);
    # Note the jump in cell numbers
    sdh_tri = SubDofHandler(dh, Set((1,3)))
    add!(sdh_tri, :u, Lagrange{RefTriangle,1}()^2)
    sdh_quad = SubDofHandler(dh, Set(2))
    add!(sdh_quad, :u, Lagrange{RefQuadrilateral,1}()^2)
    # Dofs are first created for cell 1 and 3, thereafter cell 2
    close!(dh)

    @test ndofs(dh) == 12
    @test celldofs(dh, 1) == collect(1:6)
    @test celldofs(dh, 2) == [3, 4, 7, 8, 11, 12, 5, 6]
    @test celldofs(dh, 3) == [7,8, 9, 10, 11, 12]
end

function test_field_on_subdomain()
    grid = get_2d_grid() # cell 1: quad, cell2: triangle
    dh = DofHandler(grid)

    # assume two fields: a scalar field :s and a vector field :v
    # :v lives on both cells, :s lives only on the triangle
    ip_tri = Lagrange{RefTriangle,1}()
    ip_quad = Lagrange{RefQuadrilateral,1}()

    sdh_quad = SubDofHandler(dh, Set(1))
    add!(sdh_quad, :v, ip_quad^2)
    sdh_tri = SubDofHandler(dh, Set(2))
    add!(sdh_tri, :v, ip_tri^2)
    add!(sdh_tri, :s, ip_tri)

    close!(dh)

    # retrieve field dimensions
    @test Ferrite.getfielddim(dh, :v) == 2
    @test Ferrite.getfielddim(dh, :s) ==1

    # find field in SubDofHandler
    @test Ferrite.find_field(dh.subdofhandlers[1], :v) == 1
    @test Ferrite.find_field(dh.subdofhandlers[2], :v) == 1
    @test Ferrite.find_field(dh.subdofhandlers[2], :s) == 2
    @test_throws ErrorException Ferrite.find_field(dh.subdofhandlers[1], :s)
end

function test_evaluate_at_grid_nodes()

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

    ip_quad = Lagrange{RefQuadrilateral,1}()
    ip_tri = Lagrange{RefTriangle,1}()

    dh = DofHandler(mesh)
    sdh_tri = SubDofHandler(dh, getcellset(mesh, "tris"))
    add!(sdh_tri, :v, ip_tri^2)
    sdh_quad = SubDofhandler(dh, getcellset(mesh, "quads"))
    add!(sdh_quad, :v, ip_quad^2)
    add!(sdh_quad, :s, ip_quad) # scalar field :s only on quad
    close!(dh)

    u = collect(1.:16.)

    s_nodes = evaluate_at_grid_nodes(dh, u, :s)
    @test s_nodes[1:4] ≈ [13., 14., 16., 15.]
    @test all(isnan.(s_nodes[5:6]))
    v_nodes = evaluate_at_grid_nodes(dh, u, :v)
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

# regression tests for https://github.com/KristofferC/JuAFEM.jl/issues/315 
function test_subparametric_quad()
    #linear geometry
    grid = generate_grid(Quadrilateral, (1,1))
    ip      = Lagrange{RefQuadrilateral,2}()
    
    dh = DofHandler(grid)
    add!(dh, :u, ip^2)
    close!(dh)
    
    ch = ConstraintHandler(dh)
    dbc1 = Dirichlet(:u, getfaceset(grid, "left"), (x, t) -> 0.0, 2)
    add!(ch, dbc1)
    close!(ch)
    update!(ch, 1.0)
    @test getnbasefunctions(Ferrite.getfieldinterpolation(dh.subdofhandlers[1],1)) == 18 # algebraic nbasefunctions
    @test celldofs(dh, 1) == [i for i in 1:18]
end

function test_subparametric_triangle()
    #linear geometry
    grid = generate_grid(Triangle, (1,1))

    ip = Lagrange{RefTriangle,2}()
    
    dh = DofHandler(grid)
    add!(dh, :u, ip^2)
    close!(dh)
    
    ch = ConstraintHandler(dh)
    dbc1 = Dirichlet(:u, getfaceset(grid, "left"), (x, t) -> 0.0, 2)
    add!(ch, dbc1)
    close!(ch)
    update!(ch, 1.0)
    @test getnbasefunctions(Ferrite.getfieldinterpolation(dh.subdofhandlers[1],1)) == 12 # algebraic nbasefunctions
    @test celldofs(dh, 1) == [i for i in 1:12]
end

function test_celliterator_subdomain()
    for celltype in (Line, Quadrilateral, Hexahedron)
        ip = Ferrite.default_interpolation(celltype)
        dim = Ferrite.getdim(ip)
        grid = generate_grid(celltype, ntuple(i->i==1 ? 2 : 1, dim)) # 2 cells
        dh = DofHandler(grid)
        sdh = SubDofHandler(dh, Set(2)) # only cell 2, cell 1 is not part of dh
        add!(sdh, :u, ip)
        close!(dh)

        ci = CellIterator(sdh)
        reinit!(ci.cc, 2)
        @test celldofs(ci.cc) == collect(1:length(ci.cc.dofs))
    end
end

function test_separate_fields_on_separate_domains()
    # 5_______6
    # |\      | 
    # |   \   |
    # 3______\4
    # |       |
    # |       |
    # 1_______2 
    # Given: a vector field :q defined on the quad and a scalar field :t defined on the triangles
    nodes = [Node((0.0, 0.0)),
            Node((1.0, 0.0)),
            Node((0.0, 1.0)),
            Node((1.0, 1.0)),
            Node((0.0, 2.0)),
            Node((1.0, 2.0))]
    cells = Ferrite.AbstractCell[Quadrilateral((1,2,4,3)),
            Triangle((3,4,5)),
            Triangle((4,6,5))]
    mesh = Grid(cells, nodes)
    addcellset!(mesh, "quads", Set{Int}((1,)))
    addcellset!(mesh, "tris", Set{Int}((2, 3)))

    ip_tri = Lagrange{RefTriangle,1}()
    ip_quad = Lagrange{RefQuadrilateral,1}()

    dh = DofHandler(mesh)
    sdh_quad = SubDofHandler(dh, getcellset(mesh, "quads"))
    add!(sdh_quad, :q, ip_quad^2) # vector field :q only on quad

    sdh_tri = SubDofHandler(dh, getcellset(mesh, "tris"))
    add!(sdh_tri, :t, ip_tri)
    close!(dh)

    # Expect: 8 dofs for the quad and 4 new dofs for the triangles
    @test ndofs(dh) == 12
    @test celldofs(dh, 1) == [i for i in 1:8]
    @test celldofs(dh, 2) == [9, 10, 11]
    @test celldofs(dh, 3) == [10, 12, 11]

end

function test_unique_cellsets()
    grid = generate_grid(Quadrilateral, (2, 1))
    set_u = Set(1:2)
    set_v = Set(1:1)

    ip = Lagrange{RefQuadrilateral,1}()

    # bug
    dh = DofHandler(grid)
    sdh_u = SubDofHandler(dh, set_u)
    @test_throws ErrorException SubDofHandler(dh, set_v)
end

function test_show()
    # single SubDofHandler
    grid = generate_grid(Triangle, (1,1))
    dh = DofHandler(grid)
    add!(dh, :u, Lagrange{RefTriangle, 1}()^2)
    close!(dh)
    @test repr("text/plain", dh) == string(
        repr("text/plain", typeof(dh)), "\n  Fields:\n    :u, ",
        repr("text/plain", dh.subdofhandlers[1].field_interpolations[1]), "\n  Dofs per cell: 6\n  Total dofs: 8")

    # multiple SubDofHandlers
    grid = get_2d_grid()
    dh = DofHandler(grid);
    sdh_quad = SubDofHandler(dh, Set(1))
    add!(sdh_quad, :u, Lagrange{RefQuadrilateral, 1}()^2)
    sdh_tri = SubDofHandler(dh, Set(2))
    add!(sdh_tri, :u, Lagrange{RefTriangle, 1}()^2)
    close!(dh)
    @test repr("text/plain", dh) == repr(typeof(dh)) * "\n  Fields:\n    :u, dim: 2\n  Total dofs: 10"
    @test repr("text/plain", dh.subdofhandlers[1]) == string(
        repr("text/plain", typeof(dh.subdofhandlers[1])), "\n  Cell type: Quadrilateral\n  Fields:\n    :u, ",
            repr("text/plain", dh.subdofhandlers[1].field_interpolations[1]), "\n  Dofs per cell: 8\n")
end

function test_vtk_export()
    nodes = Node.([Vec(0.0, 0.0),
                   Vec(1.0, 0.0),
                   Vec(1.0, 1.0),
                   Vec(0.0, 1.0),
                   Vec(2.0, 0.0),
            ])
    cells = [
        Quadrilateral((1, 2, 3, 4)),
        Triangle((3, 2, 5))
        ]
    grid = Grid(cells, nodes)
    ip_tri = Lagrange{RefTriangle, 1}()
    ip_quad = Lagrange{RefQuadrilateral, 1}()
    dh = DofHandler(grid)
    sdh_quad = SubDofHandler(dh, Set(1))
    add!(sdh_quad, :u, ip_quad)
    sdh_tri = SubDofHandler(dh, Set(2))
    add!(sdh_tri, :u, ip_tri)
    close!(dh)
    u = collect(1:ndofs(dh))
    filename = "mixed_2d_grid"
    vtk_grid(filename, dh) do vtk
        vtk_point_data(vtk, dh, u)
    end
    sha = bytes2hex(open(SHA.sha1, filename*".vtu"))
    @test sha == "339ab8a8a613c2f38af684cccd695ae816671607"
    rm(filename*".vtu") # clean up 
end

@testset "DofHandler" begin
    test_1d_bar_beam();
    test_2d_scalar();
    test_2d_error();
    test_2d_vector();
    test_2d_mixed_1_el();
    test_2d_mixed_2_el();
    test_face_dofs_2_tri();
    test_face_dofs_quad_tri();
    test_3d_tetrahedrons();
    test_serendipity_quad_tri();
    test_2d_mixed_field_triangles();
    test_2d_mixed_field_mixed_celltypes();
    test_3d_mixed_field_mixed_celltypes();
    test_2_element_heat_eq();
    test_element_order();
    test_field_on_subdomain();
    test_mixed_grid_show();
    test_subparametric_quad();
    test_subparametric_triangle();
    # test_evaluate_at_grid_nodes()
    test_mixed_grid_show()
    test_separate_fields_on_separate_domains();
    test_unique_cellsets()
    test_celliterator_subdomain()
    test_show()
    test_vtk_export()
end
