function import_inp(filepath_with_ext)
    problem = extract_inp(filepath_with_ext)
    return inp_to_juafem(problem)
end

function inp_to_juafem(problem::InpContent)
    _celltype = problem.celltype
    if _celltype == "CPS3"
        # Linear triangle
        celltype = Triangle
        geom_order = 1
        refshape = RefTetrahedron
        dim = 2
    elseif _celltype == "CPS6"
        # Quadratic triangle
        celltype = QuadraticTriangle
        geom_order = 2
        refshape = RefTetrahedron
        dim = 2
    elseif _celltype == "C3D4"
        # Linear tetrahedron
        celltype = Tetrahedron
        geom_order = 1
        refshape = RefTetrahedron
        dim = 3
    elseif _celltype == "C3D10"
        # Quadratic tetrahedron
        celltype = QuadraticTetrahedron
        geom_order = 2
        refshape = RefTetrahedron
        dim = 3    
    elseif _celltype == "CPS4"
        # Linear quadrilateral
        celltype = Quadrilateral
        geom_order = 1
        refshape = RefCube
        dim = 2
    elseif _celltype == "CPS8" || _celltype == "CPS8R"
        # Quadratic quadrilateral
        celltype = QuadraticQuadrilateral
        geom_order = 2
        refshape = RefCube
        dim = 2
    elseif _celltype == "C3D8" || _celltype == "C3D8R"
        # Linear hexahedron
        celltype = Hexahedron
        geom_order = 1
        refshape = RefCube
        dim = 3
    elseif _celltype == "C3D20" || _celltype == "C3D20R"
        # Quadratic hexahedron
        celltype = QuadraticHexahedron
        geom_order = 2
        refshape = RefCube
        dim = 3
    #elseif _celltype == "C3D6"
        # Linear wedge
    #elseif _celltype == "C3D15"
        # Quadratic wedge
    else
        throw("Unsupported cell type $_celltype.")
    end
    cells = celltype.(problem.cells)
    nodes = Node.(problem.node_coords)
    grid = Grid(cells, nodes)

    for k in keys(problem.cellsets)
        grid.cellsets[k] = Set(problem.cellsets[k])
    end
    for k in keys(problem.nodesets)
        grid.nodesets[k] = Set(problem.nodesets[k])
    end
    for k in keys(problem.facesets)
        grid.facesets[k] = Set(problem.facesets[k])
    end
    # Define boundary faces
    grid.boundary_matrix = extract_boundary_matrix(grid);

    dh = DofHandler(grid)
    # Isoparametric
    field_interpolation = Lagrange{dim, refshape, geom_order}()
    push!(dh, :u, dim, field_interpolation)
    close!(dh)

    ch = ConstraintHandler(dh)
    for k in keys(problem.nodedbcs)
        vec = problem.nodedbcs[k]
        f(x, t) = [vec[i][2] for i in 1:length(vec)]
        components = [vec[i][1] for i in 1:length(vec)]
        dbc = Dirichlet(:u, getnodeset(grid, k), f, components)
        add!(ch, dbc)
        close!(ch)
        update!(ch, 0.0)
    end

    return ch
end

function extract_boundary_matrix(grid::Grid{dim}) where dim
    nfaces = length(faces(grid.cells[1]))
    ncells = length(grid.cells)
    countedbefore = Dict{NTuple{dim,Int},Bool}()
    boundary_matrix = ones(Bool, nfaces, ncells) # Assume all are boundary faces
    for (ci, cell) in enumerate(getcells(grid))    
        for (fi, face) in enumerate(faces(cell))
            sface = sortface(face) # TODO: faces(cell) may as well just return the sorted list
            token = ht_keyindex2!(countedbefore, sface)
            if token > 0 # haskey(countedbefore, sface)
                boundary_matrix[fi, ci] = 0
            else # distribute new dofs
                Base._setindex!(countedbefore, true, sface, -token)# countedbefore[sface] = true,  mark the face as counted
            end
        end
    end
    sparse(boundary_matrix)
end
