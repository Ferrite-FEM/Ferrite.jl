struct PointEvalHandler{dim, C, T, U}
    dh::DofHandler{dim, C, T}
    cells::Vector{Int}
    cellvalues::Vector{U}
end

# function PointEvalHandler(grid, interpolation, points)
#     """
#     This is what you would call in reality to set up the PointEvalHandler
#     """
#     kdtree = KDTree([gn.x for gn in grid.nodes])
#     """
#     Assume that we have a conformal mesh for now so that the nearest node will 
#     be a part of the cell that the point is in
#     """
#     nnodes, _ = knn(kdtree, points, 1) 
#     nnodes = reduce(vcat, nnodes)
# end

function PointEvalHandler(dh::DofHandler{dim, C, T}, interpolation::Interpolation{dim, S}, points::AbstractVector{Vec{dim, T}}) where {dim, C, S, T<:Real}
    grid = dh.grid
    dummy_cell_itp_qr = QuadratureRule{dim, S}(1)
    dummy_face_qr = QuadratureRule{dim-1, S}(1)
    cellvalues = CellScalarValues(dummy_cell_itp_qr, interpolation)
    facevalues = FaceScalarValues(dummy_face_qr, interpolation)
    points_array = [(i, p) for (i, p) in enumerate(points)]
    cells_of_points = Vector{Int}(undef, length(points))
    cellvalues_of_points = Vector{typeof(cellvalues)}(undef, length(points))

    for cell in 1:length(grid.cells)
        cell_coords = getcoordinates(grid, cell)
        face_nodes = faces(grid.cells[cell])
        nodes_on_faces = [grid.nodes[fn[1]] for fn in face_nodes]
        deletions = []
        for (j, ipoint) in enumerate(points_array)
            i, point = ipoint
            if is_point_inside_cell(cell_coords, point, facevalues, nodes_on_faces)
                push!(deletions, j)
                local_coordinate = find_local_coordinate(interpolation, cell_coords, point)
                point_qr = QuadratureRule{2, S, T}([1], [local_coordinate])
                cells_of_points[i] = cell
                # since these are unique for each point to evaluate, we need to create a cellvalue for each point anyway, so we might as well do it all at once
                cellvalues = CellScalarValues(point_qr, interpolation)
                reinit!(cellvalues, cell_coords)
                cellvalues_of_points[i] = cellvalues 
            end
        end
        deleteat!(points_array, deletions)
    end
    if length(points_array) > 0
        println(points_array)
        error("Did not find cells for all points")
    end
    return PointEvalHandler(dh, cells_of_points, cellvalues_of_points)
end

function (peh::PointEvalHandler)(dof_values)
    [function_value(cellvalue, 1, 
                    dof_values[celldofs(peh.dh, cell)]) for (cell, cellvalue) in zip(peh.cells,
                                                                                     peh.cellvalues)]
end

pointeval(peh::PointEvalHandler, dof_values) = peh(dof_values)

function is_point_inside_cell(cell_coords, point, facevalues, nodes_on_faces)
    is_on_wrong_side_of_a_plane = false

    for face in 1:length(nodes_on_faces)
        reinit!(facevalues, cell_coords, face)
        normal = facevalues.normals[1]
        face_node = nodes_on_faces[face]
        test = dot(normal, point-face_node.x) > 0
        is_on_wrong_side_of_a_plane = is_on_wrong_side_of_a_plane || test
    end
    return !is_on_wrong_side_of_a_plane
end

function find_local_coordinate(interpolation, cell_coordinates, global_coordinate)
    """
    currently copied verbatim from https://discourse.julialang.org/t/finding-the-value-of-a-field-at-a-spatial-location-in-juafem/38975/2
    other than to make J dim x dim rather than 2x2
    """
    dim = length(global_coordinate)
    local_guess = zero(Vec{dim})
    n_basefuncs = getnbasefunctions(interpolation)
    max_iters = 10
    tol_norm = 1e-10
    for iter in 1:10
        if iter == max_iters
            error("did not find a local coordinate")
        end
        N = JuAFEM.value(interpolation, local_guess)

        global_guess = zero(Vec{dim})
        for j in 1:n_basefuncs
            global_guess += N[j] * cell_coordinates[j]
        end
        residual = global_guess - global_coordinate
        if norm(residual) <= tol_norm
            break
        end
        dNdξ = JuAFEM.derivative(interpolation, local_guess)
        J = zero(Tensor{dim, dim})
        for j in 1:n_basefuncs
            J += cell_coordinates[j] ⊗ dNdξ[j]
        end
        local_guess -= inv(J) ⋅ residual
    end
    return local_guess
end
