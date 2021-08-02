struct PointEvalHandler{DH<:AbstractDofHandler,dim,T<:Real}
    dh::DH
    cells::Vector{Union{Missing, Int}}
    local_coords::Vector{Vec{dim,T}} # TODO: store local coordinates instead of PointScalarValues (can we toss the PointScalarValues in that case?)
end

# TODO write show method for PointEvalHandler

function PointEvalHandler(dh::AbstractDofHandler, points::AbstractVector{Vec{dim, T}},
    geom_interpolations::Vector{<:Interpolation{dim}}=get_default_geom_interpolations(dh);
    ) where {dim, T<:Real}

    node_cell_dicts = [_get_node_cell_map(grid) for field_idx in 1:nfields(dh)] # why do we do this per field?
    cells, local_coords = _get_cellcoords(points, dh.grid, node_cell_dicts, geom_interpolations)

    return PointEvalHandler(dh, cells, local_coords)
end

# function interpolations need to be explicitely given, because we don't know which field we are looking for.
# Only one field at a time can be interpolated, so one function interpolation per FieldHandler (= per cellset) is sufficient.
# If several fields should be interpolated with different function interpolations, several PointEvalHandlers need to be constructed.
function PointEvalHandler(dh::MixedDofHandler{dim, T}, points::AbstractVector{Vec{dim, T}},
    geom_interpolations::Vector{<:Interpolation{dim}}=get_default_geom_interpolations(dh);
    ) where {dim, T<:Real}

    # TODO: test that geom_interpolation is compatible with grid (use this as a default)

    node_cell_dicts = [_get_node_cell_map(grid, fh.cellset) for fh in dh.fieldhandlers]
    cells, local_coords = _get_cellcoords(points, dh.grid, node_cell_dicts, geom_interpolations)

   return PointEvalHandler(dh, cells, local_coords)
end

get_default_geom_interpolations(dh::DofHandler{dim}) = [default_interpolation(first(typeof(dh.grid.cells)))]

function get_default_geom_interpolations(dh::MixedDofHandler{dim}) where {dim}
    ips = Interpolation{dim}[]
    for fh in dh.fieldhandlers
        push!(ips, default_interpolation(typeof(grid.cells[first(fh.cellset)])))
    end
    return ips
end

function _get_cellcoords(points::AbstractVector{Vec{dim,T}}, grid::Grid, node_cell_dicts::Vector{Dict{Int, Vector{Int}}}, 
     geom_interpolations::Vector{<:Interpolation{dim}}) where {dim, T<:Real}

    # set up tree structure for finding nearest nodes to points
    kdtree = KDTree([node.x for node in grid.nodes])
    nearest_nodes, _ = knn(kdtree, points, 3, true) #TODO 3 is a random value, it shouldn't matter because likely the nearest node is the one we want

    cells = Vector{Union{Missing, Int}}(undef, length(points))
    local_coords = Vector{Vec{dim, T}}(undef, length(points))

    for point_idx in 1:length(points)
        cell_found = false
        for dict_idx in 1:length(node_cell_dicts) # all cells in a node_cell_dict have the same geom_interpolation
            node_cell_dict = node_cell_dicts[dict_idx]
            geom_interpol = geom_interpolations[dict_idx]
            # loop over points
            for node in nearest_nodes[point_idx], cell in get(node_cell_dict, node, [0])
                # if node is not part of the fieldhandler, try the next node
                cell == 0 ? continue : nothing

                cell_coords = getcoordinates(grid, cell)
                is_in_cell, local_coord = point_in_cell(geom_interpol, cell_coords, points[point_idx])
                if is_in_cell
                    cell_found = true
                    cells[point_idx] = cell
                    local_coords[point_idx] = local_coord
                    break
                end
            end
        end
        !cell_found && (cells[point_idx] = missing) #error("No cell found for point $(points[point_idx]), index $point_idx.")
    end
    return cells, local_coords
end

# check if point is inside a cell based on physical coordinate
function point_in_cell(geom_interpol::Interpolation{dim,shape,order}, cell_coordinates, global_coordinate) where {dim, shape, order}
    converged, x_local = find_local_coordinate(geom_interpol, cell_coordinates, global_coordinate)
    if converged
        return _check_isoparametric_boundaries(shape, x_local), x_local
    else
        return false, x_local
    end
end

# check if point is inside a cell based on isoparametric coordinate
function _check_isoparametric_boundaries(::Type{RefCube}, x_local::Vec{dim}) where {dim}
    inside = true
    for x in x_local
        if x >= -1.0 && x<= 1.0 || abs(x) ≈ 1.0 # might happen on the boundary of the cell
             nothing
        else
            inside = false
        end
    end
    return inside
end

# check if point is inside a cell based on isoparametric coordinate
function _check_isoparametric_boundaries(::Type{RefTetrahedron}, x_local::Vec{dim}) where {dim}
    inside = true
    for x in x_local
        x >= 0.0 && x<= 1.0 ? nothing : inside = false
    end
    0.0 <= 1-sum(x_local) <= 1.0 ? nothing : inside=false
    return inside
end

# TODO: should we make iteration params optional keyword arguments?
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
    converged = false
    for iter in 1:10
        N = Ferrite.value(interpolation, local_guess)

        global_guess = zero(Vec{dim})
        for j in 1:n_basefuncs
            global_guess += N[j] * cell_coordinates[j]
        end
        residual = global_guess - global_coordinate
        if norm(residual) <= tol_norm
            converged = true
            break
        end
        dNdξ = Ferrite.derivative(interpolation, local_guess)
        J = zero(Tensor{dim, dim})
        for j in 1:n_basefuncs
            J += cell_coordinates[j] ⊗ dNdξ[j]
        end
        local_guess -= inv(J) ⋅ residual
    end
    return converged, local_guess
end

function _get_node_cell_map(grid::Grid, cellset::Set{Int}=Set{Int64}(1:getncells(grid)))
    cell_dict = Dict{Int, Vector{Int}}()
    for cellidx in cellset
        for node in grid.cells[cellidx].nodes
            if haskey(cell_dict, node)
                push!(cell_dict[node], cellidx)
            else
                cell_dict[node] = [cellidx]
            end
        end
    end
    return cell_dict
end

# some useful code:
# # values in nodal order
# # can't be used for sub/superparametric approximations
# func_interpolations::Vector{<:Interpolation{dim}}

# # retrieve RefShape for QuadratureRule
# refshape = getrefshape(func_interpol)
# point_qr = QuadratureRule{dim, refshape, T}([1], [local_coordinate])
# pv = PointScalarValues(point_qr, func_interpol)

function get_point_values(ph::PointEvalHandler, nodal_values::Vector{T}, func_interpolations = get_default_geom_interpolations(dh)) where {T<:Union{Real, AbstractTensor}}
    # TODO check for sub/superparametric approximations

    # if interpolation !== default_interpolation(typeof(ch.dh.grid.cells[first(cellset)]))
    #     @warn("adding constraint to nodeset is not recommended for sub/super-parametric approximations.")
    # end
    length(nodal_values) == getnnodes(ph.dh.grid) || error("You must supply nodal values for all nodes of the Grid.")

    npoints = length(ph.cells)
    vals = Vector{T}(undef, npoints)
    for i in eachindex(ph.cells)
        # TODO: need the PointScalarValues now. Can either compute them here or supply function for computing all of them and store in a Vector
        # Anyways: We need the fieldhandler the cells belong to. Should they be stored in the PointEvalHandler?
        node_idxs = getcells(ph.dh.grid, ph.cells[i]).nodes
        vals[i] = function_value(ph.cellvalues[i], 1, [nodal_values[node_idx] for node_idx in node_idxs])
    end
    return vals
end

# values in dof-order. They must be obtained from the same DofHandler that was used for constructing the PointEvalHandler
function get_point_values(ph::PointEvalHandler{DH}, dof_values::Vector{T}, fieldname::Symbol) where {T,DH<:MixedDofHandler}

    length(dof_values) == ndofs(ph.dh) || error("You must supply nodal values for all $(ndofs(ph.dh)) dofs.")

    npoints = length(ph.cells)
    vals = Vector{T}(undef, npoints)
    for i in eachindex(ph.cells)
        # set NaN values if no cell was found for a point
        if ismissing(ph.cells[i])
            vals[i] = first(dof_values)*NaN
            continue
        end
        cell_dofs = celldofs(ph.dh, ph.cells[i])
        #TODO damn, now we need to find the fieldhandler again --> should be stored
        # want to be able to use this no matter if dof_values are coming from L2Projector or from simulation
        for fh in ph.dh.fieldhandlers
            if ph.cells[i] ∈ fh.cellset
                dofrange = dof_range(fh, fieldname)
                field_dofs = cell_dofs[dofrange]
                vals[i] = function_value(ph.cellvalues[i], 1, dof_values[field_dofs])
            end
        end
    end
    return vals
end

function get_point_values(ph::PointEvalHandler{DH}, dof_values::Vector{T}, fieldname::Symbol) where {T,DH<:DofHandler}

    length(dof_values) == ndofs(ph.dh) || error("You must supply nodal values for all $(ndofs(ph.dh)) dofs.")

    fielddim = getfielddim(ph.dh, fieldname)

    npoints = length(ph.cells)
    # for a scalar field return a Vector of Scalars, for a vector field return a Vector of Vecs
    if fielddim == 1
        vals = Vector{T}(undef, npoints)
    else
        vals = Vector{Vec{fielddim, T}}(undef, npoints)
    end
    
    for i in eachindex(ph.cells)
        # set NaN values if no cell was found for a point
        if ismissing(ph.cells[i])
            vals[i] = first(dof_values)*NaN
            continue
        end
        cell_dofs = celldofs(ph.dh, ph.cells[i])
        dofrange = dof_range(ph.dh, fieldname)
        field_dofs = cell_dofs[dofrange]
        # reshape field_dofs so that they work with ScalarValues
        if fielddim == 1
            dof_values_reshaped = dof_values[field_dofs]
        else
            dof_values_reshaped = reinterpret(Vec{fielddim, T}, dof_values[field_dofs])
        end
        vals[i] = function_value(ph.cellvalues[i], 1, dof_values_reshaped)
    end
    return vals
end

get_point_values(ph::PointEvalHandler, dof_values::Vector{T}, ::L2Projector) where T = get_point_values(ph, dof_values, :_)

