struct PointEvalHandler{DH<:AbstractDofHandler,dim,T<:Real}
    dh::DH
    cells::Vector{Union{Missing, Int}}
    local_coords::Vector{Vec{dim,T}} # TODO: store local coordinates instead of PointScalarValues (can we toss the PointScalarValues in that case?)
    ip_idxs::Vector{Int}
end

# TODO write show method for PointEvalHandler

function PointEvalHandler(dh::AbstractDofHandler, points::AbstractVector{Vec{dim, T}},
    geom_interpolations::Vector{<:Interpolation{dim}}=get_default_geom_interpolations(dh);
    ) where {dim, T<:Real}

    node_cell_dicts = [_get_node_cell_map(dh.grid)]
    cells, local_coords, ip_idxs = _get_cellcoords(points, dh.grid, node_cell_dicts, geom_interpolations)

    return PointEvalHandler(dh, cells, local_coords, ip_idxs)
end

function Base.show(io::IO, ::MIME"text/plain", ph::PointEvalHandler)
    println(io, typeof(ph))
    println(io, "  number of points: ", length(ph.local_coords))
    n_missing = sum(ismissing.(ph.cells))
    if n_missing == 0
        println(io, "  Found corresponding cell for all points.")
    else
        println(io, "  Could not find corresponding cell for ", n_missing, " points.")
    end
end
# function interpolations need to be explicitely given, because we don't know which field we are looking for.
# Only one field at a time can be interpolated, so one function interpolation per FieldHandler (= per cellset) is sufficient.
# If several fields should be interpolated with different function interpolations, several PointEvalHandlers need to be constructed.
function PointEvalHandler(dh::MixedDofHandler{dim, T}, points::AbstractVector{Vec{dim, T}},
    geom_interpolations::Vector{<:Interpolation{dim}}=get_default_geom_interpolations(dh);
    ) where {dim, T<:Real}

    # TODO: test that geom_interpolation is compatible with grid (use this as a default)

    node_cell_dicts = [_get_node_cell_map(dh.grid, fh.cellset) for fh in dh.fieldhandlers]
    cells, local_coords, ip_idxs = _get_cellcoords(points, dh.grid, node_cell_dicts, geom_interpolations)

   return PointEvalHandler(dh, cells, local_coords, ip_idxs)
end

get_default_geom_interpolations(dh::DofHandler) = [default_interpolation(typeof(first(dh.grid.cells)))]

function get_default_geom_interpolations(dh::MixedDofHandler{dim}) where {dim}
    ips = Interpolation{dim}[]
    for fh in dh.fieldhandlers
        push!(ips, default_interpolation(typeof(dh.grid.cells[first(fh.cellset)])))
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
    ip_idxs = Vector{Int}(undef, length(points))

    for point_idx in 1:length(points)
        cell_found = false
        for ip_idx in 1:length(node_cell_dicts) # all cells in a node_cell_dict have the same geom_interpolation
            node_cell_dict = node_cell_dicts[ip_idx]
            geom_interpol = geom_interpolations[ip_idx]
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
                    ip_idxs[point_idx] = ip_idx
                    break
                end
            end
        end
        !cell_found && (cells[point_idx] = missing) #error("No cell found for point $(points[point_idx]), index $point_idx.")
    end
    return cells, local_coords, ip_idxs
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
    for iter in 1:max_iters
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

function get_func_interpolations(ph, fieldname)
    func_interpolations = []
    for fh in ph.dh.fieldhandlers
        j = findfirst(i->i == fieldname, getfieldnames(fh))
        if !(j === nothing)
            push!(func_interpolations, fh.fields[j].interpolation)
        else
            push!(func_interpolations, missing)
        end
    end
    return func_interpolations
end

function get_point_values(ph::PointEvalHandler, nodal_values::Vector{T}, func_interpolations::Vector{<:Interpolation} = get_default_geom_interpolations(ph.dh)) where {T<:Union{Real, AbstractTensor}}
    # TODO check for sub/superparametric approximations

    # if interpolation !== default_interpolation(typeof(ch.dh.grid.cells[first(cellset)]))
    #     @warn("adding constraint to nodeset is not recommended for sub/super-parametric approximations.")
    # end
    length(nodal_values) == getnnodes(ph.dh.grid) || error("You must supply nodal values for all nodes of the Grid.")

    npoints = length(ph.cells)
    vals = Vector{T}(undef, npoints)
    for i in eachindex(ph.cells)
        # TODO: would it be worth offering an option where these are precomputed?
        pv = PointScalarValues(ph.local_coords[i], func_interpolations[ph.ip_idxs[i]])
        node_idxs = getcells(ph.dh.grid, ph.cells[i]).nodes
        vals[i] = function_value(pv, 1, [nodal_values[node_idx] for node_idx in node_idxs])
    end
    return vals
end

# values in dof-order. They must be obtained from the same DofHandler that was used for constructing the PointEvalHandler
function get_point_values(ph::PointEvalHandler{DH}, dof_values::Vector{T}, fieldname::Symbol; func_interpolations = get_func_interpolations(ph, fieldname)) where {T,DH<:MixedDofHandler}

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
        for fh in ph.dh.fieldhandlers
            if ph.cells[i] ∈ fh.cellset
                dofrange = dof_range(fh, fieldname)
                field_dofs = cell_dofs[dofrange]
                pv = PointScalarValues(ph.local_coords[i], func_interpolations[ph.ip_idxs[i]])
                # reshape field_dofs so that they work with ScalarValues
                if fielddim == 1
                    dof_values_reshaped = dof_values[field_dofs]
                else
                    dof_values_reshaped = reinterpret(Vec{fielddim, T}, dof_values[field_dofs])
                end
                vals[i] = function_value(pv, 1, dof_values_reshaped)
            end
        end
    end
    return vals
end

# values in dof-order. They must be obtained from the same DofHandler that was used for constructing the PointEvalHandler
function get_point_values!(vals::Vector{T2}, ph::PointEvalHandler{DH}, dof_values::Vector{T}, fieldname::Symbol; func_interpolations = get_func_interpolations(ph, fieldname)) where {T2, T,DH<:MixedDofHandler} 
    length(dof_values) == ndofs(ph.dh) || error("You must supply nodal values for all $(ndofs(ph.dh)) dofs.")

    # TODO: should really check that T2 corresponds to return type for fielddim!

    fielddim = getfielddim(ph.dh, fieldname)

    for fh_idx in eachindex(ph.dh.fieldhandlers)
        _get_point_values!(vals, dof_values, ph, func_interpolations[fh_idx], fh_idx, fieldname, fielddim)
    end
    return vals
end

# this is just a function barrier
function _get_point_values!(
    vals::Vector{T2},
    dof_values::Vector{T},
    ph::PointEvalHandler{DH,dim,T},
    ip::Interpolation,
    fh_idx::Int,
    fieldname::Symbol,
    fielddim::Int) where {T2,dim,T,refshape,DH<:MixedDofHandler}

    local_coords = ph.local_coords
    dh = ph.dh
    cellset = Set{Int}(collect(1:length(ph.cells))[ph.ip_idxs .== fh_idx]) # TODO: should really store these sets and not ip_idxs
    pv = PointScalarValues(first(local_coords), ip)
    cell_dofs = celldofs(dh, first(cellset))
    dofrange = dof_range(dh.fieldhandlers[fh_idx], fieldname)
    for cellid in cellset
        # set NaN values if no cell was found for a point
        if ismissing(ph.cells[cellid])
            vals[cellid] = first(dof_values)*NaN
            continue
        end
        celldofs!(cell_dofs, dh, cellid)
        # field_dofs = view(cell_dofs, dofrange_field)
        reinit!(pv, local_coords[cellid], ip)
        if fielddim == 1
            @inbounds @views dof_values_reshaped = dof_values[cell_dofs[dofrange]]
        else
            dof_values_reshaped = reinterpret(Vec{fielddim, T}, dof_values[cell_dofs[dofrange]]) # TODO: should this use inbounds or views?
        end
        vals[cellid] = function_value(pv, 1, dof_values_reshaped)::T2
    end
    return vals
end


function get_point_values(ph::PointEvalHandler{DH}, dof_values::Vector{T}, fieldname::Symbol; func_interpolations = [getfieldinterpolation(ph.dh, find_field(ph.dh, fieldname))]) where {T,DH<:DofHandler}

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
        pv = PointScalarValues(ph.local_coords[i], func_interpolations[ph.ip_idxs[i]])
        vals[i] = function_value(pv, 1, dof_values_reshaped)
    end
    return vals
end

get_point_values(ph::PointEvalHandler{DH}, dof_values::Vector{T}, ::L2Projector) where {DH<:MixedDofHandler,T} = get_point_values(ph, dof_values, :_)

