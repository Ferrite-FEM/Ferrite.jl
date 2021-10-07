"""
    PointEvalHandler(dh::AbstractDofHandler, points::AbstractVector{Vec{dim,T}}, [geom_interpolations::Vector{<:Interpolation{dim}}]) where {dim, T}

A `PointEvalHandler` computes the corresponding cell for each point in `points` and its local coordinate within the cell.
When no `geom_interpolations` are given, the default interpolations of the grid are used. For using custom geometry interpolations hand in a vector of interpolations,
whose order corresponds to the order of `FieldHandlers` within the given `MixedDofHandler`. For a `DofHandler`, give a Vector with your geometry interpolation as single entry.
"""
PointEvalHandler

struct PointEvalHandler{DH<:AbstractDofHandler,dim,T<:Real}
    dh::DH
    cells::Vector{Union{Nothing, Int}}
    local_coords::Vector{Union{Nothing, Vec{dim,T}}}
    pointidx_sets::Vector{Set{Int}} # indices to access cells and local_coords
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

function PointEvalHandler(dh::AbstractDofHandler, points::AbstractVector{Vec{dim, T}},
    geom_interpolations::Vector{<:Interpolation{dim}}=get_default_geom_interpolations(dh);
    ) where {dim, T<:Real}

    getrefshape.(geom_interpolations) == getrefshape.(get_default_geom_interpolations(dh)) || error("The given geometry interpolations are incompatible with the given DofHandler.")

    node_cell_dicts = [_get_node_cell_map(dh.grid)]
    cells, local_coords, cellsets = _get_cellcoords(points, dh.grid, node_cell_dicts, geom_interpolations)

    return PointEvalHandler(dh, cells, local_coords, cellsets)
end

# function interpolations need to be explicitely given, because we don't know which field we are looking for.
# Only one field at a time can be interpolated, so one function interpolation per FieldHandler (= per cellset) is sufficient.
# If several fields should be interpolated with different function interpolations, several PointEvalHandlers need to be constructed.
function PointEvalHandler(dh::MixedDofHandler{dim, T}, points::AbstractVector{Vec{dim, T}},
    geom_interpolations::Vector{<:Interpolation{dim}}=get_default_geom_interpolations(dh);
    ) where {dim, T<:Real}

    getrefshape.(geom_interpolations) == getrefshape.(get_default_geom_interpolations(dh)) || error("The given geometry interpolations are incompatible with the given MixedDofHandler.")

    node_cell_dicts = [_get_node_cell_map(dh.grid, fh.cellset) for fh in dh.fieldhandlers]
    cells, local_coords, cellsets = _get_cellcoords(points, dh.grid, node_cell_dicts, geom_interpolations)

   return PointEvalHandler(dh, cells, local_coords, cellsets)
end

function _get_cellcoords(points::AbstractVector{Vec{dim,T}}, grid::Grid, node_cell_dicts::Vector{Dict{Int, Vector{Int}}}, 
     geom_interpolations::Vector{<:Interpolation{dim}}) where {dim, T<:Real}

    # set up tree structure for finding nearest nodes to points
    kdtree = KDTree(reinterpret(Vec{dim,T}, grid.nodes)
    nearest_nodes, _ = knn(kdtree, points, 3, true) #TODO 3 is a random value, it shouldn't matter because likely the nearest node is the one we want

    cells = Vector{Union{Nothing, Int}}(nothing, length(points))
    local_coords = Vector{Union{Nothing, Vec{dim, T}}}(nothing, length(points))
    cellset_sets = [Set{Int}() for _ in eachindex(node_cell_dicts)]

    for point_idx in 1:length(points)
        cell_found = false
        for ip_idx in 1:length(node_cell_dicts) # all cells in a node_cell_dict have the same geom_interpolation
            node_cell_dict = node_cell_dicts[ip_idx]
            geom_interpol = geom_interpolations[ip_idx]
            # loop over points
            for node in nearest_nodes[point_idx]
                possible_cells = get(node_cell_dict, node, nothing)
                possible_cells === nothing && continue # if node is not part of the fieldhandler, try the next node
                for cell in possible_cells
                    cell_coords = getcoordinates(grid, cell)
                    is_in_cell, local_coord = point_in_cell(geom_interpol, cell_coords, points[point_idx])
                    if is_in_cell
                        cell_found = true
                        cells[point_idx] = cell
                        local_coords[point_idx] = local_coord
                        push!(cellset_sets[ip_idx], point_idx)
                        break
                    end
                end
                cell_found && break
            end
            cell_found && break
        end
        if !cell_found
            @warn("No cell found for point number $point_idx, coordinate: $(points[point_idx]).")
        end
    end
    return cells, local_coords, cellset_sets
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
function _check_isoparametric_boundaries(::Type{RefCube}, x_local::Vec{dim, T}) where {dim, T}
    for x in x_local
        if abs(x) - 1.0 > sqrt(eps(T))
             return false
        end
    end
    return true
end

# check if point is inside a cell based on isoparametric coordinate
function _check_isoparametric_boundaries(::Type{RefTetrahedron}, x_local::Vec{dim, T}) where {dim, T}
    tol = sqrt(eps(T))
    for x in x_local
        if x < -tol || x - 1 > tol
            return false
        end
    end
    if sum(x_local) < -tol || sum(x_local) - 1. > tol
        return false
    end
    return true
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

# return a Dict with a key for each node that contains a vector with the adjacent cells as value
function _get_node_cell_map(grid::Grid, cellset::Set{Int}=Set{Int64}(1:getncells(grid)))
    cell_dict = Dict{Int, Vector{Int}}()
    for cellidx in cellset
        for node in grid.cells[cellidx].nodes
            v = get!(Vector{Int}, cell_dict, node)
            push!(v, cellidx)
        end
    end
    return cell_dict
end

##################################################################################################################
# points in nodal order

"""
    get_point_values(ph::PointEvalHandler, nodal_values::Vector{T}, [func_interpolations::Vector{<:Interpolation}]) where T

Return a `Vector{T}` with the field values in the points of the `PointEvalHandler`. The field values are computed based on the `nodal_values` and interpolated to the local coordinates by the `func_interpolations`.
The `nodal_values` must be given in nodal order.

If no function interpolations are given, the interpolations are determined based on the grid. For using custom function interpolations hand in a vector of interpolations,
whose order corresponds to the order of `FieldHandlers` within the `MixedDofHandler` that was used for constructing the `PointEvalHandler`. For a `DofHandler`, give a Vector with your function interpolation as single entry.

This function should not be used for superparametric approximations. Instead, use the version which is based on `dof_values`.
"""
function get_point_values(
    ph::PointEvalHandler{DH},
    nodal_values::Vector{T},
    func_interpolations::Vector{<:Interpolation} = get_default_geom_interpolations(ph.dh),
    ) where {DH<:MixedDofHandler, T<:Union{Real, AbstractTensor}}

    if func_interpolations != get_default_geom_interpolations(ph.dh)
        @warn("Obtaining point values based on nodal values is not recommended for superparametric approximations. You can igonre this warning for subparametric approximations.")
    end

    length(nodal_values) == getnnodes(ph.dh.grid) || error("You must supply nodal values for all nodes of the Grid.")

    npoints = length(ph.cells)
    vals = Vector{T}(undef, npoints)
    # for type stability with the MixedDofHandler, we want compute this per FieldHandler, as all cells in a FieldHandler share the interpolation
    for fh_idx in eachindex(ph.dh.fieldhandlers)
        _get_point_values!(vals, nodal_values, func_interpolations[fh_idx], ph, fh_idx)
    end
    _set_missing_vals!(vals, ph)
    return vals
end

function get_point_values(
    ph::PointEvalHandler{DH},
    nodal_values::Vector{T},
    func_interpolations::Vector{<:Interpolation} = get_default_geom_interpolations(ph.dh),
    ) where {DH<:DofHandler, T<:Union{Real, AbstractTensor}}

    if func_interpolations != get_default_geom_interpolations(ph.dh)
        @warn("Obtaining point values based on nodal values is not recommended for superparametric approximations. You can igonre this warning for subparametric approximations.")
    end

    length(nodal_values) == getnnodes(ph.dh.grid) || error("You must supply nodal values for all nodes of the Grid.")

    npoints = length(ph.cells)
    vals = Vector{T}(undef, npoints)
    _get_point_values!(vals, nodal_values, func_interpolations[1], ph, 1)
    _set_missing_vals!(vals, ph)
    return vals
end

# this is just a function barrier
function _get_point_values!(vals::Vector{T}, nodal_values::Vector{T}, ip::Interpolation, ph::PointEvalHandler, fh_idx::Int) where T
    # allocate quantities specific to Celltype / Interpolation
    pv = PointScalarValues(ph.local_coords[first(ph.pointidx_sets[fh_idx])], ip)
    nnodes = nnodes_per_cell(ph.dh.grid, first(ph.cells[first(ph.pointidx_sets[fh_idx])]))
    node_idxs = Vector{Int}(undef, nnodes)

    # compute point values
    for point_idx in ph.pointidx_sets[fh_idx]
        reinit!(pv, ph.local_coords[point_idx], ip)
        node_idxs[1:nnodes] .= getcells(ph.dh.grid, ph.cells[point_idx]).nodes
        @inbounds @views vals[point_idx] = function_value(pv, 1, nodal_values[node_idxs])
    end
    return vals
end

##################################################################################################################
# values in dof-order. They must be obtained from the same DofHandler that was used for constructing the PointEvalHandler
"""
    get_point_values(ph::PointEvalHandler, dof_values::Vector{T}, fieldname::Symbol, [func_interpolations::Vector{<:Interpolation}]) where T
    get_point_values(ph::PointEvalHandler, dof_values::Vector{T}, ::L2Projector, [func_interpolations::Vector{<:Interpolation}]) where T

Return a `Vector{T}` (for a 1-dimensional field) or a `Vector{Vec{fielddim, T}}` (for a vector field) with the field values of field `fieldname` in the points of the `PointEvalHandler`. The field values are computed based on the `dof_values` and interpolated to the local coordinates by the `func_interpolations`.
The `dof_values` must be given in the order of the dofs that corresponds to the `AbstractDofHandler` which was used to construct `ph`. If the `dof_values` are obtained by a `L2Projector`, the `L2Projector` can be handed over instead of the `fieldname`. 

If no function interpolations are given, the field interpolations are extracted from the `AbstractDofHandler` of `ph`. For using custom function interpolations hand in a vector of interpolations,
whose order corresponds to the order of `FieldHandlers` within the `MixedDofHandler` that was used for constructing `ph`. For a `DofHandler`, give a Vector with your function interpolation as single entry.
"""
function get_point_values(
    ph::PointEvalHandler{DH},
    dof_values::Vector{T},
    fieldname::Symbol,
    func_interpolations = get_func_interpolations(ph, fieldname)
    ) where {T,DH<:AbstractDofHandler}

    fielddim = getfielddim(ph.dh, fieldname)
    npoints = length(ph.cells)
    # for a scalar field return a Vector of Scalars, for a vector field return a Vector of Vecs
    if fielddim == 1
        vals = Vector{T}(undef, npoints)
    else
        vals = Vector{Vec{fielddim, T}}(undef, npoints)
    end
    get_point_values!(vals, ph, dof_values, fieldname, func_interpolations)
    return vals
end

get_point_values(ph::PointEvalHandler{DH}, dof_values::Vector{T}, ::L2Projector, func_interpolations = get_func_interpolations(ph, :_)) where {DH<:MixedDofHandler,T} = get_point_values(ph, dof_values, :_, func_interpolations)

# values in dof-order. They must be obtained from the same DofHandler that was used for constructing the PointEvalHandler
function get_point_values!(vals::Vector{T2},
    ph::PointEvalHandler{DH},
    dof_values::Vector{T},
    fieldname::Symbol,
    func_interpolations = get_func_interpolations(ph, fieldname)
    ) where {T2, T,DH<:MixedDofHandler} 

    length(dof_values) == ndofs(ph.dh) || error("You must supply nodal values for all $(ndofs(ph.dh)) dofs.")

    fielddim = getfielddim(ph.dh, fieldname)
    
    for fh_idx in eachindex(ph.dh.fieldhandlers)
        _get_point_values!(vals, dof_values, ph, func_interpolations[fh_idx], fh_idx, fieldname, Val(fielddim))
    end
    _set_missing_vals!(vals, ph)
    return vals
end

function get_point_values!(vals::Vector{T2},
    ph::PointEvalHandler{DH},
    dof_values::Vector{T},
    fieldname::Symbol,
    func_interpolations = get_func_interpolations(ph, fieldname)
    ) where {T2, T,DH<:DofHandler} 

    length(dof_values) == ndofs(ph.dh) || error("You must supply nodal values for all $(ndofs(ph.dh)) dofs.")

    fielddim = getfielddim(ph.dh, fieldname)
    _get_point_values!(vals, dof_values, ph, func_interpolations[1], 1, fieldname, Val(fielddim))
    _set_missing_vals!(vals, ph)
    return vals
end

# this is just a function barrier
function _get_point_values!(
    vals::Vector{T2},
    dof_values::Vector{T},
    ph::PointEvalHandler{DH,dim},
    ip::Interpolation,
    fh_idx::Int,
    fieldname::Symbol,
    fdim::Val{fielddim}) where {T2,dim,T,DH<:AbstractDofHandler, fielddim}

    # extract variables
    local_coords = ph.local_coords
    dh = ph.dh
    point_idx_set = ph.pointidx_sets[fh_idx]
    # preallocate some stuff specific to this cellset
    pv = PointScalarValues(first(local_coords), ip)
    cell_dofs = celldofs(dh, ph.cells[first(point_idx_set)])
    dofrange = _dof_range(dh, fieldname, fh_idx)

    # compute point values
    for pointid in point_idx_set
        celldofs!(cell_dofs, dh, ph.cells[pointid])
        reinit!(pv, local_coords[pointid], ip)
        @inbounds @views dof_values_reshaped = _change_format(fdim, dof_values[cell_dofs[dofrange]])
        vals[pointid] = function_value(pv, 1, dof_values_reshaped)
    end
    return vals
end

###################################################################################################################
# utils 

# set NaNs for points where no cell was found
function _set_missing_vals!(vals::Vector{T}, ph::PointEvalHandler) where T
    v = vals[first(first(ph.pointidx_sets))] # just needed for its type
    for (idx, cell) in enumerate(ph.cells)
        cell !== nothing && continue
        vals[idx] = v*NaN
    end
    return vals
end

# reshape dof_values based on fielddim
_change_format(::Val{1}, dof_values::AbstractVector{T}) where T = dof_values
_change_format(::Val{fielddim}, dof_values::AbstractVector{T}) where {fielddim, T} = reinterpret(Vec{fielddim, T}, dof_values)

## work-arounds so that we can call stuff with the same syntax for DofHandler and MixedDofHandler
_dof_range(dh::DofHandler, fieldname, ::Int) = dof_range(dh, fieldname)
_dof_range(dh::MixedDofHandler, fieldname, fh_idx::Int) = dof_range(dh.fieldhandlers[fh_idx], fieldname)

get_default_geom_interpolations(dh::DofHandler) = [default_interpolation(typeof(first(dh.grid.cells)))]
function get_default_geom_interpolations(dh::MixedDofHandler{dim}) where {dim}
    ips = Interpolation{dim}[]
    for fh in dh.fieldhandlers
        push!(ips, default_interpolation(typeof(dh.grid.cells[first(fh.cellset)])))
    end
    return ips
end

get_func_interpolations(ph::PointEvalHandler{DH}, fieldname) where DH<:DofHandler = [getfieldinterpolation(ph.dh, find_field(ph.dh, fieldname))]
function get_func_interpolations(ph::PointEvalHandler{DH}, fieldname) where DH<:MixedDofHandler
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