"""
    PointEvalHandler(dh::AbstractDofHandler, points::AbstractVector{Vec{dim,T}}, [geom_interpolations::Vector{<:Interpolation{dim}}]) where {dim, T}

A `PointEvalHandler` computes the corresponding cell for each point in `points` and its local coordinate within the cell.
When no `geom_interpolations` are given, the default interpolations of the grid are used. For using custom geometry interpolations hand in a vector of interpolations,
whose order corresponds to the order of `FieldHandlers` within the given `MixedDofHandler`. For a `DofHandler`, give a Vector with your geometry interpolation as single entry.
"""
PointEvalHandler

struct PointEvalHandler{G,dim,T<:Real}
    grid::G
    cells::Vector{Union{Nothing, Int}}
    local_coords::Vector{Union{Nothing, Vec{dim,T}}}
    pointidx_sets::Vector{Set{Int}} # indices to access cells and local_coords
end

function Base.show(io::IO, ::MIME"text/plain", ph::PointEvalHandler)
    println(io, typeof(ph))
    println(io, "  number of points: ", length(ph.local_coords))
    n_missing = sum(x -> x === nothing, ph.cells)
    if n_missing == 0
        print(io, "  Found corresponding cell for all points.")
    else
        print(io, "  Could not find corresponding cell for ", n_missing, " points.")
    end
end

function PointEvalHandler(grid::Grid, points::AbstractVector{Vec{dim,T}}) where {dim, T}
    node_cell_dicts = _get_node_cell_map(grid)
    cells, local_coords, cellsets = _get_cellcoords(points, grid, node_cell_dicts)
    return PointEvalHandler(grid, cells, local_coords, cellsets)
end

function _get_cellcoords(points::AbstractVector{Vec{dim,T}}, grid::Grid, node_cell_dicts::Dict{C,Dict{Int, Vector{Int}}}) where {dim, T<:Real, C}

    # set up tree structure for finding nearest nodes to points
    kdtree = KDTree(reinterpret(Vec{dim,T}, grid.nodes))
    nearest_nodes, _ = knn(kdtree, points, 3, true) #TODO 3 is a random value, it shouldn't matter because likely the nearest node is the one we want

    cells = Vector{Union{Nothing, Int}}(nothing, length(points))
    local_coords = Vector{Union{Nothing, Vec{dim, T}}}(nothing, length(points))
    cellset_sets = [Set{Int}() for _ in eachindex(node_cell_dicts)]

    for point_idx in 1:length(points)
        cell_found = false
        for (idx, (CT, node_cell_dict)) in enumerate(node_cell_dicts)
            geom_interpol = default_interpolation(CT)
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
                        push!(cellset_sets[idx], point_idx)
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
    tol = sqrt(eps(T))
    # All in the range [-1, 1]
    return all(x -> abs(x) - 1 < tol, x_local)
end

# check if point is inside a cell based on isoparametric coordinate
function _check_isoparametric_boundaries(::Type{RefTetrahedron}, x_local::Vec{dim, T}) where {dim, T}
    tol = sqrt(eps(T))
    # Positive and below the plane 1 - ξx - ξy - ξz
    return all(x -> x > -tol, x_local) && sum(x_local) - 1 < tol
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
        J = zero(Tensor{2, dim})
        for j in 1:n_basefuncs
            J += cell_coordinates[j] ⊗ dNdξ[j]
        end
        local_guess -= inv(J) ⋅ residual
    end
    return converged, local_guess
end

# return a Dict with a key for each node that contains a vector with the adjacent cells as value
function _get_node_cell_map(grid::Grid)
    C = eltype(grid.cells) # possibly abstract
    cell_dicts = Dict{Type{<:C}, Dict{Int, Vector{Int}}}()
    ctypes = Set{Type{<:C}}(typeof(c) for c in grid.cells)
    for ctype in ctypes
        cell_dict = cell_dicts[ctype] = Dict{Int,Vector{Int}}()
        for (cellidx, cell) in enumerate(grid.cells)
            cell isa ctype || continue
            for node in cell.nodes
                v = get!(Vector{Int}, cell_dict, node)
                push!(v, cellidx)
            end
        end
    end
    return cell_dicts
end

"""
TODO: Update docstring
    get_point_values(ph::PointEvalHandler, dof_values::Vector{T}, fieldname::Symbol, [func_interpolations::Vector{<:Interpolation}]) where T
    get_point_values(ph::PointEvalHandler, dof_values::Vector{T}, ::L2Projector, [func_interpolations::Vector{<:Interpolation}]) where T

Return a `Vector{T}` (for a 1-dimensional field) or a `Vector{Vec{fielddim, T}}` (for a vector field) with the field values of field `fieldname` in the points of the `PointEvalHandler`. The field values are computed based on the `dof_values` and interpolated to the local coordinates by the `func_interpolations`.
The `dof_values` must be given in the order of the dofs that corresponds to the `AbstractDofHandler` which was used to construct `ph`. If the `dof_values` are obtained by a `L2Projector`, the `L2Projector` can be handed over instead of the `fieldname`. 

If no function interpolations are given, the field interpolations are extracted from the `AbstractDofHandler` of `ph`. For using custom function interpolations hand in a vector of interpolations,
whose order corresponds to the order of `FieldHandlers` within the `MixedDofHandler` that was used for constructing `ph`. For a `DofHandler`, give a Vector with your function interpolation as single entry.
"""
get_point_values

function get_point_values(ph::PointEvalHandler, proj::L2Projector, dof_vals::AbstractVector)
    get_point_values(ph, proj.dh, dof_vals)
end

function get_point_values(ph::PointEvalHandler, dh::AbstractDofHandler, dof_vals::AbstractVector{T},
                           fname::Symbol=find_single_field(dh)) where {T}
    fdim = getfielddim(dh, fname)
    npoints = length(ph.cells)
    # for a scalar field return a Vector of Scalars, for a vector field return a Vector of Vecs
    if fdim == 1
        out_vals = fill!(Vector{T}(undef, npoints), T(NaN))
    else
        nanv = convert(Vec{fdim,T}, NaN * zero(Vec{fdim,T}))
        out_vals = fill!(Vector{Vec{fdim, T}}(undef, npoints), nanv)
    end
    func_interpolations = get_func_interpolations(dh, fname)
    get_point_values!(out_vals, ph, dh, dof_vals, fname, func_interpolations)
    return out_vals
end
function find_single_field(dh)
    ns = getfieldnames(dh)
    if length(ns) != 1
        throw(ArgumentError("multiple fields in DoF handler, must specify which"))
    end
    return ns[1]
end

# values in dof-order. They must be obtained from the same DofHandler that was used for constructing the PointEvalHandler
function get_point_values!(out_vals::Vector{T2},
    ph::PointEvalHandler,
    dh::MixedDofHandler,
    dof_vals::Vector{T},
    fname::Symbol,
    func_interpolations
    ) where {T2, T} 

    # TODO: I don't think this is correct??
    # length(dof_vals) == ndofs(dh) || error("You must supply nodal values for all $(ndofs(dh)) dofs.")

    fdim = getfielddim(dh, fname)
    
    for fh_idx in eachindex(dh.fieldhandlers)
        ip = func_interpolations[fh_idx]
        if ip !== nothing
            _get_point_values!(out_vals, dof_vals, ph, dh, ip, fh_idx, fname, Val(fdim))
        end
    end
    _set_missing_vals!(out_vals, ph)
    return out_vals
end

function get_point_values!(out_vals::Vector{T2},
    ph::PointEvalHandler,
    dh::DofHandler,
    dof_vals::Vector{T},
    fname::Symbol,
    func_interpolations
    ) where {T2, T} 

    # TODO: I don't think this is correct??
    length(dof_vals) == ndofs(dh) || error("You must supply nodal values for all $(ndofs(dh)) dofs.")

    fdim = getfielddim(dh, fname)
    _get_point_values!(out_vals, dof_vals, ph, dh, func_interpolations[1], 1, fname, Val(fdim))
    _set_missing_vals!(out_vals, ph)
    return out_vals
end

# this is just a function barrier (https://youtu.be/_lK4cX5xGiQ)
function _get_point_values!(
    out_vals::Vector{T2},
    dof_vals::Vector{T},
    ph::PointEvalHandler,
    dh::AbstractDofHandler,
    ip::Interpolation,
    fh_idx::Int,
    fieldname::Symbol,
    fdim::Val{fielddim}) where {T2,T,fielddim}

    # extract variables
    local_coords = ph.local_coords
    point_idx_set = ph.pointidx_sets[fh_idx]
    # preallocate some stuff specific to this cellset
    pv = PointScalarValues(first(local_coords), ip)
    cell_dofs = celldofs(dh, ph.cells[first(point_idx_set)])
    dofrange = _dof_range(dh, fieldname, fh_idx)

    # compute point values
    for pointid in point_idx_set
        celldofs!(cell_dofs, dh, ph.cells[pointid])
        reinit!(pv, local_coords[pointid], ip)
        @inbounds @views dof_vals_reshaped = _change_format(fdim, dof_vals[cell_dofs[dofrange]])
        out_vals[pointid] = function_value(pv, 1, dof_vals_reshaped)
    end
    return out_vals
end

###################################################################################################################
# utils 

# set NaNs for points where no cell was found
function _set_missing_vals!(vals::Vector{T}, ph::PointEvalHandler) where T
    # TODO: Remove this 
    return
    for (idx, cell) in enumerate(ph.cells)
        cell !== nothing && continue
        vals[idx] *= NaN
    end
    return vals
end

# reshape dof_values based on fielddim
_change_format(::Val{1}, dof_values::AbstractVector{T}) where T = dof_values
_change_format(::Val{fielddim}, dof_values::AbstractVector{T}) where {fielddim, T} = reinterpret(Vec{fielddim, T}, dof_values)

## work-arounds so that we can call stuff with the same syntax for DofHandler and MixedDofHandler
_dof_range(dh::DofHandler, fieldname, ::Int) = dof_range(dh, fieldname)
_dof_range(dh::MixedDofHandler, fieldname, fh_idx::Int) = dof_range(dh.fieldhandlers[fh_idx], fieldname)

get_default_geom_interpolations(grid::Grid) = (error(); [default_interpolation(typeof(first(dh.grid.cells)))])
function get_default_geom_interpolations(dh::MixedDofHandler{dim}) where {dim}
    ips = Interpolation{dim}[]
    error()
    for fh in dh.fieldhandlers
        push!(ips, default_interpolation(typeof(dh.grid.cells[first(fh.cellset)])))
    end
    return ips
end

get_func_interpolations(dh::DH, fieldname) where DH<:DofHandler = [getfieldinterpolation(dh, find_field(dh, fieldname))]
function get_func_interpolations(dh::DH, fieldname) where DH<:MixedDofHandler
    func_interpolations = Union{Interpolation,Nothing}[]
    for fh in dh.fieldhandlers
        j = findfirst(i -> i === fieldname, getfieldnames(fh))
        if j === nothing
            push!(func_interpolations, missing)
        else
            push!(func_interpolations, fh.fields[j].interpolation)
        end
    end
    return func_interpolations
end
