"""
    PointEvalHandler(grid::Grid, points::AbstractVector{Vec{dim,T}}; kwargs...) where {dim, T}

The `PointEvalHandler` can be used for function evaluation in *arbitrary points* in the
domain -- not just in quadrature points or nodes.

The `PointEvalHandler` takes the following keyword arguments:
 - `search_nneighbors`: How many nodes should be found in the nearest neighbor search for each
   point. Usually there is no need to change this setting. Default value: `3`.
 - `warn`: Show a warning if a point is not found. Default value: `true`.

The constructor takes a grid and a vector of coordinates for the points. The
`PointEvalHandler` computes i) the corresponding cell, and ii) the (local) coordinate
within the cell, for each point. The fields of the `PointEvalHandler` are:
 - `cells::Vector{Union{Int,Nothing}}`: vector with cell IDs for the points, with `nothing`
   for points that could not be found.
 - `local_coords::Vector{Union{Vec,Nothing}}`: vector with the local coordinates
   (i.e. coordinates in the reference configuration) for the points, with `nothing` for
   points that could not be found.

There are two ways to use the `PointEvalHandler` to evaluate functions:

 - [`get_point_values`](@ref): can be used when the function is described by
   i) a `dh::DofHandler` + `uh::Vector` (for example the FE-solution), or
   ii) a `p::L2Projector` + `ph::Vector` (for projected data).
 - Iteration with [`PointIterator`](@ref) + [`PointValues`](@ref): can be used for more
   flexible evaluation in the points, for example to compute gradients.
"""
PointEvalHandler

struct PointEvalHandler{G,dim,T<:Real}
    grid::G
    cells::Vector{Union{Nothing, Int}}
    local_coords::Vector{Union{Nothing, Vec{dim,T}}}
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

function PointEvalHandler(grid::AbstractGrid, points::AbstractVector{Vec{dim,T}}; search_nneighbors=3, warn=true) where {dim, T}
    node_cell_dicts = _get_node_cell_map(grid)
    cells, local_coords = _get_cellcoords(points, grid, node_cell_dicts, search_nneighbors, warn)
    return PointEvalHandler(grid, cells, local_coords)
end

function _get_cellcoords(points::AbstractVector{Vec{dim,T}}, grid::AbstractGrid, node_cell_dicts::Dict{C,Dict{Int, Vector{Int}}}, search_nneighbors, warn) where {dim, T<:Real, C}

    # set up tree structure for finding nearest nodes to points
    kdtree = KDTree(reinterpret(Vec{dim,T}, getnodes(grid)))
    nearest_nodes, _ = knn(kdtree, points, search_nneighbors, true) 

    cells = Vector{Union{Nothing, Int}}(nothing, length(points))
    local_coords = Vector{Union{Nothing, Vec{dim, T}}}(nothing, length(points))

    for point_idx in 1:length(points)
        cell_found = false
        for (CT, node_cell_dict) in node_cell_dicts
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
                        break
                    end
                end
                cell_found && break
            end
            cell_found && break
        end
        if !cell_found && warn
            @warn("No cell found for point number $point_idx, coordinate: $(points[point_idx]).")
        end
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
    for _ in 1:max_iters
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
function _get_node_cell_map(grid::AbstractGrid)
    cells = getcells(grid)
    C = eltype(cells) # possibly abstract
    cell_dicts = Dict{Type{<:C}, Dict{Int, Vector{Int}}}()
    ctypes = Set{Type{<:C}}(typeof(c) for c in cells)
    for ctype in ctypes
        cell_dict = cell_dicts[ctype] = Dict{Int,Vector{Int}}()
        for (cellidx, cell) in enumerate(cells)
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
    get_point_values(ph::PointEvalHandler, dh::AbstractDofHandler, dof_values::Vector{T}, [fieldname::Symbol]) where T
    get_point_values(ph::PointEvalHandler, proj::L2Projector, dof_values::Vector{T}) where T

Return a `Vector{T}` (for a 1-dimensional field) or a `Vector{Vec{fielddim, T}}` (for a
vector field) with the field values of field `fieldname` in the points of the
`PointEvalHandler`. The `fieldname` can be omitted if only one field is stored in `dh`.
The field values are computed based on the `dof_values` and interpolated to the local
coordinates by the function interpolation of the corresponding `field` stored in the
`AbstractDofHandler` or the `L2Projector`.

Points that could not be found in the domain when constructing the `PointEvalHandler` will
have `NaN`s for the corresponding entries in the output vector.
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
        out_vals = fill!(Vector{T}(undef, npoints), NaN * zero(T))
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
    length(dof_vals) == ndofs(dh) || error("You must supply values for all $(ndofs(dh)) dofs.")

    fdim = getfielddim(dh, fname)
    
    for fh_idx in eachindex(dh.fieldhandlers)
        ip = func_interpolations[fh_idx]
        if ip !== nothing
            dofrange = dof_range(dh.fieldhandlers[fh_idx], fname)
            cellset = dh.fieldhandlers[fh_idx].cellset
            _get_point_values!(out_vals, dof_vals, ph, dh, ip, cellset, Val(fdim), dofrange)
        end
    end
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
    length(dof_vals) == ndofs(dh) || error("You must supply values for all $(ndofs(dh)) dofs.")

    fdim = getfielddim(dh, fname)
    dofrange = dof_range(dh, fname)
    _get_point_values!(out_vals, dof_vals, ph, dh, func_interpolations[1], nothing, Val(fdim), dofrange)
    return out_vals
end

# function barrier with concrete type of interpolation
function _get_point_values!(
    out_vals::Vector{T2},
    dof_vals::Vector{T},
    ph::PointEvalHandler,
    dh::AbstractDofHandler,
    ip::Interpolation,
    cellset::Union{Nothing, Set{Int}},
    fdim::Val{fielddim},
    dofrange::AbstractRange{Int},
    ) where {T2,T,fielddim}

    # extract variables
    local_coords = ph.local_coords
    # preallocate some stuff specific to this cellset
    idx = findfirst(!isnothing, local_coords)
    idx === nothing && return out_vals
    pv = PointScalarValuesInternal(local_coords[idx], ip)
    first_cell = cellset === nothing ? 1 : first(cellset)
    cell_dofs = Vector{Int}(undef, ndofs_per_cell(dh, first_cell))

    # compute point values
    for pointid in eachindex(ph.cells)
        cellid = ph.cells[pointid]
        cellid === nothing && continue # next point if no cell was found for this one
        cellset !== nothing && (cellid ∈ cellset || continue) # no need to check the cellset for a regular DofHandler
        celldofs!(cell_dofs, dh, ph.cells[pointid])
        reinit!(pv, local_coords[pointid], ip)
        @inbounds @views dof_vals_reshaped = _change_format(fdim, dof_vals[cell_dofs[dofrange]])
        out_vals[pointid] = function_value(pv, 1, dof_vals_reshaped)
    end
    return out_vals
end

###################################################################################################################
# utils 

# reshape dof_values based on fielddim
_change_format(::Val{1}, dof_values::AbstractVector{T}) where T = dof_values
_change_format(::Val{fielddim}, dof_values::AbstractVector{T}) where {fielddim, T} = reinterpret(Vec{fielddim, T}, dof_values)

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

# Iteration of PointEvalHandler
"""
    PointIterator(ph::PointEvalHandler)

Create an iterator over the points in the [`PointEvalHandler`](@ref).
The elements of the iterator are either a [`PointLocation`](@ref), if the corresponding
point could be found in the grid, or `nothing`, if the point was not found.

A `PointLocation` can be used to query the cell ID with the `cellid` function, and can be used
to reinitialize [`PointValues`](@ref) with [`reinit!`](@ref).

# Examples
```julia
ph = PointEvalHandler(grid, points)

for point in PointIterator(ph)
    point === nothing && continue # Skip any points that weren't found
    reinit!(pointvalues, point)   # Update pointvalues
    # ...
end
```
"""
PointIterator

struct PointIterator{PH<:PointEvalHandler}
    ph::PH
    coords::Vector{Vec{2,Float64}}
end

function PointIterator(ph::PointEvalHandler{G}) where {D,C,T,G<:Grid{D,C,T}}
    n = nnodes_per_cell(ph.grid)
    coords = zeros(Vec{D,T}, n) # resize!d later if needed
    return PointIterator(ph, coords)
end

"""
    PointLocation

Element of a [`PointIterator`](@ref), typically used to reinitialize
[`PointValues`](@ref). Fields:
 - `cid::Int`: ID of the cell containing the point
 - `local_coord::Vec`: the local (reference) coordinate of the point
 - `coords::Vector{Vec}`: the coordinates of the cell
"""
struct PointLocation{V}
    cid::Int
    local_coord::V
    coords::Vector{V}
end

function Base.iterate(p::PointIterator, state = 1)
    if state > length(p.ph.cells)
        return nothing
    elseif p.ph.cells[state] === nothing
        return (nothing, state + 1)
    else
        cid = (p.ph.cells[state])::Int
        local_coord = (p.ph.local_coords[state])::Vec
        n = nnodes_per_cell(p.ph.grid, cid)
        cellcoords!(resize!(p.coords, n), p.ph.grid, cid)
        point = PointLocation(cid, local_coord, p.coords)
        return (point, state + 1)
    end
end

cellid(p::PointLocation) = p.cid

function reinit!(pv::PointValues, point::PointLocation)
    reinit!(pv, point.coords, point.local_coord)
    return pv
end
