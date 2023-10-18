"""
    PointEvalHandler(grid::Grid, points::AbstractVector{Vec{dim,T}}; kwargs...) where {dim, T}

The `PointEvalHandler` can be used for function evaluation in *arbitrary points* in the
domain -- not just in quadrature points or nodes.

The `PointEvalHandler` takes the following keyword arguments:
 - `search_nneighbors`: How many nodes should be found in the nearest neighbor search for each
   point. Usually there is no need to change this setting. Default value: `3`.
 - `warn`: Show a warning if a point is not found. Default value: `true`.
 - `newton_max_iters`: Maximum number of inner Newton iterations. Default value: `10`.
 - `newton_residual_tolerance`: Tolerance for the residual norm to indicate convergence in the
   inner Newton solver. Default value: `1e-10`.

The constructor takes a grid and a vector of coordinates for the points. The
`PointEvalHandler` computes i) the corresponding cell, and ii) the (local) coordinate
within the cell, for each point. The fields of the `PointEvalHandler` are:
 - `cells::Vector{Union{Int,Nothing}}`: vector with cell IDs for the points, with `nothing`
   for points that could not be found.
 - `local_coords::Vector{Union{Vec,Nothing}}`: vector with the local coordinates
   (i.e. coordinates in the reference configuration) for the points, with `nothing` for
   points that could not be found.

There are two ways to use the `PointEvalHandler` to evaluate functions:

 - [`evaluate_at_points`](@ref): can be used when the function is described by
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

function PointEvalHandler(grid::AbstractGrid{dim}, points::AbstractVector{Vec{dim,T}}; search_nneighbors=3, warn=true, max_iters = 10, tol_norm::T = 1e-10) where {dim, T}
    node_cell_dicts = _get_node_cell_map(grid)
    cells, local_coords = _get_cellcoords(points, grid, node_cell_dicts, search_nneighbors, warn, #=linesearch_max_substeps=# 4, max_iters, tol_norm)
    return PointEvalHandler(grid, cells, local_coords)
end

function _get_cellcoords(points::AbstractVector{Vec{dim,T}}, grid::AbstractGrid, node_cell_dicts::Dict{C,Dict{Int, Vector{Int}}}, search_nneighbors, warn, linesearch_max_substeps, max_iters, tol_norm) where {dim, T<:Real, C}
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
                possible_cells === nothing && continue # if node is not part of the subdofhandler, try the next node
                for cell in possible_cells
                    cell_coords = getcoordinates(grid, cell)
                    is_in_cell, local_coord = point_in_cell(geom_interpol, cell_coords, points[point_idx], warn, linesearch_max_substeps, max_iters, tol_norm::T)
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
# TODO linear case can be handled easier
function point_in_cell(geom_interpol::Interpolation{shape}, cell_coordinates, global_coordinate, warn,  linesearch_max_substeps, max_iters, tol_norm) where {shape}
    return find_local_coordinate(geom_interpol, cell_coordinates, global_coordinate, warn, linesearch_max_substeps, max_iters, tol_norm)
end

# check if point is inside a cell based on isoparametric coordinate
function _check_isoparametric_boundaries(::Type{RefHypercube{dim}}, x_local::Vec{dim, T}, tol::T) where {dim, T}
    # All in the range [-1, 1]
    return all(x -> abs(x) - 1 ≤ tol, x_local)
end

# check if point is inside a cell based on isoparametric coordinate
function _check_isoparametric_boundaries(::Type{RefSimplex{dim}}, x_local::Vec{dim, T}, tol::T) where {dim, T}
    # Positive and below the plane 1 - ξx - ξy - ξz
    return all(x -> x ≥ -tol, x_local) && sum(x_local) - 1 < tol
end

cellcenter(::Type{<:RefHypercube{dim}}, _::Type{T}) where {dim, T} = zero(Vec{dim, T})
cellcenter(::Type{<:RefSimplex{dim}}, _::Type{T}) where {dim, T} = Vec{dim, T}((ntuple(d->1/3, dim)))

# TODO GeometryValues PR helper/Mapping?
function compute_J_and_x(interpolation::IP, ξ::Vec{dim,T}, cell_coordinates::Vector{<:Vec{dim, T}}) where {IP, T, dim}
    n_basefuncs = getnbasefunctions(interpolation)
    J = zero(Tensor{2, dim, T, 2^dim})
    x = zero(Vec{dim, T})
    for j in 1:n_basefuncs
        dNdξ, N = shape_gradient_and_value(interpolation, ξ, j)
        x += N * cell_coordinates[j]
        J += cell_coordinates[j] ⊗ dNdξ
    end
    return J, x
end

function compute_x(interpolation::IP, ξ::Vec{dim,T}, cell_coordinates::Vector{<:Vec{dim, T}}) where {IP, T, dim}
    n_basefuncs = getnbasefunctions(interpolation)
    x = zero(Vec{dim, T})
    for j in 1:n_basefuncs
        N = shape_value(interpolation, ξ, j)
        x += N * cell_coordinates[j]
    end
    return x
end

# See https://discourse.julialang.org/t/finding-the-value-of-a-field-at-a-spatial-location-in-juafem/38975/2
function find_local_coordinate(interpolation::IP, cell_coordinates::Vector{V}, global_coordinate::V, warn::Bool, linesearch_max_substeps::Int , max_iters::Int, tol_norm::T) where {dim, T, V <: Vec{dim, T}, ref_shape, IP <: Interpolation{ref_shape}}
    n_basefuncs = getnbasefunctions(interpolation)
    @assert length(cell_coordinates) == n_basefuncs
    local_guess = cellcenter(ref_shape, T)
    converged = false
    for iter in 1:max_iters
        # Check if still inside element
        _check_isoparametric_boundaries(ref_shape, local_guess, sqrt(tol_norm)) || break
        # Setup J(ξ) and x(ξ)
        J, global_guess = compute_J_and_x(interpolation, local_guess, cell_coordinates)
        # Check if converged
        residual = global_guess - global_coordinate
        best_residual_norm = norm(residual) # for line search below
        if best_residual_norm ≤ tol_norm
            converged = true
            break
        end
        if det(J) ≤ 0.0
            warn && @warn "det(J) negative! Aborting! $(det(J))" 
            break
        end
        Δξ = inv(J) ⋅ residual
        # Do line search if outside/on boundary
        best_index = 1
        new_local_guess = local_guess - Δξ
        global_guess = compute_x(interpolation, new_local_guess, cell_coordinates)
        !_check_isoparametric_boundaries(ref_shape, new_local_guess, sqrt(tol_norm)) && for next_index ∈ 2:linesearch_max_substeps
            new_local_guess = local_guess - Δξ/next_index
            global_guess = compute_x(interpolation, new_local_guess, cell_coordinates)
            residual_norm = norm(global_guess - global_coordinate)
            if residual_norm < best_residual_norm && _check_isoparametric_boundaries(ref_shape, new_local_guess, sqrt(tol_norm))
                best_residual_norm = residual_norm
                best_index = next_index
            else
                break
            end
        end
        local_guess -= Δξ / best_index
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
    evaluate_at_points(ph::PointEvalHandler, dh::AbstractDofHandler, dof_values::Vector{T}, [fieldname::Symbol]) where T
    evaluate_at_points(ph::PointEvalHandler, proj::L2Projector, dof_values::Vector{T}) where T

Return a `Vector{T}` (for a 1-dimensional field) or a `Vector{Vec{fielddim, T}}` (for a
vector field) with the field values of field `fieldname` in the points of the
`PointEvalHandler`. The `fieldname` can be omitted if only one field is stored in `dh`.
The field values are computed based on the `dof_values` and interpolated to the local
coordinates by the function interpolation of the corresponding `field` stored in the
`AbstractDofHandler` or the `L2Projector`.

Points that could not be found in the domain when constructing the `PointEvalHandler` will
have `NaN`s for the corresponding entries in the output vector.
"""
evaluate_at_points

function evaluate_at_points(ph::PointEvalHandler, proj::L2Projector, dof_vals::AbstractVector)
    evaluate_at_points(ph, proj.dh, dof_vals)
end

function evaluate_at_points(ph::PointEvalHandler{<:Any, dim, T1}, dh::AbstractDofHandler, dof_vals::AbstractVector{T2},
                           fname::Symbol=find_single_field(dh)) where {dim, T1, T2}
    npoints = length(ph.cells)
    # Figure out the value type by creating a dummy PointValuesInternal
    ip = getfieldinterpolation(dh, find_field(dh, fname))
    pv = PointValuesInternal(zero(Vec{dim, T1}), ip)
    zero_val = function_value_init(pv, dof_vals)
    # Allocate the output as NaNs
    nanv = convert(typeof(zero_val), NaN * zero_val)
    out_vals = fill(nanv, npoints)
    func_interpolations = get_func_interpolations(dh, fname)
    evaluate_at_points!(out_vals, ph, dh, dof_vals, fname, func_interpolations)
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
function evaluate_at_points!(out_vals::Vector{T2},
    ph::PointEvalHandler,
    dh::DofHandler,
    dof_vals::Vector{T},
    fname::Symbol,
    func_interpolations
    ) where {T2, T}

    # TODO: I don't think this is correct??
    length(dof_vals) == ndofs(dh) || error("You must supply values for all $(ndofs(dh)) dofs.")

    for (sdh_idx, sdh) in pairs(dh.subdofhandlers)
        ip = func_interpolations[sdh_idx]
        if ip !== nothing
            dofrange = dof_range(sdh, fname)
            cellset = sdh.cellset
            _evaluate_at_points!(out_vals, dof_vals, ph, dh, ip, cellset, dofrange)
        end
    end
    return out_vals
end

# function barrier with concrete type of interpolation
function _evaluate_at_points!(
    out_vals::Vector{T2},
    dof_vals::Vector{T},
    ph::PointEvalHandler,
    dh::AbstractDofHandler,
    ip::Interpolation,
    cellset::Union{Nothing, Set{Int}},
    dofrange::AbstractRange{Int},
    ) where {T2,T}

    # extract variables
    local_coords = ph.local_coords
    # preallocate some stuff specific to this cellset
    idx = findfirst(!isnothing, local_coords)
    idx === nothing && return out_vals
    pv = PointValuesInternal(local_coords[idx], ip)
    first_cell = cellset === nothing ? 1 : first(cellset)
    cell_dofs = Vector{Int}(undef, ndofs_per_cell(dh, first_cell))
    u_e = Vector{T}(undef, ndofs_per_cell(dh, first_cell))

    # compute point values
    for pointid in eachindex(ph.cells)
        cellid = ph.cells[pointid]
        cellid === nothing && continue # next point if no cell was found for this one
        cellset !== nothing && (cellid ∈ cellset || continue) # no need to check the cellset for a regular DofHandler
        celldofs!(cell_dofs, dh, ph.cells[pointid])
        for (i, I) in pairs(cell_dofs)
            u_e[i] = dof_vals[I]
        end
        reinit!(pv, local_coords[pointid])
        out_vals[pointid] = function_value(pv, 1, u_e, dofrange)
    end
    return out_vals
end

function get_func_interpolations(dh::DofHandler, fieldname)
    func_interpolations = Union{Interpolation,Nothing}[]
    for sdh in dh.subdofhandlers
        j = _find_field(sdh, fieldname)
        if j === nothing
            push!(func_interpolations, nothing)
        else
            push!(func_interpolations, sdh.field_interpolations[j])
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

struct PointIterator{PH<:PointEvalHandler, V <: Vec}
    ph::PH
    coords::Vector{V}
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
        getcoordinates!(resize!(p.coords, n), p.ph.grid, cid)
        point = PointLocation(cid, local_coord, p.coords)
        return (point, state + 1)
    end
end

cellid(p::PointLocation) = p.cid

function reinit!(pv::PointValues, point::PointLocation)
    reinit!(pv, point.coords, point.local_coord)
    return pv
end
