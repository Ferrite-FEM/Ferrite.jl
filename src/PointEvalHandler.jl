Base.@kwdef struct NewtonLineSearchPointFinder{T}
    max_iters::Int = 10
    max_line_searches::Int = 5
    residual_tolerance::T = 1.0e-10
end

"""
    PointEvalHandler(grid::Grid, points::AbstractVector{Vec{dim, T}}; kwargs...) where {dim, T}

The `PointEvalHandler` can be used for function evaluation in *arbitrary points* in the
domain -- not just in quadrature points or nodes.

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

struct PointEvalHandler{G, T <: Real}
    grid::G
    cells::Vector{Union{Nothing, Int}}
    local_coords::Vector{Union{Nothing, Vec{1, T}, Vec{2, T}, Vec{3, T}}}
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
    return
end

# Internals:
# `PointEvalHandler` takes the following keyword arguments:
#  - `search_nneighbors`: How many nodes should be found in the nearest neighbor search for each
#    point. Usually there is no need to change this setting. Default value: `3`.
#  - `warn::Bool`: Show a warning if a point is not found. Default value: `true`.
#  - `newton_max_iters::Int`: Maximum number of inner Newton iterations. Default value: `10`.
#  - `newton_residual_tolerance`: Tolerance for the residual norm to indicate convergence in the
#    inner Newton solver. Default value: `1e-10`.
function PointEvalHandler(grid::AbstractGrid{dim}, points::AbstractVector{Vec{dim, T}}; search_nneighbors = 3, warn::Bool = true, strategy = NewtonLineSearchPointFinder()) where {dim, T}
    T_local = float(promote_type(T, get_coordinate_eltype(grid)))
    local_points = Vec{dim, T_local}[Vec{dim, T_local}(Tuple(point)) for point in points]
    node_cell_dicts = _get_node_cell_map(grid)
    cells, local_coords = _get_cellcoords(local_points, grid, node_cell_dicts, search_nneighbors, warn, strategy)
    return PointEvalHandler(grid, cells, local_coords)
end

function _get_cellcoords(points::AbstractVector{Vec{dim, T}}, grid::AbstractGrid, node_cell_dicts::Dict{C, Dict{Int, Vector{Int}}}, search_nneighbors, warn, strategy::NewtonLineSearchPointFinder) where {dim, T <: Real, C}
    # set up tree structure for finding nearest nodes to points
    node_coordinates = Vec{dim, T}[Vec{dim, T}(Tuple(get_node_coordinate(node))) for node in getnodes(grid)]
    kdtree = KDTree(node_coordinates)
    nearest_nodes, _ = knn(kdtree, points, search_nneighbors, true)

    cells = Vector{Union{Nothing, Int}}(nothing, length(points))
    local_coords = Vector{Union{Nothing, Vec{1, T}, Vec{2, T}, Vec{3, T}}}(nothing, length(points))

    for point_idx in 1:length(points)
        cell_found = false
        for (CT, node_cell_dict) in node_cell_dicts
            geom_interpol = geometric_interpolation(CT)
            # loop over points
            for node in nearest_nodes[point_idx]
                possible_cells = get(node_cell_dict, node, nothing)
                possible_cells === nothing && continue # if node is not part of the subdofhandler, try the next node
                for cell in possible_cells
                    cell_coords = getcoordinates(grid, cell)
                    is_in_cell, local_coord = find_local_coordinate(geom_interpol, cell_coords, points[point_idx], strategy; warn)
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

# check if point is inside a cell based on isoparametric coordinate
function check_isoparametric_boundaries(::Type{RefHypercube{dim}}, x_local::Vec{dim, T}, tol) where {dim, T}
    # All in the range [-1, 1]^dim
    return all(x -> abs(x) - 1 ≤ tol, x_local)
end

# check if point is inside a cell based on isoparametric coordinate
function check_isoparametric_boundaries(::Type{RefSimplex{dim}}, x_local::Vec{dim, T}, tol) where {dim, T}
    # Positive and below the plane 1 - ξx - ξy - ξz
    return all(x -> x >= -tol, x_local) && sum(x_local) <= 1 + tol
end

cellcenter(::Type{<:RefHypercube{dim}}, _::Type{T}) where {dim, T} = zero(Vec{dim, T})
cellcenter(::Type{<:RefSimplex{dim}}, _::Type{T}) where {dim, T} = Vec(ntuple(_ -> one(T) / (dim + 1), dim))

function project_to_reference_cell(::Type{<:RefHypercube{dim}}, x::Vec{dim, T}) where {dim, T}
    return Vec(ntuple(i -> clamp(x[i], -one(T), one(T)), dim))
end

function project_to_reference_cell(::Type{<:RefSimplex{dim}}, x::Vec{dim, T}) where {dim, T}
    xp = Vec(ntuple(i -> max(x[i], zero(T)), dim))
    sum(xp) <= one(T) && return xp

    # Euclidean projection onto the probability simplex. Reference simplices
    # also contain points with sum(x) < 1, handled by the early return above.
    u = sort(Tuple(x); rev = true)
    cumulative_sum = zero(T)
    threshold = zero(T)
    for i in eachindex(u)
        cumulative_sum += u[i]
        candidate_threshold = (cumulative_sum - one(T)) / i
        u[i] > candidate_threshold && (threshold = candidate_threshold)
    end
    return Vec(ntuple(i -> max(x[i] - threshold, zero(T)), dim))
end

function check_isoparametric_boundaries(::Type{RefPrism}, x_local::Vec{3}, tol)
    x, y, z = x_local
    return x >= -tol && y >= -tol && x + y <= 1 + tol && z >= -tol && z <= 1 + tol
end
cellcenter(::Type{RefPrism}, ::Type{T}) where {T} = Vec((one(T) / 3, one(T) / 3, one(T) / 2))
function project_to_reference_cell(::Type{RefPrism}, x::Vec{3, T}) where {T}
    xy = project_to_reference_cell(RefTriangle, Vec((x[1], x[2])))
    return Vec((xy[1], xy[2], clamp(x[3], zero(T), one(T))))
end

function check_isoparametric_boundaries(::Type{RefPyramid}, x_local::Vec{3}, tol)
    x, y, z = x_local
    return x >= -tol && y >= -tol && z >= -tol && x + z <= 1 + tol && y + z <= 1 + tol
end
cellcenter(::Type{RefPyramid}, ::Type{T}) where {T} = Vec((3 * one(T) / 8, 3 * one(T) / 8, one(T) / 4))
function project_to_reference_cell(::Type{RefPyramid}, x::Vec{3, T}) where {T}
    z = clamp(x[3], zero(T), one(T))
    upper = one(T) - z
    return Vec((clamp(x[1], zero(T), upper), clamp(x[2], zero(T), upper), z))
end

function check_point_search_convergence(refshape, local_guess, J, residual_norm, strategy, boundary_tolerance, warn)
    residual_norm <= strategy.residual_tolerance || return false
    check_isoparametric_boundaries(refshape, local_guess, boundary_tolerance) || return false
    detJ = calculate_detJ(J)
    if detJ <= zero(detJ)
        warn && @warn "det(J) is not positive at the converged point; rejecting the cell" detJ
        return false
    end
    return true
end

function refine_local_coordinate(interpolation, cell_coordinates, global_coordinate, local_guess)
    # Projected steps select the correct root but have non-smooth derivatives at
    # reference-cell boundaries. Two unconstrained Newton steps at the selected
    # root restore implicit derivatives (used by AD).
    for _ in 1:2
        J, global_guess = calculate_jacobian_and_spatial_coordinate(interpolation, local_guess, cell_coordinates)
        local_guess -= calculate_Jinv(J) ⋅ (global_guess - global_coordinate)
    end
    return local_guess
end

# See https://discourse.julialang.org/t/finding-the-value-of-a-field-at-a-spatial-location-in-juafem/38975/2
function find_local_coordinate(interpolation::Interpolation{refshape}, cell_coordinates::Vector{<:Vec{sdim}}, global_coordinate::Vec{sdim}, strategy::NewtonLineSearchPointFinder; warn::Bool = false) where {sdim, refshape}
    boundary_tolerance = √(strategy.residual_tolerance)

    T = promote_type(eltype(cell_coordinates[1]), eltype(global_coordinate))
    n_basefuncs = getnbasefunctions(interpolation)
    @assert length(cell_coordinates) == n_basefuncs
    local_guess = cellcenter(refshape, T)
    for iter in 1:strategy.max_iters
        # Setup J(ξ) and x(ξ)
        J, global_guess = calculate_jacobian_and_spatial_coordinate(interpolation, local_guess, cell_coordinates)
        # Check if converged
        residual = global_guess - global_coordinate
        best_residual_norm = norm(residual) # for line search below
        # Early convergence check
        best_residual_norm ≤ strategy.residual_tolerance && break
        Δξ = calculate_Jinv(J) ⋅ residual # J \ residual

        # Backtrack to the best projected trial point that decreases the residual.
        # Projection is needed to converge to points on curved cell boundaries:
        # simply rejecting an exterior Newton step can make the iteration stall.
        best_index = 0
        best_local_guess = local_guess
        for next_index in 1:strategy.max_line_searches
            new_local_guess = project_to_reference_cell(refshape, local_guess - Δξ / 2^(next_index - 1))
            global_guess = spatial_coordinate(interpolation, new_local_guess, cell_coordinates)
            residual_norm = norm(global_guess - global_coordinate)
            if residual_norm < best_residual_norm
                best_residual_norm = residual_norm
                best_index = next_index
                best_local_guess = new_local_guess
            end
        end
        best_index == 0 && break
        local_guess = best_local_guess
        # Late convergence check
        best_residual_norm ≤ strategy.residual_tolerance && break
        if iter == strategy.max_iters
            @debug println("Failed to converge in $(strategy.max_iters) iterations")
        end
    end

    J, global_guess = calculate_jacobian_and_spatial_coordinate(interpolation, local_guess, cell_coordinates)
    residual_norm = norm(global_guess - global_coordinate)
    if residual_norm <= boundary_tolerance
        local_guess = refine_local_coordinate(interpolation, cell_coordinates, global_coordinate, local_guess)
        J, global_guess = calculate_jacobian_and_spatial_coordinate(interpolation, local_guess, cell_coordinates)
        residual_norm = norm(global_guess - global_coordinate)
    end
    converged = check_point_search_convergence(refshape, local_guess, J, residual_norm, strategy, boundary_tolerance, warn)
    if converged
        @debug println("Local point finder converged with residual $residual_norm to $local_guess")
    else
        @debug println("Local point finder failed to converge inside the element: residual $residual_norm, coordinate $local_guess")
    end
    return converged, local_guess
end

# return a Dict with a key for each node that contains a vector with the adjacent cells as value
function _get_node_cell_map(grid::AbstractGrid)
    cells = getcells(grid)
    C = eltype(cells) # possibly abstract
    cell_dicts = Dict{Type{<:C}, Dict{Int, Vector{Int}}}()
    for (cellidx, cell) in enumerate(cells)
        cell_dict = get!(cell_dicts, typeof(cell)) do
            Dict{Int, Vector{Int}}()
        end
        for node in cell.nodes
            v = get!(Vector{Int}, cell_dict, node)
            push!(v, cellidx)
        end
    end
    return cell_dicts
end

"""
    evaluate_at_points(ph::PointEvalHandler, dh::AbstractDofHandler, dof_values::AbstractVector{T}, [fieldname::Symbol]) where {T}
    evaluate_at_points(ph::PointEvalHandler, proj::L2Projector, dof_values::AbstractVector{T}) where {T}

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
    return evaluate_at_points(ph, proj.dh, dof_vals)
end

function evaluate_at_points(
        ph::PointEvalHandler{<:Any, T1}, dh::AbstractDofHandler, dof_vals::AbstractVector{T2},
        fname::Symbol = find_single_field(dh)
    ) where {T1, T2}
    npoints = length(ph.cells)
    # Figure out the value type by creating a dummy PointValues
    ip = getfieldinterpolation(dh, find_field(dh, fname))
    pv = PointValues(T1, ip; update_gradients = Val(false))
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
function evaluate_at_points!(
        out_vals::AbstractVector{T2},
        ph::PointEvalHandler{<:Any, T_ph},
        dh::DofHandler,
        dof_vals::AbstractVector{T},
        fname::Symbol,
        func_interpolations
    ) where {T2, T_ph, T}

    # TODO: I don't think this is correct??
    length(dof_vals) == ndofs(dh) || error("You must supply values for all $(ndofs(dh)) dofs.")

    for (sdh_idx, sdh) in pairs(dh.subdofhandlers)
        ip = func_interpolations[sdh_idx]
        if ip !== nothing
            dofrange = dof_range(sdh, fname)
            cellset = sdh.cellset
            ip_geo = geometric_interpolation(getcelltype(sdh))

            pv = PointValues(T_ph, ip, ip_geo; update_gradients = Val(false))
            _evaluate_at_points!(out_vals, dof_vals, ph, dh, pv, cellset, dofrange)
        end
    end
    return out_vals
end

# function barrier with concrete type of PointValues
function _evaluate_at_points!(
        out_vals::AbstractVector{T2},
        dof_vals::AbstractVector{T},
        ph::PointEvalHandler,
        dh::AbstractDofHandler,
        pv::PointValues,
        cellset::Union{Nothing, AbstractSet{Int}},
        dofrange::AbstractRange{Int},
    ) where {T2, T}

    # extract variables
    local_coords = ph.local_coords

    # preallocate some stuff specific to this cellset
    idx = findfirst(!isnothing, local_coords)
    idx === nothing && return out_vals

    grid = get_grid(dh)
    first_cell = cellset === nothing ? 1 : first(cellset)
    cell_dofs = Vector{Int}(undef, ndofs_per_cell(dh, first_cell))
    u_e = Vector{T}(undef, ndofs_per_cell(dh, first_cell))
    x = getcoordinates(grid, first_cell)

    # compute point values
    for pointid in eachindex(ph.cells)
        cellid = ph.cells[pointid]
        cellid === nothing && continue # next point if no cell was found for this one
        cellset !== nothing && (cellid ∈ cellset || continue) # no need to check the cellset for a regular DofHandler
        celldofs!(cell_dofs, dh, ph.cells[pointid])
        for (i, I) in pairs(cell_dofs)
            u_e[i] = dof_vals[I]
        end
        getcoordinates!(x, grid, cellid)
        reinit!(pv, x, local_coords[pointid])
        out_vals[pointid] = function_value(pv, 1, u_e, dofrange)
    end
    return out_vals
end

function get_func_interpolations(dh::DofHandler, fieldname)
    func_interpolations = Union{Interpolation, Nothing}[]
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

struct PointIterator{PH <: PointEvalHandler, V <: Vec}
    ph::PH
    coords::Vector{V}
end

function PointIterator(ph::PointEvalHandler{G}) where {D, C, T, G <: Grid{D, C, T}}
    n = nnodes_per_cell(ph.grid)
    coords = zeros(Vec{D, T}, n) # resize!d later if needed
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
