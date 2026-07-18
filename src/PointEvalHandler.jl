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

By default, a point is only assigned to a cell if it is inside the cell (up to a small
tolerance). Passing the keyword argument `extrapolation_tolerance > 0` also assigns points
whose local coordinate is at most this far outside the reference shape of a cell.
Evaluation in such points extrapolates from the assigned cell. Cells containing the point
are preferred; among extrapolation candidates the cell with the smallest violation
of the reference shape bounds (measured in reference coordinates) is chosen. Note that,
since the tolerance is measured in reference coordinates, the physical distance a given
tolerance corresponds to scales with the size of the cell (and depends on the reference
shape).

!!! note
    The search only considers cells connected to the `search_nneighbors` (default: 3)
    nodes nearest to each point. In meshes with large differences in cell size the cell
    containing a point may be missed by this search, in which case the point is not
    found -- or, with `extrapolation_tolerance > 0`, may be assigned to a nearby cell
    that does not contain it. Increasing `search_nneighbors` makes the search more
    robust.

The keyword argument `cellset` restricts the search to those cells. This is required to
obtain correct results when evaluating a field that is only defined on a subdomain, or
that is discontinuous across subdomain interfaces, since a point is otherwise assigned to
an arbitrary cell containing it, where the field may not be defined (evaluating to `NaN`)
or where it has the value from the "wrong" side of the interface.

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
    n_missing = count(x -> x === nothing, ph.cells)
    if n_missing == 0
        print(io, "  Found corresponding cell for all points.")
    else
        print(io, "  Could not find corresponding cell for ", n_missing, " points.")
    end
    return
end

# Internals:
# `PointEvalHandler` takes the following additional keyword arguments:
#  - `search_nneighbors`: How many nodes should be found in the nearest neighbor search for each
#    point. Usually there is no need to change this setting. Default value: `3`.
#  - `warn::Bool`: Show a warning if a point is not found. Default value: `true`.
#  - `strategy`: Method used to find the local coordinate of a point within a cell. Default
#    value: `NewtonLineSearchPointFinder()`, with the fields:
#     - `max_iters::Int`: Maximum number of Newton iterations. Default value: `10`.
#     - `max_line_searches::Int`: Maximum number of line search steps. Default value: `5`.
#     - `residual_tolerance`: Tolerance for the residual norm to indicate convergence in
#       the Newton solver. Default value: `1e-10`.
function PointEvalHandler(
        grid::AbstractGrid{dim}, points::AbstractVector{Vec{dim, T}};
        search_nneighbors = 3, warn::Bool = true, extrapolation_tolerance::Real = 0.0,
        cellset::Union{IntegerCollection, Nothing} = nothing,
        strategy = NewtonLineSearchPointFinder()
    ) where {dim, T}
    extrapolation_tolerance ≥ 0 || throw(ArgumentError("extrapolation_tolerance must be non-negative"))
    cset = cellset === nothing ? nothing : convert_to_orderedset(cellset)
    node_cell_dicts = _get_node_cell_map(grid, cset)
    cells, local_coords = _get_cellcoords(points, grid, node_cell_dicts, search_nneighbors, warn, extrapolation_tolerance, strategy, cset)
    return PointEvalHandler(grid, cells, local_coords)
end

function _get_cellcoords(points::AbstractVector{Vec{dim, T}}, grid::AbstractGrid, node_cell_dicts::Dict{C, Dict{Int, Vector{Int}}}, search_nneighbors, warn, extrapolation_tolerance, strategy::NewtonLineSearchPointFinder, cellset = nothing) where {dim, T <: Real, C}
    Tg = get_coordinate_eltype(grid)
    Tl = promote_type(T, Tg) # scalar type of the local coordinates
    # set up tree structure for finding nearest nodes to points
    if cellset === nothing
        kdtree = KDTree(reinterpret(Vec{dim, Tg}, getnodes(grid)))
        nearest_nodes, _ = knn(kdtree, points, min(search_nneighbors, getnnodes(grid)), true)
    else
        # Only nodes of cells in the cellset are relevant for the search
        node_ids = Int[]
        for cell_dict in values(node_cell_dicts)
            append!(node_ids, keys(cell_dict))
        end
        unique!(sort!(node_ids))
        if isempty(node_ids)
            nearest_nodes = [Int[] for _ in 1:length(points)]
        else
            node_coords = [get_node_coordinate(grid, i) for i in node_ids]
            sub_nearest, _ = knn(KDTree(node_coords), points, min(search_nneighbors, length(node_ids)), true)
            nearest_nodes = [node_ids[idxs] for idxs in sub_nearest]
        end
    end

    cells = Vector{Union{Nothing, Int}}(nothing, length(points))
    local_coords = Vector{Union{Nothing, Vec{1, Tl}, Vec{2, Tl}, Vec{3, Tl}}}(nothing, length(points))
    inside_tolerance = √(strategy.residual_tolerance)
    visited_cells = Set{Int}()
    cell_coords = zeros(Vec{dim, Tg}, 0) # resize!d per cell

    for point_idx in 1:length(points)
        cell_found = false
        best_violation = Tl(Inf)
        empty!(visited_cells)
        for (CT, node_cell_dict) in node_cell_dicts
            geom_interpol = geometric_interpolation(CT)
            refshape = getrefshape(geom_interpol)
            # loop over points
            for node in nearest_nodes[point_idx]
                possible_cells = get(node_cell_dict, node, nothing)
                possible_cells === nothing && continue # if node is not part of the subdofhandler, try the next node
                for cell in possible_cells
                    cell ∈ visited_cells && continue # cell shared with an already searched node
                    push!(visited_cells, cell)
                    cellobj = getcells(grid, cell)
                    getcoordinates!(resize!(cell_coords, nnodes(cellobj)), grid, cellobj)
                    is_in_cell, local_coord = find_local_coordinate(geom_interpol, cell_coords, points[point_idx], strategy; warn, extrapolation_tolerance)
                    is_in_cell || continue
                    violation = boundary_violation(refshape, local_coord)
                    if violation ≤ inside_tolerance
                        cell_found = true
                        cells[point_idx] = cell
                        local_coords[point_idx] = local_coord
                        break
                    elseif violation < best_violation
                        # Outside the cell but within extrapolation_tolerance: use the
                        # closest cell unless a cell containing the point is found later
                        best_violation = violation
                        cells[point_idx] = cell
                        local_coords[point_idx] = local_coord
                    end
                end
                cell_found && break
            end
            cell_found && break
        end
        if cells[point_idx] === nothing && warn
            @warn("No cell found for point number $point_idx, coordinate: $(points[point_idx]).")
        end
    end
    return cells, local_coords
end

# Maximum violation of the reference shape boundaries in local coordinates (≤ 0 if inside)
function boundary_violation(::Type{RefHypercube{dim}}, x_local::Vec{dim}) where {dim}
    # All in the range [-1, 1]^dim
    return maximum(abs, x_local) - 1
end

function boundary_violation(::Type{RefSimplex{dim}}, x_local::Vec{dim}) where {dim}
    # Positive and below the plane 1 - ξx - ξy - ξz
    return max(-minimum(x_local), sum(x_local) - 1)
end

function boundary_violation(::Type{RefPrism}, x_local::Vec{3})
    # Triangle in the ξx-ξy plane, ξz in the range [0, 1]
    (x, y, z) = x_local
    return max(-x, -y, x + y - 1, -z, z - 1)
end

function boundary_violation(::Type{RefPyramid}, x_local::Vec{3})
    # Unit square base in the ξx-ξy plane with the apex in (0, 0, 1)
    (x, y, z) = x_local
    return max(-x, -y, -z, x + z - 1, y + z - 1)
end

# check if point is inside a cell based on isoparametric coordinate
function check_isoparametric_boundaries(refshape::Type{<:AbstractRefShape}, x_local::Vec, tol)
    return boundary_violation(refshape, x_local) ≤ tol
end

cellcenter(::Type{<:RefHypercube{dim}}, _::Type{T}) where {dim, T} = zero(Vec{dim, T})
cellcenter(::Type{<:RefSimplex{dim}}, _::Type{T}) where {dim, T} = Vec{dim, T}((ntuple(d -> 1 / (dim + 1), dim))) # barycenter
cellcenter(::Type{RefPrism}, _::Type{T}) where {T} = Vec{3, T}((1 / 3, 1 / 3, 1 / 2))
cellcenter(::Type{RefPyramid}, _::Type{T}) where {T} = Vec{3, T}((3 / 8, 3 / 8, 1 / 4)) # centroid

# Check that a converged point is within the element boundaries (up to tolerance) and
# that the element is not geometrically broken (inverted) at the point. Note that a
# negative det(J) at intermediate Newton iterates is fine: the geometric map of a
# healthy element may fold outside the element. det(J) == 0 is also accepted since the
# map of a healthy element may be degenerate in single points (e.g. the pyramid apex).
function check_converged_point(interpolation::Interpolation{refshape}, cell_coordinates, local_guess, boundary_tolerance, warn) where {refshape}
    check_isoparametric_boundaries(refshape, local_guess, boundary_tolerance) || return false
    J, _ = calculate_jacobian_and_spatial_coordinate(interpolation, local_guess, cell_coordinates)
    detJ = calculate_detJ(J)
    if detJ < 0
        warn && @warn "det(J) negative at the converged point! Aborting! $detJ"
        return false
    end
    return true
end

# See https://discourse.julialang.org/t/finding-the-value-of-a-field-at-a-spatial-location-in-juafem/38975/2
function find_local_coordinate(interpolation::Interpolation{refshape}, cell_coordinates::Vector{<:Vec{sdim}}, global_coordinate::Vec{sdim}, strategy::NewtonLineSearchPointFinder; warn::Bool = false, extrapolation_tolerance::Real = 0.0) where {sdim, refshape}
    boundary_tolerance = max(√(strategy.residual_tolerance), extrapolation_tolerance)

    T = promote_type(eltype(cell_coordinates[1]), eltype(global_coordinate))
    n_basefuncs = getnbasefunctions(interpolation)
    @assert length(cell_coordinates) == n_basefuncs
    local_guess = cellcenter(refshape, T)
    converged = false
    for iter in 1:strategy.max_iters
        # Setup J(ξ) and x(ξ)
        J, global_guess = calculate_jacobian_and_spatial_coordinate(interpolation, local_guess, cell_coordinates)
        # Check if converged
        residual = global_guess - global_coordinate
        residual_norm = norm(residual)
        # Early convergence check
        if residual_norm ≤ strategy.residual_tolerance
            converged = check_converged_point(interpolation, cell_coordinates, local_guess, boundary_tolerance, warn)
            if converged
                @debug println("Local point finder converged in $iter iterations with residual $residual_norm to $local_guess")
            else
                @debug println("Local point finder converged in $iter iterations with residual $residual_norm to a point outside the element (or with a broken geometric map): $local_guess")
            end
            break
        end
        Δξ = calculate_Jinv(J) ⋅ residual # J \ residual

        # Do line search if the new guess is outside the element
        best_index = 1
        new_local_guess = local_guess - Δξ
        global_guess = spatial_coordinate(interpolation, new_local_guess, cell_coordinates)
        best_residual_norm = norm(global_guess - global_coordinate)
        if !check_isoparametric_boundaries(refshape, new_local_guess, boundary_tolerance)
            # Among the halved steps that stay inside the element, use the one with the
            # smallest residual, but only if it also improves on the full step. Otherwise
            # the full step is taken, and if the iteration converges to a point outside
            # the element the boundary check at convergence rejects the cell.
            for next_index in 2:strategy.max_line_searches
                new_local_guess = local_guess - Δξ / 2^(next_index - 1)
                global_guess = spatial_coordinate(interpolation, new_local_guess, cell_coordinates)
                residual_norm = norm(global_guess - global_coordinate)
                if residual_norm < best_residual_norm && check_isoparametric_boundaries(refshape, new_local_guess, boundary_tolerance)
                    best_residual_norm = residual_norm
                    best_index = next_index
                end
            end
        end
        local_guess -= Δξ / 2^(best_index - 1)
        # Late convergence check
        if best_residual_norm ≤ strategy.residual_tolerance
            converged = check_converged_point(interpolation, cell_coordinates, local_guess, boundary_tolerance, warn)
            if converged
                @debug println("Local point finder converged in $iter iterations with residual $best_residual_norm to $local_guess")
            else
                @debug println("Local point finder converged in $iter iterations with residual $best_residual_norm to a point outside the element (or with a broken geometric map): $local_guess")
            end
            break
        end
        if iter == strategy.max_iters
            @debug println("Failed to converge in $(strategy.max_iters) iterations")
        end
    end
    return converged, local_guess
end

# return a Dict with a key for each node that contains a vector with the adjacent cells as value
function _get_node_cell_map(grid::AbstractGrid, cellset::Union{AbstractSet{<:Integer}, Nothing} = nothing)
    cells = getcells(grid)
    C = eltype(cells) # possibly abstract
    cell_dicts = Dict{Type{<:C}, Dict{Int, Vector{Int}}}()
    cellidxs = cellset === nothing ? eachindex(cells) : sort!(collect(cellset))
    for cellidx in cellidxs
        cell = cells[cellidx]
        cell_dict = get!(Dict{Int, Vector{Int}}, cell_dicts, typeof(cell))
        for node in cell.nodes
            push!(get!(Vector{Int}, cell_dict, node), cellidx)
        end
    end
    return cell_dicts
end

"""
    evaluate_at_points(ph::PointEvalHandler, dh::DofHandler, dof_values::AbstractVector{T}, [fieldname::Symbol]) where {T}
    evaluate_at_points(ph::PointEvalHandler, proj::L2Projector, dof_values::AbstractVector{T}) where {T}

Return a `Vector{T}` (for a 1-dimensional field) or a `Vector{Vec{fielddim, T}}` (for a
vector field) with the field values of field `fieldname` in the points of the
`PointEvalHandler`. The `fieldname` can be omitted if only one field is stored in `dh`.
The field values are computed based on the `dof_values` and interpolated to the local
coordinates by the function interpolation of the corresponding `field` stored in the
`DofHandler` or the `L2Projector`.

Points that could not be found in the domain when constructing the `PointEvalHandler` will
have `NaN`s for the corresponding entries in the output vector.
"""
evaluate_at_points

function evaluate_at_points(ph::PointEvalHandler, proj::L2Projector, dof_vals::AbstractVector)
    return evaluate_at_points(ph, proj.dh, dof_vals)
end

function evaluate_at_points(
        ph::PointEvalHandler{<:Any, T1}, dh::DofHandler, dof_vals::AbstractVector{T2},
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
    n_undefined = count(ph.cells) do cellid
        cellid === nothing && return false
        return !any(
            i -> func_interpolations[i] !== nothing && cellid ∈ dh.subdofhandlers[i].cellset,
            eachindex(dh.subdofhandlers)
        )
    end
    if n_undefined > 0
        @warn "$n_undefined point(s) were assigned to cells where field :$fname is not defined and could not be evaluated. If the field is defined on a subdomain, construct the PointEvalHandler with the corresponding `cellset` to restrict the point search."
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

!!! note
    The `PointLocation`s returned by the iterator share a single cell coordinate buffer
    (similar to `CellIterator`): a `PointLocation` is only valid until the next
    iteration.

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
    coords = Vec{D, T}[] # resize!d in iterate to the size of each cell
    return PointIterator(ph, coords)
end

"""
    PointLocation

Element of a [`PointIterator`](@ref), typically used to reinitialize
[`PointValues`](@ref). Fields:
 - `cid::Int`: ID of the cell containing the point
 - `local_coord::Vec`: the local (reference) coordinate of the point
 - `coords::Vector{Vec}`: the coordinates of the cell

Note that the local coordinate and the cell coordinates are of different dimension for
embedded cells (e.g. `Line` cells in a two-dimensional grid).
"""
struct PointLocation{VL <: Vec, VC <: Vec}
    cid::Int
    local_coord::VL
    coords::Vector{VC}
end

Base.length(p::PointIterator) = length(p.ph.cells)

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
