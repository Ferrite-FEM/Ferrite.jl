function _geometric_interpolations(dh::DofHandler)
    sdhs = dh.subdofhandlers
    getcelltype(i) = typeof(getcells(get_grid(dh), first(sdhs[i].cellset)))
    return ntuple(i -> geometric_interpolation(getcelltype(i)), length(sdhs))
end

# Which interpolations `apply_analytical!` is implemented for; used to error before any
# dof value has been assigned so that `a` is never partially modified. `apply_analytical!`
# requires the dof values to be function values at the reference coordinates (so a
# `reference_coordinates` method must exist); `project_analytical!` covers H(div) and
# 2D H(curl) interpolations instead.
function _check_apply_analytical_supported(ip_fun::Interpolation)
    applicable(reference_coordinates, ip_fun) && return nothing
    if conformity(ip_fun) isa Union{HdivConformity, HcurlConformity}
        throw(ArgumentError("apply_analytical! is not applicable for $(ip_fun), whose dofs are not nodal function values. Use project_analytical! instead."))
    end
    error("apply_analytical! is not implemented for $(ip_fun).")
end

"""
    apply_analytical!(
        a::AbstractVector, dh::AbstractDofHandler, fieldname::Symbol,
        f::Function, cellset = 1:getncells(get_grid(dh))
    )

Apply a solution `f(x)` by modifying the values in the degree of freedom vector `a`
pertaining to the field `fieldname` for all cells in `cellset`.
The function `f(x)` is given the spatial coordinate
of the degree of freedom. For scalar fields, `f(x)::Number`,
and for vector fields with dimension `dim`, `f(x)::Vec{dim}`.

This function can be used to apply initial conditions for time dependent problems.

!!! note

    This function only works for standard nodal finite element interpolations
    when the function value at the (algebraic) node is equal to the corresponding
    degree of freedom value.
    This holds for e.g. Lagrange and Serendipity interpolations, including
    sub- and superparametric elements. For interpolations whose dofs are not
    nodal function values, e.g. H(div) and H(curl) interpolations, use
    [`project_analytical!`](@ref) instead.
"""
function apply_analytical!(
        a::AbstractVector, dh::DofHandler, fieldname::Symbol, f::Function,
        cellset = 1:getncells(get_grid(dh))
    )

    fieldname ∉ getfieldnames(dh) && error("The fieldname $fieldname was not found in the dof handler")
    ip_geos = _geometric_interpolations(dh)

    # Check that all field interpolations are supported before mutating `a` to avoid
    # partial application
    for sdh in dh.subdofhandlers
        isnothing(_find_field(sdh, fieldname)) && continue
        ip_fun = getfieldinterpolation(sdh, find_field(sdh, fieldname))
        _check_apply_analytical_supported(ip_fun)
    end

    for (sdh, ip_geo) in zip(dh.subdofhandlers, ip_geos)
        isnothing(_find_field(sdh, fieldname)) && continue
        field_idx = find_field(sdh, fieldname)
        ip_fun = getfieldinterpolation(sdh, field_idx)
        field_dim = n_components(sdh, field_idx)
        celldofinds = dof_range(sdh, fieldname)
        set_intersection = if length(cellset) == length(sdh.cellset) == getncells(get_grid(dh))
            BitSet(1:getncells(get_grid(dh)))
        else
            intersect(BitSet(sdh.cellset), BitSet(cellset))
        end
        isempty(set_intersection) && continue
        _apply_analytical!(a, dh, celldofinds, field_dim, ip_fun, ip_geo, f, set_intersection)
    end
    return a
end

function _apply_analytical!(
        a::AbstractVector, dh::AbstractDofHandler, celldofinds, field_dim,
        ip_fun::Interpolation{RefShape}, ip_geo::Interpolation, f::Function, cellset
    ) where {dim, RefShape <: AbstractRefShape{dim}}

    coords = getcoordinates(get_grid(dh), first(cellset))
    ref_points = reference_coordinates(ip_fun)
    dummy_weights = zeros(length(ref_points))
    qr = QuadratureRule{RefShape}(dummy_weights, ref_points)
    # Note: Passing ip_geo as the function interpolation here, it is just a dummy.
    cv = CellValues(qr, ip_geo, ip_geo)
    c_dofs = celldofs(dh, first(cellset))
    f_dofs = zeros(Int, length(celldofinds))

    # Check f before looping
    length(f(first(coords))) == field_dim || error("length(f(x)) must be equal to dimension of the field ($field_dim)")

    for cellnr in cellset
        getcoordinates!(coords, get_grid(dh), cellnr)
        celldofs!(c_dofs, dh, cellnr)
        for (i, celldofind) in enumerate(celldofinds)
            f_dofs[i] = c_dofs[celldofind]
        end
        _apply_analytical!(a, f_dofs, coords, field_dim, cv, f)
    end
    return a
end

function _apply_analytical!(a::AbstractVector, dofs::Vector{Int}, coords::Vector{<:Vec}, field_dim, cv::CellValues, f)
    for i_dof in 1:getnquadpoints(cv)
        x_dof = spatial_coordinate(cv, i_dof, coords)
        for (idim, icval) in enumerate(f(x_dof))
            a[dofs[field_dim * (i_dof - 1) + idim]] = icval
        end
    end
    return a
end

# Which interpolations `project_analytical!` is implemented for; used to error before any
# dof value has been assigned so that `a` is never partially modified.
function _check_project_analytical_supported(ip_fun::Interpolation{RefShape}) where {dim, RefShape <: AbstractRefShape{dim}}
    cf = conformity(ip_fun)
    if cf isa HcurlConformity && dim == 3
        throw(ArgumentError("project_analytical! is not implemented for 3D H(curl) interpolations ($ip_fun), whose edge dofs require line integrals along edges"))
    elseif !(cf isa Union{HdivConformity, HcurlConformity})
        throw(ArgumentError("project_analytical! is currently only implemented for H(div) and H(curl) interpolations, got $(ip_fun). For nodal interpolations (e.g. Lagrange), use apply_analytical! instead."))
    end
    return nothing
end

"""
    project_analytical!(
        a::AbstractVector, dh::AbstractDofHandler, fieldname::Symbol,
        f::Function, cellset = 1:getncells(get_grid(dh));
        qr_order = -1
    )

Set the values in the degree of freedom vector `a` pertaining to the field `fieldname`
by cell-local projections of the function `f(x)` for all cells in `cellset`. The
function `f(x)` is given the spatial coordinate `x` and must return the full vector
value, `f(x)::Vec{dim}`.

In contrast to [`apply_analytical!`](@ref), which requires the dofs to be function
values at nodal locations, `project_analytical!` determines the dof values by local
L2 projections. It thereby supports H(div) (`RaviartThomas`, `BrezziDouglasMarini`)
and 2D H(curl) (`Nedelec`) interpolations, whose dofs are moment functionals. The
projections are:

* Dofs associated with a facet are determined by projecting the trace of `f`
  (the normal component ``f \\cdot n`` for H(div), the tangential part
  ``f \\times n`` for H(curl)) onto the trace of the finite element space on
  that (physical) facet, the same projection as for [`ProjectedDirichlet`](@ref).
* Remaining interior dofs (present for e.g. `RaviartThomas{RefTriangle, 2}`) are
  determined by a cell-local L2 minimization with the facet dofs fixed.

Note that all projections are cell-local: no global system is solved, unlike the
global L2 projection with an [`L2Projector`](@ref). This operator nevertheless
reproduces exactly any function that lies in the (global) finite element space,
and shared facet dofs get the same value from both neighboring cells (up to
roundoff), since both project the same function onto the same facet trace space.
On affine facets, the facet dof values coincide with the classical moment
functionals of these interpolations; on non-affine facets they are the
physical-L2 analogue thereof.

The quadrature order can be controlled with the keyword `qr_order`; by default,
`2 * order` of the interpolation is used, which integrates the projection
matrices exactly on affine cells. For strongly distorted or curved cells, or
rapidly varying `f`, a higher order may be beneficial.

This function can be used to apply initial conditions for time dependent problems.

!!! note
    3D H(curl) interpolations (`Nedelec` on `RefTetrahedron`/`RefHexahedron`)
    are not yet supported, since their edge dofs require line integrals along
    edges; an `ArgumentError` is thrown for these. Nodal (H1) interpolations are
    currently not supported either; use [`apply_analytical!`](@ref) for these.
"""
function project_analytical!(
        a::AbstractVector, dh::DofHandler, fieldname::Symbol, f::Function,
        cellset = 1:getncells(get_grid(dh)); qr_order::Int = -1
    )

    fieldname ∉ getfieldnames(dh) && error("The fieldname $fieldname was not found in the dof handler")
    ip_geos = _geometric_interpolations(dh)

    # Check that all field interpolations are supported before mutating `a` to avoid
    # partial application
    for sdh in dh.subdofhandlers
        isnothing(_find_field(sdh, fieldname)) && continue
        ip_fun = getfieldinterpolation(sdh, find_field(sdh, fieldname))
        _check_project_analytical_supported(ip_fun)
    end

    for (sdh, ip_geo) in zip(dh.subdofhandlers, ip_geos)
        isnothing(_find_field(sdh, fieldname)) && continue
        field_idx = find_field(sdh, fieldname)
        ip_fun = getfieldinterpolation(sdh, field_idx)
        field_dim = n_components(sdh, field_idx)
        celldofinds = dof_range(sdh, fieldname)
        set_intersection = if length(cellset) == length(sdh.cellset) == getncells(get_grid(dh))
            BitSet(1:getncells(get_grid(dh)))
        else
            intersect(BitSet(sdh.cellset), BitSet(cellset))
        end
        isempty(set_intersection) && continue
        _project_analytical!(a, dh, celldofinds, field_dim, ip_fun, ip_geo, f, set_intersection, qr_order)
    end
    return a
end

# H(div) and H(curl) interpolations: dofs are determined by L2 projection of the
# trace of f onto each facet's trace space, followed by a cell-local L2 fit for
# interior dofs. Reuses the facet projection kernels from ProjectedDirichlet.
_trace_function(::HdivConformity, f::Function) = (x, _, n) -> f(x) ⋅ n
_trace_function(::HcurlConformity, f::Function) = (x, _, n) -> f(x) × n

function _project_analytical!(
        a::AbstractVector, dh::AbstractDofHandler, celldofinds, field_dim,
        ip_fun::Interpolation{RefShape}, ip_geo::Interpolation, f::Function, cellset, qr_order::Int
    ) where {dim, RefShape <: AbstractRefShape{dim}}

    # 3D H(curl) is rejected up front in _check_project_analytical_supported
    cf = conformity(ip_fun)

    grid = get_grid(dh)
    order = _default_bc_qr_order(qr_order, ip_fun)
    fqr = FacetQuadratureRule{RefShape}(order)
    fv = FacetValues(fqr, ip_fun, ip_geo)
    facetdof_indices = dirichlet_facetdof_indices(ip_fun)
    interior_dofs = setdiff(1:getnbasefunctions(ip_fun), Iterators.flatten(facetdof_indices))

    T = eltype(a)
    max_facet_dofs = maximum(length, facetdof_indices)
    Kᶠ = zeros(T, max_facet_dofs, max_facet_dofs)
    aᶠ = zeros(T, max_facet_dofs)
    fᶠ = zeros(T, max_facet_dofs)
    cv = if isempty(interior_dofs)
        nothing
    else
        CellValues(QuadratureRule{RefShape}(order), ip_fun, ip_geo)
    end
    Kⁱ = zeros(T, length(interior_dofs), length(interior_dofs))
    aⁱ = zeros(T, length(interior_dofs))
    rⁱ = zeros(T, length(interior_dofs))

    coords = getcoordinates(grid, first(cellset))
    c_dofs = celldofs(dh, first(cellset))

    # Check f before looping
    length(f(first(coords))) == field_dim || error("length(f(x)) must be equal to dimension of the field ($field_dim)")
    g = _trace_function(cf, f)

    for cellnr in cellset
        cell = getcells(grid, cellnr)
        getcoordinates!(coords, grid, cellnr)
        celldofs!(c_dofs, dh, cellnr)
        for facetnr in 1:nfacets(cell)
            shape_nrs = facetdof_indices[facetnr]
            isempty(shape_nrs) && continue
            reinit!(fv, cell, coords, facetnr)
            solve_projected_dbc!(aᶠ, Kᶠ, fᶠ, g, fv, shape_nrs, coords, 0)
            for (i, shape_nr) in pairs(shape_nrs)
                a[c_dofs[celldofinds[shape_nr]]] = aᶠ[i]
            end
        end
        if cv !== nothing
            reinit!(cv, cell, coords)
            _solve_interior_dofs!(a, aⁱ, Kⁱ, rⁱ, c_dofs, celldofinds, interior_dofs, cv, coords, f)
        end
    end
    return a
end

# Determine the interior dofs by minimizing the cell-local L2 error with the
# (already determined) facet dofs fixed.
function _solve_interior_dofs!(
        a::AbstractVector, aⁱ::Vector, Kⁱ::Matrix, rⁱ::Vector, c_dofs, celldofinds, interior_dofs,
        cv::CellValues, coords::Vector{<:Vec}, f::Function
    )
    fill!(Kⁱ, 0)
    fill!(rⁱ, 0)
    for q_point in 1:getnquadpoints(cv)
        dV = getdetJdV(cv, q_point)
        x = spatial_coordinate(cv, q_point, coords)
        # Residual: f minus the contribution from the facet dofs
        v = f(x)
        for shape_nr in 1:getnbasefunctions(cv)
            shape_nr in interior_dofs && continue
            v -= shape_value(cv, q_point, shape_nr) * a[c_dofs[celldofinds[shape_nr]]]
        end
        for (i, I) in pairs(interior_dofs)
            Nᵢ = shape_value(cv, q_point, I)
            rⁱ[i] += (Nᵢ ⋅ v) * dV
            for (j, J) in pairs(interior_dofs)
                Nⱼ = shape_value(cv, q_point, J)
                Kⁱ[i, j] += (Nᵢ ⋅ Nⱼ) * dV
            end
        end
    end
    solve_small_spd!(aⁱ, Kⁱ, rⁱ)
    for (i, I) in pairs(interior_dofs)
        a[c_dofs[celldofinds[I]]] = aⁱ[i]
    end
    return a
end
