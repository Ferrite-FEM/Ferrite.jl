function _default_interpolations(dh::DofHandler)
    fhs = dh.fieldhandlers
    getcelltype(i) = typeof(getcells(dh.grid, first(fhs[i].cellset)))
    ntuple(i -> default_interpolation(getcelltype(i)), length(fhs))
end

"""
    apply_analytical!(
        a::AbstractVector, dh::AbstractDofHandler, fieldname::Symbol, 
        f::Function, cellset=1:getncells(dh.grid))

Apply a solution `f(x)` by modifying the values in the degree of freedom vector `a`
pertaining to the field `fieldname` for all cells in `cellset`.
The function `f(x)` are given the spatial coordinate
of the degree of freedom. For scalar fields, `f(x)::Number`,
and for vector fields with dimension `dim`, `f(x)::Vec{dim}`.

This function can be used to apply initial conditions for time dependent problems.

!!! note
    
    This function only works for standard nodal finite element interpolations
    when the function value at the (algebraic) node is equal to the corresponding
    degree of freedom value.
    This holds for e.g. Lagrange and Serendipity interpolations, including
    sub- and superparametric elements.
"""
function apply_analytical!(
    a::AbstractVector, dh::DofHandler, fieldname::Symbol, f::Function,
    cellset = 1:getncells(dh.grid))

    fieldname ∉ getfieldnames(dh) && error("The fieldname $fieldname was not found in the dof handler")
    ip_geos = _default_interpolations(dh)

    for (fh, ip_geo) in zip(dh.fieldhandlers, ip_geos)
        isnothing(_find_field(fh, fieldname)) && continue
        field_idx = find_field(fh, fieldname)
        ip_fun = getfieldinterpolation(fh, field_idx)
        field_dim = getfielddim(fh, field_idx)
        celldofinds = dof_range(fh, fieldname)
        set_intersection = if length(cellset) == length(fh.cellset) == getncells(dh.grid)
            BitSet(1:getncells(dh.grid))
        else
            intersect(BitSet(fh.cellset), BitSet(cellset))
        end
        _apply_analytical!(a, dh, celldofinds, field_dim, ip_fun, ip_geo, f, set_intersection)
    end
    return a
end

function _apply_analytical!(
    a::AbstractVector, dh::AbstractDofHandler, celldofinds, field_dim,
    ip_fun::Interpolation{dim,RefShape}, ip_geo::Interpolation, f::Function, cellset) where {dim, RefShape}

    coords = getcoordinates(dh.grid, first(cellset))
    ref_points = reference_coordinates(ip_fun)
    dummy_weights = zeros(length(ref_points))
    qr = QuadratureRule{dim, RefShape}(dummy_weights, ref_points)
    # Note: Passing ip_geo as the function interpolation here, it is just a dummy.
    cv = CellValues(qr, ip_geo, ip_geo)
    c_dofs = celldofs(dh, first(cellset))
    f_dofs = zeros(Int, length(celldofinds))

    # Check f before looping
    length(f(first(coords))) == field_dim || error("length(f(x)) must be equal to dimension of the field ($field_dim)")

    for cellnr in cellset
        getcoordinates!(coords, dh.grid, cellnr)
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
            a[dofs[field_dim*(i_dof-1)+idim]] = icval
        end
    end
    return a
end
