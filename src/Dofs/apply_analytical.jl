
function _default_interpolations(dh::MixedDofHandler)
    fhs = dh.fieldhandlers
    getcelltype(i) = typeof(getcells(dh.grid, first(fhs[i].cellset)))
    ntuple(i -> default_interpolation(getcelltype(i)), length(fhs))
end

function _default_interpolation(dh::DofHandler)
    return default_interpolation(typeof(getcells(dh.grid, 1)))
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
    cellset=1:getncells(dh.grid)
    )
    fieldname ∉ getfieldnames(dh) && error("The fieldname $fieldname was not found in the dof handler")
    ip_geo = _default_interpolation(dh)
    field_idx = find_field(dh, fieldname)
    ip_fun = getfieldinterpolation(dh, field_idx)
    celldofinds = dof_range(dh, fieldname)
    field_dim = getfielddim(dh, field_idx)
    _apply_analytical!(a, dh, celldofinds, field_dim, ip_fun, ip_geo, f, cellset)
end

function apply_analytical!(
    a::AbstractVector, dh::MixedDofHandler, fieldname::Symbol, f::Function,
    cellset=1:getncells(dh.grid),
    )
    fieldname ∉ getfieldnames(dh) && error("The fieldname $fieldname was not found in the dof handler")
    ip_geos = _default_interpolations(dh)

    for (fh, ip_geo) in zip(dh.fieldhandlers, ip_geos)
        fieldname ∈ getfieldnames(fh) || continue
        field_idx = find_field(fh, fieldname)
        ip_fun = getfieldinterpolation(fh, field_idx)
        field_dim = getfielddim(fh, field_idx)
        celldofinds = dof_range(fh, fieldname)
        _apply_analytical!(a, dh, celldofinds, field_dim, ip_fun, ip_geo, f, intersect(fh.cellset, cellset))
    end
    return a
end

function _apply_analytical!(
    a::Vector, dh::AbstractDofHandler, celldofinds, field_dim,
    ip_fun::Interpolation, ip_geo::Interpolation, f::Function, cellset)
    
    coords = getcoordinates(dh.grid, first(cellset))
    cdv = CellDofValues(ip_fun, ip_geo)
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
        _apply_analytical!(a, f_dofs, coords, field_dim, cdv, f)
    end
    return a
end

function _apply_analytical!(a::Vector, dofs::Vector{Int}, coords::Vector{<:Vec}, field_dim, cdv::CellDofValues, f)
    for i_dof in 1:getnquadpoints(cdv)
        x_dof = spatial_coordinate(cdv, i_dof, coords)
        for (idim, icval) in enumerate(f(x_dof))
            a[dofs[field_dim*(i_dof-1)+idim]] = icval
        end
    end
    return a
end