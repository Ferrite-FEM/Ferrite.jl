
function _default_interpolations(dh::MixedDofHandler)
    fhs = dh.fieldhandlers
    getcelltype(i) = typeof(getcells(dh.grid, first(fhs[i].cellset)))
    ntuple(i->default_interpolation(getcelltype(i)), length(fhs))
end

function _default_interpolation(dh::DofHandler)
    return default_interpolation(typeof(getcells(dh.grid, 1)))
end

"""
    apply_analytical!(
        a::AbstractVector, dh::AbstractDofHandler, field::Symbol, f::Function,
        cellset=1:getncells(dh.grid))

Apply initial conditions to the degree of freedom vector `a` on the field `field` 
for all cells in `cellset`.
The initial condition is given by `f(x)` where `x` is the spatial coordinate 
of the degree of freedom. For scalar fields, `f(x)::Number`, 
and for vector fields with dimension `dim`, `f(x)::Vec{dim}`
"""
function apply_analytical!(
    a::AbstractVector, dh::DofHandler, field::Symbol, f::Function, 
    cellset=1:getncells(dh.grid)
    )

    ip_geo=_default_interpolation(dh)
    field_idx = find_field(dh, field)
    ip_fun = getfieldinterpolation(dh, field_idx)
    celldofinds = dof_range(dh, field)
    field_dim = getfielddim(dh, field_idx)
    apply_analytical!(a, dh, celldofinds, field_dim, ip_fun, ip_geo, f, cellset)
end

function apply_analytical!(
    a::AbstractVector, dh::MixedDofHandler, field::Symbol, f::Function, 
    cellset=1:getncells(dh.grid),
    )
    ip_geos=_default_interpolations(dh)

    if field ∉ getfieldnames(dh)
        @warn("The field $field was not found in the dof handler")
        return a
    end
    if length(ip_geos) != length(dh.fieldhandlers)
        error("$(length(ip_geos)) ip_geos and $(length(dh.fieldhandlers)) fieldhandlers")
    end

    for (fh, ip_geo) in zip(dh.fieldhandlers, ip_geos)
        field ∈ getfieldnames(fh) || continue
        field_idx = find_field(fh, field)
        ip_fun = getfieldinterpolation(fh, field_idx)
        field_dim = getfielddim(fh, field_idx)
        celldofinds = dof_range(fh, field)
        apply_analytical!(a, dh, celldofinds, field_dim, ip_fun, ip_geo, f, intersect(fh.cellset, cellset))
    end
    return a
end

function apply_analytical!(
    a::Vector, dh::AbstractDofHandler, celldofinds, field_dim,
    ip_fun::Interpolation, ip_geo::Interpolation, f::Function, cellset)
    
    coords = getcoordinates(dh.grid, first(cellset))
    c_dofs = celldofs(dh, first(cellset))
    f_dofs = zeros(Int, length(celldofinds))

    # Check f before looping
    length(f(first(coords))) == field_dim || error("length(f(x)) must be equal to dimension of the field ($field_dim)")

    for cellnr in cellset
        cellcoords!(coords, dh, cellnr)
        celldofs!(c_dofs, dh, cellnr)
        # f_dofs .= c_dofs[celldofinds]
        foreach(i->(f_dofs[i] = c_dofs[celldofinds[i]]), 1:length(celldofinds)) 
        apply_analytical!(a, f_dofs, coords, field_dim, ip_fun, ip_geo, f)
    end
    return a
end

function apply_analytical!(a::Vector, dofs::Vector{Int}, coords::Vector{XT}, field_dim, ip_fun, ip_geo, f) where {XT<:Vec}

    getnbasefunctions(ip_geo) == length(coords) || error("coords=$coords not compatible with ip_ge=$ip_geo")
    ref_coords = reference_coordinates(ip_fun)
    length(ref_coords)*field_dim == length(dofs) || error("$ip_fun must have length(dofs)=$(length(dofs)) reference coords")

    for (i_dof, refpoint) in enumerate(ref_coords)
        x_dof = zero(XT)
        for (i,x_node) in enumerate(coords)
            x_dof += value(ip_geo, i, refpoint) * x_node
        end
        for (idim, icval) in enumerate(f(x_dof))
            a[dofs[field_dim*(i_dof-1)+idim]] = icval
        end
    end
    return a
end