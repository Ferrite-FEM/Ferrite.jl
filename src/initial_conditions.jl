
# A few additional dispatches for existing functions, to be moved to MixedDofHandler.jl
Ferrite.getfieldinterpolation(fh::FieldHandler, field_idx::Int) = fh.fields[field_idx].interpolation
Ferrite.getfielddim(fh::FieldHandler, field_idx::Int) = fh.fields[field_idx].dim

function _default_interpolations(dh::MixedDofHandler)
    fhs = dh.fieldhandlers
    getcelltype(i) = typeof(getcells(dh.grid, first(fhs[i].cellset)))
    ntuple(i->Ferrite.default_interpolation(getcelltype(i)), length(fhs))
end

function _default_interpolation(dh::DofHandler)
    return Ferrite.default_interpolation(typeof(getcells(dh.grid, 1)))
end

"""
    initial_conditions!(
        a::AbstractVector, dh::AbstractDofHandler, field::Symbol, f::Function,
        cellset=1:getncells(dh.grid))

Apply initial conditions to the degree of freedom vector `a` on the field `field`,
where the initial condition is given by `f(x)` where `x` are the spatial coordinates 
of the degree of freedom. For scalar fields, `f(x)::Number`, 
and for vector fields with dimension `dim`, `f(x)::Vec{dim}`
"""
function initial_conditions!(
    a::AbstractVector, dh::DofHandler, field::Symbol, f::Function, 
    cellset=1:getncells(dh.grid)
    )

    ip_geo=_default_interpolation(dh)
    field_idx = Ferrite.find_field(dh, field)
    ip_fun = Ferrite.getfieldinterpolation(dh, field_idx)
    celldofinds = dof_range(dh, field)
    field_dim = Ferrite.getfielddim(dh, field_idx)
    initial_conditions!(a, dh, celldofinds, field_dim, ip_fun, ip_geo, f, cellset)
end

function initial_conditions!(
    a::AbstractVector, dh::MixedDofHandler, field::Symbol, f::Function, 
    cellset=1:getncells(dh.grid),
    )
    ip_geos=_default_interpolations(dh)

    if field ∉ Ferrite.getfieldnames(dh)
        @warn("The field $field was not found in the dof handler")
        return a
    end
    if length(ip_geo) != length(dh.fieldhandlers)
        error("$(length(ip_geos)) ip_geos and $(length(dh.fieldhandlers)) fieldhandlers")
    end

    for (fh, ip_geo) in zip(dh.fieldhandlers, ip_geos)
        field ∈ Ferrite.getfieldnames(fh) || continue
        field_idx = Ferrite.find_field(fh, field)
        ip_fun = Ferrite.getfieldinterpolation(fh, field_idx)
        field_dim = Ferrite.getfielddim(fh, field_idx)
        celldofinds = dof_range(fh, field)
        initial_conditions!(a, dh, celldofinds, field_dim, ip_fun, ip_geo, f, intersect(fh.cellset, cellset))
    end
    return a
end

function initial_conditions!(
    a::Vector, dh::Ferrite.AbstractDofHandler, celldofinds, field_dim,
    ip_fun::Interpolation, ip_geo::Interpolation, f::Function, cellset)
    
    coords = getcoordinates(dh.grid, first(cellset))
    c_dofs = celldofs(dh, first(cellset))
    f_dofs = zeros(Int, length(celldofinds))

    # Check f before looping
    length(f(first(coords))) == field_dim || error("length(f(x)) must be equal to dimension of the field ($field_dim)")

    for cellnr in cellset
        Ferrite.cellcoords!(coords, dh, cellnr)
        Ferrite.celldofs!(c_dofs, dh, cellnr)
        # f_dofs .= c_dofs[celldofinds]
        foreach(i->(f_dofs[i] = c_dofs[celldofinds[i]]), 1:length(celldofinds)) 
        initial_conditions!(a, f_dofs, coords, field_dim, ip_fun, ip_geo, f)
    end
    return a
end

function initial_conditions!(a::Vector, dofs::Vector{Int}, coords::Vector{XT}, field_dim, ip_fun, ip_geo, f) where {XT<:Vec}

    getnbasefunctions(ip_geo) == length(coords) || throw("coords=$coords not compatible with ip_ge=$ip_geo")
    ref_coords = Ferrite.reference_coordinates(ip_fun)
    length(ref_coords)*field_dim == length(dofs) || error("$ip_fun must have length(dofs)=$(length(dofs)) reference coords")

    for (i_dof, refpoint) in enumerate(ref_coords)
        x_dof = zero(XT)
        for (i,x_node) in enumerate(coords)
            x_dof += Ferrite.value(ip_geo, i, refpoint) * x_node
        end
        ic = f(x_dof)
        foreach(idim->( a[dofs[field_dim*(i_dof-1)+idim]] = ic[idim] ), 1:length(ic))
    end
    return a
end