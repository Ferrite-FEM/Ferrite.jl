# Various tooling around the DofHandler to query information about dofs.

"""
    collect_dofs(dh::DofHandler; field, components, cellset) -> Vector{Int}

Return a sorted vector with the global degrees of freedom matching all of the (optional)
filtering arguments:
 - `field::Symbol`: only dofs pertaining to the field named `field`.
 - `components::Vector{Int}`: only dofs pertaining to these components of the field
   (requires specifying `field`). Component filtering is only supported for fields with
   scalar and vectorized interpolations, where dofs map one-to-one to components (see
   [`DofOrder.ComponentWise`](@ref) for details on the component convention).
 - `cellset::AbstractSet{Int}`: only dofs on the cells in `cellset`.

Without filtering arguments all dofs are returned, i.e. `collect(1:ndofs(dh))`.
"""
function collect_dofs(
        dh::DofHandler;
        field::Union{Symbol, Nothing} = nothing,
        components::Union{Vector{Int}, Nothing} = nothing,
        cellset::Union{AbstractSet{Int}, Nothing} = nothing,
    )
    # Argument checking
    if field === nothing && components !== nothing
        throw(ArgumentError("must specify field when specifying components"))
    elseif components !== nothing && !(issorted(components) && allunique(components))
        throw(ArgumentError("components must be sorted and unique"))
    elseif field !== nothing && !(field in dh.field_names)
        throw(ArgumentError("field :$field not found in the DofHandler"))
    end
    # Allocate buffers
    seen = falses(ndofs(dh))
    cc = CellCache(dh)
    filtered_dof_range = Int[]
    # Loop over the cells
    for sdh in dh.subdofhandlers
        # Compute the component-filtered dof range
        if field !== nothing
            field_idx = findfirst(==(field), sdh.field_names)
            field_idx === nothing && continue # field does not exist on this subdomain
            full_range = dof_range(sdh, field_idx)
            if components === nothing
                copy!(filtered_dof_range, full_range)
            else
                ip = sdh.field_interpolations[field_idx]
                field_dim = n_components(ip)
                if !(field_dim == 1 || ip isa VectorizedInterpolation)
                    throw(ArgumentError("component filtering is not supported for fields with non-vectorized interpolation ($(typeof(ip)))"))
                end
                if !all(x -> 1 <= x <= field_dim, components)
                    throw(ArgumentError("components out of range for field :$field with $field_dim components"))
                end
                @assert mod(length(full_range), field_dim) == 0
                empty!(filtered_dof_range)
                for (i, dof) in pairs(full_range)
                    if mod1(i, field_dim) in components
                        push!(filtered_dof_range, dof)
                    end
                end
            end
        else # field === nothing -> all dofs
            copy!(filtered_dof_range, 1:ndofs_per_cell(sdh))
        end
        # Construct the set to loop over
        if cellset === nothing || cellset === sdh.cellset
            loopset = sdh.cellset
        else
            loopset = intersect(sdh.cellset, cellset)
        end
        for ci in loopset
            # Update the cell cache and collect the dofs
            reinit!(cc, ci)
            dofs = celldofs(cc)
            for d in filtered_dof_range
                seen[dofs[d]] = true
            end
        end
    end
    return findall(seen)
end

"""
    global_dof_range(dh::DofHandler, f::Symbol)

Return the global dof range for dofs pertaining to field `f`. This requires dofs to be
globally enumerated field wise, see [`renumber!`](@ref) and `DofOrder.FieldWise` for more
details.
"""
function global_dof_range(dh::DofHandler, f::Symbol)
    f in dh.field_names || error("field :$f not found in the DofHandler")
    seen = falses(ndofs(dh))
    dofmin, dofmax = typemax(Int), typemin(Int)
    for sdh in dh.subdofhandlers
        f in sdh.field_names || continue
        frange = dof_range(sdh, f)
        for cc in CellIterator(sdh)
            dofs = celldofs(cc)
            for j in frange
                d = dofs[j]
                seen[d] = true
                dofmin = min(dofmin, d)
                dofmax = max(dofmax, d)
            end
        end
    end
    r = dofmin:dofmax
    if !all(@view seen[r])
        error("dofs for field $(f) not continuously enumerated, renumber by field")
    end
    return r
end
