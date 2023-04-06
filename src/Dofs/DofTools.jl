# Various tooling around DofHandler to query information about dofs etc.

"""
    collect_dofs(dh::DofHandler, args...) -> Set{Int}

Return the (unordered) set of global degrees of freedom matching args...

Filtering arguments:
 - `field::Symbol`:           Which field
 - `components::Vector{Int}`: Which components of the field
 - `cellset::Set{Int}`:       Which cells
 - `faceset::Set{FaceIndex}`: Which faces

"""
function collect_dofs(dh::DofHandler; filter_kwargs...)
    return collect_dofs!(Set{Int}(), dh; filter_kwargs...)
end

function collect_dofs!(
    set        :: Set,
    dh         :: DofHandler;
    field      :: Union{Symbol,                 Nothing} = nothing,
    components :: Union{Vector{Int},            Nothing} = nothing,
    cellset    :: Union{AbstractSet{Int},       Nothing} = nothing,
    faceset    :: Union{AbstractSet{FaceIndex}, Nothing} = nothing,
)
    # Argument checking
    if cellset !== nothing && faceset !== nothing
        throw(ArgumentError("must not specify both cellset and faceset"))
    elseif field === nothing && components !== nothing
        throw(ArgumentError("must specify field when specifying components"))
    elseif components !== nothing && !(issorted(components) && allunique(components))
        throw(ArgumentError("components must be sorted and unique"))
    elseif faceset !== nothing
        throw(ArgumentError("Fly, you fools!"))
    end
    # Allocate buffers
    cc = CellCache(dh)
    filtered_dof_range = Int[]
    # Loop over the cells
    for fh in dh.fieldhandlers
        # Compute the component-filtered dof range
        if field !== nothing
            field_idx = findfirst(x -> x === field, fh.field_names)
            field_idx === nothing && continue # field is specified, but does not exist here
            full_range = dof_range(fh, field_idx)
            if components === nothing
                copy!(filtered_dof_range, full_range)
            else
                field_dim = getfielddim(fh, field_idx)
                @assert mod(length(full_range), field_dim) == 0
                @assert all(x -> 1 <= x <= field_dim, components)
                resize!(filtered_dof_range, 0)
                for (i, dof) in pairs(full_range)
                    if mod1(i, field_dim) in components
                        push!(filtered_dof_range, dof)
                    end
                end
            end
        else # field === nothing -> all dofs
            @assert components === nothing
            copy!(filtered_dof_range, 1:ndofs_per_cell(fh))
        end
        # Construct the set to loop over
        if cellset === nothing || cellset === fh.cellset
            loopset = fh.cellset
        else
            loopset = intersect(fh.cellset, cellset)
        end
        for ci in loopset
            # Update the cell cache and collect the dofs
            reinit!(cc, ci)
            for d in filtered_dof_range
                push!(set, cc.dofs[d])
            end
        end
    end
    return set
end

"""
    global_dof_range(dh::DofHandler, field::Symbol)

TBW
"""
function global_dof_range(dh::DofHandler, field::Symbol)
    set = collect_dofs(dh; field=field)
    dofmin, dofmax = extrema(set)
    r = dofmin:dofmax
    length(set) == length(r) || error("dofs for field $(repr(field)) are not continuous.")
    return r
end
