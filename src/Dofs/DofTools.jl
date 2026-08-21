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
