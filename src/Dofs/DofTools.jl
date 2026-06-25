"""
    global_dof_range(dh::AbstractDofHandler, f::Symbol)

Return the global dof range for dofs pertaining to field `f`. This requires dofs to be
globally enumerated field wise, see [`renumber!`](@ref) for more details.
"""
function global_dof_range(dh::AbstractDofHandler, f::Symbol)
    set = Set{Int}()
    collect_dofs!(set, dh, f)
    dofmin, dofmax = extrema(set)
    r = dofmin:dofmax
    if length(set) != length(r)
        error("dofs for field $(f) not continuously enumerated, renumber by field")
    end
    return r
end

function collect_dofs!(set::Set{Int}, dh::DofHandler, f::Symbol)
    frange = dof_range(dh, f)
    for cc in CellIterator(dh)
        union!(set, @view cc.dofs[frange])
    end
    return set
end
function collect_dofs!(set::Set{Int}, dh::MixedDofHandler, f::Symbol)
    for fh in dh.fieldhandlers
        f in getfieldnames(fh) || continue
        frange = dof_range(fh, f)
        for cc in CellIterator(dh, fh.cellset)
            union!(set, @view cc.dofs[frange])
        end
    end
    return set
end
