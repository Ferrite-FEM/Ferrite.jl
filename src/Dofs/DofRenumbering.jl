#####################
## DoF renumbering ##
#####################

"""
    renumber!(dh::AbstractDofHandler, perm)
    renumber!(dh::AbstractDofHandler, ch::ConstraintHandler, perm)

Renumber the degrees of freedom in the DofHandler and/or ConstraintHandler according to the
permutation `perm`.

!!! warning
    The dof numbering in the DofHandler and ConstraintHandler must be always consistent. It
    is therefore necessary to either renumber *before* creating the ConstraintHandler in the
    first place, or to renumber the DofHandler and the ConstraintHandler *together*.
"""
renumber!

function renumber!(dh::AbstractDofHandler, ch::ConstraintHandler, perm::AbstractVector{<:Integer})
    @assert ch.dh === dh
    renumber!(dh, perm)
    _renumber!(ch, perm)
    return nothing
end

function renumber!(dh::Union{DofHandler, MixedDofHandler}, perm::AbstractVector{<:Integer})
    @assert isclosed(dh)
    @assert isperm(perm) && length(perm) == ndofs(dh)
    cell_dofs = dh isa DofHandler ? dh.cell_dofs :
                #= dh isa MixedDofHandler ? =# dh.cell_dofs.values
    for i in eachindex(cell_dofs)
        cell_dofs[i] = perm[cell_dofs[i]]
    end
    return dh
end

function _renumber!(ch::ConstraintHandler, perm::AbstractVector{<:Integer})
    @assert isclosed(ch)
    # To renumber the ConstraintHandler we start by renumbering master dofs in
    # ch.dofcoefficients.
    for coeffs in ch.dofcoefficients
        coeffs === nothing && continue
        for (i, (k, v)) in pairs(coeffs)
            coeffs[i] = perm[k] => v
        end
    end
    # Next we renumber ch.prescribed_dofs, empty the dofmapping dict and then (re)close! it.
    # In close! the dependent fields (ch.free_dofs, ch.inhomogeneities,
    # ch.affine_inhomogeneities, ch.dofcoefficients) will automatically be permuted since
    # they are sorted based on ch.prescribed_dofs. The dofmapping is also updated in close!,
    # but it is necessary to empty it here since otherwise it might contain keys from the
    # old numbering.
    pdofs = ch.prescribed_dofs
    for i in eachindex(pdofs)
        pdofs[i] = perm[pdofs[i]]
    end
    empty!(ch.dofmapping)
    ch.closed[] = false
    close!(ch)
    return nothing
end
