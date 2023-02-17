function Ferrite.reinit!(cc::CellCache{<:Any,<:Ferrite.AbstractGrid,<:DistributedDofHandler}, i::Int)
    cc.cellid[] = i
    if cc.flags.nodes
        Ferrite.cellnodes!(cc.nodes, cc.grid, i)
    end
    if cc.flags.coords
        Ferrite.cellcoords!(cc.coords, cc.grid, i)
    end
    if cc.dh !== nothing && cc.flags.dofs
        Ferrite.celldofs!(cc.dofs, cc.dh, i)
    end
    return cc
end
