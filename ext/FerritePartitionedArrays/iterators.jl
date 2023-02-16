"""
This is copy pasta for now.
"""
function Ferrite.CellIterator(dh::DistributedDofHandler{dim,T}, cellset::Union{AbstractVector{Int},Nothing}=nothing, flags::UpdateFlags=UpdateFlags())  where {dim,C,T}
    isconcretetype(C) || _check_same_celltype(getgrid(dh), cellset)
    N = nnodes_per_cell(getgrid(dh), cellset === nothing ? 1 : first(cellset))
    cell = ScalarWrapper(0)
    nodes = zeros(Int, N)
    coords = zeros(Vec{dim,T}, N)
    n = ndofs_per_cell(dh, cellset === nothing ? 1 : first(cellset))
    celldofs = zeros(Int, n)
    return Ferrite.CellIterator{dim,C,T,typeof(dh)}(flags, getgrid(dh), cell, nodes, coords, cellset, dh, celldofs)
end
