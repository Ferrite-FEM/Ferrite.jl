"""
    DeviceSubGrid(cells::AbstractArray, nodes::AbstractArray, global_to_local_cellid)

This is a part of a grid which can be transferred to devices (e.g. GPUs).
"""
struct DeviceSubGrid{
        sdim, C <: Ferrite.AbstractCell, T <: Real,
        CA <: AbstractArray{C, 1}, NA <: AbstractArray{Node{sdim, T}, 1},
        GTLC,
    } <: Ferrite.AbstractGrid{sdim}
    cells::CA
    nodes::NA
    global_to_local_cellid::GTLC
end

function Adapt.adapt_structure(to, grid::DeviceSubGrid)
    return DeviceSubGrid(Adapt.adapt(to, grid.cells), Adapt.adapt(to, grid.nodes), Adapt.adapt(to, grid.global_to_local_cellid))
end

# This allows us to bypass the scalar indexing in the original implementation.
Ferrite.get_coordinate_eltype(::DeviceSubGrid{<:Any, <:Any, T}) where {T} = T
Ferrite.get_coordinate_type(::DeviceSubGrid{dim, <:Any, T}) where {dim, T} = Vec{dim, T}

Ferrite.getcells(grid::DeviceSubGrid, i::Int) = grid.cells[grid.global_to_local_cellid[i]]
Ferrite.getcells(grid::DeviceSubGrid{<:Any, <:Any, <:Any, <:Any, <:Any, Nothing}, i::Int) = grid.cells[i]
