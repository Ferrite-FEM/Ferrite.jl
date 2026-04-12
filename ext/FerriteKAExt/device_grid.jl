"""
    DeviceGrid(cells::AbstractArray, nodes::AbstractArray)
    DeviceGrid(backend::KA.Backend, host_grid::AbstractGrid)

This is a grid which can be transferred to devices (e.g. GPUs).
"""
struct DeviceGrid{
        dim, C <: Ferrite.AbstractCell, T <: Real,
        CA <: AbstractArray{C, 1}, NA <: AbstractArray{Node{dim, T}, 1},
    } <: Ferrite.AbstractGrid{dim}
    cells::CA
    nodes::NA
end

function Adapt.adapt_structure(to, grid::DeviceGrid)
    return DeviceGrid(Adapt.adapt_structure(to, grid.cells), Adapt.adapt_structure(to, grid.nodes))
end

function DeviceGrid(backend, grid::AbstractGrid)
    return DeviceGrid(adapt(backend, getcells(grid)), adapt(backend, getnodes(grid)))
end

# This allows us to bypass the scalar indexing in the original implementation.
Ferrite.get_coordinate_eltype(::DeviceGrid{<:Any, <:Any, T}) where {T} = T
Ferrite.get_coordinate_type(::DeviceGrid{dim, <:Any, T}) where {dim, T} = Vec{dim, T}

#TODO: Document as slow, but ok during setup
function Ferrite.nnodes_per_cell(grid::DeviceGrid, i::Integer)
    return @allowscalar Ferrite.nnodes(grid.cells[i])
end
