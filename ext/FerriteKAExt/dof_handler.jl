struct DeviceSubDofHandler{
        dim,
        CS <: AbstractVector{Int}, CD <: AbstractVector{Int},
        CO <: AbstractVector{Int}, DR <: NamedTuple, G <: AbstractGrid{dim},
    } <: AbstractDofHandler
    cellset::CS
    cell_dofs::CD
    cell_dofs_offset::CO
    ndofs_per_cell::Int
    nnodes_per_cell::Int
    dof_ranges::DR
    grid::G
end

Ferrite.get_grid(dh::DeviceSubDofHandler) = dh.grid

Ferrite.nnodes_per_cell(sdh::DeviceSubDofHandler) = sdh.nnodes_per_cell
Ferrite.ndofs_per_cell(sdh::DeviceSubDofHandler) = sdh.ndofs_per_cell

function Ferrite.dof_range(sdh::DeviceSubDofHandler, field_idx::Int)
    return sdh.dof_ranges[field_idx]
end
function Ferrite.dof_range(sdh::DeviceSubDofHandler, field_name::Symbol)
    return sdh.dof_ranges[field_name]
end

function Ferrite.celldofs!(global_dofs::AbstractVector, sdh::DeviceSubDofHandler, i::Integer)
    copyto!(global_dofs, 1, sdh.cell_dofs, sdh.cell_dofs_offset[i], length(global_dofs))
    return global_dofs
end

# Host-only container — not sent to the device!
struct HostDofHandler{dim, G <: DeviceGrid{dim}, SDH <: DeviceSubDofHandler, DH <: AbstractDofHandler} <: AbstractDofHandler
    subdofhandlers::Vector{SDH}
    grid::G
    original_dh::DH
end

function HostDofHandler(backend, dh::DofHandler)
    gpu_grid = DeviceGrid(backend, dh.grid)
    cell_dofs = adapt(backend, dh.cell_dofs)
    cell_dofs_offset = adapt(backend, dh.cell_dofs_offset)
    subdofhandlers = map(dh.subdofhandlers) do sdh
        dof_ranges = NamedTuple(name => Ferrite.dof_range(sdh, name) for name in sdh.field_names)
        DeviceSubDofHandler(
            adapt(backend, collect(Int, sdh.cellset)), cell_dofs, cell_dofs_offset,
            sdh.ndofs_per_cell, Ferrite.nnodes_per_cell(dh.grid, first(sdh.cellset)), dof_ranges, gpu_grid
        )
    end
    return HostDofHandler(subdofhandlers, gpu_grid, dh)
end

Ferrite.ndofs_per_cell(dh::HostDofHandler, i) = Ferrite.ndofs_per_cell(dh.original_dh, i)

Ferrite.get_grid(dh::HostDofHandler) = dh.grid
