struct DeviceSubDofHandler{
        dim,
        CS <: AbstractVector{Int}, CD <: AbstractVector{Int},
        CO <: AbstractVector{Int}, FN, DR <: Tuple, G <: Ferrite.AbstractGrid{dim},
    } <: AbstractDofHandler
    cellset::CS
    cell_dofs::CD
    cell_dofs_offset::CO
    ndofs_per_cell::Int
    nnodes_per_cell::Int
    # Vector{Symbol} on host, Nothing on device — Symbol is not a bitstype and cannot
    # be stored in device memory. Use the integer index overload of dof_range on device.
    field_names::FN
    dof_ranges::DR
    grid::G
end

Ferrite.get_grid(dh::DeviceSubDofHandler) = dh.grid

function Adapt.adapt_structure(to, sdh::DeviceSubDofHandler)
    return DeviceSubDofHandler(
        Adapt.adapt_structure(to, sdh.cellset),
        Adapt.adapt_structure(to, sdh.cell_dofs),
        Adapt.adapt_structure(to, sdh.cell_dofs_offset),
        sdh.ndofs_per_cell,
        sdh.nnodes_per_cell,
        nothing,
        sdh.dof_ranges,
        Adapt.adapt_structure(to, sdh.grid),
    )
end

Ferrite.nnodes_per_cell(sdh::DeviceSubDofHandler) = sdh.nnodes_per_cell
Ferrite.ndofs_per_cell(sdh::DeviceSubDofHandler) = sdh.ndofs_per_cell

function Ferrite.dof_range(sdh::DeviceSubDofHandler, field_idx::Int)
    return sdh.dof_ranges[field_idx]
end
function Ferrite.dof_range(sdh::DeviceSubDofHandler, field_name::Symbol)
    idx = findfirst(==(field_name), sdh.field_names)
    idx === nothing && error("Field $field_name not found in DeviceSubDofHandler")
    return sdh.dof_ranges[idx]
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
        dof_ranges = Tuple(Ferrite.dof_range(sdh, i) for i in 1:length(sdh.field_names))
        DeviceSubDofHandler(
            adapt(backend, collect(Int, sdh.cellset)), cell_dofs, cell_dofs_offset,
            sdh.ndofs_per_cell, nnodes_per_cell(dh.grid, first(sdh.cellset)), copy(sdh.field_names), dof_ranges, gpu_grid
        )
    end
    return HostDofHandler(subdofhandlers, gpu_grid, dh)
end

Ferrite.ndofs_per_cell(dh::HostDofHandler, i) = Ferrite.ndofs_per_cell(dh.original_dh, i)

function Adapt.adapt_structure(to::KA.Backend, dh::DofHandler)
    return HostDofHandler(to, dh)
end

Ferrite.get_grid(dh::HostDofHandler) = dh.grid
