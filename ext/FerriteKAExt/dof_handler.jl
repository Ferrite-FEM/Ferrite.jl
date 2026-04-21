struct DeviceSubDofHandler{
        dim,
        CS <: AbstractVector{Int}, CD <: AbstractVector{Int},
        CO <: AbstractVector{Int}, FN <: NamedTuple, DR <: Tuple, G <: Ferrite.AbstractGrid{dim},
    } <: AbstractDofHandler
    cellset::CS
    cell_dofs::CD
    cell_dofs_offset::CO
    ndofs_per_cell::Int
    nnodes_per_cell::Int
    # On the host we have a Vector{Symbol}, but Symbol is not a bitstype and cannot
    # be stored in device memory. Therefore we store index of the field as a namedtuple
    field_indices::FN
    dof_ranges::DR
    grid::G
end

Ferrite.get_grid(dh::DeviceSubDofHandler) = dh.grid

function Adapt.adapt_structure(to, sdh::DeviceSubDofHandler)
    return DeviceSubDofHandler(
        Adapt.adapt(to, sdh.cellset),
        Adapt.adapt(to, sdh.cell_dofs),
        Adapt.adapt(to, sdh.cell_dofs_offset),
        sdh.ndofs_per_cell,
        sdh.nnodes_per_cell,
        sdh.field_indices,
        sdh.dof_ranges,
        Adapt.adapt(to, sdh.grid),
    )
end

Ferrite.nnodes_per_cell(sdh::DeviceSubDofHandler) = sdh.nnodes_per_cell
Ferrite.ndofs_per_cell(sdh::DeviceSubDofHandler) = sdh.ndofs_per_cell

Base.@propagate_inbounds function Ferrite.dof_range(sdh::DeviceSubDofHandler, field_idx::Int)
    return sdh.dof_ranges[field_idx]
end
Base.@propagate_inbounds function Ferrite.dof_range(sdh::DeviceSubDofHandler, field_name::Symbol)
    return Ferrite.dof_range(sdh, sdh.field_indices[field_name])
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
        field_indices = NamedTuple{ntuple(i->dh.field_names[i], length(dh.field_names))}(collect(1:length(dh.field_names)))
        DeviceSubDofHandler(
            adapt(backend, collect(Int, sdh.cellset)), cell_dofs, cell_dofs_offset,
            sdh.ndofs_per_cell, nnodes_per_cell(dh.grid, first(sdh.cellset)), field_indices, dof_ranges, gpu_grid
        )
    end
    return HostDofHandler(subdofhandlers, gpu_grid, dh)
end

function Adapt.adapt_structure(to::KA.Backend, dh::DofHandler)
    return HostDofHandler(to, dh)
end

Ferrite.get_grid(dh::HostDofHandler) = dh.grid
