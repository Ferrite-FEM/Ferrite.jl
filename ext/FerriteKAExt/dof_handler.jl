struct DeviceSubDofHandler{
        sdim,
        CS <: AbstractVector{Int}, CD <: AbstractVector{Int},
        CO <: AbstractVector{Int}, FN <: NamedTuple, DR <: Tuple, G <: AbstractGrid{sdim},
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
    # Grid information
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
struct HostDofHandler{sdim, G <: Grid{sdim}, DH <: AbstractDofHandler} <: AbstractDofHandler
    subdofhandlers::Vector
    grid::G
    original_dh::DH
end

function HostDofHandler(backend, dh::DofHandler)
    grid_cpu = get_grid(dh)
    nodes_cpu = getnodes(grid_cpu)
    nodes_gpu = adapt(backend, nodes_cpu)
    cell_dofs = adapt(backend, dh.cell_dofs)
    cell_dofs_offset = adapt(backend, dh.cell_dofs_offset)
    subdofhandlers = map(dh.subdofhandlers) do sdh
        dof_ranges = Tuple(Ferrite.dof_range(sdh, i) for i in 1:length(sdh.field_names))
        field_indices = NamedTuple{ntuple(i -> dh.field_names[i], length(dh.field_names))}(collect(1:length(dh.field_names)))
        # invert cellset and build a device container grid with only a single cell type
        cellset = collect(Int, sdh.cellset)
        global_to_local_cellid = zeros(Int, getncells(grid_cpu))
        local_cells = Vector{typeof(getcells(grid_cpu, first(cellset)))}(undef, length(cellset))
        for (i, cid) in enumerate(cellset)
            global_to_local_cellid[cid] = i
            local_cells[i] = getcells(grid_cpu, cid)
        end
        grid_gpu = DeviceSubGrid(adapt(backend, local_cells), nodes_gpu, adapt(backend, global_to_local_cellid))

        DeviceSubDofHandler(
            adapt(backend, cellset), cell_dofs, cell_dofs_offset,
            sdh.ndofs_per_cell, nnodes_per_cell(grid_cpu, first(cellset)), field_indices,
            dof_ranges, grid_gpu,
        )
    end
    return HostDofHandler(subdofhandlers, grid_cpu, dh)
end

function Adapt.adapt_structure(to::KA.Backend, dh::DofHandler)
    return HostDofHandler(to, dh)
end

Ferrite.get_grid(dh::HostDofHandler) = dh.grid
