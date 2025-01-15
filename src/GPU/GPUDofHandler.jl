# This file defines the `GPUDofHandler` type, which is a degree of freedom handler that is stored on the GPU.
# Therefore most of the functions are same as the ones defined in dof_handler.jl, but executable on the GPU.

"""
    AbstractGPUDofHandler <: Ferrite.AbstractDofHandler

Abstract type representing degree-of-freedom (DoF) handlers for GPU-based
finite element computations. This serves as the base type for GPU-specific
DoF handler implementations.
"""
abstract type AbstractGPUDofHandler <: AbstractDofHandler end

struct GPUSubDofHandler{VEC_INT, Ti, VEC_IP} <: AbstractGPUDofHandler
    cellset::VEC_INT
    field_names::VEC_INT # cannot use symbols in GPU
    field_interpolations::VEC_IP
    ndofs_per_cell::Ti
end

## IDEA: to have multiple interfaces for dofhandlers (e.g. one domain dofhandler, multiple subdomains)
struct GPUDofHandler{SUB_DOFS <: AbstractArray{<:AbstractGPUDofHandler, 1}, CDOFS <: AbstractArray{<:Number, 1}, VEC_INT <: AbstractArray{Int32, 1}, GRID <: AbstractGrid} <: AbstractGPUDofHandler
    subdofhandlers::SUB_DOFS
    cell_dofs::CDOFS
    grid::GRID
    cell_dofs_offset::VEC_INT
    cell_to_subdofhandler::VEC_INT
end


function ndofs_per_cell(dh::GPUDofHandler, cell::Ti) where {Ti <: Integer}
    sdhidx = dh.cell_to_subdofhandler[cell]
    sdhidx ∉ 1:length(dh.subdofhandlers) && return 0 # Dof handler is just defined on a subdomain
    return ndofs_per_cell(dh.subdofhandlers[sdhidx])
end
ndofs_per_cell(sdh::GPUSubDofHandler) = sdh.ndofs_per_cell
cell_dof_offset(dh::GPUDofHandler, i::Int32) = dh.cell_dofs_offset[i]
get_grid(dh::GPUDofHandler) = dh.grid

"""
    celldofs(dh::GPUDofHandler, i::Int32)

Return the cell degrees of freedom (DoFs) for the given cell index `i` in the
`GPUDofHandler`.

# Arguments
- `dh`: A `GPUDofHandler` instance.
- `i::Int32`: The index of the cell.

# Returns
A `SubArray` (view) representing the DoFs for the specified cell.
"""
function celldofs(dh::GPUDofHandler, i::Int32)
    offset = cell_dof_offset(dh, i)
    ndofs = ndofs_per_cell(dh, i)
    view = @view dh.cell_dofs[offset:(offset + ndofs - Int32(1))]
    return view
end


# TODO: Delete all below this line
"""
    LocalsGPUDofHandler{DH, LOCAL_MATRICES, LOCAL_VECTORS} <: AbstractGPUDofHandler

This object acts as a temporary data structure for storing local stiffness matrices and force vectors, when
dynamic shared memory doesn't have enough space, to be used in GPU kernel by GPU cell iterators.

# Fields
- `dh::DH`: Base DoF handler (e.g., `GPUDofHandler`).
- `Kes::LOCAL_MATRICES`: Local stiffness matrices for each cell (3rd order tensor).
- `fes::LOCAL_VECTORS`: Local force vectors for each cell (2nd order tensor).
"""
struct LocalsGPUDofHandler{DH <: AbstractDofHandler, LOCAL_MATRICES, LOCAL_VECTORS} <: AbstractGPUDofHandler
    dh::DH
    Kes::LOCAL_MATRICES
    fes::LOCAL_VECTORS
end

# Accessor functions for LocalsGPUDofHandler
dofhandler(dh::LocalsGPUDofHandler) = dh.dh
localkes(dh::LocalsGPUDofHandler) = dh.Kes
localfes(dh::LocalsGPUDofHandler) = dh.fes
cellke(dh::LocalsGPUDofHandler, i::Int32) = @view dh.Kes[i, :, :]
cellfe(dh::LocalsGPUDofHandler, i::Int32) = @view dh.fes[i, :]
