# This file defines the `GPUDofHandler` type, which is a degree of freedom handler that is stored on the GPU.
# Therefore most of the functions are same as the ones defined in dof_handler.jl, but executable on the GPU.

"""
    AbstractGPUDofHandler <: Ferrite.AbstractDofHandler

Abstract type representing degree-of-freedom (DoF) handlers for GPU-based
finite element computations. This serves as the base type for GPU-specific
DoF handler implementations.
"""
abstract type AbstractGPUDofHandler <: Ferrite.AbstractDofHandler end


struct GPUDofHandler{CDOFS <: AbstractArray{<:Number, 1}, VEC_INT <: AbstractArray{Int32, 1}, GRID <: AbstractGrid} <: AbstractGPUDofHandler
    cell_dofs::CDOFS
    grid::GRID
    cell_dofs_offset::VEC_INT
    ndofs_cell::VEC_INT
end

"""
    ndofs_per_cell(dh::GPUDofHandler, i::Int32)

Return the number of degrees of freedom (DoFs) associated with the cell at
index `i` in the `GPUDofHandler`.

# Arguments
- `dh`: A `GPUDofHandler` instance.
- `i::Int32`: The index of the cell.

# Returns
The number of DoFs for the specified cell as an `Int32`.
"""
ndofs_per_cell(dh::GPUDofHandler, i::Int32) = dh.ndofs_cell[i]

"""
    cell_dof_offset(dh::GPUDofHandler, i::Int32)

Return the offset into the `cell_dofs` array for the cell at index `i`.

# Arguments
- `dh`: A `GPUDofHandler` instance.
- `i::Int32`: The index of the cell.

# Returns
The offset in the `cell_dofs` array as an `Int32`.
"""
cell_dof_offset(dh::GPUDofHandler, i::Int32) = dh.cell_dofs_offset[i]

"""
    get_grid(dh::GPUDofHandler)

Return the computational grid associated with the `GPUDofHandler`.

# Arguments
- `dh`: A `GPUDofHandler` instance.

# Returns
The computational grid.
"""
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
