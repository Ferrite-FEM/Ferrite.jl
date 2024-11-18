# This file defines the `GPUDofHandler` type, which is a degree of freedom handler that is stored on the GPU.
# Therefore most of the functions are same as the ones defined in dof_handler.jl, but executable on the GPU.

abstract type AbstractGPUDofHandler <: AbstractDofHandler  end

struct GPUDofHandler{CDOFS <: AbstractArray{<:Number, 1}, VEC_INT <: AbstractArray{Int32, 1}, GRID <: AbstractGrid} <: AbstractGPUDofHandler
    cell_dofs::CDOFS
    grid::GRID
    cell_dofs_offset::VEC_INT
    ndofs_cell::VEC_INT
end

ndofs_per_cell(dh::GPUDofHandler, i::Int32) = dh.ndofs_cell[i]
cell_dof_offset(dh::GPUDofHandler, i::Int32) = dh.cell_dofs_offset[i]
get_grid(dh::GPUDofHandler) = dh.grid

"""
    celldofs(dh::GPUDofHandler, i::Int32)

Return the cell degrees of freedom for the given cell index `i` in the `GPUDofHandler` `dh`.
"""
function celldofs(dh::GPUDofHandler, i::Int32)
    offset = cell_dof_offset(dh, i)
    ndofs = ndofs_per_cell(dh, i)
    view = @view dh.cell_dofs[offset:(offset + ndofs - Int32(1))]
    return view
end


struct LocalsGPUDofHandler{DH <: AbstractDofHandler, LOCAL_MATRICES, LOCAL_VECTORS} <: AbstractGPUDofHandler
    dh::DH
    Kes::LOCAL_MATRICES # local stiffness matrices for each running cell (3rd order tensor (e,i,j))
    fes::LOCAL_VECTORS # local force vectors for each running cell (2nd order tensor (e,i))
end

## Accessors ##
dofhandler(dh::LocalsGPUDofHandler) = dh.dh
cellke(dh::LocalsGPUDofHandler, e::Int32) = @view dh.Kes[e, :, :]
cellfe(dh::LocalsGPUDofHandler, e::Int32) = @view dh.fes[e, :]
