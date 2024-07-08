struct GPUDofHandler{CDOFS<:AbstractArray{<:Number,1},VEC_INT<:AbstractArray{Int32,1},GRID<:AbstractGrid}<: AbstractDofHandler
    cell_dofs::CDOFS
    grid::GRID
    cell_dofs_offset::VEC_INT
    closed::Bool
    ndofs_cell::VEC_INT
end

@inline isclosed(dh::GPUDofHandler) = dh.closed

@inline ndofs_per_cell(dh::GPUDofHandler, i::Int32)= dh.ndofs_cell[i]
@inline cell_dof_offset(dh::GPUDofHandler, i::Int32) = dh.cell_dofs_offset[i]


"""
    celldofs(dh::GPUDofHandler, i::Int32)

Return the cell degrees of freedom for the given cell index `i` in the `GPUDofHandler` `dh`.
"""
function celldofs(dh::GPUDofHandler, i::Int32)
    offset = cell_dof_offset(dh, i)
    ndofs = ndofs_per_cell(dh, i)
    return @view dh.cell_dofs[offset:(offset+ndofs-Int32(1))]
end
