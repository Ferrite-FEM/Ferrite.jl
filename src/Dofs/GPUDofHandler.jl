struct GPUDofHandler{CDOFS<:AbstractArray{<:Number,1},VEC_INT<:AbstractArray{Int32,1},GRID<:AbstractGrid}<: AbstractDofHandler
    cell_dofs::CDOFS
    grid::GRID
    cell_dofs_offset::VEC_INT
    closed::Bool
    ndofs_cell::VEC_INT
end




isclosed(dh::GPUDofHandler) = dh.closed

ndofs_per_cell(dh::GPUDofHandler, i::Int32)= dh.ndofs_cell[i]

function celldofs(dh::GPUDofHandler, i::Int32)
    offset = dh.cell_dofs_offset[i]
    ndofs = ndofs_per_cell(dh, i)
   return @view dh.cell_dofs[offset:(offset+ndofs-Int32(1))]
end

# function celldofs(dh::GPUDofHandler, i::Int)
#     global_dofs = MVector{Int,ndofs_per_cell(dh, i)}(undef)
#     return celldofs!(global_dofs, dh, i)
# end

# function celldofs!(global_dofs::StaticVector{Int}, dh::GPUDofHandler, i::Int)
#     @cuassert isclosed(dh)
#     @assert length(global_dofs) == ndofs_per_cell(dh, i)
#     unsafe_copyto!(global_dofs, 1, dh.cell_dofs, dh.cell_dofs_offset[i], length(global_dofs))
#     return global_dofs
# end