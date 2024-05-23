
struct GPUDofHandler{CDOFS<:AbstractArray{<:Number,1},GRID<:AbstractGrid}<: Ferrite.AbstractDofHandler
    cell_dofs::CDOFS
    grid::GRID
end


function GPUDofHandler(dh::DofHandler)
    GPUDofHandler(dh.cell_dofs,dh.grid)
end