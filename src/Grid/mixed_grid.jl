mutable struct MixedGrid{dim, C, T<:Real} <: Ferrite.AbstractGrid{dim}
    cells::C # Tuple of concretely typed cell vectors
    ncells_per_vector::Vector{Int}
    nodes::Vector{Node{dim,T}}
    # Sets
    cellsets::Dict{String,Set{Int}}
    nodesets::Dict{String,Set{Int}}
    facesets::Dict{String,Set{FaceIndex}} 
    edgesets::Dict{String,Set{EdgeIndex}} 
    vertexsets::Dict{String,Set{VertexIndex}} 
    # Boundary matrix (faces per cell × cell)
    boundary_matrix::SparseMatrixCSC{Bool,Int}
end

function MixedGrid(cells::C,
        nodes::Vector{Node{dim,T}};
        cellsets::Dict{String,Set{Int}}=Dict{String,Set{Int}}(),
        nodesets::Dict{String,Set{Int}}=Dict{String,Set{Int}}(),
        facesets::Dict{String,Set{FaceIndex}}=Dict{String,Set{FaceIndex}}(),
        edgesets::Dict{String,Set{EdgeIndex}}=Dict{String,Set{EdgeIndex}}(),
        vertexsets::Dict{String,Set{VertexIndex}}=Dict{String,Set{VertexIndex}}(),
        boundary_matrix::SparseMatrixCSC{Bool,Int}=spzeros(Bool, 0, 0)) where {dim,C,T}
        ncells_per_type = collect(length.(cells))
    return MixedGrid(cells, ncells_per_type, nodes, cellsets, nodesets, facesets, edgesets, vertexsets, boundary_matrix)
end

struct CellId{I}
    i::Int
end

function globalid(grid::MixedGrid, cellid::Ferrite.CellId{I}) where I
    global_id = 0
    for i=1:I-1
        global_id += grid.ncells_per_vector[I]
    end
    global_id += cellid.i
    return global_id
end

Ferrite.getcells(grid::MixedGrid, cellid::Ferrite.CellId{I}) where I = grid.cells[I][cellid.i]
Ferrite.getncells(grid::MixedGrid) = sum(grid.ncells_per_vector)

# Inherently type unstable
function local_index(grid::MixedGrid, global_id::Int)
    local_idx = global_id
    for (i, ncells) in enumerate(grid.ncells_per_vector) 
        local_idx - ncells < 1 && return NewCellIndex{i}(local_idx)
        local_idx -= ncells
    end
    error("Local index corresponding to global_id=$global_id not found.")
end
