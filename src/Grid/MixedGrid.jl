
mutable struct MixedGrid{dim,C,T<:Real} <: JuAFEM.AbstractGrid
    cells::Vector{C}
    nodes::Vector{Node{dim,T}}
    # Sets
    cellsets::Dict{String,Set{Int}}
    nodesets::Dict{String,Set{Int}}
    facesets::Dict{String,Set{Tuple{Int,Int}}} # TODO: This could be Set{FaceIndex} which could result in nicer use later
    # Boundary matrix (faces per cell × cell)
    boundary_matrix::SparseMatrixCSC{Bool,Int}
end

function MixedGrid(
    cells::Vector{C},
    nodes::Vector{Node{dim,T}};
    cellsets::Dict{String,Set{Int}}=Dict{String,Set{Int}}(),
    nodesets::Dict{String,Set{Int}}=Dict{String,Set{Int}}(),
    facesets::Dict{String,Set{Tuple{Int,Int}}}=Dict{String,Set{Tuple{Int,Int}}}(),
          boundary_matrix::SparseMatrixCSC{Bool,Int}=spzeros(Bool, 0, 0)) where {dim,C,T}
    return MixedGrid(cells, nodes, cellsets, nodesets, facesets, boundary_matrix)
end
