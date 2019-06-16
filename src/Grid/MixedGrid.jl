
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

@inline function getcoordinates!(x::Vector{Vec{dim,T}}, grid::MixedGrid{dim,T}, cell::Int) where {dim,T}
    @assert length(x) == nnodes(grid.cells[cell])
    @inbounds for i in 1:nnodes(grid.cells[cell])
        x[i] = grid.nodes[grid.cells[cell].nodes[i]].x
    end
end

@inline function getcoordinates(grid::MixedGrid{dim,C,T}, cell::Int) where {dim,C,T}
    nodeidx = grid.cells[cell].nodes
    return [grid.nodes[i].x for i in nodeidx]::Vector{Vec{dim,T}}
end
