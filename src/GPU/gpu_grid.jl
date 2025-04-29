# This file defines the GPUGrid type, which is a grid that is stored on the GPU. Therefore most of the
# functions are same as the ones defined in grid.jl, but executable on the GPU.

abstract type AbstractGPUGrid{dim} <: AbstractGrid{dim} end

struct GPUGrid{dim, CELLVEC <: AbstractArray, NODEVEC <: AbstractArray} <: AbstractGPUGrid{dim}
    cells::CELLVEC
    nodes::NODEVEC
end

function GPUGrid(
        cells::CELLVEC,
        nodes::NODEVEC
    ) where {C <: AbstractCell, CELLVEC <: AbstractArray{C, 1}, NODEVEC <: AbstractArray{Node{dim, T}}} where {dim, T}
    return GPUGrid{dim, CELLVEC, NODEVEC}(cells, nodes)
end

get_coordinate_type(::GPUGrid{dim, CELLVEC, NODEVEC}) where
{C <: AbstractCell, CELLVEC <: AbstractArray{C, 1}, NODEVEC <: AbstractArray{Node{dim, T}}} where
{dim, T} = Vec{dim, T} # Node is baked into the mesh type.


# Note: For functions that takes blockIdx as an argument, we need to use Int32 explicitly,
# otherwise the compiler will not be able to infer the type of the argument and throw a dynamic function invokation error.
@inline getcells(grid::GPUGrid, v::Union{Int32, Vector{Int32}}) = grid.cells[v]
@inline getnodes(grid::GPUGrid, v::Int32) = grid.nodes[v]

"""
    getcoordinates(grid::Ferrite.GPUGrid,e::Int32)

Return the coordinates of the nodes of the element `e` in the `GPUGrid` `grid`.
"""
function getcoordinates(grid::GPUGrid, e::Int32)
    # e is the element index.
    CT = get_coordinate_type(grid)
    cell = getcells(grid, e)
    N = nnodes(cell)
    x = MVector{N, CT}(undef) # local array to store the coordinates of the nodes of the cell.
    node_ids = get_node_ids(cell)
    for i in 1:length(x)
        x[i] = get_node_coordinate(grid, node_ids[i])
    end

    return SVector(x...)
end
