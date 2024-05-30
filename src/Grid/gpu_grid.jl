# This file defines the GPUGrid type, which is a grid that is stored on the GPU. It is a subtype of AbstractGrid.
# TODO: Refactor type parameters to be more consistent with the rest of the codebase.
struct GPUGrid{dim,CELLVEC<:AbstractArray,NODEVEC<:AbstractArray}<: Ferrite.AbstractGrid{dim}
    cells::CELLVEC
    nodes::NODEVEC
end

function GPUGrid(cells::CELLVEC,
                 nodes::NODEVEC) where {C<:Ferrite.AbstractCell,CELLVEC<:AbstractArray{C,1},NODEVEC<:AbstractArray{Node{dim,T}}} where {dim,T}
    GPUGrid{dim,CELLVEC,NODEVEC}(cells,nodes)
end

get_coordinate_type(::GPUGrid{dim,CELLVEC,NODEVEC}) where 
    {C<:Ferrite.AbstractCell,CELLVEC<:AbstractArray{C,1},NODEVEC<:AbstractArray{Node{dim,T}}} where 
    {dim,T} = Vec{dim,T} # Node is baked into the mesh type.





# Note: For functions that takes blockIdx as an argument, we need to use Int32 explicitly,
# otherwise the compiler will not be able to infer the type of the argument and throw a dynamic function invokation error.   
@inline getcells(grid::GPUGrid, v::Union{Int32, Vector{Int32}}) = grid.cells[v]

# This function is used to get the coordinates of a cell on the GPU.
@inline function getcoordinates(grid::Ferrite.GPUGrid)
    # b_idx is the block index which is the same as the cell index.
    # Each block corresponds to a cell, so we can use the block index to get the cell.
    b_idx = blockIdx().x  # element index

    CT = get_coordinate_type(grid)
    cell = getcells(grid, b_idx)
    N = nnodes(cell)

    # Create a CuStaticSharedArray to store the coordinates of the cell.
    arr = CuStaticSharedArray(CT, N)
    # We are using 2D threads (each (i,j) represents an element in local stiffness matrix), so we need to get the x and y thread indices.
    # no. threads in x = no. of shape functions
    # no. threads in y = no. of shape functions
    tx = threadIdx().x
    ty = threadIdx().y
    q_point = threadIdx().z

    # no. of nodes <= no. of shape functions (and we need only one threads direction)
    if ty == 1 && q_point == 1  && tx <=N 
        arr[tx] = get_node_coordinate(grid, Ferrite.get_node_ids(cell)[tx]) 
    end

    # Sync all threads to make sure all the values are written to the array.
    sync_threads()
    # return the array as a SVector, so that it can fit with the rest of the codebase. 
    return SVector{N,CT}(arr)
end


