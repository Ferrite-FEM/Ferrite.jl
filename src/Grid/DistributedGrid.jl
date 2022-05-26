using Metis
using MPI

"""
"""
abstract type AbstractDistributedGrid{sdim} <: AbstractGrid{sdim} end

struct SharedVertex
    local_idx::VertexIndex
    remote_vertices::Dict{Int,Vector{VertexIndex}}
end

"""
"""
struct SharedFace
    local_idx::FaceIndex
    remote_edges::Dict{Int,Vector{FaceIndex}}
end

"""
"""
struct SharedEdge
    local_idx::EdgeIndex
    remote_edges::Dict{Int,Vector{EdgeIndex}}
end

"""
"""
mutable struct DistributedGrid{dim,C<:AbstractCell,T<:Real} <: AbstractDistributedGrid{dim}
    # Dense comminicator on the grid
    grid_comm::MPI.Comm
    # Sparse communicator along the shared vertex neighbors
    # We only need this one because the vertices induce the edge and face neighbors.
    interface_comm::MPI.Comm
    # Here we store the full local grid
    local_grid::Grid{dim,C,T}
    # Local copies of the shared entities of the form (local index, (process id in grid_comm, remote index))
    shared_vertices::Vector{SharedVertex}
    shared_edges::Vector{SharedEdge}
    shared_faces::Vector{SharedFace}
end

"""
"""
function DistributedGrid(grid_to_distribute::Grid, grid_comm::MPI.Comm; partition_alg = :RECURSIVE)
    grid_topology = ExclusiveTopology(grid_to_distribute)
    return DistributedGrid(grid_to_distribute, grid_topology, grid_comm; partition_alg=partition_alg)
end

"""
"""
function DistributedGrid(grid_to_distribute::Grid{dim,C,T}, grid_topology::ExclusiveTopology, grid_comm::MPI.Comm; partition_alg = :RECURSIVE) where {dim,C,T}
    N = getncells(grid_to_distribute)
    @assert N > 0

    # Set up the element connectivity graph
    xadj = Vector{Metis.idx_t}(undef, N+1)
    xadj[1] = 1
    adjncy = Vector{Metis.idx_t}(undef, 0)
    @inbounds for i in 1:N
        n_neighbors = 0
        for neighbor ∈ getneighborhood(grid_topology, grid_to_distribute, CellIndex(i))
            push!(adjncy, neighbor)
            n_neighbors += 1
        end
        xadj[i+1] = xadj[i] + n_neighbors
    end

    # Generate a partitioning
    parts = Metis.partition(
        Metis.Graph(
            Metis.idx_t(N),
            xadj,
            adjncy
        ),
        MPI.Comm_size(grid_comm);
        alg=partition_alg
    )

    # Start extraction of local grid
    # 1. Extract local cells
    local_cells = getcells(grid_to_distribute)[[i for i ∈ 1:N if parts[i] == (MPI.Comm_rank(grid_comm)+1)]]

    # 2. Find unique nodes
    local_node_index_set = Set{Int}()
    for cell ∈ local_cells
        for global_node_idx ∈ cell.nodes # @TODO abstraction
            push!(local_node_index_set, global_node_idx)
        end
    end

    # 3. Build a map for global to local node indices
    next_local_node_idx = 1
    global_to_local_node_map = Dict{Int,Int}()
    for global_node_idx ∈ local_node_index_set
        global_to_local_node_map[global_node_idx] = next_local_node_idx
        next_local_node_idx += 1
    end

    # 4. Extract local nodes
    local_nodes = Vector{Node{dim,T}}(undef,length(local_node_index_set))
    global_nodes = getnodes(grid_to_distribute)
    for global_node_idx ∈ local_node_index_set
        local_node_idx = global_to_local_node_map[global_node_idx]
        local_nodes[local_node_idx] = global_nodes[global_node_idx]
    end

    # 5. Transform cell indices
    for local_cell_idx ∈ 1:length(local_cells)
        local_cells[local_cell_idx] = C(map(global_node_idx -> global_to_local_node_map[global_node_idx], local_cells[local_cell_idx].nodes))
    end

    # 6. Extract sets
    # @TODO deduplicate the code. We should be able to merge each of these into a macro or function.
    global_to_local_cell_map = Dict{Int,Int}()
    next_local_cell_idx = 1
    for global_cell_idx ∈ 1:N
        if parts[global_cell_idx] == (MPI.Comm_rank(grid_comm)+1)
            global_to_local_cell_map[global_cell_idx] = next_local_cell_idx
            next_local_cell_idx += 1
        end
    end

    cellsets = Dict{String,Set{Int}}()
    for key ∈ keys(grid_to_distribute.cellsets)
        cellsets[key] = Set{Int}() # create empty set, so it does not crash during assembly
        for global_cell_idx ∈ grid_to_distribute.cellsets[key]
            if haskey(global_to_local_cell_map, global_cell_idx)
                push!(cellsets[key], global_to_local_cell_map[global_cell_idx])
            end
        end
    end

    nodesets = Dict{String,Set{Int}}()
    for key ∈ keys(grid_to_distribute.nodesets)
        nodesets[key] = Set{Int}() # create empty set, so it does not crash during assembly
        for global_node_idx ∈ grid_to_distribute.nodesets[key]
            if haskey(global_to_local_node_map, global_node_idx)
                push!(nodesets[key], global_to_local_node_map[global_node_idx])
            end
        end
    end

    facesets = Dict{String,Set{FaceIndex}}()
    for key ∈ keys(grid_to_distribute.facesets)
        facesets[key] = Set{FaceIndex}() # create empty set, so it does not crash during assembly
        for (global_cell_idx, i) ∈ grid_to_distribute.facesets[key]
            if haskey(global_to_local_cell_map, global_cell_idx)
                push!(facesets[key], FaceIndex(global_to_local_cell_map[global_cell_idx], i))
            end
        end
    end

    edgesets = Dict{String,Set{EdgeIndex}}()
    for key ∈ keys(grid_to_distribute.edgesets)
        edgesets[key] = Set{EdgeIndex}() # create empty set, so it does not crash during assembly
        for (global_cell_idx, i) ∈ grid_to_distribute.edgesets[key]
            if haskey(global_to_local_cell_map, global_cell_idx)
                push!(edgesets[key], EdgeIndex(global_to_local_cell_map[global_cell_idx], i))
            end
        end
    end

    vertexsets = Dict{String,Set{VertexIndex}}()
    for key ∈ keys(grid_to_distribute.vertexsets)
        vertexsets[key] = Set{VertexIndex}() # create empty set, so it does not crash during assembly
        for (global_cell_idx, i) ∈ grid_to_distribute.vertexsets[key]
            if haskey(global_to_local_cell_map, global_cell_idx)
                push!(vertexsets[key], VertexIndex(global_to_local_cell_map[global_cell_idx], i))
            end
        end
    end

    local_grid = Grid(
        local_cells,
        local_nodes,
        cellsets=cellsets,
        nodesets=nodesets,
        facesets=facesets,
        edgesets=edgesets,
        vertexsets=vertexsets
    )

    return DistributedGrid(grid_comm,grid_comm,local_grid,Vector{SharedVertex}([]),Vector{SharedEdge}([]),Vector{SharedFace}([]))
end