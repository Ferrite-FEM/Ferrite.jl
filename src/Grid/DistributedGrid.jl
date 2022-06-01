using Metis
using MPI

"""
"""
abstract type AbstractDistributedGrid{sdim} <: AbstractGrid{sdim} end

# TODO the following three structs can be merged to one struct with type parameter.
"""
"""
struct SharedVertex
    local_idx::VertexIndex
    remote_vertices::Dict{Int,Vector{VertexIndex}}
end

"""
"""
struct SharedFace
    local_idx::FaceIndex
    remote_faces::Dict{Int,Vector{FaceIndex}}
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
    # The entities consistently contain their *Index, because faces and edges are not materialized. 
    shared_vertices::Dict{VertexIndex,SharedVertex}
    shared_edges::Dict{EdgeIndex,SharedEdge}
    shared_faces::Dict{FaceIndex,SharedFace}
end

"""
"""
function DistributedGrid(grid_to_distribute::Grid{dim,C,T}, grid_comm::MPI.Comm; partition_alg = :RECURSIVE) where {dim,C,T}
    grid_topology = ExclusiveTopology(grid_to_distribute)
    return DistributedGrid(grid_to_distribute, grid_topology, grid_comm; partition_alg=partition_alg)
end

function create_partitioning(grid::Grid{dim,C,T}, grid_topology::ExclusiveTopology, n_partitions, partition_alg) where {dim,C,T}
    N = getncells(grid)
    @assert N > 0
    
    if n_partitions == 1
        return ones(N)
    end

    # Set up the element connectivity graph
    xadj = Vector{Metis.idx_t}(undef, N+1)
    xadj[1] = 1
    adjncy = Vector{Metis.idx_t}(undef, 0)
    @inbounds for i in 1:N
        n_neighbors = 0
        for neighbor ∈ getneighborhood(grid_topology, grid, CellIndex(i))
            push!(adjncy, neighbor)
            n_neighbors += 1
        end
        xadj[i+1] = xadj[i] + n_neighbors
    end

    # Generate a partitioning
    return Metis.partition(
        Metis.Graph(
            Metis.idx_t(N),
            xadj,
            adjncy
        ),
        n_partitions;
        alg=partition_alg
    )
end

"""
"""
function DistributedGrid(grid_to_distribute::Grid{dim,C,T}, grid_topology::ExclusiveTopology, grid_comm::MPI.Comm; partition_alg = :RECURSIVE) where {dim,C,T}
    N = getncells(grid_to_distribute)
    @assert N > 0

    my_rank = MPI.Comm_rank(grid_comm)+1

    parts = create_partitioning(grid_to_distribute, grid_topology, MPI.Comm_size(grid_comm), partition_alg)

    # Start extraction of local grid
    # 1. Extract local cells
    local_cells = getcells(grid_to_distribute)[[i for i ∈ 1:N if parts[i] == my_rank]]

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
    begin
        global_nodes = getnodes(grid_to_distribute)
        for global_node_idx ∈ local_node_index_set
            local_node_idx = global_to_local_node_map[global_node_idx]
            local_nodes[local_node_idx] = global_nodes[global_node_idx]
        end
    end

    # 5. Transform cell indices
    for local_cell_idx ∈ 1:length(local_cells)
        local_cells[local_cell_idx] = C(map(global_node_idx -> global_to_local_node_map[global_node_idx], local_cells[local_cell_idx].nodes))
    end

    # 6. Extract sets
    # @TODO deduplicate the code. We should be able to merge each of these into a macro or function.
    # We build this map now, so we avoid the communication later.
    global_to_local_cell_map = Dict{Int,Dict{Int,Int}}()
    for rank ∈ 1:MPI.Comm_size(grid_comm)
        global_to_local_cell_map[rank] = Dict{Int,Int}()
        next_local_cell_idx = 1
        for global_cell_idx ∈ 1:N
            if parts[global_cell_idx] == rank
                global_to_local_cell_map[rank][global_cell_idx] = next_local_cell_idx
                next_local_cell_idx += 1
            end
        end
    end

    cellsets = Dict{String,Set{Int}}()
    for key ∈ keys(grid_to_distribute.cellsets)
        cellsets[key] = Set{Int}() # create empty set, so it does not crash during assembly
        for global_cell_idx ∈ grid_to_distribute.cellsets[key]
            if haskey(global_to_local_cell_map[my_rank], global_cell_idx)
                push!(cellsets[key], global_to_local_cell_map[my_rank][global_cell_idx])
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
            if haskey(global_to_local_cell_map[my_rank], global_cell_idx)
                push!(facesets[key], FaceIndex(global_to_local_cell_map[my_rank][global_cell_idx], i))
            end
        end
    end

    edgesets = Dict{String,Set{EdgeIndex}}()
    for key ∈ keys(grid_to_distribute.edgesets)
        edgesets[key] = Set{EdgeIndex}() # create empty set, so it does not crash during assembly
        for (global_cell_idx, i) ∈ grid_to_distribute.edgesets[key]
            if haskey(global_to_local_cell_map[my_rank], global_cell_idx)
                push!(edgesets[key], EdgeIndex(global_to_local_cell_map[my_rank][global_cell_idx], i))
            end
        end
    end

    vertexsets = Dict{String,Set{VertexIndex}}()
    for key ∈ keys(grid_to_distribute.vertexsets)
        vertexsets[key] = Set{VertexIndex}() # create empty set, so it does not crash during assembly
        for (global_cell_idx, i) ∈ grid_to_distribute.vertexsets[key]
            if haskey(global_to_local_cell_map[my_rank], global_cell_idx)
                push!(vertexsets[key], VertexIndex(global_to_local_cell_map[my_rank][global_cell_idx], i))
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

    shared_vertices = Dict{VertexIndex,SharedVertex}()
    shared_edges = Dict{EdgeIndex,SharedEdge}()
    shared_faces = Dict{FaceIndex,SharedFace}()
    for (global_cell_idx,global_cell) ∈ enumerate(getcells(grid_to_distribute))
        if parts[global_cell_idx] == my_rank
            # Vertex
            for (i, _) ∈ enumerate(vertices(global_cell))
                cell_vertex = VertexIndex(global_cell_idx, i)
                remote_vertices = Dict{Int,Vector{VertexIndex}}()
                for (global_cell_neighbor_idx, j) ∈ getneighborhood(grid_topology, grid_to_distribute, cell_vertex, true)
                    other_rank = parts[global_cell_neighbor_idx]
                    if other_rank != my_rank
                        n1 = vertices(getcells(grid_to_distribute,global_cell_idx))[i]
                        n2 = vertices(getcells(grid_to_distribute,global_cell_neighbor_idx))[j]
                        if n1 == n2
                            if !haskey(remote_vertices,other_rank)
                                remote_vertices[other_rank] = Vector(undef,0)
                            end
                            @debug println("Detected shared vertex $cell_vertex neighbor $(VertexIndex(global_cell_neighbor_idx,j)) (R$my_rank)")
                            push!(remote_vertices[other_rank], VertexIndex(global_to_local_cell_map[other_rank][global_cell_neighbor_idx], j))
                        end
                    end
                end

                if length(remote_vertices) > 0
                    idx = VertexIndex(global_to_local_cell_map[my_rank][global_cell_idx], i)
                    shared_vertices[idx] = SharedVertex(idx, remote_vertices)
                end
            end

            # # Edge
            # if dim > 2
            #     for (i, global_vertex_idx) ∈ enumerate(edges(global_cell))
            #         cell_edge = EdgeIndex(global_cell_idx, i)
            #         for (global_cell_neighbor_idx, j) ∈ getneighborhood(grid_topology, grid_to_distribute, cell_edge)
            #             if parts[global_cell_neighbor_idx] != my_rank
            #                 push!(shared_edges, cell_edge)
            #             end
            #         end
            #     end
            # end

            # # Face
            # if dim > 1
            #     for (i, global_vertex_idx) ∈ enumerate(faces(global_cell))
            #         cell_face = FaceIndex(global_cell_idx, i)
            #         for (global_cell_neighbor_idx, j) ∈ getneighborhood(grid_topology, grid_to_distribute, cell_face)
            #             if parts[global_cell_neighbor_idx] != my_rank
            #                 push!(shared_faces, cell_face)
            #             end
            #         end
            #     end
            # end
        end
    end

    return DistributedGrid(grid_comm,grid_comm,local_grid,shared_vertices,shared_edges,shared_faces)
end

@inline getlocalgrid(dgrid::AbstractDistributedGrid) = dgrid.local_grid

@inline getcells(dgrid::AbstractDistributedGrid) = getcells(getlocalgrid(grid))
@inline getcells(dgrid::AbstractDistributedGrid, v::Union{Int, Vector{Int}}) = getcells(getlocalgrid(grid),v)
@inline getcells(dgrid::AbstractDistributedGrid, setname::String) = getcells(getlocalgrid(grid),setname)
"Returns the number of cells in the `<:AbstractDistributedGrid`."
@inline getncells(dgrid::AbstractDistributedGrid) = length(getcells(getlocalgrid(dgrid)))
"Returns the celltype of the `<:AbstractDistributedGrid`."
@inline getcelltype(dgrid::AbstractDistributedGrid) = eltype(getcells(getlocalgrid(dgrid)))
@inline getcelltype(dgrid::AbstractDistributedGrid, i::Int) = typeof(getcells(getlocalgrid(dgrid),i))
