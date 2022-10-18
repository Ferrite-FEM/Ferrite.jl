using Metis
using MPI

"""
"""
abstract type AbstractDistributedGrid{sdim} <: AbstractGrid{sdim} end

abstract type SharedEntity end

# TODO the following three structs can be merged to one struct with type parameter.
"""
"""
struct SharedVertex <: SharedEntity
    local_idx::VertexIndex
    remote_vertices::Dict{Int,Vector{VertexIndex}}
end

@inline remote_entities(sv::SharedVertex) = sv.remote_vertices

"""
"""
struct SharedFace <: SharedEntity
    local_idx::FaceIndex
    remote_faces::Dict{Int,Vector{FaceIndex}}
end

@inline remote_entities(sf::SharedFace) = sf.remote_faces

"""
"""
struct SharedEdge <: SharedEntity
    local_idx::EdgeIndex
    remote_edges::Dict{Int,Vector{EdgeIndex}}
end

@inline remote_entities(se::SharedEdge) = se.remote_edges

"""
@TODO docs
@TODO PArrays ready constructor
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

@inline get_shared_vertices(dgrid::AbstractDistributedGrid) = dgrid.shared_vertices
@inline get_shared_edges(dgrid::AbstractDistributedGrid) = dgrid.shared_edges
@inline get_shared_faces(dgrid::AbstractDistributedGrid) = dgrid.shared_faces

@inline get_shared_vertex(dgrid::AbstractDistributedGrid, vi::VertexIndex) = dgrid.shared_vertices[vi]
@inline get_shared_edge(dgrid::AbstractDistributedGrid, ei::EdgeIndex) = dgrid.shared_edges[ei]
@inline get_shared_face(dgrid::AbstractDistributedGrid, fi::FaceIndex) = dgrid.shared_faces[fi]

"""
"""
@inline is_shared_vertex(dgrid::AbstractDistributedGrid, vi::VertexIndex) = haskey(dgrid.shared_vertices, vi)
@inline is_shared_edge(dgrid::AbstractDistributedGrid, ei::EdgeIndex) = haskey(dgrid.shared_edges, ei)
@inline is_shared_face(dgrid::AbstractDistributedGrid, fi::FaceIndex) = haskey(dgrid.shared_faces, fi)


"""
Global dense communicator of the distributed grid.
"""
@inline global_comm(dgrid::AbstractDistributedGrid) = dgrid.grid_comm

"""
Graph communicator for shared vertices. Guaranteed to be derived from the communicator 
returned by @global_comm .
"""
@inline vertex_comm(dgrid::AbstractDistributedGrid) = dgrid.interface_comm

"""
"""
function DistributedGrid(grid_to_distribute::Grid{dim,C,T}; grid_comm::MPI.Comm = MPI.COMM_WORLD, partition_alg = :RECURSIVE) where {dim,C,T}
    grid_topology = ExclusiveTopology(grid_to_distribute)
    return DistributedGrid(grid_to_distribute, grid_topology, grid_comm; partition_alg=partition_alg)
end

function create_partitioning(grid::Grid{dim,C,T}, grid_topology::ExclusiveTopology, n_partitions, partition_alg) where {dim,C,T}
    n_cells_global = getncells(grid)
    @assert n_cells_global > 0
    
    if n_partitions == 1
        return ones(Metis.idx_t, n_cells_global)
    end

    # Set up the element connectivity graph
    xadj = Vector{Metis.idx_t}(undef, n_cells_global+1)
    xadj[1] = 1
    adjncy = Vector{Metis.idx_t}(undef, 0)
    @inbounds for i in 1:n_cells_global
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
            Metis.idx_t(n_cells_global),
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
    n_cells_global = getncells(grid_to_distribute)
    @assert n_cells_global > 0

    parts = create_partitioning(grid_to_distribute, grid_topology, MPI.Comm_size(grid_comm), partition_alg)

    DistributedGrid(grid_to_distribute::Grid{dim,C,T}, grid_topology::ExclusiveTopology, grid_comm::MPI.Comm, parts)
end

"""
"""    
function DistributedGrid(grid_to_distribute::Grid{dim,C,T}, grid_topology::ExclusiveTopology, grid_comm::MPI.Comm, parts::Vector{Int32}) where {dim,C,T}
    n_cells_global = getncells(grid_to_distribute)
    @assert n_cells_global > 0 # Empty input mesh...

    my_rank = MPI.Comm_rank(grid_comm)+1

    # Start extraction of local grid
    # 1. Extract local cells
    local_cells = getcells(grid_to_distribute)[[i for i ∈ 1:n_cells_global if parts[i] == my_rank]]
    @assert length(local_cells) > 0 # Cannot handle empty partitions yet

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
        for global_cell_idx ∈ 1:n_cells_global
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
                for other_vertex ∈ getneighborhood(grid_topology, grid_to_distribute, cell_vertex, true)
                    (global_cell_neighbor_idx, j) = other_vertex
                    other_rank = parts[global_cell_neighbor_idx]
                    if other_rank != my_rank
                        if toglobal(grid_to_distribute,cell_vertex) == toglobal(grid_to_distribute,other_vertex)
                            if !haskey(remote_vertices,other_rank)
                                remote_vertices[other_rank] = Vector(undef,0)
                            end
                            @debug println("Detected shared vertex $cell_vertex neighbor $other_vertex (R$my_rank)")
                            push!(remote_vertices[other_rank], VertexIndex(global_to_local_cell_map[other_rank][global_cell_neighbor_idx], j))
                        end
                    end
                end

                if length(remote_vertices) > 0
                    idx = VertexIndex(global_to_local_cell_map[my_rank][global_cell_idx], i)
                    shared_vertices[idx] = SharedVertex(idx, remote_vertices)
                end
            end

            # Face
            if dim > 1
                for (i, _) ∈ enumerate(faces(global_cell))
                    cell_face = FaceIndex(global_cell_idx, i)
                    remote_faces = Dict{Int,Vector{FaceIndex}}()
                    for other_face ∈ getneighborhood(grid_topology, grid_to_distribute, cell_face, true)
                        (global_cell_neighbor_idx, j) = other_face
                        other_rank = parts[global_cell_neighbor_idx]
                        if other_rank != my_rank
                            if toglobal(grid_to_distribute,cell_face) == toglobal(grid_to_distribute,other_face)
                                if !haskey(remote_faces,other_rank)
                                    remote_faces[other_rank] = Vector(undef,0)
                                end
                                @debug println("Detected shared face $cell_face neighbor $other_face (R$my_rank)")
                                push!(remote_faces[other_rank], FaceIndex(global_to_local_cell_map[other_rank][global_cell_neighbor_idx], j))
                            end
                        end
                    end

                    if length(remote_faces) > 0
                        idx = FaceIndex(global_to_local_cell_map[my_rank][global_cell_idx], i)
                        shared_faces[idx] = SharedFace(idx, remote_faces)
                    end
                end
            end

            # Edge
            if dim > 2
                for (i, _) ∈ enumerate(edges(global_cell))
                    cell_edge = EdgeIndex(global_cell_idx, i)
                    remote_edges = Dict{Int,Vector{EdgeIndex}}()
                    for other_edge ∈ getneighborhood(grid_topology, grid_to_distribute, cell_edge, true)
                        (global_cell_neighbor_idx, j) = other_edge
                        other_rank = parts[global_cell_neighbor_idx]
                        if other_rank != my_rank
                            if toglobal(grid_to_distribute,cell_edge) == toglobal(grid_to_distribute,other_edge)
                                if !haskey(remote_edges,other_edge)
                                    remote_edges[other_rank] = Vector(undef,0)
                                end
                                @debug println("Detected shared edge $cell_edge neighbor $other_edge (R$my_rank)")
                                push!(remote_edges[other_rank], EdgeIndex(global_to_local_cell_map[other_rank][global_cell_neighbor_idx], j))
                            end
                        end
                    end

                    if length(remote_edges) > 0
                        idx = EdgeIndex(global_to_local_cell_map[my_rank][global_cell_idx], i)
                        shared_edges[idx] = SharedEdge(idx, remote_edges)
                    end
                end
            end
        end
    end

    # Neighborhood graph
    neighbors_set = Set{Cint}()
    for (vi, sv) ∈ shared_vertices
        for (rank, vvi) ∈ sv.remote_vertices
            push!(neighbors_set, rank)
        end
    end
    # Adjust ranks back to to C index convention
    dest = collect(neighbors_set).-1
    degree = length(dest)
    interface_comm = MPI.Dist_graph_create(grid_comm, Cint[my_rank-1], Cint[degree], Cint.(dest))

    return DistributedGrid(grid_comm,interface_comm,local_grid,shared_vertices,shared_edges,shared_faces)
end

@inline getlocalgrid(dgrid::AbstractDistributedGrid) = dgrid.local_grid

@inline getnodes(dgrid::AbstractDistributedGrid) = getnodes(getlocalgrid(dgrid))
@inline getnodes(grid::AbstractDistributedGrid, v::Union{Int, Vector{Int}}) = getnodes(getlocalgrid(dgrid), v)
@inline getnodes(grid::AbstractDistributedGrid, setname::String) = getnodes(getlocalgrid(dgrid), setname)
@inline getnnodes(dgrid::AbstractDistributedGrid) = getnnodes(getlocalgrid(dgrid))

@inline getcells(dgrid::AbstractDistributedGrid) = getcells(getlocalgrid(dgrid))
@inline getcells(dgrid::AbstractDistributedGrid, v::Union{Int, Vector{Int}}) = getcells(getlocalgrid(dgrid),v)
@inline getcells(dgrid::AbstractDistributedGrid, setname::String) = getcells(getlocalgrid(dgrid),setname)
"Returns the number of cells in the `<:AbstractDistributedGrid`."
@inline getncells(dgrid::AbstractDistributedGrid) = getncells(getlocalgrid(dgrid))
"Returns the celltype of the `<:AbstractDistributedGrid`."
@inline getcelltype(dgrid::AbstractDistributedGrid) = eltype(getcells(getlocalgrid(dgrid)))
@inline getcelltype(dgrid::AbstractDistributedGrid, i::Int) = typeof(getcells(getlocalgrid(dgrid),i))

# Here we define the entity ownership by the process sharing an entity with lowest rank in the grid communicator.
function compute_owner(dgrid::AbstractDistributedGrid, shared_entity::SharedEntity)::Int32
    my_rank = MPI.Comm_rank(global_comm(dgrid))+1 # Shift rank up by 1 to match Julia's indexing convention
    return minimum([my_rank; [remote_rank for (remote_rank, _) ∈ remote_entities(shared_entity)]])
end

@inline getcellset(grid::AbstractDistributedGrid, setname::String) = getcellset(getlocalgrid(grid), setname)
@inline getcellsets(grid::AbstractDistributedGrid) = getcellsets(getlocalgrid(grid))

@inline getnodeset(grid::AbstractDistributedGrid, setname::String) = getnodeset(getlocalgrid(grid), setname)
@inline getnodesets(grid::AbstractDistributedGrid) = getnodeset(getlocalgrid(grid), setname)

@inline getfaceset(grid::AbstractDistributedGrid, setname::String) = getfaceset(getlocalgrid(grid), setname)
@inline getfacesets(grid::AbstractDistributedGrid) = getfaceset(getlocalgrid(grid), setname)

@inline getedgeset(grid::AbstractDistributedGrid, setname::String) = getedgeset(getlocalgrid(grid), setname)
@inline getedgesets(grid::AbstractDistributedGrid) = getedgeset(getlocalgrid(grid), setname)

@inline getvertexset(grid::AbstractDistributedGrid, setname::String) = getvertexset(getlocalgrid(grid), setname)
@inline getvertexsets(grid::AbstractDistributedGrid) = getvertexset(getlocalgrid(grid), setname)
