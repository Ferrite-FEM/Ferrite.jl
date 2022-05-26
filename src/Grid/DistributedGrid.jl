using Metis
using MPI

"""
"""
abstract type AbstractDistributedGrid{sdim} <: AbstractGrid{sdim} end

struct SharedVertex
    local_idx::VertexIndex
    remote_vertices::Dict{Int,Vector{VertexIndex}}
end

struct SharedFace
    local_idx::FaceIndex
    remote_edges::Dict{Int,Vector{FaceIndex}}
end

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
    local_grid::Grid
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
function DistributedGrid(grid_to_distribute::Grid, grid_topology::ExclusiveTopology, grid_comm::MPI.Comm; partition_alg = :RECURSIVE)
    N = getncells(grid_to_distribute)
    @assert N > 0

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

    parts = Metis.partition(
        Metis.Graph(
            Metis.idx_t(N),
            xadj,
            adjncy
        ),
        MPI.Comm_size(grid_comm);
        alg=partition_alg
    )

    return Grid(
        getcells(grid_to_distribute)[[i for i ∈ 1:N if parts[i] == (MPI.Comm_rank(grid_comm)+1)]],
        grid_to_distribute.nodes
    )
end