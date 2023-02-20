"""
"""
abstract type AbstractDistributedGrid{sdim} <: AbstractGrid{sdim} end

"""
"""
abstract type SharedEntity end

#TODO remove. This is a temporary workaround to make the transient extensions work.
global_comm(::AbstractDistributedGrid) = error("Not implemented.")
interface_comm(::AbstractDistributedGrid) = error("Not implemented.")
global_rank(::AbstractDistributedGrid) = error("Not implemented.")
compute_owner(::AbstractDistributedGrid) = error("Not implemented.")
remote_entities(::SharedEntity) = error("Not implemented.")

"""
"""
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
