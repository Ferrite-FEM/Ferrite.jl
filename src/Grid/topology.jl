
############
# Topology #
############

"""
    getneighborhood(topology, grid::AbstractGrid, cellidx::CellIndex, include_self=false)
    getneighborhood(topology, grid::AbstractGrid, faceidx::FaceIndex, include_self=false)
    getneighborhood(topology, grid::AbstractGrid, vertexidx::VertexIndex, include_self=false)
    getneighborhood(topology, grid::AbstractGrid, edgeidx::EdgeIndex, include_self=false)

Returns all connected entities of the same type as defined by the respective topology. If `include_self` is true,
the given entity is included in the returned list as well.
"""
getneighborhood

struct EntityNeighborhood{T<:Union{BoundaryIndex,CellIndex}}
    neighbor_info::Vector{T}
end

EntityNeighborhood(info::T) where T <: BoundaryIndex = EntityNeighborhood([info])
Base.zero(::Type{EntityNeighborhood{T}}) where T = EntityNeighborhood(T[])
Base.zero(::Type{EntityNeighborhood}) = EntityNeighborhood(BoundaryIndex[])
Base.length(n::EntityNeighborhood) = length(n.neighbor_info)
Base.getindex(n::EntityNeighborhood,i) = getindex(n.neighbor_info,i)
Base.firstindex(n::EntityNeighborhood) = 1
Base.lastindex(n::EntityNeighborhood) = length(n.neighbor_info)
Base.:(==)(n1::EntityNeighborhood, n2::EntityNeighborhood) = n1.neighbor_info == n2.neighbor_info
Base.iterate(n::EntityNeighborhood, state=1) = iterate(n.neighbor_info,state)

function Base.:+(n1::EntityNeighborhood{T1}, n2::EntityNeighborhood{T2}) where {T1 <:Union{BoundaryIndex,CellIndex}, T2 <:Union{BoundaryIndex,CellIndex}}
    neighbor_info = [n1.neighbor_info; n2.neighbor_info]
    return EntityNeighborhood(neighbor_info)
end

function Base.show(io::IO, ::MIME"text/plain", n::EntityNeighborhood)
    if length(n) == 0
        println(io, "No EntityNeighborhood")
    elseif length(n) == 1
        println(io, "$(n.neighbor_info[1])")
    else
        println(io, "$(n.neighbor_info...)")
    end
end

abstract type AbstractTopology end

"""
    ExclusiveTopology(cells::Vector{C}) where C <: AbstractCell
`ExclusiveTopology` saves topological (connectivity/neighborhood) data of the grid. The constructor works with an `AbstractCell`
vector for all cells that dispatch `vertices`, `faces` and in 3D `edges`.
The struct saves the highest dimensional neighborhood, i.e. if something is connected by a face and an
 edge only the face neighborhood is saved. The lower dimensional neighborhood is recomputed, if needed.

# Fields
- `vertex_to_cell::Dict{Int,Vector{Int}}`: global vertex id to all cells containing the vertex
- `cell_neighbor::Vector{EntityNeighborhood{CellIndex}}`: cellid to all connected cells
- `face_neighbor::SparseMatrixCSC{EntityNeighborhood,Int}`: `face_neighbor[cellid,local_face_id]` -> neighboring face
- `vertex_neighbor::SparseMatrixCSC{EntityNeighborhood,Int}`: `vertex_neighbor[cellid,local_vertex_id]` -> neighboring vertex
- `edge_neighbor::SparseMatrixCSC{EntityNeighborhood,Int}`: `edge_neighbor[cellid_local_vertex_id]` -> neighboring edge
"""
struct ExclusiveTopology <: AbstractTopology
    # maps a global vertex id to all cells containing the vertex
    vertex_to_cell::Dict{Int,Set{Int}}
    # index of the vector = cell id ->  all other connected cells
    cell_neighbor::Vector{EntityNeighborhood{CellIndex}}
    # face_neighbor[cellid,local_face_id] -> exclusive connected entities (not restricted to one entity)
    face_face_neighbor::SparseMatrixCSC{EntityNeighborhood{FaceIndex},Int}
    # vertex_neighbor[cellid,local_vertex_id] -> exclusive connected entities to the given vertex
    vertex_vertex_neighbor::SparseMatrixCSC{EntityNeighborhood{VertexIndex},Int}
    # edge_neighbor[cellid,local_edge_id] -> exclusive connected entities of the given edge
    edge_edge_neighbor::SparseMatrixCSC{EntityNeighborhood{EdgeIndex},Int}
    # TODO reintroduce the codimensional connectivity, e.g. 3D edge to 2D face
end

function ExclusiveTopology(cells::Vector{C}) where C <: AbstractCell
    cell_vertices_table = vertices.(cells) #needs generic interface for <: AbstractCell
    vertex_cell_table = Dict{Int,Set{Int}}()

    for (cellid, cell_vertices) in enumerate(cell_vertices_table)
       for vertex in cell_vertices
            if haskey(vertex_cell_table, vertex)
                push!(vertex_cell_table[vertex], cellid)
            else
                vertex_cell_table[vertex] = Set([cellid])
            end
        end
    end

    I_face = Int[]; J_face = Int[]; V_face = EntityNeighborhood{FaceIndex}[]
    I_edge = Int[]; J_edge = Int[]; V_edge = EntityNeighborhood{EdgeIndex}[]
    I_vertex = Int[]; J_vertex = Int[]; V_vertex = EntityNeighborhood{VertexIndex}[]
    cell_neighbor_table = Vector{EntityNeighborhood{CellIndex}}(undef, length(cells))

    for (cellid, cell) in enumerate(cells)
        cell_neighbors = reduce(union!, [Set{Int}(vertex_cell_table[vertex]) for vertex ∈ vertices(cell) if vertex_cell_table[vertex] != cellid])
        cell_neighbor_table[cellid] = EntityNeighborhood(CellIndex.(collect(cell_neighbors)))

        # Add all element connected via face to the table
        face_neighbors = Set{Int}()
        for (face_idx,face) ∈ enumerate(faces(cell))
            neighbor_candidates = Set{Int}(c for c ∈ vertex_cell_table[face[1]] if c != cellid)
            for face_vertex ∈ face[2:end]
                intersect!(neighbor_candidates, vertex_cell_table[face_vertex])
            end
            union!(face_neighbors, neighbor_candidates)
        end

        # Add all element connected via edge to the table
        if getdim(cell) > 2
            edge_neighbors = Set{Int}()
            for (edge_idx,edge) ∈ enumerate(edges(cell))
                neighbor_candidates = Set{Int}(c for c ∈ vertex_cell_table[edge[1]] if c != cellid)
                for edge_vertex ∈ edge[2:end]
                    edge_neighbor = vertex_cell_table[edge_vertex]
                    if edge_neighbor != cellid && edge_neighbor ∉ face_neighbors
                        intersect!(neighbor_candidates, edge_neighbor)
                    end
                end
                union!(edge_neighbors, neighbor_candidates)
            end
        end

        # Fill subentity connectivity tables
        for neighbor_cellid in cell_neighbors
            getdim(cell) == getdim(cells[neighbor_cellid]) || continue # TODO introduce logic to handle codimensional connectivity

            cell_local_ids = findall(x->x in cell_vertices_table[neighbor_cellid], cell_vertices_table[cellid])::Vector{Int}
            # vertex neighbor
            if length(cell_local_ids) == 1
                neighbor_local_ids = findall(x->x in cell_vertices_table[cellid], cell_vertices_table[neighbor_cellid])::Vector{Int}
                _vertex_neighbor!(V_vertex, I_vertex, J_vertex, cellid, cell, neighbor_local_ids, neighbor_cellid, cells[neighbor_cellid])
            # face neighbor
            elseif neighbor_cellid ∈ face_neighbors
                neighbor_local_ids = findall(x->x in cell_vertices_table[cellid], cell_vertices_table[neighbor_cellid])::Vector{Int}
                _face_neighbor!(V_face, I_face, J_face, cellid, cell, neighbor_local_ids, neighbor_cellid, cells[neighbor_cellid])
            # edge neighbor
            elseif getdim(cell) > 2 && neighbor_cellid ∈ edge_neighbors
                neighbor_local_ids = findall(x->x in cell_vertices_table[cellid], cell_vertices_table[neighbor_cellid])::Vector{Int}
                _edge_neighbor!(V_edge, I_edge, J_edge, cellid, cell, neighbor_local_ids, neighbor_cellid, cells[neighbor_cellid])
            end
        end
    end

    # Compute correct sparse matrix size
    celltype = eltype(cells)
    max_vertices = 0
    max_faces = 0
    max_edges = 0
    if isconcretetype(celltype)
        dim = getdim(cells[1])

        max_vertices = nvertices(cells[1])
        dim > 1 && (max_faces = nfaces(cells[1]))
        dim > 2 && (max_edges = nedges(cells[1]))
    else
        celltypes = typeof.(cells)
        for celltype in celltypes
            celltypeidx = findfirst(x->typeof(x)==celltype,cells)
            dim = getdim(cells[celltypeidx])

            max_vertices = max(max_vertices,nvertices(cells[celltypeidx]))
            dim > 1 && (max_faces = max(max_faces, nfaces(cells[celltypeidx])))
            dim > 2 && (max_edges = max(max_edges, nedges(cells[celltypeidx])))
        end
    end

    vertex_neighbor = sparse(I_vertex,J_vertex,V_vertex, length(cells), max_vertices)
    face_neighbor = sparse(I_face,J_face,V_face, length(cells), max_faces)
    edge_neighbor = sparse(I_edge,J_edge,V_edge, length(cells), max_edges)

    return ExclusiveTopology(vertex_cell_table,cell_neighbor_table,face_neighbor,vertex_neighbor,edge_neighbor)
end

function _vertex_neighbor!(V_vertex::Vector{EntityNeighborhood{VertexIndex}}, I_vertex::Array{Int}, J_vertex::Array{Int}, cellid::Int, cell::C1, neighbor::Vector{Int}, neighborid::Int, neighbor_cell::C2) where {C1 <: AbstractCell, C2 <: AbstractCell}
    vertex_neighbor = VertexIndex((neighborid, neighbor[1]))
    cell_vertex_id = findfirst(x->x==neighbor_cell.nodes[neighbor[1]], cell.nodes)::Int
    push!(V_vertex,EntityNeighborhood(vertex_neighbor))
    push!(I_vertex,cellid)
    push!(J_vertex,cell_vertex_id)
end

function _edge_neighbor!(V_edge::Vector{EntityNeighborhood{EdgeIndex}}, I_edge::Array{Int}, J_edge::Array{Int}, cellid::Int, cell::C1, neighbor::Vector{Int}, neighborid::Int, neighbor_cell::C2) where {C1 <: AbstractCell, C2 <: AbstractCell}
    neighbor_edge = neighbor_cell.nodes[neighbor]
    neighbor_edge_id = findfirst(x->issubset(x,neighbor_edge), edges(neighbor_cell))::Int
    edge_neighbor = EdgeIndex((neighborid, neighbor_edge_id))
    cell_edge_id = findfirst(x->issubset(x,neighbor_edge),edges(cell))::Int
    push!(V_edge, EntityNeighborhood(edge_neighbor))
    push!(I_edge, cellid)
    push!(J_edge, cell_edge_id)
end

function _face_neighbor!(V_face::Vector{EntityNeighborhood{FaceIndex}}, I_face::Array{Int}, J_face::Array{Int}, cellid::Int, cell::C1, neighbor::Vector{Int}, neighborid::Int, neighbor_cell::C2) where {C1 <: AbstractCell, C2 <: AbstractCell}
    neighbor_face = neighbor_cell.nodes[neighbor]
    neighbor_face_id = findfirst(x->issubset(x,neighbor_face), faces(neighbor_cell))::Int
    face_neighbor = FaceIndex((neighborid, neighbor_face_id))
    cell_face_id = findfirst(x->issubset(x,neighbor_face),faces(cell))::Int
    push!(V_face, EntityNeighborhood(face_neighbor))
    push!(I_face, cellid)
    push!(J_face, cell_face_id)
end

getcells(neighbor::EntityNeighborhood{T}) where T <: BoundaryIndex = first.(neighbor.neighbor_info)
getcells(neighbor::EntityNeighborhood{CellIndex}) = getproperty.(neighbor.neighbor_info, :idx)
getcells(neighbors::Vector{T}) where T <: EntityNeighborhood = reduce(vcat, getcells.(neighbors))
getcells(neighbors::Vector{T}) where T <: BoundaryIndex = getindex.(neighbors,1)

ExclusiveTopology(grid::AbstractGrid) = ExclusiveTopology(getcells(grid))

function getneighborhood(top::ExclusiveTopology, grid::AbstractGrid, cellidx::CellIndex, include_self=false)
    patch = getcells(top.cell_neighbor[cellidx.idx])
    if include_self
        return [patch; cellidx.idx]
    else
        return patch
    end
end

function getneighborhood(top::ExclusiveTopology, grid::AbstractGrid, faceidx::FaceIndex, include_self=false)
    if include_self
        return [top.face_face_neighbor[faceidx[1],faceidx[2]].neighbor_info; faceidx]
    else
        return top.face_face_neighbor[faceidx[1],faceidx[2]].neighbor_info
    end
end

function getneighborhood(top::ExclusiveTopology, grid::AbstractGrid, vertexidx::VertexIndex, include_self=false)
    cellid, local_vertexid = vertexidx[1], vertexidx[2]
    cell_vertices = vertices(getcells(grid,cellid))
    global_vertexid = cell_vertices[local_vertexid]
    if include_self
        vertex_to_cell = top.vertex_to_cell[global_vertexid]
        self_reference_local = Vector{VertexIndex}(undef,length(vertex_to_cell))
        for (i,cellid) in enumerate(vertex_to_cell)
            local_vertex = VertexIndex(cellid,findfirst(x->x==global_vertexid,vertices(getcells(grid,cellid)))::Int)
            self_reference_local[i] = local_vertex
        end
        return [top.vertex_vertex_neighbor[global_vertexid].neighbor_info; self_reference_local]
    else
        return top.vertex_vertex_neighbor[global_vertexid].neighbor_info
    end
end

function getneighborhood(top::ExclusiveTopology, grid::AbstractGrid{3}, edgeidx::EdgeIndex, include_self=false)
    cellid, local_edgeidx = edgeidx[1], edgeidx[2]
    cell_edges = edges(getcells(grid,cellid))
    nonlocal_edgeid = cell_edges[local_edgeidx]
    cell_neighbors = getneighborhood(top,grid,CellIndex(cellid))
    self_reference_local = EdgeIndex[]
    for cellid in cell_neighbors
        local_neighbor_edgeid = findfirst(x->issubset(x,nonlocal_edgeid),edges(getcells(grid,cellid)))::Int
        local_neighbor_edgeid === nothing && continue
        local_edge = EdgeIndex(cellid,local_neighbor_edgeid)
        push!(self_reference_local, local_edge)
    end
    if include_self
        return unique([top.edge_edge_neighbor[cellid, local_edgeidx].neighbor_info; self_reference_local; edgeidx])
    else
        return unique([top.edge_edge_neighbor[cellid, local_edgeidx].neighbor_info; self_reference_local])
    end
end

"""
    compute_vertex_star_stencils(top, grid) -> StencilType
Computes the stencils induced by the edge connectivity of the vertices.
"""
function compute_vertex_star_stencils(top::ExclusiveTopology, grid::Grid)
    cells = grid.cells
    vertex_vertex_table = Dict{Int,EntityNeighborhood{VertexIndex}}()
    # Vertex Connectivity
    for (global_vertexid,cellset) ∈ top.vertex_to_cell
        vertex_neighbors_local = VertexIndex[]
        for cell ∈ cellset
            neighbor_boundary = getdim(cells[cell]) > 2 ? collect(edges(cells[cell])) : collect(faces(cells[cell])) #get lowest dimension boundary
            neighbor_connected_faces = neighbor_boundary[findall(x->global_vertexid ∈ x, neighbor_boundary)]
            this_local_vertex = findfirst(i->toglobal(grid, VertexIndex(cell, i)) == global_vertexid, 1:nvertices(cells[cell]))
            push!(vertex_neighbors_local, VertexIndex(cell, this_local_vertex))
            other_vertices = findfirst.(x->x!=global_vertexid,neighbor_connected_faces)
            any(other_vertices .=== nothing) && continue
            neighbor_vertices_global = getindex.(neighbor_connected_faces, other_vertices)
            neighbor_vertices_local = [VertexIndex(cell,local_vertex) for local_vertex ∈ findall(x->x ∈ neighbor_vertices_global, vertices(cells[cell]))]
            append!(vertex_neighbors_local, neighbor_vertices_local)
        end
        vertex_vertex_table[global_vertexid] =  EntityNeighborhood(vertex_neighbors_local)
    end
    return vertex_vertex_table
end

"""
    getstencil(top::StencilType, grid::AbstractGrid, vertex_idx::VertexIndex) -> Iterable{EntityNeighborhood}
Get an iterateable over the stencil members for a given local entity.
"""
function getstencil(top::Dict{Int,EntityNeighborhood{VertexIndex}}, grid::AbstractGrid, vertex_idx::VertexIndex)
    return top[toglobal(grid, vertex_idx)].neighbor_info
end

"""
    compute_face_skeleton(topology, grid) -> Iterable{FaceIndex}
Creates an iterateable face skeleton. The skeleton consists of `FaceIndex` that can be used to `reinit`
`FaceValues`.
"""
function compute_face_skeleton(top::ExclusiveTopology, grid::Grid)
    face_skeleton_global = Set{NTuple}()
    face_skeleton_local = Vector{FaceIndex}()
    fs_length = length(face_skeleton_global)
    # TODO use topology to speed up :)
    for (cellid,cell) ∈ enumerate(grid.cells)
        for (local_face_id,face) ∈ enumerate(faces(cell))
            push!(face_skeleton_global, first(sortface(face)))
            fs_length_new = length(face_skeleton_global)
            if fs_length != fs_length_new
                push!(face_skeleton_local, FaceIndex(cellid,local_face_id))
                fs_length = fs_length_new
            end
        end
    end
    return face_skeleton_local
end
