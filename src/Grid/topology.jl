
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
Base.length(n::EntityNeighborhood) = length(n.neighbor_info)
Base.getindex(n::EntityNeighborhood,i) = getindex(n.neighbor_info,i)
Base.firstindex(n::EntityNeighborhood) = 1
Base.lastindex(n::EntityNeighborhood) = length(n.neighbor_info)
Base.:(==)(n1::EntityNeighborhood, n2::EntityNeighborhood) = n1.neighbor_info == n2.neighbor_info
Base.iterate(n::EntityNeighborhood, state=1) = iterate(n.neighbor_info,state)

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
    ExclusiveTopology(grid::Grid)

`ExclusiveTopology` saves topological (connectivity/neighborhood) data of the grid. The constructor works with an `AbstractCell`
vector for all cells that dispatch `vertices`, `faces` and in 3D `edges`.
The struct saves the highest dimensional neighborhood, i.e. if something is connected by a face and an
 edge only the face neighborhood is saved. The lower dimensional neighborhood is recomputed, if needed.

# Fields
- `vertex_to_cell::Vector{Set{Int}}`: global vertex id to all cells containing the vertex
- `cell_neighbor::Vector{EntityNeighborhood{CellIndex}}`: cellid to all connected cells
- `face_neighbor::Matrix{EntityNeighborhood,Int}`: `face_neighbor[cellid,local_face_id]` -> neighboring face
- `vertex_neighbor::Matrix{EntityNeighborhood,Int}`: `vertex_neighbor[cellid,local_vertex_id]` -> neighboring vertex
- `edge_neighbor::Matrix{EntityNeighborhood,Int}`: `edge_neighbor[cellid_local_vertex_id]` -> neighboring edge
- `face_skeleton::Union{Vector{FaceIndex}, Nothing}`: 

!!! note Currently mixed-dimensional queries do not work at the moment. They will be added back later.
"""
mutable struct ExclusiveTopology <: AbstractTopology
    # maps a global vertex id to all cells containing the vertex
    vertex_to_cell::Vector{Set{Int}}
    # index of the vector = cell id ->  all other connected cells
    cell_neighbor::Vector{EntityNeighborhood{CellIndex}}
    # face_neighbor[cellid,local_face_id] -> exclusive connected entities (not restricted to one entity)
    face_face_neighbor::Matrix{EntityNeighborhood{FaceIndex}}
    # vertex_neighbor[cellid,local_vertex_id] -> exclusive connected entities to the given vertex
    vertex_vertex_neighbor::Matrix{EntityNeighborhood{VertexIndex}}
    # edge_neighbor[cellid,local_edge_id] -> exclusive connected entities of the given edge
    edge_edge_neighbor::Matrix{EntityNeighborhood{EdgeIndex}}
    # lazy constructed face topology
    face_skeleton::Union{Vector{FaceIndex}, Nothing}
    # TODO reintroduce the codimensional connectivity, e.g. 3D edge to 2D face
end

function Base.show(io::IO, ::MIME"text/plain", topology::ExclusiveTopology)
    println(io, "ExclusiveTopology\n")
    print(io, "  Vertex neighbors: $(size(topology.vertex_vertex_neighbor))\n")
    print(io, "  Face neighbors: $(size(topology.face_face_neighbor))\n")
    println(io, "  Edge neighbors: $(size(topology.edge_edge_neighbor))")
end

function _num_shared_vertices(cell_a::C1, cell_b::C2) where {C1, C2}
    num_shared_vertices = 0
    for vertex ∈ vertices(cell_a)
        for vertex_neighbor ∈ vertices(cell_b)
            if vertex_neighbor == vertex
                num_shared_vertices += 1
                continue
            end
        end
    end
    return num_shared_vertices
end

function _exclusive_topology_ctor(cells::Vector{C}, vertex_cell_table::Array{Set{Int}}, vertex_table, face_table, edge_table, cell_neighbor_table) where C <: AbstractCell
    for (cell_id, cell) in enumerate(cells)
        # Gather all cells which are connected via vertices
        cell_neighbor_ids = Set{Int}()
        for vertex ∈ vertices(cell)
            for vertex_cell_id ∈ vertex_cell_table[vertex]
                if vertex_cell_id != cell_id
                    push!(cell_neighbor_ids, vertex_cell_id)
                end
            end
        end
        cell_neighbor_table[cell_id] = EntityNeighborhood(CellIndex.(collect(cell_neighbor_ids)))

        # Any of the neighbors is now sorted in the respective categories
        for cell_neighbor_id ∈ cell_neighbor_ids
            # Buffer neighbor
            cell_neighbor = cells[cell_neighbor_id]
            # TODO handle mixed-dimensional case
            getdim(cell_neighbor) == getdim(cell) || continue

            num_shared_vertices = _num_shared_vertices(cell, cell_neighbor)

            # Simplest case: Only one vertex is shared => Vertex neighbor
            if num_shared_vertices == 1
                for (lvi, vertex) ∈ enumerate(vertices(cell))
                    for (lvi2, vertex_neighbor) ∈ enumerate(vertices(cell_neighbor))
                        if vertex_neighbor == vertex
                            push!(vertex_table[cell_id, lvi].neighbor_info, VertexIndex(cell_neighbor_id, lvi2))
                            break
                        end
                    end
                end
            # Shared path
            elseif num_shared_vertices == 2
                if getdim(cell) == 2
                    _add_single_face_neighbor!(face_table, cell, cell_id, cell_neighbor, cell_neighbor_id)
                elseif getdim(cell) == 3
                    _add_single_edge_neighbor!(edge_table, cell, cell_id, cell_neighbor, cell_neighbor_id)
                else
                    @error "Case not implemented."
                end
            # Shared surface
            elseif num_shared_vertices >= 3
                _add_single_face_neighbor!(face_table, cell, cell_id, cell_neighbor, cell_neighbor_id)
            else
                @error "Found connected elements without shared vertex... Mesh broken?"
            end
        end
    end
end

function ExclusiveTopology(cells::Vector{C}) where C <: AbstractCell
    # Setup the cell to vertex table
    cell_vertices_table = vertices.(cells) #needs generic interface for <: AbstractCell
    vertex_cell_table = Set{Int}[Set{Int}() for _ ∈ 1:maximum(maximum.(cell_vertices_table))]

    # Setup vertex to cell connectivity by flipping the cell to vertex table
    for (cellid, cell_vertices) in enumerate(cell_vertices_table)
        for vertex in cell_vertices
            push!(vertex_cell_table[vertex], cellid)
        end
    end

    # Compute correct matrix size
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
        celltypes = Set(typeof.(cells))
        for celltype in celltypes
            celltypeidx = findfirst(x->typeof(x)==celltype,cells)
            dim = getdim(cells[celltypeidx])

            max_vertices = max(max_vertices,nvertices(cells[celltypeidx]))
            dim > 1 && (max_faces = max(max_faces, nfaces(cells[celltypeidx])))
            dim > 2 && (max_edges = max(max_edges, nedges(cells[celltypeidx])))
        end
    end

    # Setup matrices
    vertex_table = Matrix{EntityNeighborhood{VertexIndex}}(undef, length(cells), max_vertices)
    for j = 1:size(vertex_table,2)
        for i = 1:size(vertex_table,1)
            vertex_table[i,j] = EntityNeighborhood{VertexIndex}(VertexIndex[])
        end
    end
    face_table   = Matrix{EntityNeighborhood{FaceIndex}}(undef, length(cells), max_faces)
    for j = 1:size(face_table,2)
        for i = 1:size(face_table,1)
            face_table[i,j] = EntityNeighborhood{FaceIndex}(FaceIndex[])
        end
    end
    edge_table   = Matrix{EntityNeighborhood{EdgeIndex}}(undef, length(cells), max_edges)
    for j = 1:size(edge_table,2)
        for i = 1:size(edge_table,1)
            edge_table[i,j] = EntityNeighborhood{EdgeIndex}(EdgeIndex[])
        end
    end
    cell_neighbor_table = Vector{EntityNeighborhood{CellIndex}}(undef, length(cells))

    _exclusive_topology_ctor(cells, vertex_cell_table, vertex_table, face_table, edge_table, cell_neighbor_table)

    return ExclusiveTopology(vertex_cell_table,cell_neighbor_table,face_table,vertex_table,edge_table,nothing)
end

function _add_single_face_neighbor!(face_table, cell::C1, cell_id, cell_neighbor::C2, cell_neighbor_id) where {C1, C2}
    for (lfi, face) ∈ enumerate(faces(cell))
        uniqueface = sortface_fast(face)
        for (lfi2, face_neighbor) ∈ enumerate(faces(cell_neighbor))
            uniqueface2 = sortface_fast(face_neighbor)
            if uniqueface == uniqueface2
                push!(face_table[cell_id, lfi].neighbor_info, FaceIndex(cell_neighbor_id, lfi2))
                return
            end
        end
    end
end

function _add_single_edge_neighbor!(edge_table, cell::C1, cell_id, cell_neighbor::C2, cell_neighbor_id) where {C1, C2}
    for (lei, edge) ∈ enumerate(edges(cell))
        uniqueedge = sortedge_fast(edge)
        for (lei2, edge_neighbor) ∈ enumerate(edges(cell_neighbor))
            uniqueedge2 = sortedge_fast(edge_neighbor)
            if uniqueedge == uniqueedge2
                push!(edge_table[cell_id, lei].neighbor_info, EdgeIndex(cell_neighbor_id, lei2))
                return
            end
        end
    end
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
    vertex_star_stencils(top::ExclusiveTopology, grid::Grid) -> Vector{Int, EntityNeighborhood{VertexIndex}}()
Computes the stencils induced by the edge connectivity of the vertices.
"""
function vertex_star_stencils(top::ExclusiveTopology, grid::Grid)
    cells = grid.cells
    stencil_table = Dict{Int,EntityNeighborhood{VertexIndex}}()
    # Vertex Connectivity
    for (global_vertexid,cellset) ∈ enumerate(top.vertex_to_cell)
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
        stencil_table[global_vertexid] =  EntityNeighborhood(vertex_neighbors_local)
    end
    return stencil_table
end

"""
    getstencil(top::Dict{Int, EntityNeighborhood{VertexIndex}}, grid::AbstractGrid, vertex_idx::VertexIndex) -> EntityNeighborhood{VertexIndex}
Get an iterateable over the stencil members for a given local entity.
"""
function getstencil(top::Dict{Int, EntityNeighborhood{VertexIndex}}, grid::Grid, vertex_idx::VertexIndex)
    return top[toglobal(grid, vertex_idx)].neighbor_info
end

"""
    _faceskeleton(topology::ExclusiveTopology, grid::Grid) -> Iterable{FaceIndex}
Creates an iterateable face skeleton. The skeleton consists of `FaceIndex` that can be used to `reinit`
`FaceValues`.
"""
function _faceskeleton(top::ExclusiveTopology, grid::Grid)
    face_skeleton_global = Set{NTuple}()
    face_skeleton_local = Vector{FaceIndex}()
    fs_length = length(face_skeleton_global)
    # TODO use topology to speed up :)
    for (cellid,cell) ∈ enumerate(grid.cells)
        for (local_face_id,face) ∈ enumerate(faces(cell))
            push!(face_skeleton_global, sortface_fast(face))
            fs_length_new = length(face_skeleton_global)
            if fs_length != fs_length_new
                push!(face_skeleton_local, FaceIndex(cellid,local_face_id))
                fs_length = fs_length_new
            end
        end
    end
    return face_skeleton_local
end

"""
    face_skeleton(top::ExclusiveTopology, grid::Grid) -> Vector{FaceIndex}
Creates an iterateable face skeleton. The skeleton consists of `FaceIndex` that can be used to `reinit`
`FaceValues`.
"""
function faceskeleton(top::ExclusiveTopology, grid::Grid)
    if top.face_skeleton === nothing
        top.face_skeleton = _faceskeleton(top, grid)
    end
    return top.face_skeleton
end
