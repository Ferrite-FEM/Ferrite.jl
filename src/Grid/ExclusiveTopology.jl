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

function Base.:+(n1::EntityNeighborhood, n2::EntityNeighborhood)
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

getcells(neighbor::EntityNeighborhood{T}) where T <: BoundaryIndex = first.(neighbor.neighbor_info)
getcells(neighbor::EntityNeighborhood{CellIndex}) = getproperty.(neighbor.neighbor_info, :idx)
getcells(neighbors::Vector{T}) where T <: EntityNeighborhood = reduce(vcat, getcells.(neighbors))
getcells(neighbors::Vector{T}) where T <: BoundaryIndex = getindex.(neighbors,1)

"""
    ExclusiveTopology(cells::Vector{C}) where C <: AbstractCell
`ExclusiveTopology` saves topological (connectivity) data of the grid. The constructor works with an `AbstractCell`
vector for all cells that dispatch `vertices`, `faces` and in 3D `edges` as well as the utility functions
`face_npoints` and `edge_npoints`.
The struct saves the highest dimensional neighborhood, i.e. if something is connected by a face and an
 edge only the face neighborhood is saved. The lower dimensional neighborhood is recomputed, if needed.

# Fields
- `vertex_to_cell::Dict{Int,Vector{Int}}`: global vertex id to all cells containing the vertex
- `cell_neighbor::Vector{EntityNeighborhood{CellIndex}}`: cellid to all connected cells
- `face_neighbor::SparseMatrixCSC{EntityNeighborhood,Int}`: `face_neighbor[cellid,local_face_id]` -> neighboring face
- `vertex_neighbor::SparseMatrixCSC{EntityNeighborhood,Int}`: `vertex_neighbor[cellid,local_vertex_id]` -> neighboring vertex
- `edge_neighbor::SparseMatrixCSC{EntityNeighborhood,Int}`: `edge_neighbor[cellid_local_vertex_id]` -> neighboring edge
- `vertex_vertex_neighbor::Dict{Int,EntityNeighborhood{VertexIndex}}`: global vertex id -> all connected vertices by edge or face
- `face_skeleton::Vector{FaceIndex}`: list of unique faces in the grid 
"""
struct ExclusiveTopology <: AbstractTopology
    # maps a global vertex id to all cells containing the vertex
    vertex_to_cell::Dict{Int,Vector{Int}}
    # index of the vector = cell id ->  all other connected cells
    cell_neighbor::Vector{EntityNeighborhood{CellIndex}}
    # face_neighbor[cellid,local_face_id] -> exclusive connected entities (not restricted to one entity)
    face_neighbor::SparseMatrixCSC{EntityNeighborhood,Int}
    # vertex_neighbor[cellid,local_vertex_id] -> exclusive connected entities to the given vertex
    vertex_neighbor::SparseMatrixCSC{EntityNeighborhood,Int}
    # edge_neighbor[cellid,local_edge_id] -> exclusive connected entities of the given edge
    edge_neighbor::SparseMatrixCSC{EntityNeighborhood,Int}
    # maps global vertex id to all directly (by edge or face) connected vertices (no diagonal connection considered)
    vertex_vertex_neighbor::Dict{Int,EntityNeighborhood{VertexIndex}}
    # list of unique faces in the grid given as FaceIndex
    face_skeleton::Vector{FaceIndex}
end

function ExclusiveTopology(cells::Vector{C}) where C <: AbstractCell
    cell_vertices_table = vertices.(cells) #needs generic interface for <: AbstractCell
    vertex_cell_table = Dict{Int,Vector{Int}}() 
    
    for (cellid, cell_nodes) in enumerate(cell_vertices_table)
       for node in cell_nodes
            if haskey(vertex_cell_table, node)
                push!(vertex_cell_table[node], cellid)
            else
                vertex_cell_table[node] = [cellid]
            end
        end 
    end

    I_face = Int[]; J_face = Int[]; V_face = EntityNeighborhood[]
    I_edge = Int[]; J_edge = Int[]; V_edge = EntityNeighborhood[]
    I_vertex = Int[]; J_vertex = Int[]; V_vertex = EntityNeighborhood[]   
    cell_neighbor_table = Vector{EntityNeighborhood{CellIndex}}(undef, length(cells)) 

    for (cellid, cell) in enumerate(cells)    
        #cell neighborhood
        cell_neighbors = getindex.((vertex_cell_table,), cell_vertices_table[cellid]) # cell -> vertex -> cell
        cell_neighbors = unique(reduce(vcat,cell_neighbors)) # non unique list initially 
        filter!(x->x!=cellid, cell_neighbors) # get rid of self neighborhood
        cell_neighbor_table[cellid] = EntityNeighborhood(CellIndex.(cell_neighbors)) 

        for neighbor in cell_neighbors
            neighbor_local_ids = findall(x->x in cell_vertices_table[cellid], cell_vertices_table[neighbor])
            cell_local_ids = findall(x->x in cell_vertices_table[neighbor], cell_vertices_table[cellid])
            # vertex neighbor
            if length(cell_local_ids) == 1
                _vertex_neighbor!(V_vertex, I_vertex, J_vertex, cellid, cell, neighbor_local_ids, neighbor, cells[neighbor])
            # face neighbor
            elseif length(cell_local_ids) == face_npoints(cell)
                _face_neighbor!(V_face, I_face, J_face, cellid, cell, neighbor_local_ids, neighbor, cells[neighbor]) 
            # edge neighbor
            elseif getdim(cell) > 2 && length(cell_local_ids) == edge_npoints(cell)
                _edge_neighbor!(V_edge, I_edge, J_edge, cellid, cell, neighbor_local_ids, neighbor, cells[neighbor])
            end
        end       
    end
    
    celltype = eltype(cells)
    if isconcretetype(celltype)
        dim = getdim(cells[1])
        _nvertices = nvertices(cells[1])
        push!(V_vertex,zero(EntityNeighborhood{VertexIndex}))
        push!(I_vertex,1); push!(J_vertex,_nvertices)
        if dim > 1
            _nfaces = nfaces(cells[1])
            push!(V_face,zero(EntityNeighborhood{FaceIndex}))
            push!(I_face,1); push!(J_face,_nfaces)
        end
        if dim > 2
            _nedges = nedges(cells[1])
            push!(V_edge,zero(EntityNeighborhood{EdgeIndex}))
            push!(I_edge,1); push!(J_edge,_nedges)
        end
    else
        celltypes = typeof.(cells) 
        for celltype in celltypes
            celltypeidx = findfirst(x->typeof(x)==celltype,cells)
            dim = getdim(cells[celltypeidx])
            _nvertices = nvertices(cells[celltypeidx])
            push!(V_vertex,zero(EntityNeighborhood{VertexIndex}))
            push!(I_vertex,celltypeidx); push!(J_vertex,_nvertices)
            if dim > 1
                _nfaces = nfaces(cells[celltypeidx])
                push!(V_face,zero(EntityNeighborhood{FaceIndex}))
                push!(I_face,celltypeidx); push!(J_face,_nfaces)
            end
            if dim > 2
                _nedges = nedges(cells[celltypeidx])
                push!(V_edge,zero(EntityNeighborhood{EdgeIndex}))
                push!(I_edge,celltypeidx); push!(J_edge,_nedges)
            end
        end
    end
    face_neighbor = sparse(I_face,J_face,V_face)
    vertex_neighbor = sparse(I_vertex,J_vertex,V_vertex) 
    edge_neighbor = sparse(I_edge,J_edge,V_edge)

    vertex_vertex_table = Dict{Int,EntityNeighborhood}()
    vertex_vertex_global = Dict{Int,Vector{Int}}()
    # Vertex Connectivity
    for global_vertexid in keys(vertex_cell_table)
        #Cellset that contains given vertex 
        cellset = vertex_cell_table[global_vertexid]
        vertex_neighbors_local = VertexIndex[]
        vertex_neighbors_global = Int[]
        for cell in cellset
            neighbor_boundary = getdim(cells[cell]) == 2 ? [faces(cells[cell])...] : [edges(cells[cell])...] #get lowest dimension boundary
            neighbor_connected_faces = neighbor_boundary[findall(x->global_vertexid in x, neighbor_boundary)]
            neighbor_vertices_global = getindex.(neighbor_connected_faces, findfirst.(x->x!=global_vertexid,neighbor_connected_faces))
            neighbor_vertices_local= [VertexIndex(cell,local_vertex) for local_vertex in findall(x->x in neighbor_vertices_global, vertices(cells[cell]))]
            append!(vertex_neighbors_local, neighbor_vertices_local)
            append!(vertex_neighbors_global, neighbor_vertices_global)
        end
        vertex_vertex_table[global_vertexid] =  EntityNeighborhood(vertex_neighbors_local)
        vertex_vertex_global[global_vertexid] = vertex_neighbors_global
    end 

    # Face Skeleton
    face_skeleton_global = Set{NTuple}()
    face_skeleton_local = Vector{FaceIndex}()
    fs_length = length(face_skeleton_global)
    for (cellid,cell) in enumerate(cells)
        for (local_face_id,face) in enumerate(faces(cell))
            push!(face_skeleton_global, sortface(face))
            fs_length_new = length(face_skeleton_global)
            if fs_length != fs_length_new
                push!(face_skeleton_local, FaceIndex(cellid,local_face_id)) 
                fs_length = fs_length_new
            end
        end
    end
    return ExclusiveTopology(vertex_cell_table,cell_neighbor_table,face_neighbor,vertex_neighbor,edge_neighbor,vertex_vertex_table,face_skeleton_local)
end

ExclusiveTopology(grid::AbstractGrid) = ExclusiveTopology(getcells(grid))

function _vertex_neighbor!(V_vertex, I_vertex, J_vertex, cellid, cell, neighbor, neighborid, neighbor_cell)
    vertex_neighbor = VertexIndex((neighborid, neighbor[1]))
    cell_vertex_id = findfirst(x->x==neighbor_cell.nodes[neighbor[1]], cell.nodes)
    push!(V_vertex,EntityNeighborhood(vertex_neighbor))
    push!(I_vertex,cellid)
    push!(J_vertex,cell_vertex_id)
end

function _edge_neighbor!(V_edge, I_edge, J_edge, cellid, cell, neighbor, neighborid, neighbor_cell)
    neighbor_edge = neighbor_cell.nodes[neighbor]
    if getdim(neighbor_cell) < 3
        neighbor_edge_id = findfirst(x->issubset(x,neighbor_edge), faces(neighbor_cell))
        edge_neighbor = FaceIndex((neighborid, neighbor_edge_id))
    else
        neighbor_edge_id = findfirst(x->issubset(x,neighbor_edge), edges(neighbor_cell))
        edge_neighbor = EdgeIndex((neighborid, neighbor_edge_id))
    end
    cell_edge_id = findfirst(x->issubset(x,neighbor_edge),edges(cell))
    push!(V_edge, EntityNeighborhood(edge_neighbor))
    push!(I_edge, cellid)
    push!(J_edge, cell_edge_id)
end

function _face_neighbor!(V_face, I_face, J_face, cellid, cell, neighbor, neighborid, neighbor_cell)
    neighbor_face = neighbor_cell.nodes[neighbor]
    if getdim(neighbor_cell) == getdim(cell)
        neighbor_face_id = findfirst(x->issubset(x,neighbor_face), faces(neighbor_cell))
        face_neighbor = FaceIndex((neighborid, neighbor_face_id))
    else
        neighbor_face_id = findfirst(x->issubset(x,neighbor_face), edges(neighbor_cell))
        face_neighbor = EdgeIndex((neighborid, neighbor_face_id))
    end
    cell_face_id = findfirst(x->issubset(x,neighbor_face),faces(cell))
    push!(V_face, EntityNeighborhood(face_neighbor))
    push!(I_face, cellid)
    push!(J_face, cell_face_id)
end



"""
    getneighborhood(top::ExclusiveTopology, grid::AbstractGrid, cellidx::CellIndex, include_self=false)
    getneighborhood(top::ExclusiveTopology, grid::AbstractGrid, faceidx::FaceIndex, include_self=false)
    getneighborhood(top::ExclusiveTopology, grid::AbstractGrid, vertexidx::VertexIndex, include_self=false)
    getneighborhood(top::ExclusiveTopology, grid::AbstractGrid, edgeidx::EdgeIndex, include_self=false)

Returns all directly connected entities of the same type, i.e. calling the function with a `VertexIndex` will return
a list of directly connected vertices (connected via face/edge). If `include_self` is true, the given `*Index` is included 
in the returned list.

!!! warning
    This feature is highly experimental and very likely subjected to interface changes in the future.
"""
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
        return [top.face_neighbor[faceidx[1],faceidx[2]].neighbor_info; faceidx]
    else
        return top.face_neighbor[faceidx[1],faceidx[2]].neighbor_info
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
            local_vertex = VertexIndex(cellid,findfirst(x->x==global_vertexid,vertices(getcells(grid,cellid))))
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
        local_neighbor_edgeid = findfirst(x->issubset(x,nonlocal_edgeid),edges(getcells(grid,cellid)))
        local_neighbor_edgeid === nothing && continue
        local_edge = EdgeIndex(cellid,local_neighbor_edgeid)
        push!(self_reference_local, local_edge)
    end
    if include_self  
        return unique([top.edge_neighbor[cellid, local_edgeidx].neighbor_info; self_reference_local; edgeidx])
    else
        return unique([top.edge_neighbor[cellid, local_edgeidx].neighbor_info; self_reference_local])
    end
end

"""
    faceskeleton(grid) -> Vector{FaceIndex}
Returns an iterateable face skeleton. The skeleton consists of `FaceIndex` that can be used to `reinit` 
`FaceValues`.
"""
faceskeleton(top::ExclusiveTopology, grid::AbstractGrid) =  top.face_skeleton