
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
vector for all cells that dispatch `vertices`, `edges`, and `faces`.
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
    edge_skeleton::Union{Vector{EdgeIndex}, Nothing}
    vertex_skeleton::Union{Vector{VertexIndex}, Nothing}
    # TODO reintroduce the codimensional connectivity, e.g. 3D edge to 2D face
end

function get_facet_facet_neighborhood(t::ExclusiveTopology, g::AbstractGrid)
    return _get_facet_facet_neighborhood(t, Val(get_reference_dimensionality(g)))
end
_get_facet_facet_neighborhood(t::ExclusiveTopology, #=rdim=#::Val{1}) = t.vertex_vertex_neighbor
_get_facet_facet_neighborhood(t::ExclusiveTopology, #=rdim=#::Val{2}) = t.edge_edge_neighbor
_get_facet_facet_neighborhood(t::ExclusiveTopology, #=rdim=#::Val{3}) = t.face_face_neighbor
function _get_facet_facet_neighborhood(t::ExclusiveTopology, #=rdim=#::Val{:mixed})
    throw(ArgumentError("get_facet_facet_neightborhood is only supported for grids containing cells with the same reference dimension.
    Access the `vertex_vertex_neighbor`, `edge_edge_neighbor`, or `face_face_neighbor` fields explicitly instead."))
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
            # Shared edge
            elseif num_shared_vertices == 2
                _add_single_edge_neighbor!(edge_table, cell, cell_id, cell_neighbor, cell_neighbor_id)
            # Shared face
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
        max_faces = nfaces(cells[1])
        max_edges = nedges(cells[1])
    else
        celltypes = Set(typeof.(cells))
        for celltype in celltypes
            celltypeidx = findfirst(x->typeof(x)==celltype,cells)
            dim = getdim(cells[celltypeidx])

            max_vertices = max(max_vertices,nvertices(cells[celltypeidx]))
            max_faces = max(max_faces, nfaces(cells[celltypeidx]))
            max_edges = max(max_edges, nedges(cells[celltypeidx]))
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

    return ExclusiveTopology(vertex_cell_table,cell_neighbor_table,face_table,vertex_table,edge_table,nothing,nothing,nothing)
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
    vertex_to_cell = top.vertex_to_cell[global_vertexid]
    self_reference_local = Vector{VertexIndex}()
    sizehint!(self_reference_local, length(vertex_to_cell))
    for (i,cellid) in enumerate(vertex_to_cell)
        local_vertex = VertexIndex(cellid,findfirst(x->x==global_vertexid,vertices(getcells(grid,cellid)))::Int)
        !include_self && local_vertex == vertexidx && continue
        push!(self_reference_local, local_vertex)
    end
    return self_reference_local
end

function getneighborhood(top::ExclusiveTopology, grid::AbstractGrid, edgeidx::EdgeIndex, include_self=false)
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
        return unique([top.edge_edge_neighbor[cellid, local_edgeidx].neighbor_info; self_reference_local; edgeidx])
    else
        return unique([top.edge_edge_neighbor[cellid, local_edgeidx].neighbor_info; self_reference_local])
    end
end

function getneighborhood(top::ExclusiveTopology, grid::AbstractGrid, facetindex::FacetIndex, include_self=false)
    rdim = get_reference_dimensionality(grid)
    return _getneighborhood(Val(rdim), top, grid, facetindex, include_self)
end
_getneighborhood(::Val{1}, top, grid, facetindex::FacetIndex, include_self) = getneighborhood(top, grid, VertexIndex(facetindex...), include_self)
_getneighborhood(::Val{2}, top, grid, facetindex::FacetIndex, include_self) = getneighborhood(top, grid, EdgeIndex(facetindex...), include_self)
_getneighborhood(::Val{3}, top, grid, facetindex::FacetIndex, include_self) = getneighborhood(top, grid, FaceIndex(facetindex...), include_self)
function _getneighborhood(::Val{:mixed}, args...)
    throw(ArgumentError("getneighborhood with FacetIndex is is only supported for grids containing cells with a common reference dimension.
    For mixed-dimensionality grid, use `VertexIndex`, `EdgeIndex`, and `FaceIndex` explicitly"))
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
            neighbor_boundary = edges(cells[cell])
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
    _create_skeleton(neighborhood::Matrix{EntityNeighborhood{BI}}) where BI <: Union{FaceIndex, EdgeIndex, VertexIndex}

Materializes the skeleton from the `neighborhood` information by returning a `Vector{BI}` with `BI`s describing 
the unique entities in the grid. 

*Example:* With `BI=EdgeIndex`, and an edge between cells and 1 and 2, with vertices 2 and 5, could be described by either 
`EdgeIndex(1, 2)` or `EdgeIndex(2, 4)`, but only one of these will be in the vector returned by this function. 
"""
function _create_skeleton(neighborhood::Matrix{EntityNeighborhood{BI}}) where BI <: Union{FaceIndex, EdgeIndex, VertexIndex}
    i = 1
    skeleton = Vector{BI}(undef, length(neighborhood) - count(neighbors -> !isempty(neighbors) , neighborhood) ÷ 2)
    for (idx, entity) in pairs(neighborhood)
        isempty(entity.neighbor_info) || entity.neighbor_info[][1] > idx[1] || continue
        skeleton[i] = BI(idx[1], idx[2])
        i += 1
    end
    return skeleton
end
_create_skeleton(::Nothing) = nothing

#TODO: For the specific entities the grid input is unused
"""
    vertexskeleton(top::ExclusiveTopology, ::AbstractGrid) -> Vector{VertexIndex}

Materializes the skeleton from the `neighborhood` information by returning a `Vector{VertexIndex}` 
describing the unique vertices in the grid. (One unique vertex may have multiple `VertexIndex`, but only 
one is included in the returned `Vector`) 
"""
function vertexskeleton(top::ExclusiveTopology, ::Union{AbstractGrid,Nothing}=nothing)
    top.vertex_skeleton = _create_skeleton(top.vertex_vertex_neighbor)
    return top.vertex_skeleton
end

"""
    edgeskeleton(top::ExclusiveTopology, ::AbstractGrid) -> Vector{EdgeIndex}

Materializes the skeleton from the `neighborhood` information by returning a `Vector{EdgeIndex}` 
describing the unique edge in the grid. (One unique edge may have multiple `EdgeIndex`, but only 
one is included in the returned `Vector`) 
"""
function edgeskeleton(top::ExclusiveTopology, ::Union{AbstractGrid,Nothing}=nothing)
    top.edge_skeleton = _create_skeleton(top.edge_edge_neighbor)
    return top.edge_skeleton
end

"""
    faceskeleton(top::ExclusiveTopology, ::AbstractGrid) -> Vector{FaceIndex}

Materializes the skeleton from the `neighborhood` information by returning a `Vector{FaceIndex}` 
describing the unique faces in the grid. (One unique face may have multiple `FaceIndex`, but only 
one is included in the returned `Vector`) 
"""
function faceskeleton(top::ExclusiveTopology, ::Union{AbstractGrid,Nothing}=nothing)
    top.face_skeleton = _create_skeleton(top.face_face_neighbor)
    return top.face_skeleton
end

"""
    facetskeleton(top::ExclusiveTopology, grid::AbstractGrid)

Materializes the skeleton from the `neighborhood` information by returning a `Vector{BI}` where 
`BI <: Union{VertexIndex, EdgeIndex, FaceIndex}`. 
It describes the unique facets in the grid, and allows for dimension-independent code in the case 
that all cells have the same reference dimension. For cells with different reference dimensions, 
[`Ferrite.vertexskeleton`](@ref), [`Ferrite.edgeskeleton`](@ref), or [`Ferrite.faceskeleton`](@ref)
must be used explicitly. 
"""
function facetskeleton(top::ExclusiveTopology, grid::AbstractGrid)
    rdim = get_reference_dimensionality(grid)
    return _facetskeleton(top, Val(rdim))
end
_facetskeleton(top::ExclusiveTopology, #=rdim=#::Val{1}) = vertexskeleton(top)
_facetskeleton(top::ExclusiveTopology, #=rdim=#::Val{2}) = edgeskeleton(top)
_facetskeleton(top::ExclusiveTopology, #=rdim=#::Val{3}) = faceskeleton(top)
function _facetskeleton(::ExclusiveTopology, #=rdim=#::Val{:mixed})
    throw(ArgumentError("facetskeleton is only supported for grids containing cells with a common reference dimension.
    For mixed-dimensionality grid, use `faceskeleton`, `edgeskeleton`, and `vertexskeleton` explicitly"))
end
