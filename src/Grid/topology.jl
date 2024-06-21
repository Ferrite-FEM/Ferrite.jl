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


abstract type AbstractTopology end

"""
    ExclusiveTopology(grid::AbstractGrid)

The **experimental feature** `ExclusiveTopology` saves topological (connectivity/neighborhood) data of the grid.
Only the highest dimensional neighborhood is saved. I.e., if something is connected by a face and an
edge, only the face neighborhood is saved. The lower dimensional neighborhood is recomputed when calling getneighborhood if needed.

# Fields
- `vertex_to_cell::AbstractArray{AbstractVector{Int}, 1}`:           global vertex id to all cells containing the vertex
- `cell_neighbor::AbstractArray{AbstractVector{Int}, 1}`:            cellid to all connected cells
- `face_neighbor::AbstractArray{AbstractVector{FaceIndex}, 2}`:      `face_neighbor[cellid,   local_face_id]`   -> neighboring faces
- `edge_neighbor::AbstractArray{AbstractVector{EdgeIndex}, 2}`:      `edge_neighbor[cellid,   local_edge_id]`   -> neighboring edges
- `vertex_neighbor::AbstractArray{AbstractVector{VertexIndex}, 2}`:  `vertex_neighbor[cellid, local_vertex_id]` -> neighboring vertices
- `face_skeleton::Union{Vector{FaceIndex}, Nothing}`:     List of unique faces in the grid given as `FaceIndex`
- `edge_skeleton::Union{Vector{EdgeIndex}, Nothing}`:     List of unique edges in the grid given as `EdgeIndex`
- `vertex_skeleton::Union{Vector{VertexIndex}, Nothing}`: List of unique vertices in the grid given as `VertexIndex`

!!! warning "Limitations"
    Non-conforming grids will silently not work.
    Grids with embedded cells (different reference dimension compared
    to the spatial dimension) are not supported, and will error on construction.

"""
mutable struct ExclusiveTopology <: AbstractTopology
    vertex_to_cell::ArrayOfVectorViews{Int, 1}
    cell_neighbor::ArrayOfVectorViews{Int, 1}
    # face_face_neighbor[cellid,local_face_id] -> exclusive connected entities (not restricted to one entity)
    face_face_neighbor::ArrayOfVectorViews{FaceIndex, 2}
    # edge_edge_neighbor[cellid,local_edge_id] -> exclusive connected entities of the given edge
    edge_edge_neighbor::ArrayOfVectorViews{EdgeIndex, 2}
    # vertex_vertex_neighbor[cellid,local_vertex_id] -> exclusive connected entities to the given vertex
    vertex_vertex_neighbor::ArrayOfVectorViews{VertexIndex, 2}

    face_skeleton::Union{Vector{FaceIndex}, Nothing}
    edge_skeleton::Union{Vector{EdgeIndex}, Nothing}
    vertex_skeleton::Union{Vector{VertexIndex}, Nothing}
end

function ExclusiveTopology(grid::AbstractGrid{sdim}) where sdim
    if sdim != get_reference_dimension(grid)
        error("ExclusiveTopology does not support embedded cells (i.e. reference dimensions different from the spatial dimension)")
    end
    cells = getcells(grid)
    nnodes = getnnodes(grid)
    ncells = length(cells)

    max_vertices, max_edges, max_faces = _max_nentities_per_cell(cells)
    vertex_to_cell = build_vertex_to_cell(cells; max_vertices, nnodes)
    cell_neighbor = build_cell_neighbor(grid, cells, vertex_to_cell; ncells)

    # Here we don't use the convenience constructor taking a function,
    # since we want to do it simultaneously for 3 data-types
    facedata = sizehint!(FaceIndex[], ncells * max_faces * _getsizehint(grid, FaceIndex))
    face_face_neighbor_buf = CollectionsOfViews.ConstructionBuffer(facedata, (ncells, max_faces), _getsizehint(grid, FaceIndex))
    edgedata = sizehint!(EdgeIndex[], ncells * max_edges * _getsizehint(grid, EdgeIndex))
    edge_edge_neighbor_buf = CollectionsOfViews.ConstructionBuffer(edgedata, (ncells, max_edges), _getsizehint(grid, EdgeIndex))
    vertdata = sizehint!(VertexIndex[], ncells * max_vertices * _getsizehint(grid, VertexIndex))
    vertex_vertex_neighbor_buf = CollectionsOfViews.ConstructionBuffer(vertdata, (ncells, max_vertices), _getsizehint(grid, VertexIndex))

    for (cell_id, cell) in enumerate(cells)
        for neighbor_cell_id in cell_neighbor[cell_id]
            neighbor_cell = cells[neighbor_cell_id]
            getrefdim(neighbor_cell) == getrefdim(cell) || error("Not supported")
            num_shared_vertices = _num_shared_vertices(cell, neighbor_cell)
            if num_shared_vertices == 1
                _add_single_vertex_neighbor!(vertex_vertex_neighbor_buf, cell, cell_id, neighbor_cell, neighbor_cell_id)
            # Shared edge
            elseif num_shared_vertices == 2
                _add_single_edge_neighbor!(edge_edge_neighbor_buf, cell, cell_id, neighbor_cell, neighbor_cell_id)
            # Shared face
            elseif num_shared_vertices >= 3
                _add_single_face_neighbor!(face_face_neighbor_buf, cell, cell_id, neighbor_cell, neighbor_cell_id)
            else
                error("Found connected elements without shared vertex... Mesh broken?")
            end
        end
    end
    face_face_neighbor     = ArrayOfVectorViews(face_face_neighbor_buf)
    edge_edge_neighbor     = ArrayOfVectorViews(edge_edge_neighbor_buf)
    vertex_vertex_neighbor = ArrayOfVectorViews(vertex_vertex_neighbor_buf)
    return ExclusiveTopology(vertex_to_cell, cell_neighbor, face_face_neighbor, edge_edge_neighbor, vertex_vertex_neighbor, nothing, nothing, nothing)
end

function get_facet_facet_neighborhood(t::ExclusiveTopology, g::AbstractGrid)
    return _get_facet_facet_neighborhood(t, Val(get_reference_dimension(g)))
end
_get_facet_facet_neighborhood(t::ExclusiveTopology, #=rdim=#::Val{1}) = t.vertex_vertex_neighbor
_get_facet_facet_neighborhood(t::ExclusiveTopology, #=rdim=#::Val{2}) = t.edge_edge_neighbor
_get_facet_facet_neighborhood(t::ExclusiveTopology, #=rdim=#::Val{3}) = t.face_face_neighbor
function _get_facet_facet_neighborhood(::ExclusiveTopology, #=rdim=#::Val{:mixed})
    throw(ArgumentError("get_facet_facet_neightborhood is only supported for grids containing cells with the same reference dimension.
    Access the `vertex_vertex_neighbor`, `edge_edge_neighbor`, or `face_face_neighbor` fields explicitly instead."))
end

# Guess of how many neighbors depending on grid dimension and index type.
# Better to guess a bit too high than a bit too low.
function _getsizehint(g::AbstractGrid, ::Type{IDX}) where IDX
    CT = getcelltype(g)
    isconcretetype(CT) && return _getsizehint(getrefshape(CT)(), IDX)
    rdim = get_reference_dimension(g)::Int
    return _getsizehint(RefSimplex{rdim}(), IDX) # Simplex is "worst case", used as default.
end
_getsizehint(::AbstractRefShape,    ::Type{FaceIndex}) = 1 # Always 1 or zero if not mixed rdim

_getsizehint(::AbstractRefShape{1}, ::Type{EdgeIndex}) = 1
_getsizehint(::AbstractRefShape{2}, ::Type{EdgeIndex}) = 1
_getsizehint(::AbstractRefShape{3}, ::Type{EdgeIndex}) = 3 # Number for RefTetrahedron
_getsizehint(::RefHexahedron,       ::Type{EdgeIndex}) = 1 # Optim for RefHexahedron

_getsizehint(::AbstractRefShape{1}, ::Type{VertexIndex}) = 1
_getsizehint(::AbstractRefShape{2}, ::Type{VertexIndex}) = 3
_getsizehint(::AbstractRefShape{3}, ::Type{VertexIndex}) = 13
_getsizehint(::RefHypercube,        ::Type{VertexIndex}) = 1 # Optim for RefHypercube

_getsizehint(::AbstractRefShape{1}, ::Type{CellIndex}) = 2
_getsizehint(::AbstractRefShape{2}, ::Type{CellIndex}) = 12
_getsizehint(::AbstractRefShape{3}, ::Type{CellIndex}) = 70
_getsizehint(::RefQuadrilateral,    ::Type{CellIndex}) = 8
_getsizehint(::RefHexahedron,       ::Type{CellIndex}) = 26

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

"Return the highest number of vertices, edges, and faces per cell"
function _max_nentities_per_cell(cells::Vector{C}) where C
    if isconcretetype(C)
        cell = first(cells)
        return nvertices(cell), nedges(cell), nfaces(cell)
    else
        celltypes = Set(typeof.(cells))
        max_vertices = 0
        max_edges = 0
        max_faces = 0
        for celltype in celltypes
            celltypeidx = findfirst(x -> isa(x, celltype), cells)
            max_vertices = max(max_vertices, nvertices(cells[celltypeidx]))
            max_edges = max(max_edges, nedges(cells[celltypeidx]))
            max_faces = max(max_faces, nfaces(cells[celltypeidx]))
        end
        return max_vertices, max_edges, max_faces
    end
end

function _add_single_face_neighbor!(face_table::ConstructionBuffer, cell::AbstractCell, cell_id::Int, cell_neighbor::AbstractCell, cell_neighbor_id::Int)
    for (lfi, face) ∈ enumerate(faces(cell))
        uniqueface = sortface_fast(face)
        for (lfi2, face_neighbor) ∈ enumerate(faces(cell_neighbor))
            uniqueface2 = sortface_fast(face_neighbor)
            if uniqueface == uniqueface2
                push_at_index!(face_table, FaceIndex(cell_neighbor_id, lfi2), cell_id, lfi)
                return
            end
        end
    end
end

function _add_single_edge_neighbor!(edge_table::ConstructionBuffer, cell::AbstractCell, cell_id::Int, cell_neighbor::AbstractCell, cell_neighbor_id::Int)
    for (lei, edge) ∈ enumerate(edges(cell))
        uniqueedge = sortedge_fast(edge)
        for (lei2, edge_neighbor) ∈ enumerate(edges(cell_neighbor))
            uniqueedge2 = sortedge_fast(edge_neighbor)
            if uniqueedge == uniqueedge2
                push_at_index!(edge_table, EdgeIndex(cell_neighbor_id, lei2), cell_id, lei)
                return
            end
        end
    end
end

function _add_single_vertex_neighbor!(vertex_table::ConstructionBuffer, cell::AbstractCell, cell_id::Int, cell_neighbor::AbstractCell, cell_neighbor_id::Int)
    for (lvi, vertex) ∈ enumerate(vertices(cell))
        for (lvi2, vertex_neighbor) ∈ enumerate(vertices(cell_neighbor))
            if vertex_neighbor == vertex
                push_at_index!(vertex_table, VertexIndex(cell_neighbor_id, lvi2), cell_id, lvi)
                break
            end
        end
    end
end

function build_vertex_to_cell(cells; max_vertices, nnodes)
    vertex_to_cell = ArrayOfVectorViews(sizehint!(Int[], max_vertices * nnodes), (nnodes,); sizehint = max_vertices) do cov
            for (cellid, cell) in enumerate(cells)
                for vertex in vertices(cell)
                    push_at_index!(cov, cellid, vertex)
                end
            end
        end
    return vertex_to_cell
end

function build_cell_neighbor(grid, cells, vertex_to_cell; ncells)
    # In this case, we loop over the cells in order and all all neighbors at once.
    # Then we can create ArrayOfVectorViews directly without the CollectionsOfViews.ConstructionBuffer
    sizehint = _getsizehint(grid, CellIndex)
    data = empty!(Vector{Int}(undef, ncells * sizehint))

    indices = Vector{Int}(undef, ncells + 1)
    cell_neighbor_ids = Int[]
    n = 1
    for (cell_id, cell) in enumerate(cells)
        empty!(cell_neighbor_ids)
        for vertex ∈ vertices(cell)
            for vertex_cell_id ∈ vertex_to_cell[vertex]
                if vertex_cell_id != cell_id
                    vertex_cell_id ∈ cell_neighbor_ids || push!(cell_neighbor_ids, vertex_cell_id)
                end
            end
        end
        indices[cell_id] = n
        append!(data, cell_neighbor_ids)
        n += length(cell_neighbor_ids)
    end
    indices[end] = n
    sizehint!(data, length(data)) # Tell julia that we won't add more data
    return ArrayOfVectorViews(indices, data, LinearIndices(1:ncells))
end

function getneighborhood(top::ExclusiveTopology, grid::AbstractGrid, cellidx::CellIndex, include_self=false)
    patch = top.cell_neighbor[cellidx.idx]
    if include_self
        return view(push!(collect(patch), cellidx.idx), 1:(length(patch) + 1))
    else
        return patch
    end
end

function getneighborhood(top::ExclusiveTopology, grid::AbstractGrid, faceidx::FaceIndex, include_self=false)
    neighbors = top.face_face_neighbor[faceidx[1], faceidx[2]]
    if include_self
        return view(push!(collect(neighbors), faceidx), 1:(length(neighbors) + 1))
    else
        return neighbors
    end
end

function getneighborhood(top::ExclusiveTopology, grid::AbstractGrid{2}, edgeidx::EdgeIndex, include_self=false)
    neighbors = top.edge_edge_neighbor[edgeidx[1], edgeidx[2]]
    if include_self
        return view(push!(collect(neighbors), edgeidx), 1:(length(neighbors) + 1))
    else
        return neighbors
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
    return view(self_reference_local, 1:length(self_reference_local))
end

function getneighborhood(top::ExclusiveTopology, grid::AbstractGrid{3}, edgeidx::EdgeIndex, include_self=false)
    cellid, local_edgeidx = edgeidx[1], edgeidx[2]
    cell_edges = edges(getcells(grid, cellid))
    nonlocal_edgeid = cell_edges[local_edgeidx]
    cell_neighbors = getneighborhood(top, grid, CellIndex(cellid))
    self_reference_local = EdgeIndex[]
    for cellid in cell_neighbors
        local_neighbor_edgeid = findfirst(x -> issubset(x, nonlocal_edgeid), edges(getcells(grid, cellid)))
        local_neighbor_edgeid === nothing && continue
        local_edge = EdgeIndex(cellid,local_neighbor_edgeid)
        push!(self_reference_local, local_edge)
    end
    if include_self
        neighbors = unique([top.edge_edge_neighbor[cellid, local_edgeidx]; self_reference_local; edgeidx])
    else
        neighbors = unique([top.edge_edge_neighbor[cellid, local_edgeidx]; self_reference_local])
    end
    return view(neighbors, 1:length(neighbors))
end

function getneighborhood(top::ExclusiveTopology, grid::AbstractGrid, facetindex::FacetIndex, include_self=false)
    rdim = get_reference_dimension(grid)
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
    vertex_star_stencils(top::ExclusiveTopology, grid::Grid) -> AbstractVector{AbstractVector{VertexIndex}}
Computes the stencils induced by the edge connectivity of the vertices.
"""
function vertex_star_stencils(top::ExclusiveTopology, grid::Grid)
    cells = grid.cells
    stencil_table = ArrayOfVectorViews(VertexIndex[], (getnnodes(grid),); sizehint = 10) do buf
        # Vertex Connectivity
        for (global_vertexid,cellset) ∈ enumerate(top.vertex_to_cell)
            for cell ∈ cellset
                neighbor_boundary = edges(cells[cell])
                neighbor_connected_faces = neighbor_boundary[findall(x->global_vertexid ∈ x, neighbor_boundary)]
                this_local_vertex = findfirst(i->toglobal(grid, VertexIndex(cell, i)) == global_vertexid, 1:nvertices(cells[cell]))
                push_at_index!(buf, VertexIndex(cell, this_local_vertex), global_vertexid)
                other_vertices = findfirst.(x->x!=global_vertexid,neighbor_connected_faces)
                any(other_vertices .=== nothing) && continue
                neighbor_vertices_global = getindex.(neighbor_connected_faces, other_vertices)
                neighbor_vertices_local = [VertexIndex(cell,local_vertex) for local_vertex ∈ findall(x->x ∈ neighbor_vertices_global, vertices(cells[cell]))]
                for vertex_index in neighbor_vertices_local
                    push_at_index!(buf, vertex_index, global_vertexid)
                end
            end
        end
    end
    return stencil_table
end

"""
    getstencil(top::ArrayOfVectorViews{VertexIndex, 1}, grid::AbstractGrid, vertex_idx::VertexIndex) -> AbstractVector{VertexIndex}
Get an iterateable over the stencil members for a given local entity.
"""
function getstencil(top::ArrayOfVectorViews{VertexIndex, 1}, grid::Grid, vertex_idx::VertexIndex)
    return top[toglobal(grid, vertex_idx)]
end

"""
    _create_skeleton(neighborhood::AbstractMatrix{AbstractVector{BI}}) where BI <: Union{FaceIndex, EdgeIndex, VertexIndex}

Materializes the skeleton from the `neighborhood` information by returning a `Vector{BI}` with `BI`s describing
the unique entities in the grid.

*Example:* With `BI=EdgeIndex`, and an edge between cells and 1 and 2, with vertices 2 and 5, could be described by either
`EdgeIndex(1, 2)` or `EdgeIndex(2, 4)`, but only one of these will be in the vector returned by this function.
"""
function _create_skeleton(neighborhood::ArrayOfVectorViews{BI, 2}) where BI <: Union{FaceIndex, EdgeIndex, VertexIndex}
    i = 1
    skeleton = Vector{BI}(undef, length(neighborhood) - count(neighbors -> !isempty(neighbors) , values(neighborhood)) ÷ 2)
    for (idx, entity) in pairs(neighborhood)
        isempty(entity) || entity[][1] > idx[1] || continue
        skeleton[i] = BI(idx[1], idx[2])
        i += 1
    end
    return skeleton
end

#TODO: For the specific entities the grid input is unused
"""
    vertexskeleton(top::ExclusiveTopology, ::AbstractGrid) -> Vector{VertexIndex}

Materializes the skeleton from the `neighborhood` information by returning a `Vector{VertexIndex}`
describing the unique vertices in the grid. (One unique vertex may have multiple `VertexIndex`, but only
one is included in the returned `Vector`)
"""
function vertexskeleton(top::ExclusiveTopology, ::Union{AbstractGrid,Nothing}=nothing)
    if top.vertex_skeleton === nothing
        top.vertex_skeleton = _create_skeleton(top.vertex_vertex_neighbor)
    end
    return top.vertex_skeleton
end

"""
    edgeskeleton(top::ExclusiveTopology, ::AbstractGrid) -> Vector{EdgeIndex}

Materializes the skeleton from the `neighborhood` information by returning a `Vector{EdgeIndex}`
describing the unique edge in the grid. (One unique edge may have multiple `EdgeIndex`, but only
one is included in the returned `Vector`)
"""
function edgeskeleton(top::ExclusiveTopology, ::Union{AbstractGrid,Nothing}=nothing)
    if top.edge_skeleton === nothing
        top.edge_skeleton = _create_skeleton(top.edge_edge_neighbor)
    end
    return top.edge_skeleton
end

"""
    faceskeleton(top::ExclusiveTopology, ::AbstractGrid) -> Vector{FaceIndex}

Materializes the skeleton from the `neighborhood` information by returning a `Vector{FaceIndex}`
describing the unique faces in the grid. (One unique face may have multiple `FaceIndex`, but only
one is included in the returned `Vector`)
"""
function faceskeleton(top::ExclusiveTopology, ::Union{AbstractGrid,Nothing}=nothing)
    if top.face_skeleton === nothing
        top.face_skeleton = _create_skeleton(top.face_face_neighbor)
    end
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
    rdim = get_reference_dimension(grid)
    return _facetskeleton(top, Val(rdim))
end
_facetskeleton(top::ExclusiveTopology, #=rdim=#::Val{1}) = vertexskeleton(top)
_facetskeleton(top::ExclusiveTopology, #=rdim=#::Val{2}) = edgeskeleton(top)
_facetskeleton(top::ExclusiveTopology, #=rdim=#::Val{3}) = faceskeleton(top)
function _facetskeleton(::ExclusiveTopology, #=rdim=#::Val{:mixed})
    throw(ArgumentError("facetskeleton is only supported for grids containing cells with a common reference dimension.
    For mixed-dimensionality grid, use `faceskeleton`, `edgeskeleton`, and `vertexskeleton` explicitly"))
end
