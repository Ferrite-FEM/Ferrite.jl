############
# Topology #
############

"""
    getneighborhood(topology, grid::AbstractGrid, cellidx::CellIndex, include_self = false)
    getneighborhood(topology, grid::AbstractGrid, faceidx::FaceIndex, include_self = false)
    getneighborhood(topology, grid::AbstractGrid, vertexidx::VertexIndex, include_self = false)
    getneighborhood(topology, grid::AbstractGrid, edgeidx::EdgeIndex, include_self = false)

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

Grids with mixed reference dimensions (e.g. a 3D grid containing both `Hexahedron` and
`Quadrilateral` cells) are supported: the shared entity is classified by the number of
shared vertices, so mixed-dimensional connections are stored in `vertex_vertex_neighbor`,
`edge_edge_neighbor`, or `face_face_neighbor` accordingly. Per-entity queries with
`VertexIndex`, `EdgeIndex`, `FaceIndex`, and `FacetIndex` work for such grids (the facet
dimension is resolved per cell). Please remember that the dimensions of faces (dim=2) and
edges (dim=1) are fixed, i.e. they are defined independent of the spatial dimension of the
grid. You can consult [the entity naming page](@ref entity-naming-docs) for
details. The bulk operations [`facetskeleton`](@ref) and `get_facet_facet_neighborhood`
remain unsupported, since they assume a common facet dimension across the whole grid.

!!! warning "Limitations"
    The implementation only works with conforming grids, i.e. grids without "hanging nodes". Non-conforming grids will give unexpected results.
    Purely embedded grids, where the highest cell reference dimension is smaller than the
    spatial dimension (e.g. a shell grid of `Quadrilateral`s in 3D space), are not
    supported and will error on construction.

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
    facet_skeleton::Union{Vector{FacetIndex}, Nothing}
end

function ExclusiveTopology(grid::AbstractGrid{sdim}) where {sdim}
    cells = getcells(grid)
    if _max_reference_dimension(cells) != sdim
        error("ExclusiveTopology requires the highest cell reference dimension to equal the spatial dimension ($sdim). Purely embedded (e.g. shell) grids are not supported.")
    end
    nnodes = getnnodes(grid)
    ncells = length(cells)

    max_vertices, max_edges, max_faces = _max_nentities_per_cell(cells)
    vertex_to_cell = build_vertex_to_cell(cells; nnodes)
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
            num_shared_vertices = _num_shared_vertices(cell, neighbor_cell)
            # The number of shared vertices indicates the expected shared entity (1 -> vertex,
            # 2 -> edge, >=3 -> face). For grids with mixed reference dimensions the shared
            # vertices may not actually form that entity in both cells (e.g. a `Line` spanning
            # the face-diagonal of a `Hexahedron` shares 2 vertices that are not a common edge).
            # In that case we fall back to the next lower-dimensional entity.
            if num_shared_vertices == 1
                _add_single_vertex_neighbor!(vertex_vertex_neighbor_buf, cell, cell_id, neighbor_cell, neighbor_cell_id)
            elseif num_shared_vertices == 2 # Shared edge (or two separate vertices)
                # For cells with reference dimension <= 2 the pairwise search is over so
                # few edges that it beats collecting the shared vertices for a targeted
                # lookup first (the branch is static for concrete cell types).
                added = if getrefdim(cell) == 3
                    g1, g2, _, _ = _first_shared_vertices(cell, neighbor_cell)
                    _add_edge_neighbor!(edge_edge_neighbor_buf, cell, cell_id, neighbor_cell, neighbor_cell_id, sortedge_fast((g1, g2)))
                else
                    _add_single_edge_neighbor!(edge_edge_neighbor_buf, cell, cell_id, neighbor_cell, neighbor_cell_id)
                end
                added || _add_single_vertex_neighbor!(vertex_vertex_neighbor_buf, cell, cell_id, neighbor_cell, neighbor_cell_id)
            elseif num_shared_vertices == 3 # Shared triangular face (or lower-dimensional entities)
                g1, g2, g3, _ = _first_shared_vertices(cell, neighbor_cell)
                _add_face_neighbor!(face_face_neighbor_buf, cell, cell_id, neighbor_cell, neighbor_cell_id, sortface_fast((g1, g2, g3))) ||
                    _add_single_edge_neighbor!(edge_edge_neighbor_buf, cell, cell_id, neighbor_cell, neighbor_cell_id) ||
                    _add_single_vertex_neighbor!(vertex_vertex_neighbor_buf, cell, cell_id, neighbor_cell, neighbor_cell_id)
            elseif num_shared_vertices >= 4 # Shared quadrilateral face (or lower-dimensional entities)
                g1, g2, g3, g4 = _first_shared_vertices(cell, neighbor_cell)
                _add_face_neighbor!(face_face_neighbor_buf, cell, cell_id, neighbor_cell, neighbor_cell_id, sortface_fast((g1, g2, g3, g4))) ||
                    _add_single_face_neighbor!(face_face_neighbor_buf, cell, cell_id, neighbor_cell, neighbor_cell_id) ||
                    _add_single_edge_neighbor!(edge_edge_neighbor_buf, cell, cell_id, neighbor_cell, neighbor_cell_id) ||
                    _add_single_vertex_neighbor!(vertex_vertex_neighbor_buf, cell, cell_id, neighbor_cell, neighbor_cell_id)
            else
                error("Found connected elements without shared vertex... Mesh broken?")
            end
        end
    end
    face_face_neighbor = ArrayOfVectorViews(face_face_neighbor_buf)
    edge_edge_neighbor = ArrayOfVectorViews(edge_edge_neighbor_buf)
    vertex_vertex_neighbor = ArrayOfVectorViews(vertex_vertex_neighbor_buf)
    return ExclusiveTopology(vertex_to_cell, cell_neighbor, face_face_neighbor, edge_edge_neighbor, vertex_vertex_neighbor, nothing)
end

function get_facet_facet_neighborhood(t::ExclusiveTopology, g::AbstractGrid)
    return _get_facet_facet_neighborhood(t, Val(get_reference_dimension(g)))
end
_get_facet_facet_neighborhood(t::ExclusiveTopology, #=rdim=# ::Val{1}) = t.vertex_vertex_neighbor
_get_facet_facet_neighborhood(t::ExclusiveTopology, #=rdim=# ::Val{2}) = t.edge_edge_neighbor
_get_facet_facet_neighborhood(t::ExclusiveTopology, #=rdim=# ::Val{3}) = t.face_face_neighbor
function _get_facet_facet_neighborhood(::ExclusiveTopology, #=rdim=# ::Val{:mixed})
    throw(ArgumentError("get_facet_facet_neighborhood is only supported for grids containing cells with the same reference dimension.
    Access the `vertex_vertex_neighbor`, `edge_edge_neighbor`, or `face_face_neighbor` fields explicitly instead."))
end

# Guess of how many neighbors depending on grid dimension and index type.
# This could be possible to optimize further by studying connectivities of non-uniform
# grids, see https://github.com/Ferrite-FEM/Ferrite.jl/pull/974#discussion_r1660838649
function _getsizehint(g::AbstractGrid, ::Type{IDX}) where {IDX}
    CT = getcelltype(g)
    isconcretetype(CT) && return _getsizehint(getrefshape(CT)(), IDX)
    rdim = get_reference_dimension(g)
    rdim isa Int || (rdim = getspatialdim(g)) # Mixed reference dimensions: use spatial dim (the highest rdim).
    return _getsizehint(RefSimplex{rdim}(), IDX) # Simplex is "worst case", used as default.
end

# Highest reference dimension among the cells. Equals `get_reference_dimension` for
# grids with a single reference dimension, and is used to detect purely embedded grids.
_max_reference_dimension(cells::AbstractVector{<:AbstractCell{<:AbstractRefShape{rdim}}}) where {rdim} = rdim
_max_reference_dimension(cells::AbstractVector{<:AbstractCell}) = maximum(getrefdim, cells)
_getsizehint(::AbstractRefShape, ::Type{FaceIndex}) = 1 # Always 1 or zero if not mixed rdim

_getsizehint(::AbstractRefShape{1}, ::Type{EdgeIndex}) = 1
_getsizehint(::AbstractRefShape{2}, ::Type{EdgeIndex}) = 1
_getsizehint(::AbstractRefShape{3}, ::Type{EdgeIndex}) = 3 # Number for RefTetrahedron
_getsizehint(::RefHexahedron, ::Type{EdgeIndex}) = 1 # Optim for RefHexahedron

_getsizehint(::AbstractRefShape{1}, ::Type{VertexIndex}) = 1
_getsizehint(::AbstractRefShape{2}, ::Type{VertexIndex}) = 3
_getsizehint(::AbstractRefShape{3}, ::Type{VertexIndex}) = 13
_getsizehint(::RefHypercube, ::Type{VertexIndex}) = 1 # Optim for RefHypercube

_getsizehint(::AbstractRefShape{1}, ::Type{CellIndex}) = 2
_getsizehint(::AbstractRefShape{2}, ::Type{CellIndex}) = 12
_getsizehint(::AbstractRefShape{3}, ::Type{CellIndex}) = 70
_getsizehint(::RefQuadrilateral, ::Type{CellIndex}) = 8
_getsizehint(::RefHexahedron, ::Type{CellIndex}) = 26

function _num_shared_vertices(cell_a::C1, cell_b::C2) where {C1, C2}
    num_shared_vertices = 0
    for vertex in vertices(cell_a)
        for vertex_neighbor in vertices(cell_b)
            # Branch-free accumulation so that the loops vectorize (a cell never contains
            # the same vertex twice, so each match is a distinct shared vertex).
            num_shared_vertices += Int(vertex_neighbor == vertex)
        end
    end
    return num_shared_vertices
end

# Global ids of the first four vertices shared by `cell_a` and `cell_b` (0 if fewer), used
# to look up the shared edge or face directly instead of comparing all pairs of them.
function _first_shared_vertices(cell_a::C1, cell_b::C2) where {C1, C2}
    num_shared_vertices = 0
    g1 = g2 = g3 = 0
    for vertex in vertices(cell_a)
        for vertex_neighbor in vertices(cell_b)
            if vertex_neighbor == vertex
                num_shared_vertices += 1
                if num_shared_vertices == 1
                    g1 = vertex
                elseif num_shared_vertices == 2
                    g2 = vertex
                elseif num_shared_vertices == 3
                    g3 = vertex
                else
                    return (g1, g2, g3, vertex)
                end
                break
            end
        end
    end
    return (g1, g2, g3, 0)
end

"Return the highest number of vertices, edges, and faces per cell"
function _max_nentities_per_cell(cells::Vector{C}) where {C}
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

# Add the face neighbor when the shared face is known to consist of the shared vertices,
# with `sorted_face` their unique representation from `sortface_fast` (3 indices identify
# a face, also for quadrilateral faces, see `sortface_fast`).
# Return `true` if both cells have such a face and the entry was recorded.
function _add_face_neighbor!(face_table::ConstructionBuffer, cell::AbstractCell, cell_id::Int, cell_neighbor::AbstractCell, cell_neighbor_id::Int, sorted_face::Tuple{Int, Int, Int})
    lfi = _find_local_entity(faces(cell), sortface_fast, sorted_face)
    lfi == 0 && return false
    lfi2 = _find_local_entity(faces(cell_neighbor), sortface_fast, sorted_face)
    lfi2 == 0 && return false
    push_at_index!(face_table, FaceIndex(cell_neighbor_id, lfi2), cell_id, lfi)
    return true
end

# Like `_add_face_neighbor!` but for the edge with sorted vertices `sorted_edge`.
function _add_edge_neighbor!(edge_table::ConstructionBuffer, cell::AbstractCell, cell_id::Int, cell_neighbor::AbstractCell, cell_neighbor_id::Int, sorted_edge::Tuple{Int, Int})
    lei = _find_local_entity(edges(cell), sortedge_fast, sorted_edge)
    lei == 0 && return false
    lei2 = _find_local_entity(edges(cell_neighbor), sortedge_fast, sorted_edge)
    lei2 == 0 && return false
    push_at_index!(edge_table, EdgeIndex(cell_neighbor_id, lei2), cell_id, lei)
    return true
end

# Local index of the entity in `entities` whose sorted vertices are `sorted_entity`, or 0.
function _find_local_entity(entities::Tuple, sortfun::F, sorted_entity::Tuple) where {F}
    for (i, entity) in pairs(entities)
        sortfun(entity) == sorted_entity && return i
    end
    return 0
end

# The `_add_single_*_neighbor!` functions return `true` if a shared entity was found and
# recorded, so the caller can fall back to a lower-dimensional entity if not (relevant for
# mixed reference dimension grids, see the construction loop).
function _add_single_face_neighbor!(face_table::ConstructionBuffer, cell::AbstractCell, cell_id::Int, cell_neighbor::AbstractCell, cell_neighbor_id::Int)
    for (lfi, face) in enumerate(faces(cell))
        uniqueface = sortface_fast(face)
        for (lfi2, face_neighbor) in enumerate(faces(cell_neighbor))
            uniqueface2 = sortface_fast(face_neighbor)
            if uniqueface == uniqueface2
                push_at_index!(face_table, FaceIndex(cell_neighbor_id, lfi2), cell_id, lfi)
                return true
            end
        end
    end
    return false
end

function _add_single_edge_neighbor!(edge_table::ConstructionBuffer, cell::AbstractCell, cell_id::Int, cell_neighbor::AbstractCell, cell_neighbor_id::Int)
    for (lei, edge) in enumerate(edges(cell))
        uniqueedge = sortedge_fast(edge)
        for (lei2, edge_neighbor) in enumerate(edges(cell_neighbor))
            uniqueedge2 = sortedge_fast(edge_neighbor)
            if uniqueedge == uniqueedge2
                push_at_index!(edge_table, EdgeIndex(cell_neighbor_id, lei2), cell_id, lei)
                return true
            end
        end
    end
    return false
end

function _add_single_vertex_neighbor!(vertex_table::ConstructionBuffer, cell::AbstractCell, cell_id::Int, cell_neighbor::AbstractCell, cell_neighbor_id::Int)
    found = false
    for (lvi, vertex) in enumerate(vertices(cell))
        for (lvi2, vertex_neighbor) in enumerate(vertices(cell_neighbor))
            if vertex_neighbor == vertex
                push_at_index!(vertex_table, VertexIndex(cell_neighbor_id, lvi2), cell_id, lvi)
                found = true
                break
            end
        end
    end
    return found
end

function build_vertex_to_cell(cells; nnodes)
    # Two-pass construction: count the number of cells for each vertex, compute the view
    # offsets, and then fill in the cell ids. This allocates exactly the required memory
    # and avoids the data relocations of the generic `ConstructionBuffer` approach.
    nextidx = zeros(Int, nnodes)
    for cell in cells
        for vertex in vertices(cell)
            nextidx[vertex] += 1
        end
    end
    indices = Vector{Int}(undef, nnodes + 1)
    indices[1] = 1
    for v in 1:nnodes
        indices[v + 1] = indices[v] + nextidx[v]
        nextidx[v] = indices[v] # Next free data slot for vertex v
    end
    data = Vector{Int}(undef, indices[end] - 1)
    for (cellid, cell) in enumerate(cells)
        for vertex in vertices(cell)
            data[nextidx[vertex]] = cellid
            nextidx[vertex] += 1
        end
    end
    return ArrayOfVectorViews(indices, data, LinearIndices(1:nnodes); checkargs = false)
end

function build_cell_neighbor(grid, cells, vertex_to_cell; ncells)
    # In this case, we loop over the cells in order and all neighbors at once.
    # Then we can create ArrayOfVectorViews directly without the CollectionsOfViews.ConstructionBuffer
    sizehint = _getsizehint(grid, CellIndex)
    data = empty!(Vector{Int}(undef, ncells * sizehint))

    indices = Vector{Int}(undef, ncells + 1)
    # last_recorded_by[c] is the latest cell that pushed c as its neighbor, giving O(1)
    # deduplication instead of a linear scan over the neighbors found so far.
    last_recorded_by = zeros(Int, ncells)
    n = 1
    for (cell_id, cell) in enumerate(cells)
        indices[cell_id] = n
        for vertex in vertices(cell)
            for vertex_cell_id in vertex_to_cell[vertex]
                if vertex_cell_id != cell_id && last_recorded_by[vertex_cell_id] != cell_id
                    last_recorded_by[vertex_cell_id] = cell_id
                    push!(data, vertex_cell_id)
                    n += 1
                end
            end
        end
    end
    indices[end] = n
    # Free the excess capacity if a significant part would be reclaimed (the shrink copies
    # the data, so it is not worth it for the typically well-matched size hints).
    if 4 * length(data) < 3 * ncells * sizehint
        sizehint!(data, length(data))
    end
    return ArrayOfVectorViews(indices, data, LinearIndices(1:ncells))
end

function getneighborhood(top::ExclusiveTopology, grid::AbstractGrid, cellidx::CellIndex, include_self = false)
    patch = top.cell_neighbor[cellidx.idx]
    if include_self
        return view(push!(collect(patch), cellidx.idx), 1:(length(patch) + 1))
    else
        return patch
    end
end

function getneighborhood(top::ExclusiveTopology, grid::AbstractGrid, faceidx::FaceIndex, include_self = false)
    neighbors = top.face_face_neighbor[faceidx[1], faceidx[2]]
    if include_self
        return view(push!(collect(neighbors), faceidx), 1:(length(neighbors) + 1))
    else
        return neighbors
    end
end

function getneighborhood(top::ExclusiveTopology, grid::AbstractGrid, edgeidx::EdgeIndex, include_self = false)
    # The reference dimension of the specific cell (not the grid) determines whether the edge
    # is a facet or a proper edge. This is well-defined even for mixed-dimensional grids.
    if get_reference_dimension(grid, edgeidx[1]) == 3
        return _getneighborhood_edge3(top, grid, edgeidx, include_self)
    end
    # For cells with reference dimension <= 2 the edge is a facet, shared by at most two
    # cells, so the stored exclusive neighborhood is already complete.
    neighbors = top.edge_edge_neighbor[edgeidx[1], edgeidx[2]]
    if include_self
        return view(push!(collect(neighbors), edgeidx), 1:(length(neighbors) + 1))
    else
        return neighbors
    end
end

function getneighborhood(top::ExclusiveTopology, grid::AbstractGrid, vertexidx::VertexIndex, include_self = false)
    cellid, local_vertexid = vertexidx[1], vertexidx[2]
    cell_vertices = vertices(getcells(grid, cellid))
    global_vertexid = cell_vertices[local_vertexid]
    vertex_to_cell = top.vertex_to_cell[global_vertexid]
    self_reference_local = Vector{VertexIndex}()
    sizehint!(self_reference_local, length(vertex_to_cell))
    for (i, cellid) in enumerate(vertex_to_cell)
        local_vertex = VertexIndex(cellid, findfirst(x -> x == global_vertexid, vertices(getcells(grid, cellid)))::Int)
        !include_self && local_vertex == vertexidx && continue
        push!(self_reference_local, local_vertex)
    end
    return view(self_reference_local, 1:length(self_reference_local))
end

# Recompute the full edge neighborhood for a reference-dimension-3 cell, where an edge is
# shared by potentially many cells and only the exclusive neighborhood is stored.
function _getneighborhood_edge3(top::ExclusiveTopology, grid::AbstractGrid, edgeidx::EdgeIndex, include_self)
    cellid, local_edgeidx = edgeidx[1], edgeidx[2]
    v1, v2 = edges(getcells(grid, cellid))[local_edgeidx]
    stored_neighbors = top.edge_edge_neighbor[cellid, local_edgeidx]
    nstored = length(stored_neighbors)
    neighbors = EdgeIndex[]
    sizehint!(neighbors, nstored + 4 + include_self)
    append!(neighbors, stored_neighbors)
    for neighbor_cellid in top.cell_neighbor[cellid]
        for (i, edge) in pairs(edges(getcells(grid, neighbor_cellid)))
            if (edge[1] == v1 && edge[2] == v2) || (edge[1] == v2 && edge[2] == v1)
                # The recomputed candidates are distinct (at most one per neighbor cell),
                # so only the stored entries can duplicate them.
                candidate = EdgeIndex(neighbor_cellid, i)
                candidate ∈ view(neighbors, 1:nstored) || push!(neighbors, candidate)
                break
            end
        end
    end
    # The edge itself is never a neighbor of itself, so no duplication check is needed.
    include_self && push!(neighbors, edgeidx)
    return view(neighbors, 1:length(neighbors))
end

function getneighborhood(top::ExclusiveTopology, grid::AbstractGrid, facetindex::FacetIndex, include_self = false)
    # The facet dimension is determined by the reference dimension of the specific cell, which
    # is well-defined even for grids with mixed reference dimensions.
    rdim = get_reference_dimension(grid, facetindex[1])
    return _getneighborhood(Val(rdim), top, grid, facetindex, include_self)
end
_getneighborhood(::Val{1}, top, grid, facetindex::FacetIndex, include_self) = getneighborhood(top, grid, VertexIndex(facetindex...), include_self)
_getneighborhood(::Val{2}, top, grid, facetindex::FacetIndex, include_self) = getneighborhood(top, grid, EdgeIndex(facetindex...), include_self)
_getneighborhood(::Val{3}, top, grid, facetindex::FacetIndex, include_self) = getneighborhood(top, grid, FaceIndex(facetindex...), include_self)

"""
    vertex_star_stencils(top::ExclusiveTopology, grid::Grid) -> AbstractVector{AbstractVector{VertexIndex}}
Computes the stencils induced by the edge connectivity of the vertices.
"""
function vertex_star_stencils(top::ExclusiveTopology, grid::Grid)
    cells = getcells(grid)
    stencil_table = ArrayOfVectorViews(VertexIndex[], (getnnodes(grid),); sizehint = 10) do buf
        # Vertex Connectivity
        for (global_vertexid, cellset) in enumerate(top.vertex_to_cell)
            for cell_id in cellset
                cell_vertices = vertices(cells[cell_id])
                # The vertex itself is part of the stencil
                for (lvi, gvi) in pairs(cell_vertices)
                    if gvi == global_vertexid
                        push_at_index!(buf, VertexIndex(cell_id, lvi), global_vertexid)
                        break
                    end
                end
                # All vertices connected to it by an edge of this cell
                for (lvi, gvi) in pairs(cell_vertices)
                    gvi == global_vertexid && continue
                    for edge in edges(cells[cell_id])
                        if (edge[1] == global_vertexid && edge[2] == gvi) || (edge[2] == global_vertexid && edge[1] == gvi)
                            push_at_index!(buf, VertexIndex(cell_id, lvi), global_vertexid)
                            break
                        end
                    end
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
    _create_facet_skeleton(neighborhood::AbstractMatrix{AbstractVector{BI}}, grid::AbstractGrid) where {BI <: Union{FaceIndex, EdgeIndex, VertexIndex}}

Materializes the skeleton from the `neighborhood` information by returning a `Vector{FacetIndex}` describing the
unique facets in the grid.

*Example:* With `BI=EdgeIndex`, and an edge between cells and 1 and 2, with vertices 2 and 5, could be described by either
`EdgeIndex(1, 2)` or `EdgeIndex(2, 4)`, but only one of these will be in the vector returned by this function.
"""
function _create_facet_skeleton(neighborhood::ArrayOfVectorViews{BI, 2}, grid) where {BI <: Union{FaceIndex, EdgeIndex, VertexIndex}}
    i = 1
    skeleton = Vector{FacetIndex}(undef, length(neighborhood) - count(neighbors -> !isempty(neighbors), values(neighborhood)) ÷ 2)
    for (idx, entity) in pairs(neighborhood)
        cell_nr = idx[1]
        facet_nr = idx[2]
        if !isconcretetype(getcelltype(grid)) # Mixed grid
            # `neighborhood`, indexed by (cell_idx, facet_nr). For mixed grids,
            # `facet_nr` will sometimes be too large.
            # facet_nr > nfacets(getcells(grid, cell_nr)) && continue
            facet_nr > nfacets(getcells(grid, cell_nr)) && continue
        end
        on_boundary = isempty(entity)
        if on_boundary || entity[][1] > cell_nr # Pick the cell with the lowest nr
            skeleton[i] = FacetIndex(cell_nr, facet_nr)
            i += 1
        end
    end
    return resize!(skeleton, i - 1)
end

"""
    facetskeleton(top::ExclusiveTopology, grid::AbstractGrid)

Materializes the skeleton from the `neighborhood` information by returning an iterable over the
unique facets in the grid, described by `FacetIndex`.
"""
function facetskeleton(top::ExclusiveTopology, grid::AbstractGrid)
    if top.facet_skeleton === nothing
        rdim = get_reference_dimension(grid)
        top.facet_skeleton = if rdim == 1
            _create_facet_skeleton(top.vertex_vertex_neighbor, grid)
        elseif rdim == 2
            _create_facet_skeleton(top.edge_edge_neighbor, grid)
        elseif rdim == 3
            _create_facet_skeleton(top.face_face_neighbor, grid)
        else
            throw(ArgumentError("facetskeleton not supported for refdim = $rdim"))
        end
    end
    return top.facet_skeleton
end

"""
    entity_dim(index_type::Union{VertexIndex, EdgeIndex, FaceIndex})

Queries the dimension of an entity. Spatial and reference dimensions coincide
for this query.
"""
entity_dim(::VertexIndex) = 0
entity_dim(::EdgeIndex) = 1
entity_dim(::FaceIndex) = 2

"""
    entity_codim(grid::AbstractGrid, index_type::Union{VertexIndex, EdgeIndex, FaceIndex})

Queries the relative dimension of an entity by its index type (i.e. `spatial dimension of grid - reference dimension of entity`).
"""
entity_codim(grid::AbstractGrid, idx::Union{VertexIndex, EdgeIndex, FaceIndex}) = get_reference_dimension(grid, idx[1]) - entity_dim(idx)
