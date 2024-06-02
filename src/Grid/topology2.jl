include("../CollectionOfVectors.jl")


# =============================================================================================== #
# EntityTopology
# =============================================================================================== #

"Unique representation of a face"
struct Face
    vertices::NTuple{3, Int}
    function Face(vertices::Union{NTuple{3, Int}, NTuple{4, Int}})
        return new(sortface_fast(vertices))
    end
end
function Face(grid::AbstractGrid, idx::FaceIndex)
    vertices = faces(getcells(grid, idx[1]))[idx[2]]
    return Face(vertices)
end

"Unique representation of an edge"
struct Edge
    vertices::NTuple{2, Int}
    function Edge(vertices::NTuple{2, Int})
        return new(sortedge_fast(vertices))
    end
end
function Edge(grid::AbstractGrid, idx::EdgeIndex)
    vertices = edges(getcells(grid, idx[1]))[idx[2]]
    return Edge(vertices)
end

# A Vertex uniquely represented by its node number.

"""
    Topology indexed by the entity, i.e. a `Face`, `Edge`, or `Vertex` where the
    latter is just represented as an `Int`.
"""
struct EntityTopology <: AbstractTopology
    faceneighbors::CollectionOfVectors{OrderedDict{Face, UnitRange{Int}}, FaceIndex}
    edgeneighbors::CollectionOfVectors{OrderedDict{Edge, UnitRange{Int}}, EdgeIndex}
    vertexneighbors::CollectionOfVectors{Vector{UnitRange{Int}}, VertexIndex}
end

function EntityTopology(grid::AbstractGrid)
    return EntityTopology(
        build_neighborhood(grid,   FaceIndex, Face), #TODO: Skip in 1d and 2d
        build_neighborhood(grid,   EdgeIndex, Edge), #TODO: Skip in 1d
        build_neighborhood(grid, VertexIndex, Int))
end

function build_neighborhood(grid, ::Type{BI}, ::Type{ET}; sizehint=_getsizehint(grid, BI)) where {BI<:BoundaryIndex, ET<:Union{Edge, Face}}
    return CollectionOfVectors(OrderedDict{ET}, BI; sizehint) do b
        for (cellnr, cell) in enumerate(getcells(grid))
            for (entitynr, entity_vertices) in enumerate(boundaryfunction(BI)(cell))
                add!(b, BI(cellnr, entitynr), ET(entity_vertices))
            end
        end
    end
end

function build_neighborhood(grid, IdxType::Type{VertexIndex}, ::Type{Int}; sizehint=_getsizehint(grid, IdxType))
    return CollectionOfVectors(Vector, VertexIndex; sizehint, dims=(getnnodes(grid),)) do b
        for (cellnr, cell) in enumerate(getcells(grid))
            for (vertexnr, global_vertex) in enumerate(vertices(cell))
                add!(b, VertexIndex(cellnr, vertexnr), global_vertex)
            end
        end
    end
end

# Guess of how many neighbors depending on grid dimension and index type.
# This is just a performance optimization, and a good default is sufficient.
_getsizehint(::AbstractGrid{3}, ::Type{FaceIndex}) = 2
_getsizehint(::AbstractGrid, ::Type{FaceIndex}) = 0 # No faces exists in 2d or lower dim
_getsizehint(::AbstractGrid{dim}, ::Type{EdgeIndex}) where dim = dim^2
_getsizehint(::AbstractGrid{dim}, ::Type{VertexIndex}) where dim = 2^dim
_getsizehint(::AbstractGrid{1}, ::Type{CellIndex}) = 2
_getsizehint(::AbstractGrid{2}, ::Type{CellIndex}) = 12
function _getsizehint(g::AbstractGrid{3}, ::Type{CellIndex})
    CT = getcelltype(g)
    if isconcretetype(CT)
        RS = getrefshape(CT)
        RS === RefHexahedron && return 26
        RS === RefTetrahedron && return 70
    end
    return 70 # Assume that there are some RefTetrahedron
end

function getneighborhood(top::EntityTopology, grid::AbstractGrid, idx::FaceIndex)
    return top.faceneighbors[Face(grid, idx)]
end

function getneighborhood(top::EntityTopology, grid::AbstractGrid, idx::EdgeIndex)
    return top.edgeneighbors[Edge(grid, idx)]
end

function getneighborhood(top::EntityTopology, grid::AbstractGrid, idx::VertexIndex)
    return top.vertexneighbors[toglobal(grid, idx)]
end

function faceskeleton(top::EntityTopology, ::AbstractGrid)
    return (first(n) for n in nonempty_values(top.faceneighbors))
end

function edgeskeleton(top::EntityTopology, ::AbstractGrid)
    return (first(n) for n in nonempty_values(top.edgeneighbors))
end

function vertexskeleton(top::EntityTopology, ::AbstractGrid)
    return (first(n) for n in nonempty_values(top.vertexneighbors))
end

# =============================================================================================== #
# ExclusiveTopology and CoverTopology utils
# =============================================================================================== #
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
                add!(face_table, FaceIndex(cell_neighbor_id, lfi2), cell_id, lfi)
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
                add!(edge_table, EdgeIndex(cell_neighbor_id, lei2), cell_id, lei)
                return
            end
        end
    end
end

function _add_single_vertex_neighbor!(vertex_table::ConstructionBuffer, cell::AbstractCell, cell_id::Int, cell_neighbor::AbstractCell, cell_neighbor_id::Int)
    for (lvi, vertex) ∈ enumerate(vertices(cell))
        for (lvi2, vertex_neighbor) ∈ enumerate(vertices(cell_neighbor))
            if vertex_neighbor == vertex
                add!(vertex_table, VertexIndex(cell_neighbor_id, lvi2), cell_id, lvi)
                break
            end
        end
    end
end

function build_vertex_to_cell(cells; max_vertices, nnodes)
    vertex_to_cell = CollectionOfVectors(Vector, Int; sizehint=max_vertices, dims=(nnodes,)) do cov
            for (cellid, cell) in enumerate(cells)
                for vertex in vertices(cell)
                    add!(cov, cellid, vertex)
                end
            end
        end
    return vertex_to_cell
end

function build_cell_neighbor(grid, cells, vertex_to_cell; ncells)
    # Note: The following could be optimized, since we loop over the cells in order,
    # there is no need to use the special adaptive indexing and then compress_data! in CollectionOfVectors.
    return CollectionOfVectors(Vector, CellIndex; sizehint=_getsizehint(grid, CellIndex), dims=(ncells,)) do cov
            cell_neighbor_ids = Set{Int}()
            for (cell_id, cell) in enumerate(cells)
                empty!(cell_neighbor_ids)
                for vertex ∈ vertices(cell)
                    for vertex_cell_id ∈ vertex_to_cell[vertex]
                        if vertex_cell_id != cell_id
                            push!(cell_neighbor_ids, vertex_cell_id)
                        end
                    end
                end
                # TODO: At least "appending" values should be supported for cov::ConstructionBuffer
                for neighbor_id in cell_neighbor_ids
                    add!(cov, CellIndex(neighbor_id), cell_id)
                end
            end
        end
end


# =============================================================================================== #
# ExclusiveTopology2
# =============================================================================================== #
"ExclusiveTopology2 by using `CollectionOfVectors`"
struct ExclusiveTopology2
    # maps a global vertex id to all cells containing the vertex
    vertex_to_cell::CollectionOfVectors{Vector{UnitRange{Int}}, Int}
    # index of the vector = cell id ->  all other connected cells
    cell_neighbor::CollectionOfVectors{Vector{UnitRange{Int}}, CellIndex}
    # face_face_neighbor[cellid,local_face_id] -> exclusive connected entities (not restricted to one entity)
    face_face_neighbor::CollectionOfVectors{Matrix{UnitRange{Int}}, FaceIndex}
    # edge_edge_neighbor[cellid,local_edge_id] -> exclusive connected entities of the given edge
    edge_edge_neighbor::CollectionOfVectors{Matrix{UnitRange{Int}}, EdgeIndex}
    # vertex_vertex_neighbor[cellid,local_vertex_id] -> exclusive connected entities to the given vertex
    vertex_vertex_neighbor::CollectionOfVectors{Matrix{UnitRange{Int}}, VertexIndex}

    # list of unique faces in the grid given as FaceIndex
    face_skeleton::Union{Vector{FaceIndex}, Nothing}
    # list of unique edges in the grid given as EdgeIndex
    edge_skeleton::Union{Vector{FaceIndex}, Nothing}
    # list of unique vertices in the grid given as VertexIndex
    vertex_skeleton::Union{Vector{VertexIndex}, Nothing}
end

function ExclusiveTopology2(grid::AbstractGrid{sdim}) where sdim
    sdim == get_reference_dimension(grid) || error("ExclusiveTopology2 is only tested for non-embedded grids")
    cells = getcells(grid)
    nnodes = getnnodes(grid)
    ncells = length(cells)

    max_vertices, max_edges, max_faces = _max_nentities_per_cell(cells)
    vertex_to_cell = build_vertex_to_cell(cells; max_vertices, nnodes)
    cell_neighbor = build_cell_neighbor(grid, cells, vertex_to_cell; ncells)

    # Here we don't use the convenience constructor taking a function, since we want to do it simultaneously for 3 data-types
    # This also allows giving a sizehint to the underlying vectors
    facedata = sizehint!(FaceIndex[], ncells * max_faces * _getsizehint(grid, FaceIndex))
    face_face_neighbor_buf = ConstructionBuffer(Matrix, facedata; dims=(ncells, max_faces), sizehint=_getsizehint(grid, FaceIndex))
    edgedata = sizehint!(EdgeIndex[], ncells * max_edges * _getsizehint(grid, EdgeIndex))
    edge_edge_neighbor_buf = ConstructionBuffer(Matrix, edgedata; dims=(ncells, max_edges), sizehint=_getsizehint(grid, EdgeIndex))
    vertdata = sizehint!(VertexIndex[], ncells * max_vertices * _getsizehint(grid, VertexIndex))
    vertex_vertex_neighbor_buf = ConstructionBuffer(Matrix, vertdata; dims=(ncells, max_vertices), sizehint=_getsizehint(grid, VertexIndex))

    for (cell_id, cell) in enumerate(cells)
        for neighbor_cell_idx in cell_neighbor[cell_id]
            neighbor_cell_id = neighbor_cell_idx.idx
            neighbor_cell = cells[neighbor_cell_id]
            getrefdim(neighbor_cell) == getrefdim(cell) || error("Not supported")
            num_shared_vertices = _num_shared_vertices(cell, neighbor_cell) # See grid/topology.jl
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
    face_face_neighbor     = CollectionOfVectors(face_face_neighbor_buf)
    edge_edge_neighbor     = CollectionOfVectors(edge_edge_neighbor_buf)
    vertex_vertex_neighbor = CollectionOfVectors(vertex_vertex_neighbor_buf)
    return ExclusiveTopology2(vertex_to_cell, cell_neighbor, face_face_neighbor, edge_edge_neighbor, vertex_vertex_neighbor, nothing, nothing, nothing)
end

# =============================================================================================== #
# CoverTopology
# =============================================================================================== #
"CoverTopology (from FerriteDistributed) by using `CollectionOfVectors`"
struct CoverTopology
    # maps a global vertex id to all cells containing the vertex
    vertex_to_cell::CollectionOfVectors{Vector{UnitRange{Int}}, Int}
    # index of the vector = cell id ->  all other connected cells
    cell_neighbor::CollectionOfVectors{Vector{UnitRange{Int}}, CellIndex}
    # face_face_neighbor[cellid,local_face_id] -> exclusive connected entities (not restricted to one entity)
    face_face_neighbor::CollectionOfVectors{Matrix{UnitRange{Int}}, FaceIndex}
    # edge_edge_neighbor[cellid,local_edge_id] -> exclusive connected entities of the given edge
    edge_edge_neighbor::CollectionOfVectors{Matrix{UnitRange{Int}}, EdgeIndex}
    # vertex_vertex_neighbor[cellid,local_vertex_id] -> exclusive connected entities to the given vertex
    vertex_vertex_neighbor::CollectionOfVectors{Matrix{UnitRange{Int}}, VertexIndex}

    # list of unique faces in the grid given as FaceIndex
    face_skeleton::Union{Vector{FaceIndex}, Nothing}
    # list of unique edges in the grid given as EdgeIndex
    edge_skeleton::Union{Vector{FaceIndex}, Nothing}
    # list of unique vertices in the grid given as VertexIndex
    vertex_skeleton::Union{Vector{VertexIndex}, Nothing}
end

function CoverTopology(grid::AbstractGrid{sdim}) where sdim
    sdim == get_reference_dimension(grid) || error("CoverTopology is only tested for non-embedded grids")
    cells = getcells(grid)
    nnodes = getnnodes(grid)
    ncells = length(cells)

    max_vertices, max_edges, max_faces = _max_nentities_per_cell(cells)
    vertex_to_cell = build_vertex_to_cell(cells; max_vertices, nnodes)
    cell_neighbor = build_cell_neighbor(grid, cells, vertex_to_cell; ncells)

    # Here we don't use the convenience constructor taking a function, since we want to do it simultaneously for 3 data-types
    facedata = sizehint!(FaceIndex[], ncells * max_faces * _getsizehint(grid, FaceIndex))
    face_face_neighbor_buf = ConstructionBuffer(Matrix, facedata; dims=(ncells, max_faces), sizehint=_getsizehint(grid, FaceIndex))
    edgedata = sizehint!(EdgeIndex[], ncells * max_edges * _getsizehint(grid, EdgeIndex))
    edge_edge_neighbor_buf = ConstructionBuffer(Matrix, edgedata; dims=(ncells, max_edges), sizehint=_getsizehint(grid, EdgeIndex))
    vertdata = sizehint!(VertexIndex[], ncells * max_vertices * _getsizehint(grid, VertexIndex))
    vertex_vertex_neighbor_buf = ConstructionBuffer(Matrix, vertdata; dims=(ncells, max_vertices), sizehint=_getsizehint(grid, VertexIndex))

    for (cell_id, cell) in enumerate(cells)
        for neighbor_cell_idx in cell_neighbor[cell_id]
            neighbor_cell_id = neighbor_cell_idx.idx
            neighbor_cell = cells[neighbor_cell_id]
            getrefdim(neighbor_cell) == getrefdim(cell) || error("Not supported")
            num_shared_vertices = _num_shared_vertices(cell, neighbor_cell) # See grid/topology.jl
            if num_shared_vertices ≥ 1 # Shared vertex
                _add_single_vertex_neighbor!(vertex_vertex_neighbor_buf, cell, cell_id, neighbor_cell, neighbor_cell_id)
            end
            if num_shared_vertices ≥ 2 # Shared edge
                _add_single_edge_neighbor!(edge_edge_neighbor_buf, cell, cell_id, neighbor_cell, neighbor_cell_id)
            end
            # Shared face
            if num_shared_vertices ≥ 3 # Shared face
                _add_single_face_neighbor!(face_face_neighbor_buf, cell, cell_id, neighbor_cell, neighbor_cell_id)
            end
            if num_shared_vertices ≤ 0
                error("Found connected elements without shared vertex... Mesh broken?")
            end
        end
    end
    face_face_neighbor     = CollectionOfVectors(face_face_neighbor_buf)
    edge_edge_neighbor     = CollectionOfVectors(edge_edge_neighbor_buf)
    vertex_vertex_neighbor = CollectionOfVectors(vertex_vertex_neighbor_buf)
    return CoverTopology(vertex_to_cell, cell_neighbor, face_face_neighbor, edge_edge_neighbor, vertex_vertex_neighbor, nothing, nothing, nothing)
end
