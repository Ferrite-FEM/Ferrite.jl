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

Base.hash(x::Union{Face, Edge}, h::UInt) = hash(x.vertices, h)
Base.isequal(x::E, y::E) where {E <: Union{Face, Edge}} = x.vertices == y.vertices

# Helper to keep track of the current allocated space in `neighbors::Vector`
# in GlobalNeighborInformation during construction.
struct AdaptiveRange
    start::Int
    ncurrent::Int   # Could be UInt8
    nmax::Int       # Could be UInt8
end

struct GlobalNeighborInformation{ET, BI, CT}
    neighbors::Vector{BI}
    indices::CT             # ngbs = neighbors[indices[item]], where item is the unique representation of the entity.
    function GlobalNeighborInformation(neighbors::Vector{VertexIndex}, indices::Vector{UnitRange{Int}})
        return new{Int, VertexIndex, typeof(indices)}(neighbors, indices)
    end
    function GlobalNeighborInformation(neighbors::Vector{BI}, indices::OrderedDict{ET, UnitRange{Int}}) where {BI<:BoundaryIndex, ET}
        return new{ET, BI, typeof(indices)}(neighbors, indices)
    end
end

getneighbors(gni::GlobalNeighborInformation{ET}, idx::ET) where ET = (gni.neighbors[i] for i in gni.indices[idx])

_getsizehint(::AbstractGrid, ::Type{FaceIndex}) = 2
_getsizehint(::AbstractGrid{dim}, ::Type{EdgeIndex}) where dim = dim^2
_getsizehint(::AbstractGrid{dim}, ::Type{VertexIndex}) where dim = 2^dim

function GlobalNeighborInformation(grid, IdxType::Type{<:BoundaryIndex}, ::Type{ET}; sizehint=_getsizehint(grid, IdxType)) where {ET<:Union{Edge, Face}}
    gni = GlobalNeighborInformation(IdxType[], OrderedDict{ET, UnitRange{Int}}())
    indices_buffer = OrderedDict{ET, AdaptiveRange}()
    # Fill the information
    for (cellnr, cell) in enumerate(getcells(grid))
        for (entitynr, entity_vertices) in enumerate(boundaryfunction(IdxType)(cell))
            e = ET(entity_vertices)
            addneighbor!(gni, indices_buffer, e, IdxType(cellnr, entitynr), sizehint)
        end
    end
    # Compress the information by shifting all items in the vector to the beginning, and deleting unused space.
    compress_data!(gni, indices_buffer)
    return gni
end

function GlobalNeighborInformation(grid, IdxType::Type{VertexIndex}, ::Type{Int}; sizehint=_getsizehint(grid, IdxType))
    indices = Vector{UnitRange{Int}}(undef, getnnodes(grid))
    gni = GlobalNeighborInformation(IdxType[], indices)
    indices_buffer = fill(AdaptiveRange(0, 0, 0), getnnodes(grid))
    # Fill the information
    for (cellnr, cell) in enumerate(getcells(grid))
        for (vertexnr, global_vertex) in enumerate(vertices(cell))
            addneighbor!(gni, indices_buffer, global_vertex, VertexIndex(cellnr, vertexnr), sizehint)
        end
    end
    # Compress the information
    compress_data!(gni, indices_buffer)
    return gni
end

function addneighbor!(gni::GlobalNeighborInformation{ET}, indices_buffer, item::ET, idx::BoundaryIndex, sizehint::Int) where {ET<:Union{Edge, Face}}
    n = length(gni.neighbors)
    added_range = AdaptiveRange(n + 1, 1, sizehint)
    r = get!(indices_buffer, item) do
        # Enters only if item is not in indices_buffer
        resize!(gni.neighbors, n + sizehint)
        gni.neighbors[n+1] = idx
        added_range
    end
    r === added_range && return gni # We added a new unique entity, can exit
    # Otherwise, we need to add more neighbors to an existing entity:

    if r.ncurrent == r.nmax # Need to move to the end of the vector
        indices_buffer[item] = AdaptiveRange(n + 1, r.ncurrent + 1, r.nmax + sizehint)
        resize!(gni.neighbors, n + r.nmax + sizehint)
        for i in 1:r.ncurrent # TODO: Iterator for AdaptiveRange
            gni.neighbors[n + i] = gni.neighbors[r.start + i - 1]
        end
        gni.neighbors[n + r.ncurrent + 1] = idx
    else
        indices_buffer[item] = AdaptiveRange(r.start, r.ncurrent + 1, r.nmax)
        gni.neighbors[r.start + r.ncurrent] = idx
    end
    return gni
end

function compress_data!(gni::GlobalNeighborInformation{ET}, indices_buffer) where {ET<:Union{Edge, Face}}
    # indices_buffer values are of type AdaptiveRange and these are not overlapping. Sort by their first value.
    sort!(indices_buffer; byvalue=true, by = r -> r.start)
    sizehint!(gni.indices, length(indices_buffer))
    # NOTE: gni.indices and indices_buffer have the same keys, this could probably be optimized
    # as rehash etc. is taking some time...
    n = 0
    cnt = 1
    for (entity, r) in indices_buffer
        nstop = r.start + r.ncurrent - 1
        for (iold, inew) in zip(nstop:-1:r.start, n .+ (r.ncurrent:-1:1))
            @assert inew ≤ iold # To not overwrite
            gni.neighbors[inew] = gni.neighbors[iold]
        end
        gni.indices[entity] = (n + 1):(n + r.ncurrent)
        n += r.ncurrent
        cnt += 1
    end
    empty!(indices_buffer)
    resize!(gni.neighbors, n)
    return gni
end

function addneighbor!(gni::GlobalNeighborInformation{Int}, indices_buffer, item::Int, idx::VertexIndex, sizehint::Int)
    r = indices_buffer[item]
    n = length(gni.neighbors)
    if r.start == 0 # Not previously added
        resize!(gni.neighbors, n + sizehint)
        gni.neighbors[n+1] = idx
        indices_buffer[item] = AdaptiveRange(n + 1, 1, sizehint)
    elseif r.ncurrent == r.nmax # We have used up our space, move data to the end of the vector.
        resize!(gni.neighbors, n + r.nmax + sizehint)
        for i in 1:r.ncurrent
            gni.neighbors[n + i] = gni.neighbors[r.start + i - 1]
        end
        gni.neighbors[n + r.ncurrent + 1] = idx
        indices_buffer[item] = AdaptiveRange(n + 1, r.ncurrent + 1, r.nmax + sizehint)
    else # We have space in an already allocated section
        gni.neighbors[r.start + r.ncurrent] = idx
        indices_buffer[item] = AdaptiveRange(r.start, r.ncurrent + 1, r.nmax)
    end
    return gni
end

function compress_data!(gni::GlobalNeighborInformation{Int}, indices_buffer)
    # indices_buffer contain AdaptiveRange and these are not overlapping. Sort by their first value.
    sort!(indices_buffer; by = r -> r.start)
    n = 0
    for (entity, r) in enumerate(indices_buffer)
        r.start == 0 && continue # E.g. a node not being a vertex.
        nstop = r.start + r.ncurrent - 1
        for (iold, inew) in zip(nstop:-1:r.start, n .+ (r.ncurrent:-1:1))
            @assert inew ≤ iold # To not overwrite
            gni.neighbors[inew] = gni.neighbors[iold]
        end
        gni.indices[entity] = (n + 1):(n + r.ncurrent)
        n += r.ncurrent
    end
    empty!(indices_buffer)
    resize!(gni.neighbors, n)
    return gni
end

struct MaterializedTopology <: AbstractTopology
    faceneighbors::GlobalNeighborInformation{Face, FaceIndex, OrderedDict{Face, UnitRange{Int}}}
    edgesneighbors::GlobalNeighborInformation{Edge, EdgeIndex, OrderedDict{Edge, UnitRange{Int}}}
    vertexneighbors::GlobalNeighborInformation{Int, VertexIndex, Vector{UnitRange{Int}}}
end

function MaterializedTopology(grid::AbstractGrid)
    return MaterializedTopology(
        GlobalNeighborInformation(grid,   FaceIndex, Face), #TODO: Skip in 1d and 2d
        GlobalNeighborInformation(grid,   EdgeIndex, Edge), #TODO: Skip in 1d
        GlobalNeighborInformation(grid, VertexIndex, Int))
end

function getneighborhood(top::MaterializedTopology, grid::AbstractGrid, idx::FaceIndex)
    return getneighbors(top.faceneighbors, Face(grid, idx))
end

function getneighborhood(top::MaterializedTopology, grid::AbstractGrid, idx::EdgeIndex)
    return getneighbors(top.edgeneighbors, Edge(grid, idx))
end

function getneighborhood(top::MaterializedTopology, grid::AbstractGrid, idx::VertexIndex)
    return getneighbors(top.vertexneighbors, toglobal(grid, idx))
end
