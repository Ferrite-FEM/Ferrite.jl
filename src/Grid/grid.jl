#########################
# Main types for meshes #
#########################
"""
    Node{dim, T}

A `Node` is a point in space.

# Fields
- `x::Vec{dim,T}`: stores the coordinates
"""
struct Node{dim,T}
    x::Vec{dim,T}
end
Node(x::NTuple{dim,T}) where {dim,T} = Node(Vec{dim,T}(x))

"""
    get_node_coordinate(::Node)
    
Get the value of the node coordinate.
"""
get_node_coordinate(n::Node) = n.x

"""
    get_coordinate_type(::Node)

Get the data type of the the node coordinate.
"""
get_coordinate_type(::Node{dim,T}) where {dim,T}  = Vec{dim,T}

"""
    get_coordinate_eltype(::Node)

Get the data type of the components of the nodes coordinate.
"""
get_coordinate_eltype(::Node{dim,T}) where {dim,T} = T

##########################
# AbstractCell interface #
##########################

abstract type AbstractCell{refshape <: AbstractRefShape} end

nvertices(c::AbstractCell) = length(vertices(c))
nedges(   c::AbstractCell) = length(edges(c))
nfaces(   c::AbstractCell) = length(faces(c))
nnodes(   c::AbstractCell) = length(get_node_ids(c))

"""
    Ferrite.vertices(::AbstractCell)

Returns a tuple with the node indices (of the nodes in a grid) for each vertex in a given cell.
This function induces the [`VertexIndex`](@ref), where the second index 
corresponds to the local index into this tuple.
"""
vertices(::AbstractCell)

"""
    Ferrite.edges(::AbstractCell)

Returns a tuple of 2-tuples containing the ordered node indices (of the nodes in a grid) corresponding to
the vertices that define an *oriented edge*. This function induces the 
[`EdgeIndex`](@ref), where the second index corresponds to the local index into this tuple.

Note that the vertices are sufficient to define an edge uniquely.
"""
edges(::AbstractCell)

"""
    Ferrite.faces(::AbstractCell)
    Ferrite.faces(::Type{AbstractRefShape})

Returns a tuple of n-tuples containing the ordered node indices (of the nodes in a grid, or reference element if `AbstractRefShape` is passed) corresponding to
the vertices that define an *oriented face*. This function induces the 
[`FaceIndex`](@ref), where the second index corresponds to the local index into this tuple.

Note that the vertices are sufficient to define a face uniquely.
"""
faces(::AbstractCell)

"""
    Ferrite.default_interpolation(::AbstractCell)::Interpolation

Returns the interpolation which defines the geometry of a given cell.
"""
default_interpolation(::AbstractCell)

"""
    Ferrite.get_node_ids(c::AbstractCell)

Return the node id's for cell `c` in the order determined by the cell's reference cell.

Default implementation: `c.nodes`.
"""
get_node_ids(c::AbstractCell) = c.nodes

# Default implementations of vertices/edges/faces that work as long as get_node_ids is
# correctly implemented for the cell.

# RefLine (refdim = 1): vertices for vertexdofs, faces for BC
function vertices(c::AbstractCell{RefLine})
    ns = get_node_ids(c)
    return (ns[1], ns[2]) # v1, v2
end
function faces(c::AbstractCell{RefLine})
    ns = get_node_ids(c)
    return ((ns[1],), (ns[2],)) # f1, f2
end
function faces(::Type{RefLine})
    return ((1), (2,)) # f1, f2
end

# RefTriangle (refdim = 2): vertices for vertexdofs, faces for facedofs (edgedofs) and BC
function vertices(c::AbstractCell{RefTriangle})
    ns = get_node_ids(c)
    return (ns[1], ns[2], ns[3]) # v1, v2, v3
end
function faces(c::AbstractCell{RefTriangle})
    ns = get_node_ids(c)
    return (
        (ns[1], ns[2]), (ns[2], ns[3]), (ns[3], ns[1]), # f1, f2, f3
    )
end
function faces(::Type{RefTriangle})
    return (
        (1, 2), (2, 3), (3, 1), # f1, f2, f3
    )
end

# RefQuadrilateral (refdim = 2): vertices for vertexdofs, faces for facedofs (edgedofs) and BC
function vertices(c::AbstractCell{RefQuadrilateral})
    ns = get_node_ids(c)
    return (ns[1], ns[2], ns[3], ns[4]) # v1, v2, v3, v4
end
function faces(c::AbstractCell{RefQuadrilateral})
    ns = get_node_ids(c)
    return (
        (ns[1], ns[2]), (ns[2], ns[3]), (ns[3], ns[4]), (ns[4], ns[1]), # f1, f2, f3, f4
    )
end
function faces(::Type{RefQuadrilateral})
    return (
        (1, 2), (2, 3), (3, 4), (4, 1), # f1, f2, f3, f4
    )
end

# RefTetrahedron (refdim = 3): vertices for vertexdofs, edges for edgedofs, faces for facedofs and BC
function vertices(c::AbstractCell{RefTetrahedron})
    ns = get_node_ids(c)
    return (ns[1], ns[2], ns[3], ns[4]) # v1, v2, v3, v4
end
function edges(c::AbstractCell{RefTetrahedron})
    ns = get_node_ids(c)
    return (
        (ns[1], ns[2]), (ns[2], ns[3]), (ns[3], ns[1]), # e1, e2, e3
        (ns[1], ns[4]), (ns[2], ns[4]), (ns[3], ns[4]), # e4, e5, e6
    )
end
function faces(c::AbstractCell{RefTetrahedron})
    ns = get_node_ids(c)
    return (
        (ns[1], ns[3], ns[2]), (ns[1], ns[2], ns[4]), # f1, f2
        (ns[2], ns[3], ns[4]), (ns[1], ns[4], ns[3]), # f3, f4
    )
end
function faces(::Type{RefTetrahedron})
    return (
        (1, 3, 2), (1, 2, 4), # f1, f2
        (2, 3, 4), (1, 4, 3), # f3, f4
    )
end

# RefHexahedron (refdim = 3): vertices for vertexdofs, edges for edgedofs, faces for facedofs and BC
function vertices(c::AbstractCell{RefHexahedron})
    ns = get_node_ids(c)
    return (
        ns[1], ns[2], ns[3], ns[4], ns[5], ns[6], ns[7], ns[8], # v1, ..., v8
    )
end
function edges(c::AbstractCell{RefHexahedron})
    ns = get_node_ids(c)
    return (
        (ns[1], ns[2]), (ns[2], ns[3]), (ns[3], ns[4]), (ns[4], ns[1]), # e1, e2, e3, e4
        (ns[5], ns[6]), (ns[6], ns[7]), (ns[7], ns[8]), (ns[8], ns[5]), # e5, e6, e7, e8
        (ns[1], ns[5]), (ns[2], ns[6]), (ns[3], ns[7]), (ns[4], ns[8]), # e9, e10, e11, e12
    )
end
function faces(c::AbstractCell{RefHexahedron})
    ns = get_node_ids(c)
    return (
        (ns[1], ns[4], ns[3], ns[2]), (ns[1], ns[2], ns[6], ns[5]), # f1, f2
        (ns[2], ns[3], ns[7], ns[6]), (ns[3], ns[4], ns[8], ns[7]), # f3, f4
        (ns[1], ns[5], ns[8], ns[4]), (ns[5], ns[6], ns[7], ns[8]), # f5, f6
    )
end
function faces(::Type{RefHexahedron})
    return (
        (1, 4, 3, 2), (1, 2, 6, 5), # f1, f2
        (2, 3, 7, 6), (3, 4, 8, 7), # f3, f4
        (1, 5, 8, 4), (5, 6, 7, 8), # f5, f6
    )
end

# RefPrism (refdim = 3): vertices for vertexdofs, edges for edgedofs, faces for facedofs and BC
function vertices(c::AbstractCell{RefPrism})
    ns = get_node_ids(c)
    return (ns[1], ns[2], ns[3], ns[4], ns[5], ns[6]) # v1, ..., v6
end
function edges(c::AbstractCell{RefPrism})
    ns = get_node_ids(c)
    return (
        (ns[2], ns[1]), (ns[1], ns[3]), (ns[1], ns[4]), (ns[3], ns[2]), # e1, e2, e3, e4
        (ns[2], ns[5]), (ns[3], ns[6]), (ns[4], ns[5]), (ns[4], ns[6]), # e5, e6, e7, e8
        (ns[6], ns[5]),                                                 # e9
    )
end
function faces(c::AbstractCell{RefPrism})
    ns = get_node_ids(c)
    return (
        (ns[1], ns[3], ns[2]),        (ns[1], ns[2], ns[5], ns[4]), # f1, f2
        (ns[3], ns[1], ns[4], ns[6]), (ns[2], ns[3], ns[6], ns[5]), # f3, f4
        (ns[4], ns[5], ns[6]),                                      # f5
    )
end
function faces(::Type{RefPrism})
    return (
        (1, 3, 2),        (1, 2, 5, 4), # f1, f2
        (3, 1, 4, 6), (2, 3, 6, 5), # f3, f4
        (4, 5, 6),                                      # f5
    )
end

######################################################
# Concrete implementations of AbstractCell interface #
######################################################

# Lagrange interpolation based cells
struct Line                   <: AbstractCell{RefLine}          nodes::NTuple{ 2, Int} end
struct QuadraticLine          <: AbstractCell{RefLine}          nodes::NTuple{ 3, Int} end
struct Triangle               <: AbstractCell{RefTriangle}      nodes::NTuple{ 3, Int} end
struct QuadraticTriangle      <: AbstractCell{RefTriangle}      nodes::NTuple{ 6, Int} end
struct Quadrilateral          <: AbstractCell{RefQuadrilateral} nodes::NTuple{ 4, Int} end
struct QuadraticQuadrilateral <: AbstractCell{RefQuadrilateral} nodes::NTuple{ 9, Int} end
struct Tetrahedron            <: AbstractCell{RefTetrahedron}   nodes::NTuple{ 4, Int} end
struct QuadraticTetrahedron   <: AbstractCell{RefTetrahedron}   nodes::NTuple{10, Int} end
struct Hexahedron             <: AbstractCell{RefHexahedron}    nodes::NTuple{ 8, Int} end
struct QuadraticHexahedron    <: AbstractCell{RefHexahedron}    nodes::NTuple{27, Int} end
struct Wedge                  <: AbstractCell{RefPrism}         nodes::NTuple{ 6, Int} end

default_interpolation(::Type{Line})                   = Lagrange{RefLine,          1}()
default_interpolation(::Type{QuadraticLine})          = Lagrange{RefLine,          2}()
default_interpolation(::Type{Triangle})               = Lagrange{RefTriangle,      1}()
default_interpolation(::Type{QuadraticTriangle})      = Lagrange{RefTriangle,      2}()
default_interpolation(::Type{Quadrilateral})          = Lagrange{RefQuadrilateral, 1}()
default_interpolation(::Type{QuadraticQuadrilateral}) = Lagrange{RefQuadrilateral, 2}()
default_interpolation(::Type{Tetrahedron})            = Lagrange{RefTetrahedron,   1}()
default_interpolation(::Type{QuadraticTetrahedron})   = Lagrange{RefTetrahedron,   2}()
default_interpolation(::Type{Hexahedron})             = Lagrange{RefHexahedron,    1}()
default_interpolation(::Type{QuadraticHexahedron})    = Lagrange{RefHexahedron,    2}()
default_interpolation(::Type{Wedge})                  = Lagrange{RefPrism,         1}()

# TODO: Remove this, used for Quadrilateral3D
edges(c::Quadrilateral#=3D=#) = faces(c)

# Serendipity interpolation based cells
struct SerendipityQuadraticQuadrilateral <: AbstractCell{RefQuadrilateral} nodes::NTuple{ 8, Int} end
struct SerendipityQuadraticHexahedron    <: AbstractCell{RefHexahedron}    nodes::NTuple{20, Int} end

default_interpolation(::Type{SerendipityQuadraticQuadrilateral}) = Serendipity{RefQuadrilateral, 2}()
default_interpolation(::Type{SerendipityQuadraticHexahedron})    = Serendipity{RefHexahedron,    2}()


############
# Topology #
############
# TODO: Move topology stuff to src/Grid/Topology.jl or something

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

"""
    nvertices_on_face(cell::AbstractCell, local_face_index::Int)
Specifies for each subtype of AbstractCell how many nodes form a face.
"""
nvertices_on_face(cell::AbstractCell, local_face_index::Int) = length(faces(cell)[local_face_index])
"""
    nvertices_on_edge(::AbstractCell, local_edge_index::Int)
Specifies for each subtype of AbstractCell how many nodes form an edge.
"""
nvertices_on_edge(cell::AbstractCell, local_edge_index::Int) = length(edges(cell)[local_edge_index])

getdim(::Union{AbstractCell{refshape},Type{<:AbstractCell{refshape}}}) where {refdim, refshape <: AbstractRefShape{refdim}} = refdim

abstract type AbstractTopology end

"""
    ExclusiveTopology(cells::Vector{C}) where C <: AbstractCell
`ExclusiveTopology` saves topological (connectivity) data of the grid. The constructor works with an `AbstractCell`
vector for all cells that dispatch `vertices`, `faces` and in 3D `edges`.
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
    vertex_to_cell::Dict{Int,Set{Int}}
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

    I_face = Int[]; J_face = Int[]; V_face = EntityNeighborhood[]
    I_edge = Int[]; J_edge = Int[]; V_edge = EntityNeighborhood[]
    I_vertex = Int[]; J_vertex = Int[]; V_vertex = EntityNeighborhood[]   
    cell_neighbor_table = Vector{EntityNeighborhood{CellIndex}}(undef, length(cells)) 

    for (cellid, cell) in enumerate(cells)
        cell_neighbors = reduce(union!, [Set{Int}(vertex_cell_table[vertex]) for vertex ∈ vertices(cell) if vertex_cell_table[vertex] != cellid])
        cell_neighbor_table[cellid] = EntityNeighborhood(CellIndex.(collect(cell_neighbors)))

        face_neighbors = Set{Int}()
        for (face_idx,face) ∈ enumerate(faces(cell))
            neighbor_candidates = Set{Int}(c for c ∈ vertex_cell_table[face[1]] if c != cellid)
            for face_vertex ∈ face[2:end]
                intersect!(neighbor_candidates, vertex_cell_table[face_vertex])
            end
            union!(face_neighbors, neighbor_candidates)
        end

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

        for neighbor_cellid in cell_neighbors
            cell_local_ids = findall(x->x in cell_vertices_table[neighbor_cellid], cell_vertices_table[cellid])
            # vertex neighbor
            if length(cell_local_ids) == 1
                neighbor_local_ids = findall(x->x in cell_vertices_table[cellid], cell_vertices_table[neighbor_cellid])
                _vertex_neighbor!(V_vertex, I_vertex, J_vertex, cellid, cell, neighbor_local_ids, neighbor_cellid, cells[neighbor_cellid])
            # face neighbor
            elseif neighbor_cellid ∈ face_neighbors
                neighbor_local_ids = findall(x->x in cell_vertices_table[cellid], cell_vertices_table[neighbor_cellid])
                _face_neighbor!(V_face, I_face, J_face, cellid, cell, neighbor_local_ids, neighbor_cellid, cells[neighbor_cellid]) 
            # edge neighbor
            elseif getdim(cell) > 2 && neighbor_cellid ∈ edge_neighbors
                neighbor_local_ids = findall(x->x in cell_vertices_table[cellid], cell_vertices_table[neighbor_cellid])
                _edge_neighbor!(V_edge, I_edge, J_edge, cellid, cell, neighbor_local_ids, neighbor_cellid, cells[neighbor_cellid])
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
            neighbor_boundary = getdim(cells[cell]) > 2 ? collect(edges(cells[cell])) : collect(faces(cells[cell])) #get lowest dimension boundary
            neighbor_connected_faces = neighbor_boundary[findall(x->global_vertexid in x, neighbor_boundary)]
            other_vertices = findfirst.(x->x!=global_vertexid,neighbor_connected_faces)
            any(other_vertices .=== nothing) && continue 
            neighbor_vertices_global = getindex.(neighbor_connected_faces, other_vertices)
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
            push!(face_skeleton_global, first(sortface(face)))
            fs_length_new = length(face_skeleton_global)
            if fs_length != fs_length_new
                push!(face_skeleton_local, FaceIndex(cellid,local_face_id)) 
                fs_length = fs_length_new
            end
        end
    end
    return ExclusiveTopology(vertex_cell_table,cell_neighbor_table,face_neighbor,vertex_neighbor,edge_neighbor,vertex_vertex_table,face_skeleton_local)
end

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

getcells(neighbor::EntityNeighborhood{T}) where T <: BoundaryIndex = first.(neighbor.neighbor_info)
getcells(neighbor::EntityNeighborhood{CellIndex}) = getproperty.(neighbor.neighbor_info, :idx)
getcells(neighbors::Vector{T}) where T <: EntityNeighborhood = reduce(vcat, getcells.(neighbors))
getcells(neighbors::Vector{T}) where T <: BoundaryIndex = getindex.(neighbors,1)

abstract type AbstractGrid{dim} end

ExclusiveTopology(grid::AbstractGrid) = ExclusiveTopology(getcells(grid))

"""
    Grid{dim, C<:AbstractCell, T<:Real} <: AbstractGrid}

A `Grid` is a collection of `Cells` and `Node`s which covers the computational domain, together with Sets of cells, nodes and faces.
There are multiple helper structures to apply boundary conditions or define subdomains. They are gathered in the `cellsets`, `nodesets`,
`facesets`, `edgesets` and `vertexsets`. 

# Fields
- `cells::Vector{C}`: stores all cells of the grid
- `nodes::Vector{Node{dim,T}}`: stores the `dim` dimensional nodes of the grid
- `cellsets::Dict{String,Set{Int}}`: maps a `String` key to a `Set` of cell ids
- `nodesets::Dict{String,Set{Int}}`: maps a `String` key to a `Set` of global node ids
- `facesets::Dict{String,Set{FaceIndex}}`: maps a `String` to a `Set` of `Set{FaceIndex} (global_cell_id, local_face_id)`
- `edgesets::Dict{String,Set{EdgeIndex}}`: maps a `String` to a `Set` of `Set{EdgeIndex} (global_cell_id, local_edge_id` 
- `vertexsets::Dict{String,Set{VertexIndex}}`: maps a `String` key to a `Set` of local vertex ids
- `boundary_matrix::SparseMatrixCSC{Bool,Int}`: optional, only needed by `onboundary` to check if a cell is on the boundary, see, e.g. Helmholtz example
"""
mutable struct Grid{dim,C<:AbstractCell,T<:Real} <: AbstractGrid{dim}
    cells::Vector{C}
    nodes::Vector{Node{dim,T}}
    # Sets
    cellsets::Dict{String,Set{Int}}
    nodesets::Dict{String,Set{Int}}
    facesets::Dict{String,Set{FaceIndex}} 
    edgesets::Dict{String,Set{EdgeIndex}} 
    vertexsets::Dict{String,Set{VertexIndex}} 
    # Boundary matrix (faces per cell × cell)
    boundary_matrix::SparseMatrixCSC{Bool,Int}
end

function Grid(cells::Vector{C},
              nodes::Vector{Node{dim,T}};
              cellsets::Dict{String,Set{Int}}=Dict{String,Set{Int}}(),
              nodesets::Dict{String,Set{Int}}=Dict{String,Set{Int}}(),
              facesets::Dict{String,Set{FaceIndex}}=Dict{String,Set{FaceIndex}}(),
              edgesets::Dict{String,Set{EdgeIndex}}=Dict{String,Set{EdgeIndex}}(),
              vertexsets::Dict{String,Set{VertexIndex}}=Dict{String,Set{VertexIndex}}(),
              boundary_matrix::SparseMatrixCSC{Bool,Int}=spzeros(Bool, 0, 0)) where {dim,C,T}
    return Grid(cells, nodes, cellsets, nodesets, facesets, edgesets, vertexsets, boundary_matrix)
end

##########################
# Grid utility functions #
##########################
"""
    get_coordinate_type(::AbstractGrid)

Get the datatype for a single point in the grid.
"""
get_coordinate_type(grid::Grid{dim,C,T}) where {dim,C,T} = Vec{dim,T} # Node is baked into the mesh type.

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

"""
    toglobal(grid::AbstractGrid, vertexidx::VertexIndex) -> Int
    toglobal(grid::AbstractGrid, vertexidx::Vector{VertexIndex}) -> Vector{Int}
This function takes the local vertex representation (a `VertexIndex`) and looks up the unique global id (an `Int`).
"""
toglobal(grid::AbstractGrid,vertexidx::VertexIndex) = vertices(getcells(grid,vertexidx[1]))[vertexidx[2]]
toglobal(grid::AbstractGrid,vertexidx::Vector{VertexIndex}) = unique(toglobal.((grid,),vertexidx))

@inline getdim(::AbstractGrid{dim}) where {dim} = dim
"""
    getcells(grid::AbstractGrid) 
    getcells(grid::AbstractGrid, v::Union{Int,Vector{Int}} 
    getcells(grid::AbstractGrid, setname::String)

Returns either all `cells::Collection{C<:AbstractCell}` of a `<:AbstractGrid` or a subset based on an `Int`, `Vector{Int}` or `String`.
Whereas the last option tries to call a `cellset` of the `grid`. `Collection` can be any indexable type, for `Grid` it is `Vector{C<:AbstractCell}`.
"""
@inline getcells(grid::AbstractGrid) = grid.cells
@inline getcells(grid::AbstractGrid, v::Union{Int, Vector{Int}}) = grid.cells[v]
@inline getcells(grid::AbstractGrid, setname::String) = grid.cells[collect(getcellset(grid,setname))]
"Returns the number of cells in the `<:AbstractGrid`."
@inline getncells(grid::AbstractGrid) = length(grid.cells)
"Returns the celltype of the `<:AbstractGrid`."
@inline getcelltype(grid::AbstractGrid) = eltype(grid.cells)
@inline getcelltype(grid::AbstractGrid, i::Int) = typeof(grid.cells[i])

"""
    getnodes(grid::AbstractGrid) 
    getnodes(grid::AbstractGrid, v::Union{Int,Vector{Int}}
    getnodes(grid::AbstractGrid, setname::String)

Returns either all `nodes::Collection{N}` of a `<:AbstractGrid` or a subset based on an `Int`, `Vector{Int}` or `String`.
The last option tries to call a `nodeset` of the `<:AbstractGrid`. `Collection{N}` refers to some indexable collection where each element corresponds
to a Node.
"""
@inline getnodes(grid::AbstractGrid) = grid.nodes
@inline getnodes(grid::AbstractGrid, v::Union{Int, Vector{Int}}) = grid.nodes[v]
@inline getnodes(grid::AbstractGrid, setname::String) = grid.nodes[collect(getnodeset(grid,setname))]
"Returns the number of nodes in the grid."
@inline getnnodes(grid::AbstractGrid) = length(grid.nodes)
"Returns the number of nodes of the `i`-th cell."
@inline nnodes_per_cell(grid::AbstractGrid, i::Int=1) = nnodes(grid.cells[i])

get_node_coordinate(grid, nodeid) = get_node_coordinate(getnodes(grid, nodeid))
"Return the number type of the nodal coordinates."
@inline get_coordinate_eltype(grid::AbstractGrid) = get_coordinate_eltype(first(getnodes(grid)))

"""
    getcellset(grid::AbstractGrid, setname::String)

Returns all cells as cellid in a `Set` of a given `setname`.
"""
@inline getcellset(grid::AbstractGrid, setname::String) = grid.cellsets[setname]
"""
    getcellsets(grid::AbstractGrid)

Returns all cellsets of the `grid`.
"""
@inline getcellsets(grid::AbstractGrid) = grid.cellsets

"""
    getnodeset(grid::AbstractGrid, setname::String)

Returns all nodes as nodeid in a `Set` of a given `setname`.
"""
@inline getnodeset(grid::AbstractGrid, setname::String) = grid.nodesets[setname]
"""
    getnodesets(grid::AbstractGrid)

Returns all nodesets of the `grid`.
"""
@inline getnodesets(grid::AbstractGrid) = grid.nodesets

"""
    getfaceset(grid::AbstractGrid, setname::String)

Returns all faces as `FaceIndex` in a `Set` of a given `setname`.
"""
@inline getfaceset(grid::AbstractGrid, setname::String) = grid.facesets[setname]
"""
    getfacesets(grid::AbstractGrid)

Returns all facesets of the `grid`.
"""
@inline getfacesets(grid::AbstractGrid) = grid.facesets

"""
    getedgeset(grid::AbstractGrid, setname::String)

Returns all edges as `EdgeIndex` in a `Set` of a given `setname`.
"""
@inline getedgeset(grid::AbstractGrid, setname::String) = grid.edgesets[setname]
"""
    getedgesets(grid::AbstractGrid)

Returns all edge sets of the grid.
"""
@inline getedgesets(grid::AbstractGrid) = grid.edgesets

"""
    getedgeset(grid::AbstractGrid, setname::String)

Returns all vertices as `VertexIndex` in a `Set` of a given `setname`.
"""
@inline getvertexset(grid::AbstractGrid, setname::String) = grid.vertexsets[setname]
"""
    getvertexsets(grid::AbstractGrid)

Returns all vertex sets of the grid.
"""
@inline getvertexsets(grid::AbstractGrid) = grid.vertexsets

n_faces_per_cell(grid::Grid) = nfaces(getcelltype(grid))

# Transformations
"""
    transform!(grid::Abstractgrid, f::Function)

Transform all nodes of the `grid` based on some transformation function `f`.
"""
function transform!(g::Grid, f::Function)
    map!(n -> Node(f(get_node_coordinate(n))), g.nodes, g.nodes)
    return g
end

# Sets

_check_setname(dict, name) = haskey(dict, name) && throw(ArgumentError("there already exists a set with the name: $name"))
_warn_emptyset(set, name) = length(set) == 0 && @warn("no entities added to the set with name: $name")

"""
    addcellset!(grid::AbstractGrid, name::String, cellid::Union{Set{Int}, Vector{Int}})
    addcellset!(grid::AbstractGrid, name::String, f::function; all::Bool=true)

Adds a cellset to the grid with key `name`.
Cellsets are typically used to define subdomains of the problem, e.g. two materials in the computational domain.
The `DofHandler` can construct different fields which live not on the whole domain, but rather on a cellset.
`all=true` implies that `f(x)` must return `true` for all nodal coordinates `x` in the cell if the cell
should be added to the set, otherwise it suffices that `f(x)` returns `true` for one node. 

```julia
addcellset!(grid, "left", Set((1,3))) #add cells with id 1 and 3 to cellset left
addcellset!(grid, "right", x -> norm(x[1]) < 2.0 ) #add cell to cellset right, if x[1] of each cell's node is smaller than 2.0
```
"""
function addcellset!(grid::AbstractGrid, name::String, cellid::Union{Set{Int},Vector{Int}})
    _check_setname(grid.cellsets,  name)
    cells = Set(cellid)
    _warn_emptyset(cells, name)
    grid.cellsets[name] = cells
    grid
end

function addcellset!(grid::AbstractGrid, name::String, f::Function; all::Bool=true)
    _check_setname(grid.cellsets, name)
    cells = Set{Int}()
    for (i, cell) in enumerate(getcells(grid))
        pass = all
        for node_idx in cell.nodes
            node = grid.nodes[node_idx]
            v = f(node.x)
            all ? (!v && (pass = false; break)) : (v && (pass = true; break))
        end
        pass && push!(cells, i)
    end
    _warn_emptyset(cells, name)
    grid.cellsets[name] = cells
    grid
end

"""
    addfaceset!(grid::AbstractGrid, name::String, faceid::Union{Set{FaceIndex},Vector{FaceIndex}})
    addfaceset!(grid::AbstractGrid, name::String, f::Function; all::Bool=true) 

Adds a faceset to the grid with key `name`.
A faceset maps a `String` key to a `Set` of tuples corresponding to `(global_cell_id, local_face_id)`.
Facesets are used to initialize `Dirichlet` structs, that are needed to specify the boundary for the `ConstraintHandler`.
`all=true` implies that `f(x)` must return `true` for all nodal coordinates `x` on the face if the face
should be added to the set, otherwise it suffices that `f(x)` returns `true` for one node. 

```julia
addfaceset!(grid, "right", Set(((2,2),(4,2))) #see grid manual example for reference
addfaceset!(grid, "clamped", x -> norm(x[1]) ≈ 0.0) #see incompressible elasticity example for reference
```
"""
addfaceset!(grid::AbstractGrid, name::String, set::Union{Set{FaceIndex},Vector{FaceIndex}}) = 
    _addset!(grid, name, set, grid.facesets)
addedgeset!(grid::AbstractGrid, name::String, set::Union{Set{EdgeIndex},Vector{EdgeIndex}}) = 
    _addset!(grid, name, set, grid.edgesets)
addvertexset!(grid::AbstractGrid, name::String, set::Union{Set{VertexIndex},Vector{VertexIndex}}) = 
    _addset!(grid, name, set, grid.vertexsets)
function _addset!(grid::AbstractGrid, name::String, _set, dict::Dict)
    _check_setname(dict, name)
    set = Set(_set)
    _warn_emptyset(set, name)
    dict[name] = set
    grid
end

addfaceset!(grid::AbstractGrid, name::String, f::Function; all::Bool=true) = 
    _addset!(grid, name, f, Ferrite.faces, grid.facesets, FaceIndex; all=all)
addedgeset!(grid::AbstractGrid, name::String, f::Function; all::Bool=true) = 
    _addset!(grid, name, f, Ferrite.edges, grid.edgesets, EdgeIndex; all=all)
addvertexset!(grid::AbstractGrid, name::String, f::Function; all::Bool=true) = 
    _addset!(grid, name, f, Ferrite.vertices, grid.vertexsets, VertexIndex; all=all)
function _addset!(grid::AbstractGrid, name::String, f::Function, _ftype::Function, dict::Dict, _indextype::Type; all::Bool=true)
    _check_setname(dict, name)
    _set = Set{_indextype}()
    for (cell_idx, cell) in enumerate(getcells(grid))
        for (face_idx, face) in enumerate(_ftype(cell))
            pass = all
            for node_idx in face
                v = f(grid.nodes[node_idx].x)
                all ? (!v && (pass = false; break)) : (v && (pass = true; break))
            end
            pass && push!(_set, _indextype(cell_idx, face_idx))
        end
    end
    _warn_emptyset(_set, name)
    dict[name] = _set
    grid
end


"""
    getfaceedges(grid::AbstractGrid, face::FaceIndex)
    getfaceedges(cell::AbstractCell, face::FaceIndex)

Returns the edges represented as `Set{EdgeIndex}` in a given face represented as
`FaceIndex`.

```julia-repl
julia> using Ferrite; using Ferrite: getfaceedges

julia> grid = generate_grid(Tetrahedron, (2,1,1));

julia> getfaceedges(grid, FaceIndex(4,2))
Set{EdgeIndex} with 3 elements:
  EdgeIndex((4, 4))
  EdgeIndex((4, 5))
  EdgeIndex((4, 1))
```
"""
function getfaceedges end

"""
    getfacevertices(grid::AbstractGrid, face::FaceIndex)
    getfacevertices(cell::AbstractCell, face::FaceIndex)

Returns the vertices represented as `Set{VertexIndex}` in a given face represented as
`FaceIndex`.

```julia-repl
julia> using Ferrite; using Ferrite: getfacevertices

julia> grid = generate_grid(Tetrahedron, (2,1,1));

julia> getfacevertices(grid, FaceIndex(4,2))
Set{VertexIndex} with 3 elements:
  VertexIndex((4, 2))
  VertexIndex((4, 4))
  VertexIndex((4, 1))
```
"""
function getfacevertices end

"""
    getedgevertices(grid::AbstractGrid, edge::EdgeIndex)
    getedgevertices(cell::AbstractCell, edge::EdgeIndex)

Returns the vertices represented as `Set{VertexIndex}` in a given edge represented as
`EdgeIndex`.

```julia-repl
julia> using Ferrite; using Ferrite: getedgevertices

julia> grid = generate_grid(Tetrahedron, (2,1,1));

julia> getedgevertices(grid, EdgeIndex(4,2))
Set{EdgeIndex} with 2 elements:
  VertexIndex((4, 2))
  VertexIndex((4, 3))
```
"""
function getedgevertices end

for (func,             entity_f, subentity_f, entity_t,   subentity_t) in (
    (:getfaceedges,    :faces,   :edges,      :FaceIndex, :EdgeIndex),
    (:getfacevertices, :faces,   :vertices,   :FaceIndex, :VertexIndex),
    (:getedgevertices, :edges,   :vertices,   :EdgeIndex, :VertexIndex),
)
    @eval begin
        function $(func)(grid::AbstractGrid, entity_idx::$(entity_t))
            cell = getcells(grid)[entity_idx[1]]
            return $(func)(cell, entity_idx)
        end
        function $(func)(cell::AbstractCell, entity_idx::$(entity_t))
            _set = Set{$(subentity_t)}()
            subentities = $(subentity_f)(cell)
            entity = $(entity_f)(cell)[entity_idx[2]]
            for (subentity_idx, subentity) in pairs(subentities)
                if all(x -> x in entity, subentity)
                    push!(_set, $(subentity_t)((entity_idx[1], subentity_idx)))
                end
            end
            return _set
        end
    end
end

"""
    getfaceinstances(grid::AbstractGrid, topology::ExclusiveTopology, face::FaceIndex)

Returns all the faces as `Set{FaceIndex}` that share all their vertices with a given face
represented as `FaceIndex`. The returned set includes the input face.

```julia-repl
julia> using Ferrite; using Ferrite: getfaceinstances

julia> grid = generate_grid(Tetrahedron, (2,1,1));

julia> topology = ExclusiveTopology(grid);

julia> getfaceinstances(grid, topology, FaceIndex(4,2))
Set{FaceIndex} with 2 elements:
  FaceIndex((6, 4))
  FaceIndex((4, 2))
```
"""
function getfaceinstances end

"""
    getedgeinstances(grid::AbstractGrid, topology::ExclusiveTopology, edge::EdgeIndex)

Returns all the edges as `Set{EdgeIndex}` that share all their vertices with a given edge
represented as `EdgeIndex`.
The returned set includes the input edge.

```julia-repl
julia> using Ferrite; using Ferrite: getedgeinstances

julia> grid = generate_grid(Tetrahedron, (2,1,1));

julia> topology = ExclusiveTopology(grid);

julia> getedgeinstances(grid, topology, EdgeIndex(4,2))
Set{EdgeIndex} with 3 elements:
  EdgeIndex((4, 2))
  EdgeIndex((9, 6))
  EdgeIndex((7, 6))
```
"""
function getedgeinstances end

"""
    getvertexinstances(grid::AbstractGrid, topology::ExclusiveTopology, vertex::EdgeIndex)

Returns all the vertices as `Set{::VertexIndex}` that use a given vertex represented as
`VertexIndex` in all cells.
The returned set includes the input vertex.

```julia-repl
julia> using Ferrite; using Ferrite: getvertexinstances

julia> grid = generate_grid(Tetrahedron,(2,1,1));

julia> topology = ExclusiveTopology(grid);

julia> getvertexinstances(grid, topology, VertexIndex(4,2))
Set{VertexIndex} with 8 elements:
  VertexIndex((7, 4))
  VertexIndex((10, 4))
  VertexIndex((12, 4))
  VertexIndex((6, 3))
  VertexIndex((4, 2))
  VertexIndex((9, 4))
  VertexIndex((11, 4))
  VertexIndex((8, 4))
```
"""
function getvertexinstances end

for (func,                entity_f,  entity_t) in (
    (:getvertexinstances, :vertices, :VertexIndex),
    (:getedgeinstances,   :edges,    :EdgeIndex),
    (:getfaceinstances,   :faces,    :FaceIndex),
)
    @eval begin
      function $(func)(grid::AbstractGrid, topology::ExclusiveTopology, entity::$(entity_t))
        _set = Set{$(entity_t)}()
        cells = getcells(grid)
        cell = cells[entity[1]]
        verts = $(entity_f)(cell)[entity[2]]
        # Since we are looking for an entity that share *all* vertices, the first one can be
        # used here to query potiential neighbor cells
        for cell_idx in topology.vertex_to_cell[verts[1]] # Since all vertices should be shared, the first one can be used here
            cell_entities = $(entity_f)(cells[cell_idx])
            for (entity_idx, cell_entity) in pairs(cell_entities)
                if all(x -> x in verts, cell_entity)
                    push!(_set, $(entity_t)((cell_idx, entity_idx)))
                end
            end
        end
        return _set
      end
    end
end

"""
    filterfaces(grid::AbstractGrid, faces::Set{FaceIndex}, f::Function; all::Bool=true)

Returns the faces in `faces` that satisfy `f` as a `Set{FaceIndex}`.
`all=true` implies that `f(x)` must return `true` for all nodal coordinates `x` on the face
if the face should be added to the set, otherwise it suffices that `f(x)` returns `true` for
one node.

```julia-repl
julia> using Ferrite; using Ferrite: filterfaces

julia> grid = generate_grid(Tetrahedron, (2,2,2));

julia> topology = ExclusiveTopology(grid);

julia> addboundaryfaceset!(grid, topology, "b", x -> true);

julia> filterfaces(grid, grid.facesets["b"], x -> x[3] ≈ -1)
Set{FaceIndex} with 8 elements:
  FaceIndex((7, 1))
  FaceIndex((3, 1))
  FaceIndex((21, 1))
  FaceIndex((13, 1))
  FaceIndex((19, 1))
  FaceIndex((15, 1))
  FaceIndex((1, 1))
  FaceIndex((9, 1))
```
"""
function filterfaces end

"""
    filteredges(grid::AbstractGrid, edges::Set{EdgeIndex}, f::Function; all::Bool=true)

Returns the edges in `edges` that satisfy `f` as a `Set{EdgeIndex}`.
`all=true` implies that `f(x)` must return `true` for all nodal coordinates `x` on the face
if the face should be added to the set, otherwise it suffices that `f(x)` returns `true` for
one node.

```julia-repl
julia> using Ferrite; using Ferrite: filteredges

julia> grid = generate_grid(Tetrahedron, (1,1,1));

julia> topology = ExclusiveTopology(grid);

julia> addboundaryedgeset!(grid, topology, "b", x -> true);

julia> filteredges(grid, grid.edgesets["b"], x -> x[3] ≈ -1)
Set{EdgeIndex} with 8 elements:
  EdgeIndex((1, 2))
  EdgeIndex((3, 2))
  EdgeIndex((4, 3))
  EdgeIndex((1, 3))
  EdgeIndex((3, 3))
  EdgeIndex((1, 1))
  EdgeIndex((3, 1))
  EdgeIndex((2, 3))
```
"""
function filteredges end

"""
    filtervertices(grid::AbstractGrid, vertices::Set{VertexIndex}, f::Function; all::Bool=true)

Returns the vertices in `vertices` that satisfy `f` as a `Set{VertexIndex}`.
`all=true` implies that `f(x)` must return `true` for all nodal coordinates `x` on the face
if the face should be added to the set, otherwise it suffices that `f(x)` returns `true` for
one node.

```julia-repl
julia> using Ferrite; using Ferrite: filtervertices

julia> grid = generate_grid(Tetrahedron,(1,1,1));

julia> topology = ExclusiveTopology(grid);

julia> addboundaryvertexset!(grid, topology, "b", x -> true);

julia> filtervertices(grid, grid.vertexsets["b"], x -> x[3] ≈ -1)
Set{VertexIndex} with 12 elements:
  VertexIndex((2, 3))
  VertexIndex((4, 3))
  VertexIndex((4, 1))
  VertexIndex((3, 3))
  VertexIndex((3, 2))
  VertexIndex((1, 1))
  VertexIndex((2, 1))
  VertexIndex((3, 1))
  VertexIndex((1, 3))
  VertexIndex((5, 1))
  VertexIndex((1, 2))
  VertexIndex((6, 1))
```
"""
function filtervertices end

for (func,            entity_f,  entity_t) in (
    (:filtervertices, :vertices, :VertexIndex),
    (:filteredges,    :edges,    :EdgeIndex),
    (:filterfaces,    :faces,    :FaceIndex),
)
    @eval begin
        function $(func)(grid::AbstractGrid, set::Set{$(entity_t)}, f::Function; all::Bool=true)
            _set = Set{$(entity_t)}()
            cells = getcells(grid)
            for entity in set # entities can be edges/vertices in the face/edge
                cell = cells[entity[1]]
                cell_entities = $(entity_f)(cell)
                pass = all
                for node_idx in cell_entities[entity[2]] # using cell entities common with boundary face
                    v = f(grid.nodes[node_idx].x)
                    all ? (!v && (pass = false; break)) : (v && (pass = true; break))
                end
                pass && push!(_set, entity)
            end
            return _set
        end
    end
end

"""
    addboundaryfaceset!(grid::AbstractGrid, topology::ExclusiveTopology, name::String, f::Function; all::Bool=true)

Adds a boundary faceset to the grid with key `name`.
A faceset maps a `String` key to a `Set` of tuples corresponding to `(global_cell_id,
local_face_id)`. Facesets are used to initialize `Dirichlet` structs, that are needed to
specify the boundary for the `ConstraintHandler`. `all=true` implies that `f(x)` must return
`true` for all nodal coordinates `x` on the face if the face should be added to the set,
otherwise it suffices that `f(x)` returns `true` for one node.

```julia-repl
julia> using Ferrite

julia> grid = generate_grid(Tetrahedron, (1,1,1));

julia> topology = ExclusiveTopology(grid);

julia> addboundaryfaceset!(grid, topology, "b", x -> true);

julia> grid.facesets["b"]
Set{FaceIndex} with 12 elements:
  FaceIndex((3, 1))
  FaceIndex((4, 3))
  FaceIndex((3, 3))
  FaceIndex((4, 1))
  FaceIndex((5, 1))
  FaceIndex((2, 2))
  FaceIndex((1, 4))
  FaceIndex((2, 1))
  FaceIndex((6, 1))
  FaceIndex((6, 3))
  FaceIndex((5, 3))
  FaceIndex((1, 1))
```
"""
function addboundaryfaceset! end

"""
    addboundaryedgeset!(grid::AbstractGrid, topology::ExclusiveTopology, name::String, f::Function; all::Bool=true)

Adds a boundary edgeset to the grid with key `name`.
An edgeset maps a `String` key to a `Set` of tuples corresponding to `(global_cell_id,
local_edge_id)`. `all=true` implies that `f(x)` must return `true` for all nodal coordinates
`x` on the face if the face should be added to the set, otherwise it suffices that `f(x)`
returns `true` for one node.

```julia-repl
julia> using Ferrite

julia> grid = generate_grid(Tetrahedron, (1,1,1));

julia> topology = ExclusiveTopology(grid);

julia> addboundaryedgeset!(grid, topology, "b", x -> true);

julia> grid.edgesets["b"]
Set{EdgeIndex} with 30 elements:
  EdgeIndex((6, 6))
  EdgeIndex((2, 1))
  EdgeIndex((5, 3))
  .
  .
  .
  EdgeIndex((2, 5))
  EdgeIndex((1, 4))
```
"""
function addboundaryedgeset! end

"""
    addboundaryvertexset!(grid::AbstractGrid, topology::ExclusiveTopology, name::String, f::Function; all::Bool=true)

Adds a boundary vertexset to the grid with key `name`.
A vertexset maps a `String` key to a `Set` of tuples corresponding to `(global_cell_id,
local_vertex_id)`. `all=true` implies that `f(x)` must return `true` for all nodal
coordinates `x` on the face if the face should be added to the set, otherwise it suffices
that `f(x)` returns `true` for one node.

```julia-repl
julia> using Ferrite

julia> grid = generate_grid(Tetrahedron, (1,1,1));

julia> topology = ExclusiveTopology(grid);

julia> addboundaryvertexset!(grid, topology, "b", x -> true);

julia> grid.vertexsets["b"]
Set{VertexIndex} with 24 elements:
  VertexIndex((2, 3))
  VertexIndex((5, 2))
  VertexIndex((4, 1))
  .
  .
  .
  VertexIndex((1, 4))
  VertexIndex((4, 4))
```
"""
function addboundaryvertexset! end

for (func,                   entity_f,            entity_t,     filter_f,        instance_f,          destination) in (
    (:addboundaryfaceset!,   :((_, x)->Set([x])), :FaceIndex,   :filterfaces,    :getfaceinstances,   :(grid.facesets)),
    (:addboundaryedgeset!,   :getfaceedges,       :EdgeIndex,   :filteredges,    :getedgeinstances,   :(grid.edgesets)),
    (:addboundaryvertexset!, :getfacevertices,    :VertexIndex, :filtervertices, :getvertexinstances, :(grid.vertexsets)),
)
    @eval begin
        function $(func)(grid::AbstractGrid, topology::ExclusiveTopology, name::String, f::Function; all::Bool=true)
            _check_setname($(destination), name)
            _set = Set{$(entity_t)}()
            for (face_idx, neighborhood) in pairs(topology.face_neighbor)
                isempty(neighborhood) || continue # Skip any faces with neighbors (not on boundary)
                entities =  $(entity_f)(grid, FaceIndex((face_idx[1], face_idx[2])))
                for entity in $(filter_f)(grid, entities, f; all=all)
                    union!(_set, $(instance_f)(grid, topology, entity))
                end
            end
            _warn_emptyset(_set, name)
            $(destination)[name] = _set
            return grid
        end
    end
end

"""
    addnodeset!(grid::AbstractGrid, name::String, nodeid::Union{Vector{Int},Set{Int}})
    addnodeset!(grid::AbstractGrid, name::String, f::Function)    

Adds a `nodeset::Dict{String, Set{Int}}` to the `grid` with key `name`. Has the same interface as `addcellset`. 
However, instead of mapping a cell id to the `String` key, a set of node ids is returned.
"""
function addnodeset!(grid::AbstractGrid, name::String, nodeid::Union{Vector{Int},Set{Int}})
    _check_setname(grid.nodesets, name)
    grid.nodesets[name] = Set(nodeid)
    _warn_emptyset(grid.nodesets[name], name)
    grid
end

function addnodeset!(grid::AbstractGrid, name::String, f::Function)
    _check_setname(grid.nodesets, name)
    nodes = Set{Int}()
    for (i, n) in enumerate(getnodes(grid))
        f(n.x) && push!(nodes, i)
    end
    grid.nodesets[name] = nodes
    _warn_emptyset(grid.nodesets[name], name)
    grid
end

"""
    get_cell_coordinates!(x::Vector{Vec{dim,T}}, grid::AbstractGrid, cell::Int)
    get_cell_coordinates!(x::Vector{Vec{dim,T}}, grid::AbstractGrid, cell::AbstractCell)

Fills the vector `x` with the coordinates of a cell defined by either its cellid or the cell object itself.
"""
@inline function get_cell_coordinates!(x::Vector{Vec{dim,T}}, grid::Ferrite.AbstractGrid, cellid::Int) where {dim,T} 
    cell = getcells(grid, cellid)
    get_cell_coordinates!(x, grid, cell)
end

@inline function get_cell_coordinates!(x::Vector{Vec{dim,T}}, grid::Ferrite.AbstractGrid, cell::Ferrite.AbstractCell) where {dim,T}
    @inbounds for i in 1:length(x)
        x[i] = get_node_coordinate(grid, cell.nodes[i])
    end
    return x
end

@inline get_cell_coordinates!(x::Vector{Vec{dim,T}}, grid::AbstractGrid, cell::CellIndex) where {dim, T} = get_cell_coordinates!(x, grid, cell.idx)
@inline get_cell_coordinates!(x::Vector{Vec{dim,T}}, grid::AbstractGrid, face::FaceIndex) where {dim, T} = get_cell_coordinates!(x, grid, face.idx[1])


"""
    get_cell_coordinates(grid::AbstractGrid, cell)
Return a vector with the coordinates of the vertices of cell number `cell`.
"""
@inline function get_cell_coordinates(grid::AbstractGrid, cell::Int)
    dim = getdim(grid)
    T = get_coordinate_eltype(grid)
    _cell = getcells(grid, cell)
    N = nnodes(_cell)
    x = Vector{Vec{dim, T}}(undef, N)
    get_cell_coordinates!(x, grid, _cell)
end
@inline get_cell_coordinates(grid::AbstractGrid, cell::CellIndex) = get_cell_coordinates(grid, cell.idx)
@inline get_cell_coordinates(grid::AbstractGrid, face::FaceIndex) = get_cell_coordinates(grid, face.idx[1])

function cellnodes!(global_nodes::Vector{Int}, grid::AbstractGrid, i::Int)
    cell = getcells(grid, i)
    _cellnodes!(global_nodes, cell)
end
function _cellnodes!(global_nodes::Vector{Int}, cell::AbstractCell)
    @assert length(global_nodes) == nnodes(cell)
    @inbounds for i in 1:length(global_nodes)
        global_nodes[i] = cell.nodes[i]
    end
    return global_nodes
end

function Base.show(io::IO, ::MIME"text/plain", grid::Grid)
    print(io, "$(typeof(grid)) with $(getncells(grid)) ")
    if isconcretetype(eltype(grid.cells))
        typestrs = [repr(eltype(grid.cells))]
    else
        typestrs = sort!(repr.(Set(typeof(x) for x in grid.cells)))
    end
    join(io, typestrs, '/')
    print(io, " cells and $(getnnodes(grid)) nodes")
end

"""
    boundaryfunction(::Type{<:BoundaryIndex})

Helper function to dispatch on the correct entity from a given boundary index.
"""
boundaryfunction(::Type{<:BoundaryIndex})

boundaryfunction(::Type{FaceIndex}) = Ferrite.faces
boundaryfunction(::Type{EdgeIndex}) = Ferrite.edges
boundaryfunction(::Type{VertexIndex}) = Ferrite.vertices

for INDEX in (:VertexIndex, :EdgeIndex, :FaceIndex)
    @eval begin  
        #Constructor
        ($INDEX)(a::Int, b::Int) = ($INDEX)((a,b))

        Base.getindex(I::($INDEX), i::Int) = I.idx[i]
        
        #To be able to do a,b = faceidx
        Base.iterate(I::($INDEX), state::Int=1) = (state==3) ?  nothing : (I[state], state+1)

        #For (cellid, faceidx) in faceset
        Base.in(v::Tuple{Int, Int}, s::Set{$INDEX}) = in($INDEX(v), s)
    end
end

"""
    transfer_point_cell_to_face(point::AbstractVector, cell::AbstractCell{AbstractRefShape}, face::Int)

Transform point on a N-D cell face to the face's reference (N-1)D coordinates.
"""
transfer_point_cell_to_face

"""
    transfer_point_cell_to_face(point::AbstractVector, cell::AbstractCell{AbstractRefShape}, face::Int) 

Transform point from face's reference (N-1)D coordinates to ND coordinates on the cell's face.
"""
transfer_point_face_to_cell

"""
Mapping from 1D line to point.
"""
function transfer_point_cell_to_face(point::AbstractVector, cell::AbstractCell{RefLine}, face::Int)
    x = point[]
    face == 1 && return [ -x]
    face == 2 && return [ x]
end

"""
Mapping from 2D face of a quadrilateral to 1D line.
"""
function transfer_point_cell_to_face(point::AbstractVector, cell::AbstractCell{RefQuadrilateral}, face::Int)
    x, y = point
    face == 1 && return [ x]
    face == 2 && return [ y]
    face == 3 && return [ x]
    face == 4 && return [ y]
end

"""
Mapping from 2D face of a triangle to 1D line.
"""
function transfer_point_cell_to_face(point::AbstractVector, cell::AbstractCell{RefTriangle}, face::Int)
    x, y = point
    face == 1 && return [ x]
    face == 2 && return [ y]
    face == 3 && return [ x]
end

"""
Mapping from 3D face of a tetrahedon to 2D triangle.
"""
function transfer_point_cell_to_face(point::AbstractVector, cell::AbstractCell{RefTetrahedron}, face::Int)
    x, y, z = point
    face == 1 && return [ x,  y]
    face == 2 && return [ x,  z]
    face == 3 && return [ x,  y]
    face == 4 && return [ y,  z]
end

"""
Mapping from 3D face of a hexahedron to 2D quadrilateral.
"""
function transfer_point_cell_to_face(point::AbstractVector, cell::AbstractCell{RefHexahedron}, face::Int)
    x, y, z = point
    face == 1 && return [ x, y]
    face == 2 && return [ x, z]
    face == 3 && return [ y,  z]
    face == 4 && return [-x,  z]
    face == 5 && return [ z,  y]
    face == 6 && return [ x,  y]
end


"""
Mapping from to 0D node to 1D line vertex.
"""
function transfer_point_face_to_cell(point::AbstractVector, cell::AbstractCell{RefLine}, face::Int)
    face == 1 && return [ -1.0]
    face == 2 && return [ 1.0]
end

"""
Mapping from 1D line to 2D face of a quadrilateral.
"""
function transfer_point_face_to_cell(point::AbstractVector, cell::AbstractCell{RefQuadrilateral}, face::Int)
    x = point[]
    face == 1 && return [ x, -1]
    face == 2 && return [ 1, x]
    face == 3 && return [ x, 1]
    face == 4 && return [ -1, x]
end

"""
Mapping from 1D line to 2D face of a triangle.
"""
function transfer_point_face_to_cell(point::AbstractVector, cell::AbstractCell{RefTriangle}, face::Int)
    x = point[]
    face == 1 && return [ x, 1 - x]
    face == 2 && return [ 0, x]
    face == 3 && return [ x, 0]
end

"""
Mapping from 2D triangle to 3D face of a tetrahedon.
"""
function transfer_point_face_to_cell(point::AbstractVector, cell::AbstractCell{RefTetrahedron}, face::Int)
    x,y = point
    face == 1 && return [ x,  y,  0]
    face == 2 && return [ x,  0,  y]
    face == 3 && return [ x,  y,  1-x-y]
    face == 4 && return [ 0,  x,  y]
end

"""
Mapping from 2D quadrilateral to 3D face of a hexahedron.
"""
function transfer_point_face_to_cell(point::AbstractVector, cell::AbstractCell{RefHexahedron}, face::Int)
    x,y = point
    face == 1 && return [ y,  x, -1]
    face == 2 && return [ x, -1,  y]
    face == 3 && return [ 1,  x,  y]
    face == 4 && return [ x,  1,  y]
    face == 5 && return [-1,  y,  x]
    face == 6 && return [ x,  y,  1]
end

#################################
#### Orientation of Entities ####
#################################
# @TODO merge this code with into the logic in `ConstraintHandler`.

"""
    PathOrientationInfo

Orientation information for 1D entities.

The orientation for 1D entities is defined by the indices of the grid nodes
associated to the vertices. To give an example, the oriented path
```
1 ---> 2
```
is called *regular*, indicated by `regular=true`, while the oriented path
```
2 ---> 1
```
is called *inverted*, indicated by `regular=false`.
"""
struct PathOrientationInfo
    regular::Bool # Indicator whether the orientation is regular or inverted.
end

"""
    SurfaceOrientationInfo

Orientation information for 2D entities. Such an entity can be 
possibly flipped (i.e. the defining vertex order is reverse to the 
spanning vertex order) and the vertices can be rotated against each other.
Take for example the faces
```
1---2 2---3
| A | | B |
4---3 1---4
```
which are rotated against each other by 90° (shift index is 1) or the faces
```
1---2 2---1
| A | | B |
4---3 3---4
```
which are flipped against each other. Any combination of these can happen. 
The combination to map this local face to the defining face is encoded with
this data structure via ``rotate \\circ flip`` where the rotation is indiced by
the shift index.
    !!!NOTE TODO implement me.
"""
struct SurfaceOrientationInfo
    #flipped::Bool
    #shift_index::Int
end
# This one is the same as `SurfaceOrientationInfo` but used in interfaces. It is calculated for neighbor facet relative to current facet
struct InterfaceOrientationInfo
    flipped::Bool
    transformation::Union{Nothing, AbstractArray}
end

function transform_coordinates(src::Vector{Vector{Float64}}, dst::Vector{Vector{Float64}})
    # Calculate the transformation matrix
    src_matrix = [  src[1][1]   src[2][1]   src[3][1]
                    src[1][2]   src[2][2]   src[3][2]
                    1           1           1       ]
    dst_matrix = [  dst[1][1]   dst[2][1]   dst[3][1]
                    dst[1][2]   dst[2][2]   dst[3][2]
                    1           1           1       ]
    transform_matrix = dst_matrix / src_matrix

    return transform_matrix
end
# TODO: make this `SurfaceOrientationInfo`?
"""
    InterfaceOrientationInfo(grid::AbstractGrid, this_face::FaceIndex, neighbor_face::FaceIndex)

Orientation information for interfaces. Such an interface can be 
possibly flipped (i.e. the defining vertex order is reverse to the 
spanning vertex order) and the vertices can be rotated skewed against each other, and skewed in the case of triangles.
Take for example the faces
```
1           3
| \\         | \\
|  \\        |  \\
| A \\       | B \\ 
|    \\      |    \\
2-----3     1-----2  
```
which are skewed and rotated against each other by 90° or the faces
```
1           3
| \\         | \\
|  \\        |  \\
| A \\       | B \\ 
|    \\      |    \\
2-----3     2-----1  
```
which are flipped about the centroid vector. Any combination of these can happen. 
The combination to map neighbor facet to current facet is encoded with
this data structure via ``\\vec{p}_{\\text{there}} = [T] \\vec{p}_{\\text{here}}`` where ``[T]`` is `InterfaceOrientationInfo.transformation`.
"""
function InterfaceOrientationInfo(grid::AbstractGrid, this_face::FaceIndex, neighbor_face::FaceIndex)
    cell = getcells(grid)[this_face[1]]
    other_cell = getcells(grid)[neighbor_face[1]]
    face_nodes = faces(cell)[this_face[2]]
    neighbor_face_nodes = faces(other_cell)[neighbor_face[2]]

    !all([i ∈ face_nodes for i in neighbor_face_nodes]) && error("Passed faces do not use the same nodes")
    getdim(cell) == 1 && return(InterfaceOrientationInfo(false, nothing)) # this may change for embedded elements?

    cell_ip = cell|> typeof|> default_interpolation
    cell_coords = cell_ip|> reference_coordinates
    nodes_coord = cell_coords[[((cell_ip |>getrefshape|> faces)[this_face[2]])...]]
    nodes_coord = transfer_point_cell_to_face.(nodes_coord, Ref(cell), Ref(this_face[2]))

    other_cell_ip = other_cell|> typeof|> default_interpolation
    other_cell_coords = other_cell_ip|> reference_coordinates
    other_nodes_coord = other_cell_coords[[((other_cell_ip|> getrefshape|> faces)[neighbor_face[2]])...]]
    other_nodes_coord = transfer_point_cell_to_face.(other_nodes_coord, Ref(other_cell), Ref(neighbor_face[2]))

    getdim(cell) == 2 && return(InterfaceOrientationInfo(nodes_coord[1] == -other_nodes_coord[1], nothing))

    flipped = ([(nodes_coord[2] - nodes_coord[1])..., 0] × [(nodes_coord[3] - nodes_coord[2])..., 0])[3] > 0
    flipped = flipped == (([(other_nodes_coord[2] - other_nodes_coord[1])..., 0] × [(other_nodes_coord[3] - other_nodes_coord[2])..., 0])[3] < 0)
    flipped || reverse!.(nodes_coord)
    return InterfaceOrientationInfo(flipped, transform_coordinates(nodes_coord[1:3],other_nodes_coord[1:3]))
end
