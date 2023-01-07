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
getcoordinates(n::Node) = n.x


abstract type AbstractCell{dim,N,M} end
"""
    Cell{dim,N,M} <: AbstractCell{dim,N,M}
A `Cell` is a sub-domain defined by a collection of `Node`s as it's vertices.
However, a `cell` is not defined by the nodes but rather by the global node ids.
The parameter `dim` refers here to the geometrical/ambient dimension, i.e. the dimension of the `nodes` in the grid and **not** the topological dimension of the cell.
A `Cell` has `N` nodes and `M` faces.

# Fields
- `nodes::Ntuple{N,Int}`: N-tuple that stores the node ids. The ordering defines a cell's and its subentities' orientations.
"""
struct Cell{dim,N,M} <: AbstractCell{dim,N,M}
    nodes::NTuple{N,Int}
end
nfaces(c::C) where {C<:AbstractCell} = nfaces(typeof(c))
nfaces(::Type{<:AbstractCell{dim,N,M}}) where {dim,N,M} = M
nnodes(c::C) where {C<:AbstractCell} = nnodes(typeof(c))
nnodes(::Type{<:AbstractCell{dim,N,M}}) where {dim,N,M} = N

# Typealias for commonly used cells
const Line  = Cell{1,2,2}
const Line2D = Cell{2,2,1}
const Line3D = Cell{3,2,0}
const QuadraticLine = Cell{1,3,2}

const Triangle = Cell{2,3,3}
const QuadraticTriangle = Cell{2,6,3}

const Quadrilateral = Cell{2,4,4}
const Quadrilateral3D = Cell{3,4,1}
const QuadraticQuadrilateral = Cell{2,9,4}

const Tetrahedron = Cell{3,4,4}
const QuadraticTetrahedron = Cell{3,10,4}

const Hexahedron = Cell{3,8,6}

"""
A `CellIndex` wraps an Int and corresponds to a cell with that number in the mesh
"""
struct CellIndex
    idx::Int
end

"""
A `FaceIndex` wraps an (Int, Int) and defines a local face by pointing to a (cell, face).
"""
struct FaceIndex <: BoundaryIndex
    idx::Tuple{Int,Int} # cell and side
end

"""
A `EdgeIndex` wraps an (Int, Int) and defines a local edge by pointing to a (cell, edge).
"""
struct EdgeIndex <: BoundaryIndex
    idx::Tuple{Int,Int} # cell and side
end

"""
A `VertexIndex` wraps an (Int, Int) and defines a local vertex by pointing to a (cell, vert).
"""
struct VertexIndex <: BoundaryIndex
    idx::Tuple{Int,Int} # cell and side
end

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
    face_npoints(::AbstractCell{dim,N,M)
Specifies for each subtype of AbstractCell how many nodes form a face
"""
face_npoints(::Cell{2,N,M}) where {N,M} = 2
face_npoints(::Cell{3,4,1}) = 4 #not sure how to handle embedded cells e.g. Quadrilateral3D
"""
    edge_npoints(::AbstractCell{dim,N,M)
Specifies for each subtype of AbstractCell how many nodes form an edge
"""
edge_npoints(::Cell{3,4,1}) = 2 #not sure how to handle embedded cells e.g. Quadrilateral3D
face_npoints(::Cell{3,N,6}) where N = 4
face_npoints(::Cell{3,N,4}) where N = 3
edge_npoints(::Cell{3,N,M}) where {N,M} = 2

getdim(::Cell{dim}) where dim = dim

abstract type AbstractTopology end

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
            neighbor_local_ids = findall(x->x in cell.nodes, cells[neighbor].nodes)
            cell_local_ids = findall(x->x in cells[neighbor].nodes, cell.nodes)
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
    getneighborhood(top::ExclusiveTopology, grid::Grid{dim,C,T}, cellidx::CellIndex, include_self=false)
    getneighborhood(top::ExclusiveTopology, grid::Grid{dim,C,T}, faceidx::FaceIndex, include_self=false)
    getneighborhood(top::ExclusiveTopology, grid::Grid{dim,C,T}, vertexidx::VertexIndex, include_self=false)
    getneighborhood(top::ExclusiveTopology, grid::Grid{dim,C,T}, edgeidx::EdgeIndex, include_self=false)

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
        return [top.vertex_vertex_neighbor[global_vertexid].neighbor_info; vertexidx]
    else
        return top.vertex_vertex_neighbor[global_vertexid].neighbor_info
    end
end

function getneighborhood(top::ExclusiveTopology, grid::AbstractGrid{3}, edgeidx::EdgeIndex, include_self=false)
    if include_self 
        return [top.edge_neighbor[edgeidx[1],edgeidx[2]].neighbor_info; edgeidx]
    else
        return top.edge_neighbor[edgeidx[1],edgeidx[2]].neighbor_info
    end
end

"""
    faceskeleton(grid) -> Vector{FaceIndex}
Returns an iterateable face skeleton. The skeleton consists of `FaceIndex` that can be used to `reinit` 
`FaceValues`.
"""
faceskeleton(top::ExclusiveTopology, grid::AbstractGrid) =  top.face_skeleton

toglobal(grid::Grid,vertexidx::VertexIndex) = vertices(getcells(grid,vertexidx[1]))[vertexidx[2]]
toglobal(grid::Grid,vertexidx::Vector{VertexIndex}) = unique(toglobal.((grid,),vertexidx))

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

n_faces_per_cell(grid::Grid) = nfaces(eltype(grid.cells))

"""
    function compute_vertex_values(grid::AbstractGrid, f::Function)
    function compute_vertex_values(grid::AbstractGrid, v::Vector{Int}, f::Function)    
    function compute_vertex_values(grid::AbstractGrid, set::String, f::Function)

Given a `grid` and some function `f`, `compute_vertex_values` computes all nodal values,
 i.e. values at the nodes,  of the function `f`. 
The function implements two dispatches, where only a subset of the grid's node is used.

```julia
    compute_vertex_values(grid, x -> sin(x[1]) + cos([2]))
    compute_vertex_values(grid, [9, 6, 3], x -> sin(x[1]) + cos([2])) #compute function values at nodes with id 9,6,3
    compute_vertex_values(grid, "right", x -> sin(x[1]) + cos([2])) #compute function values at nodes belonging to nodeset right
```

"""
@inline function compute_vertex_values(nodes::Vector{Node{dim,T}}, f::Function) where{dim,T}
    map(n -> f(getcoordinates(n)), nodes)
end

@inline function compute_vertex_values(grid::AbstractGrid, f::Function)
    compute_vertex_values(getnodes(grid), f::Function)
end

@inline function compute_vertex_values(grid::AbstractGrid, v::Vector{Int}, f::Function)
    compute_vertex_values(getnodes(grid, v), f::Function)
end

@inline function compute_vertex_values(grid::AbstractGrid, set::String, f::Function)
    compute_vertex_values(getnodes(grid, set), f::Function)
end

# Transformations
"""
    transform!(grid::Abstractgrid, f::Function)

Transform all nodes of the `grid` based on some transformation function `f`.
"""
function transform!(g::AbstractGrid, f::Function)
    c = similar(g.nodes)
    for i in 1:length(c)
        c[i] = Node(f(g.nodes[i].x))
    end
    copyto!(g.nodes, c)
    g
end

# Sets

_check_setname(dict, name) = haskey(dict, name) && throw(ArgumentError("there already exists a set with the name: $name"))
_warn_emptyset(set, name) = length(set) == 0 && @warn("no entities added to the set with name: $name")

"""
    addcellset!(grid::AbstractGrid, name::String, cellid::Union{Set{Int}, Vector{Int}})
    addcellset!(grid::AbstractGrid, name::String, f::function; all::Bool=true)

Adds a cellset to the grid with key `name`.
Cellsets are typically used to define subdomains of the problem, e.g. two materials in the computational domain.
The `MixedDofHandler` can construct different fields which live not on the whole domain, but rather on a cellset.
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
addfaceset!(gird, "right", Set(((2,2),(4,2))) #see grid manual example for reference
addfaceset!(grid, "clamped", x -> norm(x[1]) ≈ 0.0) #see incompressible elasticity example for reference
```
"""
addfaceset!(grid::Grid, name::String, set::Union{Set{FaceIndex},Vector{FaceIndex}}) = 
    _addset!(grid, name, set, grid.facesets)
addedgeset!(grid::Grid, name::String, set::Union{Set{EdgeIndex},Vector{EdgeIndex}}) = 
    _addset!(grid, name, set, grid.edgesets)
addvertexset!(grid::Grid, name::String, set::Union{Set{VertexIndex},Vector{VertexIndex}}) = 
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
    getcoordinates!(x::Vector{Vec{dim,T}}, grid::AbstractGrid, cell::Int)
Fills the vector `x` with the coordinates of a cell, defined by its cell id.
"""
@inline function getcoordinates!(x::Vector{Vec{dim,T}}, grid::AbstractGrid, cell::Int) where {dim,T}
    #@assert length(x) == N
    @inbounds for i in 1:length(x)
        x[i] = grid.nodes[grid.cells[cell].nodes[i]].x
    end
end
@inline getcoordinates!(x::Vector{Vec{dim,T}}, grid::AbstractGrid, cell::CellIndex) where {dim, T} = getcoordinates!(x, grid, cell.idx)
@inline getcoordinates!(x::Vector{Vec{dim,T}}, grid::AbstractGrid, face::FaceIndex) where {dim, T} = getcoordinates!(x, grid, face.idx[1])

"""
    getcoordinates(grid::AbstractGrid, cell)
Return a vector with the coordinates of the vertices of cell number `cell`.
"""
@inline function getcoordinates(grid::AbstractGrid, cell::Int)
    # TODO pretty ugly, worth it?
    dim = typeof(grid.cells[cell]).parameters[1]
    T = typeof(grid).parameters[3]
    nodeidx = grid.cells[cell].nodes
    return [grid.nodes[i].x for i in nodeidx]::Vector{Vec{dim,T}}
end
@inline getcoordinates(grid::AbstractGrid, cell::CellIndex) = getcoordinates(grid, cell.idx)
@inline getcoordinates(grid::AbstractGrid, face::FaceIndex) = getcoordinates(grid, face.idx[1])

function Base.show(io::IO, ::MIME"text/plain", grid::Grid)
    print(io, "$(typeof(grid)) with $(getncells(grid)) ")
    typestrs = sort!(collect(Set(celltypes[typeof(x)] for x in grid.cells)))
    str = join(io, typestrs, '/')
    print(io, " cells and $(getnnodes(grid)) nodes")
end

const celltypes = Dict{DataType, String}(Cell{1,2,2}  => "Line",
                                         Cell{2,2,2}  => "2D-Line",
                                         Cell{3,2,0}  => "3D-Line",
                                         Cell{1,3,2}  => "QuadraticLine",
                                         Cell{2,3,3}  => "Triangle",
                                         Cell{2,6,3}  => "QuadraticTriangle",
                                         Cell{2,4,4}  => "Quadrilateral",
                                         Cell{3,4,1}  => "3D-Quadrilateral",
                                         Cell{2,9,4}  => "QuadraticQuadrilateral",
                                         Cell{3,4,4}  => "Tetrahedron",
                                         Cell{3,10,4} => "QuadraticTetrahedron",
                                         Cell{3,8,6}  => "Hexahedron",
                                         Cell{3,20,6} => "Cell{3,20,6}")

# Functions to uniquely identify vertices, edges and faces, used when distributing
# dofs over a mesh. For this we can ignore the nodes on edged, faces and inside cells,
# we only need to use the nodes that are vertices.
# 1D: vertices
faces(c::Union{Line,QuadraticLine}) = (c.nodes[1], c.nodes[2])
vertices(c::Union{Line,Line2D,Line3D,QuadraticLine}) = (c.nodes[1], c.nodes[2])
# 2D: vertices, faces
faces(c::Line2D) = ((c.nodes[1],c.nodes[2]),)
vertices(c::Union{Triangle,QuadraticTriangle}) = (c.nodes[1], c.nodes[2], c.nodes[3])
faces(c::Union{Triangle,QuadraticTriangle}) = ((c.nodes[1],c.nodes[2]), (c.nodes[2],c.nodes[3]), (c.nodes[3],c.nodes[1]))
vertices(c::Union{Quadrilateral,Quadrilateral3D,QuadraticQuadrilateral}) = (c.nodes[1], c.nodes[2], c.nodes[3], c.nodes[4])
faces(c::Union{Quadrilateral,QuadraticQuadrilateral}) = ((c.nodes[1],c.nodes[2]), (c.nodes[2],c.nodes[3]), (c.nodes[3],c.nodes[4]), (c.nodes[4],c.nodes[1]))
# 3D: vertices, edges, faces
edges(c::Line3D) = ((c.nodes[1],c.nodes[2]),)
vertices(c::Union{Tetrahedron,QuadraticTetrahedron}) = (c.nodes[1], c.nodes[2], c.nodes[3], c.nodes[4])
edges(c::Union{Tetrahedron,QuadraticTetrahedron}) = ((c.nodes[1],c.nodes[2]), (c.nodes[2],c.nodes[3]), (c.nodes[3],c.nodes[1]), (c.nodes[1],c.nodes[4]), (c.nodes[2],c.nodes[4]), (c.nodes[3],c.nodes[4]))
faces(c::Union{Tetrahedron,QuadraticTetrahedron}) = ((c.nodes[1],c.nodes[3],c.nodes[2]), (c.nodes[1],c.nodes[2],c.nodes[4]), (c.nodes[2],c.nodes[3],c.nodes[4]), (c.nodes[1],c.nodes[4],c.nodes[3]))
vertices(c::Union{Hexahedron,Cell{3,20,6}}) = (c.nodes[1], c.nodes[2], c.nodes[3], c.nodes[4], c.nodes[5], c.nodes[6], c.nodes[7], c.nodes[8])
edges(c::Union{Hexahedron,Cell{3,20,6}}) = ((c.nodes[1],c.nodes[2]), (c.nodes[2],c.nodes[3]), (c.nodes[3],c.nodes[4]), (c.nodes[4],c.nodes[1]), (c.nodes[5],c.nodes[6]), (c.nodes[6],c.nodes[7]), (c.nodes[7],c.nodes[8]), (c.nodes[8],c.nodes[5]), (c.nodes[1],c.nodes[5]), (c.nodes[2],c.nodes[6]), (c.nodes[3],c.nodes[7]), (c.nodes[4],c.nodes[8]))
faces(c::Union{Hexahedron,Cell{3,20,6}}) = ((c.nodes[1],c.nodes[4],c.nodes[3],c.nodes[2]), (c.nodes[1],c.nodes[2],c.nodes[6],c.nodes[5]), (c.nodes[2],c.nodes[3],c.nodes[7],c.nodes[6]), (c.nodes[3],c.nodes[4],c.nodes[8],c.nodes[7]), (c.nodes[1],c.nodes[5],c.nodes[8],c.nodes[4]), (c.nodes[5],c.nodes[6],c.nodes[7],c.nodes[8]))
edges(c::Union{Quadrilateral3D}) = ((c.nodes[1],c.nodes[2]), (c.nodes[2],c.nodes[3]), (c.nodes[3],c.nodes[4]), (c.nodes[4],c.nodes[1]))
faces(c::Union{Quadrilateral3D}) = ((c.nodes[1],c.nodes[2],c.nodes[3],c.nodes[4]),)

# random stuff
default_interpolation(::Union{Type{Line},Type{Line2D},Type{Line3D}}) = Lagrange{1,RefCube,1}()
default_interpolation(::Type{QuadraticLine}) = Lagrange{1,RefCube,2}()
default_interpolation(::Type{Triangle}) = Lagrange{2,RefTetrahedron,1}()
default_interpolation(::Type{QuadraticTriangle}) = Lagrange{2,RefTetrahedron,2}()
default_interpolation(::Union{Type{Quadrilateral},Type{Quadrilateral3D}}) = Lagrange{2,RefCube,1}()
default_interpolation(::Type{QuadraticQuadrilateral}) = Lagrange{2,RefCube,2}()
default_interpolation(::Type{Tetrahedron}) = Lagrange{3,RefTetrahedron,1}()
default_interpolation(::Type{QuadraticTetrahedron}) = Lagrange{3,RefTetrahedron,2}()
default_interpolation(::Type{Hexahedron}) = Lagrange{3,RefCube,1}()
default_interpolation(::Type{Cell{3,20,6}}) = Serendipity{3,RefCube,2}()

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
