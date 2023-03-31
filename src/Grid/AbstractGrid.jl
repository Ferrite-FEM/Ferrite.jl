# Defines all Abstract entities related to a grid type
abstract type AbstractCell{dim,N,M} end

nfaces(c::C) where {C<:AbstractCell} = nfaces(typeof(c))
nfaces(::Type{<:AbstractCell{dim,N,M}}) where {dim,N,M} = M
nedges(c::C) where {C<:AbstractCell} = length(edges(c))
nvertices(c::C) where {C<:AbstractCell} = length(vertices(c))
nnodes(c::C) where {C<:AbstractCell} = nnodes(typeof(c))
nnodes(::Type{<:AbstractCell{dim,N,M}}) where {dim,N,M} = N

"""
    Ferrite.vertices(::AbstractCell)

Returns a tuple with the node indices (of the nodes in a grid) for each vertex in a given cell.
This function induces the [`VertexIndex`](@ref), where the second index 
corresponds to the local index into this tuple.
"""
vertices(::Ferrite.AbstractCell)

"""
    Ferrite.edges(::AbstractCell)

Returns a tuple of 2-tuples containing the ordered node indices (of the nodes in a grid) corresponding to
the vertices that define an *oriented edge*. This function induces the 
[`EdgeIndex`](@ref), where the second index corresponds to the local index into this tuple.

Note that the vertices are sufficient to define an edge uniquely.
"""
edges(::Ferrite.AbstractCell)

"""
    Ferrite.faces(::AbstractCell)

Returns a tuple of n-tuples containing the ordered node indices (of the nodes in a grid) corresponding to
the vertices that define an *oriented face*. This function induces the 
[`FaceIndex`](@ref), where the second index corresponds to the local index into this tuple.

Note that the vertices are sufficient to define a face uniquely.
"""
faces(::Ferrite.AbstractCell)

"""
    Ferrite.default_interpolation(::AbstractCell)::Interpolation

Returns the interpolation which defines the geometry of a given cell.
"""
default_interpolation(::Ferrite.AbstractCell)

"""
    Ferrite.getnodeidxs(cell::AbstractCell)
    Ferrite.getnodeidxs(cell::AbstractCell, i::Integer)

Return the node indexes (in the parent grid) for all nodes in `cell`,
or for node number `i` in the given `cell`
"""
function getnodeidxs end
@inline getnodeidxs(cell::AbstractCell, i::Integer) = getnodeidxs(cell)[i]

abstract type AbstractGrid{dim} end

@inline getdim(::AbstractGrid{dim}) where {dim} = dim


"""
    getcells(grid::AbstractGrid) 
    getcells(grid::AbstractGrid, v::Union{Int,Vector{Int}} 
    getcells(grid::AbstractGrid, setname::String)

Returns either all `cells::Collection{C<:AbstractCell}` of a `<:AbstractGrid` or a subset based on an `Int`, `Vector{Int}` or `String`.
Whereas the last option tries to call a `cellset` of the `grid`. `Collection` can be any indexable type, for `Grid` it is `Vector{C<:AbstractCell}`.
"""
function getcells end
@inline getcells(grid::AbstractGrid, v::Union{Int, Vector{Int}}) = getcells(grid)[v]
@inline getcells(grid::AbstractGrid, setname::String) = getcells(grid)[collect(getcellset(grid,setname))]

"Returns the number of cells in the `<:AbstractGrid`."
@inline getncells(grid::AbstractGrid) = length(getcells(grid))

"Returns the celltype of the `<:AbstractGrid`."
@inline getcelltype(grid::AbstractGrid) = eltype(getcells(grid))
@inline getcelltype(grid::AbstractGrid, i::Int) = typeof(getcells(grid, i))

"""
    getnodes(grid::AbstractGrid) 
    getnodes(grid::AbstractGrid, v::Union{Int,Vector{Int}}
    getnodes(grid::AbstractGrid, setname::String)

Returns either all `nodes::Collection{N}` of a `<:AbstractGrid` or a subset based on an `Int`, `Vector{Int}` or `String`.
The last option tries to call a `nodeset` of the `<:AbstractGrid`. `Collection{N}` refers to some indexable collection where each element corresponds
to a Node.
"""
function getnodes end

@inline getnodes(grid::AbstractGrid, v::Union{Int, Vector{Int}}) = grid.nodes[v]
@inline getnodes(grid::AbstractGrid, setname::String) = grid.nodes[collect(getnodeset(grid,setname))]

"Returns the number of nodes in the grid."
@inline getnnodes(grid::AbstractGrid) = length(getnodes(grid))

"Returns the number of nodes of the `i`-th cell."
@inline nnodes_per_cell(grid::AbstractGrid, i::Int=1) = nnodes(getcells(grid, i))

get_node_coordinate(grid::AbstractGrid, i::Integer) = get_node_coordinate(getnodes(grid, i))

"Return the number type of the nodal coordinates."
@inline get_coordinate_eltype(grid::AbstractGrid) = get_coordinate_eltype(first(getnodes(grid)))

"Return the type of the nodal coordinates."
@inline get_coordinate_type(grid::AbstractGrid) = get_coordinate_type(first(getnodes(grid)))

"""
    getcoordinates!(x::Vector, grid::AbstractGrid, cell::Int)
    getcoordinates!(x::Vector, grid::AbstractGrid, cell::AbstractCell)

Fills the vector `x` with the coordinates of a cell defined by either its cellid or the cell object itself.
"""
@inline function getcoordinates!(x::Vector, grid::Ferrite.AbstractGrid, cellid::Int)
    cell = getcells(grid, cellid)
    getcoordinates!(x, grid, cell)
end

@inline function getcoordinates!(x::Vector, grid::Ferrite.AbstractGrid, cell::Ferrite.AbstractCell)
    @inbounds for i in 1:length(x)
        x[i] = get_node_coordinate(grid, getnodeidxs(cell, i))
    end
    return x
end

@inline getcoordinates!(x::Vector, grid::AbstractGrid, cell::CellIndex) = getcoordinates!(x, grid, cell.idx)
@inline getcoordinates!(x::Vector, grid::AbstractGrid, face::FaceIndex) = getcoordinates!(x, grid, face.idx[1])

# TODO: Deprecate one of `cellcoords!` and `getcoordinates!`, as they do the same thing
cellcoords!(global_coords::Vector{Vec{dim,T}}, grid::AbstractGrid{dim}, i::Int) where {dim,T} = getcoordinates!(global_coords, grid, i) 

"""
    getcoordinates(grid::AbstractGrid, cell)
Return a vector with the coordinates of the vertices of cell number `cell`.
"""
@inline function getcoordinates(grid::AbstractGrid, cell::Int)
    dim = getdim(grid)
    T = get_coordinate_eltype(grid)
    _cell = getcells(grid, cell)
    N = nnodes(_cell)
    x = Vector{Vec{dim,T}}(undef, N)
    getcoordinates!(x, grid, _cell)
end

@inline getcoordinates(grid::AbstractGrid, cell::CellIndex) = getcoordinates(grid, cell.idx)
@inline getcoordinates(grid::AbstractGrid, face::FaceIndex) = getcoordinates(grid, face.idx[1])

function cellnodes!(global_nodes::Vector{Int}, grid::AbstractGrid, i::Int)
    cell = getcells(grid, i)
    _cellnodes!(global_nodes, cell)
end
function _cellnodes!(global_nodes::Vector{Int}, cell::AbstractCell)
    @assert length(global_nodes) == nnodes(cell)
    @inbounds for i in 1:length(global_nodes)
        global_nodes[i] = getnodeidxs(cell, i)
    end
    return global_nodes
end

function Base.show(io::IO, ::MIME"text/plain", grid::AbstractGrid)
    print(io, "$(typeof(grid)) with $(getncells(grid)) ")
    if isconcretetype(getcelltype(grid))
        typestrs = [repr(getcelltype(grid))]
    else
        typestrs = sort!(repr.(Set(typeof(x) for x in getcells(grid))))
    end
    join(io, typestrs, '/')
    print(io, " cells and $(getnnodes(grid)) nodes")
end

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
    map(n -> f(get_node_coordinate(n)), nodes)
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

abstract type AbstractTopology end
# AbstractTopology is only used as supertype for ExclusiveTopology as of now
# Exact interface to be designed. 