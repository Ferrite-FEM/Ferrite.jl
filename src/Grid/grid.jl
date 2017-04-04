# Grid types
export Node, Cell, CellIndex, CellFace, FaceIndex, Grid

# Cell type alias
export Line, QuadraticLine,
       Triangle, QuadraticTriangle, Quadrilateral, QuadraticQuadrilateral,
       Tetrahedron, QuadraticTetrahedron, Hexahedron, QuadraticHexahedron

# Grid utilities
export getcells, getncells, getnodes, getnnodes, getcelltype,
       getcellset,  getnodeset, getfaceset, getcoordinates, getcoordinates!,
       getcellsets, getnodesets, getfacesets, onboundary, nfaces

export addnodeset!, addcellset!, transform

#########################
# Main types for meshes #
#########################
"""
A `Node` is a point in space.
"""
immutable Node{dim, T}
    x::Vec{dim, T}
end
Node{dim, T}(x::NTuple{dim, T}) = Node(Vec{dim, T}(x))
getcoordinates(n::Node) = n.x

#=
#A bit redundant but perhaps we want to put more in here:
immutable Face
    onboundary::Bool
end
onboundary(f::Face) = f.onboundary
Face() = Face(false)
=#

"""
A `Cell` is a sub-domain defined by a collection of `Node`s as it's vertices.
"""
immutable Cell{dim, N, M}
    nodes::NTuple{N, Int}
    onboundary::NTuple{M, Bool}
end

onboundary(c::Cell, face::Int) = c.onboundary[face]

(::Type{Cell{dim, N, M}}){dim, N, M}(nodes::NTuple{N}) = Cell{dim,N,M}(nodes, ntuple(i->false, Val{M}))
(::Type{Cell{dim, N, M}}){dim, N, M}(nodes::NTuple{N}, arr::AbstractArray{Bool}) = Cell{dim,N,M}(nodes, ntuple(i -> arr[i], Val{M}))


nfaces(c::Cell) = nfaces(typeof(c))
nfaces{dim, N, M}(::Type{Cell{dim, N, M}}) = M

# Typealias for commonly used cells

@compat const Line = Cell{1, 2, 2}
@compat const QuadraticLine = Cell{1, 3, 2}

@compat const Triangle = Cell{2, 3, 3}
@compat const QuadraticTriangle = Cell{2, 6, 3}

@compat const Quadrilateral = Cell{2, 4, 4}
@compat const QuadraticQuadrilateral = Cell{2, 9, 4}

@compat const Tetrahedron = Cell{3, 4, 4}
@compat const QuadraticTetrahedron = Cell{3, 10, 4}

@compat const Hexahedron = Cell{3, 8, 6}
@compat const QuadraticHexahedron = Cell{3, 20, 6} # Function interpolation for this doesn't exist in JuAFEM yet

"""
A `CellIndex` wraps an Int and corresponds to a cell with that number in the mesh
"""
immutable CellIndex
    idx::Int
end

"""
A `FaceIndex` wraps an (Int, Int) and defines a face by pointing to a (cell, face).
"""
immutable FaceIndex
    idx::Tuple{Int, Int} # cell and side
end

"""
A `Grid` is a collection of `Cells` and `Node`s which covers the computational domain, together with Sets of cells, nodes and faces.
"""
type Grid{dim, N, T <: Real, M}
    cells::Vector{Cell{dim, N, M}}
    nodes::Vector{Node{dim, T}}
    # Sets
    cellsets::Dict{String, Set{Int}}
    nodesets::Dict{String, Set{Int}}
    facesets::Dict{String, Set{Tuple{Int, Int}}}
end

function Grid{dim, N, M, T}(cells::Vector{Cell{dim, N, M}},
                         nodes::Vector{Node{dim, T}};
                         cellsets::Dict{String, Set{Int}}=Dict{String, Set{Int}}(),
                         nodesets::Dict{String, Set{Int}}=Dict{String, Set{Int}}(),
                         facesets::Dict{String, Set{Tuple{Int, Int}}}=Dict{String, Set{Tuple{Int, Int}}}(),
                        )
    return Grid(cells, nodes, cellsets, nodesets, facesets)
end

##########################
# Grid utility functions #
##########################
@inline getcells(grid::Grid) = grid.cells
@inline getcells(grid::Grid, v::Union{Int, Vector{Int}}) = grid.cells[v]
@inline getcells(grid::Grid, set::String) = grid.cells[grid.cellsets[set]]
@inline getncells(grid::Grid) = length(grid.cells)
@inline getcelltype(grid::Grid) = eltype(grid.cells)

@inline getnodes(grid::Grid) = grid.nodes
@inline getnodes(grid::Grid, v::Union{Int, Vector{Int}}) = grid.nodes[v]
@inline getnodes(grid::Grid, set::String) = grid.nodes[grid.nodesets[set]]
@inline getnnodes(grid::Grid) = length(grid.nodes)

@inline getcellset(grid::Grid, set::String) = grid.cellsets[set]
@inline getcellsets(grid::Grid) = grid.cellsets

@inline getnodeset(grid::Grid, set::String) = grid.nodesets[set]
@inline getnodesets(grid::Grid) = grid.nodesets

@inline getfaceset(grid::Grid, set::String) = grid.facesets[set]
@inline getfacesets(grid::Grid) = grid.facesets

n_faces_per_cell(grid::Grid) = nfaces(eltype(grid.cells))
getfacelist(grid::Grid) = getfacelist(eltype(grid.cells))

# Transformations

function transform!(g::Grid, f::Function)
    c = similar(g.nodes)
    for i in 1:length(c)
        c[i] = Node(f(g.nodes[i].x))
    end
    copy!(g.nodes, c)
    g
end

# Sets

_check_nodesetname(grid, name) = haskey(grid.nodesets, name) && throw(ArgumentError("There already exists a nodeset with the name: $name"))
_warn_emptyset(set) = length(set) == 0 && warn("no entities added to set")

function addcellset!(grid::Grid, name::String, cellid::Union{Set{Int}, Vector{Int}})
    haskey(grid.cellsets, name) && throw(ArgumentError("There already exists a cellset with the name: $name"))
    grid.cellsets[name] = Set(cellid)
    _warn_emptyset(grid.cellsets[name])
    grid
end

function addcellset!(grid::Grid, name::String, f::Function)
    cells = Set{Int}()
    for (i, cell) in enumerate(getcells(grid))
        all_true = true
        for node_idx in cell.nodes
            node = grid.nodes[node_idx]
            !f(node.x) && (all_true = false; break)
        end
        all_true && push!(cells, i)
    end
    grid.cellsets[name] = cells
    _warn_emptyset(grid.cellsets[name])
    grid
end

function addnodeset!(grid::Grid, name::String, nodeid::Union{Vector{Int}, Set{Int}})
    _check_nodesetname(grid, name)
    grid.nodesets[name] = Set(nodeid)
    _warn_emptyset(grid.nodesets[name])
    grid
end

function addnodeset!(grid::Grid, name::String, f::Function)
    _check_nodesetname(grid, name)
    nodes = Set{Int}()
    for (i, n) in enumerate(getnodes(grid))
        f(n.x) && push!(nodes, i)
    end
    grid.nodesets[name] = nodes
    _warn_emptyset(grid.nodesets[name])
    grid
end

"""
Updates the coordinate vector for a cell

    getcoordinates!(x::Vector{Vec}, grid::Grid, cell::Int)
    getcoordinates!(x::Vector{Vec}, grid::Grid, cell::CellIndex)
    getcoordinates!(x::Vector{Vec}, grid::Grid, face::FaceIndex)

** Arguments **

* `x`: a vector of `Vec`s, one for each vertex of the cell.
* `grid`: a `Grid`
* `cell`: a `CellIndex` corresponding to a `Cell` in the grid in the grid

** Results **

* `x`: the updated vector

"""
@inline function getcoordinates!{dim, T, N}(x::Vector{Vec{dim, T}}, grid::Grid{dim, N, T}, cell::Int)
    @assert length(x) == N
    @inbounds for i in 1:N
        x[i] = grid.nodes[grid.cells[cell].nodes[i]].x
    end
end
@inline getcoordinates!{dim, T, N}(x::Vector{Vec{dim, T}}, grid::Grid{dim, N, T}, cell::CellIndex) = getcoordinates!(x, grid, cell.idx)
@inline getcoordinates!{dim, T, N}(x::Vector{Vec{dim, T}}, grid::Grid{dim, N, T}, face::FaceIndex) = getcoordinates!(x, grid, face.idx[1])

"""
Returns a vector with the coordinates of the vertices of a cell

    getcoordinates(grid::Grid, cell::Int)
    getcoordinates(grid::Grid, cell::CellIndex)
    getcoordinates(grid::Grid, face::FaceIndex)

** Arguments **

* `grid`: a `Grid`
* `cell`: a `CellIndex` corresponding to a `Cell` in the grid in the grid

** Results **

* `x`: A `Vector` of `Vec`s, one for each vertex of the cell.

"""
@inline function getcoordinates{dim, N, T}(grid::Grid{dim, N, T}, cell::Int)
    nodeidx = grid.cells[cell].nodes
    return [grid.nodes[i].x for i in nodeidx]::Vector{Vec{dim, T}}
end
@inline getcoordinates(grid::Grid, cell::CellIndex) = getcoordinates(grid, cell.idx)
@inline getcoordinates(grid::Grid, face::FaceIndex) = getcoordinates(grid, face.idx[1])

# Iterate over cell vector
Base.start{dim, N}(c::Vector{Cell{dim, N}}) = 1
Base.next{dim, N}(c::Vector{Cell{dim, N}}, state) = (CellIndex(state), state + 1)
Base.done{dim, N}(c::Vector{Cell{dim, N}}, state) = state > length(c)
