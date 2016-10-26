# Grid types
export Node, Cell, CellIndex, CellBoundary, CellBoundaryIndex, Grid

# Cell type alias
export Line, QuadraticLine,
       Triangle, QuadraticTriangle, Quadrilateral, QuadraticQuadrilateral,
       Tetrahedron, QuadraticTetrahedron, Hexahedron, QuadraticHexahedron

# Grid utilities
export getcells, getncells, getnodes, getnnodes, getcelltype,
       getcellset, getnodeset, getcellboundaryset, getcoordinates

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

"""
A `Cell` is a sub-domain defined by a collection of `Node`s as it's vertices.
"""
immutable Cell{dim, N}
    nodes::NTuple{N, Int}
end
(::Type{Cell{dim}}){dim,N}(nodes::NTuple{N}) = Cell{dim,N}(nodes)

# Typealias for commonly used cells
typealias Line Cell{1, 2}
typealias QuadraticLine Cell{1, 3}

typealias Triangle Cell{2, 3}
typealias QuadraticTriangle Cell{2, 6}
typealias Quadrilateral Cell{2, 4}
typealias QuadraticQuadrilateral Cell{2, 9}

typealias Tetrahedron Cell{3, 4}
typealias QuadraticTetrahedron Cell{3, 10} # Function interpolation for this doesn't exist in JuAFEM yet
typealias Hexahedron Cell{3, 8}
typealias QuadraticHexahedron Cell{3, 20} # Function interpolation for this doesn't exist in JuAFEM yet

"""
A `CellIndex` is returned when looping over the cells in a grid.
"""
immutable CellIndex
    idx::Int
end

"""
A `CellBoundary` is a sub-domain of the boundary defined by the cell and the side.
"""
immutable CellBoundary
    idx::Tuple{Int, Int} # cell and side
end

"""
A `CellBoundaryIndex` is returned when looping over cell boundaries of the grid.
"""
typealias CellBoundaryIndex CellBoundary

"""
A `Grid` is a collection of `Cells` and `Node`s which covers the computational domain.
"""
immutable Grid{dim, N, T <: Real}
    cells::Vector{Cell{dim, N}}
    nodes::Vector{Node{dim, T}}
    cellboundaries::Vector{CellBoundary}
    cellsets::Dict{Symbol, Vector{Int}}
    nodesets::Dict{Symbol, Vector{Int}}
    cellboundarysets::Dict{Symbol, Vector{Int}}
end

function Grid{dim, N, T}(cells::Vector{Cell{dim, N}}, nodes::Vector{Node{dim, T}}, cellboundaries::Vector{CellBoundary};
                         cellsets::Dict{Symbol, Vector{Int}}=Dict{Symbol, Vector{Int}}(),
                         nodesets::Dict{Symbol, Vector{Int}}=Dict{Symbol, Vector{Int}}(),
                         cellboundarysets::Dict{Symbol, Vector{Int}}=Dict{Symbol, Vector{Int}}())
    return Grid(cells, nodes, cellboundaries, cellsets, nodesets, cellboundarysets)
end

##########################
# Grid utility functions #
##########################
@inline getcells(grid::Grid) = grid.cells
@inline getncells(grid::Grid) = length(grid.cells)
@inline getnodes(grid::Grid) = grid.nodes
@inline getnnodes(grid::Grid) = length(grid.nodes)
@inline getcelltype(grid::Grid) = eltype(grid.cells)
@inline getcellset(grid::Grid, set::Symbol) = grid.cells[grid.cellsets[set]]
@inline getnodeset(grid::Grid, set::Symbol) = grid.nodes[grid.nodesets[set]]
@inline getcellboundaryset(grid::Grid, set::Symbol) = grid.cellboundaries[grid.cellboundarysets[set]]

function addcellset!(grid::Grid, name::Symbol, cellid::Vector{Int})
    haskey(grid.cellsets, name) && throw(ArgumentError("There already exists a cellset with the name: $name"))
    grid.cellsets[name] = cellid
    nothing
end
function addnodeset!(grid::Grid, name::Symbol, nodeid::Vector{Int})
    haskey(grid.nodesets, name) && throw(ArgumentError("There already exists a nodeset with the name: $name"))
    grid.nodesets[name] = nodeid
    nothing
end

"""
Returns a vector with the coordinates of the vertices of a cell

    getcoordinates(grid::Grid, cell::CellIndex)
    getcoordinates(grid::Grid, boundary::CellBoundaryIndex)

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
@inline getcoordinates(grid::Grid, cell::CellIndex) = coordinates(grid, cell.idx)
@inline getcoordinates(grid::Grid, boundary::CellBoundaryIndex) = coordinates(grid, boundary.idx[1])

# Iterate over cell vector
Base.start{dim, N}(c::Vector{Cell{dim, N}}) = 1
Base.next{dim, N}(c::Vector{Cell{dim, N}}, state) = (CellIndex(state), state + 1)
Base.done{dim, N}(c::Vector{Cell{dim, N}}, state) = state > length(c)
