# Grid types
export Node, Cell, CellIndex, CellBoundary, CellBoundaryIndex, Grid

# Cell typealias
export Line, QuadraticLine,
       Triangle, QuadraticTriangle, Quadrilateral, QuadraticQuadrilateral,
       Tetrahedron, QuadraticTetrahedron, Hexahedron, QuadraticHexahedron

# Grid utilities
export cells, nodes, coordinates, n_cells, n_nodes, cell_type

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

typealias CellBoundaryIndex CellBoundary

"""
A `Grid` is a collection of `Cells` and `Node`s which covers the computational domain.
"""
immutable Grid{dim, N, T <: Real}
    cells::Vector{Cell{dim, N}}
    nodes::Vector{Node{dim, T}}
    cellbounds::Vector{CellBoundary}
    cellsets::Dict{Symbol, Vector{Int}}
    nodesets::Dict{Symbol, Vector{Int}}
    cellboundsets::Dict{Symbol, Vector{Int}}
end

function Grid{dim, N, T}(cells::Vector{Cell{dim, N}}, nodes::Vector{Node{dim, T}}, cellboundaries::Vector{CellBoundary};
                         cellsets::Dict{Symbol, Vector{Int}}=Dict{Symbol, Vector{Int}}(),
                         nodesets::Dict{Symbol, Vector{Int}}=Dict{Symbol, Vector{Int}}(),
                         cellboundsets::Dict{Symbol, Vector{Int}}=Dict{Symbol, Vector{Int}}())
    return Grid(cells, nodes, cellboundaries, cellsets, nodesets, cellboundsets)
end

# Typealias for commonly used cells
typealias Line Cell{1, 2}
typealias QuadraticLine Cell{1, 3}

typealias Triangle Cell{2, 3}
typealias QuadraticTriangle Cell{2, 6}
typealias Quadrilateral Cell{2, 4}
typealias QuadraticQuadrilateral Cell{2, 9}

typealias Tetrahedron Cell{3, 4}
# typealias QuadraticTetrahedron Cell{3, 10} # Doesn't exist in JuAFEM yet
typealias Hexahedron Cell{3, 8}
# typealias QuadraticHexahedron Cell{3, 20} # Doesn't exist in JuAFEM yet

##########################
# Grid utility functions #
##########################
@inline cells(grid::Grid) = grid.cells
@inline nodes(grid::Grid) = grid.nodes
@inline n_cells(grid::Grid) = length(grid.cells)
@inline n_nodes(grid::Grid) = length(grid.nodes)
@inline cell_type(grid::Grid) = eltype(grid.cells)

# Iterate over cell vector
Base.start{dim, N}(c::Vector{Cell{dim, N}}) = 1
Base.next{dim, N}(c::Vector{Cell{dim, N}}, state) = (CellIndex(state), state + 1)
Base.done{dim, N}(c::Vector{Cell{dim, N}}, state) = state > length(c)
# Base.eltype{dim, N}(c::Type{Vector{Cell{dim, N}}}) = CellIndex

"""
Returns a vector with the coordinates of the vertices of a cell

    coordinates{dim, N}(grid::Grid{dim, N}, cell::CellIndex)

** Arguments **

* `grid`: a `Grid`
* `cell`: a `CellIndex` corresponding to a `Cell` in the grid in the grid

** Results **

* `x`: A `Vector` of `Vec`s, one for each vertex of the cell.

"""
@inline function coordinates{dim, N, T}(grid::Grid{dim, N, T}, cell::Int)
    nodeidx = grid.cells[cell].nodes
    return [grid.nodes[i].x for i in nodeidx]::Vector{Vec{dim, T}}
end
@inline coordinates{dim, N, T}(grid::Grid{dim, N, T}, cell::CellIndex) = coordinates(grid, cell.idx)
@inline coordinates{dim, N, T}(grid::Grid{dim, N, T}, boundary::CellBoundaryIndex) = coordinates(grid, boundary.idx[1])

# reinit!(fe_cv::FECellValues, grid::Grid, cell::CellIndex) = reinit!(fe_cv, coordinates(grid, cell))
# reinit!(fe_bv::FEBoundaryValues, grid::Grid, boundary::CellBoundaryIndex) = reinit!(fe_bv, coordinates(grid, boundary), boundary.idx[2])


function get_cell_boundary_set(grid::Grid, set::Symbol)
    return grid.cellbounds[grid.cellboundsets[set]]
end
