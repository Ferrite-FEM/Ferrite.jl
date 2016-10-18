# Grid types
export Node, Cell, CellIndex, Grid

# Cell typealias
export Line, QuadraticLine,
       Triangle, QuadraticTriangle, Quadrilateral, QuadraticQuadrilateral,
       Tetrahedron, QuadraticTetrahedron, Hexahedron, QuadraticHexahedron

# Grid utilities
export cells, nodes, coordinates, n_cells, n_nodes

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
A `Cell` is a sub-domain defined by a collection of `Node`s as vertices.
"""
immutable Cell{dim, N}
    nodes::NTuple{N, Int}
end
(::Type{Cell{dim}}){dim,N}(nodes::NTuple{N}) = Cell{dim,N}(nodes)

immutable CellIndex
    idx::Int
end

immutable BoundaryIndex
    idx::CellIndex
    side::Int
end

"""
A `Grid` is a collection of `Cells` and `Node`s which covers the computational domain.
"""
immutable Grid{dim, N, T <: Real}
    cells::Vector{Cell{dim, N}}
    nodes::Vector{Node{dim, T}}
    boundaries::Dict{Int, Vector{Tuple{CellIndex,Int}}}
    # cell_sets # Dict? with symbol/name => Vector of id's
    # node_sets # Dict?
    # interpolation::FunctionSpace
end

#####################################
# Typealias for commonly used cells #
#####################################
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

# Iterate over cell vector
Base.start{dim, N}(c::Vector{Cell{dim, N}}) = 1
Base.next{dim, N}(c::Vector{Cell{dim, N}}, state) = (CellIndex(state), state + 1)
Base.done{dim, N}(c::Vector{Cell{dim, N}}, state) = state > length(c)
# Base.eltype{dim, N}(c::Type{Vector{Cell{dim, N}}}) = CellIndex

# Iterate over boundary vector
Base.start(b::Vector{Tuple{CellIndex,Int}}) = 1
Base.next(b::Vector{Tuple{CellIndex,Int}}, state) = (BoundaryIndex(b[state]...), state + 1)
Base.done(b::Vector{Tuple{CellIndex,Int}}, state) = state > length(b)
# Base.eltype(b::Vector{Tuple{CellIndex,Int}}) = BoundaryIndex

"""
Returns a vector with the coordinates of the vertices of a cell

    coordinates{dim, N}(grid::Grid{dim, N}, cell::CellIndex)

** Arguments **

* `grid`: a `Grid`
* `cell`: a `CellIndex` corresponding to a `Cell` in the grid in the grid

** Results **

* `x`: A `Vector` of `Vec`s, one for each vertex of the cell.

"""
@inline function coordinates{dim, N, T}(grid::Grid{dim, N, T}, cell::CellIndex)
    nodeidx = grid.cells[cell.idx].nodes
    return [grid.nodes[i].x for i in nodeidx]::Vector{Vec{dim, T}}
end

@inline coordinates{dim, N, T}(grid::Grid{dim, N, T}, boundary::BoundaryIndex) = coordinates(grid, boundary.idx)

reinit!(fe_cv::FECellValues, grid::Grid, cell::CellIndex) = reinit!(fe_cv, coordinates(grid, cell))
reinit!(fe_bv::FEBoundaryValues, grid::Grid, boundary::BoundaryIndex) = reinit!(fe_bv, coordinates(grid, boundary), boundary.side)
