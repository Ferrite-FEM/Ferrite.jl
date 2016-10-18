# Grid types
export Node, Cell, Grid

# Cell typealias
export Line, QuadraticLine,
       Triangle, QuadraticTriangle, Quadrilateral, QuadraticQuadrilateral,
       Tetrahedron, QuadraticTetrahedron, Hexahedron, QuadraticHexahedron

# Grid utilities
export cells, nodes, cell_coordinates, n_cells, n_nodes

#########################
# Main types for meshes #
#########################
"""
A `Node` is a point in space.
"""
immutable Node{dim, T}
    x::Vec{dim, T}
end

"""
A `Cell` is a sub-domain defined by a collection of `Node`s as vertices.
"""
immutable Cell{dim, N}
    id::Int
    nodes::NTuple{N, Int}
end

"""
A `Grid` is a collection of `Cells` and `Node`s which covers the computational domain.
"""
immutable Grid{dim, N, T <: Real}
    cells::Vector{Cell{dim, N}}
    nodes::Vector{Node{dim, T}}
    # cell_sets # Dict? with symbol/name => Vector of id's
    # node_sets # Dict?
    # interpolation::FunctionSpace
end

# Helper functions
(::Type{Cell{dim}}){dim,N}(id::Int, nodes::NTuple{N}) = Cell{dim,N}(id, nodes)


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

"""
Returns a vector with the coordinates of the vertices of a cell

    cell_coordinates{dim, N}(grid::Grid{dim, N}, cell::Cell{dim, N})

** Arguments **

* `grid`: a `Grid`
* `cell`: a `Cell` in the grid

** Results **

* `x`: A `Vector` of `Vec`s, one for each vertex of the cell.

"""
@inline cell_coordinates{dim, N}(grid::Grid{dim, N}, cell::Cell{dim, N}) = cell_coordinates(grid, cell.id)

@inline function cell_coordinates{dim, N, T}(grid::Grid{dim, N, T}, cell_id::Int)
    nodeidx = grid.cells[cell_id].nodes
    return [grid.nodes[i].x for i in nodeidx]::Vector{Vec{dim, T}}
end
