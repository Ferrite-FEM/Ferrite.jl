module Ferrite
using Reexport
@reexport using Tensors
@reexport using WriteVTK

using LinearAlgebra
using SparseArrays
using StaticArrays
using Base: @propagate_inbounds
using NearestNeighbors
using EnumX

include("exports.jl")

"""
    AbstractRefShape{refdim}

Supertype for all reference shapes, with reference dimension `refdim`. Reference shapes are
used to define grid cells, shape functions, and quadrature rules. Currently existing
reference shapes are: [`RefLine`](@ref), [`RefTriangle`](@ref), [`RefQuadrilateral`](@ref),
[`RefTetrahedron`](@ref), [`RefHexahedron`](@ref), [`RefPrism`](@ref).
"""
abstract type AbstractRefShape{refdim} end

# See src/docs.jl for detailed documentation
struct RefLine          <: AbstractRefShape{1} end
struct RefTriangle      <: AbstractRefShape{2} end
struct RefQuadrilateral <: AbstractRefShape{2} end
struct RefTetrahedron   <: AbstractRefShape{3} end
struct RefHexahedron    <: AbstractRefShape{3} end
struct RefPrism         <: AbstractRefShape{3} end

# TODO: Update interpolation definitions and enable deprecation
const RefCube = RefHexahedron

"""
Abstract type which has `CellValues` and `FaceValues` as subtypes. Each Value type is
equipped with the spatial dimension `sdim` of the grid, the reference elements' dimension
`rdim`, the data type for the values `T` and the reference element's geometry `refshape`.
"""
abstract type Values{sdim,rdim,T,refshape} end
abstract type CellValues{sdim,rdim,T,refshape} <: Values{sdim,rdim,T,refshape} end
abstract type FaceValues{sdim,rdim,T,refshape} <: Values{sdim,rdim,T,refshape} end

"""
Abstract type which is used as identifier for faces, edges and verices
"""
abstract type BoundaryIndex end

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

include("utils.jl")

# Matrix/Vector utilities
include("arrayutils.jl")

# Interpolations
include("interpolations.jl")

# Quadrature
include("Quadrature/quadrature.jl")

# FEValues
include("FEValues/cell_values.jl")
include("FEValues/face_values.jl")
include("PointEval/point_values.jl")
include("FEValues/common_values.jl")
include("FEValues/face_integrals.jl")

# Grid
include("Grid/grid.jl")
include("Grid/grid_generators.jl")
include("Grid/coloring.jl")

# Dofs
include("Dofs/DofHandler.jl")
include("Dofs/ConstraintHandler.jl")
include("Dofs/apply_analytical.jl")
include("Dofs/sparsity_pattern.jl")
include("Dofs/DofRenumbering.jl")

include("iterators.jl")

# Assembly
include("assembler.jl")

# Projection
include("L2_projection.jl")

# Export
include("Export/VTK.jl")

# Point Evaluation
include("PointEval/PointEvalHandler.jl")

# Other
include("deprecations.jl")
include("docs.jl")

end # module
