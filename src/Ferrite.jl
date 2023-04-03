module Ferrite
using Reexport
@reexport using Tensors
@reexport using WriteVTK

using LinearAlgebra
using SparseArrays
using Base: @propagate_inbounds
using NearestNeighbors
using EnumX

include("exports.jl")

"""
Represents a reference shape which quadrature rules and interpolations are defined on.
Currently, the only concrete types that subtype this type are `RefCube` in 1, 2 and 3 dimensions,
and `RefTetrahedron` in 2 and 3 dimensions.
"""
abstract type AbstractRefShape end

struct RefTetrahedron <: AbstractRefShape end
struct RefCube <: AbstractRefShape end
struct RefPrism <: AbstractRefShape end

"""
Abstract type which has `CellValues` and `FaceValues` as subtypes
"""
abstract type Values{dim,T,refshape} end
abstract type CellValues{dim,T,refshape} <: Values{dim,T,refshape} end
abstract type FaceValues{dim,T,refshape} <: Values{dim,T,refshape} end

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
include("FEValues/CellMultiValues.jl")

# Grid
include("Grid/grid.jl")
include("Grid/grid_generators.jl")
include("Grid/coloring.jl")

# Dofs
include("Dofs/MixedDofHandler.jl")
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

end # module
