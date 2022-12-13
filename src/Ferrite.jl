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

# Adaptiviy
include(joinpath("Adaptivity", "AdaptiveCells.jl"))

# Dofs
include("Dofs/DofHandler.jl")
include("Dofs/MixedDofHandler.jl")
include("Dofs/ConstraintHandler.jl")
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
