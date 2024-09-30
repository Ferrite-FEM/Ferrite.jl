module Ferrite

using Reexport: @reexport
@reexport using Tensors

using Base:
    @propagate_inbounds
using EnumX:
    EnumX, @enumx
using LinearAlgebra:
    LinearAlgebra, Symmetric, cholesky, det, norm, pinv, tr
using NearestNeighbors:
    NearestNeighbors, KDTree, knn
using OrderedCollections:
    OrderedSet
using SparseArrays:
    SparseArrays, SparseMatrixCSC, nonzeros, nzrange, rowvals, AbstractSparseMatrixCSC
using StaticArrays:
    StaticArrays, MArray, MMatrix, SArray, SMatrix, SVector
using WriteVTK:
    WriteVTK, VTKCellTypes
using Tensors:
    Tensors, AbstractTensor, SecondOrderTensor, SymmetricTensor, Tensor, Vec, gradient,
    rotation_tensor, symmetric, tovoigt!, hessian, otimesu
using ForwardDiff:
    ForwardDiff

include("CollectionsOfViews.jl")
using .CollectionsOfViews:
    CollectionsOfViews, ArrayOfVectorViews, push_at_index!, ConstructionBuffer

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
struct RefHypercube{refdim} <: AbstractRefShape{refdim} end
struct RefSimplex{refdim}   <: AbstractRefShape{refdim} end
const RefLine          = RefHypercube{1}
const RefQuadrilateral = RefHypercube{2}
const RefHexahedron    = RefHypercube{3}
const RefTriangle      = RefSimplex{2}
const RefTetrahedron   = RefSimplex{3}
struct RefPrism         <: AbstractRefShape{3} end
struct RefPyramid       <: AbstractRefShape{3} end

"""
    Ferrite.getrefdim(RefShape::Type{<:AbstractRefShape})

Get the dimension of the reference shape
"""
getrefdim(::Type{<:AbstractRefShape}) # To get correct doc filtering
getrefdim(::Type{<:AbstractRefShape{rdim}}) where rdim = rdim

abstract type AbstractCell{refshape <: AbstractRefShape} end

abstract type AbstractValues end
abstract type AbstractCellValues <: AbstractValues end
abstract type AbstractFacetValues <: AbstractValues end

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

"""
A `FacetIndex` wraps an (Int, Int) and defines a local facet by pointing to a (cell, facet).
"""
struct FacetIndex <: BoundaryIndex
    idx::Tuple{Int,Int} # cell and side
end

const AbstractVecOrSet{T} = Union{AbstractSet{T}, AbstractVector{T}}
const IntegerCollection = AbstractVecOrSet{<:Integer}

include("utils.jl")
include("PoolAllocator.jl")

# Matrix/Vector utilities
include("arrayutils.jl")

# Interpolations
include("interpolations.jl")

# Quadrature
include("Quadrature/quadrature.jl")

# FEValues
struct ValuesUpdateFlags{FunDiffOrder, GeoDiffOrder, DetJdV} end # Default constructor in common_values.jl
include("FEValues/GeometryMapping.jl")
include("FEValues/FunctionValues.jl")
include("FEValues/CellValues.jl")
include("FEValues/FacetValues.jl")
include("FEValues/InterfaceValues.jl")
include("FEValues/PointValues.jl")
include("FEValues/common_values.jl")
include("FEValues/face_integrals.jl")

# Grid
include("Grid/grid.jl")
include("Grid/topology.jl")
include("Grid/utils.jl")
include("Grid/grid_generators.jl")
include("Grid/coloring.jl")

# Dofs
include("Dofs/DofHandler.jl")
include("Dofs/ConstraintHandler.jl")
include("Dofs/apply_analytical.jl")
include("Dofs/sparsity_pattern.jl")
include("Dofs/block_sparsity_pattern.jl")
include("Dofs/DofRenumbering.jl")

include("iterators.jl")

# Assembly
include("assembler.jl")

# Projection
include("L2_projection.jl")

# Export
include("Export/VTK.jl")

# Point Evaluation
include("PointEvalHandler.jl")

# Other
include("deprecations.jl")
include("docs.jl")

end # module
