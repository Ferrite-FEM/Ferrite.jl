module Ferrite

using Reexport: @reexport
@reexport using Tensors

using Base:
    @propagate_inbounds
using EnumX:
    EnumX, @enumx
using LinearAlgebra:
    LinearAlgebra, Symmetric, Transpose, cholesky, det, issymmetric, norm,
    pinv, tr
using NearestNeighbors:
    NearestNeighbors, KDTree, knn
using OrderedCollections:
    OrderedSet
using SparseArrays:
    SparseArrays, SparseMatrixCSC, nonzeros, nzrange, rowvals, sparse
using StaticArrays:
    StaticArrays, MArray, MMatrix, SArray, SMatrix, SVector
using WriteVTK:
    WriteVTK, VTKCellTypes
using Tensors:
    Tensors, AbstractTensor, SecondOrderTensor, SymmetricTensor, Tensor, Vec, gradient,
    rotation_tensor, symmetric, tovoigt!, hessian, otimesu
using ForwardDiff:
    ForwardDiff


include("exports.jl")

"""
    AbstractRefShape{refdim}

Supertype for all reference shapes with reference dimension `refdim`. Reference shapes are
used to define grid cells, interpolations, and quadrature rules.

Currently implemented reference shapes are: [`RefLine`](@ref), [`RefTriangle`](@ref),
[`RefQuadrilateral`](@ref), [`RefTetrahedron`](@ref), [`RefHexahedron`](@ref),
[`RefPrism`](@ref).

# Examples
```julia
# Create a 1st order Lagrange interpolation on the reference triangle
interpolation = Lagrange{2, RefTriangle, 1}()

# Create a 2nd order quadrature rule for the reference quadrilateral
quad_rule = Quadrature{2, RefQuadrilateral}(2)
```

Implementation details can be found in the devdocs section on [Reference cells](@ref).
"""
abstract type AbstractRefShape{refdim} end

"""
    RefHypercube{dim} <: AbstractRefShape{dim}

Reference shape for a `dim`-dimensional hypercube. See [`AbstractRefShape`](@ref)
documentation for details.
"""
struct RefHypercube{refdim} <: AbstractRefShape{refdim} end

"""
    RefSimplex{dim} <: AbstractRefShape{dim}

Reference shape for a `dim`-dimensional simplex. See [`AbstractRefShape`](@ref)
documentation for details.
"""
struct RefSimplex{refdim} <: AbstractRefShape{refdim} end

"""
    RefLine <: AbstractRefShape{1}

Reference line/interval, alias for [`RefHypercube{1}`](@ref). See [`AbstractRefShape`](@ref)
documentation for details.
"""
const RefLine = RefHypercube{1}

"""
    RefTriangle <: AbstractRefShape{2}

Reference triangle, alias for [`RefSimplex{2}`](@ref). See [`AbstractRefShape`](@ref)
documentation for details.
"""
const RefTriangle = RefSimplex{2}

"""
    RefQuadrilateral <: AbstractRefShape{2}

Reference quadrilateral, alias for [`RefHypercube{2}`](@ref). See [`AbstractRefShape`](@ref)
documentation for details.
"""
const RefQuadrilateral = RefHypercube{2}

"""
    RefTetrahedron <: AbstractRefShape{3}

Reference tetrahedron, alias for [`RefSimplex{3}`](@ref). See [`AbstractRefShape`](@ref)
documentation for details.
"""
const RefTetrahedron = RefSimplex{3}

"""
    RefHexahedron <: AbstractRefShape{3}

Reference hexahedron, alias for [`RefHypercube{3}`](@ref). See [`AbstractRefShape`](@ref)
documentation for details.
"""
const RefHexahedron = RefHypercube{3}

"""
    RefPrism <: AbstractRefShape{3}

Reference prism. See [`AbstractRefShape`](@ref) documentation for details.
"""
struct RefPrism <: AbstractRefShape{3} end


"""
    Ferrite.getrefdim(RefShape::Type{<:AbstractRefShape})

Get the dimension of the reference shape
"""
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

# Matrix/Vector utilities
include("arrayutils.jl")

# Interpolations
include("interpolations.jl")

# Quadrature
include("Quadrature/quadrature.jl")

# FEValues
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

end # module
