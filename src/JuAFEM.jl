__precompile__()

module JuAFEM
using Compat
using ForwardDiff
using Reexport
@reexport using Tensors
@reexport using WriteVTK

import Base: show, push!
import WriteVTK: vtk_grid, vtk_point_data, DatasetFile

# Utilities

export start_assemble, assemble!, end_assemble

export CellValues, CellScalarValues, CellVectorValues
export FaceValues, FaceScalarValues, FaceVectorValues
export ScalarValues, VectorValues

export reinit!, shape_value, shape_gradient, shape_symmetric_gradient, shape_divergence, getdetJdV,
       function_value, function_gradient, function_symmetric_gradient, function_divergence, spatial_coordinate
export getfacenumber, getnormal
export Interpolation, getdim, getrefshape, getorder, getnbasefunctions, getnquadpoints
export Lagrange, Serendipity, RefTetrahedron, RefCube
export QuadratureRule, getweights, getpoints
export getVTKtype

"""
Represents a reference shape which quadrature rules and interpolations are defined on.
Currently, the only concrete types that subtype this type are `RefCube` in 1,2 and 3 dimensions,
and `RefTetrahedron` in 2 and 3 dimensions.
"""
@compat abstract type AbstractRefShape end

immutable RefTetrahedron <: AbstractRefShape end
immutable RefCube <: AbstractRefShape end

"""
Abstract type which has `CellValues` and `FaceValues` as subtypes
"""
@compat abstract type Values{dim, T, refshape} end
@compat abstract type CellValues{dim, T, refshape} <: Values{dim, T, refshape} end
@compat abstract type FaceValues{dim, T, refshape} <: Values{dim, T, refshape} end

include("utils.jl")
include("interpolations.jl")

# Quadrature
include(joinpath("Quadrature", "quadrature.jl"))

# FEValues
include(joinpath("FEValues","cell_values.jl"))
include(joinpath("FEValues","face_values.jl"))
include(joinpath("FEValues","common_values.jl"))
include(joinpath("FEValues","face_integrals.jl"))

# Grid
include(joinpath("Grid", "grid.jl"))
include(joinpath("Grid", "grid_generators.jl"))

# Dofs
include(joinpath("Dofs", "DofHandler.jl"))
include(joinpath("Dofs", "DirichletBoundaryConditions.jl"))

# Export
include(joinpath("Export", "VTK.jl"))

# Other
include("iterators.jl")
include("assembler.jl")
include("deprecations.jl")

end # module
