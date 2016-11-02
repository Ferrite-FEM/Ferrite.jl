__precompile__()

module JuAFEM
using FastGaussQuadrature
using Compat
using Reexport
@reexport using ContMechTensors
@reexport using WriteVTK

import Base: show
import WriteVTK: vtk_grid

# Utilities

export start_assemble, assemble!, end_assemble

export CellValues, CellScalarValues, CellVectorValues
export BoundaryValues, BoundaryScalarValues, BoundaryVectorValues
export ScalarValues, VectorValues

export reinit!, shape_value, shape_gradient, shape_symmetric_gradient, shape_divergence, getdetJdV, getquadrule, getfunctionspace, getgeometricspace,
       function_value, function_gradient, function_symmetric_gradient, function_divergence, spatial_coordinate
export getboundarynumber
export FunctionSpace, getdim, getrefshape, getorder, getnbasefunctions, getnquadpoints
export Lagrange, Serendipity, RefTetrahedron, RefCube
export QuadratureRule, getweights, getpoints
export getVTKtype

"""
Represents a reference shape which quadrature rules and function spaces are defined on.
Currently, the only concrete types that subtype this type is `RefTetrahedron` and `RefCube`.
"""
abstract AbstractRefShape

immutable RefTetrahedron <: AbstractRefShape end
immutable RefCube <: AbstractRefShape end

"""
Abstract type which has `CellValues` and `BoundaryValues` as subtypes
"""
abstract Values{dim, T, FS, GS}
abstract CellValues{dim, T, FS, GS}     <: Values{dim, T, FS, GS}
abstract BoundaryValues{dim, T, FS, GS} <: Values{dim, T, FS, GS}


include("function_spaces.jl")
include("quadrature.jl")
include("cell_values.jl")
include("boundary_values.jl")

typealias ScalarValues{dim, T, FS, GS} Union{CellScalarValues{dim, T, FS, GS}, BoundaryScalarValues{dim, T, FS, GS}}
typealias VectorValues{dim, T, FS, GS} Union{CellVectorValues{dim, T, FS, GS}, BoundaryVectorValues{dim, T, FS, GS}}

include("common_values.jl")
include("assembler.jl")
include("boundary_integrals.jl")
include("grid.jl")
include("grid_generators.jl")
include("VTK.jl")
include("deprecations.jl")

end # module
