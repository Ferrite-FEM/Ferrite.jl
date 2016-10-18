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

export FECellValues, reinit!, shape_value, shape_gradient, shape_divergence, detJdV, get_quadrule, get_functionspace, get_geometricspace,
                     function_value, function_gradient, function_symmetric_gradient, function_divergence, spatial_coordinate
export FEBoundaryValues, get_boundarynumber
export FunctionSpace, functionspace_n_dim, functionspace_ref_shape, functionspace_order, n_basefunctions
export Lagrange, Serendipity, RefTetrahedron, RefCube
export QuadratureRule, weights, points
export VTK_type

"""
Represents a reference shape which quadrature rules and function spaces are defined on.
Currently, the only concrete types that subtype this type is `RefTetrahedron` and `RefCube`.
"""
abstract AbstractRefShape

immutable RefTetrahedron <: AbstractRefShape end
immutable RefCube <: AbstractRefShape end

"""
Abstract type which has `FECellValues` and `FEBoundaryValues` as subtypes
"""
abstract AbstractFEValues{dim, T, FS, GS}


include("function_spaces.jl")
include("quadrature.jl")
include("fe_cell_values.jl")
include("fe_boundary_values.jl")
include("commons_abstract_fevalues.jl")
include("assembler.jl")
include("VTK.jl")
include("boundary_integrals.jl")
include("grid.jl")
include("grid_generators.jl")
include("deprecations.jl")

end # module
