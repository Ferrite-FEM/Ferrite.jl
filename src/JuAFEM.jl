__precompile__()

module JuAFEM
using InplaceOps
using FastGaussQuadrature
using Compat
using Reexport
@reexport using ContMechTensors
@reexport using WriteVTK

import Base: show
import WriteVTK: vtk_grid

# Utilities

export coordxtr, topologyxtr
export start_assemble, assemble, end_assemble
export FEValues, reinit!, shape_value, shape_gradient, shape_divergence, detJdV, get_quadrule, get_functionspace,
                 function_scalar_value, function_vector_value, function_scalar_gradient,
                 function_vector_gradient, function_vector_divergence, function_vector_symmetric_gradient
export n_dim, n_basefunctions
export Lagrange, Serendipity
export QuadratureRule, GaussQuadrature, Dim, weights, points

include("geometry_types.jl")
include("function_spaces.jl")
include("quadrature.jl")
include("fe_values.jl")
include("assembler.jl")
include("coordxtr.jl")
include("VTK.jl")

end # module
