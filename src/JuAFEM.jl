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

export start_assemble, assemble, end_assemble

export FEValues, reinit!, shape_value, shape_gradient, shape_divergence, detJdV, get_quadrule, get_functionspace, get_geometricspace,
                 function_scalar_value, function_vector_value, function_scalar_gradient,
                 function_vector_gradient, function_vector_divergence, function_vector_symmetric_gradient
export FEFaceValues
export FunctionSpace, n_dim, ref_shape, fs_order, n_basefunctions
export Lagrange, Serendipity, RefTetrahedron, RefCube
export QuadratureRule, Dim, weights, points

"""
Represents a reference shape which quadrature rules and function spaces are defined on.
Currently, the only concrete types that subtype this type is `RefTetrahedron` and `RefCube`.
"""
abstract AbstractRefShape

immutable RefTetrahedron <: AbstractRefShape end
immutable RefCube <: AbstractRefShape end



"""
Singleton type that is similar to `Val` in Base but with a more descriptive name. Used to construct some
types in a type stable way.
"""
immutable Dim{T} end

include("function_spaces.jl")
include("quadrature.jl")
include("fe_values.jl")
include("fe_face_values.jl")
include("commons_feval_feface.jl")
include("assembler.jl")
include("VTK.jl")
include("boundary_integrals.jl")

end # module
