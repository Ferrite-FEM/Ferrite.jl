#__precompile__()

module JuAFEM
using InplaceOps
using FastGaussQuadrature
using Compat
using Devectorize
using Reexport
@reexport using ContMechTensors
@reexport using WriteVTK

import Base: show
import WriteVTK: vtk_grid

# Elements
export spring1e, spring1s
export plani4e, plani8e, soli8e, plante
export plani4s, plani8s, soli8s, plants
export plani4f, plani8f, soli8f, plantf
export flw2i4e, flw2i8e, flw2te, flw3i8e
export flw2i4s, flw2i8s, flw2ts, flw3i8s
export bar2e, bar2s, bar2g

# Materials
export hooke

export vtk_grid

# Utilities
export solve_eq_sys, solveq
export extract, coordxtr, topologyxtr
export statcon
export start_assemble, assemble, assem, end_assemble, eldraw2, eldisp2, gen_quad_mesh
export FEValues, reinit!, shape_value, shape_gradient, shape_divergence, detJdV, get_quadrule, get_functionspace,
                 function_scalar_value, function_vector_value, function_scalar_gradient,
                 function_vector_gradient, function_vector_divergence, function_vector_symmetric_gradient
export n_dim, n_basefunctions
export Lagrange, Serendipity
export get_gaussrule, Dim
export FEMesh, get_element_nodes, get_element_coords, get_node_coords, get_boundary_nodes
export FEField, FEDofs, get_element_dofs, get_node_dofs, get_field, add_field, remove_field


include("materials/hooke.jl")
include("utilities/utilities.jl")
include("elements/elements.jl")

end # module
