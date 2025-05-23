using Ferrite
using Tensors
using Test
using Logging
using ForwardDiff
import SHA
using Random
using LinearAlgebra
using SparseArrays
using StaticArrays
using OrderedCollections
using WriteVTK
import Metis
using HCubature: hcubature, hquadrature

include("test_utils.jl")

# Unit tests
include("test_collectionsofviews.jl")
include("test_refshapes.jl")
include("test_interpolations.jl")
include("test_cellvalues.jl")
include("test_facevalues.jl")
include("test_interfacevalues.jl")
include("test_quadrules.jl")
include("test_assemble.jl")
include("test_dofs.jl")
include("test_sparsity_patterns.jl")
include("test_constraints.jl")
include("test_grid_dofhandler_vtk.jl")
include("test_vtk_export.jl")
include("test_abstractgrid.jl")
include("test_grid_generators.jl")
include("test_grid_addboundaryset.jl")
include("test_mixeddofhandler.jl")
include("test_l2_projection.jl")
include("test_pointevaluation.jl")
# include("test_notebooks.jl")
include("test_apply_rhs.jl")
include("test_apply_analytical.jl")
include("PoolAllocator.jl")
include("test_deprecations.jl")
include("blockarrays.jl")
include("test_assembler_extensions.jl")
include("test_continuity.jl")
include("test_examples.jl")

@test all(x -> isdefined(Ferrite, x), names(Ferrite))  # Test that all exported symbols are defined
# # See which is not defined if fails
# for name in names(Ferrite)
#     isdefined(Ferrite, name) || @warn "Ferrite.$name is not defined but $name is exported"
# end

# Integration tests
include("integration/test_simple_scalar_convergence.jl")
