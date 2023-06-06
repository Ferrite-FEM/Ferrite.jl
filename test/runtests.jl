using Ferrite
using Tensors
using Test
using Logging
using ForwardDiff
import SHA
using Random
using LinearAlgebra
using SparseArrays

const HAS_EXTENSIONS = isdefined(Base, :get_extension)

# https://github.com/JuliaLang/julia/pull/47749
const MODULE_CAN_BE_TYPE_PARAMETER = VERSION >= v"1.10.0-DEV.90"

if HAS_EXTENSIONS && MODULE_CAN_BE_TYPE_PARAMETER
    import Metis
end

const RUN_JET_TESTS = VERSION >= v"1.9"

if RUN_JET_TESTS
    using JET: @test_call
else
    # Just eat the macro on incompatible versions
    macro test_call(args...)
        nothing
    end
end

include("test_utils.jl")

# Unit tests
# include("test_interpolations.jl")
# include("test_cellvalues.jl")
# include("test_facevalues.jl")
# include("test_interfacevalues.jl")
# include("test_quadrules.jl")
# include("test_assemble.jl")
# include("test_dofs.jl")
# include("test_constraints.jl")
# include("test_grid_dofhandler_vtk.jl")
# include("test_abstractgrid.jl")
# include("test_grid_addboundaryset.jl")
# include("test_mixeddofhandler.jl")
include("test_l2_projection.jl")
include("test_pointevaluation.jl")
# include("test_notebooks.jl")
include("test_apply_rhs.jl")
include("test_apply_analytical.jl")
include("test_deprecations.jl")
HAS_EXTENSIONS && include("blockarrays.jl")
include("test_examples.jl")
@test all(x -> isdefined(Ferrite, x), names(Ferrite))  # Test that all exported symbols are defined

# Integration tests
include("integration/test_simple_scalar_convergence.jl")
