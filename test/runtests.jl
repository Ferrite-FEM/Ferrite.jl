using JuAFEM
using FastGaussQuadrature
using Base.Test
using ForwardDiff

include("test_function_spaces.jl")
include("test_fecellvalues.jl")
include("test_feboundaryvalues.jl")
include("test_quadrules.jl")
include("test_assemble.jl")
include("test_VTK.jl")
