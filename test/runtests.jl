using JuAFEM
using FactCheck
using FastGaussQuadrature


# write your own tests here
include("test_elements.jl")
include("test_materials.jl")
include("test_utilities.jl")
include("test_fevalues.jl")

FactCheck.exitstatus()