using JuAFEM
using FastGaussQuadrature

if VERSION >= v"0.5-"
    using Base.Test
else
    using BaseTestNext
    const Test = BaseTestNext
end

# write your own tests here
include("test_elements.jl")
include("test_materials.jl")
include("test_utilities.jl")
include("test_fevalues.jl")
