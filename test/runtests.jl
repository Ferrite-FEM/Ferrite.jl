using JuAFEM
using FastGaussQuadrature

if VERSION >= v"0.5-"
    using Base.Test
else
    using BaseTestNext
    const Test = BaseTestNext
end

include("test_fevalues.jl")
include("test_quadrules.jl")
include("test_assemble.jl")
