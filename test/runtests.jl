using JuAFEM
using FastGaussQuadrature
using Base.Test
using ForwardDiff

include("test_utils.jl")
include("test_interpolations.jl")
include("test_cellvalues.jl")
include("test_boundaryvalues.jl")
include("test_quadrules.jl")
include("test_assemble.jl")
include("test_grid.jl")
include("test_VTK.jl")
include("test_notebooks.jl")