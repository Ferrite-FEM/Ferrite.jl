using JuAFEM
using Tensors
using Base.Test
using ForwardDiff

include("test_utils.jl")
include("test_interpolations.jl")
include("test_cellvalues.jl")
include("test_facevalues.jl")
include("test_quadrules.jl")
include("test_assemble.jl")
include("test_grid.jl")
include("test_VTK.jl")
if VERSION.minor == 6
    include("test_notebooks.jl")
end

# Build the docs
include("../docs/make.jl")
