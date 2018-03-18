using JuAFEM
using Tensors
using Base.Test
using ForwardDiff
import SHA

include("test_utils.jl")
include("test_interpolations.jl")
include("test_cellvalues.jl")
include("test_facevalues.jl")
include("test_quadrules.jl")
include("test_grid_dofhandler_vtk.jl")
if VERSION.minor == 6
    include("test_notebooks.jl")
    include("test_examples.jl")
end

# Build the docs
include("../docs/make.jl")
