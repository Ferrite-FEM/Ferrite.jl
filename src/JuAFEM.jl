module JuAFEM

using FastGaussQuadrature
using InplaceOps

export spring1e, spring1s
export plani4e, plani4s

export hooke

include("elements/elements.jl")
include("materials/hooke.jl")

# Utilities
include("utilities/quadrature.jl")
include("utilities/shape_functions.jl")
include("utilities/linalg.jl")
include("utilities/assembler.jl")

end # module
