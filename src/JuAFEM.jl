module JuAFEM
using InplaceOps
using FastGaussQuadrature
using Compat
using Devectorize
import Base: LinAlg.chksquare, show

# Elements
export spring1e, spring1s
export plani4e, plani8e, soli8e, plante
export plani4s, plani8s, soli8s, plants
export flw2i4e, flw2i8e, flw2te, flw3i8e
export flw2i4s, flw2i8s, flw2ts, flw3i8s
export bar2e, bar2s, bar2g

# Materials
export hooke

# Utilities
export solve_eq_sys, solveq
export extract, coordxtr
export statcon
export start_assemble, assemble, assem, end_assemble, eldraw2, eldisp2, gen_quad_mesh

include("types.jl")
include("materials/hooke.jl")
include("utilities/utilities.jl")
include("elements/elements.jl")

end # module
