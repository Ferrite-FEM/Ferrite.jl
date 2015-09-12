module JuAFEM
using InplaceOps
using FastGaussQuadrature
using Requires
@lazymod Winston
using Compat
using Devectorize
import Base: LinAlg.chksquare, show

# Elements
export spring1e, spring1s
export plani4e, plani8e, soli8e, plante
export flw2i4e
export bar2e, bar2s

# Materials
export hooke

# Utilities
export solve_eq_sys, solveq
export extract_eldisp, extract
export start_assemble, assemble, assem, end_assemble, eldraw2, eldisp2, gen_quad_mesh

include("types.jl")
include("materials/hooke.jl")
include("utilities/utilities.jl")
include("elements/elements.jl")

end # module
