using BenchmarkTools
using Ferrite

include("helper.jl")

const SUITE = BenchmarkGroup()

include("benchmarks-mesh.jl")
include("benchmarks-dofs.jl")
include("benchmarks-assembly.jl")
include("benchmarks-boundary-conditions.jl")
