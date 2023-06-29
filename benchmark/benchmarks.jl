using BenchmarkTools
using Ferrite

const selected = get(ENV, "FERRITE_SELECTED_BENCHMARKS", "all")
const runall = selected == "all"

include("helper.jl")

const SUITE = BenchmarkGroup()

if runall || selected == "mesh"
    include("benchmarks-mesh.jl")
end
if runall || selected == "dofs"
    include("benchmarks-dofs.jl")
end
if runall || selected == "assembly"
    include("benchmarks-assembly.jl")
end
if runall || selected == "boundary-conditions"
    include("benchmarks-boundary-conditions.jl")
end
