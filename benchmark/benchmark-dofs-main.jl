using BenchmarkTools
using Ferrite

include("helper.jl")

const SUITE = BenchmarkGroup()

include("benchmarks-dofs.jl")
