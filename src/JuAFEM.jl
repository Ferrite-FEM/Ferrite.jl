module JuAFEM

export spring1e, spring1s

export hooke

include("elements/elements.jl")
include("materials/hooke.jl")
include("quadrature.jl")

end # module
