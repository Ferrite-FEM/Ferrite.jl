# Each test file is self-contained (own `using`s and helper `include`s) and runs in its own
# module, so the files can also be run on their own.
using Test

@testset "PureP4est" begin
    for file in ("test_octant.jl", "test_forest.jl", "test_iterator.jl")
        @testset "$file" begin
            @eval module $(Symbol(splitext(file)[1]))
            include($(joinpath(@__DIR__, file)))
            end
        end
    end
end
