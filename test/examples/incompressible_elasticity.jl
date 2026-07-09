# Test the incompressible_elasticity tutorial script
module TestIncompressibleElasticity
mktempdir() do dir
    cd(dir) do
        include(joinpath(@__DIR__, "../../docs/src/literate-tutorials/incompressible_elasticity.jl"))
    end
end
end
