# Test the hyperelasticity tutorial script
module TestHyperElasticity
    mktempdir() do dir
        cd(dir) do
            include(joinpath(@__DIR__, "../../docs/src/literate-tutorials/hyperelasticity.jl"))
        end
    end
end
