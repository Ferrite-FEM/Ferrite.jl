# Test the quasi_incompressible_hyperelasticity gallery script
module TestQuasiIncompressibleHyperElasticity
mktempdir() do dir
    cd(dir) do
        include(joinpath(@__DIR__, "../../docs/src/literate-gallery/quasi_incompressible_hyperelasticity.jl"))
    end
end
end
