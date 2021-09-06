# Test the scripts

module TestHeatEquationExample
    mktempdir() do dir
        cd(dir) do
            include(joinpath(@__DIR__, "../docs/src/literate/heat_equation.jl"))
        end
    end
end

module TestIncompressibleElasticity
    mktempdir() do dir
        cd(dir) do
            include(joinpath(@__DIR__, "../docs/src/literate/incompressible_elasticity.jl"))
        end
    end
end

module TestHyperElasticity
    mktempdir() do dir
        cd(dir) do
            include(joinpath(@__DIR__, "../docs/src/literate/hyperelasticity.jl"))
        end
    end
end

module TestQuasiIncompressibleHyperElasticity
    mktempdir() do dir
        cd(dir) do
            include(joinpath(@__DIR__, "../docs/src/literate/quasi_incompressible_hyperelasticity.jl"))
        end
    end
end