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

# module TestNavierStokesDiffeqIntegration
#     mktempdir() do dir
#         cd(dir) do
#             include(joinpath(@__DIR__, "../docs/src/literate/ns_vs_diffeq.jl"))
#         end
#     end
# end

module TestComputationalHomogenization
    include(joinpath(@__DIR__, "../docs/download_resources.jl"))
    mktempdir() do dir
        cd(dir) do
            cp(joinpath(@__DIR__, "../docs/src/examples/periodic-rve.msh"),
               joinpath(dir, "periodic-rve.msh")
            )
            include(joinpath(@__DIR__, "../docs/src/literate/computational_homogenization.jl"))
        end
    end
end

module TestStokesFlow
if !Sys.iswindows()
    mktempdir() do dir
        cd(dir) do
            include(joinpath(@__DIR__, "../docs/src/literate/stokes-flow.jl"))
        end
    end
end
end
