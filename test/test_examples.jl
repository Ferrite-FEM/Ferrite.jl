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
#
module TestComputationalHomogenization
    # Add this unregistered package here
    import Pkg; Pkg.add(Pkg.PackageSpec(url = "https://github.com/Ferrite-FEM/FerriteGmsh.jl"))
    mktempdir() do dir
        cd(dir) do
            include(joinpath(@__DIR__, "../docs/src/literate/computational_homogenization.jl"))
        end
    end
end
