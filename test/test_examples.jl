# Test the scripts

module TestHeatEquationExample
    mktempdir() do dir
        cd(dir) do
            include(joinpath(@__DIR__, "../docs/src/literate-tutorials/heat_equation.jl"))
        end
    end
end

module TestIncompressibleElasticity
    mktempdir() do dir
        cd(dir) do
            include(joinpath(@__DIR__, "../docs/src/literate-tutorials/incompressible_elasticity.jl"))
        end
    end
end

module TestHyperElasticity
    mktempdir() do dir
        cd(dir) do
            include(joinpath(@__DIR__, "../docs/src/literate-tutorials/hyperelasticity.jl"))
        end
    end
end

module TestQuasiIncompressibleHyperElasticity
    mktempdir() do dir
        cd(dir) do
            include(joinpath(@__DIR__, "../docs/src/literate-gallery/quasi_incompressible_hyperelasticity.jl"))
        end
    end
end

# module TestNavierStokesDiffeqIntegration
#     mktempdir() do dir
#         cd(dir) do
#             include(joinpath(@__DIR__, "../docs/src/literate-tutorials/ns_vs_diffeq.jl"))
#         end
#     end
# end

module TestComputationalHomogenization
    mktempdir() do dir
        cd(dir) do
            # Add already downloaded file to allow running test suite offline
            mesh_file = joinpath(@__DIR__, "../docs/src/tutorials/periodic-rve.msh")
            isfile(mesh_file) && cp(mesh_file, joinpath(dir, basename(mesh_file)))
            include(joinpath(@__DIR__, "../docs/src/literate-tutorials/computational_homogenization.jl"))
        end
    end
end

module TestStokesFlow
if !Sys.iswindows()
    mktempdir() do dir
        cd(dir) do
            include(joinpath(@__DIR__, "../docs/src/literate-tutorials/stokes-flow.jl"))
        end
    end
end
end
