# Test the computational_homogenization tutorial script
module TestComputationalHomogenization
    mktempdir() do dir
        cd(dir) do
            # Add already downloaded file to allow running test suite offline
            mesh_file = joinpath(@__DIR__, "../../docs/src/tutorials/periodic-rve.msh")
            isfile(mesh_file) && cp(mesh_file, joinpath(dir, basename(mesh_file)))
            include(joinpath(@__DIR__, "../../docs/src/literate-tutorials/computational_homogenization.jl"))
        end
    end
end
