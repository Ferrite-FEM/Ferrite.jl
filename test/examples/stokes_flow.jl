# Test the stokes-flow tutorial script
module TestStokesFlow
if !Sys.iswindows()
    mktempdir() do dir
        cd(dir) do
            include(joinpath(@__DIR__, "../../docs/src/literate-tutorials/stokes-flow.jl"))
        end
    end
end
end
