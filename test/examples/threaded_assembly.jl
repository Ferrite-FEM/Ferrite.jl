# Test the threaded_assembly how-to script
module TestMultiThreading
    mktempdir() do dir
        cd(dir) do
            # Silence the @time output
            redirect_stdout(devnull) do
                include(joinpath(@__DIR__, "../../docs/src/literate-howto/threaded_assembly.jl"))
            end
        end
    end
end
