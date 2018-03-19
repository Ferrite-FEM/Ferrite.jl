# Just make sure they run for now.
module TestHeatEquationExample
    mktempdir() do dir
        cd(dir) do
            include(joinpath(@__DIR__, "..","docs", "src", "examples", "heat_equation.jl"))
        end
    end
end
