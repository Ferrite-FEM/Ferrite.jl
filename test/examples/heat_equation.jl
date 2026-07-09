# Test the heat_equation tutorial script
module TestHeatEquationExample
    mktempdir() do dir
        cd(dir) do
            include(joinpath(@__DIR__, "../../docs/src/literate-tutorials/heat_equation.jl"))
        end
    end
end
