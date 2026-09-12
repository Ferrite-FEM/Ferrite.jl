# Test the heat_equation tutorial script
module TestPlateEquationExample
mktempdir() do dir
    cd(dir) do
        include(joinpath(@__DIR__, "../../docs/src/literate-gallery/plate_equation.jl"))
    end
end
end
