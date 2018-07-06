# generate examples
import Literate

EXAMPLEDIR = joinpath(@__DIR__, "src", "examples")
GENERATEDDIR = joinpath(@__DIR__, "src", "examples", "generated")
for example in readdir(EXAMPLEDIR)
    endswith(example, ".jl") || continue
    input = abspath(joinpath(EXAMPLEDIR, example))
    script = Literate.script(input, GENERATEDDIR)
    code = strip(read(script, String))
    mdpost(str) = replace(str, "@__CODE__" => code)
    Literate.markdown(input, GENERATEDDIR, postprocess = mdpost)
    Literate.notebook(input, GENERATEDDIR, execute = true)
end

# copy some figures to the build directory
cp(joinpath(@__DIR__, "../examples/figures/heat_square.png"),
   joinpath(@__DIR__, "src/examples/generated/heat_equation.png"); force = true)

cp(joinpath(@__DIR__, "../examples/figures/coloring.png"),
   joinpath(@__DIR__, "src/examples/generated/coloring.png"); force = true)

# remove any .vtu files in the generated dir (should not be deployed)
cd(GENERATEDDIR) do
    foreach(file -> endswith(file, ".vtu") && rm(file), readdir())
end
