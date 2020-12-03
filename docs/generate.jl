# generate examples
import Literate

EXAMPLEDIR = joinpath(@__DIR__, "src", "literate")
GENERATEDDIR = joinpath(@__DIR__, "src", "examples")
mkpath(GENERATEDDIR)
for example in readdir(EXAMPLEDIR)
    if endswith(example, ".jl")
        input = abspath(joinpath(EXAMPLEDIR, example))
        script = Literate.script(input, GENERATEDDIR)
        code = strip(read(script, String))
        mdpost(str) = replace(str, "@__CODE__" => code)
        Literate.markdown(input, GENERATEDDIR, postprocess = mdpost)
        Literate.notebook(input, GENERATEDDIR, execute = true)
    elseif any(endswith.(example, [".png", ".jpg", ".gif"]))
        cp(joinpath(EXAMPLEDIR, example), joinpath(GENERATEDDIR, example); force=true)
    else
        @warn "ignoring $example"
    end
end

# remove any .vtu files in the generated dir (should not be deployed)
cd(GENERATEDDIR) do
    foreach(file -> endswith(file, ".vtu") && rm(file), readdir())
    foreach(file -> endswith(file, ".pvd") && rm(file), readdir())
end
