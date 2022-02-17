# generate examples
import Literate

EXAMPLEDIR = joinpath(@__DIR__, "src", "literate")
GENERATEDDIR = joinpath(@__DIR__, "src", "examples")
mkpath(GENERATEDDIR)

# Download some assets
import Downloads
for (file, url) in [
        "periodic-rve.msh" => "https://raw.githubusercontent.com/Ferrite-FEM/Ferrite.jl/gh-pages/assets/periodic-rve.msh",
        "periodic-rve-coarse.msh" => "https://raw.githubusercontent.com/Ferrite-FEM/Ferrite.jl/gh-pages/assets/periodic-rve-coarse.msh",
    ]
    afile = joinpath(GENERATEDDIR, file)
    if !isfile(afile)
        Downloads.download(url, afile)
    end
end

for example in readdir(EXAMPLEDIR)
    if endswith(example, ".jl")
        input = abspath(joinpath(EXAMPLEDIR, example))
        script = Literate.script(input, GENERATEDDIR)
        code = strip(read(script, String))

        # remove "hidden" lines which are not shown in the markdown
        line_ending_symbol = occursin(code, "\r\n") ? "\r\n" : "\n"
        code_clean = join(filter(x->!endswith(x,"#hide"),split(code, r"\n|\r\n")), line_ending_symbol)
        code_clean = replace(code_clean, r"^# This file was generated .*$"m => "")
        code_clean = strip(code_clean)

        mdpost(str) = replace(str, "@__CODE__" => code_clean)
        Literate.markdown(input, GENERATEDDIR, postprocess = mdpost)
        Literate.notebook(input, GENERATEDDIR, execute = is_ci) # Don't execute locally
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
