# generate examples
import Literate

TUTORIALS_IN = joinpath(@__DIR__, "src", "literate")
TUTORIALS_OUT = joinpath(@__DIR__, "src", "tutorials")
mkpath(TUTORIALS_OUT)

HOWTO_IN = joinpath(@__DIR__, "src", "literate-howto")
HOWTO_OUT = joinpath(@__DIR__, "src", "howto")
mkpath(HOWTO_IN)
mkpath(HOWTO_OUT)

# Download some assets
include("download_resources.jl")

# Run Literate on all examples
@timeit dto "Literate." for tutorial in readdir(TUTORIALS_IN; join=true)
    name = basename(tutorial)
    if endswith(tutorial, ".jl")
        if !liveserver
            script = @timeit dto "script()" @timeit dto name Literate.script(tutorial, TUTORIALS_OUT)
            code = strip(read(script, String))
        else
            code = "<< no script output when building as draft >>"
        end

        # remove "hidden" lines which are not shown in the markdown
        line_ending_symbol = occursin(code, "\r\n") ? "\r\n" : "\n"
        code_clean = join(filter(x->!endswith(x,"#hide"),split(code, r"\n|\r\n")), line_ending_symbol)
        code_clean = replace(code_clean, r"^# This file was generated .*$"m => "")
        code_clean = strip(code_clean)

        mdpost(str) = replace(str, "@__CODE__" => code_clean)
        function nbpre(str)
            # \llbracket and \rr bracket not supported by MathJax (Jupyter/nbviewer)
            str = replace(str, "\\llbracket" => "[\\![", "\\rrbracket" => "]\\!]")
            return str
        end

        @timeit dto "markdown()" @timeit dto name begin
            Literate.markdown(tutorial, TUTORIALS_OUT, postprocess = mdpost)
        end
        if !liveserver
            @timeit dto "notebook()"  @timeit dto name begin
                Literate.notebook(tutorial, TUTORIALS_OUT, preprocess = nbpre, execute = is_ci) # Don't execute locally
            end
        end
<<<<<<< HEAD
    elseif any(endswith.(tutorial, [".png", ".jpg", ".gif"]))
        cp(tutorial, joinpath(TUTORIALS_OUT, name); force=true)
=======
    elseif any(endswith.(example, [".png", ".jpg", ".gif", ".svg"]))
        cp(joinpath(EXAMPLEDIR, example), joinpath(GENERATEDDIR, example); force=true)
>>>>>>> f8406fffda50e821e54bfb00d5effb899977dd6e
    else
        @warn "ignoring $tutorial"
    end
end

# remove any .vtu files in the generated dir (should not be deployed)
@timeit dto "remove vtk files" cd(TUTORIALS_OUT) do
    foreach(file -> endswith(file, ".vtu") && rm(file), readdir())
    foreach(file -> endswith(file, ".pvd") && rm(file), readdir())
end
