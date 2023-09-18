# generate examples
import Literate

# Tutorials
TUTORIALS_IN = joinpath(@__DIR__, "src", "literate-tutorials")
TUTORIALS_OUT = joinpath(@__DIR__, "src", "tutorials")
mkpath(TUTORIALS_OUT)

# How-to guides
HOWTO_IN = joinpath(@__DIR__, "src", "literate-howto")
HOWTO_OUT = joinpath(@__DIR__, "src", "howto")
mkpath(HOWTO_OUT)

# Code gallery
GALLERY_IN = joinpath(@__DIR__, "src", "literate-gallery")
GALLERY_OUT = joinpath(@__DIR__, "src", "gallery")
mkpath(GALLERY_OUT)

# Download some assets
include("download_resources.jl")

# Run Literate on all examples
@timeit dto "Literate." for (IN, OUT) in [(TUTORIALS_IN, TUTORIALS_OUT), (HOWTO_IN, HOWTO_OUT), (GALLERY_IN, GALLERY_OUT)], program in readdir(IN; join=true)
    name = basename(program)
    if endswith(program, "maxwell.jl")
        if !liveserver
            script = @timeit dto "script()" @timeit dto name Literate.script(program, OUT)
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
            Literate.markdown(program, OUT, postprocess = mdpost)
        end
        if !liveserver
            @timeit dto "notebook()"  @timeit dto name begin
                Literate.notebook(program, OUT, preprocess = nbpre, execute = is_ci) # Don't execute locally
            end
        end
    elseif any(endswith.(program, [".png", ".jpg", ".gif"]))
        cp(program, joinpath(OUT, name); force=true)
    else
        @warn "ignoring $program"
    end
end

# remove any .vtu files in the generated dir (should not be deployed)
@timeit dto "remove vtk files" for dir in [TUTORIALS_OUT, HOWTO_OUT, GALLERY_OUT]
    cd(dir) do
        foreach(file -> endswith(file, ".vtu") && rm(file), readdir())
        foreach(file -> endswith(file, ".pvd") && rm(file), readdir())
    end
end
