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
@timeit dto "Literate." for (IN, OUT) in [(TUTORIALS_IN, TUTORIALS_OUT), (HOWTO_IN, HOWTO_OUT), (GALLERY_IN, GALLERY_OUT)], program in readdir(IN; join = true)
    name = basename(program)
    if endswith(program, ".jl")
        if !liveserver
            script = @timeit dto "script()" @timeit dto name Literate.script(program, OUT)
            code = strip(read(script, String))
        else
            code = "<< no script output when building as draft >>"
        end

        # remove "hidden" lines which are not shown in the markdown
        line_ending_symbol = occursin(code, "\r\n") ? "\r\n" : "\n"
        code_clean = join(filter(x -> !endswith(x, "#hide"), split(code, r"\n|\r\n")), line_ending_symbol)
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
    elseif any(endswith.(program, [".png", ".jpg", ".gif", ".webp"]))
        cp(program, joinpath(OUT, name); force = true)
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

# Overview pages: compose index.md from the curated body file (kept in docs/,
# outside src/, so Documenter does not render it as its own page). The body
# contains a preamble followed by one `---`-separated section per example. Each
# section is paired, in order, with its entry in `cards` (a list of (page name,
# title, [images]) with light/dark thumbnails from docs/screenshots.py) and
# rendered as a row with the figure to the left of the description (the
# `.example-row` styling in docs/src/assets/custom.css).
function write_overview(dir, body_file, cards)
    sections = split(read(body_file, String), "\n---\n")
    preamble = popfirst!(sections)
    if length(sections) != length(cards)
        error("$(basename(body_file)): $(length(sections)) description sections but $(length(cards)) cards; they must match up one to one, in order")
    end
    io = IOBuffer()
    # Point Documenter's "Edit source" button at the tracked body file; the
    # default would be this generated (gitignored) index.md, which does not
    # exist in the repository.
    println(io, "```@meta")
    println(io, "EditURL = \"", replace(relpath(body_file, dir), '\\' => '/'), "\"")
    println(io, "```\n")
    write(io, preamble)
    for ((name, title, imgs), section) in zip(cards, sections)
        # Documenter's prettyurls defaults to true (locally and in CI), so each
        # example page is rendered as <name>/index.html.
        href = "$name/"
        println(io, "\n---\n")
        println(io, "```@raw html")
        println(io, "<div class=\"example-row\">")
        println(io, "<a class=\"example-thumb\" href=\"$href\">")
        for img in imgs
            println(io, "  <img src=\"$img\" alt=\"$title\">")
        end
        println(io, "</a>")
        println(io, "<div class=\"example-description\">")
        println(io, "```")
        println(io, "\n", strip(section), "\n")
        println(io, "```@raw html")
        println(io, "</div>")
        println(io, "</div>")
        println(io, "```")
    end
    # Only write when the content changed to not retrigger LiveServer, which
    # watches the output file.
    index_md = joinpath(dir, "index.md")
    content = String(take!(io))
    return if !isfile(index_md) || read(index_md, String) != content
        write(index_md, content)
    end
end

write_overview(
    TUTORIALS_OUT,
    joinpath(@__DIR__, "tutorials_index_body.md"),
    [
        ("heat_equation", "Heat equation", ["heat_equation-light.png", "heat_equation-dark.png"]),
        ("linear_elasticity", "Linear elasticity", ["linear_elasticity-light.png", "linear_elasticity-dark.png"]),
        ("elastodynamics", "Elastodynamics and modal analysis", ["elastodynamics-light.webp", "elastodynamics-dark.webp"]),
        ("euler_bernoulli_beam", "Euler-Bernoulli beam", ["euler_bernoulli_beam-light.png", "euler_bernoulli_beam-dark.png"]),
        ("incompressible_elasticity", "Incompressible elasticity", ["incompressible_elasticity-light.png", "incompressible_elasticity-dark.png"]),
        ("hyperelasticity", "Hyperelasticity", ["hyperelasticity-light.png", "hyperelasticity-dark.png"]),
        ("plasticity", "von Mises plasticity", ["plasticity-light.png", "plasticity-dark.png"]),
        ("transient_heat_equation", "Transient heat equation", ["transient_heat-light.webp", "transient_heat-dark.webp"]),
        ("computational_homogenization", "Computational homogenization", ["computational_homogenization-light.png", "computational_homogenization-dark.png"]),
        ("stokes-flow", "Stokes flow", ["stokes-flow-light.png", "stokes-flow-dark.png"]),
        ("porous_media", "Porous media", ["porous_media-light.webp", "porous_media-dark.webp"]),
        ("ns_vs_diffeq", "Navier–Stokes equations", ["ns_vs_diffeq-light.webp", "ns_vs_diffeq-dark.webp"]),
        ("reactive_surface", "Reactive surface", ["reactive_surface-light.webp", "reactive_surface-dark.webp"]),
        ("linear_shell", "Linear shell", ["linear_shell-light.png", "linear_shell-dark.png"]),
        ("dg_heat_equation", "DG heat equation", ["dg_heat_equation-light.png", "dg_heat_equation-dark.png"]),
    ],
)

write_overview(
    GALLERY_OUT,
    joinpath(@__DIR__, "gallery_index_body.md"),
    [
        ("helmholtz", "Helmholtz equation", ["helmholtz-light.png", "helmholtz-dark.png"]),
        ("quasi_incompressible_hyperelasticity", "Nearly incompressible hyperelasticity", ["quasi_incompressible_hyperelasticity-light.webp", "quasi_incompressible_hyperelasticity-dark.webp"]),
        ("landau", "Ginzburg–Landau minimization", ["landau_opt-light.png", "landau_opt-dark.png"]),
        ("topology_optimization", "Topology optimization", ["topology_optimization-light.webp", "topology_optimization-dark.webp"]),
    ],
)
