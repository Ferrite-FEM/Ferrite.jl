# Regenerates the example screenshots/animations for the tutorials.
#
# It runs the example scripts that write the required VTK data files into a
# temporary directory and then calls ParaView (pvbatch) on docs/screenshots.py
# to render a light/dark variant for each scene (PNG, or WebP for time-stepping
# examples).
#
# The rendered files are written to docs/screenshot-assets/ (gitignored). They
# are NOT committed to this branch; instead upload the contents of that folder
# to the `assets/` directory on the `gh-pages` branch. The docs build fetches
# them from there via docs/download_resources.jl (Ferrite.asset_url).
#
# Local preview (no upload needed): after running this script the assets sit in
# docs/screenshot-assets/, and docs/download_resources.jl copies them from there
# instead of downloading. So just run this once and then build the docs:
#   julia --project=docs docs/generate_screenshots.jl
#   julia --project=docs docs/make.jl
#
# Usage (from the repo root):
#   julia --project=docs docs/generate_screenshots.jl [--render-only] [names...]
#
# --render-only   skip running the examples and reuse the VTK files already in
#                 the data directory (handy while tuning docs/screenshots.py).
# --upload        after rendering, commit the selected examples' assets to the
#                 gh-pages `assets/` folder and push (needs push rights to the
#                 origin remote).
# names...        only (re)generate data for these examples; default is all.
#
# Requires `pvbatch` (ParaView) on PATH.

# Each example maps to the literate file that, when run, writes its VTK data.
const EXAMPLES = Dict(
    "heat_equation" => "literate-tutorials/heat_equation.jl",
    "dg_heat_equation" => "literate-tutorials/dg_heat_equation.jl",
    "linear_elasticity" => "literate-tutorials/linear_elasticity.jl",
    "incompressible_elasticity" => "literate-tutorials/incompressible_elasticity.jl",
    "plasticity" => "literate-tutorials/plasticity.jl",
    "hyperelasticity" => "literate-tutorials/hyperelasticity.jl",
    "stokes-flow" => "literate-tutorials/stokes-flow.jl",
    "computational_homogenization" => "literate-tutorials/computational_homogenization.jl",
    "linear_shell" => "literate-tutorials/linear_shell.jl",
    "heat_adaptivity" => "literate-tutorials/heat_adaptivity.jl",
    "darcy_flow" => "literate-tutorials/darcy_flow.jl",
    # How-to guides.
    "postprocessing" => "literate-howto/postprocessing.jl",
    "threaded_assembly" => "literate-howto/threaded_assembly.jl",
    # Time-stepping examples rendered as animations (write .pvd or temporal .vtkhdf).
    "transient_heat" => "literate-tutorials/transient_heat_equation.jl",
    "elastodynamics" => "literate-tutorials/elastodynamics.jl",
    "porous_media" => "literate-tutorials/porous_media.jl",
    "ns_vs_diffeq" => "literate-tutorials/ns_vs_diffeq.jl",
    "reactive_surface" => "literate-tutorials/reactive_surface.jl",
    # Code gallery.
    "helmholtz" => "literate-gallery/helmholtz.jl",
    "quasi_incompressible_hyperelasticity" => "literate-gallery/quasi_incompressible_hyperelasticity.jl",
    "landau" => "literate-gallery/landau.jl",
    "topology_optimization" => "literate-gallery/topology_optimization.jl",
    "elasticity_adaptivity" => "literate-gallery/elasticity_adaptivity.jl",
    "suspension_bridge" => "literate-gallery/suspension_bridge.jl",
)

# Extra code evaluated in the example's module after running it, e.g. to
# re-run at a finer mesh resolution for a smoother figure than the (cheap)
# default the example uses.
const POSTRUN = Dict(
    "stokes-flow" => :(main(0.02)),
    # More AMR steps than the (cheap) example default, for a properly refined mesh.
    "heat_adaptivity" => :(solve_adaptive(grid; nsteps = 7)),
    "elasticity_adaptivity" => :(solve_adaptive(forest; nsteps = 12)),
    # The example only runs the larger regularization radius; the figure comparing the
    # two needs the smaller one as well.
    "topology_optimization" => :(topopt(0.02, 0.5, 60, "small_radius"; output = false)),
)

# Output file basenames (before the -light/-dark suffix) each scene renders;
# defaults to the scene name itself. Needed to know which files in OUTDIR
# belong to a scene when uploading a subset.
const OUTPUTS = Dict(
    "linear_elasticity" => ["linear_elasticity", "linear_elasticity_stress"],
    "incompressible_elasticity" => ["incompressible_elasticity", "incompressible_elasticity_pressure"],
    "elastodynamics" => ["elastodynamics", "elastodynamics_modes"],
    "landau" => ["landau_orig", "landau_opt"],
    "postprocessing" => ["postprocessing", "postprocessing_cutline"],
    "threaded_assembly" => ["coloring"],
    "topology_optimization" => ["topology_optimization", "topology_optimization_result"],
)

const DOCS = @__DIR__
const DATADIR = joinpath(DOCS, "screenshot-data")     # gitignored VTK scratch data
const OUTDIR = joinpath(DOCS, "screenshot-assets")    # gitignored; upload to gh-pages/assets

# Examples that build on another one include it by its generated path
# (docs/src/tutorials/...), which only exists once Literate has run. Map such a
# path back to the literate source, which is what this script runs from.
function resolve_example(dir, path)
    file = normpath(joinpath(dir, path))
    isfile(file) && return file
    return replace(file, r"([/\\])(tutorials|howto|gallery)([/\\])" => s"\1literate-\2\3")
end

# Copy the rendered assets for the given scenes into the gh-pages `assets/`
# folder and push them, so docs/download_resources.jl can fetch them at build
# time. Only the selected scenes' files are uploaded: OUTDIR is persistent, so
# it may also hold stale outputs from earlier runs of other scenes.
function upload_assets(dir, names)
    bases = [base for name in names for base in get(OUTPUTS, name, [name])]
    files = filter(readdir(dir)) do f
        any(base -> startswith(f, base * "-"), bases)
    end
    for base in bases
        any(startswith(base * "-"), files) || error("no rendered files for $base in $dir")
    end
    repo = normpath(joinpath(DOCS, ".."))
    run(`git -C $repo fetch origin gh-pages`)
    wt = mktempdir()
    return try
        run(`git -C $repo worktree add --detach $wt origin/gh-pages`)
        dest = joinpath(wt, "assets")
        mkpath(dest)
        for f in files
            cp(joinpath(dir, f), joinpath(dest, f); force = true)
        end
        run(`git -C $wt add assets`)
        if success(`git -C $wt diff --cached --quiet`)
            @info "gh-pages assets already up to date"
        else
            run(`git -C $wt commit -m "Update generated example assets"`)
            run(`git -C $wt push origin HEAD:gh-pages`)
            @info "pushed updated assets to gh-pages"
        end
    finally
        run(`git -C $repo worktree remove --force $wt`)
    end
end

render_only = "--render-only" in ARGS
selected = filter(a -> !startswith(a, "--"), ARGS)
isempty(selected) && (selected = collect(keys(EXAMPLES)))

mkpath(DATADIR)
mkpath(OUTDIR)

if !render_only
    for name in selected
        script = joinpath(DOCS, "src", EXAMPLES[name])
        @info "running $name"
        # Each example runs in its own module so their `using`s and definitions
        # don't clash (e.g. several packages export `update!`).
        mod = Module(Symbol("Example_", name))
        # A module created at runtime has no `include`, which examples building on
        # another one need (e.g. the postprocessing how-to). Give it one resolving
        # relative to the example file, not to the data directory we run in.
        dir = dirname(script)
        Core.eval(mod, :(include(path) = $(Base.include)($mod, $resolve_example($dir, path))))
        cd(() -> Base.include(mod, script), DATADIR)
        if haskey(POSTRUN, name)
            cd(() -> Core.eval(mod, POSTRUN[name]), DATADIR)
        end
    end
end

run(`pvbatch $(joinpath(DOCS, "screenshots.py")) $DATADIR $OUTDIR $selected`)
@info "screenshots written to $OUTDIR"

if "--upload" in ARGS
    upload_assets(OUTDIR, selected)
end
