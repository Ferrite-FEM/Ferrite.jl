using TimerOutputs

dto = TimerOutput()
reset_timer!(dto)

const liveserver = "liveserver" in ARGS

if liveserver
    using Revise
    @timeit dto "Revise.revise()" Revise.revise()
end

using Documenter, Ferrite, FerriteGmsh, FerriteMeshParser

const is_ci = haskey(ENV, "GITHUB_ACTIONS")

# Generate tutorials and how-to guides
include("generate.jl")

# Build documentation.
@timeit dto "makedocs" makedocs(
    format = Documenter.HTML(
        assets = ["assets/custom.css", "assets/favicon.ico"],
        canonical = "https://ferrite-fem.github.io/Ferrite.jl/stable",
        collapselevel = 1,
    ),
    sitename = "Ferrite.jl",
    doctest = false,
    # strict = VERSION.minor == 6 && sizeof(Int) == 8, # only strict mode on 0.6 and Int64
    strict = false,
    draft = liveserver,
    pages = Any[
        "Home" => "index.md",
        "Tutorials" => [
            "Tutorials overview" => "tutorials/index.md",
            "tutorials/heat_equation.md",
            "tutorials/postprocessing.md",
            "tutorials/helmholtz.md",
            "tutorials/incompressible_elasticity.md",
            "tutorials/hyperelasticity.md",
            "tutorials/threaded_assembly.md",
            "tutorials/plasticity.md",
            "tutorials/transient_heat_equation.md",
            "tutorials/landau.md",
            "tutorials/linear_shell.md",
            "tutorials/quasi_incompressible_hyperelasticity.md",
            "tutorials/ns_vs_diffeq.md",
            "tutorials/computational_homogenization.md",
            # "tutorials/stokes-flow.md",
            "tutorials/topology_optimization.md",
            "tutorials/porous_media.md",
        ],
        "Topic guides" => [
            "Topic guide overview" => "topics/index.md",
            "topics/fe_intro.md",
            "topics/degrees_of_freedom.md",
            "topics/assembly.md",
            "topics/boundary_conditions.md",
            "topics/constraints.md",
            "topics/grid.md",
            "topics/export.md"
        ],
        "Reference" => [
            "Reference overview" => "reference/index.md",
            "reference/quadrature.md",
            "reference/interpolations.md",
            "reference/fevalues.md",
            "reference/dofhandler.md",
            "reference/assembly.md",
            "reference/boundary_conditions.md",
            "reference/grid.md",
            "reference/export.md",
            "reference/utils.md",
        ],
        "How-to guides" => [
            "How-to guide overview" => "howto/index.md",
        ],
        "devdocs/index.md",
        ],
)

# make sure there are no *.vtu files left around from the build
@timeit dto "remove vtk files" cd(joinpath(@__DIR__, "build", "tutorials")) do
    foreach(file -> endswith(file, ".vtu") && rm(file), readdir())
end


# Deploy built documentation
if !liveserver
    @timeit dto "deploydocs" deploydocs(
        repo = "github.com/Ferrite-FEM/Ferrite.jl.git",
        push_preview=true,
        versions = [
            "stable" => "v^",
            "v#.#",
            "v0.3.13",
            "v0.3.12",
            "v0.3.11",
            "v0.3.10",
            "v0.3.9",
            "v0.3.8",
            "v0.3.7",
            "v0.3.6",
            "v0.3.5",
            "dev" => "dev"
        ]
    )
end

print_timer(dto)
