using TimerOutputs

dto = TimerOutput()
reset_timer!(dto)

const liveserver = "liveserver" in ARGS

if liveserver
    using Revise
    @timeit dto "Revise.revise()" Revise.revise()
end

using Documenter, DocumenterCitations, Ferrite, FerriteGmsh, FerriteMeshParser
using Documenter, Ferrite, FerriteGmsh, FerriteMeshParser
using SparseArrays, LinearAlgebra

using BlockArrays
const FerriteBlockArrays = Base.get_extension(Ferrite, :FerriteBlockArrays)

const is_ci = haskey(ENV, "GITHUB_ACTIONS")

# Generate tutorials and how-to guides
include("generate.jl")

# Changelog
include("changelog.jl")
create_documenter_changelog()

bibtex_plugin = CitationBibliography(
    joinpath(@__DIR__, "src", "assets", "references.bib"),
    style=:numeric
)

# Build documentation.
@timeit dto "makedocs" makedocs(
    format = Documenter.HTML(
        assets = [
            "assets/custom.css",
            "assets/citations.css",
            "assets/favicon.ico"
        ],
        canonical = "https://ferrite-fem.github.io/Ferrite.jl/stable",
        collapselevel = 1,
    ),
    sitename = "Ferrite.jl",
    doctest = false,
    warnonly = true,
    draft = liveserver,
    pages = Any[
        "Home" => "index.md",
        # hide("Changelog" => "changelog.md"),
        "Tutorials" => [
            "Tutorials overview" => "tutorials/index.md",
            "tutorials/heat_equation.md",
            "tutorials/linear_elasticity.md",
            "tutorials/incompressible_elasticity.md",
            "tutorials/hyperelasticity.md",
            "tutorials/plasticity.md",
            "tutorials/transient_heat_equation.md",
            "tutorials/computational_homogenization.md",
            "tutorials/stokes-flow.md",
            "tutorials/porous_media.md",
            "tutorials/ns_vs_diffeq.md",
            "tutorials/linear_shell.md",
            "tutorials/dg_heat_equation.md",
        ],
        "Topic guides" => [
            "Topic guide overview" => "topics/index.md",
            "topics/fe_intro.md",
            "topics/FEValues.md",
            "topics/degrees_of_freedom.md",
            "topics/sparse_matrix.md",
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
            "reference/sparsity_pattern.md",
            "reference/assembly.md",
            "reference/boundary_conditions.md",
            "reference/grid.md",
            "reference/export.md",
            "reference/utils.md",
        ],
        "How-to guides" => [
            "How-to guide overview" => "howto/index.md",
            "howto/postprocessing.md",
            "howto/threaded_assembly.md",
        ],
        "gallery/index.md",
        # "Code gallery" => [
        #     "Code gallery overview" => "gallery/index.md",
        #     "gallery/helmholtz.md",
        #     "gallery/quasi_incompressible_hyperelasticity.md",
        #     "gallery/landau.md",
        #     "gallery/topology_optimization.md",
        # ],
        "devdocs/index.md",
        "references.md",
        ],
    plugins = [
        bibtex_plugin,
    ]
)

# make sure there are no *.vtu files left around from the build
@timeit dto "remove vtk files" cd(joinpath(@__DIR__, "build", "tutorials")) do
    foreach(file -> endswith(file, ".vtu") && rm(file), readdir())
end

# Insert some <br> in the side menu
for (root, _, files) in walkdir(joinpath(@__DIR__, "build")), file in joinpath.(root, files)
    endswith(file, ".html") || continue
    str = read(file, String)
    # Insert <br> after "Reference" (before "Code gallery")
    str = replace(str, r"""(<li(?: class="is-active")?><a class="tocitem" href(?:="[\./\w]+")?>Code gallery</a></li>)""" => s"<br>\1")
    write(file, str)
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
