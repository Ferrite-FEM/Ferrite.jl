Base.HOME_PROJECT[] = abspath(Base.HOME_PROJECT[]) # JuliaLang/julia/pull/28625

using Documenter, JuAFEM

# Generate examples
include("generate.jl")

GENERATEDEXAMPLES = [joinpath("examples", "generated", f) for f in (
    "heat_equation.md",
    "incompressible_elasticity.md",
    "threaded_assembly.md",
    "plasticity.md"
    )]

# Build documentation.
makedocs(
    format = Documenter.HTML(prettyurls = haskey(ENV, "HAS_JOSH_K_SEAL_OF_APPROVAL")), # disable for local builds
    sitename = "JuAFEM.jl",
    doctest = false,
    # strict = VERSION.minor == 6 && sizeof(Int) == 8, # only strict mode on 0.6 and Int64
    strict = false,
    pages = Any[
        "Home" => "index.md",
        "manual/fe_intro.md",
        "Manual" => [
            "manual/cell_integration.md",
            "manual/degrees_of_freedom.md",
            "manual/assembly.md",
            "manual/boundary_conditions.md",
            "manual/grid.md",
            "manual/export.md"
            ],
        "Examples" => GENERATEDEXAMPLES,
        "API Reference" => [
            "reference/quadrature.md",
            "reference/interpolations.md",
            "reference/fevalues.md",
            "reference/dofhandler.md",
            "reference/assembly.md",
            "reference/boundary_conditions.md",
            "reference/grid.md",
            "reference/export.md"
            ]
        ],
)

# make sure there are no *.vtu files left around from the build
cd(joinpath(@__DIR__, "build", "examples", "generated")) do
    foreach(file -> endswith(file, ".vtu") && rm(file), readdir())
end


# Deploy built documentation from Travis.
deploydocs(
    repo = "github.com/KristofferC/JuAFEM.jl.git",
)
