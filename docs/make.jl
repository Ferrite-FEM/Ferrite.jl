Pkg.checkout("Documenter")

using Documenter, JuAFEM

# Generate examples
include("generate.jl")

GENERATEDEXAMPLES = [joinpath("examples", "generated", f) for f in readdir(GENERATEDDIR) if endswith(f, ".md")]

# Build documentation.
makedocs(
    format = :html,
    sitename = "JuAFEM.jl",
    doctest = true,
    strict = VERSION.minor == 6 && sizeof(Int) == 8, # only strict mode on 0.6 and Int64
    pages = Any[
        "Home" => "index.md",
        "man/fe_intro.md",
        "man/getting_started.md",
        "Library" => ["lib/maintypes.md",
                      "lib/utility_functions.md"],
        "Examples" => GENERATEDEXAMPLES
        ]
        )

# Deploy built documentation from Travis.
deploydocs(
    repo = "github.com/KristofferC/JuAFEM.jl.git",
    target = "build",
    julia = "0.6", # deploy from release bot
    deps = nothing,
    make = nothing,
)
