using Documenter, JuAFEM

# Build documentation.
makedocs(
    modules = [JuAFEM],
    format = :html,
    sitename = "JuAFEM.jl",
    doctest = true,
    strict = true,
    pages = Any[
        "Home" => "index.md",
        "man/fe_intro.md",
        "man/getting_started.md",
        "Library" => ["lib/maintypes.md",
                      "lib/utility_functions.md"]]
        )

# Deploy built documentation from Travis.
deploydocs(
    repo = "github.com/KristofferC/JuAFEM.jl.git",
    target = "build",
    julia = "0.5",
    deps = nothing,
    make = nothing,
)
