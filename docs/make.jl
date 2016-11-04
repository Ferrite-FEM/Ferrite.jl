using Documenter, JuAFEM

# Build documentation.
makedocs(
    format = :html,
    sitename = "JuAFEM.jl",
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
    deps = nothing,
    make = nothing,
)
