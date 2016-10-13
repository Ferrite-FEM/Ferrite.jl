using Documenter, JuAFEM

# Build documentation.
makedocs(
    format = :html,
    sitename = "JuAFEM.jl",
    pages = Any[
        "Home" => "index.md",
        "Installation" => "man/installation.md",
        "man/basic_usage.md","lib/maintypes.md", "lib/utility_functions.md"]
        )

# Deploy built documentation from Travis.
deploydocs(
    # options
    repo = "github.com/KristofferC/JuAFEM.jl.git"
)
