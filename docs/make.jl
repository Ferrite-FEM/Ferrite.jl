using Documenter, JuAFEM

# Build documentation.
makedocs(
    format = :html,
    sitename = "JuAFEM.jl",
    doctest = true,
    strict = false, # VERSION.minor == 6, # only strict mode on release bot
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
    julia = "0.6", # deploy from release bot
    deps = nothing,
    make = nothing,
)
