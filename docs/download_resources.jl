# Download some assets necessary for docs/testing not stored in the repo
import Downloads

# Tutorials
const directory = joinpath(@__DIR__, "src", "tutorials")
mkpath(directory)

for (file, url) in [
        "periodic-rve.msh" => "https://raw.githubusercontent.com/Ferrite-FEM/Ferrite.jl/gh-pages/assets/periodic-rve.msh",
        "periodic-rve-coarse.msh" => "https://raw.githubusercontent.com/Ferrite-FEM/Ferrite.jl/gh-pages/assets/periodic-rve-coarse.msh",
        "transient_heat.gif" => "https://raw.githubusercontent.com/Ferrite-FEM/Ferrite.jl/gh-pages/assets/transient_heat.gif",
        "transient_heat_colorbar.svg" => "https://raw.githubusercontent.com/Ferrite-FEM/Ferrite.jl/gh-pages/assets/transient_heat_colorbar.svg",
        "porous_media.gif" => "https://raw.githubusercontent.com/Ferrite-FEM/Ferrite.jl/gh-pages/assets/porous_media.gif",
        "porous_media_0p25.inp" => "https://raw.githubusercontent.com/Ferrite-FEM/Ferrite.jl/gh-pages/assets/porous_media_0p25.inp",
        "reactive_surface.gif" => "https://raw.githubusercontent.com/Ferrite-FEM/Ferrite.jl/gh-pages/assets/reactive_surface.gif",
        "nsdiffeq.gif" => "https://raw.githubusercontent.com/Ferrite-FEM/Ferrite.jl/gh-pages/assets/nsdiffeq.gif",
        "linear_elasticity.svg" => "https://raw.githubusercontent.com/Ferrite-FEM/Ferrite.jl/gh-pages/assets/linear_elasticity.svg",
        "linear_elasticity_stress.png" => "https://raw.githubusercontent.com/Ferrite-FEM/Ferrite.jl/gh-pages/assets/linear_elasticity_stress.png",
    ]
    afile = joinpath(directory, file)
    if !isfile(afile)
        Downloads.download(url, afile)
    end
end

# Topics
const topics_directory = joinpath(@__DIR__, "src", "topics", "downloaded_assets")
mkpath(topics_directory)

for (file, url) in [
        "ProjectedDirichlet.svg" => "https://raw.githubusercontent.com/Ferrite-FEM/Ferrite.jl/gh-pages/assets/ProjectedDirichlet.svg",
    ]
    afile = joinpath(topics_directory, file)
    isfile(afile) || Downloads.download(url, afile)
end
