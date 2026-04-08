# Download some assets necessary for docs/testing not stored in the repo
import Downloads

# Tutorials
const directory = joinpath(@__DIR__, "src", "tutorials")
mkpath(directory)

for file in [
        "transient_heat.gif", "transient_heat_colorbar.svg",
        "porous_media.gif",
        "reactive_surface.gif",
        "nsdiffeq.gif",
        "linear_elasticity.svg", "linear_elasticity_stress.png",
    ]
    afile = joinpath(directory, file)
    if !isfile(afile)
        Downloads.download(Ferrite.asset_url(file), afile)
    end
end

const howto_directory = joinpath(@__DIR__, "src", "howto")
mkpath(howto_directory)

for (file, url) in [
        "proj_tutorial_setup.png" => "https://raw.githubusercontent.com/Ferrite-FEM/Ferrite.jl/refs/heads/gh-pages/assets/proj_tutorial_setup.png",
        "proj_tutorial_results.png" => "https://raw.githubusercontent.com/Ferrite-FEM/Ferrite.jl/refs/heads/gh-pages/assets/proj_tutorial_results.png",
    ]
    afile = joinpath(howto_directory, file)
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
