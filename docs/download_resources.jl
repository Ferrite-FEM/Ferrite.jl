# Download some assets necessary for docs/testing not stored in the repo
import Downloads

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
    ]
    afile = joinpath(directory, file)
    if !isfile(afile)
        Downloads.download(url, afile)
    end
end
