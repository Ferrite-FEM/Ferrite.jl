# Download some assets necessary for docs/testing not stored in the repo
import Downloads

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
