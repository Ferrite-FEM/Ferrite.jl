#!/usr/bin/env julia

using PGFPlotsX, Ferrite, FerriteGmsh

# Add Julia colors
push!(empty!(PGFPlotsX.CUSTOM_PREAMBLE),
    "\\definecolor{julia-blue}{rgb}{0.251, 0.388, 0.847}",
    "\\definecolor{julia-green}{rgb}{0.22, 0.596, 0.149}",
    "\\definecolor{julia-purple}{rgb}{0.584, 0.345, 0.698}",
    "\\definecolor{julia-red}{rgb}{0.796, 0.235, 0.2}",
)

function ferrite_logo(; bounding_box=true, mesh=true)
    # Run neper
    ## Tessalation
    success(`neper -T -dim 2 -n 6 -id 4 -reg 1`)
    ## Meshing
    success(`neper -M n6-id4.tess -dim 2 -rcl 2`)
    ## Read the mesh
    grid = redirect_stdout(devnull) do
        saved_file_to_grid("n6-id4.msh")
    end

    # Create the tex code of the grid
    io = IOBuffer()
    for (i, n) in enumerate(grid.nodes)
        println(io, "\\node [] (N$(i)) at $(n.x.data) {};")
    end
    colormap = Dict(
        "1" => "julia-purple",
        "2" => "julia-red",
        "3" => "julia-red",
        "4" => "julia-blue",
        "5" => "julia-purple",
        "6" => "julia-green"
    )
    for (setk, setv) in grid.cellsets
        color = colormap[setk]
        for c in setv
            cell = grid.cells[c]
            print(io, "\\draw [color=$(mesh ? "black" : color), fill=$(color)] ")
            join(io, ("(N$i.center)" for i in cell.nodes), " -- ")
            println(io, " -- cycle;")
        end
    end
    tex_grid = String(take!(io))

    # Create the plot
    logo = @pgf TikzPicture(
        {
        scale = "5"
        },
        """
        % Bounding box
        $(bounding_box ? "" : "%")\\draw [line width=4pt] (0, 0) -- (1, 0) -- (1, 1) -- (0, 1) -- cycle;
        $(tex_grid)
        """
    )
    return logo
end

logo = ferrite_logo()

# Save it
PGFPlotsX.save("logo.pdf", logo)
PGFPlotsX.save("logo.svg", logo)
PGFPlotsX.save("logo.png", logo)

# favicon.ico generated based on logo.svg from https://realfavicongenerator.net
