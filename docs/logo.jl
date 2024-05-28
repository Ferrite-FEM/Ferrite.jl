#!/usr/bin/env julia

using PGFPlotsX, Ferrite, FerriteGmsh

# Add Julia colors
push!(empty!(PGFPlotsX.CUSTOM_PREAMBLE),
    raw"\definecolor{julia-blue}{rgb}{0.251, 0.388, 0.847}",
    raw"\definecolor{julia-green}{rgb}{0.22, 0.596, 0.149}",
    raw"\definecolor{julia-purple}{rgb}{0.584, 0.345, 0.698}",
    raw"\definecolor{julia-red}{rgb}{0.796, 0.235, 0.2}",
    raw"\usepackage{anyfontsize}",
    raw"\usepackage{fontspec}",
    raw"\setmainfont{JuliaMono}",
)

function ferrite_logo(; bounding_box=true, mesh=true)
    # Run neper
    ## Tessalation
    success(`neper -T -dim 2 -n 6 -id 4 -reg 1`)
    ## Meshing
    success(`neper -M n6-id4.tess -dim 2 -order 1 -rcl 2`)
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
        "face1" => "julia-purple",
        "face2" => "julia-red",
        "face3" => "julia-red",
        "face4" => "julia-blue",
        "face5" => "julia-purple",
        "face6" => "julia-green"
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

# Create versions with the name

## Horizontal
function ferrite_logo_horizontal(p)
    horizontal = @pgf TikzPicture(
    raw"\node at (2.5, 0) {\pgftext{\includegraphics{" * abspath(p) * "}}};",
    raw"\node[anchor=west] at (5.5, -0.2) {{\fontsize{80}{105}\selectfont Ferrite.jl}};",
    raw"\node at (0,  2.5) {};",
    raw"\node at (0, -2.5) {};",
    )
    return horizontal
end
p = ferrite_logo_horizontal("logo.pdf")
PGFPlotsX.save("logo-horizontal.svg", p)
PGFPlotsX.save("logo-horizontal.pdf", p)
PGFPlotsX.save("logo-horizontal.png", p)

## Square with the name below
function ferrite_logo_named(p)
    logo = @pgf TikzPicture(
    raw"\node at (3, 0) {\pgftext{\includegraphics{" * abspath(p) * "}}};",
    # x = 3.0 would center the text, but that looks a bit off. x = 3.11 puts the middle
    # point between the vertical bar of the "F" and the vertical bar of the "l" in the middle.
    raw"\node[] at (3.11, -3.7) {{\fontsize{35}{35}\selectfont {\color{black}Ferrite.jl}}};",
    # raw"\node[] at (3.0, -3.4) {{\fontsize{25}{25}\selectfont {\color{black}Ferrite.jl}}};",
    raw"\node at (0,  2.7) {};",
    # raw"\draw (3, -2) -- (3, -5);",
    # raw"\draw (6.5, -2) -- (6.5, -5);",
    # raw"\draw (-0.5, -2) -- (-0.5, -5);",
    )
    return logo
end
p = ferrite_logo_named("logo.pdf")
PGFPlotsX.save("logo-name.svg", p)
PGFPlotsX.save("logo-name.pdf", p)
PGFPlotsX.save("logo-name.png", p)
