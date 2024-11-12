"""
    ColoringDofHandler{Ti<:Integer,VECS<:Vector{Vector{Ti}},DH<:AbstractDofHandler}

A mutable struct that encapsulates a DofHandler and different colors for the cells, in order to be used
in CPU multithreading scheme.
"""
mutable struct ColoringDofHandler{Ti <: Integer, VECS <: Vector{Vector{Ti}}, DH <: AbstractDofHandler}
    dh::DH
    colors::VECS
    current_color::Ti
end


function init_colordh(dh::AbstractDofHandler)
    grid = get_grid(dh)
    colors = create_coloring(grid)
    return ColoringDofHandler(dh, colors, 0)
end


## Accessors ##
dofhandler(cd::ColoringDofHandler) = cd.dh
colors(cd::ColoringDofHandler) = cd.colors
eles_in_color(cd::ColoringDofHandler, color::Ti) where {Ti <: Integer} = cd.colors[color]
current_color(cd::ColoringDofHandler) = cd.current_color
current_color!(cd::ColoringDofHandler, color::Ti) where {Ti <: Integer} = (cd.current_color = color)
ncolors(cd::ColoringDofHandler) = cd |> colors |> length
