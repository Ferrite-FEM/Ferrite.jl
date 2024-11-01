using Ferrite
import CairoMakie as Plt

function plot_shape_function(ip::VectorInterpolation{2, RefShape}, qr::Int, i::Int) where {RefShape<:Ferrite.AbstractRefShape{2}}
    return plot_shape_function(ip, QuadratureRule{RefShape}(qr), i)
end

function plot_shape_function(ip::VectorInterpolation{2, RefShape}, qr::QuadratureRule{RefShape}, i::Int) where {RefShape<:Ferrite.AbstractRefShape{2}}
    points = Ferrite.getpoints(qr)
    N = map(ξ -> Ferrite.reference_shape_value(ip, ξ, i), points)
    vertices = Ferrite.reference_coordinates(Lagrange{RefShape, 1}())
    push!(vertices, first(vertices))

    fig = Plt.Figure()
    ax = Plt.Axis(fig[1,1])
    Plt.lines!(ax, first.(vertices), last.(vertices))
    Plt.arrows!(ax, first.(points), last.(points), first.(N), last.(N); lengthscale = 0.3)
    return fig
end

function plot_global_shape_function(ip, qr, nx, ny, i)
    #TODO: Plot a single global shape function to investigate continuity
end
