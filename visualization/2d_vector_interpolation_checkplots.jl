using Ferrite
import CairoMakie as Plt

function facet_quadrature(::Type{RefShape}, nqp::Int) where RefShape
    fqr = FacetQuadratureRule{RefShape}(nqp)
    points = [p for rule in fqr.face_rules for p in rule.points]
    weights = [w for rule in fqr.face_rules for w in rule.weights]
    return QuadratureRule{RefShape}(weights, points)
end

function plot_shape_function(ip::VectorInterpolation{2, RefShape}, qr::Int, i::Int) where {RefShape<:Ferrite.AbstractRefShape{2}}
    return plot_shape_function(ip, QuadratureRule{RefShape}(qr), i)
end

function plot_shape_function(ip::VectorInterpolation{2, RefShape}, qr::QuadratureRule{RefShape}, i::Int) where {RefShape<:Ferrite.AbstractRefShape{2}}
    points = Ferrite.getpoints(qr)
    N = map(ξ -> Ferrite.reference_shape_value(ip, ξ, i), points)
    vertices = Ferrite.reference_coordinates(Lagrange{RefShape, 1}())
    push!(vertices, first(vertices))

    fig = Plt.Figure()
    ax = Plt.Axis(fig[1,1]; aspect=Plt.DataAspect())
    Plt.xlims!(ax, (-1.0, 1.5))
    Plt.lines!(ax, first.(vertices), last.(vertices))
    Plt.arrows!(ax, first.(points), last.(points), first.(N), last.(N); lengthscale = 0.1)
    Plt.scatter!(ax, first.(points), last.(points))
    return fig
end

function plot_global_shape_function(ip::VectorInterpolation{2, RefShape}; qr_order::Int=0, nel, i, qr=nothing) where RefShape
    fig = Plt.Figure()
    ax = Plt.Axis(fig[1,1])
    _qr = qr === nothing ? QuadratureRule{RefShape}(qr_order) : qr
    CT = RefShape === RefTriangle ? Triangle : Quadrilateral
    grid = generate_grid(CT, (nel, nel))
    dh = close!(add!(DofHandler(grid), :u, ip))
    points = Vec{2, Float64}[]
    directions = Vec{2, Float64}[]
    cv = CellValues(_qr, ip, Lagrange{RefShape, 1}())
    cell_contour = getcoordinates(grid, 1)
    resize!(cell_contour, length(cell_contour) + 1)
    for cell in CellIterator(dh)
        copyto!(cell_contour, getcoordinates(cell))
        cell_contour[end] = cell_contour[1] # Close contour
        Plt.lines!(ax, first.(cell_contour), last.(cell_contour))
        if i ∈ celldofs(cell)
            reinit!(cv, cell)
            for q_point in 1:getnquadpoints(cv)
                x = spatial_coordinate(cv, q_point, getcoordinates(cell))
                shape_index = findfirst(x -> x == i, celldofs(cell))
                push!(points, x)
                push!(directions, shape_value(cv, q_point, shape_index))
            end
        end
    end
    Plt.arrows!(ax, first.(points), last.(points), first.(directions), last.(directions); lengthscale = 0.1)
    Plt.scatter!(ax, first.(points), last.(points))
    return fig
end
