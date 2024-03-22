using Ferrite
import FerriteViz as Viz
import GLMakie as Plt


function refshape_shapevalues(ip::VectorInterpolation{3, RefHexahedron}, shape_nr; npoints = 5)
    x, y, z, u, v, w = (zeros(npoints^3) for _ in 1:6)
    idx = 1
    for i in 1:npoints
        for j in 1:npoints
            for k in 1:npoints
                x[idx], y[idx], z[idx] = (i-1, j-1, k-1) .* (2/(npoints-1)) .- 1
                u[idx], v[idx], w[idx] = shape_value(ip, Vec((x[idx], y[idx], z[idx])), shape_nr)
                idx += 1
            end
        end
    end
    return x, y, z, u, v, w
end

refcell(::Type{RefHexahedron})  = Hexahedron(ntuple(i->i, 8))
refcell(::Type{RefTetrahedron}) = Tetrahedron(ntuple(i->i, 4))
refip(::Type{RefShape}) where {RefShape <: Ferrite.AbstractRefShape} = Lagrange{RefShape,1}()


function plot_refcell(::Type{RefShape}; kwargs...) where {RefShape <: Ferrite.AbstractRefShape{3}}
    fig = Plt.Figure()
    ax = Plt.Axis3(fig[1,1]; xlabel="ξ₁", ylabel="ξ₂", zlabel="ξ₃")
    plot_refcell!(ax, RefShape; kwargs...)
    return fig, ax
end

function plot_refcell!(ax, ::Type{RefShape}; vertices=true, edges=true, faces=true) where {RefShape <: Ferrite.AbstractRefShape{3}}
    plot_vertices!(ax, RefShape; label=vertices)
    plot_edges!(ax, RefShape; label=edges)
    plot_faces!(ax, RefShape; label=faces)
    return ax
end

function plot_vertices!(ax, ::Type{RefShape}; label) where {RefShape <: Ferrite.AbstractRefShape{3}}
    ξ = Ferrite.reference_coordinates(refip(RefShape))
    ξ1, ξ2, ξ3 = (getindex.(ξ, i) for i in 1:3)
    Plt.scatter!(ax, ξ1, ξ2, ξ3)
    if label
        Plt.text!(ax, ξ1, ξ2, ξ3; text=[string(i) for i in 1:length(ξ)])
    end
    return ax
end

function plot_edges!(ax, ::Type{RefShape}; label) where {RefShape <: Ferrite.AbstractRefShape{3}}
    arrowheadpos = 2/3
    cell = refcell(RefShape)
    ξ = Ferrite.reference_coordinates(refip(RefShape))
    x, y, z, u, v, w = (zeros(length(Ferrite.edges(cell))) for _ in 1:6)
    for (k, e) in enumerate(Ferrite.edges(cell))
        ξa, ξb = getindex.((ξ,), e)
        Plt.lines!(ax, ([ξa[i], ξb[i]] for i in 1:3)...)
        x[k], y[k], z[k] = ξa
        u[k], v[k], w[k] = ξb - ξa
    end
    Plt.arrows!(ax, x, y, z, u * arrowheadpos, v * arrowheadpos, w * arrowheadpos; linewidth=0.05, arrowsize=0.1)
    if label
        s = arrowheadpos + (1-arrowheadpos)/6
        Plt.text!(ax, x + u*s, y + v*s, z + w*s; text=[string(i) for i in 1:length(x)])
    end
end

plot_faces!(ax, RefShape; label) = nothing

function testit(nshape=1)
    ip = Nedelec{3,RefHexahedron,1}();

    fig, ax = plot_refcell(RefHexahedron; vertices=true, edges=true);
    vals = refshape_shapevalues(ip, nshape)
    lengths = sqrt.(vals[4].^2 + vals[5].^2 + vals[6].^2)
    Plt.arrows!(ax, vals...; lengthscale=0.1, arrowsize=0.1, color=lengths)
    return fig
end

# Possible tests 
#= 
1) Check shape_value(ip, ξ, i) ⋅ v_edge[i] = |shape_value(ip, ξ, i)| (checks alignment)
2) Check ∫ Ni ⋅ v dL = 1 on each edge 
3) Check shape_value(ip, ξ, i) ⋅ v_edge[j] = 0 for i≠j 
=#