# Maxwell's equations 
# For simplicity, we start with the very basic example from 
# https://www.math.colostate.edu/~bangerth/videos.676.33.html
# Specifically, 
# ```math
# \int_\Omega \left[\mathrm{curl}(\boldsymbol{\delta u}) \cdot \mathrm{curl}(\boldsymbol{u})
# + \mathrm{div}(\boldsymbol{\delta u}) \mathrm{div}(\boldsymbol{u})\right] \mathrm{d}\Omega = 0
# ```
# As noted in the lecture, standard Lagrange elements are not sufficient to solve this problem accurately,
# and we therefore use the Nedelec interpolation.  
import CairoMakie as M
using Ferrite
import Ferrite: Nedelec, RaviartThomas
import Ferrite: reference_coordinates
ip = Nedelec{2,RefTriangle,1}()
ip = RaviartThomas{2,RefTriangle,1}()
ip_geo = Lagrange{RefTriangle,1}()
ref_x = reference_coordinates(ip_geo)

x_vertices = [ref_x..., ref_x[1]]

function create_testpoints(npts)
    inside(x) = (1 - x[1]) > x[2]
    x = Vec{2,Float64}[]
    for i in 0:(npts-1)
        for j in i:(npts-1)
            push!(x, Vec((i/(npts-1)), 1 - j/(npts-1)))
        end
    end
    return x
end
x = create_testpoints(10)

w = ones(length(x))*0.5/length(x)
qr = QuadratureRule{RefTriangle}(w, x)

fig = M.Figure(resolution=(1000,1000)); 
for i in 1:3
    ax=M.Axis(fig[i,1]; aspect=M.DataAspect());
    M.lines!(ax, first.(x_vertices), last.(x_vertices))
    v = shape_value.((ip,), x, i)
    M.scatter!(ax, first.(x), last.(x))
    M.arrows!(ax, first.(x), last.(x), first.(v), last.(v); lengthscale=0.25)
end
display(fig)

fig2 = M.Figure(resolution=(1000,1000))
cv = CellValues(qr, ip, ip_geo)
ref_x[1] = Vec(4.0,0.0)
reinit!(cv, ref_x, Triangle((1,2,3)))
x_vertices = [ref_x..., ref_x[1]]
for i in 1:3
    ax=M.Axis(fig2[i,1]; aspect=M.DataAspect());
    M.lines!(ax, first.(x_vertices), last.(x_vertices))
    x_qp = spatial_coordinate.((cv,), 1:length(x), (ref_x,))
    @show x_qp ≈ x # should be false
    v = shape_value.((cv,), 1:length(x), i)
    M.scatter!(ax, first.(x_qp), last.(x_qp))
    M.arrows!(ax, first.(x_qp), last.(x_qp), first.(v), last.(v); lengthscale=0.25)
end
fig2