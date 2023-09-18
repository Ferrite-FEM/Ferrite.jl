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
using Ferrite 
import Ferrite: Nedelec
import CairoMakie as M
ip = Nedelec{2,RefTriangle,1}()
grid = generate_grid(Triangle, (2,2))
dh = DofHandler(grid)
add!(dh, :B, ip)
close!(dh)

ip_geo = Ferrite.default_interpolation(Triangle)
qr = QuadratureRule{RefTriangle}(10)
cv = CellValues(qr, ip, ip_geo)

n_qp = getncells(grid)*getnquadpoints(cv)
coords = (zeros(n_qp), zeros(n_qp))
vectors = (zeros(n_qp), zeros(n_qp))

for nr in 1:(ndofs(dh))
    u = zeros(ndofs(dh))
    u[nr] = 1.0

    for cell_nr in 1:getncells(grid)
        x = getcoordinates(grid, cell_nr)
        reinit!(cv, x, getcells(grid, cell_nr))
        ue = u[celldofs(dh, cell_nr)]
        for q_point in 1:getnquadpoints(cv)
            i = getnquadpoints(cv)*(cell_nr-1) + q_point
            qp_x = spatial_coordinate(cv, q_point, x)
            v = function_value(cv, q_point, ue)
            sfac = norm(v) ≈ 0 ? NaN : 1.0 # Skip plotting zero-vector points
            coords[1][i] = sfac*qp_x[1]
            coords[2][i] = sfac*qp_x[2]
            vectors[1][i] = v[1]
            vectors[2][i] = v[2]
        end
    end

    fig = M.Figure()
    ax = M.Axis(fig[1,1]; aspect=M.DataAspect())
    for cellnr in 1:getncells(grid)
        x = getcoordinates(grid, cellnr)
        push!(x, x[1])
        M.lines!(ax, first.(x), last.(x), color=:black)
    end
    M.arrows!(ax, coords..., vectors...; lengthscale=0.1)
    display(fig)
end
#=
mutable struct NewCellCache{T,dim,CT}
    const x::Vector{Vec{dim,T}}
    const dofs::Vector{Int}
    cell::CT 
end
=#