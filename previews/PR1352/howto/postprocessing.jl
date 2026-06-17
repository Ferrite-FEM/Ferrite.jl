include("../tutorials/heat_equation.jl");

function compute_heat_fluxes(cellvalues::CellValues, dh::DofHandler, a::AbstractVector{T}) where {T}

    nqp = getnquadpoints(cellvalues)

    # Allocate storage for the fluxes to store
    q = [Vec{2, T}[] for _ in 1:getncells(dh.grid)]

    for cell in CellIterator(dh)
        q_cell = q[cellid(cell)]
        aᵉ = a[celldofs(cell)]
        reinit!(cellvalues, cell)

        for q_point in 1:nqp
            q_qp = - function_gradient(cellvalues, q_point, aᵉ)
            push!(q_cell, q_qp)
        end
    end
    return q
end

q_gp = compute_heat_fluxes(cellvalues, dh, u);

projector = L2Projector(ip, grid);

q_projected = project(projector, q_gp, qr);

VTKGridFile("heat_equation_flux", grid) do vtk
    write_projection(vtk, projector, q_projected, "q")
end;

points = [Vec((x, 0.75)) for x in range(-1.0, 1.0, length = 101)];

ph = PointEvalHandler(grid, points);

q_points = evaluate_at_points(ph, projector, q_projected);

u_points = evaluate_at_points(ph, dh, u, :u);

import Plots

Plots.plot(getindex.(points, 1), u_points, xlabel = "x (coordinate)", ylabel = "u (temperature)", label = nothing)

Plots.plot(getindex.(points, 1), getindex.(q_points, 1), xlabel = "x (coordinate)", ylabel = "q_x (flux in x-direction)", label = nothing)

# This file was generated using Literate.jl, https://github.com/fredrikekre/Literate.jl
