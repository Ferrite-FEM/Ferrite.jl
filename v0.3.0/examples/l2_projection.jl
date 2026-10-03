include("heat_equation.jl");

function compute_heat_fluxes(cellvalues::CellScalarValues{dim,T}, dh::DofHandler, a) where {dim,T}

    n = getnbasefunctions(cellvalues)
    cell_dofs = zeros(Int, n)
    nqp = getnquadpoints(cellvalues)

    # Allocate storage for the fluxes to store
    q = [Vec{2,T}[] for _ in 1:getncells(dh.grid)]

    for (cell_num, cell) in enumerate(CellIterator(dh))
        q_cell = q[cell_num]
        celldofs!(cell_dofs, dh, cell_num)
        aᵉ = a[cell_dofs]
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

q_nodes = project(projector, q_gp, qr);

vtk_grid("heat_equation_flux", grid) do vtk
    vtk_point_data(vtk, q_nodes, "q")
end;

# This file was generated using Literate.jl, https://github.com/fredrikekre/Literate.jl

