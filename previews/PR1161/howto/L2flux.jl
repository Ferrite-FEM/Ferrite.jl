using Ferrite
RefShape = RefQuadrilateral
grid = generate_grid(Quadrilateral, 1 .* (10, 10))
addcellset!(grid, "low_k", x -> x[2] < -1.0e-3 || x[1] > 1.0e-3)

ipu = Lagrange{RefShape, 2}()
dh = close!(add!(DofHandler(grid), :T, ipu))
qr = QuadratureRule{RefShape}(2)
cv = CellValues(qr, ipu, Lagrange{RefShape, 1}())

function solve_fe(dh, cv, low_k_set)
    K = allocate_matrix(dh)
    f = zeros(ndofs(dh))

    Ke = zeros(getnbasefunctions(cv), getnbasefunctions(cv))
    assembler = start_assemble(K)
    for cell in CellIterator(dh)
        reinit!(cv, cell)
        fill!(Ke, 0)
        k = cellid(cell) in low_k_set ? 0.1 : 1.0
        for q_point in 1:getnquadpoints(cv)
            dΩ = getdetJdV(cv, q_point)
            for i in 1:getnbasefunctions(cv)
                ∇Ni = shape_gradient(cv, q_point, i)
                for j in 1:getnbasefunctions(cv)
                    ∇Nj = shape_gradient(cv, q_point, j)
                    Ke[i, j] += k * (∇Ni ⋅ ∇Nj) * dΩ
                end
            end
        end
        assemble!(assembler, celldofs(cell), Ke)
    end

    ch = ConstraintHandler(dh)
    add!(ch, Dirichlet(:T, getfacetset(dh.grid, "left"), Returns(-1.0)))
    add!(ch, Dirichlet(:T, getfacetset(dh.grid, "right"), Returns(+1.0)))
    close!(ch)
    apply!(K, f, ch)
    return K \ f
end

a = solve_fe(dh, cv, getcellset(grid, "low_k"));

function calculate_qp_flux(cv, a, cell, low_k_set)
    k = cellid(cell) in low_k_set ? 0.1 : 1.0
    reinit!(cv, cell)
    ae = a[celldofs(cell)]
    return [-k * function_gradient(cv, q_point, ae) for q_point in 1:getnquadpoints(cv)]
end

qp_data = [calculate_qp_flux(cv, a, cell, getcellset(grid, "low_k")) for cell in CellIterator(dh)];

function project_and_export(name, dofhandler, sol, grid, qr_rhs, ip, type, data)
    proj = L2Projector{type}(grid)
    add!(proj, 1:getncells(grid), ip; qr_rhs)
    close!(proj)
    return VTKGridFile(name, dofhandler; write_discontinuous = true) do vtk
        write_solution(vtk, dofhandler, sol)
        Ferrite.write_cellset(vtk, grid, "low_k")
        write_projection(vtk, proj, project(proj, data), "q")
    end
end

project_and_export("proj_L2", dh, a, grid, qr, DiscontinuousLagrange{RefShape, 1}(), :scalar, qp_data)
project_and_export("proj_H1", dh, a, grid, qr, Lagrange{RefShape, 1}(), :scalar, qp_data)
project_and_export("proj_Hdiv", dh, a, grid, qr, RaviartThomas{RefShape, 1}(), :tensor, qp_data)

# This file was generated using Literate.jl, https://github.com/fredrikekre/Literate.jl
