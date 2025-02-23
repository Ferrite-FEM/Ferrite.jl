#=
# Exporting L2-projected flux
Conservation of some quantity, $u$, (e.g. mass, energy, electrons),
leads the strong forms like
```math
\dot{u} + \mathrm{div}(\boldsymbol{q}) = f
```
where $\boldsymbol{q}$ is the flux of the conserved quantity,
and $f$ is a source term representing supply of the quantity in question
per volume at the considered point (e.g. microwave heating for energy conservation).

In the case of the heat equation, $u$ represents the temperature and $\boldsymbol{q}$ the
energy flux, which is often assumed to follow Fourier's law,
```math
\boldsymbol{q} = -k \nabla u
```
Normally, $u$ is calculated in the finite element problem (see e.g. [Tutorial 1: Heat equation](heat_equation.md)).
Given a continuous $u$, $\boldsymbol{q}$ can still be discontinuous at the boundary between two materials,
if there is a jump in $k$. However, the flux normal to the boundary will still be continuous in
order to fulfill the conservation law. Specifically, the flux is in a $H(\mathrm{div})$ function
space. Since Ferrite has interpolations for such function spaces, we can actually calculate the fluxes
at the integration points, and then project the flux onto such a space for a more accurate visualization.

This how-to will demonstrate the difference in this case between `Lagrange` ($H^1$) and
and `RaviartThomas` $H(\mathrm{div})$ interpolations for this. We consider a simple geometry
with a lower part having a tenth of the heat conductivity of the upper part.
The entire part is then subjected to a horizontally linearly decreasing temperature field.

![Different projections for H1 and H(div)](L2_flux_export.png)
Figure 1: Projecting the flux onto the $H^1$ space gives artifacts at the interface between the
two materials, as the flux in the horizontal direction is discontinuous there. However, the H(div)
space allows this discontinuity.

TODO: Update explanation to FE solution to show continuity of H(div) as well

We start by setting up the problem
=#
using Ferrite

grid = generate_grid(Triangle, (10, 6), Vec((-1.0, -0.6)), Vec((1.0, 0.6)))
addcellset!(grid, "low_k", x -> x[2] < -1.0e-3 || x[1] > 0.5)

ipu = Lagrange{RefTriangle, 2}()
dh = close!(add!(DofHandler(grid), :u, ipu))
qr = QuadratureRule{RefTriangle}(2)
cv = CellValues(qr, ipu, Lagrange{RefTriangle, 1}())

# Next, we assign a given temperature solution.
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
    add!(ch, Dirichlet(:u, getfacetset(dh.grid, "left"), Returns(-1.0)))
    add!(ch, Dirichlet(:u, getfacetset(dh.grid, "right"), Returns(+1.0)))
    close!(ch)
    apply!(K, f, ch)
    return K \ f
end

a = solve_fe(dh, cv, getcellset(grid, "low_k"))

#a = zeros(ndofs(dh))
#apply_analytical!(a, dh, :u, x -> -x[1]) # Create temperature solution linear variation in x₁


# We then calculate the flux in each quadrature point
function calculate_qp_flux(cv, a, cell, low_k_set)
    k = cellid(cell) in low_k_set ? 0.1 : 1.0
    reinit!(cv, cell)
    ae = a[celldofs(cell)]
    return [-k * function_gradient(cv, q_point, ae) for q_point in 1:getnquadpoints(cv)]
end

qp_data = [calculate_qp_flux(cv, a, cell, getcellset(grid, "low_k")) for cell in CellIterator(dh)]

# Before setting up the L2Projectors
function create_l2_projector(grid, qr_rhs, ip, type)
    proj = L2Projector{type}(grid)
    add!(proj, 1:getncells(grid), ip; qr_rhs)
    close!(proj)
    return proj
end
proj_H1 = create_l2_projector(grid, qr, Lagrange{RefTriangle, 2}(), :scalar)
VTKGridFile("qproj_H1", dh) do vtk
    write_solution(vtk, dh, a)
    Ferrite.write_cellset(vtk, grid, "low_k")
    write_projection(vtk, proj_H1, project(proj_H1, qp_data), "q")
end

proj_Hdiv = create_l2_projector(grid, qr, RaviartThomas{RefTriangle, 2}(), :tensor)
VTKGridFile("qproj_Hdiv", dh; write_discontinuous = true) do vtk
    write_solution(vtk, dh, a)
    write_projection(vtk, proj_Hdiv, project(proj_Hdiv, qp_data), "q")
end
