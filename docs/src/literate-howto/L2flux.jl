#=
# L2Projection on different spaces
After having solved a finite element problem, we often want to visualize secondary
quantities, such as fluxes and stresses. This how-to describes some problems that
can occur in such visualizations if such fields are not continuous, for example
when solving a problem with multiple parts with different material properties.

As a prototype problem, we consider stationary heat flow with out any heat source.
While the full code for solving the stationary heat equation is provided for
completeness, please see the [heat equation tutorial](@ref tutorial-heat-equation)
for further details. Please also see the
[post processing how-to](@ref howto-postprocessing) for an introduction to the
`L2Projector`.

The Partial Differential Equation (PDE) for the stationary heat flow without a heat source is,
```math
\begin{align*}
\mathrm{div}(\boldsymbol{q}) &= 0\quad \text{in } \Omega \\
\end{align*}
```
where $\boldsymbol{q} = -k(\boldsymbol{x}) \nabla T$ is the heat flux that
we will visualize. The heat conductivity, $k$, has a different value in the
$\Omega_h$ and $\Omega_l$ domains:
```math
k(\boldsymbol{x}) = \left\lbrace
\begin{matrix}
1.0, \boldsymbol{x}\in\Omega_h \\
0.1, \boldsymbol{x}\in\Omega_l
\end{matrix}\right.
```

On the left and right boundaries, the temperature, $T$, is prescribed to -1 and +1, respectively.
The top and bottom boundaries are insulated, and zero-valued Neumann boundary conditions,
$q_n = 0$, are applied. The normal flux is given as $q_n = \boldsymbol{q}\cdot\boldsymbol{n}$, where
$\boldsymbol{n}$ is the outwards pointing normal vector.

![setup](proj_tutorial_setup.png)

**Figure 1:** Mesh and domains (left) and temperature solution field (right).

We create the described geometry and grid using the following code,
=#
using Ferrite
grid = generate_grid(QuadraticQuadrilateral, (20, 20))
addcellset!(grid, "low_k", x -> x[2] < 1.0e-10 || x[1] > -1.0e-10)
nothing #hide

#=
Although the geometry only have straight edges, `QuadraticQuadrilateral`s
are used to enhance the quality of the Paraview visualization.


Next, we solve the described FE problem, following the [heat equation tutorial](@ref tutorial-heat-equation),
with the extra complication of a varying heat conductivity and non-homogeneous Dirichlet boundary
conditions,
=#

qr = QuadratureRule{RefQuadrilateral}(2)
ip = Lagrange{RefQuadrilateral, 2}()
ipg = geometric_interpolation(getcelltype(grid))
cv = CellValues(qr, ip, ipg)

dh = close!(add!(DofHandler(grid), :T, ip))

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

a = solve_fe(dh, cv, getcellset(grid, "low_k"))
nothing #hide

#=
Even though the temperature is continuous, the flux, $\boldsymbol{q}$, becomes discontinuous
due to the jump of heat conductivity between $\Omega_h$ and $\Omega_l$. It will therefore not
be correct to project he heat flux onto a continuous field. On the other hand, the conservation
law requires that the divergence of the heat flux doesn't go to infinity. This translates to a
heat flux that is continuous in the normal direction. Ferrite can describe so-called $H(\mathrm{div})$
function spaces, and here we will use `RaviartThomas` interpolations to do so.

Specifically, the difference between projecting onto `Lagrange` ($H^1$), `DiscontinuousLagrange`
($L_2$), and `RaviartThomas` $H(\mathrm{div})$ interpolations will be demonstrated. To project the
fluxes, we first calculate the fluxes in the quadrature points,
=#

function calculate_qp_flux(cv, a, cell, low_k_set)
    k = cellid(cell) in low_k_set ? 0.1 : 1.0
    reinit!(cv, cell)
    ae = a[celldofs(cell)]
    return [-k * function_gradient(cv, q_point, ae) for q_point in 1:getnquadpoints(cv)]
end

qp_data = [calculate_qp_flux(cv, a, cell, getcellset(grid, "low_k")) for cell in CellIterator(dh)];

#=
Next, we project the solution and export it for each interpolation.
=#
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

project_and_export("proj_H1", dh, a, grid, qr, Lagrange{RefQuadrilateral, 1}(), :scalar, qp_data)
project_and_export("proj_L2", dh, a, grid, qr, DiscontinuousLagrange{RefQuadrilateral, 1}(), :scalar, qp_data)
project_and_export("proj_Hdiv", dh, a, grid, qr, RaviartThomas{RefQuadrilateral, 1}(), :tensor, qp_data)
nothing #hide

#=
![Different projections for H1 and H(div)](proj_tutorial_results.png)

**Figure 2:** $q_1$ flux projected onto different function spaces:
a) `Lagrange{RefQuadrilateral, 1}`, b) `DiscontinuousLagrange{RefQuadrilateral, 1}`,
and c) `RaviartThomas{RefQuadrilateral, 1}`.

We first observe that when projecting the discontinuous flux field onto a continuous field defined
by the `Lagrange` interpolation (corresponding to an $H^1$-conforming space), non-physical artifacts
appear at the material interface. These artifacts are due to the continuity constraint of the $H^1$ space,
which forces the flux to be continuous across the interface, even where a physical jump is expected.

To account for the physical discontinuity in the flux, we instead project onto a discontinuous space
using `DiscontinuousLagrange` (an $L^2$ space). This allows the horizontal flux to exhibit a jump
across the interface, in accordance with physical expectations. However, near the internal corner of
$\Omega_\mathrm{l}$, the normal flux component becomes discontinuous across cell edges,
which leads to a local imbalance in the energy conservation across cell boundaries.

To address this, we project the flux onto an $H(\mathrm{div})$-conforming space using the `RaviartThomas`
interpolation. This space enforces continuity of the normal component of the flux across cell boundaries,
while allowing tangential discontinuities. The resulting projected flux field is thus continuous in the
direction of the flow, satisfying the local conservation of energy, while still accommodating tangential
jumps at the material interface.
=#
