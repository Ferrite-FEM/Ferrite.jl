# # [Kirchhoff-Love Plate Equation](@id tutorial-plate)
# ![](plate_equation.png)
# TODO ![](plate_equation-light.png)
# TODO ![](plate_equation-dark.png)
#
# *Figure 1*: The deflection $w$ for a simply supported plate with uniform load.
# ## Introduction
# In this example, we solve the Kirchhoff-Love equation for thin plates for linear isotropy. This is a fourth-order partial differential equation used to model the deflection $w$ of a plate subject to transverse loading.
# The governing biharmonic equation is:
# ```math
# D \Delta^2 w = q
# ```
# where $\Delta$ is the Laplacian operator defined as $\Delta = \nabla \cdot \nabla$, and D is the flexural rigidity defined by the Young's modulus $E$, thickness $t$, and Poisson's ratio $\nu$:
# ```math
# D = \frac{Et^3}{12(1-\nu^2)}
# ```
# In this tutorial we will model a simply supported plate, for which the boundary conditions on $\Gamma$ are:
# ```math
# \begin{aligned}
# w &= 0 \\
# \Delta w &= 0 \quad \text{(zero bending moment)}
# \end{aligned}
# ```
# We use the following weak formulation:
# ```math
# D\int_\Omega \Delta w \Delta v \, d\Omega - \int_\Omega v q \, d\Omega = 0
# \quad \forall v \in V
# ```
# where
# ```math
# V := \{ v \in H^2(\Omega) \;:\; v = 0 \text{ on } \partial\Omega \}.
# ```
# Here, $v$ is a test function belonging to the Sobolev space of functions with square-integrable values, gradients, and Hessians, satisfying the essential boundary condition $v=0$ on $\partial\Omega$. The vanishing bending moment condition is imposed naturally through the weak formulation.
#
# ## Notes on FE-approximation of the biharmonic equation
# Since this weak form require the shape functions of the FE approximation to be in $H^2(\Omega)$, the standard $C^0$ Lagrange elements are not suitable for discretising the finite element space. Instead, one must either use a $C^0$ interior penalty (C0IP) approach or employ finite elements that provide $C^1$ continuity.
# Here, we demonstrate the latter approach and use the fifth-order Argyris element. This is a Hermite-type element with additional degrees of freedom at the vertices and along the edges that represent the gradient and Hessian of the deflection field \(w\).

# ## Code
# The following code followes a standard Ferrite solution procedure.

using Ferrite
using SparseArrays

# First we define some parameters
L = 2.0         # Side length
q0 = 10000.0    # Load
E = 200.0e9       # Stiffness
t = 0.01        # Thickness
ν = 0.3         # Poisson's radtio
penalty = 1.0e12  # Penalty stiffness
D = (E * t^3) / (12 * (1 - ν^2)); # Flexural stiffness

grid = generate_grid(Triangle, (31, 31), Vec((0.0, 0.0)), Vec((L, L)));

# We use the Argyris interpolation as and FE approximation.
ip = Argyris{RefTriangle, 5}()
dh = DofHandler(grid)
add!(dh, :w, ip)
close!(dh);

# We define the boundary conditions for the simply supported plate by prescribing the
# PointValue() DOFs at the vertices of the triangles. Note, however, that for the
# Argyris element, the restriction of the fifth-order polynomial to an edge is not
# determined solely by the values at its two endpoints. Consequently, prescribing
# the PointValue() DOFs at the boundary vertices does not enforce zero deflection
# everywhere along the boundary, and the deflection may therefore deviate slightly
# from zero between the boundary nodes.
# The solution converges to the correct solution under mesh refinement. However,
# more accurate enforcement of the boundary condition can be obtained by imposing
# w = 0 along the entire boundary, for example using a penalty or Nitsche method.
∂Ω = union(
    getfacetset(grid, "left"),
    getfacetset(grid, "right"),
    getfacetset(grid, "top"),
    getfacetset(grid, "bottom"),
)

ch = ConstraintHandler(dh)
add!(ch, Dirichlet(:w, ∂Ω, x -> 0.0; functional = PointValue()))
close!(ch)

# For the CellValues and FacetValues we need to requeest to update the hessians.
qr = QuadratureRule{RefTriangle}(8)
cellvalues = CellValues(qr, ip; update_hessians = true);

fqr = FacetQuadratureRule{RefTriangle}(8)
facetvalues = FacetValues(fqr, ip; update_hessians = true);

# For the current BVP, there is a known analytical solution (Navier’s solution) against which we can compare our numerical results.
function w_analytical(pos::Vec{2}, L, q0, D; n_terms = 50)
    x, y = pos
    w = 0.0
    constant_factor = (16 * q0 * L^4) / (D * pi^6)

    for m in 1:2:n_terms
        for n in 1:2:n_terms
            denom = m * n * (m^2 + n^2)^2
            num = sin(m * pi * x / L) * sin(n * pi * y / L)
            w += num / denom
        end
    end

    return constant_factor * w
end;

# Now we define the element routine.
function element_routine!(ke, fe, cellvalues, D, q0)
    for iqp in 1:getnquadpoints(cellvalues)
        dV = getdetJdV(cellvalues, iqp)
        for i in 1:getnbasefunctions(cellvalues)
            v = shape_value(cellvalues, iqp, i)
            fe[i] += (q0 * v) * dV
            Δw = shape_laplacian(cellvalues, iqp, i)
            for j in 1:getnbasefunctions(cellvalues)
                Δv = shape_laplacian(cellvalues, iqp, j)
                ke[i, j] += D * (Δw * Δv) * dV
            end
        end
    end
    return
end;

# Next, we assemble the contributions from the element plate stiffnesses and the stiffness arising from the penalty-based boundary constraint.
function doassemble!(K, f, cellvalues, facetvalues, dh, D, q0, penalty)

    n = getnbasefunctions(cellvalues)
    ke = zeros(n, n)
    fe = zeros(n)

    assembler = start_assemble(K, f)

    #Assemble plate element stiffnesses
    for celldata in CellIterator(dh)
        fill!(ke, 0.0)
        fill!(fe, 0.0)
        reinit!(cellvalues, celldata)
        element_routine!(ke, fe, cellvalues, D, q0)
        assemble!(assembler, celldofs(celldata), ke, fe)
    end
    
    return
end;

# Create stiffness matrix, assemble and solve:
K = allocate_matrix(dh);
f = zeros(ndofs(dh))
doassemble!(K, f, cellvalues, facetvalues, dh, D, q0, penalty);
apply!(K, f, ch)
u = K \ f;

# Export solution to VTK/Paraview
VTKGridFile("plate_equation", dh) do vtk
    write_solution(vtk, dh, u)
end;

# To test the solution, we query the deflection at the center of the plate and compare it with the analtyical solution:
mid_point = Vec((L / 2, L / 2))
ph = PointEvalHandler(grid, [mid_point])
w_fem = evaluate_at_points(ph, dh, u, :w) |> first #0.03548889438239366
w_ana = w_analytical(mid_point, L, q0, D) #0.035488713207468166

using Test
@test w_fem ≈ w_ana atol = 1.0e-6

# We can also note that the the deflection on the boundary os not exactly equal to zero
mid_point = Vec((0.0, L/2))
ph = PointEvalHandler(grid, [mid_point])
w_edge_fem = evaluate_at_points(ph, dh, u, :w) |> first
println("Deflection at $(mid_point) on the boundary: $(w_edge_fem)")

