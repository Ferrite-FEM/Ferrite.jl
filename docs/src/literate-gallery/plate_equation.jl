# # [Kirchhoff-Love Plate Equation](@id tutorial-plate)
# In this example, we solve the Kirchhoff-Love equation for thin plates. This is a fourth-order partial differential equation used to model the deflection $w$ of a plate subject to transverse loading.
# The governing biharmonic equation is:
# ```math
# D \Delta^2 w = q
# ```
# where $\Delta$ is the Laplacian operator, and D is the flexural rigidity defined by the Young's modulus $E$, thickness $t$, and Poisson's ratio $\nu$:
# ```math
# D = \frac{Et^3}{12(1-\nu^2)}
# ```
# In this tutorial we will model a simply supported plate, for which the boundary conditions on $\Gamma$ are:
# ```math
# w = 0
# ```
# For the weak form, we define the curvature vector $\boldsymbol{\kappa}$ and the constitutive matrix $\boldsymbol{C}$:
# ```math
# \boldsymbol{\kappa}(w) = \begin{bmatrix} \frac{\partial^2 w}{\partial x^2} \frac{\partial^2 w}{\partial x \partial y} \\ \frac{\partial^2 w}{\partial x \partial y} \ \frac{\partial^2 w}{\partial y^2} \end{bmatrix}, \quad
# \mathbf{C} = D \begin{bmatrix} 1 & \nu & 0 \\ \nu & 1 & 0 \\ 0 & 0 & \frac{1-\nu}{2} \end{bmatrix}
# ```
# We use the following weak formulation:
# ```math
# \int_\Omega \boldsymbol{\kappa}(\delta w):\mathbf{C}:\boldsymbol{\kappa}(w) \, d\Omega - \int_\Omega \delta w \cdot q \, d\Omega = 0 \quad \forall \delta w \in H^2_0(\Omega)
# ```
# To enforce the simply supported boundary conditions, we use a penalty approach to impose the constraint $w = 0$. The reason for this choice will be made clear later in the tutorial. The penalty term adds the following contribution to the weak form:
# ```math
# \rho\int_\Gamma w \delta w \, d\Gamma
# ```
# with $\rho$ being the penalty stiffness.
# 
# ## Notes on FE-approximation of the biharmonic equation
# Since this weak form involves second-order derivatives (the Hessian) of the shape functions, the standard $C^0$ Lagrange elements are not suitable for discretising the finite element space. Instead, one must either use a $C^0$ interior penalty (C0IP) approach or employ finite elements that provide $C^1$ continuity.
# Here, we demonstrate the latter approach and use the fifth-order Argyris element. This is a Hermite-type element with additional degrees of freedom at the vertices and along the edges that represent the gradient and Hessian of the deflection field \(w\). 

# ## Code
# The following code followes a standard Ferrite solution procedure.

using Ferrite
using SparseArrays

# First we define some parameters
L = 2.0         # Side length 
q0 = 10000.0    # Load
E = 200e9       # Stiffness
t = 0.01        # Thickness
ν = 0.3         # Poisson's radtio
penalty = 1e12  # Penalty stiffness
D = (E * t^3) / (12 * (1 - ν^2)) # Flexural stiffness
C_voigt = D * [1.0 ν 0.0; 
               ν 1.0 0.0; 
               0.0 0.0 (1-ν)/2] 
C = fromvoigt(SymmetricTensor{4,2}, C_voigt)

grid = generate_grid(Triangle, (20, 20), Vec((0.0, 0.0)), Vec((L,L)))

# We use the Argyris interpolation as and FE approximation.
ip = Argyris{RefTriangle, 5}()
dh = DofHandler(grid)
add!(dh, :w, ip)
close!(dh)

# For the CellValues and FacetValues we need to requeest to update the hessians.
qr = QuadratureRule{RefTriangle}(8)
cellvalues = CellValues(qr, ip; update_hessians=true);

fqr = FacetQuadratureRule{RefTriangle}(8)
facetvalues = FacetValues(fqr, ip; update_hessians=true);

# For the current BVP, There is a known analtyical solution (Navier's solution) that we can compare with.
# This is a good testing strategy for PDE codes and known as the method of manufactured solutions.
function w_analytical(pos::Vec{2}, L, q0, D; n_terms=50)
    x,y = pos
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
function element_routine!(ke, fe, cellvalues, C, q0)
    for iqp in 1:getnquadpoints(cellvalues)
        dV = getdetJdV(cellvalues, iqp)
        for i in 1:getnbasefunctions(cellvalues)
            v = shape_value(cellvalues, iqp, i)
            fe[i] += (q0*v) *dV
            δκ = shape_hessian(cellvalues, iqp, i)
            for j in 1:getnbasefunctions(cellvalues)
                Δκ = shape_hessian(cellvalues, iqp, j)
                ke[i,j] += (δκ ⊡ C ⊡ Δκ) * dV
            end
        end
    end
end

# To enforce the boundary condition, we use the penalty method. Currently the ConstraintHandler does not fully support Dirichlet constraints on Hermitian elements (like Argyris).
function bc_routine!(ke, facetvalues, penalty)
    for iqp in 1:getnquadpoints(facetvalues)
        dV = getdetJdV(facetvalues, iqp)
        for i in 1:getnbasefunctions(facetvalues)
            Ni = shape_value(facetvalues, iqp, i)
            for j in 1:getnbasefunctions(facetvalues)
                Nj = shape_value(facetvalues, iqp, j)
                ke[i,j] += penalty*(Ni * Nj) * dV
            end
        end
    end
end

# Here we create a standard assembly routine.
function doassemble!(
        cellvalues::CellValues, facetvalues::FacetValues, K::SparseMatrixCSC, f::Vector, dh::DofHandler, C::SymmetricTensor, q0::Float64, penalty::Float64
    )

    n = getnbasefunctions(cellvalues)
    ke = zeros(n, n)
    fe = zeros(n)

    assembler = start_assemble(K, f)
    for celldata in CellIterator(dh)
        fill!(ke, 0.0)
        fill!(fe, 0.0)
        reinit!(cellvalues, celldata)
        element_routine!(ke, fe, cellvalues, C, q0)
        assemble!(assembler, celldofs(celldata), ke, fe)
    end

    ∂Ω = union(
        getfacetset(grid, "left"),
        getfacetset(grid, "right"),
        getfacetset(grid, "top"),
        getfacetset(grid, "bottom"),
    );

    for celldata in FacetIterator(dh, ∂Ω)
        fill!(ke, 0.0)
        reinit!(facetvalues, celldata)
        bc_routine!(ke, facetvalues, penalty)
        assemble!(assembler, celldofs(celldata), ke)
    end
end

# Create stiffness matrix, assemble and solve:
K = allocate_matrix(dh);
f = zeros(ndofs(dh))
doassemble!(cellvalues, facetvalues, K, f, dh, C, q0, penalty);
u = K \ f

# Export solution to VTK/Paraview
VTKGridFile("plate_equation", dh) do vtk
    write_solution(vtk, dh, u)
end

# To test the solution, we query the deflection at the center of the plate and compare it with the analtyical solution:
mid_point = Vec((L/2,L/2))
ph = PointEvalHandler(grid, [mid_point, ])
w_fem = evaluate_at_points(ph, dh, u, :w) |> first #0.03548889438239366 
w_ana = w_analytical(mid_point, L, q0, D) #0.035488713207468166

using Test 
@test w_fem ≈ w_ana atol=1e-


