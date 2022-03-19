# # Hyperelasticity
#
# **Keywords**: *hyperelasticity*, *finite strain*, *large deformations*, *Newton's method*,
# *conjugate gradient*, *automatic differentiation*
#
# ![hyperelasticity.png](hyperelasticity.png)
#
# *Figure 1*: Cube loaded in torsion modeled with a hyperelastic material model and
# finite strain.
#
#-
#md # !!! tip
#md #     This example is also available as a Jupyter notebook:
#md #     [`hyperelasticity.ipynb`](@__NBVIEWER_ROOT_URL__/examples/hyperelasticity.ipynb).
#-
# ## Introduction
#
# In this example we will solve a problem in a finite strain setting using an
# hyperelastic material model. In order to compute the stress we will use automatic
# differentiation, to solve the non-linear system we use Newton's
# method, and for solving the Newton increment we use conjugate gradients.
#
# The weak form is expressed in terms of the first Piola-Kirchoff stress ``\mathbf{P}``
# as follows: Find ``u \in \mathbb{U}`` such that
#
# ```math
# \int_\Omega \nabla \delta \mathbf{u} : \mathbf{P}(\mathbf{u})\ \mathrm{d}\Omega =
# \int_\Omega \delta \mathbf{u} \cdot \mathbf{b}\ \mathrm{d}\Omega + \int_{\Gamma^\mathrm{N}}
# \delta \mathbf{u} \cdot \mathbf{t}\ \mathrm{d}\Gamma
# \quad \forall \delta \mathbf{u} \in \mathbb{U}^0,
# ```
#
# where ``\mathbf{u}`` is the unknown displacement field, ``\mathbf{b}`` is the body force, ``\mathbf{t}``
# is the traction on the Neumann part of the boundary, and where ``\mathbb{U}`` and ``\mathbb{U}^0`` are
# suitable trial and test sets. ``\Omega`` denotes the reference domain, which is also called reference
# or material domain. Gradients are defined with respect to the reference domain. Note that for
# large deformation problems it is also possibile that gradients and integrals are defined on the
# deformed domain, which is also called the current or spatial domain, depending on the specific
# formulation.
#

using Ferrite, Tensors, TimerOutputs, ProgressMeter
import KrylovMethods, IterativeSolvers

# ## Hyperelastic material model
#
# The stress can be derived from an energy potential, defined in
# terms of the right Cauchy-Green tensor ``\mathbf{C} = \mathbf{F}^{\mathrm{T}} \mathbf{F}``,
# where ``\mathbf{F} = \mathbf{I} + \nabla \mathbf{u}`` is the deformation gradient.
# We shall use a neo-Hookean model, where the potential can be written as
#
# ```math
# \Psi(\mathbf{C}) = \frac{\mu}{2} (I_C - 3) - \mu \ln(J) + \frac{\lambda}{2} \ln(J)^2,
# ```
#
# where ``I_C = \mathrm{tr}(\mathbf{C})``, ``J = \sqrt{\det(\mathbf{C})}`` and ``\mu`` and ``\lambda`` material parameters.
# From the potential we obtain the second Piola-Kirchoff stress ``\mathbf{S}`` as
#
# ```math
# \mathbf{S} = 2 \frac{\partial \Psi}{\partial \mathbf{C}},
# ```
#
# and the tangent of ``\mathbf{S}`` as
#
# ```math
# \frac{\partial \mathbf{S}}{\partial \mathbf{C}} = 4 \frac{\partial \Psi}{\partial \mathbf{C}}.
# ```
#
# We can implement the material model as follows, where we utilize automatic differentiation
# for the stress and the tangent, and thus only define the potential:

struct NeoHooke
    μ::Float64
    λ::Float64
end

function Ψ(C, mp::NeoHooke)
    μ = mp.μ
    λ = mp.λ
    Ic = tr(C)
    J = sqrt(det(C))
    return μ / 2 * (Ic - 3) - μ * log(J) + λ / 2 * log(J)^2
end

function constitutive_driver(C, mp::NeoHooke)
    ## Compute all derivatives in one function call
    ∂²Ψ∂C², ∂Ψ∂C = Tensors.hessian(y -> Ψ(y, mp), C, :all)
    S = 2.0 * ∂Ψ∂C
    ∂S∂C = 4.0 * ∂²Ψ∂C²
    return S, ∂S∂C
end;

# Finally, for the finite element problem we need ``\mathbf{P}`` and
# ``\frac{\partial \mathbf{P}}{\partial \mathbf{F}}``, which can be
# obtained by using the following relations:
#
# ```math
# \begin{align*}
# \mathbf{P} &= \mathbf{F} \cdot \mathbf{S},\\
# \frac{\partial \mathbf{P}}{\partial \mathbf{F}} &= [\mathbf{F} \bar{\otimes} \mathbf{I}] :
# \frac{\partial \mathbf{S}}{\partial \mathbf{C}} : [\mathbf{F}^\mathrm{T} \bar{\otimes} \mathbf{I}]
# + \mathbf{I} \bar{\otimes} \mathbf{S}.
# \end{align*}
# ```
#
# ## Newton's Method
#
# As mentioned above, to deal with the non-linear weak form we first linearize
# the problem such that we can apply Newton's method, and then apply the FEM to
# discretize the problem. Skipping a detailed derivation Newton's method can
# be expressed as:
# Given some initial guess ``\mathbf{u}^0``, find a sequence ``\mathbf{u}^{k}`` by iterating
#
# ```math
# \mathbf{u}^{k+1} = \mathbf{u}^{k} - \Delta \mathbf{u}^{k}
# ```
#
# util some termination condition has been met. Therin we determine ``\Delta \mathbf{u}^{k}``
#
# ```math
# \mathbf{K}(\mathbf{u}^{k}) \Delta \mathbf{u}^{k} = \mathbf{g}(\mathbf{u}^{k})
# ```
#
# where $K$ is the Jacobi matrix and $g$ the global residual, such that
#
# ```math
# K_{ij} = \int_{\Omega} \nabla \delta u_{i} : \frac{\partial \mathbf{P}}{\partial \mathbf{F}} : \nabla \delta u_{j} \, \mathrm{d} \Omega
# ```
# and
#
# ```math
# g_{i} = \int_{\Omega} \nabla \delta u_{i} : \mathbf{P} - \delta u_{i} \cdot \mathbf{b} \, \mathrm{d} \Omega
# ```
#
# ## Finite element assembly
#
# The element routine for assembling the residual and tangent stiffness is implemented
# as usual, with loops over quadrature points and shape functions:

function assemble_element!(ke, ge, cell, cv, fv, mp, ue)
    ## Reinitialize cell values, and reset output arrays
    reinit!(cv, cell)
    fill!(ke, 0.0)
    fill!(ge, 0.0)

    b = Vec{3}((0.0, -0.5, 0.0)) # Body force
    t = Vec{3}((0.1, 0.0, 0.0)) # Traction
    ndofs = getnbasefunctions(cv)

    for qp in 1:getnquadpoints(cv)
        dΩ = getdetJdV(cv, qp)
        ## Compute deformation gradient F and right Cauchy-Green tensor C
        ∇u = function_gradient(cv, qp, ue)
        F = one(∇u) + ∇u
        C = tdot(F)
        ## Compute stress and tangent
        S, ∂S∂C = constitutive_driver(C, mp)
        P = F ⋅ S
        I = one(S)
        ∂P∂F = otimesu(F, I) ⊡ ∂S∂C ⊡ otimesu(F', I) + otimesu(I, S)

        ## Loop over test functions
        for i in 1:ndofs
            ## Test function + gradient
            δui = shape_value(cv, qp, i)
            ∇δui = shape_gradient(cv, qp, i)
            ## Add contribution to the residual from this test function
            ge[i] += ( ∇δui ⊡ P - δui ⋅ b ) * dΩ

            ∇δui∂P∂F = ∇δui ⊡ ∂P∂F # Hoisted computation
            for j in 1:ndofs
                ∇δuj = shape_gradient(cv, qp, j)
                ## Add contribution to the tangent
                ke[i, j] += ( ∇δui∂P∂F ⊡ ∇δuj ) * dΩ
            end
        end
    end

    ## Surface integral for the traction
    for face in 1:nfaces(cell)
        if onboundary(cell, face)
            reinit!(fv, cell, face)
            for q_point in 1:getnquadpoints(fv)
                dΓ = getdetJdV(fv, q_point)
                for i in 1:ndofs
                    δui = shape_value(fv, q_point, i)
                    ge[i] -= (δui ⋅ t) * dΓ
                end
            end
        end
    end
end;

# Assembling global residual and tangent
function assemble_global!(K, f, dh, cv, fv, mp, u)
    n = ndofs_per_cell(dh)
    ke = zeros(n, n)
    ge = zeros(n)

    ## start_assemble resets K and f
    assembler = start_assemble(K, f)

    ## Loop over all cells in the grid
    @timeit "assemble" for cell in CellIterator(dh)
        global_dofs = celldofs(cell)
        ue = u[global_dofs] # element dofs
        @timeit "element assemble" assemble_element!(ke, ge, cell, cv, fv, mp, ue)
        assemble!(assembler, global_dofs, ge, ke)
    end
end;

# Define a main function, with a loop for Newton iterations

function solve()
    reset_timer!()

    ## Generate a grid
    N = 10
    L = 1.0
    left = zero(Vec{3})
    right = L * ones(Vec{3})
    grid = generate_grid(Tetrahedron, (N, N, N), left, right)

    ## Material parameters
    E = 10.0
    ν = 0.3
    μ = E / (2(1 + ν))
    λ = (E * ν) / ((1 + ν) * (1 - 2ν))
    mp = NeoHooke(μ, λ)

    ## Finite element base
    ip = Lagrange{3, RefTetrahedron, 1}()
    qr = QuadratureRule{3, RefTetrahedron}(1)
    qr_face = QuadratureRule{2, RefTetrahedron}(1)
    cv = CellVectorValues(qr, ip)
    fv = FaceVectorValues(qr_face, ip)

    ## DofHandler
    dh = DofHandler(grid)
    push!(dh, :u, 3) # Add a displacement field
    close!(dh)

    function rotation(X, t, θ = deg2rad(60.0))
        x, y, z = X
        return t * Vec{3}(
            (0.0,
            L/2 - y + (y-L/2)*cos(θ) - (z-L/2)*sin(θ),
            L/2 - z + (y-L/2)*sin(θ) + (z-L/2)*cos(θ)
            ))
    end

    dbcs = ConstraintHandler(dh)
    ## Add a homogenous boundary condition on the "clamped" edge
    dbc = Dirichlet(:u, getfaceset(grid, "right"), (x,t) -> [0.0, 0.0, 0.0], [1, 2, 3])
    add!(dbcs, dbc)
    dbc = Dirichlet(:u, getfaceset(grid, "left"), (x,t) -> rotation(x, t), [1, 2, 3])
    add!(dbcs, dbc)
    close!(dbcs)
    t = 0.5
    Ferrite.update!(dbcs, t)

    ## Pre-allocation of vectors for the solution and Newton increments
    _ndofs = ndofs(dh)
    un = zeros(_ndofs) # previous solution vector
    u  = zeros(_ndofs)
    Δu = zeros(_ndofs)
    ΔΔu = zeros(_ndofs)
    apply!(un, dbcs)

    ## Create sparse matrix and residual vector
    K = create_sparsity_pattern(dh)
    g = zeros(_ndofs)


    ## Perform Newton iterations
    newton_itr = -1
    NEWTON_TOL = 1e-8
    prog = ProgressMeter.ProgressThresh(NEWTON_TOL, "Solving:")

    while true; newton_itr += 1
        u .= un .+ Δu # Current guess
        assemble_global!(K, g, dh, cv, fv, mp, u)
        normg = norm(g[Ferrite.free_dofs(dbcs)])
        apply_zero!(K, g, dbcs)
        ProgressMeter.update!(prog, normg; showvalues = [(:iter, newton_itr)])

        if normg < NEWTON_TOL
            break
        elseif newton_itr > 30
            error("Reached maximum Newton iterations, aborting")
        end

        ## Compute increment using cg! from IterativeSolvers.jl
        @timeit "linear solve (KrylovMethods.cg)" ΔΔu′, flag, relres, iter, resvec = KrylovMethods.cg(K, g; maxIter = 1000)
        @assert flag == 0
        @timeit "linear solve (IterativeSolvers.cg!)" IterativeSolvers.cg!(ΔΔu, K, g; maxiter=1000)

        apply_zero!(ΔΔu, dbcs)
        Δu .-= ΔΔu
    end

    ## Save the solution
    @timeit "export" begin
        vtk_grid("hyperelasticity", dh) do vtkfile
            vtk_point_data(vtkfile, dh, u)
        end
    end

    print_timer(title = "Analysis with $(getncells(grid)) elements", linechars = :ascii)
    return u
end


# Run the simulation

u = solve();

## test the result                #src
using Test                        #src
@test norm(u) ≈ 4.865189736192834 #src

#md # ## Plain program
#md #
#md # Here follows a version of the program without any comments.
#md # The file is also available here: [`hyperelasticity.jl`](hyperelasticity.jl).
#md #
#md # ```julia
#md # @__CODE__
#md # ```
