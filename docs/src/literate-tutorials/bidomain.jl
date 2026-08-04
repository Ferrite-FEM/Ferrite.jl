# The full spiral wave takes a while to develop, so the CI run only integrates a       #src
# few time steps -- just enough to check that the program still runs. The figure, in    #src
# turn, is generated on a finer mesh than the one the tutorial itself runs on, see      #src
# docs/generate_screenshots.jl.                                                         #src
if isdefined(Main, :is_ci) #hide
    IS_CI = Main.is_ci     #hide
else                       #hide
    IS_CI = false          #hide
end                        #hide
IS_FIGURE = isdefined(Main, :is_figure) && Main.is_figure #hide
nothing                    #hide
# # [Bidomain and DifferentialEquations.jl](@id tutorial-bidomain)
#
# ![](bidomain-light.webp)
# ![](bidomain-dark.webp)
#
# *Figure 1*: Spiral wave of the transmembrane potential $\varphi_{\textrm{m}}$,
# obtained by breaking up a planar excitation wave front with a region of
# refractory tissue.
#
#-
#md # !!! tip
#md #     This example is also available as a Jupyter notebook:
#md #     [`bidomain.ipynb`](@__NBVIEWER_ROOT_URL__/tutorials/bidomain.ipynb)
#-
# ## Introduction
#
# In this example we will implement the [Bidomain Model](https://en.wikipedia.org/wiki/Bidomain_model) with the help of [DifferentialEquations.jl](https://github.com/SciML/DifferentialEquations.jl).
# This model is used to simulate the excitable media, most commonly cardiac tissue. For more information about the derivation,
# check out the linked wikipedia article.
#
# The Bidomain model in parabolic-elliptic form is given as the following system
#
# ```math
# \begin{aligned}
# 	\chi C_{\textrm{m}} \frac{\partial \varphi_{\textrm{m}}}{\partial t} &= \nabla \cdot (\bm{\kappa}_{\textrm{i}} \nabla \varphi_{\textrm{m}}) + \nabla \cdot (\bm{\kappa}_{\textrm{i}} \nabla \varphi_{\textrm{e}}) - \chi I_{\textrm{ion}}(\varphi_{\textrm{m}}, \mathbf{s}) & \textrm{on} \: \Omega \times (0,T] \\
# 	0 &= \nabla \cdot (\bm{\kappa}_{\textrm{i}} \nabla \varphi_{\textrm{m}}) + \nabla \cdot (\bm{\kappa}_e + \bm{\kappa}_{\textrm{i}}) \nabla \varphi_{\textrm{e}} & \textrm{on} \: \Omega \times (0,T] \\
# 	\frac{\partial \mathbf{s}}{\partial t} &= \mathbf{g}(\varphi_{\textrm{m}}, \mathbf{s}) & \textrm{on} \: \Omega \times (0,T]
# \end{aligned}
# ```
#
# For the scope of this example we utilize the FitzHugh-Nagumo neuronal cell cell model, given by
#
# ```math
# \begin{aligned}
# 	I_{\textrm{ion}}(\varphi_{\textrm{m}}, \mathbf{s}) &= - \varphi_{\textrm{m}}(1 - \varphi_{\textrm{m}})(\varphi_{\textrm{m}} - a) + s \\
# 	g(\varphi_{\textrm{m}}, \mathbf{s}) &= e(b\varphi_{\textrm{m}} - c s - d)
# \end{aligned}
# ```
#
# with parameters and initial conditions as stated in: Alfonso Bueno-Orovio, David Kay, and Kevin Burrage. "Fourier spectral methods for fractional-in-space reaction-diffusion equations." BIT Numerical mathematics 54.4 (2014): 937-954.
#
# To utilize [DifferentialEquations.jl](https://github.com/SciML/DifferentialEquations.jl) we first have to discretize the system with Ferrite into a system of ordinary differential equations (ODEs) in mass matrix form. Therefore we have first to transform it into a weak form
#
# ```math
# \begin{aligned}
# 	\int_\Omega \chi  C_{\textrm{m}} \frac{\partial \varphi_{\textrm{m}}}{\partial t} v_1 \textrm{d}\Omega &= -\int_\Omega (\bm{\kappa}_{\textrm{i}} \nabla \varphi_{\textrm{m}} + \bm{\kappa}_{\textrm{i}} \nabla \varphi_{\textrm{e}}) \cdot \nabla v_1 \textrm{d}\Omega + \int_\Omega \chi (\varphi_{\textrm{m}}(1 - \varphi_{\textrm{m}})(\varphi_{\textrm{m}} - a) - s) v_1 \textrm{d}\Omega \\
# 	0 &= -\int_\Omega (\bm{\kappa}_{\textrm{i}} \nabla \varphi_{\textrm{m}} + (\bm{\kappa}_e + \bm{\kappa}_{\textrm{i}}) \nabla \varphi_{\textrm{e}}) \cdot \nabla v_2 \textrm{d}\Omega \\
# 	\int_\Omega \frac{\partial s}{\partial t} v_3 \textrm{d}\Omega &= \int_\Omega e(b\varphi_{\textrm{m}} - c s - d) v_3 \textrm{d}\Omega
# \end{aligned}
# ```
#
# where we assume no flux boundary condition for $\varphi_{\textrm{m}}, \varphi_{\textrm{e}}$, except in one point. This models a grounding through a Dirichlet condition of zero in this point.
#
# Please note that technically speaking we obtain a [differential-algebraic system of equations](https://en.wikipedia.org/wiki/Differential-algebraic_system_of_equations) (DAE), so note that we cannot apply all ODE solvers to the resulting system. However, [DifferentialEquations.jl](https://github.com/SciML/DifferentialEquations.jl) expects for some solvers to state the DAE as an ODE in mass matrix form and because this form arises naturally in finite element methods for many common problems, let us stick with it. In this example the required Jacobians for the ODE solver are computed via automatic differentiation, but in optimized implementations they can also be manually provided.
#
# Discretizing the provided weak form yields a semi-linear system of ODEs in mass matrix form:
#
# ```math
# \mathcal{M}
# \begin{bmatrix}
#   \frac{\partial\tilde{\varphi}_\textrm{m}}{\partial t} \\
#   \frac{\partial\tilde{\varphi}_\textrm{e}}{\partial t} \\
#   \frac{\partial \tilde{s}}{\partial t}
# \end{bmatrix}
# =
# \mathcal{L}
# \begin{bmatrix}
#   \tilde{\varphi}_\textrm{m} \\
#   \tilde{\varphi}_\textrm{e} \\
#   \tilde{s}
# \end{bmatrix}
# +
# \mathcal{N}(
#   \tilde{\varphi}_\textrm{m},
#   \tilde{\varphi}_\textrm{e},
#   \tilde{s})
# ```
#
# ## Commented Program
#
using Ferrite, SparseArrays, LinearAlgebra, BlockArrays, VTKHDF
# Instead of using a self written time integrator,
# we will use in this example a time integrator from the
# [DifferentialEquations.jl](https://github.com/SciML/DifferentialEquations.jl) ecosystem,
# from which we will use adaptive time stepping. Besides this, almost any ODE solver you can
# imagine is available. We do not need the complete suite: `DiffEqBase` provides the problem
# and solution interface (which can also handle DAEs in mass matrix form), the stiff
# Rosenbrock solver `Rodas5P` lives in `OrdinaryDiffEqRosenbrock`, `SciMLIterators` lets us
# step the solution and write it out as we go, and `ADTypes` is how we tell the solver to
# differentiate the right hand side with forward-mode AD.
import DiffEqBase: ODEFunction, ODEProblem, init, NoInit
import OrdinaryDiffEqRosenbrock: Rodas5P
import SciMLIterators: intervals
import ADTypes: AutoForwardDiff
#
# Now, we define the computational domain and cellvalues. We exploit the fact that all fields of
# the Bidomain model are approximated with the same Ansatz. Hence, we use one CellValues struct for all three fields.
nel = 60
Δt = 0.1
T = 1000.0
if IS_FIGURE  #hide
    nel = 100 #hide
end           #hide
if IS_CI      #hide
    T = 10.0  #hide
end           #hide
grid = generate_grid(Quadrilateral, (nel, nel), Vec{2}((0.0, 0.0)), Vec{2}((2.5, 2.5)))
addnodeset!(grid, "ground", x -> x[1] ≈ 0.0 && x[2] ≈ 0.0)
ip = Lagrange{RefQuadrilateral, 1}()
qr = QuadratureRule{RefQuadrilateral}(2)
cellvalues = CellValues(qr, ip);
#
# We need to intialize a DofHandler. The DofHandler needs to be aware of three different fields
# which are all first order approximations. After adding all fields to the DofHandler, we `close`
# it and thereby distribute the dofs of the problem.
dh = DofHandler(grid)
add!(dh, :ϕₘ, ip)
add!(dh, :ϕₑ, ip)
add!(dh, :s, ip)
close!(dh);
#
# The linear parts of the Bidomain equations contribute to the stiffness and mass matrix, respectively.
# So, we allocate the matrices for those terms.
K = allocate_matrix(dh)
M = allocate_matrix(dh);
#
# Material related parameters are stored in the struct `FHNParameters`
Base.@kwdef struct FHNParameters
    a::Float64 = 0.1
    b::Float64 = 0.5
    c::Float64 = 1.0
    d::Float64 = 0.0
    e::Float64 = 0.01
end;
#
# Within the equations of the model, spatial dependent parameters occur such as κₑ, κᵢ, Cₘ and χ.
# For the sake of simplicity we kept them constant.
# Nonetheless, we show how one can model spatial dependent coefficients. Hence, the unused function argument `x`
function κₑ(x)
    return SymmetricTensor{2, 2, Float64}((3.5e-5, 0, 2.5e-5))
end;

function κᵢ(x)
    return SymmetricTensor{2, 2, Float64}((4.5e-5, 0, 2.0e-6))
end;

function Cₘ(x)
    return 1.0
end;

function χ(x)
    return 1.0
end;
#
# Boundary conditions are added to the problem in the usual way.
# Please check out the other examples for an in depth explanation.
# Here we force the extracellular porential to be zero at the boundary.
ch = ConstraintHandler(dh)
∂Ω = getnodeset(grid, "ground")
dbc = Dirichlet(:ϕₑ, ∂Ω, (x, t) -> 0)
add!(ch, dbc)
close!(ch)
update!(ch, 0.0);
#
# We first write a helper to assemble the linear parts. Note that we can precompute and cache linear parts. In the used notation subscripts indicate dependent coefficients.
#
# ```math
# \mathcal{M}
# =
# \begin{bmatrix}
#   M_{\chi C_\textrm{m}} & 0 & 0 \\
#   0 & 0 & 0 \\
#   0 & 0 & M
# \end{bmatrix}
# \qquad
# \mathcal{L}
# =
# \begin{bmatrix}
#   -M_{a\chi}-K_{\bm{\kappa}_{\textrm{i}}} & -K_{\bm{\kappa}_{\textrm{i}}} & -M_{\chi} \\
#   -K_{\bm{\kappa}_{\textrm{i}}} & -K_{\bm{\kappa}_{\textrm{i}}+\bm{\kappa}_{\textrm{e}}} & 0 \\
#   M_{be} & 0 & -M_{bc}
# \end{bmatrix}
# ```
#
# In the following function, `doassemble_linear!`, we assemble all linear parts of the system that stay same over all time steps.
# This follows from the used Method of Lines, where we first discretize in space and afterwards in time.
function doassemble_linear!(cellvalues::CellValues, K::SparseMatrixCSC, M::SparseMatrixCSC, dh::DofHandler; params::FHNParameters = FHNParameters())
    n_ϕₘ = getnbasefunctions(cellvalues)
    n_ϕₑ = getnbasefunctions(cellvalues)
    n_s = getnbasefunctions(cellvalues)
    ntotal = n_ϕₘ + n_ϕₑ + n_s
    n_basefuncs = getnbasefunctions(cellvalues)
    #We use BlockedArrays to write into the right places of Ke
    Ke = BlockedArray(zeros(ntotal, ntotal), [n_ϕₘ, n_ϕₑ, n_s], [n_ϕₘ, n_ϕₑ, n_s])
    Me = BlockedArray(zeros(ntotal, ntotal), [n_ϕₘ, n_ϕₑ, n_s], [n_ϕₘ, n_ϕₑ, n_s])

    assembler_K = start_assemble(K)
    assembler_M = start_assemble(M)

    #Here the block indices of the variables are defined.
    ϕₘ▄, ϕₑ▄, s▄ = 1, 2, 3

    #Now we iterate over all cells of the grid
    @inbounds for cell in CellIterator(dh)
        fill!(Ke, 0)
        fill!(Me, 0)
        #get the coordinates of the current cell
        coords = getcoordinates(cell)

        Ferrite.reinit!(cellvalues, cell)
        #loop over all Gauss points
        for q_point in 1:getnquadpoints(cellvalues)
            #get the spatial coordinates of the current gauss point
            coords_qp = spatial_coordinate(cellvalues, q_point, coords)
            #based on the gauss point coordinates, we get the spatial dependent
            #material parameters
            κₑ_loc = κₑ(coords_qp)
            κᵢ_loc = κᵢ(coords_qp)
            Cₘ_loc = Cₘ(coords_qp)
            χ_loc = χ(coords_qp)
            dΩ = getdetJdV(cellvalues, q_point)
            for i in 1:n_basefuncs
                Nᵢ = shape_value(cellvalues, q_point, i)
                ∇Nᵢ = shape_gradient(cellvalues, q_point, i)
                for j in 1:n_basefuncs
                    Nⱼ = shape_value(cellvalues, q_point, j)
                    ∇Nⱼ = shape_gradient(cellvalues, q_point, j)
                    #diffusion parts
                    Ke[BlockIndex((ϕₘ▄, ϕₘ▄), (i, j))] -= ((κᵢ_loc ⋅ ∇Nᵢ) ⋅ ∇Nⱼ) * dΩ
                    Ke[BlockIndex((ϕₘ▄, ϕₑ▄), (i, j))] -= ((κᵢ_loc ⋅ ∇Nᵢ) ⋅ ∇Nⱼ) * dΩ
                    Ke[BlockIndex((ϕₑ▄, ϕₘ▄), (i, j))] -= ((κᵢ_loc ⋅ ∇Nᵢ) ⋅ ∇Nⱼ) * dΩ
                    Ke[BlockIndex((ϕₑ▄, ϕₑ▄), (i, j))] -= (((κₑ_loc + κᵢ_loc) ⋅ ∇Nᵢ) ⋅ ∇Nⱼ) * dΩ
                    #linear reaction parts
                    Ke[BlockIndex((ϕₘ▄, ϕₘ▄), (i, j))] -= params.a * Nᵢ * Nⱼ * dΩ
                    Ke[BlockIndex((ϕₘ▄, s▄), (i, j))] -= Nᵢ * Nⱼ * dΩ
                    Ke[BlockIndex((s▄, ϕₘ▄), (i, j))] += params.e * params.b * Nᵢ * Nⱼ * dΩ
                    Ke[BlockIndex((s▄, s▄), (i, j))] -= params.e * params.c * Nᵢ * Nⱼ * dΩ
                    #mass matrices
                    Me[BlockIndex((ϕₘ▄, ϕₘ▄), (i, j))] += Cₘ_loc * χ_loc * Nᵢ * Nⱼ * dΩ
                    Me[BlockIndex((s▄, s▄), (i, j))] += Nᵢ * Nⱼ * dΩ
                end
            end
        end

        assemble!(assembler_K, celldofs(cell), Ke)
        assemble!(assembler_M, celldofs(cell), Me)
    end
    return K, M
end;

# Regarding the non-linear parts, while the affine term could be cached, for the sake of simplicity we simply recompute it in each call to the right hand side of the system.
# ```math
# \mathcal{N}(
#   \tilde{\varphi}_\textrm{m},
#   \tilde{\varphi}_\textrm{e},
#   \tilde{s})
# =
# \begin{bmatrix}
#   -(\int_\Omega \chi ((\sum_i -\tilde{\varphi}_{m,i} u_{1,i})^3 + \tilde{\varphi}_{m,i} (1+a) u_{1,i})^2)v_{1,j} \textrm{d}\Omega)_j \\
#   0 \\
#   (\int_\Omega de v_{3,j} \textrm{d}\Omega)_j
# \end{bmatrix}
# ```
# It is important to note, that we have to sneak in the boundary conditions into the evaluation of the non-linear term.
#
# The function `apply_nonlinear!` describes the nonlinear change of the system.
# It takes the change vector `du`, the current available solution `u`, the parameter
# collection `p` and the current time `t`. The parameter collection is a `NamedTuple` used to
# pass the `dh::DofHandler`, the `ch::ConstraintHandler`, the stiffness matrix `K`, the
# `cellvalues` and the constant material parameters `FHNParameters()` into the right hand
# side, which is the interface prescribed by
# [DifferentialEquations.jl](https://github.com/SciML/DifferentialEquations.jl).
function apply_nonlinear!(du, u, p, t)
    (; dh, params, cellvalues) = p
    n_basefuncs = getnbasefunctions(cellvalues)

    for cell in CellIterator(dh)
        Ferrite.reinit!(cellvalues, cell)
        _celldofs = celldofs(cell)
        ϕₘ_celldofs = _celldofs[dof_range(dh, :ϕₘ)]
        s_celldofs = _celldofs[dof_range(dh, :s)]
        ϕₘe = u[ϕₘ_celldofs]
        coords = getcoordinates(cell)
        for q_point in 1:getnquadpoints(cellvalues)
            x_qp = spatial_coordinate(cellvalues, q_point, coords)
            χ_loc = χ(x_qp)
            dΩ = getdetJdV(cellvalues, q_point)
            val = function_value(cellvalues, q_point, ϕₘe)
            nl_contrib = -val^3 + (1 + params.a) * val^2
            for j in 1:n_basefuncs
                Nⱼ = shape_value(cellvalues, q_point, j)
                du[ϕₘ_celldofs[j]] += χ_loc * nl_contrib * Nⱼ * dΩ
                du[s_celldofs[j]] -= params.e * params.d * Nⱼ * dΩ
            end
        end
    end
    return
end;
#
# We assemble the linear parts into `K` and `M`, respectively.
K, M = doassemble_linear!(cellvalues, K, M, dh);
# Now we apply *once* the boundary conditions to these parts.
apply!(K, ch)
apply!(M, ch);
#
# In the function `bidomain!` we model the actual time dependent DAE problem. This function takes
# the same parameters as `apply_nonlinear!`.
function bidomain!(du, u, p, t)
    mul!(du, p.K, u)
    apply_nonlinear!(du, u, p, t)
    apply_zero!(du, p.ch)
    return
end;
# In the following code block we define the initial condition of the problem. We first
# initialize a zero vector of length `ndofs(dh)` and fill it afterwards in a for loop over all cells.
u₀ = zeros(ndofs(dh));
for cell in CellIterator(dh)
    _celldofs = celldofs(cell)
    ϕₘ_celldofs = _celldofs[dof_range(dh, :ϕₘ)]
    s_celldofs = _celldofs[dof_range(dh, :s)]
    for (i, coordinate) in enumerate(getcoordinates(cell))
        if coordinate[2] >= 1.25
            u₀[s_celldofs[i]] = 0.1
        end
        if coordinate[1] <= 1.25 && coordinate[2] <= 1.25
            u₀[ϕₘ_celldofs[i]] = 1.0
        end
    end
end

# We can now state and solve the `ODEProblem`. Since the jacobian of our problem is large and sparse it is advantageous to avoid building a dense matrix (with dense solver) where possible. In [DifferentialEquations.jl](https://github.com/SciML/DifferentialEquations.jl) we can enforce using sparse jacobian matrices by providing a prototype jacobian with proper sparsity pattern, see [here](https://docs.sciml.ai/DiffEqDocs/stable/examples/advanced_ode_example/#Declaring-a-Sparse-Jacobian-with-Automatic-Sparsity-Detection) for details. In our problem it turns out that the K captures this pattern sufficiently, so for the sake of simplicity we simply use it in this example.
jac_sparsity = sparse(K)

f = ODEFunction(bidomain!, mass_matrix = M; jac_prototype = jac_sparsity)
p = (; K, dh, ch, params = FHNParameters(), cellvalues)
prob_mm = ODEProblem(f, u₀, (0.0, T), p);
#
# We integrate with the adaptive Rosenbrock method `Rodas5P`, which handles the singular mass
# matrix of our DAE. Its Jacobian is computed by forward-mode automatic differentiation of
# `bidomain!`. Thanks to the prototype above this is done sparsely: the columns of the
# pattern are colored such that structurally independent ones can be differentiated
# simultaneously, so that one Jacobian costs 27 directional derivatives instead of 11163.
# Since our initial condition already satisfies the algebraic constraint we skip the DAE
# initialization with `NoInit()`.
timestepper = Rodas5P(autodiff = AutoForwardDiff());
integrator = init(
    prob_mm, timestepper; initializealg = NoInit(), dt = Δt,
    adaptive = true, abstol = 1.0e-4, reltol = 1.0e-3,
    progress = true, progress_steps = 1,
);
#
# Instead of keeping the full solution in memory we step the integrator ourselves and export
# the fields as we go, into a single temporal VTKHDF file which can be viewed in
# [ParaView](https://www.paraview.org/). The internal (adaptive) steps are much finer than
# what is needed for a visualization, so we only write out a frame every `Δt_export`.
function export_solution(integrator, dh, filename; Δt_export = 5.0)
    t_next_export = 0.0
    vtkhdf = VTKHDFGridFile(filename, dh; temporal = true)
    for (u, t) in intervals(integrator)
        t < t_next_export && continue
        t_next_export = t + Δt_export
        write_timestep(vtkhdf, t) do vtk
            ## ParaView derives the name of a color map from the name of the data array,
            ## dropping all non-ASCII characters in the process -- which would make ϕₘ and
            ## ϕₑ share a single color map. Hence we export with ASCII names instead of
            ## letting `write_solution` name the arrays after the fields.
            for (field, name) in ((:ϕₘ, "phi_m"), (:ϕₑ, "phi_e"), (:s, "s"))
                write_node_data(vtk, evaluate_at_grid_nodes(dh, u, field), name)
            end
        end
    end
    close(vtkhdf)
    return
end

export_solution(integrator, dh, "bidomain.vtkhdf");

using Test                                                #hide
if IS_CI                                                  #hide
    @test all(isfinite, integrator.u)                     #hide
    ## the grounded dof must stay at zero                 #hide
    @test iszero(integrator.u[only(ch.prescribed_dofs)])  #hide
end                                                       #hide
nothing                                                   #hide
#md # ## [Plain Program](@id bidomain-plain-program)
#md #
#md # Below follows a version of the program without any comments.
#md # The file is also available here: [bidomain.jl](bidomain.jl)
#md #
#md # ```julia
#md # @__CODE__
#md # ```
