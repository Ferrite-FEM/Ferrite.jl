# First we load Ferrite and some other packages we need
using Ferrite, SparseArrays, BlockArrays, LinearAlgebra, UnPack, LinearSolve
# Since we do not need the complete DifferentialEquations suite, we just load the required ODE infrastructure, which can also handle
# DAEs in mass matrix form.
using OrdinaryDiffEq

# We start off by defining our only material parameter.
ν = 1.0/1000.0; #dynamic viscosity

# x_cells = round(Int, 55/3)                  #hide
# y_cells = round(Int, 41/3)                  #hide
x_cells = round(Int, 1)                  #hide
y_cells = round(Int, 1)                  #hide
grid = generate_grid(Quadrilateral, (x_cells, y_cells), Vec{2}((0.0, 0.0)), Vec{2}((0.55, 0.41)));   #hide

# ### Function Space
# To ensure stability we utilize the Taylor-Hood element pair Q2-Q1.
# We have to utilize the same quadrature rule for the pressure as for the velocity, because in the weak form the
# linear pressure term is tested against a quadratic function.
ip_v = Lagrange{RefQuadrilateral, 2}()^2
qr = QuadratureRule{RefQuadrilateral}(4)
cellvalues_v = CellValues(qr, ip_v);

ip_p = Lagrange{RefQuadrilateral, 1}()
cellvalues_p = CellValues(qr, ip_p);

dh = DofHandler(grid)
add!(dh, :v, ip_v)
add!(dh, :p, ip_p)
close!(dh);

# ### Boundary Conditions
# As in the DFG benchmark we apply no-slip conditions to the top, bottom and
# cylinder boundary. The no-slip condition states that the velocity of the
# fluid on this portion of the boundary is fixed to be zero.
ch = ConstraintHandler(dh);
                                   #src
nosplip_face_names = ["top", "bottom"]                                  #hide
∂Ω_noslip = union(getfaceset.((grid, ), nosplip_face_names)...);
noslip_bc = Dirichlet(:v, ∂Ω_noslip, (x, t) -> Vec((0.0,0.0)), [1,2])
add!(ch, noslip_bc);

# The left boundary has a parabolic inflow with peak velocity of 1.5. This
# ensures that for the given geometry the Reynolds number is 100, which
# is already enough to obtain some simple vortex streets. By increasing the
# velocity further we can obtain stronger vortices - which may need additional
# refinement of the grid. Note that we have to smoothly ramp up the velocity,
# because the Dirichlet constraints cannot be properly enforced yet, causing
# convergence issues.
∂Ω_inflow = getfaceset(grid, "left");

vᵢₙ(t) = 1.5/(1+exp(-2.0*(0.0-2.0)))  #inflow velocity
parabolic_inflow_profile(x,t) = [4*vᵢₙ(t)*x[2]*(0.41-x[2])/0.41^2, 0.0] # TODO vec
inflow_bc = Dirichlet(:v, ∂Ω_inflow, parabolic_inflow_profile, [1,2])
add!(ch, inflow_bc);

# The outflow boundary condition has been applied on the right side of the
# cylinder when the weak form has been derived by setting the boundary integral
# to zero. It is also called the do-nothing condition. Other outflow conditions
# are also possible.
∂Ω_free = getfaceset(grid, "right");

∂Ωref = Set([VertexIndex(1,1)])
pressure_bc = Dirichlet(:p, ∂Ωref, (x,t) -> 0.0)
add!(ch, pressure_bc);

close!(ch)
update!(ch, 0.0);

# ### Linear System Assembly
# Next we describe how the block mass matrix and the Stokes matrix are assembled.
#
# For the block mass matrix $M$ we remember that only the first equation had a time derivative
# and that the block mass matrix corresponds to the term arising from discretizing the time
# derivatives. Hence, only the upper left block has non-zero components.
function assemble_mass_matrix(cellvalues_v::CellValues, cellvalues_p::CellValues, M::SparseMatrixCSC, dh::DofHandler)
    ## Allocate a buffer for the local matrix and some helpers, together with the assembler.
    n_basefuncs_v = getnbasefunctions(cellvalues_v)
    n_basefuncs_p = getnbasefunctions(cellvalues_p)
    n_basefuncs = n_basefuncs_v + n_basefuncs_p
    v▄, p▄ = 1, 2
    Mₑ = PseudoBlockArray(zeros(n_basefuncs, n_basefuncs), [n_basefuncs_v, n_basefuncs_p], [n_basefuncs_v, n_basefuncs_p])

    ## It follows the assembly loop as explained in the basic tutorials.
    mass_assembler = start_assemble(M)
    for cell in CellIterator(dh)
        fill!(Mₑ, 0)
        Ferrite.reinit!(cellvalues_v, cell)

        for q_point in 1:getnquadpoints(cellvalues_v)
            dΩ = getdetJdV(cellvalues_v, q_point)
            ## Remember that we assemble a vector mass term, hence the dot product.
            ## There is only one time derivative on the left hand side, so only one mass block is non-zero.
            for i in 1:n_basefuncs_v
                φᵢ = shape_value(cellvalues_v, q_point, i)
                for j in 1:n_basefuncs_v
                    φⱼ = shape_value(cellvalues_v, q_point, j)
                    Mₑ[BlockIndex((v▄, v▄), (i, j))] += φᵢ ⋅ φⱼ * dΩ
                end
            end
        end
        assemble!(mass_assembler, celldofs(cell), Mₑ)
    end

    return M
end;

# Next we discuss the assembly of the Stokes matrix appearing on the right hand side.
# Remember that we use the same function spaces for trial and test, hence the
# matrix has the following block form
# ```math
#   K = \begin{bmatrix}
#       A & B^{\textrm{T}} \\
#       B & 0
#   \end{bmatrix}
# ```
# which is also called saddle point matrix. These problems are known to have
# a non-trivial kernel, which is a reflection of the strong form as discussed
# in the theory portion if this example.
function assemble_stokes_matrix(cellvalues_v::CellValues, cellvalues_p::CellValues, ν, K::SparseMatrixCSC, dh::DofHandler)
    ## Again, some buffers and helpers
    n_basefuncs_v = getnbasefunctions(cellvalues_v)
    n_basefuncs_p = getnbasefunctions(cellvalues_p)
    n_basefuncs = n_basefuncs_v + n_basefuncs_p
    v▄, p▄ = 1, 2
    Kₑ = PseudoBlockArray(zeros(n_basefuncs, n_basefuncs), [n_basefuncs_v, n_basefuncs_p], [n_basefuncs_v, n_basefuncs_p])

    ## Assembly loop
    stiffness_assembler = start_assemble(K)
    for cell in CellIterator(dh)
        ## Don't forget to initialize everything
        fill!(Kₑ, 0)

        Ferrite.reinit!(cellvalues_v, cell)
        Ferrite.reinit!(cellvalues_p, cell)

        for q_point in 1:getnquadpoints(cellvalues_v)
            dΩ = getdetJdV(cellvalues_v, q_point)
            # Assemble local viscosity block of $A$
            #+
            for i in 1:n_basefuncs_v
                ∇φᵢ = shape_gradient(cellvalues_v, q_point, i)
                for j in 1:n_basefuncs_v
                    ∇φⱼ = shape_gradient(cellvalues_v, q_point, j)
                    Kₑ[BlockIndex((v▄, v▄), (i, j))] -= ν * ∇φᵢ ⊡ ∇φⱼ * dΩ
                end
            end
            # Assemble local pressure and incompressibility blocks of $B^{\textrm{T}}$ and $B$.
            #+
            for j in 1:n_basefuncs_p
                ψ = shape_value(cellvalues_p, q_point, j)
                for i in 1:n_basefuncs_v
                    divφ = shape_divergence(cellvalues_v, q_point, i)
                    Kₑ[BlockIndex((v▄, p▄), (i, j))] += (divφ * ψ) * dΩ
                    Kₑ[BlockIndex((p▄, v▄), (j, i))] += (ψ * divφ) * dΩ
                end
            end
        end

        ## Assemble `Kₑ` into the Stokes matrix `K`.
        assemble!(stiffness_assembler, celldofs(cell), Kₑ)
    end
    return K
end;

# ### Solution of the semi-discretized system via DifferentialEquations.jl
# First we assemble the linear portions for efficiency. These matrices are
# assumed to be constant over time.
T = 1.0
Δt₀ = 0.1
Δt_save = 0.025

M = create_sparsity_pattern(dh);
M = assemble_mass_matrix(cellvalues_v, cellvalues_p, M, dh);

K = create_sparsity_pattern(dh);
K = assemble_stokes_matrix(cellvalues_v, cellvalues_p, ν, K, dh);

u₀ = zeros(ndofs(dh))
apply!(u₀, ch);
jac_sparsity = create_sparsity_pattern(dh);

struct RHSparams
    K::SparseMatrixCSC
    ch::ConstraintHandler
    dh::DofHandler
    cellvalues_v::CellValues
    u::Vector
end
p = RHSparams(K, ch, dh, cellvalues_v, copy(u₀))

function ferrite_limiter!(u, integrator, p, t)
    update!(p.ch, t)
    apply!(u, p.ch)
end

function stokes!(du,u_uc,p::RHSparams,t)
    @show "rhs",t 
    @unpack K,ch,dh,cellvalues_v,u = p

    u .= u_uc
    update!(ch, t)
    apply!(u, ch)

    mul!(du, K, u) # du .= K * u

    # Ferrite.compute_dirichlet_rates!(du, ch, t)
    # apply_zero!(du, ch)
end;

function stokes_jac!(J,u,p,t)
    @show "jac", t

    @unpack K = p

    J .= K

    # apply!(J, ch)
end;

@show M .- Δt₀*K
@show det(M .- Δt₀*K)

u = copy(u₀)
un = copy(u₀)
du = copy(u₀)
pvd = paraview_collection("stokes-debug.pvd");
for t in 0.0:Δt₀:T
    # Setup linear system
    A = M .- Δt₀*K # K = J for linear problem
    # apply!(A, ch) # This should happen here
    # Setup residual
    r = M*un
    # Inner solve
    u .= A \ r
    @show u
    # Rate update
    du .= K*u #stokes!(du,u,p,t)
    @show res = norm(M*(u - un)/Δt₀ - du)
    # Update solution
    un .= u
    # Write back
    vtk_grid("stokes-debug-$t.vtu", dh) do vtk
        vtk_point_data(vtk,dh,u)
        vtk_save(vtk)
        pvd[t] = vtk
    end
end
vtk_save(pvd);

rhs = ODEFunction(stokes!, mass_matrix=M; jac=stokes_jac!, jac_prototype=jac_sparsity)
# rhs = ODEFunction(stokes!, mass_matrix=OrdinaryDiffEq.I; jac=stokes_jac!, jac_prototype=jac_sparsity)
# rhs = ODEFunction(stokes!, mass_matrix=M; jac_prototype=jac_sparsity)
problem = ODEProblem(rhs, u₀, (0.0,T), p);

struct FreeDofErrorNorm
    ch::ConstraintHandler
end
(fe_norm::FreeDofErrorNorm)(u::Union{AbstractFloat, Complex}, t) = DiffEqBase.ODE_DEFAULT_NORM(u, t)
(fe_norm::FreeDofErrorNorm)(u::AbstractArray, t) = DiffEqBase.ODE_DEFAULT_NORM(u[fe_norm.ch.free_dofs], t)


mutable struct FerriteBackslash
end
OrdinaryDiffEq.NonlinearSolve.LinearSolve.needs_concrete_A(::FerriteBackslash) = true
function SciMLBase.solve!(cache::OrdinaryDiffEq.NonlinearSolve.LinearSolve.LinearCache, alg::FerriteBackslash; kwargs...)
    @unpack A,b,u = cache
    if verbose == true
        println("solving Ax=b")
    end
    apply_zero!(A,b,p.ch)
    u .= A \ b
    @show u
    return u
end
# timestepper = ImplicitEuler(linsolve = FerriteBackslash(), step_limiter! = ferrite_limiter!)
# timestepper = ImplicitEuler(linsolve = FerriteBackslash(), nlsolve=NonlinearSolveAlg(OrdinaryDiffEq.NonlinearSolve.NewtonRaphson(linsolve=FerriteBackslash())), step_limiter! = ferrite_limiter!)
timestepper = ImplicitEuler(linsolve = UMFPACKFactorization(reuse_symbolic=false),step_limiter! = ferrite_limiter!)

integrator = init(
    problem, timestepper, initializealg=NoInit(), dt=Δt₀,
    adaptive=false, abstol=1e-5, reltol=1e-4,
    progress=true, progress_steps=1,
    saveat=Δt_save, verbose=true,
    internalnorm=FreeDofErrorNorm(ch)
);

pvd = paraview_collection("stokes.pvd");
integrator = TimeChoiceIterator(integrator, 0.0:Δt_save:T)
for (u,t) in integrator
    # We ignored the Dirichlet constraints in the solution vector up to now,
    # so we have to bring them back now.
    #+
    vtk_grid("stokes-$t.vtu", dh) do vtk
        vtk_point_data(vtk,dh,u)
        vtk_save(vtk)
        pvd[t] = vtk
    end
end
vtk_save(pvd);
