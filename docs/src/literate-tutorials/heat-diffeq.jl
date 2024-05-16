# First we load Ferrite and some other packages we need
using Ferrite, SparseArrays, BlockArrays, LinearAlgebra, UnPack, LinearSolve
# Since we do not need the complete DifferentialEquations suite, we just load the required ODE infrastructure, which can also handle
# DAEs in mass matrix form.
using OrdinaryDiffEq

####################################################################################################################################################################################
######################################################################### Ferrite Stuff To Get Matrices ############################################################################
####################################################################################################################################################################################

# x_cells = round(Int, 55/3)                  #hide
# y_cells = round(Int, 41/3)                  #hide
x_cells = round(Int, 1)                  #hide
y_cells = round(Int, 1)                  #hide
grid = generate_grid(Quadrilateral, (x_cells, y_cells), Vec{2}((0.0, 0.0)), Vec{2}((0.55, 0.41)));   #hide

# ### Function Space
# To ensure stability we utilize the Taylor-Hood element pair Q2-Q1.
# We have to utilize the same quadrature rule for the pressure as for the velocity, because in the weak form the
# linear pressure term is tested against a quadratic function.
ip_v = Lagrange{RefQuadrilateral, 1}()
qr = QuadratureRule{RefQuadrilateral}(2)
cellvalues_v = CellValues(qr, ip_v);

dh = DofHandler(grid)
add!(dh, :v, ip_v)
close!(dh);

# ### Boundary Conditions
ch = ConstraintHandler(dh);

∂Ω_inflow = getfaceset(grid, "left");
uᵢₙ(x,t) = 1.5/(1+exp(-2.0*(t-2.0)))  
inflow_bc = Dirichlet(:v, ∂Ω_inflow, uᵢₙ)
add!(ch, inflow_bc);
close!(ch)
update!(ch, 0.0);

# ### Linear System Assembly
# Next we describe how the block mass matrix and the Stokes matrix are assembled.
#
# For the block mass matrix $M$ we remember that only the first equation had a time derivative
# and that the block mass matrix corresponds to the term arising from discretizing the time
# derivatives. Hence, only the upper left block has non-zero components.
function assemble_mass_matrix(cellvalues_v::CellValues, M::SparseMatrixCSC, dh::DofHandler)
    ## Allocate a buffer for the local matrix and some helpers, together with the assembler.
    n_basefuncs_v = getnbasefunctions(cellvalues_v)
    Mₑ = zeros(n_basefuncs_v, n_basefuncs_v)

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
                    Mₑ[i,j] += φᵢ * φⱼ * dΩ
                end
            end
        end
        assemble!(mass_assembler, celldofs(cell), Mₑ)
    end

    return M
end;

function assemble_heat_matrix(cellvalues_v::CellValues, K::SparseMatrixCSC, dh::DofHandler)
    ## Again, some buffers and helpers
    n_basefuncs_v = getnbasefunctions(cellvalues_v)
    Kₑ = zeros(n_basefuncs_v, n_basefuncs_v)

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
                    Kₑ[i,j] -= ∇φᵢ ⋅ ∇φⱼ * dΩ
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
M = assemble_mass_matrix(cellvalues_v, M, dh);

K = create_sparsity_pattern(dh);
K = assemble_heat_matrix(cellvalues_v, K, dh);

u₀ = zeros(ndofs(dh))
apply!(u₀, ch);
jac_sparsity = create_sparsity_pattern(dh);


####################################################################################################################################################################################
################################################################################# Unrolled Solver ##################################################################################
####################################################################################################################################################################################

@show Matrix(M .- Δt₀*K)
@show cond(Matrix(M .- Δt₀*K))

u = copy(u₀)
un = copy(u₀)
du = copy(u₀)
pvd = paraview_collection("heat-debug.pvd");
for t in 0.0:Δt₀:T
    # Setup linear system
    A = M .- Δt₀*K
    # apply!(A, ch)
    # Setup residual
    r = M*un
    # Inner solve
    u .= A \ r
    @show u
    # Rate update
    du .= K*u #heat!(du,u,p,t)
    @show res = norm(M*(u - un)/Δt₀ - du)
    # Update solution
    un .= u
    # Write back
    vtk_grid("heat-debug-$t.vtu", dh) do vtk
        vtk_point_data(vtk,dh,u)
        vtk_save(vtk)
        pvd[t] = vtk
    end
end
vtk_save(pvd);

####################################################################################################################################################################################
######################################################################## OrdinaryDiffEq Utils ##################################################################################
####################################################################################################################################################################################

struct RHSparams
    K::SparseMatrixCSC
    ch::ConstraintHandler
    dh::DofHandler
    cellvalues_v::CellValues
    u::Vector
end
p = RHSparams(K, ch, dh, cellvalues_v, copy(u₀))

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

function ferrite_limiter!(u, integrator, p, t)
    update!(p.ch, t)
    apply!(u, p.ch)
end

####################################################################################################################################################################################
######################################################################## OrdinaryDiffEq with Mass ##################################################################################
####################################################################################################################################################################################

function heat!(du,u_uc,p::RHSparams,t)
    @show "rhs",t 
    @unpack K,ch,dh,cellvalues_v,u = p

    u .= u_uc
    update!(ch, t)
    apply!(u, ch)

    mul!(du, K, u) # du .= K * u

    Ferrite.compute_dirichlet_rates!(du, ch, t)
    # apply_zero!(du, ch)
end;

function heat_jac!(J,u,p,t)
    @show "jac", t

    @unpack K = p

    J .= K

    # apply!(J, ch)
end;

rhs = ODEFunction(heat!, mass_matrix=M; jac=heat_jac!, jac_prototype=jac_sparsity)
# rhs = ODEFunction(heat!, mass_matrix=OrdinaryDiffEq.I; jac=heat_jac!, jac_prototype=jac_sparsity)
# rhs = ODEFunction(heat!, mass_matrix=M; jac_prototype=jac_sparsity)
problem = ODEProblem(rhs, u₀, (0.0,T), p);

# timestepper = ImplicitEuler(linsolve = FerriteBackslash(), step_limiter! = ferrite_limiter!)
# timestepper = ImplicitEuler(linsolve = FerriteBackslash(), nlsolve=NonlinearSolveAlg(OrdinaryDiffEq.NonlinearSolve.NewtonRaphson(linsolve=FerriteBackslash())), step_limiter! = ferrite_limiter!) # Errors during AD, but jac is given
timestepper = ImplicitEuler(step_limiter! = ferrite_limiter!)

integrator = init(
    problem, timestepper, initializealg=NoInit(), dt=Δt₀,
    adaptive=false, abstol=1e-5, reltol=1e-4,
    progress=true, progress_steps=1,
    saveat=Δt_save, verbose=true,
    internalnorm=FreeDofErrorNorm(ch)
);

pvd = paraview_collection("heat.pvd");
integrator = TimeChoiceIterator(integrator, 0.0:Δt_save:T)
for (u,t) in integrator
    # We ignored the Dirichlet constraints in the solution vector up to now,
    # so we have to bring them back now.
    #+
    vtk_grid("heat-$t.vtu", dh) do vtk
        vtk_point_data(vtk,dh,u)
        vtk_save(vtk)
        pvd[t] = vtk
    end
end
vtk_save(pvd);

####################################################################################################################################################################################
######################################################################## OrdinaryDiffEq without Mass ###############################################################################
####################################################################################################################################################################################

function heat_no_mass!(du,u_uc,p::RHSparams,t)
    @show "rhs",t 
    @unpack K,ch,dh,cellvalues_v,u = p

    u .= u_uc
    update!(ch, t)
    apply!(u, ch)

    mul!(du, K, u) # du .= K * u

    Mdu = M \ du
    # Ferrite.compute_dirichlet_rates!(du, ch, t)
    du .= Mdu
end;

function heat_no_mass_jac!(J,u,p,t)
    @show "jac", t

    @unpack K = p

    J .= Matrix(inv(Matrix(M))*Matrix(K))
    @show typeof(J)
    # apply!(J, ch)
end;

rhs = ODEFunction(heat_no_mass!, jac=heat_no_mass_jac!, jac_prototype=Matrix(jac_sparsity))
# rhs = ODEFunction(heat_no_mass!, mass_matrix=OrdinaryDiffEq.I; jac=heat_no_mass_jac!, jac_prototype=jac_sparsity)
# rhs = ODEFunction(heat_no_mass!, mass_matrix=M; jac_prototype=jac_sparsity)
problem = ODEProblem(rhs, u₀, (0.0,T), p);

# timestepper = ImplicitEuler(linsolve = FerriteBackslash(), step_limiter! = ferrite_limiter!)
# timestepper = ImplicitEuler(linsolve = FerriteBackslash(), nlsolve=NonlinearSolveAlg(OrdinaryDiffEq.NonlinearSolve.NewtonRaphson(linsolve=FerriteBackslash())), step_limiter! = ferrite_limiter!) # Errors during AD, but jac is given
timestepper = ImplicitEuler(step_limiter! = ferrite_limiter!)

integrator = init(
    problem, timestepper, initializealg=NoInit(), dt=Δt₀,
    adaptive=false, abstol=1e-5, reltol=1e-4,
    progress=true, progress_steps=1,
    saveat=Δt_save, verbose=true,
    internalnorm=FreeDofErrorNorm(ch)
);

pvd = paraview_collection("heat_no_mass.pvd");
integrator = TimeChoiceIterator(integrator, 0.0:Δt_save:T)
for (u,t) in integrator
    # We ignored the Dirichlet constraints in the solution vector up to now,
    # so we have to bring them back now.
    #+
    vtk_grid("heat-$t.vtu", dh) do vtk
        vtk_point_data(vtk,dh,u)
        vtk_save(vtk)
        pvd[t] = vtk
    end
end
vtk_save(pvd);
