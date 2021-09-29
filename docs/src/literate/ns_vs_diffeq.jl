# # Incompressible Navier-Stokes Equations via [DifferentialEquations.jl]()
#
# ![](https://user-images.githubusercontent.com/9196588/134514213-76d91d34-19ab-47c2-957e-16bb0c8669e1.gif)
#
#
# In this example we focus on a simple but visually appealing problem from
# fluid dynamics, namely vortex shedding. This problem is also known as
# [von-Karman vortex streets](https://en.wikipedia.org/wiki/K%C3%A1rm%C3%A1n_vortex_street). Within this example, we show how to utilize [DifferentialEquations.jl](https://github.com/SciML/DifferentialEquations.jl)
# in tandem with Ferrite.jl. To keep things simple we use a naive approach
# to discretize the system.
#
# ## Remarks on DifferentialEquations.jl
#
# Many "time step solvers" of [DifferentialEquations.jl](https://github.com/SciML/DifferentialEquations.jl) assume that that the
# problem is provided in mass matrix form. The incompressible Navier-Stokes
# equations as stated above yield a DAE in this form after applying a spatial
# discretization technique - in our case FEM. The mass matrix form a ODEs and DAEs
# is given as:
# ```math
#   M(t) \frac{du}{dt} = f(u,t)
# ```
# where $M$ is a possibly time-dependent mass matrix, $u$ our vector of unkowns
# and $f$ the right hand side. For us $f$ can be seen as the spatial discretization
# of all linear and non-linear operators depending on $u$ and $t$, but not on the time
# derivative of $u$.
#
# ## Some Theory on the Incompressible Navier-Stokes Equations
#
# ### Problem Description in Strong Form
#
# The incompressible Navier-Stokes equations can be stated as the system
#
# ```math
#  \begin{aligned}
#    \partial_t v &= \underbrace{\nu \Delta v}_{\text{viscosity}} - \underbrace{(v \cdot \nabla) v}_{\text{advection}} - \underbrace{\nabla p}_{\text{pressure}} \\
#               0 &= \underbrace{\nabla \cdot v}_{\text{incompressibility}}
#  \end{aligned}
# ```
#
# where $v$ is the unknown velocity field, $p$ the unknown pressure field,
# $\nu$ the dynamic viscosity and $\Delta$ the Laplacian. In the derivation we assumed
# a constant density of 1 for the fluid and negligible coupling between the velocity components.
# Finally we see that the pressure term appears only in combination with the gradient
# operator, so for any solution $p$ the function $p + c$ is also an admissible solution, if
# we do not impose Dirichlet conditions on the pressure. To resolve this we introduce the
# implicit constraint that $ \int_\Omega p = 0 $.
#
#
# Our setup is derived from [Turek's DFG benchmark](http://www.mathematik.tu-dortmund.de/~featflow/en/benchmarks/cfdbenchmarking/flow/dfg_benchmark1_re20.html).
# We model a channel with size $0.41 \times 2.2$ and a hole of radius $0.05$ centered at $(0.2, 0.2)$.
# The left side has a parabolic inflow profile, which is ramped up over time, modeled as the time dependent
# Dirichlet condition
# ```math
#  v(x,y,t)
#  =
#  \begin{bmatrix}
#      4 v_{in}(t) \cdot y \cdot (0.41-y)/0.41^2 \\
#      0
#  \end{bmatrix}
# ```
# where $v_{in}(t) = \text{clamp}(t, 0.0, 1.0)$. With a viscosity of $\nu = 0.001$
# this is enough to induce turbulence behind the cylinder which leads to vortex shedding. The top and bottom of our
# channel have no-slip conditions, i.e. $v = [0,0]^T$, while the right boundary has the do-nothing boundary condtion
# $\nu \partial_n v - p n = 0$ to model outflow. With these boundary conditions we can choose the zero solution as a
# feasible initial condition.
#
# ### Derivation of Semi-Discrete Weak Form
#
# By multiplying test functions $\varphi$ and $\psi$ from a suitable test function space on the strong form,
# followed by integrating over the domain and applying partial integration to the pressure and viscosity terms
# we can obtain the following weak form
# ```math
#  \begin{aligned}
#    \int \partial_t v \cdot \varphi &= - \int \nu \nabla v : \nabla \varphi - \int (v \cdot \nabla) v \cdot \varphi + \int p (\nabla \cdot \varphi) + \int_{\partial \Omega_{N}} \underbrace{(\nu \partial_n v - p n )}_{=0} \cdot \varphi \\
#                                  0 &= (\int \nabla \cdot v) \psi
#  \end{aligned}
# ```
# for all possible test functions from the suitable space.
#
# Now we can discretize the problem as usual with the finite element method
# utilizing Taylor-Hood elements (Q2Q1) to yield a stable discretization in
# mass matrix form:
# ```math
#  \underbrace{\begin{bmatrix}
#      M_v & 0 \\
#       0  & 0
#  \end{bmatrix}}_{:=M}
#  \begin{bmatrix}
#      \mathrm{d}_t\hat{\mathbf{v}} \\
#      \mathrm{d}\hat{p}
#  \end{bmatrix}
#  =
#  \underbrace{\begin{bmatrix}
#       A & B^T \\
#       B & 0
#  \end{bmatrix}}_{:=K}
#  \begin{bmatrix}
#      \hat{\mathbf{v}} \\
#      \hat{p}
#  \end{bmatrix}
#  +
#  \begin{bmatrix}
#      N(\hat{\mathbf{v}}, \hat{\mathbf{v}}, \hat{\varphi}) \\
#      0
#  \end{bmatrix}
# ```
# Here $M$ is the singular block mass matrix, $K$ is the discretized Stokes operator and $N$ the non-linear advection term.
#
#
# ## Commented Program
#
# Now we solve the problem in Ferrite with [DifferentialEquations.jl](https://github.com/SciML/DifferentialEquations.jl). What follows is a program spliced with comments.
# The full program, without comments, can be found in the next [section](@ref ns_vs_diffeq-plain-program).
#
# First we load Ferrite, and some other packages we need
using Ferrite, SparseArrays, BlockArrays, LinearAlgebra, UnPack
# Since we do not need the complete DifferentialEquations suite, we just load the required ODE infrastructure, which can also handle
# DAEs in mass matrix form.
using OrdinaryDiffEq

# We start off by defining our only material parameter.
ν = 1.0/1000.0; #dynamic viscosity

# Next a fine 2D rectangular grid has to be generated. We leave the cell size parametric for flexibility when
# playing around with the code. Note that the mesh is pretty fine, leading to a high memory consumption when
# feeding the equation system to direct solvers.
dim = 2
cell_scale_factor = 2.0
x_cells = round(Int, cell_scale_factor*220)
y_cells = round(Int, cell_scale_factor*41)
# CI chokes if the grid is too fine. :)     #src
x_cells = round(Int, 55/3)                  #hide
y_cells = round(Int, 41/3)                  #hide
grid = generate_grid(Quadrilateral, (x_cells, y_cells), Vec{2}((0.0, 0.0)), Vec{2}((2.2, 0.41)));

# Next we carve a hole $B_{0.05}(0.2,0.2)$ in the mesh by deleting the cells and update the boundary face sets.
# This code will be replaced once a proper mesh interface is avaliable.
cell_indices = filter(ci->norm(mean(map(i->grid.nodes[i].x-[0.2,0.2], Ferrite.vertices(grid.cells[ci]))))>0.05, 1:length(grid.cells))
hole_cell_indices = filter(ci->norm(mean(map(i->grid.nodes[i].x-[0.2,0.2], Ferrite.vertices(grid.cells[ci]))))<=0.05, 1:length(grid.cells));
hole_face_ring = Set{FaceIndex}()
for hci ∈ hole_cell_indices
    push!(hole_face_ring, FaceIndex((hci+1, 4)))
    push!(hole_face_ring, FaceIndex((hci-1, 2)))
    push!(hole_face_ring, FaceIndex((hci-x_cells, 3)))
    push!(hole_face_ring, FaceIndex((hci+x_cells, 1)))
end
grid.facesets["hole"] = Set(filter(x->x.idx[1] ∉ hole_cell_indices, collect(hole_face_ring)));
cell_indices_map = map(ci->norm(mean(map(i->grid.nodes[i].x-[0.2,0.2], Ferrite.vertices(grid.cells[ci]))))>0.05 ? indexin([ci], cell_indices)[1] : 0, 1:length(grid.cells))
grid.cells = grid.cells[cell_indices]
for facesetname in keys(grid.facesets)
    grid.facesets[facesetname] = Set(map(fi -> FaceIndex( cell_indices_map[fi.idx[1]] ,fi.idx[2]), collect(grid.facesets[facesetname])))
end;

# We test against full development of the flow - so regenerate the grid                              #src
grid = generate_grid(Quadrilateral, (x_cells, y_cells), Vec{2}((0.0, 0.0)), Vec{2}((0.55, 0.41)));   #hide

# ### Function space
# To ensure stability we utilize the Taylor-Hood element pair Q2-Q1.
# We have to utilize the same quadrature rule because in the weak form the
# linear pressure term is tested against a quadratic function.
ip_v = Lagrange{dim, RefCube, 2}()
ip_geom = Lagrange{dim, RefCube, 1}()
qr = QuadratureRule{dim, RefCube}(4)
cellvalues_v = CellVectorValues(qr, ip_v, ip_geom);

ip_p = Lagrange{dim, RefCube, 1}()
cellvalues_p = CellScalarValues(qr, ip_p, ip_geom);

dh = DofHandler(grid)
push!(dh, :v, dim, ip_v)
push!(dh, :p, 1, ip_p)
close!(dh);

# ### Boundary conditions
# As in the DFG benchmark we apply no-slip conditions to the top, bottom and
# cylinder boundary. The no-slip condition states that the velocity of the
# fluid on this portion of the boundary is fixed to be zero.
ch = ConstraintHandler(dh);

nosplip_face_names = ["top", "bottom", "hole"];
# No hole for the test present                                          #src
nosplip_face_names = ["top", "bottom"]                                  #hide
∂Ω_noslip = union(getfaceset.((grid, ), nosplip_face_names)...);
noslip_bc = Dirichlet(:v, ∂Ω_noslip, (x, t) -> [0,0], [1,2])
add!(ch, noslip_bc);

# The left boundary has a parabolic inflow with peak velocity of 1.0. This
# ensures that for the given geometry the Reynolds number is 100, which
# is already enough to obtain some simple vortex streets. By increasing the
# velocity further we can obtain stronger vortices - which may need additional
# refinement of the grid.
∂Ω_inflow = getfaceset(grid, "left");

vᵢₙ(t) = clamp(t, 0.0, 1.0)*1.0 #inflow velocity
vᵢₙ(t) = clamp(t, 0.0, 1.0)*0.3 #hide
parabolic_inflow_profile((x,y),t) = [4*vᵢₙ(t)*y*(0.41-y)/0.41^2,0]
inflow_bc = Dirichlet(:v, ∂Ω_inflow, parabolic_inflow_profile, [1,2])
add!(ch, inflow_bc);

# The outflow boundary condition has been applied on the right side of the
# cylinder when the weak form has been derived by setting the boundary integral
# to zero. It is also called the do-nothing condition. Other outflow conditions
# are also possible.
∂Ω_free = getfaceset(grid, "right");

close!(ch)
update!(ch, 0.0);

# ### Linear System Assembly
# Next we describe how the block mass matrix and the Stokes matrix are assembled.
#
# For the block mass matrix we remember that only the first equation had a time derivative
# and that the block mass matrix corresponds to the term arising from discretizing the time
# derivatives. Hence, only the upper left block has non-zero components.
function assemble_mass_matrix(cellvalues_v::CellVectorValues{dim}, cellvalues_p::CellScalarValues{dim}, M::SparseMatrixCSC, dh::DofHandler) where {dim}
    # We start again by allocating a buffer for the local matrix and some helpers, together with the assembler.
    #+
    n_basefuncs_v = getnbasefunctions(cellvalues_v)
    n_basefuncs_p = getnbasefunctions(cellvalues_p)
    n_basefuncs = n_basefuncs_v + n_basefuncs_p
    v▄, p▄ = 1, 2
    Mₑ = PseudoBlockArray(zeros(n_basefuncs, n_basefuncs), [n_basefuncs_v, n_basefuncs_p], [n_basefuncs_v, n_basefuncs_p])

    # It follows the assembly loop as explained in the basic tutorials.
    #+
    mass_assembler = start_assemble(M)
    @inbounds for cell in CellIterator(dh)
        fill!(Mₑ, 0)
        Ferrite.reinit!(cellvalues_v, cell)

        for q_point in 1:getnquadpoints(cellvalues_v)
            dΩ = getdetJdV(cellvalues_v, q_point)
            # Remember that we assemble a vector mass term, hence the dot product.
            #+
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

# Next we discuss the assembly of the Stokes matrix.
# Remember that we use the same function spaces for trial and test, hence the
# matrix has the following block form
# ```math
#   K = \begin{bmatrix}
#       A & B^T \\
#       B & 0
#   \end{bmatrix}
# ```
# which is also called saddle point matrix. These problems are known to have
# a non-trivial kernel, which is a reflection of the strong form as discussed
# in the theory portion if this example.
function assemble_stokes_matrix(cellvalues_v::CellVectorValues{dim}, cellvalues_p::CellScalarValues{dim}, ν, K::SparseMatrixCSC, dh::DofHandler) where {dim}
    # Again, we start again by allocating a buffer for the local matrix and some helpers, together with the assembler.
    #+
    n_basefuncs_v = getnbasefunctions(cellvalues_v)
    n_basefuncs_p = getnbasefunctions(cellvalues_p)
    n_basefuncs = n_basefuncs_v + n_basefuncs_p
    v▄, p▄ = 1, 2
    Kₑ = PseudoBlockArray(zeros(n_basefuncs, n_basefuncs), [n_basefuncs_v, n_basefuncs_p], [n_basefuncs_v, n_basefuncs_p])

    stiffness_assembler = start_assemble(K)

    @inbounds for cell in CellIterator(dh)
        fill!(Kₑ, 0)

        Ferrite.reinit!(cellvalues_v, cell)
        Ferrite.reinit!(cellvalues_p, cell)

        for q_point in 1:getnquadpoints(cellvalues_v)
            dΩ = getdetJdV(cellvalues_v, q_point)
            # Viscosity term $A$
            #+
            for i in 1:n_basefuncs_v
                ∇φᵢ = shape_gradient(cellvalues_v, q_point, i)
                for j in 1:n_basefuncs_v
                    ∇φⱼ = shape_gradient(cellvalues_v, q_point, j)
                    Kₑ[BlockIndex((v▄, v▄), (i, j))] -= ν * ∇φᵢ ⊡ ∇φⱼ * dΩ
                end
            end
            # Pressure + Incompressibility term $B$ - note the symmetry.
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

        # The last step in the element loop is to assemble `Kₑ` and `fe`
        # into the global `K` and `f` with `assemble!`.
        #+
        assemble!(stiffness_assembler, celldofs(cell), Kₑ)
    end
    return K
end;

# ### Solution of the semi-discretized system via DifferentialEquations.jl
# First we assemble the linear portions for efficiency. These matrices are
# assumed to be constant over time.
T = 10.0
Δt₀ = 0.01
Δt_save = 0.1

M = create_sparsity_pattern(dh);
M = assemble_mass_matrix(cellvalues_v, cellvalues_p, M, dh);

K = create_sparsity_pattern(dh);
K = assemble_stokes_matrix(cellvalues_v, cellvalues_p, ν, K, dh);

# These are our initial conditions. We start from the zero solution, because it
# is trivially admissible if the Dirichlet conditions are zero everywhere on the
# Dirichlet boundary for t=0. Note that the time stepper is also doing fine if the
# Dirichlet condition is non-zero and not too pathological.
u₀ = zeros(ndofs(dh))
apply!(u₀, ch);

# At the time of writing this example we have no clean way to hook into the
# nonlinear solver backend to apply the Dirichlet BCs. As a hotfix we override
# the newton initialization. We cannot solve all emerging issues by developing
# a customized newton algorithm here. This hack should only be seen as an
# intermediate step towards integration with OrdinaryDiffEq.jl.
function OrdinaryDiffEq.initialize!(nlsolver::OrdinaryDiffEq.NLSolver{<:NLNewton,true}, integrator)
    ## This block is copy pasta from OrdinaryDiffEq
    @unpack u,uprev,t,dt,opts = integrator
    @unpack z,tmp,cache = nlsolver
    @unpack weight = cache

    cache.invγdt = inv(dt * nlsolver.γ)
    cache.tstep = integrator.t + nlsolver.c * dt
    OrdinaryDiffEq.calculate_residuals!(weight, fill!(weight, one(eltype(u))), uprev, u,
                         opts.abstol, opts.reltol, opts.internalnorm, t);

    # Before starting the nonlinear solve we have to set the time correctly.
    # Note that ch is a global variable for now.
    #+
    update!(ch, cache.tstep);

    # The update of `u` takes `uprev + z` or `tmp + z` most of the time, so we have
    # to enforce Dirichlet BCs here. Note that these mutations may break the
    # error estimators.
    #+
    apply!(uprev, ch)
    apply!(tmp, ch)
    apply_zero!(z, ch);

    nothing
end;

# For the linear equations we can cleanly integrate with the linear solver
# interface provided by the DifferentialEquations ecosystem. We use a direct
# solver for simplicity, altough it comes with some issues. Implementing
# GMRES with efficient preconditioner is left open for future work.
mutable struct FerriteLinSolve{CH,F,T<:Factorization}
    ch::CH
    factorization::F
    A::T
end

FerriteLinSolve(ch) = FerriteLinSolve(ch,lu,lu(sparse(ones(1,1))))
function (p::FerriteLinSolve)(::Type{Val{:init}},f,u0_prototype)
    FerriteLinSolve(p.ch)
end

function (p::FerriteLinSolve)(x,A,b,update_matrix=false;reltol=nothing, kwargs...)
    if update_matrix
        ## Apply Dirichlet BCs
        apply_zero!(A, b, p.ch)
        ## Update factorization
        p.A = p.factorization(A)
    end
    ldiv!(x, p.A, b)
    apply_zero!(x, p.ch)
    return nothing
end;

# DifferentialEquations assumes dense matrices by default, which is not
# feasible for semi-discretization of finite element models. We communicate
# that a sparse matrix with specified pattern should be utilized through the
# `jac_prototyp` argument. Additionally, we have to provide the mass matrix.
# To apply the nonlinear portion of the Navier-Stokes problem we simply hand
# over the dof handler to the right hand side as a parameter in addition to
# the pre-assembled linear part (which is time independent) to save some
# runtime.
jac_sparsity = sparse(K)
function navierstokes!(du,u,p,t)
    K,dh,cellvalues_v = p
    du .= K * u

    n_basefuncs = getnbasefunctions(cellvalues_v)

    ## Nonlinear contribution
    for cell in CellIterator(dh)
        Ferrite.reinit!(cellvalues_v, cell)
        ## Trilinear form evaluation
        all_celldofs = celldofs(cell)
        v_celldofs = all_celldofs[dof_range(dh, :v)]
        v_cell = u[v_celldofs]
        for q_point in 1:getnquadpoints(cellvalues_v)
            dΩ = getdetJdV(cellvalues_v, q_point)
            ∇v = function_gradient(cellvalues_v, q_point, v_cell)
            v = function_value(cellvalues_v, q_point, v_cell)
            for j in 1:n_basefuncs
                φⱼ = shape_value(cellvalues_v, q_point, j)
                # Note that in Tensors.jl the definition $\textrm{grad} v = \nabla v$ holds.
                # With this information it can be quickly shown in index notation that
                # ```math
                # [(v \cdot \nabla) v]_i = v_j (\partial_j v_i) = [v (\nabla v)^T]_i
                # ```
                # where we should pay attentation to the transpose of the gradient.
                #+
                du[v_celldofs[j]] -= v ⋅ ∇v' ⋅ φⱼ * dΩ
            end
        end
    end
end;
rhs = ODEFunction(navierstokes!, mass_matrix=M; jac_prototype=jac_sparsity)
p = [K, dh, cellvalues_v]
problem = ODEProblem(rhs, u₀, (0.0,T), p);

# Now we can put everything together by specifying how to solve the problem.
# We want to use the adaptive implicit Euler method with our custom linear
# solver, which helps in the enforcement of the Dirichlet BCs. Further we
# enable the progress bar with the `progess` and `progress_steps` arguments.
# Finally we have to communicate the time step length and initialization
# algorithm. Since we start with a valid initial state we do not use one of
# DifferentialEquations.jl initialization algorithms.
# NOTE: At the time of writing this [no index 2 initialization is implemented](https://github.com/SciML/OrdinaryDiffEq.jl/issues/1019).
#
# To visualize the result we export the grid and our fields
# to VTK-files, which can be viewed in [ParaView](https://www.paraview.org/)
# by utilizing the corresponding pvd file.
timestepper = ImplicitEuler(linsolve=FerriteLinSolve(ch))
integrator = init(
    problem, timestepper, initializealg=NoInit(), dt=Δt₀,
    adaptive=true, abstol=1e-3, reltol=1e-3,
    progress=true, progress_steps=1,
    saveat=Δt_save);

pvd = paraview_collection("vortex-street.pvd");
integrator = TimeChoiceIterator(integrator, 0.0:Δt_save:T)
for (u,t) in integrator
    #compress=false flag because otherwise each vtk file will be stored in memory
    vtk_grid("vortex-street-$t.vtu", dh; compress=false) do vtk
        vtk_point_data(vtk,dh,u)
        vtk_save(vtk)
        pvd[t] = vtk
    end
end
vtk_save(pvd);

# Test the result for full proper development of the flow                   #src
function compute_divergence(dh, u, cellvalues_v)                            #src
    divv = 0.0                                                              #src
    @inbounds for (i,cell) in enumerate(CellIterator(dh))                   #src
        Ferrite.reinit!(cellvalues_v, cell)                                 #src
        for q_point in 1:getnquadpoints(cellvalues_v)                       #src
            dΩ = getdetJdV(cellvalues_v, q_point)                           #src
                                                                            #src
            all_celldofs = celldofs(cell)                                   #src
            v_celldofs = all_celldofs[dof_range(dh, :v)]                    #src
            v_cell = u[v_celldofs]                                          #src
                                                                            #src
            divv += function_divergence(cellvalues_v, q_point, v_cell) * dΩ #src
        end                                                                 #src
    end                                                                     #src
    return divv                                                             #src
end                                                                         #src
@testset "INS OrdinaryDiffEq" begin                                         #src
    u = integrator.integrator.u                                             #src
    Δdivv = abs(compute_divergence(dh, u, cellvalues_v))                    #src
    @test isapprox(Δdivv, 0.0, atol=1e-12)                                  #src
                                                                            #src
    Δv = 0.0                                                                #src
    for cell in CellIterator(dh)                                            #src
        Ferrite.reinit!(cellvalues_v, cell)                                 #src
        all_celldofs = celldofs(cell)                                       #src
        v_celldofs = all_celldofs[dof_range(dh, :v)]                        #src
        v_cell = u[v_celldofs]                                              #src
        coords = getcoordinates(cell)                                       #src
        for q_point in 1:getnquadpoints(cellvalues_v)                       #src
            dΩ = getdetJdV(cellvalues_v, q_point)                           #src
            coords_qp = spatial_coordinate(cellvalues_v, q_point, coords)   #src
            v = function_value(cellvalues_v, q_point, v_cell)               #src
            Δv += norm(v - parabolic_inflow_profile(coords_qp, T))^2*dΩ     #src
        end                                                                 #src
    end                                                                     #src
    @test isapprox(sqrt(Δv), 0.0, atol=1e-3)                                #src
end;                                                                        #src

#md # ## [Plain Program](@id ns_vs_diffeq-plain-program)
#md #
#md # Below follows a version of the program without any comments.
#md # The file is also available here: [ns_vs_diffeq.jl](ns_vs_diffeq.jl)
#md #
#md # ```julia
#md # @__CODE__
#md # ```