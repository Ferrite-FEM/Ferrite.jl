# # Incompressible Navier-Stokes Equations via [DifferentialEquations.jl]()
#
# ![](https://user-images.githubusercontent.com/9196588/134514213-76d91d34-19ab-47c2-957e-16bb0c8669e1.gif)
#
#
# In this example we focus on a simple but visually appealing problem from
# fluid dynamics, namely vortex shedding, which is also known as
# von-Karman vortex streets, to show how to utilize [DifferentialEquations.jl]()
# in tandem with [Ferrite.jl]().
#
# ## Remarks on DifferentialEquations.jl
#
# The "timestep solvers" of [DifferentialEquations.jl]() assume that that the
# problem is provided in mass matrix form. The incompressible Navier-Stokes
# equations can be rewritten in this form after discretization, yielding a DAE.
#
# ## Incompressible Navier-Stokes Equations
#
# ### Strong Form
#
# The incompressible Navier-Stokes equations can be stated as the system
#
# ```math
#  \begin{aligned}
#    \partial_t v &= \nu \Delta v - (v \cdot \nabla) v - \nabla p \\
#               0 &= \nabla \cdot v
#  \end{aligned}
# ```
#
# where $v$ is the unknown velocity field, $p$ the unknown pressure field
# and $\nu$ the dynamic viscosity. We assume a constant density of 1 for the fluid.
#
#
# ### Weak Form
#
# ```math
#  \begin{aligned}
#    \int \partial_t v \cdot \varphi &= - \int \nu \nabla v : \nabla \varphi - \int (v \cdot \nabla) v \cdot \varphi + \int p \nabla \cdot \varphi + \int_{\partial \Omega_{N}} (\nu \partial_n v - p n ) \cdot \varphi \\
#                                  0 &= \int \nabla \cdot v \psi
#  \end{aligned}
# ```
#
# where $\varphi$ and $\psi$ are suitable test functions. As the
#
# Now we can discretize the problem as usual with the finite element method
# utilizing Taylor-Hood elements (Q2Q1) to yield a stable discretization.
# ```math
#  M [\hat{\mathbf{v}}, \hat{p}] = K [\hat{\mathbf{v}}, \hat{p}] + [N(\hat{\mathbf{v}}, \hat{\mathbf{v}}, \hat{\varphi}), 0]
# ```
# Here M is the singular block mass matrix, K is the discretized Stokes operator and N the non-linear advective term.
#
# Details on the geometry and boundary conditions can be taken from [Turek's DFG benchmark](http://www.mathematik.tu-dortmund.de/~featflow/en/benchmarks/cfdbenchmarking/flow/dfg_benchmark1_re20.html).
#
# ## Commented Program
#
# Now we solve the problem in Ferrite. What follows is a program spliced with comments.
# The full program, without comments, can be found in the next [section](@ref ns_vs_diffeq-plain-program).
#
# First we load Ferrite, and some other packages we need
using Ferrite, SparseArrays, BlockArrays, LinearAlgebra, UnPack
# Since we do note need the complete DifferentialEquations suite just load the required ODE infrastructure.
using OrdinaryDiffEq

# We start  generating a simple grid with 20x20 quadrilateral elements
# using `generate_grid`. The generator defaults to the unit square,
# so we don't need to specify the corners of the domain.
x_cells = round(Int, 220)
y_cells = round(Int, 41)
# CI chokes if the grid is too fine. :)     #src
x_cells = round(Int, 55/3)                  #hide
y_cells = round(Int, 41/3)                  #hide
grid = generate_grid(Quadrilateral, (x_cells, y_cells), Vec{2}((0.0, 0.0)), Vec{2}((2.2, 0.41)));

# Carve hole in the mesh and update boundaries.
cell_indices = filter(ci->norm(mean(map(i->grid.nodes[i].x-[0.2,0.2], Ferrite.vertices(grid.cells[ci]))))>0.05, 1:length(grid.cells))
hole_cell_indices = filter(ci->norm(mean(map(i->grid.nodes[i].x-[0.2,0.2], Ferrite.vertices(grid.cells[ci]))))<=0.05, 1:length(grid.cells));
## Gather all faces in the ring and touching the ring
hole_face_ring = Set{FaceIndex}()
for hci ∈ hole_cell_indices
    push!(hole_face_ring, FaceIndex((hci+1, 4)))
    push!(hole_face_ring, FaceIndex((hci-1, 2)))
    push!(hole_face_ring, FaceIndex((hci-x_cells, 3)))
    push!(hole_face_ring, FaceIndex((hci+x_cells, 1)))
end
grid.facesets["hole"] = Set(filter(x->x.idx[1] ∉ hole_cell_indices, collect(hole_face_ring)));
## Finally update the node and cell indices
cell_indices_map = map(ci->norm(mean(map(i->grid.nodes[i].x-[0.2,0.2], Ferrite.vertices(grid.cells[ci]))))>0.05 ? indexin([ci], cell_indices)[1] : 0, 1:length(grid.cells))
grid.cells = grid.cells[cell_indices]
for facesetname in keys(grid.facesets)
    grid.facesets[facesetname] = Set(map(fi -> FaceIndex( cell_indices_map[fi.idx[1]] ,fi.idx[2]), collect(grid.facesets[facesetname])))
end;

# We test against full development of the flow - so regenerate the grid                              #src
grid = generate_grid(Quadrilateral, (x_cells, y_cells), Vec{2}((0.0, 0.0)), Vec{2}((0.55, 0.41)));   #hide

# ### Trial and test functions
# A `CellValues` facilitates the process of evaluating values and gradients of
# test and trial functions (among other things). Since the problem
# is a scalar problem we will use a `CellScalarValues` object. To define
# this we need to specify an interpolation space for the shape functions.
# We use Lagrange functions (both for interpolating the function and the geometry)
# based on the reference "cube". We also define a quadrature rule based on the
# same reference cube. We combine the interpolation and the quadrature rule
# to a `CellScalarValues` object.
dim = 2
T = 10.0
Δt₀ = 0.01
Δt_save = 0.1

ν = 1.0/1000.0                  #dynamic viscosity
vᵢₙ(t) = clamp(t, 0.0, 1.0)*1.0 #inflow velocity
vᵢₙ(t) = clamp(t, 0.0, 1.0)*0.3 #hide

ip_v = Lagrange{dim, RefCube, 2}()
ip_geom = Lagrange{dim, RefCube, 1}()
qr_v = QuadratureRule{dim, RefCube}(4)
cellvalues_v = CellVectorValues(qr_v, ip_v, ip_geom);

ip_p = Lagrange{dim, RefCube, 1}()
#Note that the pressure term comes in combination with a higher order test function...
qr_p = QuadratureRule{dim, RefCube}(4) # = qr_v
cellvalues_p = CellScalarValues(qr_p, ip_p, ip_geom);

# ### Degrees of freedom
# Next we need to define a `DofHandler`, which will take care of numbering
# and distribution of degrees of freedom for our approximated fields.
# We create the `DofHandler` and then add a single field called `u`.
# Lastly we `close!` the `DofHandler`, it is now that the dofs are distributed
# for all the elements.
dh = DofHandler(grid)
push!(dh, :v, dim, ip_v)
push!(dh, :p, 1, ip_p)
close!(dh);

# Now that we have distributed all our dofs we can create our tangent matrix,
# using `create_sparsity_pattern`. This function returns a sparse matrix
# with the correct elements stored.
M = create_sparsity_pattern(dh);
K = create_sparsity_pattern(dh);

# ### Boundary conditions
# In Ferrite constraints like Dirichlet boundary conditions
# are handled by a `ConstraintHandler`.
ch = ConstraintHandler(dh);

# Next we need to add constraints to `ch`. For this problem we define
# homogeneous Dirichlet boundary conditions on the whole boundary, i.e.
# the `union` of all the face sets on the boundary.
nosplip_face_names = ["top", "bottom", "hole"];
# No hole for the test present                                          #src
nosplip_face_names = ["top", "bottom"]                                  #hide
∂Ω_noslip = union(getfaceset.((grid, ), nosplip_face_names)...);
∂Ω_inflow = getfaceset(grid, "left");
∂Ω_free = getfaceset(grid, "right");

# Now we are set up to define our constraint. We specify which field
# the condition is for, and our combined face set `∂Ω`. The last
# argument is a function which takes the spatial coordinate $x$ and
# the current time $t$ and returns the prescribed value. In this case
# it is trivial -- no matter what $x$ and $t$ we return $0$. When we have
# specified our constraint we `add!` it to `ch`.
noslip_bc = Dirichlet(:v, ∂Ω_noslip, (x, t) -> [0,0], [1,2])
add!(ch, noslip_bc)

parabolic_inflow_profile((x,y),t) = [4*vᵢₙ(t)*y*(0.41-y)/0.41^2,0]
inflow_bc = Dirichlet(:v, ∂Ω_inflow, parabolic_inflow_profile, [1,2])
add!(ch, inflow_bc);

# We also need to `close!` and `update!` our boundary conditions. When we call `close!`
# the dofs which will be constrained by the boundary conditions are calculated and stored
# in our `ch` object. Since the boundary conditions are, in this case,
# independent of time we can `update!` them directly with e.g. $t = 0$.
close!(ch)
update!(ch, 0.0);

function assemble_mass_matrix(cellvalues_v::CellVectorValues{dim}, cellvalues_p::CellScalarValues{dim}, M::SparseMatrixCSC, dh::DofHandler) where {dim}
    # We allocate the element stiffness matrix and element force vector
    # just once before looping over all the cells instead of allocating
    # them every time in the loop.
    #+
    n_basefuncs_v = getnbasefunctions(cellvalues_v)
    n_basefuncs_p = getnbasefunctions(cellvalues_p)
    n_basefuncs = n_basefuncs_v + n_basefuncs_p
    v▄, p▄ = 1, 2
    Mₑ = PseudoBlockArray(zeros(n_basefuncs, n_basefuncs), [n_basefuncs_v, n_basefuncs_p], [n_basefuncs_v, n_basefuncs_p])

    # Next we define the global force vector `f` and use that and
    # the stiffness matrix `K` and create an assembler. The assembler
    # is just a thin wrapper around `f` and `K` and some extra storage
    # to make the assembling faster.
    #+
    stiffness_assembler = start_assemble(M)

    # It is now time to loop over all the cells in our grid. We do this by iterating
    # over a `CellIterator`. The iterator caches some useful things for us, for example
    # the nodal coordinates for the cell, and the local degrees of freedom.
    #+
    @inbounds for cell in CellIterator(dh)
        # Always remember to reset the element stiffness matrix and
        # force vector since we reuse them for all elements.
        #+
        fill!(Mₑ, 0)

        # For each cell we also need to reinitialize the cached values in `cellvalues`.
        #+
        Ferrite.reinit!(cellvalues_v, cell)

        # It is now time to loop over all the quadrature points in the cell and
        # assemble the contribution to `Kₑ` and `fe`. The integration weight
        # can be queried from `cellvalues` by `getdetJdV`.
        #+
        for q_point in 1:getnquadpoints(cellvalues_v)
            dΩ = getdetJdV(cellvalues_v, q_point)
            # For each quadrature point we loop over all the (local) shape functions.
            # We need the value and gradient of the testfunction `v` and also the gradient
            # of the trial function `u`. We get all of these from `cellvalues`.
            #+
            #Mass term
            for i in 1:n_basefuncs_v
                φᵢ = shape_value(cellvalues_v, q_point, i)
                for j in 1:n_basefuncs_v
                    φⱼ = shape_value(cellvalues_v, q_point, j)
                    Mₑ[BlockIndex((v▄, v▄), (i, j))] += φᵢ ⋅ φⱼ * dΩ
                end
            end
        end

        # The last step in the element loop is to assemble `Kₑ` and `fe`
        # into the global `K` and `f` with `assemble!`.
        #+
        assemble!(stiffness_assembler, celldofs(cell), Mₑ)
    end
    return M
end

M = assemble_mass_matrix(cellvalues_v, cellvalues_p, M, dh);

# ### Assembling the linear system
# Note: We assume negligible coupling between the velocity components in the viscosity assembly.
function assemble_stokes_matrix(cellvalues_v::CellVectorValues{dim}, cellvalues_p::CellScalarValues{dim}, ν, K::SparseMatrixCSC, dh::DofHandler) where {dim}
    # We allocate the element stiffness matrix and element force vector
    # just once before looping over all the cells instead of allocating
    # them every time in the loop.
    #+
    n_basefuncs_v = getnbasefunctions(cellvalues_v)
    n_basefuncs_p = getnbasefunctions(cellvalues_p)
    n_basefuncs = n_basefuncs_v + n_basefuncs_p
    v▄, p▄ = 1, 2
    Kₑ = PseudoBlockArray(zeros(n_basefuncs, n_basefuncs), [n_basefuncs_v, n_basefuncs_p], [n_basefuncs_v, n_basefuncs_p])

    # Next we define the global force vector `f` and use that and
    # the stiffness matrix `K` and create an assembler. The assembler
    # is just a thin wrapper around `f` and `K` and some extra storage
    # to make the assembling faster.
    #+
    stiffness_assembler = start_assemble(K)

    # It is now time to loop over all the cells in our grid. We do this by iterating
    # over a `CellIterator`. The iterator caches some useful things for us, for example
    # the nodal coordinates for the cell, and the local degrees of freedom.
    #+
    @inbounds for cell in CellIterator(dh)
        # Always remember to reset the element stiffness matrix and
        # force vector since we reuse them for all elements.
        #+
        fill!(Kₑ, 0)

        # For each cell we also need to reinitialize the cached values in `cellvalues`.
        #+
        Ferrite.reinit!(cellvalues_v, cell)
        Ferrite.reinit!(cellvalues_p, cell)

        # It is now time to loop over all the quadrature points in the cell and
        # assemble the contribution to `Kₑ` and `fe`. The integration weight
        # can be queried from `cellvalues` by `getdetJdV`.
        #+
        for q_point in 1:getnquadpoints(cellvalues_v)
            dΩ = getdetJdV(cellvalues_v, q_point)
            # For each quadrature point we loop over all the (local) shape functions.
            # We need the value and gradient of the testfunction `v` and also the gradient
            # of the trial function `u`. We get all of these from `cellvalues`.
            #+
            #Viscosity term
            for i in 1:n_basefuncs_v
                ∇φᵢ = shape_gradient(cellvalues_v, q_point, i)
                for j in 1:n_basefuncs_v
                    ∇φⱼ = shape_gradient(cellvalues_v, q_point, j)
                    Kₑ[BlockIndex((v▄, v▄), (i, j))] -= ν * ∇φᵢ ⊡ ∇φⱼ * dΩ
                end
            end
            #Pressure + Incompressibility term - note the symmetry.
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
end

# ### Solution of the system
# The last step is to solve the system. First we call `doassemble`
# to obtain the global stiffness matrix `K` and force vector `f`.
K = assemble_stokes_matrix(cellvalues_v, cellvalues_p, ν, K, dh);

# These are our initial conditions. We start from the zero solution as
# discussed above.
u₀ = zeros(ndofs(dh))
apply!(u₀, ch)

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

    # The update of u takes uprev + z or tmp + z most of the time, so we have
    # to enforce Dirichlet BCs here. Note that these mutations may break the
    # error estimators.
    #+
    apply!(uprev, ch)
    apply!(tmp, ch)
    apply_zero!(z, ch);

    nothing
end

# For the linear equations we can cleanly integrate with the linear solver
# interface provided by the DifferentialEquations ecosystem. We use a direct
# solver for simplicity, altough it comes with some issues. Implementing
# GMRES with efficient preconditioner is left open for future work.
mutable struct FerriteLinSolve{CH,F}
    ch::CH
    factorization::F
    A
end
FerriteLinSolve(ch) = FerriteLinSolve(ch,lu,nothing)
function (p::FerriteLinSolve)(::Type{Val{:init}},f,u0_prototype)
    FerriteLinSolve(ch)
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
end

# DifferentialEquations assumes dense matrices by default, which is not
# feasible for semi-discretization of finize element models. We communicate
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
                # Note that the order the gradient term is now on the left, which is the correct thing to do here.
                # It can be quickly shown through index notation.
                # ```math
                # [(v \cdot \nabla) v]_i = v_j \partial_j v_i = \partial_j v_i v_j = (\nabla v) v
                # ```
                #+
                du[v_celldofs[j]] -= ∇v ⋅ v ⋅ φⱼ * dΩ
            end
        end
    end
end;
rhs = ODEFunction(navierstokes!, mass_matrix=M; jac_prototype=jac_sparsity)
p = [K, dh, cellvalues_v]
problem = ODEProblem(rhs, u₀, (0.0,T), p);

# Now we can put everything together by specifying how to solve the problem.
# We want to use the modified extended BDF2 method with our custom linear
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
begin                                                                       #src
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
end                                                                         #src

#md # ## [Plain Program](@id ns_vs_diffeq-plain-program)
#md #
#md # Below follows a version of the program without any comments.
#md # The file is also available here: [ns_vs_diffeq.jl](ns_vs_diffeq.jl)
#md #
#md # ```julia
#md # @__CODE__
#md # ```