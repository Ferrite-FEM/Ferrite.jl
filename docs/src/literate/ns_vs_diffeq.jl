# # Incompressible Navier-Stokes Equations via [DifferentialEquations.jl]()
#
# ![](vortex-shedding.gif)
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
#    \partial_t v &= \eta \Delta v - (v \cdot \nabla) v - \nabla p \\
#    0 &= \nabla \cdot v
#  \end{aligned}
# ```
#
# where $v$ is the unknown velocity field, $p$ the unknown pressure field
# and $\eta$ the dynamic viscosity. We assume constant density of the fluid.
#
# ![](rect-domain-with-hole.png)
#
# Here $\Omega$ denotes the fluid domain, $\partial \Omega_{in}$, $\partial \Omega_{out}$ BCs (no-slip and inflow/outflow)...
#
# ### Weak Form
#
# ```math
#  \begin{aligned}
#    \int \partial_t v \cdot \phi &= - \int \nu \nabla v : \nabla \phi - \int (v \cdot \nabla) v \cdot \phi + \int p \nabla \cdot \phi + \int_{\partial \Omega_{out}} \\
#    0 &= \int \nabla \cdot v \psi
#  \end{aligned}
# ```
#
# where $\phi$ and $\psi$ are suitable test functions.
#
# Now we can discretize the problem as usual with the finite element method
# utilizing Taylor-Hood elements (Q2Q1) to yield a stable discretization.
#
# ```math
#  M [...] = K [...] + N([...])
# ```
#
#
# ## Commented Program
#
# Now we solve the problem in Ferrite. What follows is a program spliced with comments.
# The full program, without comments, can be found in the next [section](@ref ns_vs_diffeq-plain-program).
#
# First we load Ferrite, and some other packages we need
using Ferrite, SparseArrays, BlockArrays, OrdinaryDiffEq, LinearAlgebra, UnPack
# We start  generating a simple grid with 20x20 quadrilateral elements
# using `generate_grid`. The generator defaults to the unit square,
# so we don't need to specify the corners of the domain.
x_cells = 220
y_cells = 41
grid = generate_grid(Quadrilateral, (x_cells, y_cells), Vec{2}((0.0, 0.0)), Vec{2}((2.2, 0.41)));

# Carve hole in the mesh and update boundaries.
cell_indices = filter(ci->norm(mean(map(i->grid.nodes[i].x-[0.2,0.2], Ferrite.vertices(grid.cells[ci]))))>0.05, 1:length(grid.cells))
hole_cell_indices = filter(ci->norm(mean(map(i->grid.nodes[i].x-[0.2,0.2], Ferrite.vertices(grid.cells[ci]))))<=0.05, 1:length(grid.cells))
# Gather all faces in the ring and touching the ring
hole_face_ring = Set{FaceIndex}()
for hci ∈ hole_cell_indices
    push!(hole_face_ring, FaceIndex((hci+1, 4)))
    push!(hole_face_ring, FaceIndex((hci-1, 2)))
    push!(hole_face_ring, FaceIndex((hci-x_cells, 3)))
    push!(hole_face_ring, FaceIndex((hci+x_cells, 1)))
end
grid.facesets["hole"] = Set(filter(x->x.idx[1] ∉ hole_cell_indices, collect(hole_face_ring)))

cell_indices_map = map(ci->norm(mean(map(i->grid.nodes[i].x-[0.2,0.2], Ferrite.vertices(grid.cells[ci]))))>0.05 ? indexin([ci], cell_indices)[1] : 0, 1:length(grid.cells))
grid.cells = grid.cells[cell_indices]
for facesetname in keys(grid.facesets)
    grid.facesets[facesetname] = Set(map(fi -> FaceIndex( cell_indices_map[fi.idx[1]] ,fi.idx[2]), collect(grid.facesets[facesetname])))
end

# grid = saved_file_to_grid("holed_plate.msh")

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
T = 5
Δt₀ = 0.01
Δt_save = 0.05

ν = 0.001 #dynamic viscosity
vᵢₙ(t) = 1.0 #inflow velocity

ip_v = Lagrange{dim, RefCube, 2}()
ip_geom = Lagrange{dim, RefCube, 1}()
qr_v = QuadratureRule{dim, RefCube}(3)
cellvalues_v = CellVectorValues(qr_v, ip_v, ip_geom);

ip_p = Lagrange{dim, RefCube, 1}()
#Note that the pressure term comes in combination with a higher order test function...
qr_p = qr_v
cellvalues_p = CellScalarValues(qr_p, ip_p);

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
∂Ω_noslip = union(getfaceset.((grid, ), ["top", "bottom", "hole"])...);
∂Ω_inflow = getfaceset(grid, "left");
∂Ω_free = getfaceset(grid, "right");

# Now we are set up to define our constraint. We specify which field
# the condition is for, and our combined face set `∂Ω`. The last
# argument is a function which takes the spatial coordinate $x$ and
# the current time $t$ and returns the prescribed value. In this case
# it is trivial -- no matter what $x$ and $t$ we return $0$. When we have
# specified our constraint we `add!` it to `ch`.
noslip_bc = Dirichlet(:v, ∂Ω_noslip, (x, t) -> [0,0], [1,2])
add!(ch, noslip_bc);
inflow_bc = Dirichlet(:v, ∂Ω_inflow, (x, t) -> [clamp(t, 0.0, 1.0)*4*vᵢₙ(t)*x[2]*(0.41-x[2])/0.41^2,0], [1,2])
add!(ch, inflow_bc);

# We also need to `close!` and `update!` our boundary conditions. When we call `close!`
# the dofs which will be constrained by the boundary conditions are calculated and stored
# in our `ch` object. Since the boundary conditions are, in this case,
# independent of time we can `update!` them directly with e.g. $t = 0$.
close!(ch)
update!(ch, 0.0);

# ### Assembling the linear system
# Now we have all the pieces needed to assemble the linear system, $K u = f$.
# We define a function, `doassemble` to do the assembly, which takes our `cellvalues`,
# the sparse matrix and our DofHandler as input arguments. The function returns the
# assembled stiffness matrix, and the force vector.
function assemble_linear(cellvalues_v::CellVectorValues{dim}, cellvalues_p::CellScalarValues{dim}, ν, M::SparseMatrixCSC, K::SparseMatrixCSC, dh::DofHandler) where {dim}
    # We allocate the element stiffness matrix and element force vector
    # just once before looping over all the cells instead of allocating
    # them every time in the loop.
    #+
    n_basefuncs_v = getnbasefunctions(cellvalues_v)
    n_basefuncs_p = getnbasefunctions(cellvalues_p)
    n_basefuncs = n_basefuncs_v + n_basefuncs_p
    v▄, p▄ = 1, 2
    Me = PseudoBlockArray(zeros(n_basefuncs, n_basefuncs), [n_basefuncs_v, n_basefuncs_p], [n_basefuncs_v, n_basefuncs_p])
    Ke = PseudoBlockArray(zeros(n_basefuncs, n_basefuncs), [n_basefuncs_v, n_basefuncs_p], [n_basefuncs_v, n_basefuncs_p])

    # Next we define the global force vector `f` and use that and
    # the stiffness matrix `K` and create an assembler. The assembler
    # is just a thin wrapper around `f` and `K` and some extra storage
    # to make the assembling faster.
    #+
    f = zeros(ndofs(dh))
    stiffness_assembler = start_assemble(K)
    mass_assembler = start_assemble(M)

    # It is now time to loop over all the cells in our grid. We do this by iterating
    # over a `CellIterator`. The iterator caches some useful things for us, for example
    # the nodal coordinates for the cell, and the local degrees of freedom.
    #+
    @inbounds for cell in CellIterator(dh)
        # Always remember to reset the element stiffness matrix and
        # force vector since we reuse them for all elements.
        #+
        fill!(Me, 0)
        fill!(Ke, 0)

        # For each cell we also need to reinitialize the cached values in `cellvalues`.
        #+
        Ferrite.reinit!(cellvalues_v, cell)
        Ferrite.reinit!(cellvalues_p, cell)

        # It is now time to loop over all the quadrature points in the cell and
        # assemble the contribution to `Ke` and `fe`. The integration weight
        # can be queried from `cellvalues` by `getdetJdV`.
        #+
        for q_point in 1:getnquadpoints(cellvalues_v)
            dΩ = getdetJdV(cellvalues_v, q_point)
            #Mass term
            for i in 1:n_basefuncs_v
                v = shape_value(cellvalues_v, q_point, i)
                for j in 1:n_basefuncs_v
                    φ = shape_value(cellvalues_v, q_point, j)
                    Me[BlockIndex((v▄, v▄),(i, j))] += (v ⋅ φ) * dΩ
                end
            end
            # For each quadrature point we loop over all the (local) shape functions.
            # We need the value and gradient of the testfunction `v` and also the gradient
            # of the trial function `u`. We get all of these from `cellvalues`.
            #+
            #Viscosity term
            for i in 1:n_basefuncs_v
                ∇v = shape_gradient(cellvalues_v, q_point, i)
                for j in 1:n_basefuncs_v
                    ∇φ = shape_gradient(cellvalues_v, q_point, j)
                    Ke[BlockIndex((v▄, v▄), (i, j))] -= ν * (∇v ⊡ ∇φ) * dΩ
                end
            end
            #Incompressibility term
            for i in 1:n_basefuncs_p
                ψ = shape_value(cellvalues_p, q_point, i)
                for j in 1:n_basefuncs_v
                    divv = shape_divergence(cellvalues_v, q_point, j)
                    Ke[BlockIndex((p▄, v▄), (i, j))] += (ψ * divv) * dΩ
                end
            end
            #Pressure term
            dΩ = getdetJdV(cellvalues_p, q_point)
            for i in 1:n_basefuncs_v
                divφ = shape_divergence(cellvalues_v, q_point, i)
                for j in 1:n_basefuncs_p
                    p = shape_value(cellvalues_p, q_point, j)
                    Ke[BlockIndex((v▄, p▄), (i, j))] += (p * divφ) * dΩ
                end
            end
        end

        # The last step in the element loop is to assemble `Ke` and `fe`
        # into the global `K` and `f` with `assemble!`.
        #+
        assemble!(stiffness_assembler, celldofs(cell), Ke)
        assemble!(mass_assembler, celldofs(cell), Me)
    end
    return M, K
end

# ### Solution of the system
# The last step is to solve the system. First we call `doassemble`
# to obtain the global stiffness matrix `K` and force vector `f`.
M, K = assemble_linear(cellvalues_v, cellvalues_p, ν, M, K, dh);

# At the time of writing this example we have no clean way to hook into the
# nonlinear solver backend to apply the Dirichlet BCs. As a hotfix we override
# the newton initialization. We cannot solve all emerging issues by developing
# a customized newton algorithm here. This hack should only be seen as an
# intermediate step towards integration with OrdinaryDiffEq.jl.
function OrdinaryDiffEq.initialize!(nlsolver::OrdinaryDiffEq.NLSolver{<:NLNewton,true}, integrator)
    @unpack u,uprev,t,dt,opts = integrator
    @unpack z,tmp,cache = nlsolver
    @unpack weight = cache

    cache.invγdt = inv(dt * nlsolver.γ)
    cache.tstep = integrator.t + nlsolver.c * dt
    OrdinaryDiffEq.calculate_residuals!(weight, fill!(weight, one(eltype(u))), uprev, u,
                         opts.abstol, opts.reltol, opts.internalnorm, t);

    # Before starting the nonlinear solve we have to set the time correctly. Note that ch is a global variable.
    #+
    update!(ch, t);

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
# interface provided by the DifferentialEquations ecosystem.
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

# These are our initial conditions. We start from the zero solution as
# discussed above.
u₀ = zeros(ndofs(dh))
apply!(u₀, ch)

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
    K,dh,cellvalues = p
    du .= K * u

    n_basefuncs = getnquadpoints(cellvalues)

    ## Nonlinaer contribution
    for cell in CellIterator(dh)
        ## Trilinear form evaluation
        v_celldofs = celldofs(cell)
        Ferrite.reinit!(cellvalues, cell)
        v_cell = u[v_celldofs[dof_range(dh, :v)]]
        for q_point in 1:getnquadpoints(cellvalues)
            dΩ = getdetJdV(cellvalues, q_point)
            v_div = function_divergence(cellvalues, q_point, v_cell)
            v_val = function_value(cellvalues, q_point, v_cell)
            nl_contrib = - v_div * v_val
            for j in 1:n_basefuncs
                Nⱼ = shape_value(cellvalues, q_point, j)
                du[v_celldofs[j]] += nl_contrib ⋅ Nⱼ * dΩ
            end
        end
    end
end;
rhs = ODEFunction(navierstokes!, mass_matrix=M; jac_prototype=jac_sparsity)
p = [K, dh, cellvalues_v]
problem = ODEProblem(rhs, u₀, (0.0,T), p);

# Now we can put everything together by specifying how to solve the problem.
# We want to use the modified extended BDF2 method with our custom linear
# solver, which helps in the enforcement of the Dirichlet BDs. Further we
# enable the progress bar with the `progess` and `progress_steps` arguments.
# Finally we have to communicate the time step length and initialization
# algorithm. Since we start with a valid initial state we do not use one of
# DifferentialEquations.jl initialization algorithms.
# NOTE: At the time of writing this [no index 2 initialization is implemented](https://github.com/SciML/OrdinaryDiffEq.jl/issues/1019).
sol = solve(problem, MEBDF2(linsolve=FerriteLinSolve(ch)), progress=true, progress_steps=1, dt=Δt₀, saveat=Δt_save, initializealg=NoInit());

# ### Exporting to VTK
# To visualize the result we export the grid and our field `u`
# to a VTK-file, which can be viewed in e.g. [ParaView](https://www.paraview.org/).
pvd = paraview_collection("vortex-street.pvd");
# Now, we loop over all timesteps and solution vectors, in order to append them to the paraview collection.
for (solution,t) in zip(sol.u, sol.t)
    #compress=false flag because otherwise each vtk file will be stored in memory
    vtk_grid("vortex-street-$t.vtu", dh; compress=false) do vtk
        vtk_point_data(vtk,dh,solution)
        vtk_save(vtk)
        pvd[t] = vtk
    end
end
vtk_save(pvd);

# ## test the result                #src
# using Test                        #src
# @test norm(u) ≈ 3.307743912641305 #src

#md # ## [Plain Program](@id ns_vs_diffeq-plain-program)
#md #
#md # Below follows a version of the program without any comments.
#md # The file is also available here: [ns_vs_diffeq.jl](ns_vs_diffeq.jl)
#md #
#md # ```julia
#md # @__CODE__
#md # ```