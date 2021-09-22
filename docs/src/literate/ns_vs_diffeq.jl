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
using Ferrite, SparseArrays, BlockArrays, LinearAlgebra, UnPack
using Preconditioners
# using OrdinaryDiffEq
using FerriteGmsh
using IterativeSolvers
# We start  generating a simple grid with 20x20 quadrilateral elements
# using `generate_grid`. The generator defaults to the unit square,
# so we don't need to specify the corners of the domain.
x_cells = round(Int, 32)
y_cells = round(Int, 32)
grid = generate_grid(Quadrilateral, (x_cells, y_cells), Vec{2}((0.0, 0.0)), Vec{2}((1.0, 1.0)));

# x_cells = round(Int, 1.25*220)
# y_cells = round(Int, 1.25*41)
# grid = generate_grid(Quadrilateral, (x_cells, y_cells), Vec{2}((0.0, 0.0)), Vec{2}((2.2, 0.41)));

# # Carve hole in the mesh and update boundaries.
# cell_indices = filter(ci->norm(mean(map(i->grid.nodes[i].x-[0.2,0.2], Ferrite.vertices(grid.cells[ci]))))>0.05, 1:length(grid.cells))
# hole_cell_indices = filter(ci->norm(mean(map(i->grid.nodes[i].x-[0.2,0.2], Ferrite.vertices(grid.cells[ci]))))<=0.05, 1:length(grid.cells))
# # Gather all faces in the ring and touching the ring
# hole_face_ring = Set{FaceIndex}()
# for hci ∈ hole_cell_indices
#     push!(hole_face_ring, FaceIndex((hci+1, 4)))
#     push!(hole_face_ring, FaceIndex((hci-1, 2)))
#     push!(hole_face_ring, FaceIndex((hci-x_cells, 3)))
#     push!(hole_face_ring, FaceIndex((hci+x_cells, 1)))
# end
# grid.facesets["hole"] = Set(filter(x->x.idx[1] ∉ hole_cell_indices, collect(hole_face_ring)))

# cell_indices_map = map(ci->norm(mean(map(i->grid.nodes[i].x-[0.2,0.2], Ferrite.vertices(grid.cells[ci]))))>0.05 ? indexin([ci], cell_indices)[1] : 0, 1:length(grid.cells))
# grid.cells = grid.cells[cell_indices]
# for facesetname in keys(grid.facesets)
#     grid.facesets[facesetname] = Set(map(fi -> FaceIndex( cell_indices_map[fi.idx[1]] ,fi.idx[2]), collect(grid.facesets[facesetname])))
# end

# grid = saved_file_to_grid("holed_plate.msh")

# grid = generate_grid(Quadrilateral, (50, 50), Vec{2}((0.0, 0.0)), Vec{2}((2π, 2π)));

# # ### 2D Taylor-Green Solution
# F(t) = exp(-2*ν*t)
# #vₐₙₐ(x,y,t) = [cos(y)*sin(x)*F(t), -sin(y)*cos(x)*F(t)]
# vₐₙₐ(x,y,t) = [cos(y)*sin(x)*F(t), -sin(y)*cos(x)*F(t)]
# pₐₙₐ(x,y,t) = -(cos(2*x) + cos(2*y))/4.0*F(t)^2

# vₐₙₐ(x,y,t) = [-256*x^2*(x-1)^2*y*(y-1)*(2*y-1), 256*y^2*(y-1)^2*x*(x-1)*(2*x-1)]
# pₐₙₐ(x,y,t) = 150*(x-0.5)*(y-0.5)

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

ϵᵣ = 0.0# stabilization strength
γ = 1.0# other stabilization term for div-grad stabilization...

# ν = 0.001 #dynamic viscosity
ν = 1.0/100.0
vᵢₙ(t) = 0.3 #inflow velocity
#vᵢₙ(t) = clamp(t, 0.0, 1.0)*0.3 #inflow velocity

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
# ∂Ω_noslip = union(getfaceset.((grid, ), ["top", "bottom", "hole"])...);
# ∂Ω_noslip = union(getfaceset.((grid, ), ["top", "bottom"])...);
# ∂Ω_inflow = getfaceset(grid, "left");
# ∂Ω_free = getfaceset(grid, "right");

# ∂Ω = union(getfaceset.((grid, ), ["top", "bottom", "left", "right"])...);

∂Ω_noslip = union(getfaceset.((grid, ), ["bottom", "left", "right"])...);
∂Ω_lidflow = getfaceset(grid, "top");

# Now we are set up to define our constraint. We specify which field
# the condition is for, and our combined face set `∂Ω`. The last
# argument is a function which takes the spatial coordinate $x$ and
# the current time $t$ and returns the prescribed value. In this case
# it is trivial -- no matter what $x$ and $t$ we return $0$. When we have
# specified our constraint we `add!` it to `ch`.
noslip_bc = Dirichlet(:v, ∂Ω_noslip, (x, t) -> [0,0], [1,2])
add!(ch, noslip_bc);

# inflow_bc = Dirichlet(:v, ∂Ω_inflow, ((x,y), t) -> [4*vᵢₙ(t)*y*(0.41-y)/0.41^2,0], [1,2])
# add!(ch, inflow_bc);

lid_bc = Dirichlet(:v, ∂Ω_lidflow, (x, t) -> [1.0,0.0], [1,2])
add!(ch, lid_bc);

# add!(ch, Dirichlet(:v, ∂Ω, (x, t) -> vₐₙₐ(x[1], x[2], t), [1,2]));
# add!(ch, Dirichlet(:p, ∂Ω, (x, t) -> pₐₙₐ(x[1], x[2], t)));

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
    n_basefuncs_v_dim = Int(n_basefuncs_v/dim)
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
                φᵢ = shape_value(cellvalues_v, q_point, i)
                for j in 1:n_basefuncs_v
                    φⱼ = shape_value(cellvalues_v, q_point, j)
                    Me[BlockIndex((v▄, v▄),(i, j))] += (φᵢ ⋅ φⱼ) * dΩ
                end
            end
            # For each quadrature point we loop over all the (local) shape functions.
            # We need the value and gradient of the testfunction `v` and also the gradient
            # of the trial function `u`. We get all of these from `cellvalues`.
            #+
            ## Viscosity term
            # for i in 1:n_basefuncs_v
            #     ∇φᵢ = shape_symmetric_gradient(cellvalues_v, q_point, i)
            #     for j in 1:n_basefuncs_v
            #         ∇φⱼ = shape_symmetric_gradient(cellvalues_v, q_point, j)
            #         Ke[BlockIndex((v▄, v▄), (i, j))] -= ν * 2* ∇φᵢ ⊡ ∇φⱼ * dΩ
            #     end
            # end
            for i in 1:n_basefuncs_v
                ∇φᵢ = shape_gradient(cellvalues_v, q_point, i)
                for j in 1:n_basefuncs_v
                    ∇φⱼ = shape_gradient(cellvalues_v, q_point, j)
                    Ke[BlockIndex((v▄, v▄), (i, j))] -= ν * ∇φᵢ ⊡ ∇φⱼ * dΩ
                end
            end
            #Pressure term
            # for j in 1:n_basefuncs_p
            #     ψ = shape_value(cellvalues_p, q_point, j)
            #     for i in 1:n_basefuncs_v_dim
            #         for d in 1:dim
            #             comp_idx = dim*(i-1) + d
            #             #comp_idx = n_basefuncs_v_dim*(d-1) + i
            #             ∂φ∂xd = cellvalues_v.dNdx[comp_idx,q_point][d,d]
            #             Ke[BlockIndex((v▄, p▄), (comp_idx, j))] += (ψ * ∂φ∂xd) * dΩ
            #             Ke[BlockIndex((p▄, v▄), (j, comp_idx))] += (ψ * ∂φ∂xd) * dΩ
            #         end
            #     end
            # end
            for j in 1:n_basefuncs_p
                ψ = shape_value(cellvalues_p, q_point, j)
                for i in 1:n_basefuncs_v
                    divφ = shape_divergence(cellvalues_v, q_point, i)
                    Ke[BlockIndex((v▄, p▄), (i, j))] += (ψ * divφ) * dΩ
                    Ke[BlockIndex((p▄, v▄), (j, i))] += (ψ * divφ) * dΩ
                end
                #pressure stabilization
                Ke[BlockIndex((p▄, p▄), (j, j))] += ϵᵣ * ψ * ψ  * dΩ
            end
            #Incompressibility term
            # for i in 1:n_basefuncs_p
            #     ψ = shape_value(cellvalues_p, q_point, i)
            #     for j in 1:n_basefuncs_v
            #         divv = shape_divergence(cellvalues_v, q_point, j)
            #         Ke[BlockIndex((p▄, v▄), (i, j))] += (ψ * divv) * dΩ
            #     end
            # end
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
# function OrdinaryDiffEq.initialize!(nlsolver::OrdinaryDiffEq.NLSolver{<:NLNewton,true}, integrator)
#     @unpack u,uprev,t,dt,opts = integrator
#     @unpack z,tmp,cache = nlsolver
#     @unpack weight = cache

#     cache.invγdt = inv(dt * nlsolver.γ)
#     cache.tstep = integrator.t + nlsolver.c * dt
#     OrdinaryDiffEq.calculate_residuals!(weight, fill!(weight, one(eltype(u))), uprev, u,
#                          opts.abstol, opts.reltol, opts.internalnorm, t);

#     # Before starting the nonlinear solve we have to set the time correctly. Note that ch is a global variable.
#     #+
#     update!(ch, t);

#     # The update of u takes uprev + z or tmp + z most of the time, so we have
#     # to enforce Dirichlet BCs here. Note that these mutations may break the
#     # error estimators.
#     #+
#     apply!(uprev, ch)
#     apply!(tmp, ch)
#     apply_zero!(z, ch);

#     nothing
# end

# # For the linear equations we can cleanly integrate with the linear solver
# # interface provided by the DifferentialEquations ecosystem.
# mutable struct FerriteLinSolve{CH,F}
#     ch::CH
#     factorization::F
#     A
# end
# FerriteLinSolve(ch) = FerriteLinSolve(ch,lu,nothing)
# function (p::FerriteLinSolve)(::Type{Val{:init}},f,u0_prototype)
#     FerriteLinSolve(ch)
# end
# function (p::FerriteLinSolve)(x,A,b,update_matrix=false;reltol=nothing, kwargs...)
#     if update_matrix
#         ## Apply Dirichlet BCs
#         apply_zero!(A, b, p.ch)
#         ## Update factorization
#         p.A = p.factorization(A)
#     end
#     ldiv!(x, p.A, b)
#     apply_zero!(x, p.ch)
#     return nothing
# end

# These are our initial conditions. We start from the zero solution as
# discussed above.
u₀ = zeros(ndofs(dh))
apply!(u₀, ch)

# L2-projection of initial conditions
# function project_solution(cellvalues_v::CellVectorValues{dim}, cellvalues_p::CellScalarValues{dim}, dh::DofHandler, t) where {dim}
#     # We allocate the element stiffness matrix and element force vector
#     # just once before looping over all the cells instead of allocating
#     # them every time in the loop.
#     #+
#     n_basefuncs_v = getnbasefunctions(cellvalues_v)
#     n_basefuncs_p = getnbasefunctions(cellvalues_p)
#     n_basefuncs = n_basefuncs_v + n_basefuncs_p
#     v▄, p▄ = 1, 2
#     Me = PseudoBlockArray(zeros(n_basefuncs, n_basefuncs), [n_basefuncs_v, n_basefuncs_p], [n_basefuncs_v, n_basefuncs_p])
#     fe = PseudoBlockArray(zeros(n_basefuncs), [n_basefuncs_v, n_basefuncs_p])

#     # Next we define the global force vector `f` and use that and
#     # the stiffness matrix `K` and create an assembler. The assembler
#     # is just a thin wrapper around `f` and `K` and some extra storage
#     # to make the assembling faster.
#     #+
#     f = zeros(ndofs(dh))
#     M = create_sparsity_pattern(dh)
#     mass_assembler = start_assemble(M, f)

#     # It is now time to loop over all the cells in our grid. We do this by iterating
#     # over a `CellIterator`. The iterator caches some useful things for us, for example
#     # the nodal coordinates for the cell, and the local degrees of freedom.
#     #+
#     @inbounds for cell in CellIterator(dh)
#         # Always remember to reset the element stiffness matrix and
#         # force vector since we reuse them for all elements.
#         #+
#         fill!(Me, 0)
#         fill!(fe, 0)

#         coords = getcoordinates(cell)

#         # For each cell we also need to reinitialize the cached values in `cellvalues`.
#         #+
#         Ferrite.reinit!(cellvalues_v, cell)
#         Ferrite.reinit!(cellvalues_p, cell)

#         # It is now time to loop over all the quadrature points in the cell and
#         # assemble the contribution to `Ke` and `fe`. The integration weight
#         # can be queried from `cellvalues` by `getdetJdV`.
#         #+
#         for q_point in 1:getnquadpoints(cellvalues_v)
#             dΩ = getdetJdV(cellvalues_v, q_point)
#             coords_qp = spatial_coordinate(cellvalues_v, q_point, coords)

#             v_local = vₐₙₐ(coords_qp[1], coords_qp[2], t)
#             p_local = pₐₙₐ(coords_qp[1], coords_qp[2], t)

#             #Mass terms
#             for i in 1:n_basefuncs_v
#                 v = shape_value(cellvalues_v, q_point, i)
#                 for j in 1:n_basefuncs_v
#                     φ = shape_value(cellvalues_v, q_point, j)
#                     Me[BlockIndex((v▄, v▄),(i, j))] += (v ⋅ φ) * dΩ
#                 end
#                 fe[BlockIndex((v▄),(i))] += v_local ⋅ v * dΩ
#             end
#             for i in 1:n_basefuncs_p
#                 p = shape_value(cellvalues_p, q_point, i)
#                 for j in 1:n_basefuncs_p
#                     φ = shape_value(cellvalues_p, q_point, j)
#                     Me[BlockIndex((p▄, p▄),(i, j))] += (p ⋅ φ) * dΩ
#                 end
#                 fe[BlockIndex((p▄),(i))] += p_local * p * dΩ
#             end
#         end

#         # The last step in the element loop is to assemble `Ke` and `fe`
#         # into the global `K` and `f` with `assemble!`.
#         #+
#         assemble!(mass_assembler, celldofs(cell), Me, fe)
#     end
#     return M\f
# end
# u₀ = project_solution(cellvalues_v, cellvalues_p, dh, 0.0)
# apply!(u₀, ch)

# Steady-State Stokes Flow
using ProgressMeter
# SBDF-1
function solve_transient_stokes()
    update!(ch, 0.0)
    apply!(u₀,ch)

    u_prev = copy(u₀)
    u = copy(u₀)

    pvd = paraview_collection("transient-stokes-flow.pvd");

    t = 0.0
    vtk_grid("transient-stokes-flow-$t.vtu", dh; compress=false) do vtk
        vtk_point_data(vtk,dh,u)
        vtk_save(vtk)
        pvd[t] = vtk
    end

    #A = M/Δt₀ - K
    #rhsdata = get_rhs_data(ch, A)
    #apply!(A,ch)
    #A_f = lu(A)

    @showprogress for t = Δt₀:Δt₀:T
        A = M/Δt₀ - K
        b = M/Δt₀*u_prev
        ## Nonlinear contribution
        b_nl = zeros(length(b))
        n_basefuncs = getnbasefunctions(cellvalues_v)
        for cell in CellIterator(dh)
            Ferrite.reinit!(cellvalues_v, cell)
            ## Trilinear form evaluation
            all_celldofs = celldofs(cell)
            v_celldofs = all_celldofs[dof_range(dh, :v)]
            v_cell = u_prev[v_celldofs]
            for q_point in 1:getnquadpoints(cellvalues_v)
                dΩ = getdetJdV(cellvalues_v, q_point)
                v_grad = function_gradient(cellvalues_v, q_point, v_cell)
                v_val = function_value(cellvalues_v, q_point, v_cell)
                for j in 1:n_basefuncs
                    φⱼ = shape_value(cellvalues_v, q_point, j)
                    b_nl[v_celldofs[j]] += (v_val ⋅ v_grad) ⋅ φⱼ * dΩ
                end
            end
        end
        apply_zero!(b_nl, ch)
        println("NL: ", norm(b_nl))
        b += b_nl # COmment this line for stokes operator...
        apply_zero!(b, ch)

        update!(ch, t);
        apply!(A,b,ch)
        #apply_rhs!(rhsdata, b, ch)
        u = cg(A,b)
        #u = A\b

        if !all(isfinite.(u))
            break
        end

        vtk_grid("transient-stokes-flow-$t.vtu", dh; compress=false) do vtk
            vtk_point_data(vtk,dh,u)
            vtk_save(vtk)
            pvd[t] = vtk
        end
        u_prev .= u
        GC.gc()
    end

    vtk_save(pvd)

    # uₐ = project_solution(cellvalues_v, cellvalues_p, dh, T)
    # vtk_grid("taylor-green-final.vtu", dh; compress=false) do vtk
    #     vtk_point_data(vtk,dh,uₐ)
    #     vtk_save(vtk)
    # end
    # println("Error Norm: ", norm(u-project_solution(cellvalues_v, cellvalues_p, dh, T)))
end

function compute_divergence(dh, u, cellvalues_v)
    divv = 0.0
    @inbounds for (i,cell) in enumerate(CellIterator(dh))
        # For each cell we also need to reinitialize the cached values in `cellvalues`.
        #+
        Ferrite.reinit!(cellvalues_v, cell)

        # It is now time to loop over all the quadrature points in the cell and
        # assemble the contribution to `Ke` and `fe`. The integration weight
        # can be queried from `cellvalues` by `getdetJdV`.
        #+
        for q_point in 1:getnquadpoints(cellvalues_v)
            dΩ = getdetJdV(cellvalues_v, q_point)

            all_celldofs = celldofs(cell)
            v_celldofs = all_celldofs[dof_range(dh, :v)]
            v_cell = u[v_celldofs]

            divv += function_divergence(cellvalues_v, q_point, v_cell) * dΩ
        end
    end
    return divv
end

function solve_manufactured()
    A = copy(K)
    f = zeros(ndofs(dh))

    v▄ = 1

    n_basefuncs_v = getnbasefunctions(cellvalues_v)
    n_basefuncs_v_dim = Int(n_basefuncs_v/dim)
    n_basefuncs_p = getnbasefunctions(cellvalues_p)
    n_basefuncs = n_basefuncs_v + n_basefuncs_p

    assembler = start_assemble(A, f)
    A .= -K
    Kₑ = PseudoBlockArray(zeros(n_basefuncs, n_basefuncs), [n_basefuncs_v, n_basefuncs_p], [n_basefuncs_v, n_basefuncs_p])
    fₑ = PseudoBlockArray(zeros(n_basefuncs), [n_basefuncs_v, n_basefuncs_p])

    @inbounds for cell in CellIterator(dh)
        # Always remember to reset the element stiffness matrix and
        # force vector since we reuse them for all elements.
        #+
        fill!(Kₑ, 0)
        fill!(fₑ, 0)

        # For each cell we also need to reinitialize the cached values in `cellvalues`.
        #+
        Ferrite.reinit!(cellvalues_v, cell)
        Ferrite.reinit!(cellvalues_p, cell)

        coords = getcoordinates(cell)
        # It is now time to loop over all the quadrature points in the cell and
        # assemble the contribution to `Ke` and `fe`. The integration weight
        # can be queried from `cellvalues` by `getdetJdV`.
        #+
        for q_point in 1:getnquadpoints(cellvalues_v)
            dΩ = getdetJdV(cellvalues_v, q_point)

            (x,y) = spatial_coordinate(cellvalues_v, q_point, coords)
            for j in 1:n_basefuncs_v_dim
                    x_comp_idx = dim*(j-1) + 1
                    y_comp_idx = dim*(j-1) + 2
                    δφₓ₁ⱼ = shape_value(cellvalues_v, q_point, x_comp_idx)[1]
                    δφₓ₂ⱼ = shape_value(cellvalues_v, q_point, y_comp_idx)[2]
                    # Solution on [0,1]²
                    # vₓ₁(x,y) = -256x²(x-1)²y(y-1)(2y-1)
                    # vₓ₂(x,y) = 256y²(y-1)²x(x-1)(2x-1)
                    #   p(x,y) = 150(x-0.5)(y-0.5)
                    ∂²vₓ₁∂ₓ₁² = -256*(4*3*x^2 - 3*2*2*x + 2)*y*(y-1)*(2*y-1)
                    ∂²vₓ₁∂ₓ₂² = -256*x^2*(x-1)^2*(3*2*2*y - 3*2*1)
                    ∂²vₓ₂∂ₓ₁² = 256*(4*3*y^2 - 3*2*2*y + 2)*x*(x-1)*(2*x-1)
                    ∂²vₓ₂∂ₓ₂² = 256*y^2*(y-1)^2*(3*2*2*x - 3*2*1)
                    ∂p∂ₓ₁ = 150*(y-0.5)
                    ∂p∂ₓ₂ = 150*(x-0.5)
                    fₑ[BlockIndex((v▄), (x_comp_idx))] += (-ν * (∂²vₓ₁∂ₓ₁² + ∂²vₓ₁∂ₓ₂²) + ∂p∂ₓ₁ ) * δφₓ₁ⱼ * dΩ
                    fₑ[BlockIndex((v▄), (y_comp_idx))] += (-ν * (∂²vₓ₂∂ₓ₁² + ∂²vₓ₂∂ₓ₂²) + ∂p∂ₓ₂ ) * δφₓ₂ⱼ * dΩ
            end
        end

        assemble!(assembler, celldofs(cell), fₑ, Kₑ)
    end

    update!(ch, T)
    apply!(A,f,ch)
    #u = A\f
    u = cg(A,f)
    vtk_grid("analytic-steady-stokes-flow.vtu", dh; compress=false) do vtk
        vtk_point_data(vtk,dh,u)
        vtk_save(vtk)
    end

    vL₂err² = 0.0
    pL₂err² = 0.0
    @inbounds for cell in CellIterator(dh)
        Ferrite.reinit!(cellvalues_v, cell)
        Ferrite.reinit!(cellvalues_p, cell)

        all_celldofs = celldofs(cell)
        v_celldofs = all_celldofs[dof_range(dh, :v)]
        p_celldofs = all_celldofs[dof_range(dh, :p)]
        v_cell = u[v_celldofs]
        p_cell = u[p_celldofs]

        coords = getcoordinates(cell)
        # It is now time to loop over all the quadrature points in the cell and
        # assemble the contribution to `Ke` and `fe`. The integration weight
        # can be queried from `cellvalues` by `getdetJdV`.
        #+
        for q_point in 1:getnquadpoints(cellvalues_v)
            dΩ = getdetJdV(cellvalues_v, q_point)

            (x,y) = spatial_coordinate(cellvalues_v, q_point, coords)
            v = function_value(cellvalues_v, q_point, v_cell)
            p = function_value(cellvalues_p, q_point, p_cell)

            vL₂err² += (v[1] - vₐₙₐ(x,y,T)[1])^2 * dΩ
            vL₂err² += (v[2] - vₐₙₐ(x,y,T)[2])^2 * dΩ
            pL₂err² += (p - pₐₙₐ(x,y,T))^2 * dΩ
        end
    end

    println("L₂ error v: ", √vL₂err²)
    println("L₂ error p: ", √pL₂err²)
end

function pressure_correction!(u, dh, cellvalues_p)
    p_int = 0.0
    for cell in CellIterator(dh)
        Ferrite.reinit!(cellvalues_p, cell)
        all_celldofs = celldofs(cell)
        p_celldofs = all_celldofs[dof_range(dh, :p)]
        p_cell = u[p_celldofs]
        for q_point in 1:getnquadpoints(cellvalues_p)
            dΩ = getdetJdV(cellvalues_p, q_point)
            p = function_value(cellvalues_p, q_point, p_cell)
            p_int += p * dΩ
        end
    end

    pressure_dofs = []

    for cell in CellIterator(dh)
        Ferrite.reinit!(cellvalues_p, cell)
        all_celldofs = celldofs(cell)
        p_celldofs = all_celldofs[dof_range(dh, :p)]
        append!(pressure_dofs, p_celldofs)
        # for q_point in 1:getnquadpoints(cellvalues_p)
        #     dΩ = getdetJdV(cellvalues_p, q_point)
        #     for j in 1:getnbasefunctions(cellvalues_p)
        #         u[p_celldofs[j]] -= p_int * dΩ
        #     end
        # end
            # for j in 1:getnbasefunctions(cellvalues_p)
            #     u[p_celldofs[j]] -= p_int
            # end
    end
    sort!(pressure_dofs)
    unique!(pressure_dofs)

    for dof ∈ pressure_dofs
        u[dof] -= p_int
    end

    return p_int
end

function navier_stokes_residuals(dh, u, cellvalues_v, cellvalues_p)
    r = zeros(ndofs(dh))
    for cell in CellIterator(dh)
        Ferrite.reinit!(cellvalues_v, cell)
        Ferrite.reinit!(cellvalues_p, cell)
        all_celldofs = celldofs(cell)
        v_celldofs = all_celldofs[dof_range(dh, :v)]
        p_celldofs = all_celldofs[dof_range(dh, :p)]
        v_cell = u[v_celldofs]
        p_cell = u[p_celldofs]
        for q_point in 1:getnquadpoints(cellvalues_v)
            dΩ = getdetJdV(cellvalues_v, q_point)
            divv = function_divergence(cellvalues_v, q_point, v_cell)
            ∇v = function_gradient(cellvalues_v, q_point, v_cell)
            v = function_value(cellvalues_v, q_point, v_cell)
            p = function_value(cellvalues_p, q_point, p_cell)
            for j in 1:getnbasefunctions(cellvalues_v)
                ∇φⱼ = shape_gradient(cellvalues_v, q_point, j)
                divφⱼ = shape_divergence(cellvalues_v, q_point, j)
                φⱼ = shape_value(cellvalues_v, q_point, j)
                r[v_celldofs[j]] -= ν * ∇v ⊡ ∇φⱼ * dΩ
                r[v_celldofs[j]] -= (v ⋅ ∇v) ⋅ φⱼ * dΩ
                r[v_celldofs[j]] += p * divφⱼ * dΩ
            end
            for j in 1:getnbasefunctions(cellvalues_p)
                ψⱼ = shape_value(cellvalues_p, q_point, j)
                r[p_celldofs[j]] += divv * ψⱼ * dΩ
            end
        end
    end
    return r
end

function solve_steady_stokes()
    A = copy(K)
    f = zeros(ndofs(dh))
    update!(ch, T)
    apply!(A,f,ch)
    u = A\f
    println("Res1 ", norm(navier_stokes_residuals(dh, u, cellvalues_v, cellvalues_p)[ch.free_dofs]))
    pressure_correction!(u, dh, cellvalues_p)
    println("Res2 ", norm(navier_stokes_residuals(dh, u, cellvalues_v, cellvalues_p)[ch.free_dofs]))
    # u = gmres(A,f)
    # println("Res3 ", norm(navier_stokes_residuals(dh, u, cellvalues_v, cellvalues_p)[ch.free_dofs]))
    vtk_grid("steady-stokes-flow.vtu", dh; compress=false) do vtk
        vtk_point_data(vtk,dh,u)
        vtk_save(vtk)
    end

    #f = zeros(ndofs(dh))
    ## Nonlinear contribution
    for cell in CellIterator(dh)
        Ferrite.reinit!(cellvalues_v, cell)
        ## Trilinear form evaluation
        all_celldofs = celldofs(cell)
        v_celldofs = all_celldofs[dof_range(dh, :v)]
        v_cell = u[v_celldofs]
        for q_point in 1:getnquadpoints(cellvalues_v)
            dΩ = getdetJdV(cellvalues_v, q_point)
            v = function_value(cellvalues_v, q_point, v_cell)
            ∇v = function_gradient(cellvalues_v, q_point, v_cell)
            for j in 1:getnbasefunctions(cellvalues_v)
                φⱼ = shape_value(cellvalues_v, q_point, j)
                # for i in 1:getnbasefunctions(cellvalues_v)
                #     φᵢ = shape_value(cellvalues_v, q_point, i)
                #     ∇φᵢ = shape_gradient(cellvalues_v, q_point, i)
                #     f[v_celldofs[j]] += ((u[v_celldofs[i]]*φᵢ) ⋅ (u[v_celldofs[i]]*∇φᵢ)) ⋅ φⱼ * dΩ
                # end
                f[v_celldofs[j]] += v ⋅ ∇v ⋅ φⱼ * dΩ
            end
        end
    end
    vtk_grid("trilinearform.vtu", dh; compress=false) do vtk
        vtk_point_data(vtk,dh,f)
        vtk_save(vtk)
    end
    # return

    u = zeros(ndofs(dh))
    uᶞ = zeros(ndofs(dh))
    n_newton = 20
    newton_itr = -1
    NEWTON_TOL = 1e-8

    n_basefuncs_v = getnbasefunctions(cellvalues_v)
    n_basefuncs_v_dim = Int(n_basefuncs_v/dim)
    n_basefuncs_p = getnbasefunctions(cellvalues_p)
    n_basefuncs = n_basefuncs_v + n_basefuncs_p

    Jₑ = PseudoBlockArray(zeros(n_basefuncs, n_basefuncs), [n_basefuncs_v, n_basefuncs_p], [n_basefuncs_v, n_basefuncs_p])
    Rₑ = PseudoBlockArray(zeros(n_basefuncs), [n_basefuncs_v, n_basefuncs_p])

    v▄, p▄ = 1, 2

    prog = ProgressMeter.ProgressThresh(NEWTON_TOL, "Solving:")
    J = copy(K)
    # apply!(u, ch)
    R = navier_stokes_residuals(dh, u, cellvalues_v, cellvalues_p)
    last_res = norm(R[ch.free_dofs])
    R = zeros(ndofs(dh))
    α = 1.0
    # Newton line search formulation: uᵢ₊₁ = uᵢ - α J⁻¹ R(uᵢ)
    # We solve for J uᶞᵢ = R(uᵢ) where J is the Gateux derivative of R
    while true
        newton_itr += 1

        # Next we define the global force vector `f` and use that and
        # the stiffness matrix `K` and create an assembler. The assembler
        # is just a thin wrapper around `f` and `K` and some extra storage
        # to make the assembling faster.
        #+
        # f = zeros(ndofs(dh))
        jac_assembler = start_assemble(J, R)
        # J .= K
        # R .= K*u
        # R .= navier_stokes_residuals(dh, u, cellvalues_v, cellvalues_p)

        # It is now time to loop over all the cells in our grid. We do this by iterating
        # over a `CellIterator`. The iterator caches some useful things for us, for example
        # the nodal coordinates for the cell, and the local degrees of freedom.
        #+
        @inbounds for cell in CellIterator(dh)
            # Always remember to reset the element stiffness matrix and
            # force vector since we reuse them for all elements.
            #+
            fill!(Jₑ, 0.0)
            fill!(Rₑ, 0.0)

            # For each cell we also need to reinitialize the cached values in `cellvalues`.
            #+
            Ferrite.reinit!(cellvalues_v, cell)
            Ferrite.reinit!(cellvalues_p, cell)

            all_celldofs = celldofs(cell)
            v_celldofs = all_celldofs[dof_range(dh, :v)]
            p_celldofs = all_celldofs[dof_range(dh, :p)]
            v_cell = u[v_celldofs]
            p_cell = u[p_celldofs]

            # It is now time to loop over all the quadrature points in the cell and
            # assemble the contribution to `Ke` and `fe`. The integration weight
            # can be queried from `cellvalues` by `getdetJdV`.
            #+
            for q_point in 1:getnquadpoints(cellvalues_v)
                dΩ = getdetJdV(cellvalues_v, q_point)

                # For each quadrature point we loop over all the (local) shape functions.
                # We need the value and gradient of the testfunction `v` and also the gradient
                # of the trial function `u`. We get all of these from `cellvalues`.
                #+
                #Viscosity term in J - done in K
                #Pressure term in J - done in K
                #Incompressibility term in J - done in K
                ## Trilinear form evaluations
                ∇v = function_gradient(cellvalues_v, q_point, v_cell)
                v = function_value(cellvalues_v, q_point, v_cell)
                divv = function_divergence(cellvalues_v, q_point, v_cell)
                p = function_value(cellvalues_p, q_point, p_cell)
                for i ∈ 1:n_basefuncs_v
                    φᵢ = shape_value(cellvalues_v, q_point, i)
                    ∇φᵢ = shape_gradient(cellvalues_v, q_point, i)
                    divφᵢ = shape_divergence(cellvalues_v, q_point, i)
                    for j ∈ 1:n_basefuncs_v
                        φⱼ = shape_value(cellvalues_v, q_point, j)
                        ∇φⱼ = shape_gradient(cellvalues_v, q_point, j)
                        divφⱼ = shape_divergence(cellvalues_v, q_point, j)

                        Jₑ[BlockIndex((v▄, v▄), (i, j))] += ν * ∇φⱼ ⊡ ∇φᵢ * dΩ

                        Jₑ[BlockIndex((v▄, v▄), (i, j))] += ∇v ⋅ φⱼ ⋅ φᵢ * dΩ
                        Jₑ[BlockIndex((v▄, v▄), (i, j))] += ∇φⱼ ⋅ v ⋅ φᵢ * dΩ
                        # Jₑ[BlockIndex((v▄, v▄), (i, j))] += φⱼ ⋅ ∇v ⋅ φᵢ * dΩ
                        # Jₑ[BlockIndex((v▄, v▄), (i, j))] += v ⋅ ∇φⱼ ⋅ φᵢ * dΩ

                        # Jₑ[BlockIndex((v▄, v▄), (i, j))] += γ * divφⱼ * divφᵢ * dΩ #CURRENT

                        # Jₑ[BlockIndex((v▄, v▄), (j, i))] -= (v ⋅ ∇φᵢ + φᵢ ⋅ ∇v) ⋅ φⱼ * dΩ
                        ## Stablizitaion term
                        # Jₑ[BlockIndex((v▄, v▄), (j, i))] -= γ * divφᵢ * divφⱼ * dΩ
                    end
                    for j ∈ 1:n_basefuncs_p
                        ψⱼ = shape_value(cellvalues_p, q_point, j)
                        Jₑ[BlockIndex((p▄, v▄), (j, i))] -= ψⱼ * divφᵢ * dΩ
                        Jₑ[BlockIndex((v▄, p▄), (i, j))] -= divφᵢ * ψⱼ * dΩ
                    end
                    # Rₑ[BlockIndex((v▄), (j))] -= v ⋅ ∇v ⋅ φᵢ * dΩ
                    ## Stablizitaion term
                    # Rₑ[BlockIndex((v▄), (j))] -= γ * divv ⋅ divφⱼ * dΩ
                    # for i in 1:n_basefuncs_v_dim
                    #     for d in 1:dim
                    #         comp_idx = dim*(i-1) + d
                    #         #comp_idx = n_basefuncs_v_dim*(d-1) + i
                    #         ∂φ∂xd = cellvalues_v.dNdx[comp_idx,q_point][d,d]
                    #         Rₑ[BlockIndex((v▄), (j))] += (ψ * ∂φ∂xd) * dΩ
                    #         Rₑ[BlockIndex((p▄), (j))] += (ψ * ∂φ∂xd) * dΩ
                    #     end
                    # end
                    # Rₑ[BlockIndex((v▄), (j))] -= ν * ∇v ⊡ ∇φⱼ * dΩ
                    # Rₑ[BlockIndex((v▄), (j))] -= v ⋅ ∇v ⋅ φⱼ * dΩ
                    # Rₑ[BlockIndex((v▄), (j))] += p * divφⱼ * dΩ

                    Rₑ[BlockIndex((v▄), (i))] -= ν * ∇v ⊡ ∇φᵢ * dΩ
                    Rₑ[BlockIndex((v▄), (i))] -= ∇v ⋅ v ⋅ φᵢ * dΩ
                    # Rₑ[BlockIndex((v▄), (i))] -= v ⋅ ∇v ⋅ φᵢ * dΩ
                    Rₑ[BlockIndex((v▄), (i))] += p * divφᵢ * dΩ

                    # Rₑ[BlockIndex((v▄), (i))] -= γ * divv * divφᵢ * dΩ #CURRENT
                end
                for i in 1:n_basefuncs_p
                    ψᵢ = shape_value(cellvalues_p, q_point, i)
                    divψᵢ = shape_divergence(cellvalues_p, q_point, i)
                    ## Stablizitaion Term
                    #     Rₑ[BlockIndex((p▄), (j))] += divv * ψⱼ * dΩ
                    for j in 1:n_basefuncs_p
                        ψⱼ = shape_value(cellvalues_p, q_point, j)
                        ## Stablizitaion Term
                        #Jₑ[BlockIndex((v▄, v▄), (j, i))] -= (1.0/γ) *  ψⱼ * ψᵢ * dΩ
                        ## ???
                        # Jₑ[BlockIndex((p▄, p▄), (j, i))] -= ψᵢ * ψⱼ * dΩ
                        # Jₑ[BlockIndex((p▄, p▄), (i, j))] += ψᵢ * ψⱼ * dΩ #CURRENT
                    end
                    Rₑ[BlockIndex((p▄), (i))] += divv * ψᵢ * dΩ
                end
            end

            # The last step in the element loop is to assemble `Ke` and `fe`
            # into the global `K` and `f` with `assemble!`.
            #+
            assemble!(jac_assembler, all_celldofs, Rₑ, Jₑ)
        end
        #R[ch.prescribed_dofs] .= 0.0
        # vtk_grid("steady-navier-stokes-flow-$newton_itr-residual-raw.vtu", dh; compress=false) do vtk
        #     vtk_point_data(vtk,dh,R)
        #     vtk_save(vtk)
        # end
        if newton_itr == 0
            apply!(J,R,ch)
        else
            apply_zero!(J, R, ch)
        end
        norm_res = norm(R[ch.free_dofs])
        ProgressMeter.update!(prog, norm_res; showvalues = [(:iter, newton_itr), (:divv, compute_divergence(dh, u, cellvalues_v)), (:delta, norm(uᶞ)), (:line, α)])
        if norm_res < NEWTON_TOL
            println("Reached tolerance of $norm_res after $newton_itr Newton iterations, aborting")
            break
        elseif newton_itr > n_newton
            println("Reached maximum Newton iterations, aborting")
            break
        end

        uᶞ = J\R
        # uᶞ = gmres(J,R)
        if newton_itr > 0
            apply_zero!(uᶞ, ch)
        else
            apply!(uᶞ, ch)
        end
        # if newton_itr == 0
            u .+= uᶞ
        # else # line search
        #     α = 2.0
        #     uₛ = copy(u)
        #     while α > 1e-5
        #         α /= 2.0
        #         uₛ = u + α*uᶞ
        #         norm_res = norm(navier_stokes_residuals(dh, uₛ, cellvalues_v, cellvalues_p)[ch.free_dofs])
        #         if norm_res < last_res
        #             break
        #         end
        #     end
        #     last_res = norm_res
        #     u .= uₛ
        # end
        pressure_correction!(u, dh, cellvalues_p)

        # vtk_grid("steady-navier-stokes-flow-$newton_itr-residual.vtu", dh; compress=false) do vtk
        #     vtk_point_data(vtk,dh,R)
        #     vtk_save(vtk)
        # end

        # vtk_grid("steady-navier-stokes-flow-$newton_itr-delta.vtu", dh; compress=false) do vtk
        #     vtk_point_data(vtk,dh,uᶞ)
        #     vtk_save(vtk)
        # end

        vtk_grid("steady-navier-stokes-flow-$newton_itr.vtu", dh; compress=false) do vtk
            vtk_point_data(vtk,dh,u)
            vtk_save(vtk)
        end
    end



    # ## Fixed Point Iteration
    # FP_TOL = 1e-9
    # prog = ProgressMeter.ProgressThresh(FP_TOL, "Solving:")
    # fp_iter = -1
    # u_prev = copy(u)
    # while true;
    #     fp_iter += 1

    #     u_prev .= u

    #     A = copy(K)
    #     f = zeros(ndofs(dh))

    #     ## Nonlinear contribution
    #     for cell in CellIterator(dh)
    #         Ferrite.reinit!(cellvalues_v, cell)
    #         ## Trilinear form evaluation
    #         all_celldofs = celldofs(cell)
    #         v_celldofs = all_celldofs[dof_range(dh, :v)]
    #         v_cell = u_prev[v_celldofs]
    #         for q_point in 1:getnquadpoints(cellvalues_v)
    #             dΩ = getdetJdV(cellvalues_v, q_point)
    #             ∇v = function_gradient(cellvalues_v, q_point, v_cell)
    #             v = function_value(cellvalues_v, q_point, v_cell)
    #             # for j in 1:getnbasefunctions(cellvalues_v)
    #             #     φⱼ = shape_value(cellvalues_v, q_point, j)
    #             #     f[v_celldofs[j]] += (v ⋅ ∇v) ⋅ φⱼ * dΩ
    #             # end
    #             for j in 1:getnbasefunctions(cellvalues_v)
    #                 φⱼ = shape_value(cellvalues_v, q_point, j)
    #                 for i in 1:getnbasefunctions(cellvalues_v)
    #                     φᵢ = shape_value(cellvalues_v, q_point, i)
    #                     ∇φᵢ = shape_gradient(cellvalues_v, q_point, i)
    #                     f[v_celldofs[j]] += ((v_cell[i]*φᵢ) ⋅ (v_cell[i]*∇φᵢ)) ⋅ φⱼ * dΩ
    #                 end
    #             end
    #         end
    #     end

    #     for dof in ch.prescribed_dofs
    #         f[dof] = 0.0
    #     end

    #     apply!(A,f,ch)
    #     u = A\f
    #     pressure_correction!(u, dh, cellvalues_p)
    #     #gmres!(u,A,f; abstol=1e-6, reltol=1e-6)

    #     norm_rel = norm(u_prev - u)
    #     res = norm(navier_stokes_residuals(dh, u, cellvalues_v, cellvalues_p)[ch.free_dofs])
    #     ProgressMeter.update!(prog, norm_rel; showvalues = [(:iter, fp_iter), (:divv, compute_divergence(dh, u, cellvalues_v)), (:res, res)])

    #     vtk_grid("steady-navier-stokes-flow-$fp_iter.vtu", dh; compress=false) do vtk
    #         vtk_point_data(vtk,dh,u)
    #         vtk_save(vtk)
    #     end

    #     if norm_rel < FP_TOL
    #         println("FPI converged with $norm_rel.")
    #         break;
    #     end
    # end

    # ## Newton from Verfürth's numerics III script
    # NEWTON_TOL = 1e-9
    # prog = ProgressMeter.ProgressThresh(NEWTON_TOL, "Solving:")
    # newton_iter = -1
    # u_prev = copy(u)
    # while true;
    #     newton_iter += 1

    #     u_prev .= u

    #     A = copy(K)
    #     f = zeros(ndofs(dh))

    #     ## Nonlinear contributions
    #     for cell in CellIterator(dh)
    #         Ferrite.reinit!(cellvalues_v, cell)
    #         all_celldofs = celldofs(cell)
    #         v_celldofs = all_celldofs[dof_range(dh, :v)]
    #         v_cell = u_prev[v_celldofs]
    #         for q_point in 1:getnquadpoints(cellvalues_v)
    #             dΩ = getdetJdV(cellvalues_v, q_point)
    #             ∇v = function_gradient(cellvalues_v, q_point, v_cell)
    #             v = function_value(cellvalues_v, q_point, v_cell)
    #             for j in 1:getnbasefunctions(cellvalues_v)
    #                 φⱼ = shape_value(cellvalues_v, q_point, j)
    #                 for i in 1:getnbasefunctions(cellvalues_v)
    #                     φᵢ = shape_value(cellvalues_v, q_point, i)
    #                     ∇φᵢ = shape_gradient(cellvalues_v, q_point, i)
    #                     A[v_celldofs[j],v_celldofs[i]] -= φᵢ ⋅ ∇v ⋅ φⱼ * dΩ
    #                     A[v_celldofs[j],v_celldofs[i]] -= v ⋅ ∇φᵢ ⋅ φⱼ * dΩ
    #                     f[v_celldofs[j]] -= v ⋅ ∇v ⋅ φⱼ * dΩ
    #                 end
    #             end
    #         end
    #     end

    #     apply!(A,f,ch)
    #     u = A\f
    #     pressure_correction!(u, dh, cellvalues_p)
    #     #gmres!(u,A,f; abstol=1e-6, reltol=1e-6)

    #     u_diff = u_prev - u
    #     vtk_grid("steady-navier-stokes-flow-$newton_iter-diff.vtu", dh; compress=false) do vtk
    #         vtk_point_data(vtk,dh,u_diff)
    #         vtk_save(vtk)
    #     end
    #     norm_rel = norm(u_prev - u)
    #     res = norm(navier_stokes_residuals(dh, u, cellvalues_v, cellvalues_p)[ch.free_dofs])
    #     ProgressMeter.update!(prog, norm_rel; showvalues = [(:iter, newton_iter), (:divv, compute_divergence(dh, u, cellvalues_v)), (:res, res)])

    #     vtk_grid("steady-navier-stokes-flow-$newton_iter.vtu", dh; compress=false) do vtk
    #         vtk_point_data(vtk,dh,u)
    #         vtk_save(vtk)
    #     end

    #     if norm_rel < NEWTON_TOL
    #         println("Newton converged with residual $res.")
    #         break;
    #     end
    # end
end
# solve_stokes()
throw EOFError()

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

    n_basefuncs = getnbasefunctions(cellvalues)

    ## Nonlinear contribution
    for cell in CellIterator(dh)
        Ferrite.reinit!(cellvalues_v, cell)
        ## Trilinear form evaluation
        all_celldofs = celldofs(cell)
        v_celldofs = all_celldofs[dof_range(dh, :v)]
        v_cell = u_prev[v_celldofs]
        for q_point in 1:getnquadpoints(cellvalues)
            dΩ = getdetJdV(cellvalues, q_point)
            #v_div = function_divergence(cellvalues, q_point, v_cell)
            v_grad = function_gradient(cellvalues, q_point, v_cell)
            v_val = function_value(cellvalues, q_point, v_cell)
            nl_contrib = v_val ⋅ v_grad
            for j in 1:n_basefuncs
                φⱼ = shape_value(cellvalues, q_point, j)
                du[v_celldofs[j]] -= nl_contrib ⋅ φⱼ * dΩ
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
sol = solve(problem, ImplicitEuler(linsolve=FerriteLinSolve(ch)), adaptive=false, progress=true, abstol=1e-3, reltol=1e-3, progress_steps=1, dt=Δt₀, saveat=Δt_save, initializealg=NoInit());

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