# Putting this flag to false reproduces the figure shown in the example #src
# We check for laminar flow development in the CI                       #src
if @isdefined is_ci    #hide
    IS_CI = is_ci      #hide
else                   #hide
    IS_CI = false      #hide
end                    #hide
# # [Reactive Surface](@id tutorial-reactive-surface)
#
# ![](reactive_surface.gif)
#
# *Figure 1*: Reactant concentration field of the Gray-Scott model on the unit sphere.
#
#-
#md # !!! tip
#md #     This example is also available as a Jupyter notebook:
#md #     [`reactive_surface.ipynb`](@__NBVIEWER_ROOT_URL__/examples/reactive_surface.ipynb).
#-
#
# ## Introduction
#
# This tutorial gives a quick tutorial on how to assemble and solve time-dependent problems
# on embedded surfaces.
#
# For this showcase we use the well known Gray-Scott model, which is a well-known reaction-diffusion
# system to study pattern formation. The strong form is given by
#
# ```math
#  \begin{aligned}
#    \partial_t r_1 &= \nabla \cdot (D_1 \nabla r_1) - r_1*r_2^2 + F *(1 - r_1) \quad \textbf{x} \in \Omega, \\
#    \partial_t r_2 &= \nabla \cdot (D_2 \nabla r_2) + r_1*r_2^2 - r_2*(F + k ) \quad \textbf{x} \in \Omega,
#  \end{aligned}
# ```
#
# where $r_1$ and $r_2$ are the reaction fields, $D_1$ and $D_2$ the diffusion tensors,
# $k$ is the conversion rate, $F$ is the feed rate and $\Omega$ the domain. Depending on the choice of
# parameters a different pattern can be observed. Please also note that the domain does not have a 
# boundary. The corresponding weak form can be derived as usual.
#
# For simplicity we will solve the problem with the Lie-Troter-Godunov operator splitting technique with
# the classical reaction-diffusion split. In this method we split our problem in two problems, i.e. a heat
# problem and a pointwise reaction problem, and solve them alternatingly to advance in time.
#
#-
# ## Commented Program
#
# Now we solve the problem in Ferrite. What follows is a program spliced with comments.
#md # The full program, without comments, can be found in the next [section](@ref reactive_surface-plain-program).
#
# First we load Ferrite, and some other packages we need

using Ferrite, FerriteGmsh
using BlockArrays, SparseArrays, LinearAlgebra

# ### Assembly routines
# The following assembly routines are similar to these found in previous tutorials.

function assemble_element_mass!(Me::Matrix, cellvalues::CellValues)
    n_basefuncs = getnbasefunctions(cellvalues)
    ## Reset to 0
    fill!(Me, 0)
    ## Loop over quadrature points
    for q_point in 1:getnquadpoints(cellvalues)
        ## Get the quadrature weight
        dΩ = getdetJdV(cellvalues, q_point)
        ## Loop over test shape functions
        for i in 1:n_basefuncs
            δuᵢ = shape_value(cellvalues, q_point, i)
            ## Loop over trial shape functions
            for j in 1:n_basefuncs
                δuⱼ = shape_value(cellvalues, q_point, j)
                ## Add contribution to Ke
                Me[2*i-1, 2*j-1] += (δuᵢ * δuⱼ) * dΩ
                Me[2*i  , 2*j  ] += (δuᵢ * δuⱼ) * dΩ
            end
        end
    end
    return nothing
end

function assemble_element_diffusion!(De::Matrix, cellvalues::CellValues)
    n_basefuncs = getnbasefunctions(cellvalues)
    ## Reset to 0
    fill!(De, 0)
    ## Loop over quadrature points
    for q_point in 1:getnquadpoints(cellvalues)
        ## Get the quadrature weight
        dΩ = getdetJdV(cellvalues, q_point)
        ## Loop over test shape functions
        for i in 1:n_basefuncs
            ∇δuᵢ = shape_gradient(cellvalues, q_point, i)
            ## Loop over trial shape functions
            for j in 1:n_basefuncs
                ∇δuⱼ = shape_gradient(cellvalues, q_point, j)
                ## Add contribution to Ke
                De[2*i-1, 2*j-1] += 2*0.00008 * (∇δuᵢ ⋅ ∇δuⱼ) * dΩ
                De[2*i  , 2*j  ] += 2*0.00004 * (∇δuᵢ ⋅ ∇δuⱼ) * dΩ
            end
        end
    end
    return nothing
end

function assemble_matrices!(M::SparseMatrixCSC, D::SparseMatrixCSC, cellvalues::CellValues, dh::DofHandler)
    n_basefuncs = getnbasefunctions(cellvalues)

    ## Allocate the element stiffness matrix and element force vector
    Me = zeros(2*n_basefuncs, 2*n_basefuncs)
    De = zeros(2*n_basefuncs, 2*n_basefuncs)

    ## Create an assembler
    M_assembler = start_assemble(M)
    D_assembler = start_assemble(D)
    ## Loop over all cels
    for cell in CellIterator(dh)
        ## Reinitialize cellvalues for this cell
        reinit!(cellvalues, cell)
        ## Compute element contribution
        assemble_element_mass!(Me, cellvalues)
        assemble!(M_assembler, celldofs(cell), Me)

        assemble_element_diffusion!(De, cellvalues)
        assemble!(D_assembler, celldofs(cell), De)
    end
    return nothing
end

# ### Initial condition setup
# Time-dependent problems always need an initial condition from which the time evolution starts.
# In this tutorial we set the concentration of reactant 1 to $1$ and the concentration of reactant
# 2 to $0$ for all nodal dof with associated coordinate $z \leq 0.9$ on the sphere. Since the
# simulation would be pretty boring with a steady-state initial condition, we introduce some
# heterogeneity by setting the dofs associated to top part of the sphere (i.e. dofs with $z > 0.9$
# to store the reactant concentrations of $0.5$ and $0.25$ for the reactants 1 and 2 respectively.
function setup_initial_conditions!(u₀::Vector, cellvalues::CellValues, dh::DofHandler)
    u₀ .= ones(ndofs(dh))
    u₀[2:2:end] .= 0.0

    n_basefuncs = getnbasefunctions(cellvalues)

    for cell in CellIterator(dh)
        reinit!(cellvalues, cell)

        coords = getcoordinates(cell)
        dofs = celldofs(cell)
        uₑ = @view u₀[dofs]
        rv₀ₑ = reshape(uₑ, (2, n_basefuncs))

        for i in 1:n_basefuncs            
            if coords[i][3] > 0.9
                rv₀ₑ[1, i] = 0.5
                rv₀ₑ[2, i] = 0.25
            end
        end
    end

    u₀ .+= 0.01*rand(ndofs(dh))
end

# ### Simulation routines
# Now we define a function to setup and solve the problem with given feed and conversion rates
# $F$ and $k$, as well as the time step length and for how long we want to solve the model.
function gray_scott_sphere(F, k, Δt, T, refinements)
    ## We start by setting up grid, dof handler and the matrices for the heat problem.
    gmsh.initialize()

    ## Add a unit sphere
    gmsh.model.occ.addSphere(0.0,0.0,0.0,1.0)
    gmsh.model.occ.synchronize()

    gmsh.model.mesh.renumberNodes()
    gmsh.model.mesh.renumberElements()

    ## Generate surface elements and refine several times
    gmsh.model.mesh.generate(2)
    for _ in 1:refinements
        gmsh.model.mesh.refine()
    end

    ## Create a grid out of it
    nodes = tonodes()
    elements, _ = toelements(2)
    gmsh.finalize()
    grid = Grid(elements, nodes);

    ip = Lagrange{RefTriangle, 1}()
    ip_geo = Lagrange{RefTriangle, 1}()
    qr = QuadratureRule{RefTriangle}(2)
    cellvalues = CellValues(qr, ip, ip_geo^3);

    dh = DofHandler(grid);
    add!(dh, :reactants, ip^2);
    close!(dh);

    M = create_sparsity_pattern(dh; coupling=[true false;false true])
    D = create_sparsity_pattern(dh; coupling=[true false;false true])
    assemble_matrices!(M, D, cellvalues, dh);

    ## Since the heat problem is linear and has no time dependent parameters, we precompute the
    ## decomposition of the system matrix to speed up the linear system solver.
    A = M + Δt .* D
    Alu = cholesky(A)

    ## Now we setup buffers for the time dependent solution and fill the initial condition.
    uₜ   = zeros(ndofs(dh))
    uₜ₋₁ = ones(ndofs(dh))
    setup_initial_conditions!(uₜ₋₁, cellvalues, dh)

    ## And prepare output for visualization.
    pvd = VTKFileCollection("reactive-surface.pvd", dh);
    addstep!(pvd, 0.0) do io
        write_solution(io, dh, uₜ₋₁)
    end

    ## This is now the main solve loop.
    for (iₜ, t) ∈ enumerate(Δt:Δt:T)
        ## First we solve the heat problem
        uₜ .= Alu \ (M * uₜ₋₁)

        ## Then we solve the point-wise reaction problem with the solution of
        ## the heat problem as initial guess.
        rvₜ = reshape(uₜ, (2, length(grid.nodes)))
        for i ∈ 1:length(grid.nodes)
            r₁ = rvₜ[1, i]
            r₂ = rvₜ[2, i]
            rvₜ[1, i] += Δt*( -r₁*r₂^2 + F *(1 - r₁) )
            rvₜ[2, i] += Δt*(  r₁*r₂^2 - r₂*(F + k ) )
        end

        ## The solution is then stored every 10th step to vtk files for
        ## later visualization purposes.
        if (iₜ % 10) == 0
            addstep!(pvd, t) do io
                write_solution(io, dh, uₜ₋₁)
            end
        end

        ## Finally we totate the solution to initialize the next timestep.
        uₜ₋₁ .= uₜ
    end

    close(pvd);
end

## This parametrization gives the spot pattern shown in the gif above.
if !IS_CI #src
gray_scott_sphere(0.06, 0.062, 10.0, 32000.0, 3)
else #src
gray_scott_sphere(0.06, 0.062, 10.0, 20.0, 0) #src
end #src

#md # ## [Plain program](@id reactive_surface-plain-program)
#md #
#md # Here follows a version of the program without any comments.
#md # The file is also available here: [`reactive_surface.jl`](reactive_surface.jl).
#md #
#md # ```julia
#md # @__CODE__
#md # ```
