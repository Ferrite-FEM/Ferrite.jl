# # [Discontinuous Galerkin Heat equation](@id tutorial-dg-heat-equation)
#
# ![](dg_heat_equation.png)
#
# *Figure 1*: Temperature field on the unit square with an internal uniform heat source
# solved with inhomogeneous Dirichlet boundary conditions on the left and right boundaries and flux on the top and bottom boundaries.
#
#-
#md # !!! tip
#md #     This example is also available as a Jupyter notebook:
#md #     [`dg_heat_equation.ipynb`](@__NBVIEWER_ROOT_URL__/examples/dg_heat_equation.ipynb).
#-
#
# ## Introduction
#
# The heat equation is the "Hello, world!" equation of finite elements.
# Here we solve the equation on a unit square with a uniform internal source, using the interior penalty formulation.
# The strong form of the (linear) heat equation is given by
#
# ```math
#  -\nabla \cdot (k \nabla u) = f  \quad \textbf{x} \in \Omega,
# ```
#
# where $u$ is the unknown temperature field, $k$ the heat conductivity,
# $f$ the heat source and $\Omega$ the domain. For simplicity we set $f = 1$
# and $k = 1$. We will consider inhomogeneous Dirichlet boundary conditions such that
# ```math
# u(\textbf{x}) = 1 \quad \textbf{x} \in \partial \Omega_u^+, \\
# u(\textbf{x}) = -1 \quad \textbf{x} \in \partial \Omega_u^-,
# ```
# and Neumann boundary conditions such that
# ```math
# \nabla u(\textbf{x}) \cdot \vec{n} = 1 \quad \textbf{x} \in \partial \Omega_n^+, \\
# \nabla u(\textbf{x}) \cdot \vec{n} = -1 \quad \textbf{x} \in \partial \Omega_n^-,
# ```
# where $\partial \Omega$ denotes the boundaries of $\Omega$ characterized by their normals directions
# as the following:
#
# | Boundary              | Normal direction |
# |-----------------------|------------------|
# | $\partial \Omega_u^+$ | (1 , 0)          |
# | $\partial \Omega_u^-$ | (-1 , 0)         |
# | $\partial \Omega_n^+$ | (0 , 1)          |
# | $\partial \Omega_n^-$ | (0 , -1)         |
# Defining $\sigma$ as the gradient of the temperature field the equation can be expressed as
# ```math
#  \sigma = \nabla u,\\
#  -\nabla \cdot \sigma = 1,
# ```
# Multiplying by test functions $ \tau $ and $ \delta u $ respectively and integrating
# over the domain,
# ```math
#  \int_\Omega \sigma \cdot \tau dΩ = \int_\Omega \nabla u \cdot \tau dΩ,\\
#  -\int_\Omega \nabla \cdot \sigma \delta u dΩ = \int_\Omega \delta u dΩ,
# ```
# Integratig by parts and applying divergence theorem,
# ```math
#  \int_\Omega \sigma \cdot \tau dΩ = \int_\Omega u \nabla \cdot \tau dΩ + \int_\Gamma \hat{u} \tau \cdot n dΓ,\\
#  \int_\Omega \sigma \cdot \nabla \delta u dΩ = \int_\Omega \delta u dΩ + \int_\Gamma \delta u \hat{\sigma} \cdot n dΓ,
# ```
# Where $n$ is the outwards pointing normal, and $\Gamma$ is the union of the elements' boundaries.
# Using the following definitions for the jumps and averages of numerical fluxes,
# ```math
#  \{u\} = \frac{1}{2}(u^+ + u^-),\quad [[u]] = u^+ \cdot n^+ + u^- \cdot n^-\\
# ```
# and substituting the integrals of form
# ```math
#  \int_\Gamma q \phi \cdot n dΓ = \int_\Gamma [[q]] \cdot \{\phi\} dΓ + \int_\Gamma \{q\} [[\phi]] dΓ,
# ```
# with the jumps and averages results in
# ```math
#  \int_\Omega \sigma \cdot \tau dΩ = \int_\Omega u \nabla \cdot \tau dΩ + \int_\Gamma [[\hat{u}]] \cdot \{\tau\} dΓ + \int_\Gamma \{\hat{u}\} [[\tau]] dΓ,\\
#  \int_\Omega \sigma \cdot \nabla \delta u dΩ = \int_\Omega \delta u dΩ + \int_\Gamma [[\delta u]] \cdot \{\hat{\sigma}\} dΓ + \int_\Gamma \{\delta u\} [[\hat{\sigma}]] dΓ,
# ```
# Integrating $ \int_\Omega \nabla u \cdot \tau dΩ $ by parts and applying divergence theorem
# without using numerical flux, then substitute in the equation to obtain a weak form.
# ```math
#  \int_\Omega \sigma \cdot \tau dΩ = \int_\Omega \nabla u \cdot \tau dΩ + \int_\Gamma [[\hat{u} - u]] \cdot \{\tau\} dΓ + \int_\Gamma \{\hat{u} - u\} [[\tau]] dΓ,\\
#  \int_\Omega \sigma \cdot \nabla \delta u dΩ = \int_\Omega \delta u dΩ + \int_\Gamma [[\delta u]] \cdot \{\hat{\sigma}\} dΓ + \int_\Gamma \{\delta u\} [[\hat{\sigma}]] dΓ,
# ```
# Substituting
# ```math
#  \tau = \nabla \delta u,\\
# ```
# results in
# ```math
#  \int_\Omega \sigma \cdot \nabla \delta u dΩ = \int_\Omega \nabla u \cdot \nabla \delta u dΩ + \int_\Gamma [[\hat{u} - u]] \cdot \{\nabla \delta u\} dΓ + \int_\Gamma \{\hat{u} - u\} [[\nabla \delta u]] dΓ,\\
#  \int_\Omega \sigma \cdot \nabla \delta u dΩ = \int_\Omega \delta u dΩ + \int_\Gamma [[\delta u]] \cdot \{\hat{\sigma}\} dΓ + \int_\Gamma \{\delta u\} [[\hat{\sigma}]] dΓ,
# ```
# Combining the two equations,
# ```math
#  \int_\Omega \nabla u \cdot \nabla \delta u dΩ + \int_\Gamma [[\hat{u} - u]] \cdot \{\nabla \delta u\} dΓ + \int_\Gamma \{\hat{u} - u\} [[\nabla \delta u]] dΓ - \int_\Gamma [[\delta u]] \cdot \{\hat{\sigma}\} dΓ - \int_\Gamma \{\delta u\} [[\hat{\sigma}]] dΓ = \int_\Omega \delta u dΩ,\\
# ```
# The numerical fluxes chosen for the interior penalty method are $\hat{u} = \{u\}$, $\hat{\sigma} = \{\nabla u\} - \alpha([[u]])$, such choice results in $[[\hat{u}]] = 0$, $\{\hat{u}\} = \{u\}$, $[[\hat{\sigma}]] = 0$, $\{\hat{\sigma}\} = \{\nabla u\} - \alpha([[u]])$
# for the interfaces between elements $\Gamma^0 : \Gamma / \partial \Omega$. While $u$ is well defined on $\partial \Omega$ such that $\hat{u} = u$, $\hat{\sigma} = \nabla u$ resulting in
# ```math
#  \int_\Omega \nabla u \cdot \nabla \delta u dΩ - \int_{\Gamma^0} [[u]] \cdot \{\nabla \delta u\} d\Gamma^0 - \int_{\Gamma^0} [[\delta u]] \cdot \{\nabla u\} - [[\delta u]] \cdot \alpha([[u]]) d\Gamma^0 - \int_{\partial \Omega} \nabla u \cdot \vec{n} \delta u d\partial \Omega = \int_\Omega \delta u dΩ,\\
# ```
# Where
# ```math
# \alpha([[u]]) = \mu [[u]]
# ```
# with $\eta > 0$ and taken as $\eta = (order + 1)^{dim}$
# The resulting weak form is given given as follows: Find ``u \in \mathbb{U}`` such that
# ```math
#  \int_\Omega \nabla u \cdot \nabla \delta u dΩ - \int_{\Gamma^0} [[u]] \cdot \{\nabla \delta u\} + [[\delta u]] \cdot \{\nabla u\} d\Gamma^0 + \int_{\Gamma^0} \mu [[u]] ⋅ [[\delta u]] d\Gamma^0 - \int_{\partial \Omega} \nabla u \cdot \vec{n} \delta u d\partial \Omega = \int_\Omega \delta u dΩ,\\
# ```
# Since $\partial \Omega$ is constrained with both Dirichlet and Neumann boundary conditions the term $\int_{\partial \Omega} \nabla u \cdot \vec{n} \delta u d\partial \Omega$ can be expressed as an integral over $\partial \Omega_n$, where $\partial \Omega_n$ is the boundaries with only prescribed Neumann boundary condition,
# ```math
#  \int_\Omega \nabla u \cdot \nabla \delta u dΩ - \int_{\Gamma^0} [[u]] \cdot \{\nabla \delta u\} + [[\delta u]] \cdot \{\nabla u\} d\Gamma^0 + \int_{\Gamma^0} \mu [[u]] ⋅ [[\delta u]] d\Gamma^0 = \int_\Omega \delta u dΩ + \int_{\partial \Omega} (\nabla u \cdot \vec{n}) \delta u d\partial \Omega,\\
# ```
# where $\delta u \in \mathbb{T}$ is a test function, and where $\mathbb{U}$ and $\mathbb{T}$ are suitable
# trial and test function sets, respectively.
#-
# ## Commented Program
#
# Now we solve the problem in Ferrite. What follows is a program spliced with comments.
#md # The full program, without comments, can be found in the next [section](@ref heat_equation-DG-plain-program).
#
# First we load Ferrite, and some other packages we need
using Ferrite, SparseArrays
# We start by generating a simple grid with 20x20 quadrilateral elements
# using `generate_grid`. The generator defaults to the unit square,
# so we don't need to specify the corners of the domain.
grid = generate_grid(Quadrilateral, (20, 20));
# We construct the topology information which is used later for generating the sparsity pattern for stiffness matrix.
topology = ExclusiveTopology(grid)

# ### Trial and test functions
# `CellValues`, `FaceValues`, and `InterfaceValues` facilitate the process of evaluating values and gradients of
# test and trial functions (among other things). To define
# these we need to specify an interpolation space for the shape functions.
# We use `DiscontinuousLagrange` functions
# based on the two-dimensional reference quadrilateral. We also define a quadrature rule based on
# the same reference element. We combine the interpolation and the quadrature rule
# to `CellValues` and `InterfaceValues` object. Note that `InterfaceValues` object contains two `FaceValues` objects which can be used individually.
ip = DiscontinuousLagrange{RefQuadrilateral, 1}()
qr = QuadratureRule{RefQuadrilateral}(2)
# For `FaceValues` and `InterfaceValues` we use `FaceQuadratureRule`
face_qr = FaceQuadratureRule{RefQuadrilateral}(2)
cellvalues = CellValues(qr, ip);
interfacevalues = InterfaceValues(face_qr, ip)
# ### Degrees of freedom
# Next we need to define a `DofHandler`, which will take care of numbering
# and distribution of degrees of freedom for our approximated fields.
# We create the `DofHandler` and then add a single scalar field called `:u` based on
# our interpolation `ip` defined above.
# Lastly we `close!` the `DofHandler`, it is now that the dofs are distributed
# for all the elements.
dh = DofHandler(grid)
add!(dh, :u, ip)
close!(dh);

# Now that we have distributed all our dofs we can create our tangent matrix,
# using `create_sparsity_pattern`. This function returns a sparse matrix
# with the correct entries stored. We need to pass the topology and the cross-element coupling matrix when we're using
# discontinuous interpolations. The cross-element coupling matrix is of size [1,1] in this case as
# we have only one field and one DofHandler.
K = create_sparsity_pattern(dh, topology = topology, cross_coupling = trues(1,1))

# ### Boundary conditions
# In Ferrite constraints like Dirichlet boundary conditions
# are handled by a `ConstraintHandler`.
ch = ConstraintHandler(dh);

# Next we need to add constraints to `ch`. For this problem we define
# inhomogeneous Dirichlet boundary conditions on the two boundaries.
# We define ∂Ωₙ as the `union` of the face sets with Neumann boundary conditions.
∂Ωₙ = union(
    getfaceset(grid, "top"),
    getfaceset(grid, "bottom"),
);

# Now we are set up to define our constraint. We specify which field
# the condition is for, and our face sets. The last
# argument is a function of the form $f(\textbf{x})$ or $f(\textbf{x}, t)$,
# where $\textbf{x}$ is the spatial coordinate and
# $t$ the current time, and returns the prescribed value.
add!(ch, Dirichlet(:u, getfaceset(grid, "right"), (x, t) -> 1.0));
add!(ch, Dirichlet(:u, getfaceset(grid, "left"), (x, t) -> -1.0));

# Finally we also need to `close!` our constraint handler. When we call `close!`
# the dofs corresponding to our constraints are calculated and stored
# in our `ch` object.
close!(ch)

# Note that if one or more of the constraints are time dependent we would use
# [`update!`](@ref) to recompute prescribed values in each new timestep.

# ### Assembling the linear system
#
# Now we have all the pieces needed to assemble the linear system, $K u = f$.
# Assembling of the global system is done by looping over all the elements in order to
# compute the element contributions ``K_e`` and ``f_e``, all the interfaces
# to compute their contributions ``K_i``, and all the Neumann boundary faces to compute their
# contributions ``f_e`` which are then assembled to the
# appropriate place in the global ``K`` and ``f``.
#
# #### Element assembly
# We define the functions
# * `assemble_element!` to compute the contribution of volume integrals over an element.
# * `assemble_interface!` to compute the contribution of surface integrals over an interface.
# * `assemble_boundary!` to compute the contribution of surface integrals over a boundary face.
#
# The function takes pre-allocated `ke`, `ki`, and `fe` (they are allocated once and
# then reused for all elements) so we first need to make sure that they are all zeroes at
# the start of the function by using `fill!`. Then we loop over all the quadrature points,
# and for each quadrature point we loop over all the (local) shape functions. The *values
# objects provide values needed to calculate contributions as follows:
# * `cellvalues`:
#     * test function value ($\delta u$)
#     * test and trial functions gradient ($\nabla u$ and $\nabla \delta u$)
# * `interfacevalues`:
#     * jump in test and trial functions values
#     * average of test and trial functions gradients
#     * test function value ($δu$) for boundary face integrals
#
# !!! note "Notation"
#     Comparing with the brief finite element introduction in [Introduction to FEM](@ref),
#     the variables `δu`, `∇δu` and `∇u` are actually $\phi_i(\textbf{x}_q)$, $\nabla
#     \phi_i(\textbf{x}_q)$ and $\nabla \phi_j(\textbf{x}_q)$, i.e. the evaluation of the
#     trial and test functions in the quadrature point ``\textbf{x}_q``. However, to
#     underline the strong parallel between the weak form and the implementation, this
#     example uses the symbols appearing in the weak form.

function assemble_element!(Ke::Matrix, fe::Vector, cellvalues::CellValues)
    n_basefuncs = getnbasefunctions(cellvalues)
    ## Reset to 0
    fill!(Ke, 0)
    fill!(fe, 0)
    ## Loop over quadrature points
    for q_point in 1:getnquadpoints(cellvalues)
        ## Get the quadrature weight
        dΩ = getdetJdV(cellvalues, q_point)
        ## Loop over test shape functions
        for i in 1:n_basefuncs
            δu  = shape_value(cellvalues, q_point, i)
            ∇δu = shape_gradient(cellvalues, q_point, i)
            ## Add contribution to fe
            fe[i] += δu * dΩ
            ## Loop over trial shape functions
            for j in 1:n_basefuncs
                ∇u = shape_gradient(cellvalues, q_point, j)
                ## Add contribution to Ke
                Ke[i, j] += (∇δu ⋅ ∇u) * dΩ
            end
        end
    end
    return Ke, fe
end

function assemble_interface!(Ki::Matrix, iv::InterfaceValues, dim::Int, order::Int)
    ## Reset to 0
    fill!(Ki, 0)
    ## Loop over quadrature points
    for q_point in 1:getnquadpoints(iv)
        ## Get the normal to face A
        normal = getnormal(iv, q_point)
        ## Get the quadrature weight
        dΓ = getdetJdV(iv, q_point)
        ## Loop over test shape functions
        for i in 1:getnbasefunctions(iv)
            ## Multiply the jump by the normal, as the definition used in Ferrite doesn't include the normals.
            δu_jump = shape_value_jump(iv, q_point, i) * normal
            ∇δu_avg = shape_gradient_average(iv, q_point, i)
            ## Loop over trial shape functions
            for j in 1:getnbasefunctions(iv)
                ## Multiply the jump by the normal, as the definition used in Ferrite doesn't include the normals.
                u_jump = shape_value_jump(iv, q_point, j) * normal
                ∇u_avg = shape_gradient_average(iv, q_point, j)
                ## Add contribution to Ki          
                Ki[i, j] += -(δu_jump ⋅ ∇u_avg + ∇δu_avg ⋅ u_jump)*dΓ + (1 + order)^dim * 20/2 * (δu_jump ⋅ u_jump)*dΓ
            end
        end
    end
    return Ki
end

function assemble_boundary!(fe::Vector, fv::FaceValues)
    ## Reset to 0
    fill!(fe, 0)
    ## Loop over quadrature points
    for q_point in 1:getnquadpoints(fv)
        ## Get the normal to face A
        normal = getnormal(fv, q_point)
        ## Get the quadrature weight
        ∂Ω = getdetJdV(fv, q_point)
        ## Loop over test shape functions
        for i in 1:getnbasefunctions(fv)
            ## Multiply the jump by the normal, as the definition used in Ferrite doesn't include the normals.
            δu = shape_value(fv, q_point, i)
            fe[i] = normal[2] * δu * ∂Ω
        end
    end
    return fe
end
#md nothing # hide

# #### Global assembly
# We define the function `assemble_global` to loop over the elements and do the global
# assembly. The function takes our `cellvalues`, `interfacevalues`, the sparse matrix `K`, and our DofHandler
# as input arguments and returns the assembled global stiffness matrix, and the assembled
# global force vector. We start by allocating `Ke`, `Ki`, `fe`, and the global force vector `f`.
# We also create an assembler by using `start_assemble`. The assembler lets us assemble into
# `K` and `f` efficiently. We then start the loop over all the elements. In each loop
# iteration we reinitialize `cellvalues` (to update derivatives of shape functions etc.),
# compute the element contribution with `assemble_element!`, and then assemble into the
# global `K` and `f` with `assemble!`.
#
# !!! note "Notation"
#     Comparing again with [Introduction to FEM](@ref), `f` and `u` correspond to
#     $\underline{\hat{f}}$ and $\underline{\hat{u}}$, since they represent the discretized
#     versions. However, through the code we use `f` and `u` instead to reflect the strong
#     connection between the weak form and the Ferrite implementation.

function assemble_global(cellvalues::CellValues, interfacevalues::InterfaceValues, K::SparseMatrixCSC, dh::DofHandler)
    ## Allocate the element stiffness matrix and element force vector
    n_basefuncs = getnbasefunctions(cellvalues)
    Ke = zeros(n_basefuncs, n_basefuncs)
    fe = zeros(n_basefuncs)
    Ki = zeros(n_basefuncs * 2, n_basefuncs * 2)
    ## Allocate global force vector f
    f = zeros(ndofs(dh))
    ## Create an assembler
    assembler = start_assemble(K, f)
    ## Loop over all cells
    for cell in CellIterator(dh)
        ## Reinitialize cellvalues for this cell
        reinit!(cellvalues, cell)
        ## Compute volume integral contribution
        assemble_element!(Ke, fe, cellvalues)
        ## Assemble Ke and fe into K and f
        assemble!(assembler, celldofs(cell), Ke, fe)
    end
    ## get interpolation dimensions and order to use in the penalty weight
    ip = Ferrite.function_interpolation(interfacevalues.here)
    order = Ferrite.getorder(ip)
    dim = Ferrite.getdim(ip)
    ## Loop over all interfaces
    for ic in InterfaceIterator(dh)
        ## Reinitialize interfacevalues for this interface
        reinit!(interfacevalues, ic)
        ## Compute interface surface integrals contribution
        assemble_interface!(Ki, interfacevalues, dim, order)
        ## Assemble Ki into K
        assemble!(assembler, interfacedofs(ic), Ki)
    end
    ## Loop over domain boundaries with Neumann boundary conditions
    for fc in FaceIterator(dh, ∂Ωₙ)
        ## Reinitialize face_values_a for this boundary face
        reinit!(interfacevalues.here, fc)
        ## Compute boundary face surface integrals contribution
        assemble_boundary!(fe, interfacevalues.here)
        ## Assemble fe into f
        assemble!(f, celldofs(fc), fe)
    end
    return K, f
end
#md nothing # hide

# ### Solution of the system
# The last step is to solve the system. First we call `assemble_global`
# to obtain the global stiffness matrix `K` and force vector `f`.
K, f = assemble_global(cellvalues, interfacevalues, K, dh);

# To account for the boundary conditions we use the `apply!` function.
# This modifies elements in `K` and `f` respectively, such that
# we can get the correct solution vector `u` by using `\`.
apply!(K, f, ch)
u = K \ f;

# ### Exporting to VTK
# To visualize the result we export the grid and our field `u`
# to a VTK-file, which can be viewed in e.g. [ParaView](https://www.paraview.org/).
vtk_grid("dg_heat_equation", dh) do vtk
    vtk_point_data(vtk, dh, u)
end

## test the result                #src
#using Test                        #src
#@test norm(u) ≈ 3.307743912641305 #src

#md # ## [Plain program](@id heat_equation-DG-plain-program)
#md #
#md # Here follows a version of the program without any comments.
#md # The file is also available here: [`dg_heat_equation.jl`](dg_heat_equation.jl).
#md #
#md # ```julia
#md # @__CODE__
#md # ```