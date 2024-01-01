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
# This example was developed
# as part of the Google summer of code funded project
# ["Discontinuous Galerkin Infrastructure For the finite element toolbox Ferrite.jl"](https://summerofcode.withgoogle.com/programs/2023/projects/SLGbRNI5)
#
# ## Introduction
#
# This tutorial extends [tutorial 1: Heat equation](heat_equation.md) by using discontinuous Galerkin method.
# The reader is expected to have gone throught [tutorial 1: Heat equation](heat_equation.md) before proceeding with this tutorial.
# The main differences between the two tutorials are the interface integral terms in the weak form, the boundary conditions, and
# some implementation differences explained in the commented program.
# The strong form used in this tutorial is
# ```math
#  -\nabla \cdot (\nabla u) = 1  \quad \textbf{x} \in \Omega,
# ```
# 
# With the inhomogeneous Dirichlet boundary conditions
# ```math
# u(\textbf{x}) = 1 \quad \textbf{x} \in \partial \Omega_D^+ = \lbrace\textbf{x} : x_1 = 1.0\rbrace, \\
# u(\textbf{x}) = -1 \quad \textbf{x} \in \partial \Omega_D^- = \lbrace\textbf{x} : x_1 = -1.0\rbrace,
# ```
# and Neumann boundary conditions
# ```math
# \nabla u(\textbf{x}) \cdot \boldsymbol{n} = 1 \quad \textbf{x} \in \partial \Omega_N^+ = \lbrace\textbf{x} : x_2 = 1.0\rbrace, \\
# \nabla u(\textbf{x}) \cdot \boldsymbol{n} = -1 \quad \textbf{x} \in \partial \Omega_N^- = \lbrace\textbf{x} : x_2 = -1.0\rbrace,
# ```
#
# The definitions of jumps and averages used in this examples are
# ```math
#  \{u\} = \frac{1}{2}(u^+ + u^-),\quad \llbracket u\rrbracket  = u^+ \boldsymbol{n}^+ + u^- \boldsymbol{n}^-\\
# ```
# !!! details "Derivation of the weak form for homogeneous Dirichlet boundary condition"
#     Defining $\boldsymbol{\sigma}$ as the gradient of the temperature field the equation can be expressed as
#     ```math
#      \boldsymbol{\sigma} = \nabla u,\\
#      -\nabla \cdot \boldsymbol{\sigma} = 1,
#     ```
#     Multiplying by test functions $ \boldsymbol{\tau} $ and $ \delta u $ respectively and integrating
#     over the domain,
#     ```math
#      \int_\Omega \boldsymbol{\sigma} \cdot \boldsymbol{\tau} \,\mathrm{d}\Omega = \int_\Omega \nabla u \cdot \boldsymbol{\tau} \,\mathrm{d}\Omega,\\
#      -\int_\Omega \nabla \cdot \boldsymbol{\sigma} \delta u \,\mathrm{d}\Omega = \int_\Omega \delta u \,\mathrm{d}\Omega,
#     ```
#     Integrating by parts and applying divergence theorem,
#     ```math
#      \int_\Omega \boldsymbol{\sigma} \cdot \boldsymbol{\tau} \,\mathrm{d}\Omega = \int_\Omega u \nabla \cdot \boldsymbol{\tau} \,\mathrm{d}\Omega + \int_\Gamma \hat{u} \boldsymbol{\tau} \cdot \boldsymbol{n} \,\mathrm{d}\Gamma,\\
#      \int_\Omega \boldsymbol{\sigma} \cdot \nabla \delta u \,\mathrm{d}\Omega = \int_\Omega \delta u \,\mathrm{d}\Omega + \int_\Gamma \delta u \boldsymbol{\hat{\sigma}} \cdot \boldsymbol{n} \,\mathrm{d}\Gamma,
#     ```
#     Where $\boldsymbol{n}$ is the outwards pointing normal, and $\Gamma$ is the union of the elements' boundaries.
#     Substituting the integrals of form
#     ```math
#      \int_\Gamma q \boldsymbol{\phi} \cdot \boldsymbol{n} \,\mathrm{d}\Gamma = \int_\Gamma \llbracket q\rrbracket  \cdot \boldsymbol{\{\phi\}} \,\mathrm{d}\Gamma + \int_{\Gamma^0} \{q\} \llbracket \phi\rrbracket  \,\mathrm{d}\Gamma^0,
#     ```
#     where $\Gamma^0 : \Gamma \setminus \partial \Omega$, with the jumps and averages results in
#     ```math
#      \int_\Omega \boldsymbol{\sigma} \cdot \boldsymbol{\tau} \,\mathrm{d}\Omega = \int_\Omega u \nabla \cdot \boldsymbol{\tau} \,\mathrm{d}\Omega + \int_\Gamma \llbracket \hat{u}\rrbracket  \cdot \boldsymbol{\{\tau\}} \,\mathrm{d}\Gamma + \int_{\Gamma^0} \{\hat{u}\} \llbracket \tau\rrbracket  \,\mathrm{d}\Gamma^0,\\
#      \int_\Omega \boldsymbol{\sigma} \cdot \nabla \delta u \,\mathrm{d}\Omega = \int_\Omega \delta u \,\mathrm{d}\Omega + \int_\Gamma \llbracket \delta u\rrbracket  \cdot \boldsymbol{\{\hat{\sigma}\}} \,\mathrm{d}\Gamma + \int_{\Gamma^0} \{\delta u\} \llbracket \hat{\sigma}\rrbracket  \,\mathrm{d}\Gamma^0,
#     ```
#     Integrating $ \int_\Omega \nabla u \cdot \boldsymbol{\tau} \,\mathrm{d}\Omega $ by parts and applying divergence theorem
#     without using numerical flux, then substitute in the equation to obtain a weak form.
#     ```math
#      \int_\Omega \boldsymbol{\sigma} \cdot \boldsymbol{\tau} \,\mathrm{d}\Omega = \int_\Omega \nabla u \cdot \boldsymbol{\tau} \,\mathrm{d}\Omega + \int_\Gamma \llbracket \hat{u} - u\rrbracket  \cdot \boldsymbol{\{\tau\}} \,\mathrm{d}\Gamma + \int_{\Gamma^0} \{\hat{u} - u\} \llbracket \tau\rrbracket  \,\mathrm{d}\Gamma^0,\\
#      \int_\Omega \boldsymbol{\sigma} \cdot \nabla \delta u \,\mathrm{d}\Omega = \int_\Omega \delta u \,\mathrm{d}\Omega + \int_\Gamma \llbracket \delta u\rrbracket  \cdot \boldsymbol{\{\hat{\sigma}\}} \,\mathrm{d}\Gamma + \int_{\Gamma^0} \{\delta u\} \llbracket \hat{\sigma}\rrbracket  \,\mathrm{d}\Gamma^0,
#     ```
#     Substituting
#     ```math
#      \boldsymbol{\tau} = \nabla \delta u,\\
#     ```
#     results in
#     ```math
#      \int_\Omega \boldsymbol{\sigma} \cdot \nabla \delta u \,\mathrm{d}\Omega = \int_\Omega \nabla u \cdot \nabla \delta u \,\mathrm{d}\Omega + \int_\Gamma \llbracket \hat{u} - u\rrbracket  \cdot \boldsymbol{\{\nabla \delta u\}} \,\mathrm{d}\Gamma + \int_{\Gamma^0} \{\hat{u} - u\} \llbracket \nabla \delta u\rrbracket  \,\mathrm{d}\Gamma^0,\\
#      \int_\Omega \boldsymbol{\sigma} \cdot \nabla \delta u \,\mathrm{d}\Omega = \int_\Omega \delta u \,\mathrm{d}\Omega + \int_\Gamma \llbracket \delta u\rrbracket  \cdot \boldsymbol{\{\hat{\sigma}\}} \,\mathrm{d}\Gamma + \int_{\Gamma^0} \{\delta u\} \llbracket \hat{\sigma}\rrbracket  \,\mathrm{d}\Gamma^0,
#     ```
#     Combining the two equations,
#     ```math
#      \int_\Omega \nabla u \cdot \nabla \delta u \,\mathrm{d}\Omega + \int_\Gamma \llbracket \hat{u} - u\rrbracket  \cdot \boldsymbol{\{\nabla \delta u\}} \,\mathrm{d}\Gamma + \int_{\Gamma^0} \{\hat{u} - u\} \llbracket \nabla \delta u\rrbracket  \,\mathrm{d}\Gamma^0 - \int_\Gamma \llbracket \delta u\rrbracket  \cdot \boldsymbol{\{\hat{\sigma}\}} \,\mathrm{d}\Gamma - \int_{\Gamma^0} \{\delta u\} \llbracket \hat{\sigma}\rrbracket  \,\mathrm{d}\Gamma^0 = \int_\Omega \delta u \,\mathrm{d}\Omega,\\
#     ```
#     The numerical fluxes chosen for the interior penalty method are $\boldsymbol{\hat{\sigma}} = \{\nabla u\} - \alpha(\llbracket u\rrbracket)$ on $\Gamma$, $\hat{u} = \{u\}$ on the interfaces between elements $\Gamma^0 : \Gamma \setminus \partial \Omega$, 
#     and $\hat{u} = 0$ on $\partial \Omega$. Such choice results in $\boldsymbol{\{\hat{\sigma}\}} = \{\nabla u\} - \alpha(\llbracket u\rrbracket)$, $\llbracket \hat{u}\rrbracket  = 0$, $\{\hat{u}\} = \{u\}$, $\llbracket \boldsymbol{\hat{\sigma}}\rrbracket  = 0$ and the equation becomes
#     ```math
#      \int_\Omega \nabla u \cdot \nabla \delta u \,\mathrm{d}\Omega - \int_\Gamma \llbracket u\rrbracket  \cdot \boldsymbol{\{\nabla \delta u\}}  \,\mathrm{d}\Gamma - \int_\Gamma \llbracket \delta u\rrbracket  \cdot \boldsymbol{\{\nabla u\}} - \llbracket \delta u\rrbracket  \cdot \alpha(\llbracket u\rrbracket)  \,\mathrm{d}\Gamma = \int_\Omega \delta u \,\mathrm{d}\Omega,\\
#     ```
#     Where
#     ```math
#      \alpha(\llbracket u\rrbracket) = \mu \llbracket u\rrbracket
#     ```
#     Where $\mu = \eta h^{-1}$, the weak form becomes
#     ```math
#      \int_\Omega \nabla u \cdot \nabla \delta u \,\mathrm{d}\Omega - \int_\Gamma \llbracket u \rrbracket \cdot \boldsymbol{\{\nabla \delta u\}} + \llbracket \delta u \rrbracket  \cdot \boldsymbol{\{\nabla u\}}  \,\mathrm{d}\Gamma + \int_\Gamma \frac{\eta}{h} \llbracket u\rrbracket  \cdot \llbracket \delta u\rrbracket   \,\mathrm{d}\Gamma = \int_\Omega \delta u \,\mathrm{d}\Omega,\\
#     ```
# Since $\partial \Omega$ is constrained with both Dirichlet and Neumann boundary conditions the term $\int_{\partial \Omega} \nabla u \cdot \boldsymbol{n} \delta u \,\mathrm{d} \partial \Omega$ can be expressed as an integral over $\partial \Omega_N$, where $\partial \Omega_N$ is the boundaries with only prescribed Neumann boundary condition,
# The resulting weak form is given given as follows: Find $u \in \mathbb{U}$ such that
# ```math
#  \int_\Omega \nabla u \cdot \nabla \delta u \,\mathrm{d}\Omega - \int_{\Gamma^0} \llbracket u\rrbracket  \cdot \boldsymbol{\{\nabla \delta u\}} + \llbracket \delta u\rrbracket  \cdot \boldsymbol{\{\nabla u\}}  \,\mathrm{d}\Gamma^0 + \int_{\Gamma^0} \frac{\eta}{h} \llbracket u\rrbracket \cdot \llbracket \delta u\rrbracket   \,\mathrm{d}\Gamma^0 = \int_\Omega \delta u \,\mathrm{d}\Omega + \int_{\partial \Omega_N} (\nabla u \cdot \boldsymbol{n}) \delta u \,\mathrm{d} \partial \Omega_N,\\
# ```
# where $h$ is the characteristic mesh size (the maximum diameter of the cells), and $\eta$ is a large enough positive number independent of $h$ [Mu:2014:IP](@cite),
# $\delta u \in \mathbb{T}$ is a test function, and where $\mathbb{U}$ and $\mathbb{T}$ are suitable
# trial and test function sets, respectively.
#
# More details on DG formulations for elliptic problems can be found in [Cockburn:2002:unifiedanalysis](@cite)
#-
# ## Commented Program
#
# Now we solve the problem in Ferrite. What follows is a program spliced with comments.
#md # The full program, without comments, can be found in the next [section](@ref heat_equation-DG-plain-program).
#
# First we load Ferrite and other packages, and generate grid just like the [heat equation tutorial](heat_equation.md)
using Ferrite, SparseArrays
grid = generate_grid(Quadrilateral, (20, 20));
# We calculate the parameter $h$ as the maximum cell diameter.
getdistance(p1::Vec{N, T},p2::Vec{N, T}) where {N, T} = norm(p1-p2);
getdiameter(cell_coords::Vector{Vec{N, T}}) where {N, T} = maximum(getdistance.(cell_coords, reshape(cell_coords, (1,:))))
h = maximum(i -> getdiameter(getcoordinates(grid, i)), 1:getncells(grid))
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
qr = QuadratureRule{RefQuadrilateral}(2);
# For `FaceValues` and `InterfaceValues` we use `FaceQuadratureRule`
face_qr = FaceQuadratureRule{RefQuadrilateral}(2)
cellvalues = CellValues(qr, ip);
facevalues = FaceValues(face_qr, ip);
interfacevalues = InterfaceValues(facevalues)
# ### Degrees of freedom
# Degrees of freedom distribution is handled using `DofHandler` as usual
dh = DofHandler(grid)
add!(dh, :u, ip)
close!(dh);

# However, when generating the sparsity pattern we need to pass the topology and the cross-element coupling matrix when we're using
# discontinuous interpolations. The cross-element coupling matrix is of size [1,1] in this case as
# we have only one field and one DofHandler.
K = create_sparsity_pattern(dh, topology = topology, cross_coupling = trues(1,1));

# ### Boundary conditions
# The Dirichlet boundary conditions are treated 
# as usual by a `ConstraintHandler`.
ch = ConstraintHandler(dh)
add!(ch, Dirichlet(:u, getfaceset(grid, "right"), (x, t) -> 1.0))
add!(ch, Dirichlet(:u, getfaceset(grid, "left"), (x, t) -> -1.0))
close!(ch);

# Furthermore, we define $\partial \Omega_N$ as the `union` of the face sets with Neumann boundary conditions for later use
∂Ωₙ = union(
    getfaceset(grid, "top"),
    getfaceset(grid, "bottom"),
);


# ### Assembling the linear system
#
# Now we have all the pieces needed to assemble the linear system, $K u = f$.
# Assembling of the global system is done by looping over all the elements in order to
# compute the element contributions ``K_e`` and ``f_e``, all the interfaces
# to compute their contributions ``K_i``, and all the Neumann boundary faces to compute their
# contributions ``f_e`` which are then assembled to the
# appropriate place in the global ``K`` and ``f``.
#
# #### Local assembly
# We define the functions
# * `assemble_element!` to compute the contributions ``K_e`` and ``f_e`` of volume integrals over an element using `cellvalues`.
# * `assemble_interface!` to compute the contribution ``K_i`` of surface integrals over an interface using `interfacevalues`.
# * `assemble_boundary!` to compute the contribution ``f_e`` of surface integrals over a boundary face using `facevalues`.

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

function assemble_interface!(Ki::Matrix, iv::InterfaceValues, μ::Float64)
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
                Ki[i, j] += -(δu_jump ⋅ ∇u_avg + ∇δu_avg ⋅ u_jump)*dΓ +  μ * (δu_jump ⋅ u_jump) * dΓ
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
            δu = shape_value(fv, q_point, i)
            boundary_flux = normal[2]
            fe[i] = boundary_flux * δu * ∂Ω
        end
    end
    return fe
end
#md nothing # hide

# #### Global assembly
# We define the function `assemble_global` to loop over all elements and internal faces (interfaces), as well as the external faces involved in Neumann boundary conditions. 

function assemble_global(cellvalues::CellValues, facevalues::FaceValues, interfacevalues::InterfaceValues, K::SparseMatrixCSC, dh::DofHandler, h::Float64)
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
    ## get interpolation dimensions and order to use in the penalty weight, using the $(1 + order)^{dim}$ for $\eta$
    ip = Ferrite.function_interpolation(facevalues)
    order = Ferrite.getorder(ip)
    dim = Ferrite.getdim(ip)
    μ = (1 + order)^dim / h
    ## Loop over all interfaces
    for ic in InterfaceIterator(dh)
        ## Reinitialize interfacevalues for this interface
        reinit!(interfacevalues, ic)
        ## Compute interface surface integrals contribution
        assemble_interface!(Ki, interfacevalues, μ)
        ## Assemble Ki into K
        assemble!(assembler, interfacedofs(ic), Ki)
    end
    ## Loop over domain boundaries with Neumann boundary conditions
    for fc in FaceIterator(dh, ∂Ωₙ)
        ## Reinitialize face_values_a for this boundary face
        reinit!(facevalues, fc)
        ## Compute boundary face surface integrals contribution
        assemble_boundary!(fe, facevalues)
        ## Assemble fe into f
        assemble!(f, celldofs(fc), fe)
    end
    return K, f
end
K, f = assemble_global(cellvalues, facevalues, interfacevalues, K, dh, h);
#md nothing # hide

# ### Solution of the system
# The solution of the system is independent of the discontinuous discretization and the application of constraints, linear solve, and exporting is done as usual.

apply!(K, f, ch)
u = K \ f;
vtk_grid("dg_heat_equation", dh) do vtk
    vtk_point_data(vtk, dh, u)
end

## test the result                #src
using Test                        #src
@test norm(u) ≈ 27.88892990564881 #src

#md # ## References
#md # ```@bibliography
#md # Pages = ["tutorials/dg_heat_equation.md"]
#md # Canonical = false
#md # ```

#md # ## [Plain program](@id heat_equation-DG-plain-program)
#md #
#md # Here follows a version of the program without any comments.
#md # The file is also available here: [`dg_heat_equation.jl`](dg_heat_equation.jl).
#md #
#md # ```julia
#md # @__CODE__
#md # ```
