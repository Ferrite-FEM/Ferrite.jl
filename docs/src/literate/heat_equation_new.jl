# # Heat Equation
#
# ![](heat_square.png)
#
# *Figure 1*: Temperature field on the unit square with an internal uniform heat source
# solved with homogeneous Dirichlet boundary conditions on the boundary.
#
#-
#md # !!! tip
#md #     This example is also available as a Jupyter notebook:
#md #     [`heat_equation.ipynb`](@__NBVIEWER_ROOT_URL__/examples/heat_equation.ipynb).
#-
#
# ## Introduction
#
# The heat equation is the "Hello, world!" equation of finite elements.
# Here we solve the equation on a unit square, with a uniform internal source.
# The strong form of the (linear) heat equation is given by
#
# ```math
#  -\nabla \cdot (k \nabla u) = f  \quad \textbf{x} \in \Omega,
# ```
#
# where $u$ is the unknown temperature field, $k$ the heat conductivity,
# $f$ the heat source and $\Omega$ the domain. For simplicity we set $f = 1$
# and $k = 1$. We will consider homogeneous Dirichlet boundary conditions such that
# ```math
# u(\textbf{x}) = 0 \quad \textbf{x} \in \partial \Omega,
# ```
# where $\partial \Omega$ denotes the boundary of $\Omega$.
# The resulting weak form is given given as follows: Find ``u \in \mathbb{U}`` such that
# ```math
# \int_{\Omega} \nabla \delta u \cdot \nabla u \ d\Omega = \int_{\Omega} \delta u \ d\Omega \quad \forall \delta u \in \mathbb{T},
# ```
# where $\delta u$ is a test function, and where $\mathbb{U}$ and $\mathbb{T}$ are suitable
# trial and test function sets, respectively.
#-
# ## Commented Program
#
# Now we solve the problem in Ferrite. What follows is a program spliced with comments.
#md # The full program, without comments, can be found in the next [section](@ref heat_equation-plain-program).
#
# First we load Ferrite, and some other packages we need
using Ferrite, SparseArrays
# We start by generating a simple mesh with 20x20 quadrilateral elements
# using `generate_mesh`. The generator defaults to the unit square,
# so we don't need to specify the corners of the domain.
nel_x = 10
nel_y = 10
mesh_quad = generate_mesh(QuadrilateralElement, (nel_x, nel_y));
#mesh = mesh_quad

nodes = mesh_quad.nodes
elements = Union{QuadrilateralElement, Ferrite.TriangleElement}[]
n_nodes_x = nel_x+1
n_nodes_y = nel_y+1
n_nodes = n_nodes_x*n_nodes_y
node_array = reshape(collect(1:n_nodes), (n_nodes_x, n_nodes_y))
for i ∈ 1:2:(n_nodes_x-2)
    for j ∈ 1:2:(n_nodes_y-2)
        push!(elements, QuadrilateralElement((node_array[i,j], node_array[i+1,j], node_array[i+1,j+1], node_array[i,j+1])))
        push!(elements, QuadrilateralElement((node_array[i+1,j+1], node_array[i+2,j+1], node_array[i+2,j+2], node_array[i+1,j+2])))

        push!(elements, Ferrite.TriangleElement((node_array[i+1,j], node_array[i+2,j], node_array[i+1,j+1])))
        push!(elements, Ferrite.TriangleElement((node_array[i+1,j+1], node_array[i+2,j], node_array[i+2,j+1])))

        push!(elements, Ferrite.TriangleElement((node_array[i+1-1,j+1], node_array[i+2-1,j+1], node_array[i+1-1,j+1+1])))
        push!(elements, Ferrite.TriangleElement((node_array[i+1-1,j+1+1], node_array[i+2-1,j+1], node_array[i+2-1,j+1+1])))
    end
end
mesh = Ferrite.Mesh(elements, nodes)

# ### Trial and test functions
# A `CellValues` facilitates the process of evaluating values and gradients of
# test and trial functions (among other things). Since the problem
# is a scalar problem we will use a `CellScalarValues` object. To define
# this we need to specify an interpolation space for the shape functions.
# We use Lagrange functions (both for interpolating the function and the geometry)
# based on the two-dimensional reference "cube". We also define a quadrature rule based on
# the same reference cube. We combine the interpolation and the quadrature rule
# to a `CellScalarValues` object.
dim = 2
ip_quad = Lagrange{dim, RefCube, 1}()
ip_geo_quad = Lagrange{dim, RefCube, 1}()
qr_quad = QuadratureRule{dim, RefCube}(2)
ip_tri = Lagrange{dim, Ferrite.RefSimplex, 1}()
ip_geo_tri = Lagrange{dim, Ferrite.RefSimplex, 1}()
qr_tri = QuadratureRule{dim, Ferrite.RefSimplex}(2)
cellvalues = Dict([
    Ferrite.TriangleElement => CellScalarValues(qr_tri, ip_tri, ip_geo_tri),
    QuadrilateralElement => CellScalarValues(qr_quad, ip_quad, ip_geo_quad)
])

# ### Degrees of freedom
# Next we need to define a `NewDofHandler`, which will take care of numbering
# and distribution of degrees of freedom for our approximated fields.
# We create the `NewDofHandler` and then add a single scalar field called `:u`.
# Lastly we `close!` the `NewDofHandler`, it is now that the dofs are distributed
# for all the elements.
dh = NewDofHandler(mesh)
ip_field = Lagrange{dim, Union{RefCube, Ferrite.RefSimplex}, 1}()
Ferrite.add_field!(dh, :u, 1, ip_field)
close!(dh);

# u = [i for i ∈ 1:ndofs(dh)]

# vtk_grid("heat_equation", dh) do vtk
#     vtk_point_data(vtk, u, "u")
# end

# Now that we have distributed all our dofs we can create our tangent matrix,
# using `create_sparsity_pattern`. This function returns a sparse matrix
# with the correct entries stored.
K = create_sparsity_pattern(dh)

# ### Boundary conditions
# In Ferrite constraints like Dirichlet boundary conditions
# are handled by a `ConstraintHandler`.
ch = ConstraintHandler(dh);

# Next we need to add constraints to `ch`. For this problem we define
# homogeneous Dirichlet boundary conditions on the whole boundary, i.e.
# the `union` of all the face sets on the boundary.
# ∂Ω = union(
#     getfaceset(mesh, "left"),
#     getfaceset(mesh, "right"),
#     getfaceset(mesh, "top"),
#     getfaceset(mesh, "bottom"),
# );

# Now we are set up to define our constraint. We specify which field
# the condition is for, and our combined face set `∂Ω`. The last
# argument is a function which takes the spatial coordinate $\textbf{x}$ and
# the current time $t$ and returns the prescribed value. In this case
# it is trivial -- no matter what $\textbf{x}$ and $t$ we return $0$. When we have
# specified our constraint we `add!` it to `ch`.
# dbc = Dirichlet(:u, Set([FaceIndex(1,1)]), (x, t) -> 0)
# add!(ch, dbc);

# We also need to `close!` and `update!` our boundary conditions. When we call `close!`
# the dofs corresponding to our constraints are calculated and stored
# in our `ch` object. Since the boundary conditions are, in this case,
# independent of time we can `update!` them directly with e.g. $t = 0$.
# close!(ch)
# update!(ch, 0.0);

# ### Assembling the linear system
#
# Now we have all the pieces needed to assemble the linear system, $K u = f$.
# Assembling of the global system is done by looping over all the elements in order to
# compute the element contributions ``K_e`` and ``f_e``, which are then assembled to the
# appropriate place in the global ``K`` and ``f``.
#
# #### Element assembly
# We define the function `assemble_element!` (see below) which computes the contribution for
# an element. The function takes pre-allocated `ke` and `fe` (they are allocated once and
# then reused for all elements) so we first need to make sure that they are all zeroes at
# the start of the function by using `fill!`. Then we loop over all the quadrature points,
# and for each quadrature point we loop over all the (local) shape functions. We need the
# value and gradient of the test function, `δu` and also the gradient of the trial function
# `u`. We get all of these from `cellvalues`.
#
# !!! note "Notation"
#     Comparing with the brief finite element introduction in [Introduction to FEM](@ref),
#     the variables `δu`, `∇δu` and `∇u` are actually $\phi_i(\textbf{x}_q)$, $\nabla
#     \phi_i(\textbf{x}_q)$ and $\nabla \phi_j(\textbf{x}_q)$, i.e. the evaluation of the
#     trial and test functions in the quadrature point ``\textbf{x}_q``. However, to
#     underline the strong parallel between the weak form and the implementation, this
#     example uses the symbols appearing in the weak form.

function assemble_element!(Ke::Matrix, fe::Vector, cellvalues::CellScalarValues)
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
#md nothing # hide

# #### Global assembly
# We define the function `assemble_global` to loop over the elements and do the global
# assembly. The function takes our `cellvalues`, the sparse matrix `K`, and our NewDofHandler
# as input arguments and returns the assembled global stiffness matrix, and the assembled
# global force vector. We start by allocating `Ke`, `fe`, and the global force vector `f`.
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

function handle_element(cellvalues, dh, assembler, elidx, element::E) where {E <: Ferrite.Element}
    #TODO buffers
    ##TODO how to improve here?
    elementvalues = cellvalues[typeof(element)]
    coords = Vector{Vec{dim,Float64}}(undef, length(element.nodes))
    Ferrite.elementcoords!(coords, dh.mesh, element)
    reinit!(elementvalues, coords)

    n_basefuncs = getnbasefunctions(elementvalues)
    Ke = zeros(n_basefuncs, n_basefuncs)
    fe = zeros(n_basefuncs)

    ## Compute element contribution
    assemble_element!(Ke, fe, elementvalues)

    ## Assemble Ke and fe into K and f
    cur_elementdofs = zeros(Int, Ferrite.ndofs_per_element(dh, elidx))
    Ferrite.elementdofs!(cur_elementdofs, dh, elidx)
    assemble!(assembler, cur_elementdofs, Ke, fe)
end

function assemble_global(cellvalues, K::SparseMatrixCSC, dh::NewDofHandler)
    ## Allocate global force vector f
    f = zeros(ndofs(dh))
    ## Create an assembler
    assembler = start_assemble(K, f)
    ## Loop over all cels
    for (elidx, element) ∈ enumerate(dh.mesh.elements)
        handle_element(cellvalues, dh, assembler, elidx, element)
    end
    return K, f
end
#md nothing # hide

# ### Solution of the system
# The last step is to solve the system. First we call `assemble_global`
# to obtain the global stiffness matrix `K` and force vector `f`.
K, f = assemble_global(cellvalues, K, dh);
K[1,:] .= 0.0
K[:,1] .= 0.0
K[1,1] = 1.0
# To account for the boundary conditions we use the `apply!` function.
# This modifies elements in `K` and `f` respectively, such that
# we can get the correct solution vector `u` by using `\`.
apply!(K, f, ch)
u = K \ f;

# ### Exporting to VTK
# To visualize the result we export the mesh and our field `u`
# to a VTK-file, which can be viewed in e.g. [ParaView](https://www.paraview.org/).
vtk_grid("heat_equation", dh) do vtk
    #FIXME export is wrong like this
    vtk_point_data(vtk, u, "u")
end

## test the result                #src
using Test                        #src
@test norm(u) ≈ 3.307743912641305 #src

#md # ## [Plain program](@id heat_equation-plain-program)
#md #
#md # Here follows a version of the program without any comments.
#md # The file is also available here: [`heat_equation.jl`](heat_equation.jl).
#md #
#md # ```julia
#md # @__CODE__
#md # ```
