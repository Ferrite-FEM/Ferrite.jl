# # [Linear elasticity](@id tutorial-linear-elasticity)
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
#  -\boldsymbol{\sigma} \cdot \boldsymbol{\nabla} = \boldsymbol{b}  \quad \textbf{x} \in \Omega,
# ```
#
# where $\boldsymbol{\sigma}$ is the stress tensor, $\boldsymbol{b}$ is the body force and
# $\Omega$ the domain.
#
# In this example, we use linear elasticity, such that
# ```math
# \boldsymbol{\sigma} = \boldsymbol{E} : \boldsymbol \varepsilon
# ```
# where $\boldsymbol{E}$ is the elastic stiffness tensor and $\boldsymbol{\varepsilon}$ is
# the small strain tensor that is computed from the displacement field $\boldsymbol{u}$ as
# ```math
# \boldsymbol{\varepsilon} = \frac{1}{2} \left(
#   \boldsymbol{\nabla} \otimes \boldsymbol{u}
#   +
#   \boldsymbol{u} \otimes \boldsymbol{\nabla}
# \right)
# ```
#
# The resulting weak form is given given as follows: Find ``\boldsymbol{u} \in \mathbb{U}`` such that
# ```math
# \int_\Omega 
#   \boldsymbol{\sigma} : \left(\delta \boldsymbol{u} \otimes \boldsymbol{\nabla} \right)
# \, \mathrm{d}V
# =
# \int_{\partial\Omega}
#   \boldsymbol{t}^\ast \cdot \delta \boldsymbol{u}
# \, \mathrm{d}A
# +
# \int_\Omega
#   \boldsymbol{b} \cdot \delta \boldsymbol{u}
# \, \mathrm{d}V
# \quad \forall \, \delta \boldsymbol{u} \in \mathbb{T},
# ```
# where $\delta \boldsymbol{u}$ is a vector valued test function, and where $\mathbb{U}$ and
# $\mathbb{T}$ are suitable trial and test function sets, respectively.
#-

# First we load Ferrite, and some other packages we need
using Ferrite, FerriteGmsh, SparseArrays
# Like for the Heat Equation example, we will use a unit square - but here we'll load the grid of the Ferrite logo! This is done by `FerriteGmsh` here. 
grid = togrid("logo.geo") # TODO: where to store correctly and how to refer to this?
# By default the grid lacks the facesets for the boundaries, so we add them by Ferrite here.
# Note that approximate comparison to 0.0 doesn't work well, so we use a tolerance instead.
addfaceset!(grid, "right", x->x[1] ≈ 1.0)
addfaceset!(grid, "top", x->x[2] ≈ 1.0)
addfaceset!(grid, "left", x->x[1] < 1e-6) 
addfaceset!(grid, "bottom", x->x[2] < 1e-6)

# ### Trial and test functions
# We use linear Lagrange functions as test and trial functions. The grid is composed of triangular elements, thus we need the Lagrange functions defined on `RefTriangle`. All currently available interpolations can be found under [`Interpolation`](@ref).
# Since the displacement field $\boldsymbol{u}$ is a vector valued field, we vectorize the interpolation by the spatial dimension `dim`.
dim = 2
order = 1 # linear interpolation
ip = Lagrange{RefTriangle, order}()^dim
qr = QuadratureRule{RefQuadrilateral}(2) # 2 quadrature points per spatial dimension
cellvalues = CellValues(qr, ip)

# ### Degrees of freedom
# For distributing degrees of freedom, we define a `DofHandler`. The `DofHandler` knows that `u` has two degrees of freedom per node because we vectorized the interpolation above.
dh = DofHandler(grid)
add!(dh, :u, ip)
close!(dh);

# ### Boundary conditions
# Now, we add boundary conditions. We simply support the bottom and the left side and prescribe a displacement upwards on the top edge.
# The last argument to `Dirichlet` determines which components of the field should be constrained.
ch = ConstraintHandler(dh)
add!(ch, Dirichlet(:u, getfaceset(grid, "bottom"), (x, t) -> 0.0, 2))
add!(ch, Dirichlet(:u, getfaceset(grid, "left"), (x, t) -> 0.0, 1))
add!(ch, Dirichlet(:u, getfaceset(grid, "top"), (x, t) -> 0.1, 2))
close!(ch)

# ### Element routine
function assemble_cell!(ke, re, cellvalues, cell, material, ue)
    fill!(ke, 0.0)
    fill!(re, 0.0)

    n_basefuncs = getnbasefunctions(cellvalues)
    reinit!(cellvalues, cell)

    for q_point in 1:getnquadpoints(cellvalues)
        ## For each integration point, compute stress and material stiffness
        ε = function_symmetric_gradient(cellvalues, q_point, ue) # Total strain
        σ, ∂σ∂ε = material_routine(ε, material)

        dΩ = getdetJdV(cellvalues, q_point)
        for i in 1:n_basefuncs
            ∇Nᵢ = shape_gradient(cellvalues, q_point, i)# shape_symmetric_gradient(cellvalues, q_point, i)
            re[i] += σ ⊡ ∇Nᵢ * dΩ # add internal force to residual
            for j in 1:n_basefuncs
                ∇ˢʸᵐNⱼ = shape_symmetric_gradient(cellvalues, q_point, j)
                ke[i, j] += ∇Nᵢ ⊡ ∂σ∂ε ⊡ ∇ˢʸᵐNⱼ * dΩ
            end
        end
    end
end


# #### Global assembly
# We define the function `assemble_global` to loop over the elements and do the global
# assembly. The function takes our `cellvalues`, the sparse matrix `K`, and our DofHandler
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

function assemble_global(cellvalues::CellValues, K::SparseMatrixCSC, dh::DofHandler)
    ## Allocate the element stiffness matrix and element force vector
    n_basefuncs = getnbasefunctions(cellvalues)
    ke = zeros(n_basefuncs, n_basefuncs)
    fe = zeros(n_basefuncs)
    ## Allocate global force vector f
    f = zeros(ndofs(dh))
    ## Create an assembler
    assembler = start_assemble(K, f)
    ## Loop over all cels
    for cell in CellIterator(dh)
        ## Reinitialize cellvalues for this cell
        reinit!(cellvalues, cell)
        ## Compute element contribution
        @views ue = a[celldofs(cell)]
        assemble_cell!(ke, re, cellvalues, get_cell_coordinates(cell), material, ue)
        ## Assemble ke and fe into K and f
        assemble!(assembler, celldofs(cell), ke, fe)
    end
    return K, f
end
#md nothing # hide

# ### Solution of the system
# The last step is to solve the system. First we call `assemble_global`
# to obtain the global stiffness matrix `K` and force vector `f`.
K, f = assemble_global(cellvalues, K, dh);

# To account for the boundary conditions we use the `apply!` function.
# This modifies elements in `K` and `f` respectively, such that
# we can get the correct solution vector `u` by using `\`.
apply!(K, f, ch)
u = K \ f;

# ### Exporting to VTK
# To visualize the result we export the grid and our field `u`
# to a VTK-file, which can be viewed in e.g. [ParaView](https://www.paraview.org/).
vtk_grid("heat_equation", dh) do vtk
    vtk_point_data(vtk, dh, u)
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
