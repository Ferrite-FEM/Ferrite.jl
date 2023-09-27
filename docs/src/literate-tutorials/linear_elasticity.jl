# # [Linear elasticity](@id tutorial-linear-elasticity)
#
# ![](linear_elasticity.png)
#
# *Figure 1*: Linear elastically deformed Ferrite logo.
#
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
# $\mathbb{T}$ are suitable trial and test function sets, respectively. The boundary traction
# is denoted $\boldsymbol{t}^\ast$ and body forces are denoted $\boldsymbol{b}$.
#
# However, for this example we will neglect all external loads and thus the weak form reads:
# ```math
# \int_\Omega 
#   \boldsymbol{\sigma} : \left(\delta \boldsymbol{u} \otimes \boldsymbol{\nabla} \right)
# \, \mathrm{d}V
# =
# 0 \,.
# ```

# First we load Ferrite, and some other packages we need.
using Ferrite, FerriteGmsh, SparseArrays
# Like for the Heat Equation example, we will use a unit square - but here we'll load the grid of the Ferrite logo! This is done by `FerriteGmsh` here. 
grid = togrid("src/literate-tutorials/logo.geo") # TODO: where to store correctly and how to refer to this?
# By default the grid lacks the facesets for the boundaries, so we add them by Ferrite here.
# Note that approximate comparison to 0.0 doesn't work well, so we use a tolerance instead.
addfaceset!(grid, "top", x->x[2] ≈ 1.0)
addfaceset!(grid, "left", x->x[1] < 1e-6) 
addfaceset!(grid, "bottom", x->x[2] < 1e-6);

# ### Trial and test functions
# We use linear Lagrange functions as test and trial functions. The grid is composed of triangular elements, thus we need the Lagrange functions defined on `RefTriangle`. All currently available interpolations can be found under [`Interpolation`](@ref).
#
# Since the displacement field $\boldsymbol{u}$ is vector valued, we use vector valued shape functions $\boldsymbol{N}_i$ to approximate the test and trial functions:
# ```math
# \boldsymbol{u} \approx \sum_{i=1}^N \boldsymbol{N}_i \left(\boldsymbol{x}\right) \, \hat{u}_i
# \qquad
# \delta \boldsymbol{u} \approx \sum_{i=1}^N \boldsymbol{N}_i \left(\boldsymbol{x}\right) \, \delta \hat{u}_i
# ```
# Here $N$ is the number of nodal variables and $\hat{u}_i$ / $\delta\hat{u}_i$ represent the i-th nodal value.
# Using the Einstein summation convention, we can write this in short form as
# $\boldsymbol{u} \approx \boldsymbol{N}_i \, \hat{u}_i$ and $\delta\boldsymbol{u} \approx \boldsymbol{N}_i \, \delta\hat{u}_i$.
# 
# Here we use linear triangular elements (also called constant strain triangles) with a single
# quadrature point.
# The vector valued shape functions are constructed by raising the interpolation
# to the power `dim` (the dimension) since the displacement field has one component in each
# spatial dimension.
dim = 2
order = 1 # linear interpolation
ip = Lagrange{RefTriangle, order}()^dim # vector valued interpolation
qr = QuadratureRule{RefTriangle}(1) # 1 quadrature point
cellvalues = CellValues(qr, ip);

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
close!(ch);

# ### Material routine
# For the sake of structuring the program, we introduce a material routine here.
# We define a data structure that holds the material constants, here it stores the
# shear modulus $G$ and the bulk modulus $K$. The material routine then computes the stress
# $\boldsymbol{\sigma}$ and its tangent $\frac{\partial\boldsymbol{\sigma}}{\partial\boldsymbol{\varepsilon}}$
# based on the input strain $\boldsymbol{\varepsilon}$. Here, we use automatic differentiation
# for computing the stress tangent. You can read more about automatic differentiation on tensor
# operations in the [`Tensors.jl` docs](https://ferrite-fem.github.io/Tensors.jl/stable/man/automatic_differentiation/).
struct Elasticity
    G::Float64
    K::Float64
end

function material_routine(material::Elasticity, ε::SymmetricTensor{2})
    (; G, K) = material
    stress(ε) = 2G * dev(ε) + K * tr(ε) * one(ε)
    ∂σ∂ε, σ = gradient(stress, ε, :all)
    return σ, ∂σ∂ε
end

E = 200e3 # Young's modulus [MPa]
ν = 0.3 # Poisson's ratio [-]
material = Elasticity(E/2(1+ν), E/3(1-2ν));

# ### Element routine
# The residual vector and stiffness matrix follow from the weak form such that
# ```math
# \left(\underline{R}\right)_i
# =
# \int_\Omega
#   \boldsymbol{\sigma} : \boldsymbol{\nabla} \boldsymbol{N}_i
# \, \mathrm{d}V
# \,, \qquad
# \left(\underline{\underline{K}}\right)_{ij}
# =
# \int_\Omega
#   \left(
#       \frac{\partial \boldsymbol{\sigma}}{\partial \boldsymbol{\varepsilon}}
#       :
#       \boldsymbol{\nabla}^\mathrm{sym} \boldsymbol{N}_j
#   \right)
#   :
#   \boldsymbol{\nabla} \boldsymbol{N}_i
# \, \mathrm{d}V
# ```
# The element routine computes the local residual vector `re` and stiffness matrix `ke`
# for a single element. `ke` and `re` are pre-allocated and reused for all elements.
function assemble_cell!(ke, fe, cellvalues, material, ue)
    fill!(ke, 0.0)
    fill!(fe, 0.0)

    n_basefuncs = getnbasefunctions(cellvalues)
    for q_point in 1:getnquadpoints(cellvalues)
        ## For each integration point, compute strain, stress and material stiffness
        ε = function_symmetric_gradient(cellvalues, q_point, ue)
        σ, ∂σ∂ε = material_routine(material, ε)

        dΩ = getdetJdV(cellvalues, q_point)
        for i in 1:n_basefuncs
            ∇Nᵢ = shape_gradient(cellvalues, q_point, i)# shape_symmetric_gradient(cellvalues, q_point, i)
            fe[i] += σ ⊡ ∇Nᵢ * dΩ # add internal force to residual
            for j in 1:n_basefuncs
                ∇ˢʸᵐNⱼ = shape_symmetric_gradient(cellvalues, q_point, j)
                ke[i, j] += (∂σ∂ε ⊡ ∇ˢʸᵐNⱼ) ⊡ ∇Nᵢ * dΩ
            end
        end
    end
end
#md nothing # hide

# #### Global assembly
# We define the function `assemble_global` to loop over the elements and do the global
# assembly. The function takes the sparse matrix `K`, a guess of the displacement vector
# (this would be more relevant for non-linear problems), our DofHandler, our `cellvalues` and
# the material definition that we want to use for the stress computation
# as input arguments and fills the assembled global stiffness matrix, and the assembled
# global force vector. 
function assemble_global!(K, f, a, dh, cellvalues, material)
    ## Allocate the element stiffness matrix and element force vector
    n_basefuncs = getnbasefunctions(cellvalues)
    ke = zeros(n_basefuncs, n_basefuncs)
    fe = zeros(n_basefuncs)
    ## Create an assembler
    assembler = start_assemble(K, f)
    ## Loop over all cells
    for cell in CellIterator(dh)
        reinit!(cellvalues, cell) # update spatial derivatives based on element coordinates
        @views ue = a[celldofs(cell)]
        ## Compute element contribution
        assemble_cell!(ke, fe, cellvalues, material, ue)
        ## Assemble ke and fe into K and f
        assemble!(assembler, celldofs(cell), ke, fe)
    end
    return K, f
end
#md nothing # hide

# ### Solution of the system
# The last step is to solve the system. First we call `assemble_global`
# to obtain the global stiffness matrix `K` and force vector `f`.
K = create_sparsity_pattern(dh)
f = zeros(ndofs(dh))
a = zeros(ndofs(dh))
assemble_global!(K, f, a, dh, cellvalues, material);

# To account for the Dirichlet boundary conditions we use the `apply!` function.
# This modifies elements in `K` and `f` respectively, such that
# we can get the correct solution vector `u` by using `\`.
apply!(K, f, ch)
u = K \ f;

# ### Exporting to VTK
# To visualize the result we export the grid and our field `u`
# to a VTK-file, which can be viewed in e.g. [ParaView](https://www.paraview.org/).
vtk_grid("linear_elasticity", dh) do vtk
    vtk_point_data(vtk, dh, u)
    vtk_cellset(vtk, grid)
end
#md nothing # hide

#md # ## [Plain program](@id heat_equation-plain-program)
#md #
#md # Here follows a version of the program without any comments.
#md # The file is also available here: [`heat_equation.jl`](heat_equation.jl).
#md #
#md # ```julia
#md # @__CODE__
#md # ```
