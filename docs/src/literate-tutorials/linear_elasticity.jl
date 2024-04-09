# # [Linear elasticity](@id tutorial-linear-elasticity)
#
# ![](linear_elasticity.png)
#
# *Figure 1*: Linear elastically deformed Ferrite logo.
#
#md # !!! tip
#md #     This example is also available as a Jupyter notebook:
#md #     [`linear_elasticity.ipynb`](@__NBVIEWER_ROOT_URL__/examples/linear_elasticity.ipynb).
#-
#
# ## Introduction
#
# The classical first finite element problem to solve in solid mechanics is a linear balance
# of momentum problem. We will use this to introduce a vector valued field, as well as the
# [`Tensors.jl`](https://github.com/Ferrite-FEM/Tensors.jl) toolbox.
#
# The strong form of the balance of momentum is given by
# ```math
#  -\boldsymbol{\sigma} \cdot \boldsymbol{\nabla} = \boldsymbol{b}  \quad \textbf{x} \in \Omega,
# ```
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
# However, for this example we will neglect body foces and thus the weak form reads:
# ```math
# \int_\Omega 
#   \boldsymbol{\sigma} : \left(\delta \boldsymbol{u} \otimes \boldsymbol{\nabla} \right)
# \, \mathrm{d}V
# =
# \int_{\partial\Omega}
#   \boldsymbol{t}^\ast \cdot \delta \boldsymbol{u}
# \, \mathrm{d}A \,.
# ```
#
# Finally, we choose to operate on a 2-dimensional problem under plain strain conditions.
# First we load Ferrite, and some other packages we need.
using Ferrite, FerriteGmsh, SparseArrays
# Like for the Heat Equation example, we will use a unit square - but here we'll load the grid of the Ferrite logo! This is done by loading [`logo.geo`](logo.geo) with [`FerriteGmsh.jl`](https://github.com/Ferrite-FEM/FerriteGmsh.jl) here.
FerriteGmsh.Gmsh.initialize() # hide
FerriteGmsh.Gmsh.gmsh.option.set_number("General.Verbosity", 2) #hide
grid = togrid("logo.geo");
#md nothing # hide
# By default the grid lacks the facesets for the boundaries, so we add them by Ferrite here.
# [`addfaceset!`](@ref) allows to add facesets to the grid based on coordinates.
# Note that approximate comparison to 0.0 doesn't work well, so we use a tolerance instead.
addfaceset!(grid, "top", x->x[2] ≈ 1.0) # faces for which x[2] ≈ 1.0 for all nodes
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
cellvalues = CellValues(qr, ip)
qr_face = FaceQuadratureRule{RefTriangle}(1)
facevalues = FaceValues(qr_face, ip);

# ### Degrees of freedom
# For distributing degrees of freedom, we define a `DofHandler`. The `DofHandler` knows that
# `u` has two degrees of freedom per node because we vectorized the interpolation above.
dh = DofHandler(grid)
add!(dh, :u, ip)
close!(dh);

# ### Boundary conditions
# Now, we add boundary conditions. We simply support the bottom and the left side and
# prescribe a displacement upwards on the top edge.
# The last argument to `Dirichlet` determines which components of the field should be
# constrained.
ch = ConstraintHandler(dh)
add!(ch, Dirichlet(:u, getfaceset(grid, "bottom"), (x, t) -> 0.0, 2))
add!(ch, Dirichlet(:u, getfaceset(grid, "left"), (x, t) -> 0.0, 1))
#add!(ch, Dirichlet(:u, getfaceset(grid, "top"), (x, t) -> 0.1, 2))
close!(ch);

# ### Material behavior
# Next, we need to define the material behavior. In this example, we use linear elasticity,
# such that
# ```math
# \boldsymbol{\sigma} = \boldsymbol{\mathrm{E}} : \boldsymbol \varepsilon
# ```
# where $\boldsymbol{\mathrm{E}}$ is the elastic stiffness tensor and
# $\boldsymbol{\varepsilon}$ is the small strain tensor.
# Consequently, we need to set up the elastic stiffness tensor $\boldsymbol{\mathrm{E}}$.
#
# Here, we operate in plane strain conditions, keep in mind that the plane stress stiffness 
# tensor is defined differently.
Emod = 200e3 # Young's modulus [MPa]
ν = 0.3 # Poisson's ratio [-]
#md nothing # hide

λ = Emod*ν / ((1 + ν) * (1 - 2ν)) # 1st Lamé parameter
μ = Emod / (2(1 + ν)) # 2nd Lamé parameter
I = one(SymmetricTensor{2, dim}) # 2nd order unit tensor
II = one(SymmetricTensor{4, dim}) # 4th order symmetric unit tensor
𝐄 = 2μ * II + λ * (I ⊗ I); # elastic stiffness tensor

# ### Element routine
# The stiffness matrix follows from the weak form such that
# ```math
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
# The element routine computes the local  stiffness matrix `ke`
# for a single element. `ke` is pre-allocated and reused for all elements.
#
# Note that the elastic stiffness tensor $\boldsymbol{\mathrm{E}}$ is constant.
# Thus is needs to be computed and once and can then be used for all integration points.
function assemble_cell!(ke, cellvalues, ∂σ∂ε)
    fill!(ke, 0.0)
    n_basefuncs = getnbasefunctions(cellvalues)
    for q_point in 1:getnquadpoints(cellvalues)
        dΩ = getdetJdV(cellvalues, q_point)
        for i in 1:n_basefuncs
            ∇Nᵢ = shape_gradient(cellvalues, q_point, i)# shape_symmetric_gradient(cellvalues, q_point, i)
            for j in 1:n_basefuncs
                ∇ˢʸᵐNⱼ = shape_symmetric_gradient(cellvalues, q_point, j)
                ke[i, j] += (∂σ∂ε ⊡ ∇ˢʸᵐNⱼ) ⊡ ∇Nᵢ * dΩ
            end
        end
    end
    return ke
end
#md nothing # hide

# ### Global assembly
# We define the function `assemble_global` to loop over the elements and do the global
# assembly. The function takes the preallocated sparse matrix `K`, our DofHandler `dh`, our
# `cellvalues` and the elastic stiffness tensor `∂σ∂ε` as input arguments and computes the
# global stiffness matrix `K`.
function assemble_global!(K, dh, cellvalues, ∂σ∂ε)
    ## Allocate the element stiffness matrix
    n_basefuncs = getnbasefunctions(cellvalues)
    ke = zeros(n_basefuncs, n_basefuncs)
    ## Create an assembler
    assembler = start_assemble(K)
    ## Loop over all cells
    for cell in CellIterator(dh)
        reinit!(cellvalues, cell) # update spatial derivatives based on element coordinates
        ## Compute element contribution
        assemble_cell!(ke, cellvalues, ∂σ∂ε)
        ## Assemble ke and fe into K and f
        assemble!(assembler, celldofs(cell), ke)
    end
    return K
end
#md nothing # hide

function assemble_external_forces!(f_ext, dh, faceset, facevalues, prescribed_traction)
    fe_ext = zeros(getnbasefunctions(facevalues))
    for face in FaceIterator(dh, faceset)
        reinit!(facevalues, face)
        fill!(fe_ext, 0.0)
        for qp in 1:getnquadpoints(facevalues)
            X = spatial_coordinate(facevalues, qp, getcoordinates(face))
            tₚ = prescribed_traction(X)
            dΓ = getdetJdV(facevalues, qp)
            for i in 1:getnbasefunctions(facevalues)
                Nᵢ = shape_value(facevalues, qp, i)
                fe_ext[i] += tₚ ⋅ Nᵢ * dΓ
            end
        end
        assemble!(f_ext, celldofs(face), fe_ext)
    end
    return f_ext
end
#md nothing # hide

# ### Solution of the system
# The last step is to solve the system. First we allocate the global stiffness matrix `K`
# and assemble it.
K = create_sparsity_pattern(dh)
assemble_global!(K, dh, cellvalues, ∂σ∂ε);
# Then we allocate the external force vector. 
f_ext = zeros(ndofs(dh));
assemble_external_forces!(f_ext, dh, getfaceset(grid, "top"), facevalues, x->Vec(0.0, 20e3*x[1]))

# To account for the Dirichlet boundary conditions we use the `apply!` function.
# This modifies elements in `K` and `f` respectively, such that
# we can get the correct solution vector `u` by using `\`.
apply!(K, f_ext, ch)
u = K \ f_ext;

# ### Exporting to VTK
# To visualize the result we export the grid and our field `u`
# to a VTK-file, which can be viewed in e.g. [ParaView](https://www.paraview.org/).
vtk_grid("linear_elasticity", dh) do vtk
    vtk_point_data(vtk, dh, u)
    vtk_cellset(vtk, grid) # export cellsets of grains for logo-coloring
end
#md nothing # hide

#md # ## [Plain program](@id linear_elasticity-plain-program)
#md #
#md # Here follows a version of the program without any comments.
#md # The file is also available here: [`linear_elasticity.jl`](linear_elasticity).
#md #
#md # ```julia
#md # @__CODE__
#md # ```
