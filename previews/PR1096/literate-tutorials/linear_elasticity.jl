# # [Linear elasticity](@id tutorial-linear-elasticity)
#
# ![](linear_elasticity.svg)
#
# *Figure 1*: Linear elastically deformed 1mm $\times$ 1mm Ferrite logo.
#
#md # !!! tip
#md #     This tutorial is also available as a Jupyter notebook:
#md #     [`linear_elasticity.ipynb`](@__NBVIEWER_ROOT_URL__/tutorials/linear_elasticity.ipynb).
#-
#
# ## Introduction
#
# The classical first finite element problem to solve in solid mechanics is a linear balance
# of momentum problem. We will use this to introduce a vector valued field, the displacements
# $\boldsymbol{u}(\boldsymbol{x})$. In addition, some features of the
# [`Tensors.jl`](https://github.com/Ferrite-FEM/Tensors.jl) toolbox are demonstrated.
#
# ### Strong form
# The strong form of the balance of momentum for quasi-static loading is given by
# ```math
# \begin{alignat*}{2}
#   \mathrm{div}(\boldsymbol{\sigma}) + \boldsymbol{b} &= 0 \quad &&\boldsymbol{x} \in \Omega \\
#   \boldsymbol{u} &= \boldsymbol{u}_\mathrm{D} \quad &&\boldsymbol{x} \in \Gamma_\mathrm{D} \\
#   \boldsymbol{n} \cdot \boldsymbol{\sigma} &= \boldsymbol{t}_\mathrm{N} \quad &&\boldsymbol{x} \in \Gamma_\mathrm{N}
# \end{alignat*}
# ```
# where $\boldsymbol{\sigma}$ is the (Cauchy) stress tensor and $\boldsymbol{b}$ the body force.
# The domain, $\Omega$, has the boundary $\Gamma$, consisting of a Dirichlet part, $\Gamma_\mathrm{D}$, and
# a Neumann part, $\Gamma_\mathrm{N}$, with outward pointing normal vector $\boldsymbol{n}$.
# $\boldsymbol{u}_\mathrm{D}$ denotes prescribed displacements on $\Gamma_\mathrm{D}$,
# while $\boldsymbol{t}_\mathrm{N}$ the known tractions on $\Gamma_\mathrm{N}$.
#
# In this tutorial, we use linear elasticity, such that
# ```math
# \boldsymbol{\sigma} = \mathsf{C} : \boldsymbol{\varepsilon}, \quad
# \boldsymbol{\varepsilon} = \left[\mathrm{grad}(\boldsymbol{u})\right]^\mathrm{sym}
# ```
# where $\mathsf{C}$ is the 4th order elastic stiffness tensor and
# $\boldsymbol{\varepsilon}$ the small strain tensor.
# The colon, $:$, represents the double contraction,
# $\sigma_{ij} = \mathsf{C}_{ijkl} \varepsilon_{kl}$, and the superscript $\mathrm{sym}$
# denotes the symmetric part.
#
# ### Weak form
# The resulting weak form is given given as follows: Find ``\boldsymbol{u} \in \mathbb{U}`` such that
# ```math
# \int_\Omega
#   \mathrm{grad}(\delta \boldsymbol{u}) : \boldsymbol{\sigma}
# \, \mathrm{d}\Omega
# =
# \int_{\Gamma}
#   \delta \boldsymbol{u} \cdot \boldsymbol{t}
# \, \mathrm{d}\Gamma
# +
# \int_\Omega
#   \delta \boldsymbol{u} \cdot \boldsymbol{b}
# \, \mathrm{d}\Omega
# \quad \forall \, \delta \boldsymbol{u} \in \mathbb{T},
# ```
# where $\mathbb{U}$ and $\mathbb{T}$ denote suitable trial and test function spaces.
# $\delta \boldsymbol{u}$ is a vector valued test function and
# $\boldsymbol{t} = \boldsymbol{\sigma}\cdot\boldsymbol{n}$ is the traction vector on
# the boundary. In this tutorial, we will neglect body forces (i.e. $\boldsymbol{b} = \boldsymbol{0}$) and the weak form reduces to
# ```math
# \int_\Omega
#   \mathrm{grad}(\delta \boldsymbol{u}) : \boldsymbol{\sigma}
# \, \mathrm{d}\Omega
# =
# \int_{\Gamma}
#   \delta \boldsymbol{u} \cdot \boldsymbol{t}
# \, \mathrm{d}\Gamma \,.
# ```
#
# ### Finite element form
# Finally, the finite element form is obtained by introducing the finite element shape functions.
# Since the displacement field, $\boldsymbol{u}$, is vector valued, we use vector valued shape functions
# $\delta\boldsymbol{N}_i$ and $\boldsymbol{N}_i$ to approximate the test and trial functions:
# ```math
# \boldsymbol{u} \approx \sum_{i=1}^N \boldsymbol{N}_i (\boldsymbol{x}) \, \hat{u}_i
# \qquad
# \delta \boldsymbol{u} \approx \sum_{i=1}^N \delta\boldsymbol{N}_i (\boldsymbol{x}) \, \delta \hat{u}_i
# ```
# Here $N$ is the number of nodal variables, with $\hat{u}_i$ and $\delta\hat{u}_i$ representing the $i$-th nodal value.
# Using the Einstein summation convention, we can write this in short form as
# $\boldsymbol{u} \approx \boldsymbol{N}_i \, \hat{u}_i$ and $\delta\boldsymbol{u} \approx \delta\boldsymbol{N}_i \, \delta\hat{u}_i$.
#
# Inserting the these into the weak form, and noting that that the equation should hold for all $\delta \hat{u}_i$, we get
# ```math
# \underbrace{\int_\Omega \mathrm{grad}(\delta \boldsymbol{N}_i) : \boldsymbol{\sigma}\ \mathrm{d}\Omega}_{f_i^\mathrm{int}} = \underbrace{\int_\Gamma \delta \boldsymbol{N}_i \cdot \boldsymbol{t}\ \mathrm{d}\Gamma}_{f_i^\mathrm{ext}}
# ```
# Inserting the linear constitutive relationship, $\boldsymbol{\sigma} = \mathsf{C}:\boldsymbol{\varepsilon}$,
# in the internal force vector, $f_i^\mathrm{int}$, yields the linear equation
# ```math
# \underbrace{\left[\int_\Omega \mathrm{grad}(\delta \boldsymbol{N}_i) : \mathsf{C} : \left[\mathrm{grad}(\boldsymbol{N}_j)\right]^\mathrm{sym}\ \mathrm{d}\Omega\right]}_{K_{ij}}\ \hat{u}_j = f_i^\mathrm{ext}
# ```
#
# ## Implementation
# First we load Ferrite, and some other packages we need.
using Ferrite, FerriteGmsh, SparseArrays
# As in the heat equation tutorial, we will use a unit square - but here we'll load the grid of the Ferrite logo!
# This is done by downloading [`logo.geo`](logo.geo) and loading it using [FerriteGmsh.jl](https://github.com/Ferrite-FEM/FerriteGmsh.jl),
using Downloads: download
logo_mesh = "logo.geo"
asset_url = "https://raw.githubusercontent.com/Ferrite-FEM/Ferrite.jl/gh-pages/assets/"
isfile(logo_mesh) || download(string(asset_url, logo_mesh), logo_mesh)

FerriteGmsh.Gmsh.initialize() #hide
FerriteGmsh.Gmsh.gmsh.option.set_number("General.Verbosity", 2) #hide
grid = togrid(logo_mesh);
FerriteGmsh.Gmsh.finalize(); #hide
#md nothing #hide
# The generated grid lacks the facetsets for the boundaries, so we add them by using Ferrite's
# [`addfacetset!`](@ref). It allows us to add facetsets to the grid based on coordinates.
# Note that approximate comparison to 0.0 doesn't work well, so we use a tolerance instead.
addfacetset!(grid, "top", x -> x[2] ≈ 1.0) # facets for which x[2] ≈ 1.0 for all nodes
addfacetset!(grid, "left", x -> abs(x[1]) < 1.0e-6)
addfacetset!(grid, "bottom", x -> abs(x[2]) < 1.0e-6);

# ### Trial and test functions
# In this tutorial, we use the same linear Lagrange shape functions to approximate both the
# test and trial spaces, i.e. $\delta\boldsymbol{N}_i = \boldsymbol{N}_i$.
# As our grid is composed of triangular elements, we need the Lagrange functions defined
# on a `RefTriangle`. All currently available interpolations can be found under
# [`Interpolation`](@ref reference-interpolation).
#
# Here we use linear triangular elements (also called constant strain triangles).
# The vector valued shape functions are constructed by raising the interpolation
# to the power `dim` (the dimension) since the displacement field has one component in each
# spatial dimension.
dim = 2
order = 1 # linear interpolation
ip = Lagrange{RefTriangle, order}()^dim; # vector valued interpolation

# In order to evaluate the integrals, we need to specify the quadrature rules to use. Due to the
# linear interpolation, a single quadrature point suffices, both inside the cell and on the facet.
# In 2d, a facet is the edge of the element.
qr = QuadratureRule{RefTriangle}(1) # 1 quadrature point
qr_face = FacetQuadratureRule{RefTriangle}(1);

# Finally, we collect the interpolations and quadrature rules into the `CellValues` and `FacetValues`
# buffers, which we will later use to evaluate the integrals over the cells and facets.
cellvalues = CellValues(qr, ip)
facetvalues = FacetValues(qr_face, ip);

# ### Degrees of freedom
# For distributing degrees of freedom, we define a `DofHandler`. The `DofHandler` knows that
# `u` has two degrees of freedom per node because we vectorized the interpolation above.
dh = DofHandler(grid)
add!(dh, :u, ip)
close!(dh);

# ### Boundary conditions
# We set Dirichlet boundary conditions by fixing the motion normal to the bottom and left
# boundaries. The last argument to `Dirichlet` determines which components of the field should be
# constrained. If no argument is given, all components are constrained by default.
ch = ConstraintHandler(dh)
add!(ch, Dirichlet(:u, getfacetset(grid, "bottom"), (x, t) -> 0.0, 2))
add!(ch, Dirichlet(:u, getfacetset(grid, "left"), (x, t) -> 0.0, 1))
close!(ch);

# In addition, we will use Neumann boundary conditions on the top surface, where
# we add a traction vector of the form
# ```math
# \boldsymbol{t}_\mathrm{N}(\boldsymbol{x}) = (20e3) x_1 \boldsymbol{e}_2\ \mathrm{N}/\mathrm{mm}^2
# ```
traction(x) = Vec(0.0, 20.0e3 * x[1]);

# On the right boundary, we don't do anything, resulting in a zero traction Neumann boundary.
# In order to assemble the external forces, $f_i^\mathrm{ext}$, we need to iterate over all
# facets in the relevant facetset. We do this by using the [`FacetIterator`](@ref).

function assemble_external_forces!(f_ext, dh, facetset, facetvalues, prescribed_traction)
    ## Create a temporary array for the facet's local contributions to the external force vector
    fe_ext = zeros(getnbasefunctions(facetvalues))
    for facet in FacetIterator(dh, facetset)
        ## Update the facetvalues to the correct facet number
        reinit!(facetvalues, facet)
        ## Reset the temporary array for the next facet
        fill!(fe_ext, 0.0)
        ## Access the cell's coordinates
        cell_coordinates = getcoordinates(facet)
        for qp in 1:getnquadpoints(facetvalues)
            ## Calculate the global coordinate of the quadrature point.
            x = spatial_coordinate(facetvalues, qp, cell_coordinates)
            tₚ = prescribed_traction(x)
            ## Get the integration weight for the current quadrature point.
            dΓ = getdetJdV(facetvalues, qp)
            for i in 1:getnbasefunctions(facetvalues)
                Nᵢ = shape_value(facetvalues, qp, i)
                fe_ext[i] += tₚ ⋅ Nᵢ * dΓ
            end
        end
        ## Add the local contributions to the correct indices in the global external force vector
        assemble!(f_ext, celldofs(facet), fe_ext)
    end
    return f_ext
end
#md nothing #hide

# ### Material behavior
# Next, we need to define the material behavior, specifically the elastic stiffness tensor, $\mathsf{C}$.
# In this tutorial, we use plane strain linear isotropic elasticity, with Hooke's law as
# ```math
# \boldsymbol{\sigma} = 2G \boldsymbol{\varepsilon}^\mathrm{dev} + 3K \boldsymbol{\varepsilon}^\mathrm{vol}
# ```
# where $G$ is the shear modulus and $K$ the bulk modulus. This expression can be written as
# $\boldsymbol{\sigma} = \mathsf{C}:\boldsymbol{\varepsilon}$, with
# ```math
#  \mathsf{C} := \frac{\partial \boldsymbol{\sigma}}{\partial \boldsymbol{\varepsilon}}
# ```
# The volumetric, $\boldsymbol{\varepsilon}^\mathrm{vol}$,
# and deviatoric, $\boldsymbol{\varepsilon}^\mathrm{dev}$ strains, are defined as
# ```math
# \begin{align*}
# \boldsymbol{\varepsilon}^\mathrm{vol} &= \frac{\mathrm{tr}(\boldsymbol{\varepsilon})}{3}\boldsymbol{I}, \quad
# \boldsymbol{\varepsilon}^\mathrm{dev} &= \boldsymbol{\varepsilon} - \boldsymbol{\varepsilon}^\mathrm{vol}
# \end{align*}
# ```
#
# Starting from Young's modulus, $E$, and Poisson's ratio, $\nu$, the shear and bulk modulus are
# ```math
# G = \frac{E}{2(1 + \nu)}, \quad K = \frac{E}{3(1 - 2\nu)}
# ```
Emod = 200.0e3 # Young's modulus [MPa]
ν = 0.3      # Poisson's ratio [-]

Gmod = Emod / (2(1 + ν))  # Shear modulus
Kmod = Emod / (3(1 - 2ν)) # Bulk modulus

# Finally, we demonstrate `Tensors.jl`'s automatic differentiation capabilities when
# calculating the elastic stiffness tensor
C = gradient(ϵ -> 2 * Gmod * dev(ϵ) + 3 * Kmod * vol(ϵ), zero(SymmetricTensor{2, 2}));

#md # !!! details "Plane stress instead of plane strain?"
#md #     In order to change this tutorial to consider plane stress instead of plane strain,
#md #     the elastic stiffness tensor should be changed to reflect this. The plane stress elasticity
#md #     stiffness matrix in Voigt notation for engineering shear strains, is given as
#md #     ```math
#md #     \underline{\underline{\boldsymbol{E}}} = \frac{E}{1 - \nu^2}\begin{bmatrix}
#md #     1 & \nu & 0 \\
#md #     \nu & 1 & 0 \\
#md #     0 & 0 & (1 - \nu)/2
#md #     \end{bmatrix}
#md #     ```
#md #     This matrix can be converted into the 4th order elastic stiffness tensor as
#md #     ```julia
#md #     C_voigt = Emod * [1.0 ν 0.0; ν 1.0 0.0; 0.0 0.0 (1-ν)/2] / (1 - ν^2)
#md #     C = fromvoigt(SymmetricTensor{4,2}, E_voigt)
#md #     ```
#md #
# ### Element routine
# To calculate the global stiffness matrix, $K_{ij}$, the element routine computes the
# local stiffness matrix `ke` for a single element and assembles it into the global matrix.
# `ke` is pre-allocated and reused for all elements.
#
# Note that the elastic stiffness tensor $\mathsf{C}$ is constant.
# Thus is needs to be computed and once and can then be used for all integration points.
function assemble_cell!(ke, cellvalues, C)
    for q_point in 1:getnquadpoints(cellvalues)
        ## Get the integration weight for the quadrature point
        dΩ = getdetJdV(cellvalues, q_point)
        for i in 1:getnbasefunctions(cellvalues)
            ## Gradient of the test function
            ∇Nᵢ = shape_gradient(cellvalues, q_point, i)
            for j in 1:getnbasefunctions(cellvalues)
                ## Symmetric gradient of the trial function
                ∇ˢʸᵐNⱼ = shape_symmetric_gradient(cellvalues, q_point, j)
                ke[i, j] += (∇Nᵢ ⊡ C ⊡ ∇ˢʸᵐNⱼ) * dΩ
            end
        end
    end
    return ke
end
#md nothing #hide
#
# ### Global assembly
# We define the function `assemble_global` to loop over the elements and do the global
# assembly. The function takes the preallocated sparse matrix `K`, our DofHandler `dh`, our
# `cellvalues` and the elastic stiffness tensor `C` as input arguments and computes the
# global stiffness matrix `K`.
function assemble_global!(K, dh, cellvalues, C)
    ## Allocate the element stiffness matrix
    n_basefuncs = getnbasefunctions(cellvalues)
    ke = zeros(n_basefuncs, n_basefuncs)
    ## Create an assembler
    assembler = start_assemble(K)
    ## Loop over all cells
    for cell in CellIterator(dh)
        ## Update the shape function gradients based on the cell coordinates
        reinit!(cellvalues, cell)
        ## Reset the element stiffness matrix
        fill!(ke, 0.0)
        ## Compute element contribution
        assemble_cell!(ke, cellvalues, C)
        ## Assemble ke into K
        assemble!(assembler, celldofs(cell), ke)
    end
    return K
end
#md nothing #hide

# ### Solution of the system
# The last step is to solve the system. First we allocate the global stiffness matrix `K`
# and assemble it.
K = allocate_matrix(dh)
assemble_global!(K, dh, cellvalues, C);
# Then we allocate and assemble the external force vector.
f_ext = zeros(ndofs(dh))
assemble_external_forces!(f_ext, dh, getfacetset(grid, "top"), facetvalues, traction);

# To account for the Dirichlet boundary conditions we use the `apply!` function.
# This modifies elements in `K` and `f`, such that we can get the
# correct solution vector `u` by using solving the linear equation system $K_{ij} \hat{u}_j = f^\mathrm{ext}_i$,
apply!(K, f_ext, ch)
u = K \ f_ext;

# ### Postprocessing
# In this case, we want to analyze the displacements, as well as the stress field.
# We calculate the stress in each quadrature point, and then export it in two different
# ways:
# 1) Constant in each cell (matching the approximation of constant strains in each element).
#    Note that a current limitation is that cell data for second order tensors must be exported
#    component-wise (see issue #768)
# 2) Interpolated using the linear lagrange ansatz functions via the [`L2Projector`](@ref).

function calculate_stresses(grid, dh, cv, u, C)
    qp_stresses = [
        [zero(SymmetricTensor{2, 2}) for _ in 1:getnquadpoints(cv)]
            for _ in 1:getncells(grid)
    ]
    avg_cell_stresses = tuple((zeros(getncells(grid)) for _ in 1:3)...)
    for cell in CellIterator(dh)
        reinit!(cv, cell)
        cell_stresses = qp_stresses[cellid(cell)]
        for q_point in 1:getnquadpoints(cv)
            ε = function_symmetric_gradient(cv, q_point, u, celldofs(cell))
            cell_stresses[q_point] = C ⊡ ε
        end
        σ_avg = sum(cell_stresses) / getnquadpoints(cv)
        avg_cell_stresses[1][cellid(cell)] = σ_avg[1, 1]
        avg_cell_stresses[2][cellid(cell)] = σ_avg[2, 2]
        avg_cell_stresses[3][cellid(cell)] = σ_avg[1, 2]
    end
    return qp_stresses, avg_cell_stresses
end

qp_stresses, avg_cell_stresses = calculate_stresses(grid, dh, cellvalues, u, C);

# We now use the the L2Projector to project the stress-field onto the piecewise linear
# finite element space that we used to solve the problem.
proj = L2Projector(Lagrange{RefTriangle, 1}(), grid)
stress_field = project(proj, qp_stresses, qr);

# Export also values for each color in the logo. #src
color_data = zeros(Int, getncells(grid))         #hide
colors = [                                       #hide
    "1" => 1, "5" => 1, # purple                 #hide
    "2" => 2, "3" => 2, # red                    #hide
    "4" => 3,           # blue                   #hide
    "6" => 4,            # green                  #hide
]                                            #hide
for (key, color) in colors                       #hide
    for i in getcellset(grid, key)               #hide
        color_data[i] = color                    #hide
    end                                          #hide
end                                              #hide
#md nothing                                      #hide
#
# To visualize the result we export to a VTK-file. Specifically, an unstructured
# grid file, `.vtu`, is created, which can be viewed in e.g.
# [ParaView](https://www.paraview.org/).
VTKGridFile("linear_elasticity", dh) do vtk
    write_solution(vtk, dh, u)
    for (i, key) in enumerate(("11", "22", "12"))
        write_cell_data(vtk, avg_cell_stresses[i], "sigma_" * key)
    end
    write_projection(vtk, proj, stress_field, "stress field")
    Ferrite.write_cellset(vtk, grid)
    write_cell_data(vtk, color_data, "colors") #hide
end

# We used the displacement field to visualize the deformed logo in *Figure 1*,
# and in *Figure 2*, we demonstrate the difference between the interpolated stress
# field and the constant stress in each cell.

# ![](linear_elasticity_stress.png)
#
# *Figure 2*: Vertical normal stresses (MPa) exported using the `L2Projector` (left)
# and constant stress in each cell (right).

# The mesh produced by gmsh is not stable between different OS        #src
# For linux, we therefore test carefully, and for other OS we provide #src
# a coarse test to indicate introduced errors before running CI.      #src
using Test                                                            #hide
linux_result = 0.31742879147646924                                    #hide
@test abs(norm(u) - linux_result) < 0.01                              #hide
Sys.islinux() && @test norm(u) ≈ linux_result                         #hide
nothing                                                               #hide

#md # ## [Plain program](@id linear_elasticity-plain-program)
#md #
#md # Here follows a version of the program without any comments.
#md # The file is also available here: [`linear_elasticity.jl`](linear_elasticity.jl).
#md #
#md # ```julia
#md # @__CODE__
#md # ```
