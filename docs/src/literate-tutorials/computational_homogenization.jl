# # [Computational homogenization](@id tutorial-computational-homogenization)
#
# ![](computational_homogenization-light.png)
# ![](computational_homogenization-dark.png)
#
# *Figure 1*: von Mises stress in an RVE with 5 stiff inclusions embedded in a softer matrix
# material that is loaded in shear. The problem is solved by using homogeneous Dirichlet
# boundary conditions (left) and (strong) periodic boundary conditions (right).
#
#-
#md # !!! tip
#md #     This example is also available as a Jupyter notebook:
#md #     [`computational_homogenization.ipynb`](@__NBVIEWER_ROOT_URL__/tutorials/computational_homogenization.ipynb).
#-
#
# ## Introduction
#
# In this example we will solve the Representative Volume Element (RVE) problem for
# computational homogenization of linear elasticity and compute the effective/homogenized
# stiffness of an RVE with 5 stiff circular inclusions embedded in a softer matrix material
# (see Figure 1).
#
# It is possible to obtain upper and lower bounds on the stiffness analytically, see for
# example [Rule of mixtures](https://en.wikipedia.org/wiki/Rule_of_mixtures). An upper
# bound is obtained from the Voigt model, where the *strain* is assumed to be the same in
# the two constituents,
#
# ```math
# \mathsf{E}_\mathrm{Voigt} = v_\mathrm{m} \mathsf{E}_\mathrm{m} +
# (1 - v_\mathrm{m}) \mathsf{E}_\mathrm{i}
# ```
#
# where ``v_\mathrm{m}`` is the volume fraction of the matrix material, and where
# ``\mathsf{E}_\mathrm{m}`` and ``\mathsf{E}_\mathrm{i}`` are the individual stiffness for
# the matrix material and the inclusions, respectively. The lower bound is obtained from
# the Reuss model, where the *stress* is assumed to be the same in the two constituents,
#
# ```math
# \mathsf{E}_\mathrm{Reuss} = \left(v_\mathrm{m} \mathsf{E}_\mathrm{m}^{-1} +
# (1 - v_\mathrm{m}) \mathsf{E}_\mathrm{i}^{-1} \right)^{-1}.
# ```
#
# However, neither of these assumptions are, in general, very close to the "truth" which is
# why it is of interest to computationally find the homogenized properties for a given RVE.
#
# The canonical version of the RVE problem can be formulated as follows:
# For given homogenized field ``\bar{\boldsymbol{u}}``, ``\bar{\boldsymbol{\varepsilon}} =
# \boldsymbol{\varepsilon}[\bar{\boldsymbol{u}}]``, find ``\boldsymbol{u} \in
# \mathbb{U}_\Box``, ``\boldsymbol{t} \in \mathbb{T}_\Box`` such that
#
# ```math
# \frac{1}{|\Omega_\Box|} \int_{\Omega_\Box}\boldsymbol{\varepsilon}[\delta\boldsymbol{u}]
# : \mathsf{E} : \boldsymbol{\varepsilon}[\boldsymbol{u}]\ \mathrm{d}\Omega
# - \frac{1}{|\Omega_\Box|} \int_{\Gamma_\Box}\delta \boldsymbol{u} \cdot
# \boldsymbol{t}\ \mathrm{d}\Gamma = 0 \quad
# \forall \delta \boldsymbol{u} \in \mathbb{U}_\Box,\quad (1\mathrm{a})\\
# - \frac{1}{|\Omega_\Box|} \int_{\Gamma_\Box}\delta \boldsymbol{t} \cdot
# \boldsymbol{u}\ \mathrm{d}\Gamma = - \bar{\boldsymbol{\varepsilon}} :
# \left[ \frac{1}{|\Omega_\Box|} \int_{\Gamma_\Box}\delta \boldsymbol{t} \otimes
# [\boldsymbol{x} - \bar{\boldsymbol{x}}]\ \mathrm{d}\Gamma \right]
# \quad \forall \delta \boldsymbol{t} \in \mathbb{T}_\Box, \quad (1\mathrm{b})
# ```
#
# where ``\boldsymbol{u} = \bar{\boldsymbol{\varepsilon}} \cdot [\boldsymbol{x} -
# \bar{\boldsymbol{x}}] + \boldsymbol{u}^\mu``, where ``\Omega_\Box`` and ``|\Omega_\Box|``
# are the domain and volume of the RVE, where ``\Gamma_\Box`` is the boundary, and where
# ``\mathbb{U}_\Box``, ``\mathbb{T}_\Box`` are set of "sufficiently regular" functions
# defined on the RVE.
#
# Equvilantly, it is possible to instead solve for the fluctuation field ``\boldsymbol{u}^\mu``, whereby the problem statement is:
# Find ``\boldsymbol{u}^\mu \in \mathbb{U}_\Box``, ``\boldsymbol{t} \in \mathbb{T}_\Box`` such that
#
# ```math
# \frac{1}{|\Omega_\Box|} \int_{\Omega_\Box}\boldsymbol{\varepsilon}[\delta\boldsymbol{u}^\mu]
# : \mathsf{E} : \boldsymbol{\varepsilon}[\boldsymbol{u}^\mu]\ \mathrm{d}\Omega
# - \frac{1}{|\Omega_\Box|} \int_{\Gamma_\Box}\delta \boldsymbol{u}^\mu \cdot
# \boldsymbol{t}\ \mathrm{d}\Gamma 
# = - \frac{1}{|\Omega_\Box|} \int_{\Omega_\Box}\boldsymbol{\varepsilon}[\delta\boldsymbol{u}^\mu]
# : \mathsf{E} : \bar{\boldsymbol{\varepsilon}}\ \mathrm{d}\Omega \quad
# \forall \delta \boldsymbol{u} \in \mathbb{U}_\Box,\quad (2\mathrm{a})\\
# - \frac{1}{|\Omega_\Box|} \int_{\Gamma_\Box}\delta \boldsymbol{t} \cdot
# \boldsymbol{u}^\mu\ \mathrm{d}\Gamma = 0
# \quad \forall \delta \boldsymbol{t} \in \mathbb{T}_\Box, \quad (2\mathrm{b})
# ```
# which is what we we will solve for in this tutorial.
#
# This system is not solvable without introducing extra restrictions on ``\mathbb{U}_\Box``,
# ``\mathbb{T}_\Box``. In this example we will consider the common cases of Dirichlet
# boundary conditions and (strong) periodic boundary conditions.
#
# **Dirichlet boundary conditions**
#
# We can introduce the more restrictive sets of $\mathbb{U}_\Box^{\mathrm{D},0}$:
#
# $$\mathbb{U}_\Box^{\mathrm{D},0} := \left\{\boldsymbol{u} \in \mathbb{U}_\Box|\ \boldsymbol{u} = \boldsymbol{0}\ \mathrm{on}\ \Gamma_\Box\right\}$$
#
# and use this as trial and test sets to obtain a solvable RVE problem pertaining to
# Dirichlet boundary conditions. Eq. $(2\mathrm{b})$ is trivially fulfilled, and the boundary 
# traction integral in Eq. $(2\mathrm{a})$ vanishes, and we are left with the following problem: 

# Find $\boldsymbol{u}^\mu \in \mathbb{U}_\Box^{\mathrm{D},0}$ that solve
#
# $$\frac{1}{|\Omega_\Box|} \int_{\Omega_\Box}\boldsymbol{\varepsilon}[\delta\boldsymbol{u}] : \mathsf{E} : \boldsymbol{\varepsilon}[\boldsymbol{u}^\mu]\ \mathrm{d}\Omega = - \frac{1}{|\Omega_\Box|} \int_{\Omega_\Box}\boldsymbol{\varepsilon}[\delta\boldsymbol{u}] : \mathsf{E} : \bar{\boldsymbol{\varepsilon}}\ \mathrm{d}\Omega \quad \forall \delta \boldsymbol{u} \in \mathbb{U}_\Box^{\mathrm{D},0}.$$
#
# **Periodic boundary conditions**
#
# The RVE problem pertaining to periodic boundary conditions is obtained by restricting
# $\boldsymbol{u}^\mu$ to be periodic, and $\boldsymbol{t}$ anti-periodic across the
# RVE. Similarly as for Dirichlet boundary conditions, the boundary traction integral 
# vanishes with these restrictions. By substituting the kinematic split and moving the 
# known macroscopic terms to the right-hand side, we are left with the following problem:
#
# Find $\boldsymbol{u}^\mu \in \mathbb{U}_\Box^{\mathrm{P},0}$ such that
#
# $$\frac{1}{|\Omega_\Box|} \int_{\Omega_\Box}\boldsymbol{\varepsilon}[\delta\boldsymbol{u}] : \mathsf{E} : \boldsymbol{\varepsilon}[\boldsymbol{u}^\mu]\ \mathrm{d}\Omega = - \frac{1}{|\Omega_\Box|} \int_{\Omega_\Box}\boldsymbol{\varepsilon}[\delta\boldsymbol{u}] : \mathsf{E} : \bar{\boldsymbol{\varepsilon}}\ \mathrm{d}\Omega \quad \forall \delta \boldsymbol{u} \in \mathbb{U}_\Box^{\mathrm{P},0},$$
#
# where
#
# $$\mathbb{U}_\Box^{\mathrm{P},0} := \left\{\boldsymbol{u} \in \mathbb{U}_\Box| \ \llbracket \boldsymbol{u} \rrbracket_\Box = \boldsymbol{0} \ \mathrm{on}\ \Gamma_\Box^+\right\}$$
#
# where $\llbracket \bullet \rrbracket_\Box = \bullet(\boldsymbol{x}^+) - \bullet(\boldsymbol{x}^-)$ defines the "jump" over the RVE, i.e. the difference between
# the value on the image part $\Gamma_\Box^+$ (coordinate $\boldsymbol{x}^+$) and the
# mirror part $\Gamma_\Box^-$ (coordinate $\boldsymbol{x}^-$) of the boundary.
# To make sure this restriction holds in a strong sense we need a periodic mesh.
#
# Note that it would be possible to solve for the total $\boldsymbol{u}$ directly by
# instead enforcing the jump to be equal to the jump in the macroscopic part,
# $\boldsymbol{u}^\mathrm{M}$, i.e.
#
# $$\llbracket \boldsymbol{u} \rrbracket_\Box = \llbracket \boldsymbol{u}^\mathrm{M} \rrbracket_\Box = \llbracket \bar{\boldsymbol{\varepsilon}} \cdot [\boldsymbol{x} - \bar{\boldsymbol{x}}] \rrbracket_\Box = \bar{\boldsymbol{\varepsilon}} \cdot [\boldsymbol{x}^+ - \boldsymbol{x}^-].$$
#
# **Neumann boundary conditions**
#
# For Neumann (or static) boundary conditions, we make a "weak" assumption on the 
# RVE-boundary tractions. Specifically, we assume the boundary traction $\boldsymbol{t}$ 
# is generated by a uniform macroscopic stress tensor $\bar{\boldsymbol{\sigma}}$, such 
# that $\boldsymbol{t} = \bar{\boldsymbol{\sigma}} \cdot \boldsymbol{n}$ on $\Gamma_\Box$.
# 
# To establish a suitable variational setting, we define the space of admissible
# displacements as the unconstrained space, with the exception that rigid body motions 
# must be restricted (e.g., by pinning a single node) to ensure a unique solution:
#
# $$\mathbb{U}_\Box^{\mathrm{N},0} := \left\{\boldsymbol{u} \in \mathbb{U}_\Box|\ \mathrm{Rigid\ body\ motions\ are\ constrained}\right\}$$
#
# By substituting our traction assumption into the boundary integral of Eq. $(2\mathrm{a})$, 
# we can apply the divergence theorem to rewrite the traction term as a volume integral 
# for any symmetric tensor $\bar{\boldsymbol{\sigma}} \in \mathbb{R}^{3 \times 3}_\mathrm{sym}$:
#
# $$\frac{1}{|\Omega_\Box|} \int_{\Gamma_\Box} (\bar{\boldsymbol{\sigma}} \cdot \boldsymbol{n}) \cdot \delta \boldsymbol{u}\ \mathrm{d}\Gamma = \bar{\boldsymbol{\sigma}} : \left[ \frac{1}{|\Omega_\Box|} \int_{\Omega_\Box} \boldsymbol{\varepsilon}[\delta \boldsymbol{u}]\ \mathrm{d}\Omega \right] \quad \forall \delta \boldsymbol{u} \in \mathbb{U}_\Box^{\mathrm{N},0}$$
#
# Because we are operating under *macroscale strain control* (solving for a prescribed 
# macroscopic strain $\bar{\boldsymbol{\varepsilon}}$), the macroscopic stress $\bar{\boldsymbol{\sigma}}$ 
# is not known upfront. Instead, it becomes an unknown variable in a mixed problem. 
# It acts as a Lagrange multiplier that enforces the kinematic requirement that the 
# volume average of the fluctuation strain $\boldsymbol{\varepsilon}[\boldsymbol{u}^\mu]$ must be zero.
#
# By substituting the traction identity into our fluctuation-based weak form, we 
# arrive at the following mixed problem:
#
# For a given macroscale strain $\bar{\boldsymbol{\varepsilon}} \in \mathbb{R}^{3 \times 3}_\mathrm{sym}$,
# find $\boldsymbol{u}^\mu \in \mathbb{U}_\Box^{\mathrm{N},0}$ and $\bar{\boldsymbol{\sigma}} \in \mathbb{R}^{3 \times 3}_\mathrm{sym}$
# that solve:
#
# $$\frac{1}{|\Omega_\Box|} \int_{\Omega_\Box}\boldsymbol{\varepsilon}[\delta\boldsymbol{u}] : \mathsf{E} : \boldsymbol{\varepsilon}[\boldsymbol{u}^\mu]\ \mathrm{d}\Omega - \bar{\boldsymbol{\sigma}} : \left[ \frac{1}{|\Omega_\Box|} \int_{\Omega_\Box} \boldsymbol{\varepsilon}[\delta \boldsymbol{u}]\ \mathrm{d}\Omega \right] = - \frac{1}{|\Omega_\Box|} \int_{\Omega_\Box}\boldsymbol{\varepsilon}[\delta\boldsymbol{u}] : \mathsf{E} : \bar{\boldsymbol{\varepsilon}}\ \mathrm{d}\Omega \quad \forall \delta \boldsymbol{u} \in \mathbb{U}_\Box^{\mathrm{N},0}$$
#
# $$- \left[ \frac{1}{|\Omega_\Box|} \int_{\Omega_\Box} \boldsymbol{\varepsilon}[\boldsymbol{u}^\mu]\ \mathrm{d}\Omega \right] : \delta\bar{\boldsymbol{\sigma}} = 0 \quad \forall \delta\bar{\boldsymbol{\sigma}} \in \mathbb{R}^{3 \times 3}_\mathrm{sym}$$
#
# To implement this mixed Neumann boundary condition problem in practice, we need to parameterize the unknown 
# macroscopic stress tensor $\bar{\boldsymbol{\sigma}}$. Since $\bar{\boldsymbol{\sigma}}$ is a 
# symmetric second-order tensor, it has three independent components in 2D. 
# We represent it with a vector-valued global Lagrange parameter field $\bar{\boldsymbol{\lambda}} \in \mathbb{R}^3$ 
# through an orthonormal tensor basis:
#
# $$E_1 = \boldsymbol{e}_1 \otimes \boldsymbol{e}_1, \quad E_2 = \boldsymbol{e}_2 \otimes \boldsymbol{e}_2, \quad E_3 = \frac{1}{\sqrt{2}}(\boldsymbol{e}_1 \otimes \boldsymbol{e}_2 + \boldsymbol{e}_2 \otimes \boldsymbol{e}_1),$$
#
# i.e., $\bar{\boldsymbol{\sigma}} = \sum_{\alpha=1}^3 \bar{\lambda}_\alpha E_\alpha$. 
#
# **Homogenization of effective properties**
#
# In general it is necessary to compute the homogenized stress and the stiffness on the fly,
# but since we in this example consider linear elasticity it is possible to compute the
# effective properties once and for all for a given RVE configuration. We do this by
# computing sensitivity fields for every independent strain component (6 in 3D, 3 in 2D).
# Thus, for a 2D problem, as in the implementation below, we compute sensitivities
# ``\hat{\boldsymbol{u}}_{11}``, ``\hat{\boldsymbol{u}}_{22}``, and
# ``\hat{\boldsymbol{u}}_{12} = \hat{\boldsymbol{u}}_{21}`` by using
#
# ```math
# \bar{\boldsymbol{\varepsilon}} = \begin{pmatrix}1 & 0\\ 0 & 0\end{pmatrix}, \quad
# \bar{\boldsymbol{\varepsilon}} = \begin{pmatrix}0 & 0\\ 0 & 1\end{pmatrix}, \quad
# \bar{\boldsymbol{\varepsilon}} = \begin{pmatrix}0 & 0.5\\ 0.5 & 0\end{pmatrix}
# ```
#
# as the input to the RVE problem. When the sensitivities are solved we can compute the
# entries of the homogenized stiffness as follows
#
# ```math
# \mathsf{E}_{ijkl} = \frac{\partial\ \bar{\sigma}_{ij}}{\partial\ \bar{\varepsilon}_{kl}}
# = \bar{\sigma}_{ij}(\hat{\boldsymbol{u}}_{kl}),
# ```
#
# where the homogenized stress, ``\bar{\boldsymbol{\sigma}}(\boldsymbol{u})``, is computed
# as the volume average of the stress in the RVE, i.e.
## **Periodic boundary conditions (fluctuation formulation)**

# For (strong) periodic boundary conditions we also work with the fluctuation unknown
# ``\boldsymbol{u}^\mu``. The periodic formulation requires that the fluctuation field is
# periodic across matching boundary faces, while the traction is anti-periodic. Writing the
# total displacement as before, the variational problem for the fluctuation field reads:
#
# ```math
# \frac{1}{|\Omega_\Box|} \int_{\Omega_\Box} \boldsymbol{\varepsilon}[\delta\boldsymbol{u}^\mu]
# : \mathsf{E} : (\bar{\boldsymbol{\varepsilon}} + \boldsymbol{\varepsilon}[\boldsymbol{u}^\mu])\ \mathrm{d}\Omega = 0
# \quad \forall \delta\boldsymbol{u}^\mu \in \mathbb{U}_\Box^{\mathrm{P},0},
# ```
#
# with the periodic fluctuation space
#
# ```math
# \mathbb{U}_\Box^{\mathrm{P},0} := \left\{\boldsymbol{v} \in \mathbb{U}_\Box\ \middle|\ \llbracket \boldsymbol{v} \rrbracket_\Box = \boldsymbol{0}\ \mathrm{on}\ \Gamma_\Box^+\right\}.
# ```
#
# In the implementation, enforcing these constraints in a strong sense requires a periodic
# mesh and identifying pairs of boundary dofs (mirror ↔ image). The macroscopic strain
# ``\bar{\boldsymbol{\varepsilon}}`` again appears as a known contribution in the local
# element integrals and in the right-hand side when assembling the system for
# ``\boldsymbol{u}^\mu``.
#
# ```math
# \bar{\boldsymbol{\sigma}}(\boldsymbol{u}) :=
# \frac{1}{|\Omega_\Box|} \int_{\Omega_\Box} \boldsymbol{\sigma}\ \mathrm{d}\Omega =
# \frac{1}{|\Omega_\Box|} \int_{\Omega_\Box}
# \mathsf{E} : \boldsymbol{\varepsilon}[\boldsymbol{u}]\ \mathrm{d}\Omega.
# ```


# ## Commented program
#
# Now we will see how this can be implemented in Ferrite. What follows is a program
# with comments in between which describe the different steps.
#md # You can also find the same program without comments at the end of the page,
#md # see [Plain program](@ref homogenization-plain-program).

using Ferrite, SparseArrays, LinearAlgebra
using Test #src

# We first load the mesh file `"periodic-rve.msh"` (or `"periodic-rve-coarse.msh"`
# for a coarser mesh). The mesh is generated with [Gmsh](https://gmsh.info/),
# and we read it in as a Ferrite `Grid` using
# the [FerriteGmsh.jl](https://github.com/Ferrite-FEM/FerriteGmsh.jl) package:

using FerriteGmsh
using Downloads: Downloads

meshfile = "periodic-rve.msh" #!nb
#src notebook: use coarse mesh to decrease build time
#src   script: use the fine mesh
#src markdown: use the coarse mesh to decrease build time, but make it look like the fine
#md meshfile = "periodic-rve-coarse.msh" #hide
#nb meshfile = "periodic-rve-coarse.msh"
isfile(meshfile) || Downloads.download(Ferrite.asset_url(meshfile), meshfile)

grid = togrid(meshfile)

# We manually add a vertex set with a corner node 
#TODO: add this in the meshfile?
corner_min, corner_max = Ferrite.bounding_box(grid)
addvertexset!(grid, "min_corner", x -> x ≈ corner_min)
addvertexset!(grid, "max_corner", x -> x ≈ corner_max)
# Next we construct the interpolation and quadrature rule, and combining them into
# cellvalues as usual:

dim = 2
ip = Lagrange{RefTriangle, 1}()^dim
qr = QuadratureRule{RefTriangle}(2)
cellvalues = CellValues(qr, ip);

# We define a dof handler with a displacement field `:u`:
dh = DofHandler(grid)
add!(dh, :u, ip)
add!(dh_neumann, :λ, Ferrite.SystemVariable{SymmetricTensor{2,2,Float64}}())
close!(dh);

#For Neumann boundary conditions, we also have to add a system variable for σ-bar
dh_neumann = DofHandler(grid)
add!(dh_neumann, :u, ip)
add!(dh_neumann, :λ, Ferrite.SystemVariable{SymmetricTensor{2,2,Float64}}())
close!(dh_neumann);

dofhandlers = (dirichlet = dh, periodic = dh, neumann = dh_neumann);

# Now we need to define boundary conditions. As discussed earlier we will solve the problem
# using (i) homogeneous Dirichlet boundary conditions, and (ii) periodic Dirichlet boundary
# conditions. We construct two different constraint handlers, one for each case. The
# [`Dirichlet`](@ref) boundary condition we have seen in many other examples. Here we simply
# define the condition that the field, `:u`, should have both components prescribed to `0`
# on the full boundary:

ch_dirichlet = ConstraintHandler(dofhandlers.dirichlet)
dirichlet = Dirichlet(
    :u,
    union(getfacetset.(Ref(grid), ["left", "right", "top", "bottom"])...),
    (x, t) -> [0, 0],
    [1, 2]
)
add!(ch_dirichlet, dirichlet)
close!(ch_dirichlet)

# For periodic boundary conditions we use the [`PeriodicDirichlet`](@ref) constraint type,
# which is very similar to the `Dirichlet` type, but instead of a passing a facetset we pass
# a vector with "facet pairs", i.e. the mapping between mirror and image parts of the
# boundary. In this example the `"left"` and `"bottom"` boundaries are mirrors, and the
# `"right"` and `"top"` boundaries are the images.

ch_periodic = ConstraintHandler(dofhandlers.periodic);
periodic = PeriodicDirichlet(
    :u,
    ["left" => "right", "bottom" => "top"],
    [1, 2]
)
add!(ch_periodic, periodic)
close!(ch_periodic)

# For Neumann boundary conditions, we need to remove rigid body motion. We do this by...
ch_neumann = ConstraintHandler(dofhandlers.neumann);
neumann_bc1 = Dirichlet(:u, getvertexset(grid, "min_corner"), (x,t) -> Vec((0.0,0.0)))
neumann_bc2 = Dirichlet(:u, getvertexset(grid, "max_corner"), (x,t) -> 0.0, [1])
add!(ch_neumann, neumann_bc1)
add!(ch_neumann, neumann_bc2)
close!(ch_neumann)

# This will now constrain any degrees of freedom located on the mirror boundaries to
# the matching degree of freedom on the image boundaries. Internally this will create
# a number of `AffineConstraint`s of the form `u_i = 1 * u_j + 0`:
# ```julia
# a = AffineConstraint(u_m, [u_i => 1], 0)
# ```
# where `u_m` is the degree of freedom on the mirror and `u_i` the matching one on the
# image part. `PeriodicDirichlet` is thus simply just a more convenient way of
# constructing such affine constraints since it computes the degree of freedom mapping
# automatically.
#
# To simplify things we group the constraint handlers into a named tuple

ch = (dirichlet = ch_dirichlet, periodic = ch_periodic, neumann = ch_neumann);

# We can now construct the sparse matrix. Note that, since we are using affine constraints,
# which need to modify the matrix sparsity pattern in order to account for the constraint
# equations, we construct the matrix for the periodic case by passing both the dof handler
# and the constraint handler.

K_dirichlet = allocate_matrix(dofhandlers.dirichlet)
K_periodic = allocate_matrix(dofhandlers.periodic, ch.periodic)

sparsity = init_sparsity_pattern(dofhandlers.neumann)
add_sparsity_entries!(sparsity, dofhandlers.neumann)
add_system_variable_entires!(sparsity, dofhandlers.neumann, 1:getncells(grid), :λ)
K_neumann = allocate_matrix(sparsity)

K = (
    dirichlet = K_dirichlet,
    periodic = K_periodic,
    neumann = K_neumann
);


# We define the fourth order elasticity tensor for the matrix material, and define the
# inclusions to have 10 times higher stiffness

λ, μ = 1.0e10, 7.0e9 # Lamé parameters
δ(i, j) = i == j ? 1.0 : 0.0
Em = SymmetricTensor{4, 2}(
    (i, j, k, l) -> λ * δ(i, j) * δ(k, l) + μ * (δ(i, k) * δ(j, l) + δ(i, l) * δ(j, k))
)
Ei = 10 * Em;

# As mentioned above, in order to compute the apparent/homogenized stiffness we will solve
# the problem repeatedly with different macroscale strain tensors to compute the sensitivity
# of the homogenized stress, ``\bar{\boldsymbol{\sigma}}``, w.r.t. the macroscopic strain,
# ``\bar{\boldsymbol{\varepsilon}}``. The corresponding unit strains are defined below,
# and will result in three different right-hand-sides:

εᴹ = [
    SymmetricTensor{2, 2}([1.0 0.0; 0.0 0.0]), # ε_11 loading
    SymmetricTensor{2, 2}([0.0 0.0; 0.0 1.0]), # ε_22 loading
    SymmetricTensor{2, 2}([0.0 0.5; 0.5 0.0]), # ε_12/ε_21 loading
];

# The assembly function is nothing strange, and in particular there is no impact from the
# choice of boundary conditions, so the same function can be used for both cases. Since
# we want to solve the system 3 times, once for each macroscopic strain component, we
# assemble 3 right-hand-sides.

function assemble_kuu!(cellvalues::CellValues, K::SparseMatrixCSC, dh::DofHandler, εᴹ)

    n_basefuncs = getnbasefunctions(cellvalues)
    ndpc = ndofs_per_cell(dh)
    Ke = zeros(ndpc, ndpc)
    fe = zeros(ndpc, length(εᴹ))
    f = zeros(ndofs(dh), length(εᴹ))
    assembler = start_assemble(K)

    for cell in CellIterator(dh)

        E = cellid(cell) in getcellset(dh.grid, "inclusions") ? Ei : Em
        reinit!(cellvalues, cell)
        fill!(Ke, 0)
        fill!(fe, 0)

        for q_point in 1:getnquadpoints(cellvalues)
            dΩ = getdetJdV(cellvalues, q_point)
            for i in 1:n_basefuncs
                δεi = shape_symmetric_gradient(cellvalues, q_point, i)
                for j in 1:n_basefuncs
                    δεj = shape_symmetric_gradient(cellvalues, q_point, j)
                    Ke[i, j] += (δεi ⊡ E ⊡ δεj) * dΩ
                end
                for (rhs, ε) in enumerate(εᴹ)
                    σᴹ = E ⊡ ε
                    fe[i, rhs] += (- δεi ⊡ σᴹ) * dΩ
                end
            end
        end

        cdofs = celldofs(cell)
        assemble!(assembler, cdofs, Ke)
        f[cdofs, :] .+= fe
    end
    return f
end;

# For the problem statement with Neumann boundary conditions, we must additionally assemble 
# the off-diagonal coupling submatrix corresponding. As discussed, the macroscopic stress tensor $\bar{\boldsymbol{\sigma}}$ is parameterized using three global 
# Lagrange multipliers representing the coordinates along the orthonormal tensor basis $\boldsymbol{E}_\alpha$. Programmatically, we handle this by constructing a `CellValues` container using a `GlobalConstant` 
# vector interpolation. This interpolation functions like standard CellValues, returning 
# constant unit vectors $\boldsymbol{e}_\alpha \in \mathbb{R}^3$ across each cell. Since Ferrite currently does not have Tensorial interpolation, we then map these 
# vector components to their corresponding 2D symmetric tensor basis $\boldsymbol{E}_\alpha$ via `basis_to_tensor`. With this construct, we can implement the assembly function to closely match the mathematical notation.
function assemble_kuσ!(cv_u::CellValues, K::SparseMatrixCSC, dh::DofHandler)

    #Create the cellvalues for the lagrange basis.
    #TODO: Wait for tensor valued interpolations.
    EBASIS = (
        SymmetricTensor{2, 2}((1.0, 0.0, 0.0)),      # E₁ = e₁ ⊗ e₁
        SymmetricTensor{2, 2}((0.0, 0.0, 1.0)),      # E₂ = e₂ ⊗ e₂
        SymmetricTensor{2, 2}((0.0, 1 / √2, 0.0)),   # E₃ = (e₁ ⊗ e₂ + e₂ ⊗ e₁)/√2
    )
    basis_to_tensor(e::Vec{3}) = sum(e[α] * EBASIS[α] for α in 1:3)
    tensor_to_basis(ε::SymmetricTensor{2, 2}) = Vec{3}(α -> ε ⊡ EBASIS[α]);
    cv_σ = CellValues(qr, GlobalConstant{RefTriangle}()^3, Lagrange{RefTriangle,1}());

    #Get the dof indices for the lagrange parameters
    λdofs = Ferrite.system_variable_dofs(dh, :λ)
    ndpc = ndofs_per_cell(dh)
    n_basefuncs = getnbasefunctions(cellvalues)
    nλdofs = length(λdofs) 
    Ke = zeros(ndpc, nλdofs)
    assembler = start_assemble(K, fillzero=false)

    for cell in CellIterator(dh)
        reinit!(cv_u, cell)
        fill!(Ke, 0.0)
        
        for q_point in 1:getnquadpoints(cv_u)
            dΩ = getdetJdV(cv_u, q_point)
            for i in 1:n_basefuncs
                δεi = shape_symmetric_gradient(cv_u, q_point, i)
                for j in 1:nλdofs
                    δσj = shape_value(cv_σ, q_point, j) |> basis_to_tensor
                    Ke[i, j] += (δεi ⊡ δσj) * dΩ
                end
            end
        end
        cdofs = celldofs(cell)
        assemble!(assembler, cdofs, λdofs, Ke)
        assemble!(assembler, λdofs, cdofs, Ke')
    end
end;
# We can now assemble the system. The assembly function modifies the matrix in-place, but
# return the right hand side(s) which we collect in another named tuple.

f_dirichlet = assemble_kuu!(cellvalues, K.dirichlet, dofhandlers.dirichlet, εᴹ)
f_periodic = assemble_kuu!(cellvalues, K.periodic, dofhandlers.periodic, εᴹ)
f_neumann = assemble_kuu!(cellvalues, K.neumann, dofhandlers.neumann, εᴹ)
assemble_kuσ!(cellvalues, K.neumann, dofhandlers.neumann)

rhs = (
    dirichlet = f_dirichlet,
    periodic = f_periodic,
    neumann = f_neumann
);

# The next step is to solve the systems. Since application of boundary conditions, using
# the [`apply!`](@ref) function, modifies both the matrix and the right hand sides we can
# not use it directly in this case since we want to reuse the matrix again for the next
# right hand sides. We could of course re-assemble the matrix for every right hand side,
# but that would not be very efficient. Instead we will use the [`get_rhs_data`](@ref)
# function, together with [`apply_rhs!`](@ref) in a later step. This will extract the
# necessary data from the matrix such that we can apply it for all the different right
# hand sides. Note that we call `apply!` with just the matrix and no right hand side.

rhsdata = (
    dirichlet = get_rhs_data(ch.dirichlet, K.dirichlet),
    periodic = get_rhs_data(ch.periodic, K.periodic),
    neumann = get_rhs_data(ch.neumann, K.neumann),
)

apply!(K.dirichlet, ch.dirichlet)
apply!(K.periodic, ch.periodic)
apply!(K.neumann, ch.neumann)

# We can now solve the problem(s). Note that we only use `apply_rhs!` in the loops below.
# The boundary conditions are already applied to the matrix above, so we only need to
# modify the right hand side.

u = (
    dirichlet = Vector{Float64}[],
    periodic = Vector{Float64}[],
    neumann = Vector{Float64}[],
)

for i in 1:size(rhs.dirichlet, 2)
    rhs_i = @view rhs.dirichlet[:, i]                  # Extract this RHS
    apply_rhs!(rhsdata.dirichlet, rhs_i, ch.dirichlet) # Apply BC
    u_i = cholesky(Symmetric(K.dirichlet)) \ rhs_i     # Solve
    apply!(u_i, ch.dirichlet)                          # Apply BC on the solution
    push!(u.dirichlet, u_i)                            # Save the solution vector
end

for i in 1:size(rhs.periodic, 2)
    rhs_i = @view rhs.periodic[:, i]                   # Extract this RHS
    apply_rhs!(rhsdata.periodic, rhs_i, ch.periodic)   # Apply BC
    u_i = cholesky(Symmetric(K.periodic)) \ rhs_i      # Solve
    apply!(u_i, ch.periodic)                           # Apply BC on the solution
    push!(u.periodic, u_i)                             # Save the solution vector
end

for i in 1:size(rhs.neumann, 2)
    rhs_i = @view rhs.neumann[:, i]                   # Extract this RHS
    apply_rhs!(rhsdata.neumann, rhs_i, ch.neumann)    # Apply BC
    u_i = K.neumann \ rhs_i      # Solve
    apply!(u_i, ch.neumann)                           # Apply BC on the solution
    push!(u.neumann, u_i)                             # Save the solution vector
end

# When the solution(s) are known we can compute the averaged stress,
# ``\bar{\boldsymbol{\sigma}}`` in the RVE. We define a function that does this, and also
# returns the von Mises stress in every quadrature point for visualization.

function compute_stress(cellvalues::CellValues, dh::DofHandler, u, εᴹ)
    σvM_qpdata = zeros(getnquadpoints(cellvalues), getncells(dh.grid))
    σ̄Ω = zero(SymmetricTensor{2, 2})
    Ω = 0.0 # Total volume
    for cell in CellIterator(dh)
        E = cellid(cell) in getcellset(dh.grid, "inclusions") ? Ei : Em
        reinit!(cellvalues, cell)
        for q_point in 1:getnquadpoints(cellvalues)
            dΩ = getdetJdV(cellvalues, q_point)
            εμ = function_symmetric_gradient(cellvalues, q_point, u[celldofs(cell)])
            σ = E ⊡ (εᴹ + εμ)
            σvM_qpdata[q_point, cellid(cell)] = sqrt(3 / 2 * dev(σ) ⊡ dev(σ))
            Ω += dΩ # Update total volume
            σ̄Ω += σ * dΩ # Update integrated stress
        end
    end
    σ̄ = σ̄Ω / Ω
    return σvM_qpdata, σ̄
end;

# We now compute the homogenized stress and von Mises stress for all cases

σ̄ = (
    dirichlet = SymmetricTensor{2, 2}[],
    periodic = SymmetricTensor{2, 2}[],
    neumann = SymmetricTensor{2, 2}[]
)
σ = (
    dirichlet = Vector{Float64}[],
    periodic = Vector{Float64}[],
    neumann = Vector{Float64}[]
)

projector = L2Projector(ip, grid)

for i in 1:3
    σ_qp, σ̄_i = compute_stress(cellvalues, dofhandlers.dirichlet, u.dirichlet[i], εᴹ[i])
    proj = project(projector, σ_qp, qr)
    push!(σ.dirichlet, proj)
    push!(σ̄.dirichlet, σ̄_i)
end

for i in 1:3
    σ_qp, σ̄_i = compute_stress(cellvalues, dofhandlers.periodic, u.periodic[i], εᴹ[i])
    proj = project(projector, σ_qp, qr)
    push!(σ.periodic, proj)
    push!(σ̄.periodic, σ̄_i)
end

for i in 1:3
    σ_qp, σ̄_i = compute_stress(cellvalues, dofhandlers.neumann, u.neumann[i], εᴹ[i])
    proj = project(projector, σ_qp, qr)
    push!(σ.neumann, proj)
    push!(σ̄.neumann, σ̄_i)
end

# The remaining thing is to compute the homogenized stiffness. As mentioned in the
# introduction we can find all the components from the average stress of the sensitivity
# fields that we have solved for
#
# ```math
# \mathsf{E}_{ijkl} = \bar{\sigma}_{ij}(\hat{\boldsymbol{u}}_{kl}).
# ```
#
# So we have now already computed all the components, and just need to gather the data in
# a fourth order tensor:

E_dirichlet = SymmetricTensor{4, 2}() do i, j, k, l
    if k == l == 1
        σ̄.dirichlet[1][i, j] # ∂σ∂ε_**11
    elseif k == l == 2
        σ̄.dirichlet[2][i, j] # ∂σ∂ε_**22
    else
        σ̄.dirichlet[3][i, j] # ∂σ∂ε_**12 and ∂σ∂ε_**21
    end
end

E_periodic = SymmetricTensor{4, 2}() do i, j, k, l
    if k == l == 1
        σ̄.periodic[1][i, j]
    elseif k == l == 2
        σ̄.periodic[2][i, j]
    else
        σ̄.periodic[3][i, j]
    end
end

E_neumann = SymmetricTensor{4, 2}() do i, j, k, l
    if k == l == 1
        σ̄.neumann[1][i, j]
    elseif k == l == 2
        σ̄.neumann[2][i, j]
    else
        σ̄.neumann[3][i, j]
    end
end

# We can check that the results are what we expect, namely that the stiffness with Dirichlet
# boundary conditions is higher than when using periodic boundary conditions, and that
# the Reuss assumption is a lower bound, and the Voigt assumption an upper bound. We first
# compute the volume fraction of the matrix, and then the Voigt and Reuss bounds:

function matrix_volume_fraction(grid, cellvalues)
    V = 0.0 # Total volume
    Vm = 0.0 # Volume of the matrix
    for c in CellIterator(grid)
        reinit!(cellvalues, c)
        is_matrix = !(cellid(c) in getcellset(grid, "inclusions"))
        for qp in 1:getnquadpoints(cellvalues)
            dΩ = getdetJdV(cellvalues, qp)
            V += dΩ
            if is_matrix
                Vm += dΩ
            end
        end
    end
    return Vm / V
end

vm = matrix_volume_fraction(grid, cellvalues)
#-
E_voigt = vm * Em + (1 - vm) * Ei
E_reuss = inv(vm * inv(Em) + (1 - vm) * inv(Ei));

# We can now compare the different computed stiffness tensors. We expect
# ``E_\mathrm{Reuss} \leq E_\mathrm{PeriodicBC} \leq E_\mathrm{DirichletBC} \leq
# E_\mathrm{Voigt}``. A simple thing to compare are the eigenvalues of the tensors. Here
# we look at the first eigenvalue:

ev = (first ∘ eigvals).((E_reuss, E_neumann, E_periodic, E_dirichlet, E_voigt))
@test issorted(ev) #src
round.(ev; digits = -8)

# Finally, we export the solution and the stress field to a VTK file. For the export we
# also compute the macroscopic part of the displacement.

uM = zeros(ndofs(dh))

VTKGridFile("homogenization", dh) do vtk
    for i in 1:3
        ## Compute macroscopic solution
        apply_analytical!(uM, dh, :u, x -> εᴹ[i] ⋅ x)
        ## Dirichlet
        write_solution(vtk, dh, uM + u.dirichlet[i], "_dirichlet_$i")
        write_projection(vtk, projector, σ.dirichlet[i], "σvM_dirichlet_$i")
        ## Periodic
        write_solution(vtk, dh, uM + u.periodic[i], "_periodic_$i")
        write_projection(vtk, projector, σ.periodic[i], "σvM_periodic_$i")
        ## Neumann. Note, we are only interseted in the resulting displacment, not the lagrange parameters.
        write_solution(vtk, dh, uM + u.neumann[i][1:ndofs(dh)], "_neumann$i")
        write_projection(vtk, projector, σ.neumann[i], "σvM_neumann_$i")
    end
end;

# Just another way to compute the stiffness for testing purposes               #src
function homogenize_test(u::Matrix, dh, cv, E_incl, E_mat)                     #src
    ĒΩ = zero(SymmetricTensor{4, 2})                                           #src
    Ω = 0.0                                                                    #src
    ue = zeros(ndofs_per_cell(dh), 3)                                          #src
    for cell in CellIterator(dh)                                               #src
        reinit!(cv, cell)                                                      #src
        for (localdof, globaldof) in enumerate(celldofs(cell))                 #src
            for i in 1:3                                                       #src
                ue[localdof, i] = u[globaldof, i]                              #src
            end                                                                #src
        end                                                                    #src
        E = cellid(cell) in getcellset(dh.grid, "inclusions") ? E_incl : E_mat #src
        for qp in 1:getnquadpoints(cv)                                         #src
            dΩ = getdetJdV(cv, qp)                                             #src
            Ω += dΩ                                                            #src
            ## compute u^ij and u^kl                                           #src
            Ē′ = SymmetricTensor{4, 2}() do i, j, k, l                         #src
                ij = i == j == 1 ? 1 : i == j == 2 ? 2 : 3                     #src
                kl = k == l == 1 ? 1 : k == l == 2 ? 2 : 3                     #src
                εij = function_symmetric_gradient(cv, qp, view(ue, :, ij)) +   #src
                    symmetric((basevec(Vec{2}, i) ⊗ basevec(Vec{2}, j)))       #src
                εkl = function_symmetric_gradient(cv, qp, view(ue, :, kl)) +   #src
                    symmetric((basevec(Vec{2}, k) ⊗ basevec(Vec{2}, l)))       #src
                return (εij ⊡ E ⊡ εkl) * dΩ                                    #src
            end                                                                #src
            ĒΩ += Ē′                                                           #src
        end                                                                    #src
    end                                                                        #src
    return ĒΩ / Ω                                                              #src
end                                                                            #src

@test homogenize_test(reduce(hcat, u.dirichlet), dh, cellvalues, Ei, Em) ≈ E_dirichlet #src
@test homogenize_test(reduce(hcat, u.periodic), dh, cellvalues, Ei, Em) ≈ E_periodic #src
@test homogenize_test(reduce(hcat, u.neumann), dh, cellvalues, Ei, Em) ≈ E_neumann #src

#md # ## [Plain program](@id homogenization-plain-program)
#md #
#md # Here follows a version of the program without any comments.
#md # The file is also available here:
#md # [`computational_homogenization.jl`](computational_homogenization.jl).
#md #
#md # ```julia
#md # @__CODE__
#md # ```
