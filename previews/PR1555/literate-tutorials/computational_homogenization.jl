# # [Computational homogenization](@id tutorial-computational-homogenization)
#
# ![](computational_homogenization-light.png)
# ![](computational_homogenization-dark.png)
#
# [[*Figure 1*](@ref tutorial-computational-homogenization-figure-1)](@id tutorial-computational-homogenization-figure-1):
# von Mises stress in an RVE with 5 stiff inclusions embedded in a softer matrix
# material that is loaded in shear. The problem is solved by using homogeneous Dirichlet
# boundary conditions (left), (strong) periodic boundary conditions (middle), and Neumann
# boundary conditions (right).
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
# (see [Figure 1](@ref tutorial-computational-homogenization-figure-1)).
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
# where ``\Omega_\Box`` and ``|\Omega_\Box|`` are the domain and volume of the RVE, where
# ``\Gamma_\Box`` is the boundary with outward unit normal ``\boldsymbol{n}``, where
# ``\bar{\boldsymbol{x}}`` is the center of the RVE, and where ``\mathbb{U}_\Box``,
# ``\mathbb{T}_\Box`` are sets of "sufficiently regular" functions defined on the RVE and
# on its boundary, respectively. Eq. ``(1\mathrm{a})`` is the equilibrium equation of the
# RVE, loaded by the boundary traction ``\boldsymbol{t}``, and Eq. ``(1\mathrm{b})``
# weakly enforces that the fluctuation, i.e. the difference between ``\boldsymbol{u}`` and
# the macroscopic displacement ``\bar{\boldsymbol{\varepsilon}} \cdot [\boldsymbol{x} -
# \bar{\boldsymbol{x}}]``, vanishes on the boundary (in the sense of the test tractions
# ``\delta\boldsymbol{t} \in \mathbb{T}_\Box``).
#
# We split the displacement into a macroscopic part and a fluctuation part,
#
# ```math
# \boldsymbol{u} = \boldsymbol{u}^\mathrm{M} + \boldsymbol{u}^\mu =
# \bar{\boldsymbol{\varepsilon}} \cdot [\boldsymbol{x} - \bar{\boldsymbol{x}}] +
# \boldsymbol{u}^\mu,
# \quad \Rightarrow \quad
# \boldsymbol{\varepsilon}[\boldsymbol{u}] = \bar{\boldsymbol{\varepsilon}} +
# \boldsymbol{\varepsilon}[\boldsymbol{u}^\mu].
# ```
#
# Since the macroscopic part is known, the test functions are pure fluctuations,
# ``\delta\boldsymbol{u} = \delta\boldsymbol{u}^\mu``. Inserting the split into
# Eq. ``(1)``, and moving the known macroscopic contributions to the right hand side, we
# obtain the equivalent problem of finding the fluctuation field: For given
# ``\bar{\boldsymbol{\varepsilon}}``, find ``\boldsymbol{u}^\mu \in \mathbb{U}_\Box``,
# ``\boldsymbol{t} \in \mathbb{T}_\Box`` such that
#
# ```math
# \frac{1}{|\Omega_\Box|} \int_{\Omega_\Box}\boldsymbol{\varepsilon}[\delta\boldsymbol{u}]
# : \mathsf{E} : \boldsymbol{\varepsilon}[\boldsymbol{u}^\mu]\ \mathrm{d}\Omega
# - \frac{1}{|\Omega_\Box|} \int_{\Gamma_\Box}\delta \boldsymbol{u} \cdot
# \boldsymbol{t}\ \mathrm{d}\Gamma
# = - \frac{1}{|\Omega_\Box|} \int_{\Omega_\Box}\boldsymbol{\varepsilon}[\delta\boldsymbol{u}]
# : \mathsf{E} : \bar{\boldsymbol{\varepsilon}}\ \mathrm{d}\Omega \quad
# \forall \delta \boldsymbol{u} \in \mathbb{U}_\Box,\quad (2\mathrm{a})\\
# - \frac{1}{|\Omega_\Box|} \int_{\Gamma_\Box}\delta \boldsymbol{t} \cdot
# \boldsymbol{u}^\mu\ \mathrm{d}\Gamma = 0
# \quad \forall \delta \boldsymbol{t} \in \mathbb{T}_\Box. \quad (2\mathrm{b})
# ```
#
# This is the form of the problem that we will solve in this tutorial, i.e. the
# fluctuation field ``\boldsymbol{u}^\mu`` is the primary unknown. The macroscopic strain
# enters as a known "body load" on the right hand side of Eq. ``(2\mathrm{a})``.
#
# The system in Eq. ``(2)`` is not solvable without introducing extra restrictions on
# ``\mathbb{U}_\Box``, ``\mathbb{T}_\Box``. In this example we will consider the common
# cases of Dirichlet boundary conditions, (strong) periodic boundary conditions, and
# Neumann boundary conditions.
#
# **Dirichlet boundary conditions**
#
# We can introduce the more restrictive set of ``\mathbb{U}_\Box``:
#
# ```math
# \mathbb{U}_\Box^{\mathrm{D},0} := \left\{\boldsymbol{u} \in \mathbb{U}_\Box|\
# \boldsymbol{u} = \boldsymbol{0}\ \mathrm{on}\ \Gamma_\Box\right\},
# ```
#
# and use this as trial and test set to obtain a solvable RVE problem pertaining to
# Dirichlet boundary conditions. Eq. ``(2\mathrm{b})`` is trivially fulfilled, the boundary
# traction term in Eq. ``(2\mathrm{a})`` vanishes, and we are left with the following
# problem: Find ``\boldsymbol{u}^\mu \in \mathbb{U}_\Box^{\mathrm{D},0}`` that solve
#
# ```math
# \frac{1}{|\Omega_\Box|} \int_{\Omega_\Box}\boldsymbol{\varepsilon}[\delta\boldsymbol{u}]
# : \mathsf{E} : \boldsymbol{\varepsilon}[\boldsymbol{u}^\mu]\ \mathrm{d}\Omega =
# - \frac{1}{|\Omega_\Box|} \int_{\Omega_\Box}\boldsymbol{\varepsilon}[\delta\boldsymbol{u}]
# : \mathsf{E} : \bar{\boldsymbol{\varepsilon}}\ \mathrm{d}\Omega
# \quad \forall \delta \boldsymbol{u} \in \mathbb{U}_\Box^{\mathrm{D},0}.
# ```
#
# Note that this is equivalent to prescribing the total displacement to
# ``\boldsymbol{u} = \bar{\boldsymbol{\varepsilon}} \cdot [\boldsymbol{x} -
# \bar{\boldsymbol{x}}]`` on ``\Gamma_\Box``.
#
# **Periodic boundary conditions**
#
# The RVE problem pertaining to periodic boundary conditions is obtained by restricting
# ``\boldsymbol{u}^\mu`` to be periodic, and ``\boldsymbol{t}`` anti-periodic across the
# RVE. With these restrictions both boundary integrals in Eq. ``(2)`` vanish: in
# Eq. ``(2\mathrm{a})`` the contributions from the image and the mirror part of the
# boundary cancel since ``\delta\boldsymbol{u}`` is periodic and ``\boldsymbol{t}``
# anti-periodic, and Eq. ``(2\mathrm{b})`` is fulfilled for the same reason since
# ``\boldsymbol{u}^\mu`` is periodic and ``\delta\boldsymbol{t}`` anti-periodic. We are
# left with the following problem: Find
# ``\boldsymbol{u}^\mu \in \mathbb{U}_\Box^{\mathrm{P},0}`` such that
#
# ```math
# \frac{1}{|\Omega_\Box|} \int_{\Omega_\Box}\boldsymbol{\varepsilon}[\delta\boldsymbol{u}]
# : \mathsf{E} : \boldsymbol{\varepsilon}[\boldsymbol{u}^\mu]\ \mathrm{d}\Omega =
# - \frac{1}{|\Omega_\Box|} \int_{\Omega_\Box}\boldsymbol{\varepsilon}[\delta\boldsymbol{u}]
# : \mathsf{E} : \bar{\boldsymbol{\varepsilon}}\ \mathrm{d}\Omega
# \quad \forall \delta \boldsymbol{u} \in \mathbb{U}_\Box^{\mathrm{P},0},
# ```
#
# where
#
# ```math
# \mathbb{U}_\Box^{\mathrm{P},0} := \left\{\boldsymbol{u} \in \mathbb{U}_\Box|
# \ \llbracket \boldsymbol{u} \rrbracket_\Box = \boldsymbol{0}
# \ \mathrm{on}\ \Gamma_\Box^+\right\}
# ```
#
# where ``\llbracket \bullet \rrbracket_\Box = \bullet(\boldsymbol{x}^+) -
# \bullet(\boldsymbol{x}^-)`` defines the "jump" over the RVE, i.e. the difference between
# the value on the image part ``\Gamma_\Box^+`` (coordinate ``\boldsymbol{x}^+``) and the
# mirror part ``\Gamma_\Box^-`` (coordinate ``\boldsymbol{x}^-``) of the boundary.
# To make sure this restriction holds in a strong sense we need a periodic mesh.
# Periodicity determines ``\boldsymbol{u}^\mu`` only up to a constant translation, which
# also has to be removed, see the implementation below.
#
# Note that it would be possible to solve for the total ``\boldsymbol{u}`` directly by
# instead enforcing the jump to be equal to the jump in the macroscopic part,
# ``\boldsymbol{u}^\mathrm{M}``, i.e.
#
# ```math
# \llbracket \boldsymbol{u} \rrbracket_\Box =
# \llbracket \boldsymbol{u}^\mathrm{M} \rrbracket_\Box =
# \llbracket \bar{\boldsymbol{\varepsilon}} \cdot [\boldsymbol{x} - \bar{\boldsymbol{x}}]
# \rrbracket_\Box =
# \bar{\boldsymbol{\varepsilon}} \cdot [\boldsymbol{x}^+ - \boldsymbol{x}^-].
# ```
#
# **Neumann boundary conditions**
#
# For Neumann boundary conditions we instead restrict the tractions: they are assumed to be
# generated by a constant (unknown) macroscopic stress tensor
# ``\bar{\boldsymbol{\sigma}}``, i.e. we restrict ``\mathbb{T}_\Box`` to
#
# ```math
# \mathbb{T}_\Box^\mathrm{N} := \left\{\boldsymbol{t} \in \mathbb{T}_\Box|\
# \boldsymbol{t} = \bar{\boldsymbol{\sigma}} \cdot \boldsymbol{n}\ \mathrm{on}\
# \Gamma_\Box,\ \bar{\boldsymbol{\sigma}} \in \mathbb{R}^{2 \times 2}_\mathrm{sym}\right\},
# ```
#
# where ``\mathbb{R}^{2 \times 2}_\mathrm{sym}`` is the set of symmetric second order
# tensors (in 2D). The displacement is not restricted on the boundary. The traction is
# thus parametrized by the three independent components of ``\bar{\boldsymbol{\sigma}}``
# alone, and, by using the divergence theorem and the symmetry of
# ``\bar{\boldsymbol{\sigma}}``, the boundary integrals in Eq. ``(2)`` can be rewritten as
# volume integrals:
#
# ```math
# \frac{1}{|\Omega_\Box|} \int_{\Gamma_\Box} \boldsymbol{v} \cdot
# (\bar{\boldsymbol{\sigma}} \cdot \boldsymbol{n})\ \mathrm{d}\Gamma =
# \bar{\boldsymbol{\sigma}} : \left[ \frac{1}{|\Omega_\Box|} \int_{\Gamma_\Box}
# \boldsymbol{v} \otimes \boldsymbol{n}\ \mathrm{d}\Gamma \right] =
# \bar{\boldsymbol{\sigma}} : \left[ \frac{1}{|\Omega_\Box|} \int_{\Omega_\Box}
# \boldsymbol{\varepsilon}[\boldsymbol{v}]\ \mathrm{d}\Omega \right],
# ```
#
# for any displacement field ``\boldsymbol{v}``. Using this identity with
# ``\boldsymbol{v} = \delta\boldsymbol{u}`` in Eq. ``(2\mathrm{a})``, and with
# ``\boldsymbol{v} = \boldsymbol{u}^\mu`` (and ``\delta\bar{\boldsymbol{\sigma}}`` in place of
# ``\bar{\boldsymbol{\sigma}}``) in Eq. ``(2\mathrm{b})``, we arrive at the following mixed
# problem: Find ``\boldsymbol{u}^\mu \in \mathbb{U}_\Box^{\mathrm{N},0}`` and
# ``\bar{\boldsymbol{\sigma}} \in \mathbb{R}^{2 \times 2}_\mathrm{sym}`` such that
#
# ```math
# \frac{1}{|\Omega_\Box|} \int_{\Omega_\Box}\boldsymbol{\varepsilon}[\delta\boldsymbol{u}]
# : \mathsf{E} : \boldsymbol{\varepsilon}[\boldsymbol{u}^\mu]\ \mathrm{d}\Omega
# - \bar{\boldsymbol{\sigma}} : \left[ \frac{1}{|\Omega_\Box|} \int_{\Omega_\Box}
# \boldsymbol{\varepsilon}[\delta \boldsymbol{u}]\ \mathrm{d}\Omega \right]
# = - \frac{1}{|\Omega_\Box|} \int_{\Omega_\Box}\boldsymbol{\varepsilon}[\delta\boldsymbol{u}]
# : \mathsf{E} : \bar{\boldsymbol{\varepsilon}}\ \mathrm{d}\Omega
# \quad \forall \delta \boldsymbol{u} \in \mathbb{U}_\Box^{\mathrm{N},0}, \quad (3\mathrm{a})\\
# - \delta\bar{\boldsymbol{\sigma}} : \left[ \frac{1}{|\Omega_\Box|} \int_{\Omega_\Box}
# \boldsymbol{\varepsilon}[\boldsymbol{u}^\mu]\ \mathrm{d}\Omega \right] = 0
# \quad \forall \delta\bar{\boldsymbol{\sigma}} \in \mathbb{R}^{2 \times 2}_\mathrm{sym}.
# \quad (3\mathrm{b})
# ```
#
# Eq. ``(3\mathrm{b})`` states that the volume average of the fluctuation strain is zero,
# i.e. that the volume average of the total strain equals the prescribed macroscopic strain
# ``\bar{\boldsymbol{\varepsilon}}``. The macroscopic stress ``\bar{\boldsymbol{\sigma}}``
# is thus the Lagrange multiplier that enforces this constraint. Note that
# ``\bar{\boldsymbol{\sigma}}`` is an unknown of the problem even though the problem is
# still strain-driven, i.e. ``\bar{\boldsymbol{\varepsilon}}`` is prescribed.
#
# All terms in Eq. ``(3)`` depend on the displacements only through their strains.
# Consequently, ``\boldsymbol{u}^\mu`` is only determined up to a rigid body motion,
# which must be removed in order to obtain a unique solution, i.e. we use
#
# ```math
# \mathbb{U}_\Box^{\mathrm{N},0} := \left\{\boldsymbol{u} \in \mathbb{U}_\Box|\
# \boldsymbol{u}\ \mathrm{contains\ no\ rigid\ body\ motion}\right\}.
# ```
#
# In practice this is realized by prescribing exactly as many point displacement
# components as there are rigid body modes (3 in 2D: two translations and one rotation),
# see the implementation below. For the same reason, a rigid body motion
# ``\delta\boldsymbol{u}`` makes both sides of Eq. ``(3\mathrm{a})`` vanish, so the
# loading is self-equilibrated and these point constraints do not give rise to any
# reaction forces, i.e. they do not affect the strain and stress fields in the RVE.
#
# Finally, since Eq. ``(3\mathrm{a})`` is unaffected by adding a rigid body motion to
# ``\delta\boldsymbol{u}``, it holds for all ``\delta\boldsymbol{u} \in
# \mathbb{U}_\Box``. In particular, choosing ``\delta\boldsymbol{u} =
# \delta\bar{\boldsymbol{\varepsilon}} \cdot [\boldsymbol{x} - \bar{\boldsymbol{x}}]``
# for an arbitrary symmetric ``\delta\bar{\boldsymbol{\varepsilon}}`` shows that
#
# ```math
# \bar{\boldsymbol{\sigma}} = \frac{1}{|\Omega_\Box|} \int_{\Omega_\Box} \mathsf{E} :
# (\bar{\boldsymbol{\varepsilon}} + \boldsymbol{\varepsilon}[\boldsymbol{u}^\mu])
# \ \mathrm{d}\Omega,
# ```
#
# i.e. the Lagrange multiplier is equal to the volume average of the stress in the RVE.
#
# The three types of boundary conditions differ in how much they restrict the fluctuation
# field: Dirichlet boundary conditions are the most restrictive, while Neumann boundary
# conditions are the least restrictive (the only restriction is that the average
# fluctuation strain is zero, which periodic and Dirichlet boundary conditions also
# fulfill). As a consequence, the resulting homogenized stiffness is the highest for
# Dirichlet, and the lowest for Neumann boundary conditions.
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
#
# ```math
# \bar{\boldsymbol{\sigma}}(\boldsymbol{u}) :=
# \frac{1}{|\Omega_\Box|} \int_{\Omega_\Box} \boldsymbol{\sigma}\ \mathrm{d}\Omega =
# \frac{1}{|\Omega_\Box|} \int_{\Omega_\Box}
# \mathsf{E} : \boldsymbol{\varepsilon}[\boldsymbol{u}]\ \mathrm{d}\Omega.
# ```
#
# In this tutorial the macroscopic strain ``\bar{\boldsymbol{\varepsilon}}`` is
# *prescribed* -- the RVE problem is strain-driven. The complementary, stress-driven,
# case -- prescribing the average stress and solving for the then unknown
# ``\bar{\boldsymbol{\varepsilon}}`` -- is treated in the
# [stress-driven homogenization](@ref tutorial-stress-driven-homogenization) tutorial,
# which builds on this one.


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

# Next we construct the interpolation and quadrature rule, and combining them into
# cellvalues as usual:

dim = 2
ip = Lagrange{RefTriangle, 1}()^dim
qr = QuadratureRule{RefTriangle}(2)
cellvalues = CellValues(qr, ip);

# For the Neumann problem the macroscopic stress ``\bar{\boldsymbol{\sigma}}`` is an
# additional unknown. It is a single symmetric tensor for the whole RVE, i.e. it has no
# spatial variation and is not attached to the mesh. In Ferrite such unknowns are modeled
# as [algebraic variables](@ref topic-algebraic-variables), declared with an
# [`AlgebraicVariable`](@ref). The corresponding [`AlgebraicValues`](@ref) give access to
# the (constant) basis tensors that we need in the assembly below:

σ̄var = AlgebraicVariable{SymmetricTensor{2, 2}}()
algebraic_values = AlgebraicValues(σ̄var);

# We define a dof handler with a displacement field `:u`:
dh = DofHandler(grid)
add!(dh, :u, ip)
close!(dh);

# For Neumann boundary conditions we need a second dof handler which, in addition to the
# displacement field, also contains the algebraic variable `:σbar`. The dofs of the
# algebraic variable are numbered after all the displacement dofs, so the displacement
# dofs are numbered identically in the two dof handlers.
dh_neumann = DofHandler(grid)
add!(dh_neumann, :u, ip)
add!(dh_neumann, :σbar, σ̄var)
close!(dh_neumann);

dofhandlers = (dirichlet = dh, periodic = dh, neumann = dh_neumann);

# Now we need to define boundary conditions. As discussed earlier we will solve the problem
# using (i) homogeneous Dirichlet boundary conditions, (ii) periodic Dirichlet boundary
# conditions, and (iii) Neumann boundary conditions. We construct three different
# constraint handlers, one for each case. The
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
close!(ch_dirichlet);

# For periodic boundary conditions we use the [`PeriodicDirichlet`](@ref) constraint type,
# which is very similar to the `Dirichlet` type, but instead of a passing a facetset we pass
# a vector with "facet pairs", i.e. the mapping between mirror and image parts of the
# boundary. The facet pairs are computed with [`collect_periodic_facets`](@ref). In this
# example the `"left"` and `"bottom"` boundaries are mirrors, and the `"right"` and `"top"`
# boundaries are the images.

periodic_facets = collect_periodic_facets(grid, "left", "right")
collect_periodic_facets!(periodic_facets, grid, "bottom", "top")
periodic = PeriodicDirichlet(:u, periodic_facets, [1, 2]);

# This will constrain any degrees of freedom located on the mirror boundaries to
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
# The periodic constraints determine the fluctuation field only up to a constant
# translation, so we also need to remove this rigid body motion. Since the periodicity ties
# the four corners of the RVE together we do this by prescribing the fluctuation to zero
# in all of them, using a regular `Dirichlet` constraint on a vertexset with the corners.
# We add it after the `PeriodicDirichlet` constraint, which means that it replaces the
# periodic constraints between the corners, so all corners are simply prescribed to zero.
# Note that it is generally not enough to lock just one of the corners: adding a
# constraint for a degree of freedom that is already constrained replaces the old
# constraint, so either the periodic constraint or the `Dirichlet` constraint of that
# corner would be lost.

addvertexset!(grid, "corners", x -> all(abs.(x) .≈ 0.5))
corners = Dirichlet(:u, getvertexset(grid, "corners"), x -> zero(Vec{2}), [1, 2])

ch_periodic = ConstraintHandler(dofhandlers.periodic)
add!(ch_periodic, periodic)
add!(ch_periodic, corners)
close!(ch_periodic);

# For Neumann boundary conditions the fluctuation field is only determined up to a rigid
# body motion, which we must remove. This is done with three point constraints that remove
# the two translations and the rotation: both displacement components are prescribed to
# zero in the bottom left corner, and the horizontal component is prescribed to zero in the
# top right corner. As discussed in the introduction these constraints only remove the
# rigid body motion, i.e. they do not affect the strain or stress in the RVE.

addvertexset!(grid, "bottom_left", x -> x ≈ Vec((-0.5, -0.5)))
addvertexset!(grid, "top_right", x -> x ≈ Vec((0.5, 0.5)))

ch_neumann = ConstraintHandler(dofhandlers.neumann)
add!(ch_neumann, Dirichlet(:u, getvertexset(grid, "bottom_left"), x -> zero(Vec{2}), [1, 2]))
add!(ch_neumann, Dirichlet(:u, getvertexset(grid, "top_right"), x -> 0.0, [1]))
close!(ch_neumann);

# To simplify things we group the constraint handlers into a named tuple

ch = (dirichlet = ch_dirichlet, periodic = ch_periodic, neumann = ch_neumann);

# We can now construct the sparse matrix. Note that, since we are using affine constraints,
# which need to modify the matrix sparsity pattern in order to account for the constraint
# equations, we construct the matrix for the periodic case by passing both the dof handler
# and the constraint handler. For the Neumann problem we also have to declare where the
# algebraic variable couples to the displacement field, such that the corresponding
# entries are added to the sparsity pattern. This is done with a [`CellCoupling`](@ref):
# ``\bar{\boldsymbol{\sigma}}`` couples to the displacement in every cell of the RVE (the
# tuple form is bidirectional, i.e. both the ``\boldsymbol{u}``-``\bar{\boldsymbol{\sigma}}``
# and the ``\bar{\boldsymbol{\sigma}}``-``\boldsymbol{u}`` blocks are added). There is no
# coupling of ``\bar{\boldsymbol{\sigma}}`` with itself, since that block is zero.

K_dirichlet = allocate_matrix(dofhandlers.dirichlet)
K_periodic = allocate_matrix(dofhandlers.periodic, ch.periodic)
coupling = CellCoupling(1:getncells(grid); algebraic_coupling = (:u, :σbar))
K_neumann = allocate_matrix(dofhandlers.neumann; algebraic_couplings = (coupling,))

K = (
    dirichlet = K_dirichlet,
    periodic = K_periodic,
    neumann = K_neumann,
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
# choice of boundary conditions, so the same function can be used for all cases. Since
# we want to solve the system 3 times, once for each macroscopic strain component, we
# assemble 3 right-hand-sides.

function assemble_kuu!(cellvalues::CellValues, K::SparseMatrixCSC, dh::DofHandler, εᴹ, Ei, Em)

    n_basefuncs = getnbasefunctions(cellvalues)
    ndpc = ndofs_per_cell(dh)
    Ke = zeros(ndpc, ndpc)
    fe = zeros(ndpc, length(εᴹ))
    f = zeros(ndofs(dh), length(εᴹ))
    assembler = start_assemble(K)
    inclusions = getcellset(dh.grid, "inclusions")

    for cell in CellIterator(dh)

        E = cellid(cell) in inclusions ? Ei : Em
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

# For the problem with Neumann boundary conditions we must additionally assemble the
# coupling blocks between the displacement and the macroscopic stress. The algebraic
# variable `AlgebraicVariable{SymmetricTensor{2, 2}}` has one dof per independent
# component, and the dof values ``\bar{s}_\alpha`` represent the tensor in the (constant)
# basis
# ```math
# \boldsymbol{E}_1 = \boldsymbol{e}_1 \otimes \boldsymbol{e}_1, \quad
# \boldsymbol{E}_2 = \boldsymbol{e}_1 \otimes \boldsymbol{e}_2 +
# \boldsymbol{e}_2 \otimes \boldsymbol{e}_1, \quad
# \boldsymbol{E}_3 = \boldsymbol{e}_2 \otimes \boldsymbol{e}_2,
# ```
# i.e. ``\bar{\boldsymbol{\sigma}} = \sum_{\alpha=1}^3 \bar{s}_\alpha
# \boldsymbol{E}_\alpha``. The basis tensors are obtained with
# [`algebraic_basis_value`](@ref) and play the role of (constant) shape functions for
# ``\bar{\boldsymbol{\sigma}}``. Inserting the discretizations in the weak form gives the
# coupling block
# ```math
# (K^e_{u\sigma})_{i\alpha} = - \int_{\Omega^e} \boldsymbol{\varepsilon}[\boldsymbol{N}_i]
# : \boldsymbol{E}_\alpha\ \mathrm{d}\Omega,
# ```
# and, since the problem is symmetric, ``\boldsymbol{K}^e_{\sigma u} =
# (\boldsymbol{K}^e_{u\sigma})^\mathrm{T}``. (We have multiplied both equations by
# ``|\Omega_\Box|``, which does not change the solution.) The algebraic dofs are global,
# i.e. shared by all cells, so we query them once outside of the loop with
# [`algebraic_dofs`](@ref) and assemble the two rectangular blocks with the
# `assemble!(assembler, rowdofs, coldofs, Ke)` method. Note that we pass `fillzero = false`
# to [`start_assemble`](@ref) since the ``\boldsymbol{K}_{uu}`` block is already assembled
# into the matrix.
function assemble_kuσ!(cv_u::CellValues, av_σ::AlgebraicValues, K::SparseMatrixCSC, dh::DofHandler)
    σdofs = algebraic_dofs(dh, :σbar)
    n_basefuncs = getnbasefunctions(cv_u)
    nσ = getnbasefunctions(av_σ)
    Ke = zeros(n_basefuncs, nσ)
    assembler = start_assemble(K; fillzero = false)

    for cell in CellIterator(dh)
        reinit!(cv_u, cell)
        fill!(Ke, 0)

        for q_point in 1:getnquadpoints(cv_u)
            dΩ = getdetJdV(cv_u, q_point)
            for i in 1:n_basefuncs
                δεi = shape_symmetric_gradient(cv_u, q_point, i)
                for α in 1:nσ
                    Eα = algebraic_basis_value(av_σ, α)
                    Ke[i, α] -= (δεi ⊡ Eα) * dΩ
                end
            end
        end
        cdofs = celldofs(cell)
        assemble!(assembler, cdofs, σdofs, Ke)
        assemble!(assembler, σdofs, cdofs, transpose(Ke))
    end
    return K
end;

# We can now assemble the system. The assembly function modifies the matrix in-place, but
# return the right hand side(s) which we collect in another named tuple.

f_dirichlet = assemble_kuu!(cellvalues, K.dirichlet, dofhandlers.dirichlet, εᴹ, Ei, Em)
f_periodic = assemble_kuu!(cellvalues, K.periodic, dofhandlers.periodic, εᴹ, Ei, Em)
f_neumann = assemble_kuu!(cellvalues, K.neumann, dofhandlers.neumann, εᴹ, Ei, Em)
assemble_kuσ!(cellvalues, algebraic_values, K.neumann, dofhandlers.neumann)

rhs = (
    dirichlet = f_dirichlet,
    periodic = f_periodic,
    neumann = f_neumann,
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
# modify the right hand side. Note that the matrix for the Neumann problem is symmetric but
# indefinite (it has a saddle point structure, with a zero block for
# ``\bar{\boldsymbol{\sigma}}``), so we can not use a Cholesky factorization for it.

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
    apply_rhs!(rhsdata.neumann, rhs_i, ch.neumann)     # Apply BC
    u_i = K.neumann \ rhs_i                            # Solve
    apply!(u_i, ch.neumann)                            # Apply BC on the solution
    push!(u.neumann, u_i)                              # Save the solution vector
end

# When the solution(s) are known we can compute the averaged stress,
# ``\bar{\boldsymbol{\sigma}}`` in the RVE. We define a function that does this, and also
# returns the von Mises stress in every quadrature point for visualization.

function compute_stress(cellvalues::CellValues, dh::DofHandler, u, εᴹ, Ei, Em)
    σvM_qpdata = zeros(getnquadpoints(cellvalues), getncells(dh.grid))
    σ̄Ω = zero(SymmetricTensor{2, 2})
    Ω = 0.0 # Total volume
    inclusions = getcellset(dh.grid, "inclusions")
    ue = zeros(eltype(u), ndofs_per_cell(dh))
    for cell in CellIterator(dh)
        E = cellid(cell) in inclusions ? Ei : Em
        reinit!(cellvalues, cell)
        ue .= @view u[celldofs(cell)]
        for q_point in 1:getnquadpoints(cellvalues)
            dΩ = getdetJdV(cellvalues, q_point)
            εμ = function_symmetric_gradient(cellvalues, q_point, ue)
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
    neumann = SymmetricTensor{2, 2}[],
)
σ = (
    dirichlet = Vector{Float64}[],
    periodic = Vector{Float64}[],
    neumann = Vector{Float64}[],
)

projector = L2Projector(ip, grid)

for i in 1:3
    σ_qp, σ̄_i = compute_stress(cellvalues, dofhandlers.dirichlet, u.dirichlet[i], εᴹ[i], Ei, Em)
    proj = project(projector, σ_qp, qr)
    push!(σ.dirichlet, proj)
    push!(σ̄.dirichlet, σ̄_i)
end

for i in 1:3
    σ_qp, σ̄_i = compute_stress(cellvalues, dofhandlers.periodic, u.periodic[i], εᴹ[i], Ei, Em)
    proj = project(projector, σ_qp, qr)
    push!(σ.periodic, proj)
    push!(σ̄.periodic, σ̄_i)
end

for i in 1:3
    σ_qp, σ̄_i = compute_stress(cellvalues, dofhandlers.neumann, u.neumann[i], εᴹ[i], Ei, Em)
    proj = project(projector, σ_qp, qr)
    push!(σ.neumann, proj)
    push!(σ̄.neumann, σ̄_i)
end

# For the Neumann problem the macroscopic stress is also part of the solution vector, since
# it is the Lagrange multiplier ``\bar{\boldsymbol{\sigma}}``. As shown in the
# introduction it is equal to the volume average of the stress, which we can verify by
# extracting it from the solution with [`algebraic_value`](@ref):

σ̄_multiplier = [algebraic_value(dofhandlers.neumann, u_i, :σbar) for u_i in u.neumann]
@test σ̄_multiplier ≈ σ̄.neumann #src
σ̄_multiplier ≈ σ̄.neumann

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
# boundary conditions is higher than when using periodic boundary conditions, which in turn
# is higher than when using Neumann boundary conditions, and that the Reuss assumption is a
# lower bound, and the Voigt assumption an upper bound. We first
# compute the volume fraction of the matrix, and then the Voigt and Reuss bounds:

function matrix_volume_fraction(grid, cellvalues)
    V = 0.0 # Total volume
    Vm = 0.0 # Volume of the matrix
    inclusions = getcellset(grid, "inclusions")
    for c in CellIterator(grid)
        reinit!(cellvalues, c)
        is_matrix = !(cellid(c) in inclusions)
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
# ``E_\mathrm{Reuss} \leq E_\mathrm{NeumannBC} \leq E_\mathrm{PeriodicBC} \leq
# E_\mathrm{DirichletBC} \leq E_\mathrm{Voigt}``. A simple thing to compare are the eigenvalues of the tensors. Here
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
        ## Neumann (only the displacement dofs, which are numbered first)
        write_solution(vtk, dh, uM + u.neumann[i][1:ndofs(dh)], "_neumann_$i")
        write_projection(vtk, projector, σ.neumann[i], "σvM_neumann_$i")
    end
end;

# Just another way to compute the stiffness for testing purposes               #src
function homogenize_test(u::Matrix, dh, cv, E_incl, E_mat)                     #src
    ĒΩ = zero(SymmetricTensor{4, 2})                                           #src
    Ω = 0.0                                                                    #src
    ue = zeros(ndofs_per_cell(dh), 3)                                          #src
    inclusions = getcellset(dh.grid, "inclusions")                             #src
    for cell in CellIterator(dh)                                               #src
        reinit!(cv, cell)                                                      #src
        for (localdof, globaldof) in enumerate(celldofs(cell))                 #src
            for i in 1:3                                                       #src
                ue[localdof, i] = u[globaldof, i]                              #src
            end                                                                #src
        end                                                                    #src
        E = cellid(cell) in inclusions ? E_incl : E_mat                        #src
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
@test homogenize_test(reduce(hcat, u.neumann), dh_neumann, cellvalues, Ei, Em) ≈ E_neumann #src

#md # ## [Plain program](@id homogenization-plain-program)
#md #
#md # Here follows a version of the program without any comments.
#md # The file is also available here:
#md # [`computational_homogenization.jl`](computational_homogenization.jl).
#md #
#md # ```julia
#md # @__CODE__
#md # ```
