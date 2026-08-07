# ```@meta
# Draft = false
# ```
# # [Magnetostatics with H(curl) elements](@id tutorial-magnetostatics)
#
# ![](magnetostatics-light.png)
# ![](magnetostatics-dark.png)
#
# *Figure 1*: Magnetic vector potential ``A`` (left) and flux density magnitude
# ``|\boldsymbol{B}|`` (right, logarithmic scale) around a current-carrying conductor
# surrounded by an iron core: the core soaks up almost all of the magnetic flux.
#
#-
#md # !!! tip
#md #     This tutorial is also available as a Jupyter notebook:
#md #     [`magnetostatics.ipynb`](@__NBVIEWER_ROOT_URL__/tutorials/magnetostatics.ipynb).
#-
#
# ## Introduction
#
# In this tutorial we solve a magnetostatic problem: a long straight conductor carries a
# current, and an iron core placed around it — the working principle of a current
# transformer — collects the magnetic flux circling the conductor. Instead of the standard
# approach of discretizing a potential and computing the magnetic field by differentiation
# afterwards, we solve for the magnetic field ``\boldsymbol{H}`` directly as an unknown of
# its own. This requires a finite element space designed for magnetic fields — across a
# material interface the *tangential* component of ``\boldsymbol{H}`` must be continuous,
# while its normal component may jump — which is exactly what the
# ``H(\mathrm{curl})``-conforming *Nédélec* ("edge") elements provide.
#
# The motivation for computing ``\boldsymbol{H}`` directly is that the resulting
# approximation satisfies Ampère's law *exactly*: the circulation
# ``\oint \boldsymbol{H} \cdot \mathrm{d}\boldsymbol{l}`` around any closed loop of
# element edges equals the enclosed current, to machine precision, on any mesh. This is
# the very law a current transformer exploits, and more generally the property to preserve
# whenever the current linkage — the magnetomotive force — is the quantity of interest. At
# the end of this tutorial we solve the same problem with the standard potential-based
# formulation and compare.
#
# This tutorial is the ``H(\mathrm{curl})`` sibling of the
# [Darcy flow tutorial](@ref tutorial-darcy-flow), and follows the same route: readers
# familiar with that tutorial will recognize the structure of the mixed problem, with the
# divergence traded for the curl and the normal component for the tangential one. It
# teaches the following new concepts:
# - Formulating and assembling a **mixed problem** with an ``H(\mathrm{curl})``
#   interpolation (`Nedelec`) paired with a discontinuous approximation of the magnetic
#   vector potential (`DiscontinuousLagrange`), which acts as a Lagrange multiplier
# - Prescribing the tangential trace ``\boldsymbol{H} \times \boldsymbol{n}`` on a
#   boundary with [`ProjectedDirichlet`](@ref)
# - Evaluating curls of shape functions and solution fields with [`shape_curl`](@ref) and
#   [`function_curl`](@ref)
# - Verifying that Ampère's law holds exactly, element-wise and as a loop integral
# - Quantifying the tangential continuity of ``\boldsymbol{H}`` across element facets
#   using the [`InterfaceIterator`](@ref) and [`InterfaceValues`](@ref)
#
# ### Strong form
#
# The magnetostatic equations for the magnetic field ``\boldsymbol{H}`` and the flux
# density ``\boldsymbol{B}`` are Ampère's law, Gauss's law for magnetism, and the
# constitutive relation with permeability ``\mu``:
# ```math
# \mathrm{curl}(\boldsymbol{H}) = J, \qquad
# \mathrm{div}(\boldsymbol{B}) = 0, \qquad
# \boldsymbol{B} = \mu \boldsymbol{H}.
# ```
# We consider a two-dimensional problem: the cross-section of a long straight conductor,
# with the current density ``\boldsymbol{J} = J \boldsymbol{e}_z`` perpendicular to the
# plane and the magnetic field ``\boldsymbol{H} = (H_x, H_y)`` in the plane. In 2D the
# curl of a vector field is a scalar, and the curl of an out-of-plane scalar field is an
# in-plane vector,
# ```math
# \mathrm{curl}(\boldsymbol{H}) =
# \frac{\partial H_y}{\partial x} - \frac{\partial H_x}{\partial y},
# \qquad
# \mathrm{rot}(A) =
# \left( \frac{\partial A}{\partial y},\, -\frac{\partial A}{\partial x} \right),
# ```
# where the second operator is used to satisfy Gauss's law identically: writing
# ``\boldsymbol{B} = \mathrm{rot}(A)`` with the (out-of-plane) magnetic vector potential
# ``A`` gives ``\mathrm{div}(\boldsymbol{B}) = 0`` for any ``A``. The system to solve for
# the pair ``(\boldsymbol{H}, A)`` is then
# ```math
# \begin{alignat*}{2}
# \mu \boldsymbol{H} - \mathrm{rot}(A) &= \boldsymbol{0} \quad && \boldsymbol{x} \in \Omega \\
# \mathrm{curl}(\boldsymbol{H}) &= J \quad && \boldsymbol{x} \in \Omega \\
# \boldsymbol{H} \cdot \boldsymbol{t} &= 0 \quad && \boldsymbol{x} \in \Gamma_\mathrm{H} \\
# A &= 0 \quad && \boldsymbol{x} \in \Gamma_\mathrm{B}
# \end{alignat*}
# ```
# with ``\boldsymbol{t}`` the (counterclockwise) unit tangent of the boundary.
#
# The geometry is a conductor with a square cross-section, centered in a square domain,
# carrying the uniform current density ``J``. It is surrounded, at some distance, by a
# square iron core with a permeability three orders of magnitude higher than the
# surrounding air. Contour lines of ``A`` are the magnetic flux lines: they circle the
# conductor, and the highly permeable core "short-circuits" them, collecting nearly all
# of the flux. Since everything is symmetric with respect to the two coordinate axes we
# model only the upper right quadrant, ``\Omega = [0, 1]^2``:
# - On the two symmetry lines (the left and bottom edges) the field crosses at right
#   angles — it circles the conductor — so the tangential component vanishes there:
#   ``\boldsymbol{H} \cdot \boldsymbol{t} = 0`` on ``\Gamma_\mathrm{H}``.
# - The top and right edges truncate the (in reality unbounded) air region: ``A = 0``
#   makes the outer boundary a flux line, i.e. ``\boldsymbol{B} \cdot \boldsymbol{n} = 0``
#   ("magnetic insulation"), a standard far-field approximation.
#
# ### Weak form
#
# Multiplying the first equation with a (vector valued) test function
# ``\delta\boldsymbol{H}`` and integrating the potential term by parts using
# ```math
# \int_\Omega \mathrm{rot}(A) \cdot \delta\boldsymbol{H}\, \mathrm{d}\Omega =
# \int_\Omega A\, \mathrm{curl}(\delta\boldsymbol{H})\, \mathrm{d}\Omega
# - \int_{\partial\Omega} A\, (\delta\boldsymbol{H} \cdot \boldsymbol{t})\, \mathrm{d}\Gamma
# ```
# gives
# ```math
# \int_\Omega \mu\, \delta\boldsymbol{H} \cdot \boldsymbol{H}\, \mathrm{d}\Omega
# - \int_\Omega \mathrm{curl}(\delta\boldsymbol{H})\, A\, \mathrm{d}\Omega
# = - \int_{\Gamma_\mathrm{B}} (\delta\boldsymbol{H} \cdot \boldsymbol{t})\, A_\mathrm{D}\, \mathrm{d}\Gamma
# ```
# where the boundary integral vanished on ``\Gamma_\mathrm{H}`` because the test functions
# satisfy ``\delta\boldsymbol{H} \cdot \boldsymbol{t} = 0`` there. Just as in the
# [Darcy flow tutorial](@ref tutorial-darcy-flow) the roles of the boundary conditions are
# reversed compared to the potential-based formulation: the condition on the potential
# enters *weakly* (naturally) through the right hand side — and since ``A_\mathrm{D} = 0``
# here, the term drops out entirely — while the condition on the field is *essential* and
# is built into the trial and test spaces. Ampère's law is tested with a scalar test
# function ``\delta A`` (multiplied by ``-1`` to obtain a symmetric system):
# ```math
# - \int_\Omega \delta A\, \mathrm{curl}(\boldsymbol{H})\, \mathrm{d}\Omega
# = - \int_\Omega \delta A\, J\, \mathrm{d}\Omega
# ```
#
# The weak form also tells us which spaces the fields, and the corresponding test
# functions, live in. The only derivative that appears is the curl of the field, so
# ``\boldsymbol{H}`` and ``\delta\boldsymbol{H}`` must belong to ``H(\mathrm{curl})`` —
# the space of square-integrable vector fields with square-integrable curl — here with
# ``\boldsymbol{H} \cdot \boldsymbol{t} = 0`` on ``\Gamma_\mathrm{H}`` built in. The
# potential ``A`` and its test function ``\delta A`` are never differentiated, so they
# only need to be square-integrable: ``A \in L^2``. Membership in ``H(\mathrm{curl})``
# needs only the *tangential* component to be continuous across element boundaries — the
# normal component may jump. This is exactly the physical interface condition: across a
# material interface ``\boldsymbol{H} \cdot \boldsymbol{t}`` is continuous while
# ``\boldsymbol{H} \cdot \boldsymbol{n}`` jumps by the permeability ratio (such that
# ``\boldsymbol{B} \cdot \boldsymbol{n}`` is continuous). The discrete spaces chosen next
# mirror exactly this regularity — and no more.
#
# ### Choice of finite element spaces
#
# The pair of spaces for ``(\boldsymbol{H}, A)`` cannot be chosen freely — stability
# requires them to satisfy the LBB (inf-sup) condition. Here we use the lowest order
# stable pair: the field is interpolated with `Nedelec` functions (first kind), which are
# associated with the element *edges* and have continuous tangential components across
# them, and the potential with element-wise constants (`DiscontinuousLagrange` of order
# 0). In 2D the Nédélec functions are simply the Raviart-Thomas functions rotated by 90°,
# so this pair inherits the stability of the Raviart-Thomas pair used in the
# [Darcy flow tutorial](@ref tutorial-darcy-flow). Higher order pairs follow the same
# pattern: `Nedelec{RefTriangle, k}` together with
# `DiscontinuousLagrange{RefTriangle, k-1}`. See [Gatica2014](@cite) for an accessible
# introduction to mixed methods, [BofBreFor:2013:mfe](@cite) for the full theory, and
# [Bossavit1998](@cite) and [Monk2003](@cite) for the electromagnetic setting, where edge
# elements are the natural — and in 3D essentially unavoidable — choice.
#
# !!! note "Naming convention"
#     Ferrite's `Nedelec{RefTriangle, 1}` is the *lowest order* Nédélec element — the
#     classic *Whitney edge element* — which much of the literature (and e.g.
#     [DefElement](https://defelement.org/elements/nedelec1.html)) denotes
#     ``\mathrm{N1}_0`` or ``\mathrm{ND}_0``.
#
# It is worth spelling out what the degrees of freedom of an ``H(\mathrm{curl})``
# interpolation actually are, since it explains much of what follows: they are not point
# values, but *moments of the tangential component along an edge*,
# ``\int_E (\boldsymbol{N} \cdot \boldsymbol{t})\, s\, \mathrm{d}\Gamma`` for a set of
# weight functions ``s`` — for the lowest order element simply the average tangential
# component along the edge. Two consequences follow. Neighboring elements share these edge
# dofs, which is precisely why the tangential component is continuous across facets and
# the normal component is not. And there are no nodal support points at which a boundary
# value could be prescribed, so the essential boundary condition needs different treatment
# than usual.
#
# Testing Ampère's law with ``\delta A = 1`` on a single element immediately shows where
# the exactness comes from: since ``\delta A`` may be chosen as the indicator function of
# any single element ``K`` (the potential space is discontinuous!), the discrete solution
# satisfies ``\int_K \mathrm{curl}(\boldsymbol{H}_h)\, \mathrm{d}\Omega = \int_K J\, \mathrm{d}\Omega``
# for *every element individually*. By Stokes' theorem the left hand side is the
# circulation ``\oint_{\partial K} \boldsymbol{H}_h \cdot \mathrm{d}\boldsymbol{l}``, and
# since the tangential component is continuous, contributions of shared edges cancel when
# elements are combined: the circulation around *any* closed loop of element edges equals
# the enclosed current exactly. With a continuous potential space this argument is not
# available, and indeed the standard method satisfies Ampère's law only approximately.
#
# ## Implementation
#
# We start by loading the packages, and generating a triangulated unit square,
# representing the upper right quadrant of the device.
using Ferrite, SparseArrays, LinearAlgebra

grid = generate_grid(Triangle, (40, 40), Vec(0.0, 0.0), Vec(1.0, 1.0));

# ### Material regions
# The conductor occupies ``x, y \leq 0.2`` (the quadrant's quarter of the full conductor)
# and the core is the band ``0.5 \leq \max(x, y) \leq 0.7``. We collect their cells in
# cellsets (a cell belongs to a set if all its nodes are inside the region — note the half
# cell size tolerance `h/2`, which makes the sets insensitive to floating point roundoff
# in the node coordinates), and store the permeability and current density as one value
# per cell. We work in normalized units where the permeability of air is ``\mu = 1`` and
# the current density in the conductor is ``J = 1``; the relative permeability of the iron
# core is ``\mu_\mathrm{r} = 1000``.
h = 1 / 40 # cell size
addcellset!(grid, "conductor", x -> x[1] ≤ 0.2 + h / 2 && x[2] ≤ 0.2 + h / 2)
addcellset!(grid, "core", x -> 0.5 - h / 2 ≤ max(x[1], x[2]) ≤ 0.7 + h / 2)

μ = fill(1.0, getncells(grid))                    # air (and conductor) permeability
μ[collect(getcellset(grid, "core"))] .= 1000.0    # core permeability

Jz = zeros(getncells(grid))                       # current density
Jz[collect(getcellset(grid, "conductor"))] .= 1.0;

# The total current enclosed by the modeled quadrant is the current density times the
# conductor's cross-section area within the quadrant:
I_enclosed = 1.0 * 0.2^2

# ### Interpolations, DoF distribution and boundary conditions
# The magnetic field `:H` uses the Nédélec interpolation whose degrees of freedom sit on
# the edges, and the potential field `:A` a single constant per element. The geometry is
# interpolated with the standard linear Lagrange functions.
ip_H = Nedelec{RefTriangle, 1}()
ip_A = DiscontinuousLagrange{RefTriangle, 0}()
ip_geo = Lagrange{RefTriangle, 1}()

# The quadrature order is chosen such that all terms of the weak form are integrated
# exactly: the dominating term is ``\mu\, \delta\boldsymbol{N} \cdot \boldsymbol{N}`` — a
# product of two (component-wise) linear Nédélec functions — which is quadratic, so a rule
# of order 2 suffices (the terms involving the potential and the curl of the field are
# element-wise constant and thus require even less).
qr = QuadratureRule{RefTriangle}(2)
cv_H = CellValues(qr, ip_H, ip_geo)
cv_A = CellValues(qr, ip_A, ip_geo)

facet_qr = FacetQuadratureRule{RefTriangle}(2)
fv_H = FacetValues(facet_qr, ip_H, ip_geo);

# Distributing the dofs works as usual,
dh = DofHandler(grid)
add!(dh, :H, ip_H)
add!(dh, :A, ip_A)
close!(dh);

# but for the essential boundary condition ``\boldsymbol{H} \cdot \boldsymbol{t} = 0`` we
# cannot use a regular `Dirichlet` condition: as discussed above, the Nédélec
# interpolation has no nodal support points where a value could be prescribed. Instead,
# [`ProjectedDirichlet`](@ref) determines the edge dof values by an L2 projection of the
# prescribed tangential trace onto the facet — which is exactly the kind of quantity the
# edge dofs represent. The prescribed function receives the coordinate, time, and facet
# normal as arguments and returns the tangential trace
# ``\boldsymbol{H} \times \boldsymbol{n}`` as a *vector*, which for a 2d problem points
# out of the plane — hence the three-component zero below.
ch = ConstraintHandler(dh)
add!(ch, ProjectedDirichlet(:H, union(getfacetset(grid, "left"), getfacetset(grid, "bottom")), (x, t, n) -> zero(Vec{3})))
close!(ch);

# ### Assembly
# The element routine assembles the symmetric block system
# ```math
# \begin{bmatrix}
# \underline{\underline{M}} & \underline{\underline{C}}^\mathrm{T} \\
# \underline{\underline{C}} & \underline{\underline{0}}
# \end{bmatrix}
# \begin{bmatrix} \underline{H} \\ \underline{A} \end{bmatrix}
# =
# \begin{bmatrix} \underline{0} \\ \underline{g} \end{bmatrix},
# \quad
# M_{ij} = \int_\Omega \mu\, \delta\boldsymbol{N}_i \cdot \boldsymbol{N}_j\, \mathrm{d}\Omega,
# \quad
# C_{ij} = -\int_\Omega N_i\, \mathrm{curl}(\boldsymbol{N}_j)\, \mathrm{d}\Omega,
# \quad
# g_{i} = -\int_\Omega N_i\, J\, \mathrm{d}\Omega
# ```
# Since both fields live on the same cells, we assemble one local matrix containing all
# blocks, using [`dof_range`](@ref) to place the entries correctly — the same pattern as
# in the [Darcy flow tutorial](@ref tutorial-darcy-flow). The new ingredient is
# [`shape_curl`](@ref) for the Nédélec functions; note that it returns the curl as a
# (one-component, since the curl of a 2d field has only the out-of-plane component)
# vector, of which we take the first component. Had the natural boundary condition been
# inhomogeneous, ``A_\mathrm{D} \neq 0``, the corresponding boundary integral would be
# assembled with a `FacetIterator` exactly like the pressure condition in the
# [Darcy flow tutorial](@ref tutorial-darcy-flow); here it vanishes and the source term
# ``g`` is a volume integral instead.
function assemble_magnetostatics!(K, f, dh, cv_H, cv_A, μ, Jz)
    range_H = dof_range(dh, :H)
    range_A = dof_range(dh, :A)
    n_dofs = ndofs_per_cell(dh)
    Ke = zeros(n_dofs, n_dofs)
    fe = zeros(n_dofs)
    assembler = start_assemble(K, f)
    for cell in CellIterator(dh)
        reinit!(cv_H, cell)
        reinit!(cv_A, cell)
        fill!(Ke, 0.0)
        fill!(fe, 0.0)
        μᵉ = μ[cellid(cell)]
        Jᵉ = Jz[cellid(cell)]
        for qp in 1:getnquadpoints(cv_H)
            dΩ = getdetJdV(cv_H, qp)
            ## M and Cᵀ blocks (test function δH)
            for (i, I) in pairs(range_H)
                δNH = shape_value(cv_H, qp, i)
                curl_δNH = shape_curl(cv_H, qp, i)[1]
                for (j, J) in pairs(range_H)
                    NH = shape_value(cv_H, qp, j)
                    Ke[I, J] += μᵉ * (δNH ⋅ NH) * dΩ
                end
                for (j, J) in pairs(range_A)
                    NA = shape_value(cv_A, qp, j)
                    Ke[I, J] -= curl_δNH * NA * dΩ
                end
            end
            ## C block and source term (test function δA)
            for (i, I) in pairs(range_A)
                δNA = shape_value(cv_A, qp, i)
                fe[I] -= δNA * Jᵉ * dΩ
                for (j, J) in pairs(range_H)
                    curl_NH = shape_curl(cv_H, qp, j)[1]
                    Ke[I, J] -= δNA * curl_NH * dΩ
                end
            end
        end
        assemble!(assembler, celldofs(cell), Ke, fe)
    end
    return K, f
end
#md nothing #hide

# Now we can assemble and solve the linear system as usual:
K = allocate_matrix(dh)
f = zeros(ndofs(dh))
assemble_magnetostatics!(K, f, dh, cv_H, cv_A, μ, Jz)

apply!(K, f, ch)
a = K \ f;

# ### Postprocessing
#
# Neither of the two fields has nodal values that could be exported directly: the
# potential is constant in each cell, so we export it as cell data, and the flux density
# ``\boldsymbol{B} = \mu \boldsymbol{H}`` we evaluate in the quadrature points and project
# onto a linear Lagrange interpolation with the [`L2Projector`](@ref) for visualization.
A_cells = [a[celldofs(dh, cellid)[dof_range(dh, :A)[1]]] for cellid in 1:getncells(grid)]

function collect_qp_flux_densities(dh, cv_H, a, μ)
    qp_B = [
        [zero(Vec{2}) for _ in 1:getnquadpoints(cv_H)]
            for _ in 1:getncells(dh.grid)
    ]
    for cell in CellIterator(dh)
        reinit!(cv_H, cell)
        aᵉ = a[celldofs(cell)][dof_range(dh, :H)]
        for qp in 1:getnquadpoints(cv_H)
            qp_B[cellid(cell)][qp] = μ[cellid(cell)] * function_value(cv_H, qp, aᵉ)
        end
    end
    return qp_B
end
qp_B = collect_qp_flux_densities(dh, cv_H, a, μ)

proj = L2Projector(Lagrange{RefTriangle, 1}(), grid)
B_projected = project(proj, qp_B, qr)

VTKGridFile("magnetostatics", dh) do vtk
    write_cell_data(vtk, A_cells, "A")
    write_projection(vtk, proj, B_projected, "B")
    Ferrite.write_cellset(vtk, grid, "conductor")
    Ferrite.write_cellset(vtk, grid, "core")
end;

# Visualizing the potential and the flux density magnitude (Figure 1) shows the flux
# circling the conductor and concentrating inside the core — note the logarithmic color
# scale: the flux density in the core is three orders of magnitude larger than in the
# air just outside it.
#
# ### Ampère's law, exactly
#
# We now verify the properties that motivated the mixed formulation in the first place.
# First we verify that Ampère's law holds *element-wise* by computing
# ``\int_K \left[ \mathrm{curl}(\boldsymbol{H}_h) - J \right] \mathrm{d}\Omega`` for every
# element, using [`function_curl`](@ref):
function cell_ampere_residuals(dh, cv_H, a, Jz)
    residuals = zeros(getncells(dh.grid))
    for cell in CellIterator(dh)
        reinit!(cv_H, cell)
        aᵉ = a[celldofs(cell)][dof_range(dh, :H)]
        for qp in 1:getnquadpoints(cv_H)
            curl_H = function_curl(cv_H, qp, aᵉ)[1]
            residuals[cellid(cell)] += (curl_H - Jz[cellid(cell)]) * getdetJdV(cv_H, qp)
        end
    end
    return residuals
end
maximum(abs, cell_ampere_residuals(dh, cv_H, a, Jz))

# This is zero to machine precision: every element carries exactly the circulation that
# its share of the current dictates, no matter how coarse the mesh.
#
# Second, we compute the circulation
# ``\oint \boldsymbol{H} \cdot \boldsymbol{t}\, \mathrm{d}\Gamma`` around the domain
# boundary, which by Ampère's law must equal the enclosed current. The two symmetry edges
# contribute nothing (``\boldsymbol{H} \cdot \boldsymbol{t} = 0`` is built into the
# solution space there), so we integrate over the top and right edges, rotating the
# outward facet normal a quarter turn to get the counterclockwise tangent:
function circulation(dh, fv_H, facetset, a)
    C = 0.0
    for facet in FacetIterator(dh, facetset)
        reinit!(fv_H, facet)
        aᵉ = a[celldofs(facet)][dof_range(dh, :H)]
        for qp in 1:getnquadpoints(fv_H)
            n = getnormal(fv_H, qp)
            t = Vec(-n[2], n[1])
            C += (function_value(fv_H, qp, aᵉ) ⋅ t) * getdetJdV(fv_H, qp)
        end
    end
    return C
end
C_outer = circulation(dh, fv_H, union(getfacetset(grid, "top"), getfacetset(grid, "right")), a)
C_outer - I_enclosed

# Again exact to machine precision. Note a piece of physics embedded in this number: it is
# entirely independent of the core. However permeable, the core concentrates the flux
# passing through it but cannot change any circulation of ``\boldsymbol{H}`` — a net
# current cannot be magnetically shielded, and the exact discrete solution reproduces
# this.
#
# ### Flux concentration in the core
#
# The core's job in a current transformer is to collect the flux, so let us quantify how
# well it does it. We compare the flux through the core's cross-section — the segment
# ``0.5 \leq x \leq 0.7`` of the bottom symmetry edge — with the flux through the
# equally wide air segment just outside it, by integrating
# ``\boldsymbol{B} \cdot \boldsymbol{e}_y`` over the corresponding facets:
addfacetset!(grid, "core cut", x -> x[2] ≈ 0.0 && 0.5 - h / 2 ≤ x[1] ≤ 0.7 + h / 2)
addfacetset!(grid, "air cut", x -> x[2] ≈ 0.0 && 0.75 - h / 2 ≤ x[1] ≤ 0.95 + h / 2)

function flux_through_cut(dh, fv_H, facetset, a, μ)
    Φ = 0.0
    for facet in FacetIterator(dh, facetset)
        reinit!(fv_H, facet)
        aᵉ = a[celldofs(facet)][dof_range(dh, :H)]
        for qp in 1:getnquadpoints(fv_H)
            B = μ[cellid(facet)] * function_value(fv_H, qp, aᵉ)
            Φ += B[2] * getdetJdV(fv_H, qp)
        end
    end
    return Φ
end
Φ_core = flux_through_cut(dh, fv_H, getfacetset(grid, "core cut"), a, μ)
Φ_air = flux_through_cut(dh, fv_H, getfacetset(grid, "air cut"), a, μ)
Φ_core / Φ_air

# The core carries three orders of magnitude more flux than the air region behind it —
# essentially all flux lines that reach the core stay inside it (see Figure 1).
#
# ### Comparison with the standard (potential-based) formulation
#
# For contrast, we solve the same problem with the standard single-field formulation:
# find the potential ``A \in \mathbb{U}`` (linear Lagrange interpolation) such that
# ```math
# \int_\Omega \nu\, \mathrm{rot}(\delta A) \cdot \mathrm{rot}(A)\, \mathrm{d}\Omega
# = \int_\Omega \delta A\, J\, \mathrm{d}\Omega
# \quad \forall\, \delta A
# ```
# with the reluctivity ``\nu = 1/\mu``, and recover the field afterwards as
# ``\boldsymbol{H}_h = \nu\, \mathrm{rot}(A_h)``. Since
# ``\mathrm{rot}(\delta A) \cdot \mathrm{rot}(A) = \mathrm{grad}(\delta A) \cdot \mathrm{grad}(A)``
# this is precisely the [heat equation tutorial](@ref tutorial-heat-equation) with a
# source term. The boundary conditions swap roles compared to the mixed formulation:
# ``A = 0`` on the outer boundary is now essential, while the symmetry condition
# ``\boldsymbol{H} \cdot \boldsymbol{t} = 0`` is natural and needs no code at all. We
# state the program compactly:
function solve_primal(grid, μ, Jz)
    ip = Lagrange{RefTriangle, 1}()
    cv = CellValues(QuadratureRule{RefTriangle}(2), ip)
    dh = DofHandler(grid)
    add!(dh, :A, ip)
    close!(dh)
    ch = ConstraintHandler(dh)
    add!(ch, Dirichlet(:A, union(getfacetset(grid, "top"), getfacetset(grid, "right")), Returns(0.0)))
    close!(ch)
    K = allocate_matrix(dh)
    f = zeros(ndofs(dh))
    assembler = start_assemble(K, f)
    Ke = zeros(ndofs_per_cell(dh), ndofs_per_cell(dh))
    fe = zeros(ndofs_per_cell(dh))
    for cell in CellIterator(dh)
        reinit!(cv, cell)
        fill!(Ke, 0.0)
        fill!(fe, 0.0)
        ν = 1 / μ[cellid(cell)]
        for qp in 1:getnquadpoints(cv)
            dΩ = getdetJdV(cv, qp)
            for i in 1:getnbasefunctions(cv)
                δNA = shape_value(cv, qp, i)
                ∇δNA = shape_gradient(cv, qp, i)
                fe[i] += δNA * Jz[cellid(cell)] * dΩ
                for j in 1:getnbasefunctions(cv)
                    Ke[i, j] += ν * (∇δNA ⋅ shape_gradient(cv, qp, j)) * dΩ
                end
            end
        end
        assemble!(assembler, celldofs(cell), Ke, fe)
    end
    apply!(K, f, ch)
    return dh, K \ f
end
dh_primal, A_primal = solve_primal(grid, μ, Jz);

# The recovered field ``\nu\, \mathrm{rot}(A_h)`` is integrated directly on the outer
# boundary to check the circulation:
function circulation_primal(dh, μ, facetset, A)
    fv = FacetValues(FacetQuadratureRule{RefTriangle}(2), Lagrange{RefTriangle, 1}())
    C = 0.0
    for facet in FacetIterator(dh, facetset)
        reinit!(fv, facet)
        Aᵉ = A[celldofs(facet)]
        for qp in 1:getnquadpoints(fv)
            n = getnormal(fv, qp)
            t = Vec(-n[2], n[1])
            ∇A = function_gradient(fv, qp, Aᵉ)
            H = Vec(∇A[2], -∇A[1]) / μ[cellid(facet)]
            C += (H ⋅ t) * getdetJdV(fv, qp)
        end
    end
    return C
end
C_primal = circulation_primal(dh_primal, μ, union(getfacetset(grid, "top"), getfacetset(grid, "right")), A_primal)

# Comparing Ampère's law for the two methods:
using Printf
@printf("mixed:  circulation = %10.6f, relative error = %9.2e\n", C_outer, (C_outer - I_enclosed) / I_enclosed)
@printf("primal: circulation = %10.6f, relative error = %9.2e\n", C_primal, (C_primal - I_enclosed) / I_enclosed)

# The primal method misses a few percent of the enclosed current on this mesh — not
# because current is lost (globally the formulation is consistent), but because the
# recovered field commits a local error that the loop integral picks up. The error is
# concentrated where the field is approximated worst: at the corners of the conductor and
# the core, and along the material interfaces. To make this local defect visible we
# integrate the absolute tangential jump
# ``\int_\Gamma \left| [\![ \boldsymbol{H}_h ]\!] \cdot \boldsymbol{t} \right| \mathrm{d}\Gamma``
# over all interior facets, using the [`InterfaceIterator`](@ref) and
# [`InterfaceValues`](@ref) known from the
# [Discontinuous Galerkin tutorial](@ref tutorial-dg-heat-equation):
function tangential_jump_mixed(dh, a, topology)
    iv = InterfaceValues(FacetQuadratureRule{RefTriangle}(2), Nedelec{RefTriangle, 1}(), Lagrange{RefTriangle, 1}())
    range_H = dof_range(dh, :H)
    J = 0.0
    for ic in InterfaceIterator(dh, topology)
        reinit!(iv, ic)
        aᵉ = vcat(a[celldofs(ic.a)][range_H], a[celldofs(ic.b)][range_H])
        for qp in 1:getnquadpoints(iv)
            n = getnormal(iv, qp)
            t = Vec(-n[2], n[1])
            jump = function_value_jump(iv, qp, aᵉ) ⋅ t
            J += abs(jump) * getdetJdV(iv, qp)
        end
    end
    return J
end

function tangential_jump_primal(dh, A, μ, topology)
    iv = InterfaceValues(FacetQuadratureRule{RefTriangle}(2), Lagrange{RefTriangle, 1}())
    J = 0.0
    for ic in InterfaceIterator(dh, topology)
        reinit!(iv, ic)
        Aᵉ = vcat(A[celldofs(ic.a)], A[celldofs(ic.b)])
        for qp in 1:getnquadpoints(iv)
            n = getnormal(iv, qp)
            t = Vec(-n[2], n[1])
            ∇A_here = function_gradient(iv, qp, Aᵉ; here = true)
            ∇A_there = function_gradient(iv, qp, Aᵉ; here = false)
            H_here = Vec(∇A_here[2], -∇A_here[1]) / μ[cellid(ic.a)]
            H_there = Vec(∇A_there[2], -∇A_there[1]) / μ[cellid(ic.b)]
            J += abs((H_here - H_there) ⋅ t) * getdetJdV(iv, qp)
        end
    end
    return J
end

topology = ExclusiveTopology(grid)
J_mixed = tangential_jump_mixed(dh, a, topology)
J_primal = tangential_jump_primal(dh_primal, A_primal, μ, topology)
@printf("tangential jump, mixed:  %9.2e\n", J_mixed / I_enclosed)
@printf("tangential jump, primal: %9.2e\n", J_primal / I_enclosed)

# This measure separates the two methods clearly: relative to the enclosed current, the
# accumulated tangential mismatch of the primal solution is several times *larger* than
# the current itself (it is dominated by the facets at the air-core interface, where the
# permeability contrast is large), while the mixed solution is tangentially continuous to
# machine precision.
#
# The comparison is not one-sided, however — the two formulations have precisely
# complementary strengths. The primal flux density ``\mathrm{rot}(A_h)`` of a continuous
# ``A_h`` is exactly divergence free and has an exactly continuous normal component: the
# primal method satisfies Gauss's law exactly and Ampère's law approximately. The mixed
# method does the opposite: ``\boldsymbol{B} = \mu \boldsymbol{H}_h`` satisfies Ampère's
# law exactly, while ``\mathrm{div}(\boldsymbol{B})`` and the continuity of
# ``\boldsymbol{B} \cdot \boldsymbol{n}`` hold only weakly. Which formulation to prefer
# therefore depends on which law matters most for the application — here, where the
# current linkage is the point, Ampère wins. The mixed method additionally pays with a
# larger, indefinite (saddle point) system, just like in the
# [Darcy flow tutorial](@ref tutorial-darcy-flow). Finally, a note on dimensionality: in
# 2D the potential is a scalar and the primal method needs nothing beyond standard
# Lagrange elements, but in 3D the vector potential ``\boldsymbol{A}`` is a vector field
# whose natural conforming space is itself ``H(\mathrm{curl})`` — so for 3D
# magnetostatics, Nédélec elements appear in the primal formulation too, and the
# machinery from this tutorial is required either way.
#
# ## Suggestions for tweaking the program
# - Increase the order of the interpolation pair to `Nedelec{RefTriangle, 2}` /
#   `DiscontinuousLagrange{RefTriangle, 1}`, and verify with a manufactured solution,
#   e.g. ``A = \sin(\pi x)\sin(\pi y)`` with the corresponding current density
#   ``J = \mathrm{curl}(\mathrm{rot}(A)/\mu)``, that the convergence rates increase
#   accordingly.
# - Cut an air gap into the core (remove a thin band of cells from the "core" cellset)
#   and watch the collected flux ``\Phi_\mathrm{core}`` collapse — the gap dominates the
#   reluctance of the whole magnetic circuit, which is why real transformer cores are
#   stacked to avoid gaps.
# - Increase the core permeability to ``\mu_\mathrm{r} = 10^5``. The circulation stays
#   exact, but observe what happens to the condition number of the system matrix.
# - Replace the triangles with quadrilaterals (`Nedelec{RefQuadrilateral, 1}`).
#
# Test the result                                                     #src
using Test                                                            #hide
@test maximum(abs, cell_ampere_residuals(dh, cv_H, a, Jz)) < 1.0e-10  #hide
@test abs(C_outer - I_enclosed) / I_enclosed < 1.0e-8                 #hide
@test J_mixed / I_enclosed < 1.0e-12                                  #hide
@test J_primal / I_enclosed > 1.0                                     #hide
@test 1.0e-3 < abs(C_primal - I_enclosed) / I_enclosed < 1.0e-1       #hide
@test Φ_core / Φ_air > 500                                            #hide
@test Φ_core ≈ 7.289465 atol = 1.0e-4                                 #hide
nothing                                                               #hide

#md # ## References
#md # ```@bibliography
#md # Pages = ["magnetostatics.md"]
#md # Canonical = false
#md # ```

#md # ## [Plain program](@id magnetostatics-plain-program)
#md #
#md # Here follows a version of the program without any comments.
#md # The file is also available here: [`magnetostatics.jl`](magnetostatics.jl).
#md #
#md # ```julia
#md # @__CODE__
#md # ```
