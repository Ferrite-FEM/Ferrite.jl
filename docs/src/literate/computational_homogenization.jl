# # Computational homogenization
#
# ![](rve_homogenization.png)
#
# Figure 1: *von Mises stress in an RVE with 5 stiff inclusions embedded in a softer matrix
# material that is loaded in shear. The problem is solved by using homogeneous Dirichlet
# boundary conditions (left) and (strong) periodic boundary conditions (right).*
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
# This system is not solvable without introducing extra restrictions on ``\mathbb{U}_\Box``,
# ``\mathbb{T}_\Box``. In this example we will consider the common cases of Dirichlet
# boundary conditions and (strong) periodic boundary conditions.
#
# **Dirichlet boundary conditions**
#
# We can introduce the more restrictive sets of ``\mathbb{U}_\Box``:
#
# ```math
# \begin{align*}
# \mathbb{U}_\Box^\mathrm{D} &:= \left\{\boldsymbol{u} \in \mathbb{U}_\Box|\ \boldsymbol{u}
# = \bar{\boldsymbol{\varepsilon}} \cdot [\boldsymbol{x} - \bar{\boldsymbol{x}}]
# \ \mathrm{on}\ \Gamma_\Box\right\},\\
# \mathbb{U}_\Box^{\mathrm{D},0} &:= \left\{\boldsymbol{u} \in \mathbb{U}_\Box|\ \boldsymbol{u}
# = \boldsymbol{0}\ \mathrm{on}\ \Gamma_\Box\right\},
# \end{align*}
# ```
#
# and use these as trial and test sets to obtain a solvable RVE problem pertaining to
# Dirichlet boundary conditions. Eq. ``(1\mathrm{b})`` is trivially fulfilled, the second
# term of Eq. ``(1\mathrm{a})`` vanishes, and we are left with the following problem:
# Find ``\boldsymbol{u} \in \mathbb{U}_\Box^\mathrm{D}`` that solve
#
# ```math
# \frac{1}{|\Omega_\Box|} \int_{\Omega_\Box}\boldsymbol{\varepsilon}[\delta\boldsymbol{u}]
# : \mathsf{E} : \boldsymbol{\varepsilon}[\boldsymbol{u}]\ \mathrm{d}\Omega = 0
# \quad \forall \delta \boldsymbol{u} \in \mathbb{U}_\Box^{\mathrm{D},0}.
# ```
#
# Note that, since ``\boldsymbol{u} = \bar{\boldsymbol{\varepsilon}} \cdot [\boldsymbol{x} -
# \bar{\boldsymbol{x}}] + \boldsymbol{u}^\mu``, this problem is equivalent to solving for
# ``\boldsymbol{u}^\mu \in \mathbb{U}_\Box^{\mathrm{D},0}``, which is what we will do in
# the implementation.
#
# **Periodic boundary conditions**
#
# The RVE problem pertaining to periodic boundary conditions is obtained by restricting
# ``\boldsymbol{u}^\mu`` to be periodic, and ``\boldsymbol{t}`` anti-periodic across the
# RVE. Similarly as for Dirichlet boundary conditions, Eq. ``(1\mathrm{b})`` is directly
# fulfilled, and the second term in Eq. ``(1\mathrm{a})`` vanishes, with these restrictions,
# and we are left with the following problem:
# Find ``\boldsymbol{u}^\mu \in \mathbb{U}_\Box^{\mathrm{P},0}`` such that
#
# ```math
# \frac{1}{|\Omega_\Box|} \int_{\Omega_\Box}\boldsymbol{\varepsilon}[\delta\boldsymbol{u}]
# : \mathsf{E} : (\bar{\boldsymbol{\varepsilon}} + \boldsymbol{\varepsilon}
# [\boldsymbol{u}^\mu])\ \mathrm{d}\Omega = 0
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
# as the input to the RVE problem. When the sensitivies are solved we can compute the
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


# ## Commented Program
#
# Now we will see how this can be implemented in `Ferrite`. What follows is a program
# with comments in between which describe the different steps.
#md # You can also find the same program without comments at the end of the page,
#md # see [Plain program](@ref homogenization-plain-program).

using Ferrite, SparseArrays, LinearAlgebra
using Test #src

# We first load the mesh file [`periodic-rve.msh`](periodic-rve.msh)
# ([`periodic-rve-coarse.msh`](periodic-rve-coarse.msh) for a coarser mesh). The mesh is
# generated with [`gmsh`](https://gmsh.info/), and we read it in as a `Ferrite` grid using
# the [`FerriteGmsh`](https://github.com/Ferrite-FEM/FerriteGmsh.jl) package:

using FerriteGmsh
#src notebook: use coarse mesh to decrease build time
#src   script: use the fine mesh
#src markdown: use the coarse mesh to decrease build time, but make it look like the fine
#nb ## grid = saved_file_to_grid("periodic-rve.msh")
#nb grid = saved_file_to_grid("periodic-rve-coarse.msh")
#jl ## grid = saved_file_to_grid("periodic-rve-coarse.msh")
#jl grid = saved_file_to_grid("periodic-rve.msh")
#md grid = saved_file_to_grid("periodic-rve.msh")
#-
#md grid = redirect_stdout(devnull) do                #hide
#md     saved_file_to_grid("periodic-rve-coarse.msh") #hide
#md end                                               #hide

grid = saved_file_to_grid("periodic-rve.msh") #src

# Next we construct the interpolation and quadrature rule, and combining them into
# cellvalues as usual:

dim = 2
ip = Lagrange{dim, RefTetrahedron, 1}()
qr = QuadratureRule{dim, RefTetrahedron}(2)
cellvalues = CellVectorValues(qr, ip);

# We define a dof handler with a displacement field `:u`:
dh = DofHandler(grid)
push!(dh, :u, 2)
close!(dh);

# Now we need to define boundary conditions. As discussed earlier we will solve the problem
# using (i) homogeneous Dirichlet boundary conditions, and (ii) periodic Dirichlet boundary
# conditions. We construct two different constraint handlers, one for each case. The
# [`Dirichlet`](@ref) boundary condition we have seen in many other examples. Here we simply
# define the condition that the field, `:u`, should have both components prescribed to `0`
# on the full boundary:

ch_dirichlet = ConstraintHandler(dh)
dirichlet = Dirichlet(
    :u,
    union(getfaceset.(Ref(grid), ["left", "right", "top", "bottom"])...),
    (x, t) ->  [0, 0],
    [1, 2]
)
add!(ch_dirichlet, dirichlet)
close!(ch_dirichlet)
update!(ch_dirichlet, 0.0)

# For periodic boundary conditions we use the [`PeriodicDirichlet`](@ref) constraint type,
# which is very similar to the `Dirichlet` type, but instead of a passing a faceset we pass
# a vector with "face pairs", i.e. the mapping between mirror and image parts of the
# boundary. In this example the `"left"` and `"bottom"` boundaries are mirrors, and the
# `"right"` and `"top"` boundaries are the mirrors.

ch_periodic = ConstraintHandler(dh);
periodic = PeriodicDirichlet(
    :u,
    ["left" => "right", "bottom" => "top"],
    [1, 2]
)
add!(ch_periodic, periodic)
close!(ch_periodic)
update!(ch_periodic, 0.0)

# This will now constrain any degrees of freedom located on the mirror boundaries to
# the matching degree of freedom on the image boundaries. Internally this will create
# a number of [`AffineConstraint`](@ref)s of the form `u_i = 1 * u_j + 0`:
# ```julia
# a = AffineConstraint(u_m, [u_i => 1], 0)
# ```
# where `u_m` is the degree of freedom on the mirror and `u_i` the matching one on the
# image part. `PeriodicDirichlet` is thus simply just a more convenient way of
# constructing such affine constraints since it computes the degree of freedom mapping
# automatically.
#
# To simplify things we group the constraint handlers into a named tuple

ch = (dirichlet = ch_dirichlet, periodic = ch_periodic);

# We can now construct the sparse matrix. Note that, since we are using affine constraints,
# which need to modify the matrix sparsity pattern in order to account for the constraint
# equations, we construct the matrix for the periodic case by passing both the dof handler
# and the constraint handler.

K = (
    dirichlet = create_sparsity_pattern(dh),
    periodic  = create_sparsity_pattern(dh, ch.periodic),
);

# We define the fourth order elasticity tensor for the matrix material, and define the
# inclusions to have 10 times higher stiffness

λ, μ = 1e10, 7e9 # Lamé parameters
δ(i,j) = i == j ? 1.0 : 0.0
Em = SymmetricTensor{4, 2}(
    (i,j,k,l) -> λ * δ(i,j) * δ(k,l) + μ * (δ(i,k) * δ(j,l) + δ(i,l) * δ(j,k))
)
Ei = 10 * Em;

# As mentioned above, in order to compute the apparent/homogenized stiffness we will solve
# the problem repeatedly with different macroscale strain tensors to compute the sensitvity
# of the homogenized stress, ``\bar{\boldsymbol{\sigma}}``, w.r.t. the macroscopic strain,
# ``\bar{\boldsymbol{\varepsilon}}``. The corresponding unit strains are defined below,
# and will result in three different right-hand-sides:

εᴹ = [
      SymmetricTensor{2,2}([1.0 0.0; 0.0 0.0]), # ε_11 loading
      SymmetricTensor{2,2}([0.0 0.0; 0.0 1.0]), # ε_22 loading
      SymmetricTensor{2,2}([0.0 0.5; 0.5 0.0]), # ε_12/ε_21 loading
];

# The assembly function is nothing strange, and in particular there is no impact from the
# choice of boundary conditions, so the same function can be used for both cases. Since
# we want to solve the system 3 times, once for each macroscopic strain component, we
# assemble 3 right-hand-sides.

function doassemble!(cellvalues::CellVectorValues, K::SparseMatrixCSC, dh::DofHandler, εᴹ)

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
                    fe[i, rhs] += ( - δεi ⊡ σᴹ) * dΩ
               end
            end
        end

        cdofs = celldofs(cell)
        assemble!(assembler, cdofs, Ke)
        f[cdofs, :] .+= fe
    end
    return f
end;

# We can now assemble the system. The assembly function modifies the matrix in-place, but
# return the right hand side(s) which we collect in another named tuple.

rhs = (
    dirichlet = doassemble!(cellvalues, K.dirichlet, dh, εᴹ),
    periodic  = doassemble!(cellvalues, K.periodic,  dh, εᴹ),
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
    periodic  = get_rhs_data(ch.periodic,  K.periodic),
)

apply!(K.dirichlet, ch.dirichlet)
Kp = copy(K.periodic) #hide
apply!(K.periodic,  ch.periodic)

# We can now solve the problem(s). Note that we only use `apply_rhs!` in the loops below.
# The boundary conditions are already applied to the matrix above, so we only need to
# modify the right hand side.

u = (
    dirichlet = Vector{Float64}[],
    periodic  = Vector{Float64}[],
)

for i in 1:size(rhs.dirichlet, 2)
    rhs_i = @view rhs.dirichlet[:, i]                  # Extract this RHS
    apply_rhs!(rhsdata.dirichlet, rhs_i, ch.dirichlet) # Apply BC
    u_i = cholesky(Symmetric(K.dirichlet)) \ rhs_i     # Solve
    apply!(u_i, ch.dirichlet)                          # Apply BC on the solution
    push!(u.dirichlet, u_i)                            # Save the solution vector
end

rhs_p = copy(rhs.periodic) #hide
for i in 1:size(rhs.periodic, 2)
    rhs_i = @view rhs.periodic[:, i]                   # Extract this RHS
    apply_rhs!(rhsdata.periodic, rhs_i, ch.periodic)   # Apply BC
    rhs_i = @view rhs_p[:, i] #hide
    Kpp = copy(Kp) #hide
    apply!(Kpp, rhs_i, ch.periodic) #hide
    copy!(K.periodic, Kpp) #hide
    u_i = cholesky(Symmetric(K.periodic)) \ rhs_i      # Solve
    apply!(u_i, ch.periodic)                           # Apply BC on the solution
    push!(u.periodic, u_i)                             # Save the solution vector
end

# When the solution(s) are known we can compute the averaged stress,
# ``\bar{\boldsymbol{\sigma}}`` in the RVE. We define a function that does this, and also
# returns the von Mise stress in every quadrature point for visualization.

function compute_stress(cellvalues::CellVectorValues, dh::DofHandler, u, εᴹ)
    σvM_qpdata = zeros(getnquadpoints(cellvalues), getncells(dh.grid))
    σ̄Ω = zero(SymmetricTensor{2,2})
    Ω = 0.0 # Total volume
    for cell in CellIterator(dh)
        E = cellid(cell) in getcellset(dh.grid, "inclusions") ? Ei : Em
        reinit!(cellvalues, cell)
        for q_point in 1:getnquadpoints(cellvalues)
            dΩ = getdetJdV(cellvalues, q_point)
            εμ = function_symmetric_gradient(cellvalues, q_point, u[celldofs(cell)])
            σ = E ⊡ (εᴹ + εμ)
            σvM_qpdata[q_point, cellid(cell)] = sqrt(3/2 * dev(σ) ⊡ dev(σ))
            Ω += dΩ # Update total volume
            σ̄Ω += σ * dΩ # Update integrated stress
        end
    end
    σ̄ = σ̄Ω / Ω
    return σvM_qpdata, σ̄
end;

# We now compute the homogenized stress and von Mise stress for all cases

σ̄ = (
    dirichlet = SymmetricTensor{2,2}[],
    periodic  = SymmetricTensor{2,2}[],
)
σ = (
     dirichlet = Vector{Float64}[],
     periodic  = Vector{Float64}[],
)

projector = L2Projector(ip, grid)

for i in 1:3
    σ_qp, σ̄_i = compute_stress(cellvalues, dh, u.dirichlet[i], εᴹ[i])
    proj = project(projector, σ_qp, qr; project_to_nodes=false)
    push!(σ.dirichlet, proj)
    push!(σ̄.dirichlet, σ̄_i)
end

for i in 1:3
    σ_qp, σ̄_i = compute_stress(cellvalues, dh, u.periodic[i], εᴹ[i])
    proj = project(projector, σ_qp, qr; project_to_nodes=false)
    push!(σ.periodic, proj)
    push!(σ̄.periodic, σ̄_i)
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

E_dirichlet = SymmetricTensor{4,2}((i, j, k, l) -> begin
    if k == l == 1
        σ̄.dirichlet[1][i, j] # ∂σ∂ε_**11
    elseif k == l == 2
        σ̄.dirichlet[2][i, j] # ∂σ∂ε_**22
    else
        σ̄.dirichlet[3][i, j] # ∂σ∂ε_**12 and ∂σ∂ε_**21
    end
end)

E_periodic = SymmetricTensor{4,2}((i, j, k, l) -> begin
    if k == l == 1
        σ̄.periodic[1][i, j]
    elseif k == l == 2
        σ̄.periodic[2][i, j]
    else
        σ̄.periodic[3][i, j]
    end
end);

# We can check that the result are what we expect, namely that the stiffness with Dirichlet
# boundary conditions is higher than when using periodic boundary conditions, and that
# the Reuss assumption is an lower bound, and the Voigt assumption a upper bound. We first
# compute the volume fraction of the matrix, and then the Voigt and Reuss bounds:

function matrix_volume_fraction(grid, cellvalues)
    V  = 0.0 # Total volume
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
E_voigt = vm * Em + (1-vm) * Ei
E_reuss = inv(vm * inv(Em) + (1-vm) * inv(Ei));

# We can now compare the different computed stiffness tensors. We expect
# ``E_\mathrm{Reuss} \leq E_\mathrm{PeriodicBC} \leq E_\mathrm{DirichletBC} \leq
# E_\mathrm{Voigt}``. A simple thing to compare are the eigenvalues of the tensors. Here
# we look at the first eigenvalue:

ev = (first ∘ eigvals).((E_reuss, E_periodic, E_dirichlet, E_voigt))
@test issorted(ev) #src
round.(ev; digits=-8)

# Finally, we export the solution and the stress field to a VTK file. For the export we
# also compute the macroscopic part of the displacement.

chM = ConstraintHandler(dh)
add!(chM, Dirichlet(:u, Set(1:getnnodes(grid)), (x, t) -> εᴹ[Int(t)] ⋅ x, [1, 2]))
close!(chM)
uM = zeros(ndofs(dh))

vtk_grid("homogenization", dh) do vtk
    for i in 1:3
        ## Compute macroscopic solution
        update!(chM, i)
        apply!(uM, chM)
        ## Dirichlet
        vtk_point_data(vtk, dh, uM + u.dirichlet[i], "_dirichlet_$i")
        vtk_point_data(vtk, projector, σ.dirichlet[i], "σvM_dirichlet_$i")
        ## Periodic
        vtk_point_data(vtk, dh, uM + u.periodic[i], "_periodic_$i")
        vtk_point_data(vtk, projector, σ.periodic[i], "σvM_periodic_$i")
    end
end;

# Just another way to compute the stiffness for testing purposes               #src
function homogenize_test(u::Matrix, dh, cv, E_incl, E_mat)                     #src
    ĒΩ = zero(SymmetricTensor{4,2})                                            #src
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
            Ē′ = SymmetricTensor{4,2}((i, j, k, l) -> begin                    #src
                ij = i == j == 1 ? 1 : i == j == 2 ? 2 : 3                     #src
                kl = k == l == 1 ? 1 : k == l == 2 ? 2 : 3                     #src
                εij = function_symmetric_gradient(cv, qp, view(ue, :, ij)) +   #src
                          symmetric((basevec(Vec{2}, i) ⊗ basevec(Vec{2}, j))) #src
                εkl = function_symmetric_gradient(cv, qp, view(ue, :, kl)) +   #src
                          symmetric((basevec(Vec{2}, k) ⊗ basevec(Vec{2}, l))) #src
                return (εij ⊡ E ⊡ εkl) * dΩ                                    #src
            end)                                                               #src
            ĒΩ += Ē′                                                           #src
        end                                                                    #src
    end                                                                        #src
    return ĒΩ / Ω                                                              #src
end                                                                            #src

@test homogenize_test(reduce(hcat, u.dirichlet), dh, cellvalues, Ei, Em) ≈ E_dirichlet #src
@test homogenize_test(reduce(hcat, u.periodic), dh, cellvalues, Ei, Em) ≈ E_periodic #src

#md # ## [Plain program](@id homogenization-plain-program)
#md #
#md # Below follows a version of the program without any comments.
#md # The file is also available here:
#md # [computational_homogenization.jl](computational_homogenization.jl)
#md #
#md # ```julia
#md # @__CODE__
#md # ```
