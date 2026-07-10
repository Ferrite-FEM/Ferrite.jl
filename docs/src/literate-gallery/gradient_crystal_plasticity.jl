# # [Gradient crystal plasticity of a polycrystal](@id gallery-gradient-crystal-plasticity)
#
# ![](gradient_crystal_plasticity.png)
#
# ## Introduction
#
# This example solves a small-strain *gradient (higher-order) crystal plasticity* problem on a
# polycrystalline representative volume element (RVE), reproducing (on a modern Ferrite stack)
# the subscale model of
#
# > K. Carlsson, F. Larsson, K. Runesson, *Bounds on the effective response for gradient crystal
# > inelasticity based on homogenization and virtual testing*, Int. J. Numer. Methods Eng.
# > 2019;119:281–304. [doi:10.1002/nme.6050](https://doi.org/10.1002/nme.6050).
#
# Equation numbers below refer to that paper.
#
# The RVE is a periodic polycrystal generated with [Neper](https://neper.info), meshed into
# tetrahedra, and read into Ferrite with [`FerriteMeshParser`](https://github.com/Ferrite-FEM/FerriteMeshParser.jl).
# We use [`FerriteInterfaceElements`](https://github.com/Ferrite-FEM/FerriteInterfaceElements.jl)
# to *split* the mesh along the grain boundaries so that each grain carries its own microstress
# degrees of freedom (the slip systems, and hence these fields, are grain-specific), while the
# displacement is kept continuous — perfectly bonded grain boundaries — by tying the
# duplicated nodes with affine constraints. Leaving the microstresses unconstrained weakly
# enforces the *micro-hard* condition ``\gamma_\alpha = 0`` (Eq. 14a) on grain boundaries and the
# outer boundary alike, as in the paper's numerical examples (see below). We apply a macroscopic
# simple shear and compute the homogenized response ``\bar\sigma_{12}`` vs. ``\bar\gamma_{12}``
# (cf. Figure 7 in the paper), and illustrate the internal-length (grain-size) effect.
#
# !!! note "Scope"
#     The paper develops *bounds* on the effective response by combining two formulations
#     (a *primal* and a *semi-dual* one) with four sets of RVE boundary conditions — Dirichlet or
#     Neumann on the displacement, and micro-hard (Eq. 14a) or micro-free (Eq. 14b) on the
#     microstresses. This tutorial implements **only one** of these combinations: the **semi-dual**
#     formulation with a **Dirichlet** displacement boundary condition and **micro-hard**
#     microstress conditions — the paper's ``u``D-``\xi``N case (its upper-bound variant), with
#     micro-hard grain boundaries as in the paper's numerical examples. Micro-free conditions
#     (essential constraints on ``\boldsymbol\xi`` in this format), the Neumann/periodic
#     displacement conditions, and the primal formulation are not covered here.
#
# ### Constitutive model
#
# Each grain is an FCC single crystal with 12 slip systems ``(\boldsymbol s_\alpha,\boldsymbol m_\alpha)``.
# Small strains split additively into elastic and plastic parts, and the plastic strain
# originates from slip on the slip systems (Eq. 3),
# ```math
# \boldsymbol\varepsilon = \boldsymbol\varepsilon^\mathrm{e} + \boldsymbol\varepsilon^\mathrm{p},
# \qquad
# \boldsymbol\varepsilon^\mathrm{p} = \sum_\alpha \gamma_\alpha\, \boldsymbol P_\alpha,
# \qquad
# \boldsymbol P_\alpha = \operatorname{sym}(\boldsymbol s_\alpha\otimes\boldsymbol m_\alpha),
# ```
# with stress ``\boldsymbol\sigma = \mathbb C : (\boldsymbol\varepsilon-\boldsymbol\varepsilon^\mathrm p)``
# (Eq. 84). The slips ``\gamma_\alpha`` follow a viscoplastic (Norton) flow rule (Eqs. 86–87)
# driven by a resolved shear stress (the Schmid stress, Eq. 85) that is augmented by the
# *gradient microstress divergence* ``\chi_\alpha`` (from the microforce balance, Eqs. 11b, 13):
# ```math
# \dot\gamma_\alpha = \frac{1}{t_*}\left(\frac{|\tau_\alpha|}{C}\right)^n \operatorname{sign}(\tau_\alpha),
# \qquad
# \tau_\alpha = \boldsymbol\sigma:\boldsymbol P_\alpha + \chi_\alpha .
# ```
# The higher-order part is written in a *dual* (semi-dual) form: the global unknowns are the
# displacement ``\boldsymbol u`` and, per slip system, two scalar *microstress* fields
# ``\xi^\perp_\alpha`` (edge) and ``\xi^\odot_\alpha`` (screw), obtained by projecting the vector
# microstress ``\boldsymbol\xi_\alpha = \xi^\perp_\alpha\boldsymbol s_\alpha + \xi^\odot_\alpha\boldsymbol l_\alpha``
# (Eq. 92, with ``\boldsymbol l_\alpha = \boldsymbol m_\alpha\times\boldsymbol s_\alpha`` the paper's
# ``\boldsymbol k_\alpha``). They enter the flow rule through
# ```math
# \chi_\alpha = \boldsymbol s_\alpha\cdot\nabla\xi^\perp_\alpha + \boldsymbol l_\alpha\cdot\nabla\xi^\odot_\alpha,
# ```
# and are governed by a microforce balance. Its weak form is Eq. 30b; with the quadratic defect
# energy (Eqs. 88, 90) the dual relation is ``g^\perp_\alpha = \xi^\perp_\alpha/H_\mathrm g`` (Eqs. 94–95),
# so for every test function ``\delta\xi``
# ```math
# \int_\Omega\!\Big(\tfrac{\xi^\perp_\alpha}{H_\mathrm g}\,\delta\xi
#   + \gamma_\alpha\,(\boldsymbol s_\alpha\cdot\nabla\delta\xi)\Big)\,\mathrm d\Omega = 0 ,
# ```
# where ``H_\mathrm g = H\ell^2`` is the defect-energy modulus (Eq. 88), ``\ell`` an internal
# length (and likewise for ``\xi^\odot`` with ``\boldsymbol l_\alpha``). Note that ``\delta\xi``
# is unrestricted on all boundaries: integrating by parts shows that this weakly enforces
# ``\gamma_\alpha = 0`` there — the *micro-hard* condition (Eq. 14a). In the semi-dual format the
# essential/natural roles are swapped compared to the primal format (Remark 5 in the paper);
# micro-free (Eq. 14b) would instead require the essential constraint
# ``\boldsymbol\xi_\alpha\cdot\boldsymbol n = 0``. Setting ``H_\mathrm g\to 0``
# suppresses the gradient term and recovers conventional *local* crystal plasticity; increasing
# ``\ell`` strengthens the size effect. The macroscopic stress is the volume average
# ``\bar{\boldsymbol\sigma}=\langle\boldsymbol\sigma\rangle`` (Eq. 38), obtained by imposing
# ``\boldsymbol u=\bar{\boldsymbol\varepsilon}\cdot\boldsymbol x`` on the RVE boundary (Eq. 36).
#
# ## Commented program

using Ferrite, FerriteMeshParser, FerriteInterfaceElements
using Tensors, LinearAlgebra, Random, TimerOutputs
using LinearSolve: LinearProblem, init, solve!, UMFPACKFactorization

# OpenBLAS's threading has large overhead for the many small (12×12) factorizations in the
# local return map; single-threaded BLAS is faster here.
BLAS.set_num_threads(1)
import Plots
import Downloads

# ### Crystal plasticity material
#
# The 12 FCC slip systems ``\{111\}\langle110\rangle`` (4 planes × 3 directions):

function fcc_slip_systems()
    planes = [Vec{3}((1.0, 1.0, 1.0)), Vec{3}((-1.0, 1.0, 1.0)), Vec{3}((1.0, -1.0, 1.0)), Vec{3}((1.0, 1.0, -1.0))]
    dirs = [
        [Vec{3}((1.0, -1.0, 0.0)), Vec{3}((1.0, 0.0, -1.0)), Vec{3}((0.0, 1.0, -1.0))],
        [Vec{3}((1.0, 1.0, 0.0)), Vec{3}((1.0, 0.0, 1.0)), Vec{3}((0.0, 1.0, -1.0))],
        [Vec{3}((1.0, 1.0, 0.0)), Vec{3}((1.0, 0.0, -1.0)), Vec{3}((0.0, 1.0, 1.0))],
        [Vec{3}((1.0, -1.0, 0.0)), Vec{3}((1.0, 0.0, 1.0)), Vec{3}((0.0, 1.0, 1.0))],
    ]
    s = Vec{3, Float64}[]
    m = Vec{3, Float64}[]
    for (mp, ds) in zip(planes, dirs), d in ds
        push!(m, mp / norm(mp))
        push!(s, d / norm(d))
    end
    return s, m
end

# A uniformly random rotation (from a random unit quaternion), used to give each grain a
# random crystal orientation.
function rand_rotation(rng)
    u1 = rand(rng)
    u2 = rand(rng)
    u3 = rand(rng)
    w = sqrt(1 - u1) * sinpi(2u2)
    x = sqrt(1 - u1) * cospi(2u2)
    y = sqrt(u1) * sinpi(2u3)
    z = sqrt(u1) * cospi(2u3)
    return Tensor{2, 3}(
        (
            1 - 2(y^2 + z^2), 2(x * y + z * w), 2(x * z - y * w),
            2(x * y - z * w), 1 - 2(x^2 + z^2), 2(y * z + x * w),
            2(x * z + y * w), 2(y * z - x * w), 1 - 2(x^2 + y^2),
        )
    )
end

# The material stores the (rotated) slip systems and precomputes the quantities needed in the
# local return map: ``\mathbb C:\boldsymbol P_\alpha`` and ``D_{\alpha\beta}=\boldsymbol P_\alpha:\mathbb C:\boldsymbol P_\beta``.
struct CrystalPlasticity
    n::Float64
    C::Float64
    tstar::Float64
    ℂ::SymmetricTensor{4, 3, Float64, 36}
    P::Vector{SymmetricTensor{2, 3, Float64, 6}}
    s::Vector{Vec{3, Float64}}
    l::Vector{Vec{3, Float64}}
    CP::Vector{SymmetricTensor{2, 3, Float64, 6}}
    D::Matrix{Float64}
end

function CrystalPlasticity(; E, ν, n, C, tstar, R)
    λ = E * ν / ((1 + ν) * (1 - 2ν))
    μ = E / (2(1 + ν))
    δ(i, j) = i == j ? 1.0 : 0.0
    ℂ = SymmetricTensor{4, 3}((i, j, k, l) -> λ * δ(i, j) * δ(k, l) + μ * (δ(i, k) * δ(j, l) + δ(i, l) * δ(j, k)))
    s0, m0 = fcc_slip_systems()
    N = length(s0)
    s = [R ⋅ sα for sα in s0]
    m = [R ⋅ mα for mα in m0]
    l = [m[α] × s[α] for α in 1:N]
    P = [symmetric(s[α] ⊗ m[α]) for α in 1:N]
    CP = [ℂ ⊡ P[α] for α in 1:N]
    D = [P[α] ⊡ ℂ ⊡ P[β] for α in 1:N, β in 1:N]
    return CrystalPlasticity(n, C, tstar, ℂ, P, s, l, CP, D)
end

# One random-orientation material per grain (seeded for reproducibility). Units are MPa;
# the elastic and viscous parameters follow Table 1 of the paper (the gradient parameters ``H``
# and ``\ell`` are set per simulation below).
function build_materials(ngrain)
    rng = MersenneTwister(1234)
    return [CrystalPlasticity(; E = 200.0e3, ν = 0.3, n = 2.0, C = 1.0e3, tstar = 1.0e3, R = rand_rotation(rng)) for _ in 1:ngrain]
end

# ### Local return map
#
# Given the strain ``\boldsymbol\varepsilon`` and the per-slip gradient contribution
# ``\chi_\alpha`` at a quadrature point, solve the flow rule (Eq. 87, backward Euler as in Eq. 27)
# for the slips ``\gamma_\alpha``, and return the stress together with the consistent sensitivities
# ``\partial\boldsymbol\sigma/\partial\boldsymbol\varepsilon``, ``\partial\boldsymbol\sigma/\partial\chi_\beta``,
# ``\partial\gamma_\alpha/\partial\boldsymbol\varepsilon`` and ``\partial\gamma_\alpha/\partial\chi_\beta``.
# The slip-system system is small but grows with the number of slip systems, so we use ordinary
# (mutable) arrays rather than `StaticArrays`. To avoid allocating them anew at every quadrature
# point, the caller passes in a workspace of preallocated arrays, and the linear algebra is done
# in place with `lu!`/`ldiv!`. Note that the returned arrays are views into this workspace: they
# are only valid until the next `local_solve` call.
function local_solve_workspace(N)
    return (;
        γ = zeros(N), τ = zeros(N), dg = zeros(N), R = zeros(N),
        J = zeros(N, N), Jinv = zeros(N, N), dγdχ = zeros(N, N),
        dγdε = fill(zero(SymmetricTensor{2, 3}), N), dσdχ = fill(zero(SymmetricTensor{2, 3}), N),
        χ = zeros(N), ξp = zeros(N), ξo = zeros(N),
    )
end

# Stress, resolved shear ``\tau_\alpha`` and its flow-rate derivative, at the current slips `γ`.
function local_state!(τ, dg, mp::CrystalPlasticity, ε, χ, γ)
    N = length(γ)
    εp = zero(ε)
    for α in 1:N
        εp += γ[α] * mp.P[α]
    end
    σ = mp.ℂ ⊡ (ε - εp)
    for α in 1:N
        τ[α] = (σ ⊡ mp.P[α]) + χ[α]
        dg[α] = mp.n / mp.C * (abs(τ[α]) / mp.C)^(mp.n - 1)
    end
    return σ
end

function local_solve(mp::CrystalPlasticity, ε, χ, γn, Δt, ws)
    N = length(mp.P)
    c = Δt / mp.tstar
    (; γ, τ, dg, R, J, Jinv, dγdχ, dγdε, dσdχ) = ws
    copyto!(γ, γn)
    local σ
    converged = false
    for _ in 1:100
        σ = local_state!(τ, dg, mp, ε, χ, γ)
        for α in 1:N
            R[α] = γ[α] - γn[α] - c * (abs(τ[α]) / mp.C)^mp.n * sign(τ[α])
            for β in 1:N
                J[α, β] = (α == β ? 1.0 : 0.0) + c * dg[α] * mp.D[α, β]
            end
        end
        if norm(R) < 1.0e-11
            converged = true
            break
        end
        ldiv!(lu!(J), R)
        γ .-= R
    end
    converged || error("local return map did not converge")
    ## consistent sensitivities from the converged local Jacobian
    σ = local_state!(τ, dg, mp, ε, χ, γ)
    for α in 1:N, β in 1:N
        J[α, β] = (α == β ? 1.0 : 0.0) + c * dg[α] * mp.D[α, β]
    end
    copyto!(Jinv, I)
    ldiv!(lu!(J), Jinv)
    for α in 1:N, β in 1:N
        dγdχ[α, β] = Jinv[α, β] * c * dg[β]                        # ∂γ_α/∂χ_β
    end
    fill!(dγdε, zero(ε))                                           # ∂γ_α/∂ε
    for α in 1:N, κ in 1:N
        dγdε[α] += Jinv[α, κ] * c * dg[κ] * mp.CP[κ]
    end
    dσdε = mp.ℂ                                                    # ∂σ/∂ε
    for β in 1:N
        dσdε -= mp.CP[β] ⊗ dγdε[β]
    end
    fill!(dσdχ, zero(ε))                                           # ∂σ/∂χ_β
    for β in 1:N, α in 1:N
        dσdχ[β] -= mp.CP[α] * dγdχ[α, β]
    end
    return σ, γ, dσdε, dσdχ, dγdε, dγdχ
end

# ### Bulk element
#
# The coupled bulk element assembles the equilibrium residual (Eq. 30a) and the two microforce
# balances (Eq. 30b, one for ``\xi^\perp`` and one for ``\xi^\odot``), together with the consistent
# tangent. The ξ equations are written with an overall minus sign so that the tangent is symmetric.
# `dofmap` holds the local (within-cell) dof ranges of `:u`, `ξperp_α`, `ξodot_α`.
function bulk_element!(re, Ke, cv, ae, mp::CrystalPlasticity, γn, Δt, Hg, dofmap, ws, to)
    fill!(re, 0)
    fill!(Ke, 0)
    N = length(mp.P)
    nu = getnbasefunctions(cv.u)
    nn = getnbasefunctions(cv.ξ)
    (; χ, ξp, ξo, Ip, Io, S, L) = ws
    ue = @view ae[dofmap.u]
    for qp in 1:getnquadpoints(cv)
        dΩ = getdetJdV(cv, qp)
        ε = function_symmetric_gradient(cv.u, qp, ue)
        ## gradient contribution χ_α = s_α·∇ξ⊥_α + l_α·∇ξ⊙_α, and nodal ξ values
        for α in 1:N
            xp = @view ae[dofmap.p[α]]
            xo = @view ae[dofmap.o[α]]
            ξp[α] = function_value(cv.ξ, qp, xp)
            ξo[α] = function_value(cv.ξ, qp, xo)
            ∇ξp = function_gradient(cv.ξ, qp, xp)
            ∇ξo = function_gradient(cv.ξ, qp, xo)
            χ[α] = mp.s[α] ⋅ ∇ξp + mp.l[α] ⋅ ∇ξo
        end
        ## slip-projected shape gradients, used repeatedly in the tangent blocks below
        for k in 1:nn
            ∇Nk = shape_gradient(cv.ξ, qp, k)
            for α in 1:N
                S[k, α] = mp.s[α] ⋅ ∇Nk
                L[k, α] = mp.l[α] ⋅ ∇Nk
            end
        end
        σ, γ, dσdε, dσdχ, dγdε, dγdχ = @timeit to "local solve" local_solve(mp, ε, χ, γn[qp], Δt, ws)
        ## residual: equilibrium (Eq. 30a)
        for i in 1:nu
            δε = shape_symmetric_gradient(cv.u, qp, i)
            re[i] += (δε ⊡ σ) * dΩ
        end
        ## residual: microforce balances (Eq. 30b)
        @inbounds for k in 1:nn
            Nk = shape_value(cv.ξ, qp, k)
            for α in 1:N
                re[Ip[k, α]] += ((ξp[α] / Hg) * Nk + γ[α] * S[k, α]) * dΩ
                re[Io[k, α]] += ((ξo[α] / Hg) * Nk + γ[α] * L[k, α]) * dΩ
            end
        end
        ## tangent: u–u  (δε_i : dσ/dε : δε_j)
        for i in 1:nu
            δεi = shape_symmetric_gradient(cv.u, qp, i)
            for j in 1:nu
                δεj = shape_symmetric_gradient(cv.u, qp, j)
                Ke[i, j] += (δεi ⊡ dσdε ⊡ δεj) * dΩ
            end
        end
        ## tangent: u–ξ and ξ–u
        @inbounds for i in 1:nu
            δεi = shape_symmetric_gradient(cv.u, qp, i)
            for β in 1:N
                c = (δεi ⊡ dσdχ[β]) * dΩ
                for k in 1:nn
                    Ke[i, Ip[k, β]] += c * S[k, β]
                    Ke[i, Io[k, β]] += c * L[k, β]
                end
            end
        end
        @inbounds for α in 1:N
            for j in 1:nu
                δεj = shape_symmetric_gradient(cv.u, qp, j)
                v = (dγdε[α] ⊡ δεj) * dΩ
                for k in 1:nn
                    Ke[Ip[k, α], j] += S[k, α] * v
                    Ke[Io[k, α], j] += L[k, α] * v
                end
            end
        end
        ## tangent: ξ–ξ, gradient part (loop order keeps the Ke writes column-local) …
        @inbounds for β in 1:N, kp in 1:nn
            sβ = S[kp, β] * dΩ
            lβ = L[kp, β] * dΩ
            for α in 1:N
                gs = dγdχ[α, β] * sβ
                gl = dγdχ[α, β] * lβ
                for k in 1:nn
                    Ke[Ip[k, α], Ip[kp, β]] += S[k, α] * gs
                    Ke[Ip[k, α], Io[kp, β]] += S[k, α] * gl
                    Ke[Io[k, α], Ip[kp, β]] += L[k, α] * gs
                    Ke[Io[k, α], Io[kp, β]] += L[k, α] * gl
                end
            end
        end
        ## … and mass-type part (only on the α = β diagonal)
        @inbounds for kp in 1:nn, k in 1:nn
            d = shape_value(cv.ξ, qp, k) * shape_value(cv.ξ, qp, kp) / Hg * dΩ
            for α in 1:N
                Ke[Ip[k, α], Ip[kp, α]] += d
                Ke[Io[k, α], Io[kp, α]] += d
            end
        end
    end
    @views re[(nu + 1):end] .*= -1
    @views Ke[(nu + 1):end, :] .*= -1
    return re, Ke
end

# ### Mesh
#
# The mesh is produced with [Neper](https://neper.info): a Voronoi tessellation into `ngrain`
# grains, meshed into linear tetrahedra. The two commands are wrapped in `generate_mesh`, which is
# only invoked when `regenerate = true` is passed (and requires the `neper` executable on the
# `PATH`); otherwise the committed `.inp` file next to this tutorial is used. `rcl` is Neper's
# relative characteristic length — smaller means a finer mesh.
function generate_mesh(meshfile, ngrain, rcl)
    base = first(splitext(meshfile))
    run(`neper -T -n $ngrain -id 1 -dim 3 -domain "cube(1,1,1)" -o $base`)
    run(`neper -M $base.tess -order 1 -rcl $rcl -format inp -o $base`)
    return meshfile
end

# Neper writes one element set `poly1`, `poly2`, … per grain, which `FerriteMeshParser` turns
# into cell sets. It also writes node sets for the faces, edges and corners of the cube; we skip
# the automatic facet-set generation (`generate_facetsets = false`, which would emit warnings for
# the edge/corner sets) and instead define the six RVE faces ourselves from coordinate predicates.
# `insert_interfaces` then duplicates the nodes on grain boundaries so that neighbouring grains no
# longer share degrees of freedom there — this lets each grain hold its own microstress fields
# ``\xi`` (the slip systems differ between grains, so these fields cannot be shared).
# Displacement continuity is restored later with constraints,
# so the interface cells themselves are not assembled.
function load_grid(meshfile, ngrain; regenerate = false, rcl = 1.0)
    if regenerate
        generate_mesh(meshfile, ngrain, rcl)
    elseif !isfile(meshfile)
        repo_copy = joinpath(pkgdir(Ferrite), "docs", "src", "literate-gallery", meshfile)
        isfile(repo_copy) ? cp(repo_copy, meshfile) : Downloads.download(Ferrite.asset_url(meshfile), meshfile)
    end
    rawgrid = get_ferrite_grid(meshfile; generate_facetsets = false)
    for (name, f) in (
            "x0" => x -> abs(x[1]) < 1.0e-6, "x1" => x -> abs(x[1] - 1) < 1.0e-6,
            "y0" => x -> abs(x[2]) < 1.0e-6, "y1" => x -> abs(x[2] - 1) < 1.0e-6,
            "z0" => x -> abs(x[3]) < 1.0e-6, "z1" => x -> abs(x[3] - 1) < 1.0e-6,
        )
        addfacetset!(rawgrid, name, f)
    end
    grainnames = ["poly$i" for i in 1:ngrain]
    grid = insert_interfaces(rawgrid, grainnames)
    nbulk = getncells(rawgrid)
    bulkset = union((getcellset(grid, n) for n in grainnames)...)
    ## map each bulk cell to its grain (used to pick the crystal orientation)
    cellgrain = zeros(Int, nbulk)
    for (g, name) in enumerate(grainnames), c in getcellset(grid, name)
        cellgrain[c] = g
    end
    return grid, bulkset, nbulk, cellgrain
end

# ### Degrees of freedom
#
# The displacement is a vector Lagrange field. The 24 microstress fields
# ``\xi^\perp_1..\xi^\perp_{12}`` and ``\xi^\odot_1..\xi^\odot_{12}`` are scalar Lagrange fields.
# All fields live in the bulk grains; the interface cells are not part of any `SubDofHandler`
# and hence carry no dofs. `dofmap` records the local dof range of each field for the element routine.
function build_dofhandler(grid, bulkset, ip_u, ip_s, nslip)
    dh = DofHandler(grid)
    sdh = SubDofHandler(dh, bulkset)
    add!(sdh, :u, ip_u)
    for α in 1:nslip
        add!(sdh, Symbol("ξperp_$α"), ip_s)
    end
    for α in 1:nslip
        add!(sdh, Symbol("ξodot_$α"), ip_s)
    end
    close!(dh)
    dofmap = (
        u = dof_range(sdh, :u),
        p = [dof_range(sdh, Symbol("ξperp_$α")) for α in 1:nslip],
        o = [dof_range(sdh, Symbol("ξodot_$α")) for α in 1:nslip],
    )
    return dh, sdh, dofmap
end

# Bundle the whole discretization (grid, materials, dof handler, cell values) into one object.
# A single `MultiFieldCellValues` holds the values for both fields, sharing one quadrature rule
# and geometric mapping: `cv.u` for the displacement and `cv.ξ` for the (scalar) microstresses.
# The rule is second order since the ``\xi``-equations contain the mass-type term
# ``N_k N_p / H_\mathrm g``, which a one-point rule would under-integrate (to a rank-one element
# matrix).
function build_model(meshfile, ngrain; regenerate = false, rcl = 1.0)
    grid, bulkset, nbulk, cellgrain = load_grid(meshfile, ngrain; regenerate, rcl)
    materials = build_materials(ngrain)
    nslip = length(first(materials).s)   # number of slip systems (12 for FCC)
    ip_u = Lagrange{RefTetrahedron, 1}()^3
    ip_s = Lagrange{RefTetrahedron, 1}()
    dh, sdh, dofmap = build_dofhandler(grid, bulkset, ip_u, ip_s, nslip)
    qr = QuadratureRule{RefTetrahedron}(2)
    cv = MultiFieldCellValues(qr, (u = ip_u, ξ = ip_s))
    return (; grid, bulkset, nbulk, cellgrain, materials, nslip, dh, sdh, dofmap, cv)
end

# ### Boundary conditions
#
# We impose a macroscopic simple shear ``\bar{\boldsymbol\varepsilon}(t)=\bar\Gamma(t)\,\operatorname{sym}(\boldsymbol e_1\otimes\boldsymbol e_2)``
# as a Dirichlet condition ``\boldsymbol u = \bar{\boldsymbol\varepsilon}(t)\cdot\boldsymbol x`` on the RVE
# boundary (Eq. 36), ramping the engineering shear ``\bar\gamma_{12}=2\bar\varepsilon_{12}`` up to `Γmax`.
# The microstress fields carry no essential boundary condition, which (as discussed in the
# introduction) weakly enforces the micro-hard condition ``\gamma_\alpha = 0`` (Eq. 14a) also on
# the outer RVE boundary.
function macroscopic_strain(t, Γmax, T)
    e1 = Vec{3}((1.0, 0.0, 0.0))
    e2 = Vec{3}((0.0, 1.0, 0.0))
    return symmetric((Γmax * t / T) * (e1 ⊗ e2))
end

# To keep the grain boundaries perfectly bonded, we tie together the displacement dofs
# of the coincident nodes that `insert_interfaces` created. We build a node → `u`-dof lookup
# (linear tetrahedron, so vertex ``k`` component ``c`` sits at local index ``(k-1)\cdot 3 + c``),
# group nodes by coordinate, and, for every interior group with more than one node, constrain the
# extra copies to a master with an `AffineConstraint`. Groups on the outer boundary are already made
# continuous by the Dirichlet condition, so they are skipped — and since a dof must not be both
# affine-tied and Dirichlet-prescribed, we verify after `close!` that the Dirichlet condition
# indeed covers all of their dofs.
function build_constraints(model, loading)
    (; grid, bulkset, dh, dofmap) = model
    (; Γmax, T) = loading
    udof = Dict{Int, NTuple{3, Int}}()
    for cc in CellIterator(dh, bulkset)
        cdofs = celldofs(cc)
        ur = dofmap.u
        ns = getcells(grid, cellid(cc)).nodes
        for (k, n) in enumerate(ns)
            udof[n] = (cdofs[ur[(k - 1) * 3 + 1]], cdofs[ur[(k - 1) * 3 + 2]], cdofs[ur[(k - 1) * 3 + 3]])
        end
    end
    onboundary(x) = any(c -> abs(c) < 1.0e-6 || abs(c - 1) < 1.0e-6, x)   # unit-cube RVE boundary (same tolerance as the facet sets)
    groups = Dict{Vec{3, Float64}, Vector{Int}}()
    for n in 1:getnnodes(grid)
        push!(get!(groups, get_node_coordinate(grid, n), Int[]), n)
    end
    ∂Ω = union((getfacetset(grid, s) for s in ("x0", "x1", "y0", "y1", "z0", "z1"))...)
    ch = ConstraintHandler(dh)
    add!(ch, Dirichlet(:u, ∂Ω, (x, t) -> macroscopic_strain(t, Γmax, T) ⋅ x, [1, 2, 3]))
    for (x, ns) in groups
        (length(ns) == 1 || onboundary(x)) && continue
        master = first(ns)
        for other in ns[2:end], c in 1:3
            add!(ch, AffineConstraint(udof[other][c], [udof[master][c] => 1.0], 0.0))
        end
    end
    close!(ch)
    ## the skipped groups on the boundary must be fully covered by the Dirichlet condition
    prescribed = Set(ch.prescribed_dofs)
    for (x, ns) in groups
        (length(ns) > 1 && onboundary(x)) || continue
        for n in ns, c in 1:3
            udof[n][c] in prescribed || error("unconstrained duplicated boundary node at $x")
        end
    end
    return ch
end

# ### Global assembly and homogenization

function assemble_system!(K, r, model, u, state, Δt, Hg, to)
    (; dh, sdh, bulkset, cv, materials, cellgrain, dofmap, nslip) = model
    assembler = start_assemble(K, r)
    nb = ndofs_per_cell(sdh)
    re = zeros(nb)
    Ke = zeros(nb, nb)
    ## extend the local-solve workspace with element scratch: the local dof indices of
    ## ξ⊥_α, ξ⊙_α (identical for every cell) and the slip-projected shape gradients
    nn = getnbasefunctions(cv.ξ)
    ws = (;
        local_solve_workspace(nslip)...,
        Ip = [dofmap.p[α][k] for k in 1:nn, α in 1:nslip],
        Io = [dofmap.o[α][k] for k in 1:nn, α in 1:nslip],
        S = zeros(nn, nslip), L = zeros(nn, nslip),
    )
    for cc in CellIterator(dh, bulkset)
        reinit!(cv, cc)
        cid = cellid(cc)
        bulk_element!(re, Ke, cv, u[celldofs(cc)], materials[cellgrain[cid]], state[cid], Δt, Hg, dofmap, ws, to)
        assemble!(assembler, celldofs(cc), Ke, re)
    end
    return K, r
end

# After each converged step we commit the slip state and compute the homogenized stress
# ``\bar{\boldsymbol\sigma}=\tfrac1V\int_\Omega\boldsymbol\sigma\,\mathrm d\Omega`` (Eq. 38).
# We also record the cell averages of the stress and the accumulated slip
# ``\sum_\alpha|\gamma_\alpha|`` for the VTK export below — recomputing them from the committed
# state after the fact would implicitly take one more time step.
function homogenize_and_commit!(model, u, state, Δt)
    (; dh, bulkset, nbulk, cv, materials, cellgrain, dofmap, nslip) = model
    σ̄ = zero(SymmetricTensor{2, 3})
    V = 0.0
    σcell = fill(zero(SymmetricTensor{2, 3}), nbulk)
    γeff = zeros(nbulk)
    ws = local_solve_workspace(nslip)
    for cc in CellIterator(dh, bulkset)
        reinit!(cv, cc)
        cid = cellid(cc)
        ae = u[celldofs(cc)]
        mp = materials[cellgrain[cid]]
        ue = @view ae[dofmap.u]
        Vc = 0.0
        for qp in 1:getnquadpoints(cv)
            dΩ = getdetJdV(cv, qp)
            ε = function_symmetric_gradient(cv.u, qp, ue)
            χ = ws.χ
            for α in 1:nslip
                ∇ξp = function_gradient(cv.ξ, qp, (@view ae[dofmap.p[α]]))
                ∇ξo = function_gradient(cv.ξ, qp, (@view ae[dofmap.o[α]]))
                χ[α] = mp.s[α] ⋅ ∇ξp + mp.l[α] ⋅ ∇ξo
            end
            σ, γ = local_solve(mp, ε, χ, state[cid][qp], Δt, ws)
            state[cid][qp] .= γ
            σcell[cid] += σ * dΩ
            γeff[cid] += sum(abs, γ) * dΩ
            Vc += dΩ
        end
        σ̄ += σcell[cid]
        V += Vc
        σcell[cid] /= Vc
        γeff[cid] /= Vc
    end
    return σ̄ / V, σcell, γeff
end

# ### Time-stepping driver
#
# For a given defect-energy modulus `Hg` (= ``H\ell^2``), ramp the shear over `T/Δt` steps,
# solving the coupled nonlinear system with Newton at each step, and return the
# ``(\bar\gamma_{12}, \bar\sigma_{12})`` history. A `TimerOutput` records the time spent in the
# main phases (assembly, and within it the local return map; the linear solve; homogenization).
#
# The linearized system is solved through
# [LinearSolve.jl](https://github.com/SciML/LinearSolve.jl), so the algorithm can be swapped by
# passing e.g. `linsolve = KrylovJL_GMRES()` instead of the default sparse LU. The cache created
# by `init` is reused across all solves; since the sparsity pattern never changes, the symbolic
# factorization is only computed once.
function simulate(model, ch, loading, Hg; linsolve = UMFPACKFactorization())
    (; dh, nbulk, nslip, cv) = model
    (; Γmax, T, Δt) = loading
    to = TimerOutput()
    K = allocate_matrix(dh, ch)
    r = zeros(ndofs(dh))
    u = zeros(ndofs(dh))
    lincache = init(LinearProblem(K, r), linsolve)
    state = [[zeros(nslip) for _ in 1:getnquadpoints(cv)] for _ in 1:nbulk]
    γhist = [0.0]
    σhist = [0.0]
    times = Δt:Δt:T
    nsteps = length(times)
    local σcell, γeff
    for (istep, t) in enumerate(times)
        update!(ch, t)
        apply!(u, ch)
        iters = 0
        converged = false
        for iter in 1:25
            iters = iter
            @timeit to "assemble" assemble_system!(K, r, model, u, state, Δt, Hg, to)
            apply_zero!(K, r, ch)
            res = norm(r[ch.free_dofs])
            ## in an interactive session, overwrite a single status line with the Newton residual
            if isinteractive()
                print("\r  step $(lpad(istep, 2))/$nsteps   iter $(lpad(iter, 2))   ‖r‖ = $(round(res, sigdigits = 3))        ")
                flush(stdout)
            end
            if res < 1.0e-5
                converged = true
                break
            end
            Δu = @timeit to "linear solve" begin
                lincache.A = K   # flag K and r as updated
                lincache.b = r
                solve!(lincache).u
            end
            apply_zero!(Δu, ch)
            u .-= Δu
        end
        converged || error("global Newton did not converge in $iters iterations")
        σ̄, σcell, γeff = @timeit to "homogenize" homogenize_and_commit!(model, u, state, Δt)
        push!(γhist, Γmax * t / T)
        push!(σhist, σ̄[1, 2])
        ## finished step: overwrite the status line with a summary and move on
        println("\r  step $(lpad(istep, 2))/$nsteps   γ̄₁₂ = $(rpad(round(Γmax * t / T, digits = 4), 6))   σ̄₁₂ = $(round(σ̄[1, 2], digits = 1)) MPa   ($iters iters)      ")
    end
    return (; γ = γhist, σ = σhist, u, σcell, γeff, to)
end

# ### VTK output
#
# Write the final (deformed) polycrystal to a VTK file for ParaView: the displacement and all
# microstress dofs as nodal fields, and per-cell the grain id, the shear stress ``\sigma_{12}``,
# the von Mises stress, and the accumulated slip ``\sum_\alpha|\gamma_\alpha|`` (the cell averages
# recorded when the last step was committed). The `InterfaceCell`s from `insert_interfaces` cannot
# be written to VTK, so we build a grid of the bulk tetrahedra only (sharing the same nodes) for
# the output.
function export_vtk(filename, model, u, σcell, γeff)
    (; grid, nbulk, cellgrain, dh, nslip) = model
    σ12 = [σ[1, 2] for σ in σcell]
    σvM = [sqrt(3 / 2) * norm(dev(σ)) for σ in σcell]
    bulkgrid = Grid([getcells(grid, c) for c in 1:nbulk], getnodes(grid))
    return VTKGridFile(filename, bulkgrid) do vtk
        write_node_data(vtk, evaluate_at_grid_nodes(dh, u, :u), "u")
        for α in 1:nslip
            write_node_data(vtk, evaluate_at_grid_nodes(dh, u, Symbol("ξperp_$α")), "xiperp_$α")
            write_node_data(vtk, evaluate_at_grid_nodes(dh, u, Symbol("ξodot_$α")), "xiodot_$α")
        end
        write_cell_data(vtk, cellgrain, "grain")
        write_cell_data(vtk, σ12, "sigma_12")
        write_cell_data(vtk, σvM, "sigma_vonMises")
        write_cell_data(vtk, γeff, "gamma_eff")
    end
end

# ### Length-scale (grain-size) study
#
# We sweep the internal length ``\ell`` (relative to the RVE size) at a fixed gradient modulus
# ``H``. A vanishing ``\ell`` gives the local response (plastic saturation); increasing ``\ell``
# strengthens the slip pile-up at the micro-hard grain boundaries and stiffens the macroscopic
# response. The loading follows the paper (``\bar\Gamma = 0.1`` ramped over ``T = 15`` s) with a
# somewhat coarser time step; ``H`` and the swept ``\ell`` values are chosen to give a clear size
# effect on this small RVE rather than to match Table 1 (where ``\ell/L \approx 10^{-3}``), so
# the curves are not quantitatively comparable to Figure 7 of the paper.

ngrain = 10
model = build_model("gradient_crystal_plasticity.inp", ngrain; regenerate = false)
loading = (Γmax = 0.1, T = 15.0, Δt = 1.0)
ch = build_constraints(model, loading)

H = 1.0e5
ℓs = [0.005, 0.05, 0.1, 0.15]
results = map(ℓs) do ℓ
    println("Internal length ℓ = $ℓ  (Hg = H·ℓ² = $(H * ℓ^2)):")
    (ℓ, simulate(model, ch, loading, H * ℓ^2))
end

plt = Plots.plot(;
    xlabel = "shear strain  γ̄₁₂", ylabel = "shear stress  σ̄₁₂ [MPa]",
    legend = :bottomright, title = "Polycrystal size effect"
)
for (ℓ, res) in results
    Plots.plot!(plt, res.γ, res.σ; marker = :circle, label = "ℓ = $(ℓ)")
end
Plots.savefig(plt, "gradient_crystal_plasticity.png")

# The smallest ``\ell`` recovers the local response (note the plastic saturation), while
# larger internal lengths harden the polycrystal — the grain-size effect.
#
# Finally, we write the deformed polycrystal for the smallest internal length to a VTK file,
# containing the displacement and microstress dofs together with the resulting stress fields.

ℓ_vtk, res_vtk = first(results)
export_vtk("gradient_crystal_plasticity", model, res_vtk.u, res_vtk.σcell, res_vtk.γeff)

# The `TimerOutput` collected during that last simulation shows where the time is spent — the
# element assembly (with the local return map nested inside), the linear solve, and homogenization:

print_timer(res_vtk.to)

#md # ## [Plain program](@id gallery-gradient-crystal-plasticity-plain-program)
#md #
#md # Here follows a version of the program without any comments.
#md #
#md # ```julia
#md # @__CODE__
#md # ```
