# # [Parameter identification and inverse problems](@id tutorial-parameter-identification)
#
# **Keywords**: *inverse problem*, *parameter identification*, *adjoint method*,
# *automatic differentiation*, *implicit differentiation*, *optimization*, *plasticity*.
#
# ## Introduction
#
# So far in the tutorials we have solved *forward* problems: given a set of material
# parameters and boundary conditions, compute the response of the structure. This
# tutorial is about the *inverse* problem: given (noisy) measurements of the response,
# recover the material parameters that produced them. This is the bread and butter of
# *model calibration* — fitting a constitutive model to experimental data.
#
# The workflow we build is generic:
#
# 1. **Forward solve.** Solve the finite element problem for a trial set of parameters
#    ``\boldsymbol{\theta}``.
# 2. **Synthetic data.** For a "true" parameter set we solve once, sample the solution
#    at a handful of sensor locations and add measurement noise. This plays the role of
#    experimental data.
# 3. **Misfit.** Define an objective ``J(\boldsymbol{\theta})`` measuring the distance
#    between the simulated and measured sensor values.
# 4. **Gradient.** Compute ``\mathrm{d}J/\mathrm{d}\boldsymbol{\theta}`` efficiently with
#    the **adjoint method**, verify it against finite differences, and hand it to a
#    gradient-based optimizer.
#
# The star of the show is the gradient. A finite element solution ``\boldsymbol{u}`` is
# defined *implicitly* as the root of a residual ``\boldsymbol{R}(\boldsymbol{u},
# \boldsymbol{\theta}) = \boldsymbol{0}``. We never have an explicit formula
# ``\boldsymbol{u}(\boldsymbol{\theta})`` to differentiate. The adjoint method sidesteps
# this using the *implicit function theorem*: it produces the full gradient with respect
# to *all* parameters at the cost of a **single extra linear solve** — one that reuses the
# tangent matrix we already assembled and factorized for the forward Newton solve.
#
# We do this twice, at two different scales:
#
# - **Part 1 (linear elasticity)** identifies the stiffness of several material regions.
#   Here the implicit function is the global equilibrium ``\boldsymbol{R} = \boldsymbol{0}``.
# - **Part 2 (plasticity)** identifies yield and hardening parameters. Now the element
#   residual *itself* hides a local nonlinear solve — the plastic return mapping that
#   produces the internal state variables. Differentiating through it is a *second*
#   application of the implicit function theorem, which we delegate to
#   [`ImplicitDifferentiation.jl`](https://github.com/gdalle/ImplicitDifferentiation.jl).
#
# ## The adjoint method
#
# Let the discrete equilibrium be ``\boldsymbol{R}(\boldsymbol{u}, \boldsymbol{\theta}) =
# \boldsymbol{0}``, which defines the solution ``\boldsymbol{u}(\boldsymbol{\theta})``
# implicitly, and let the objective be ``J(\boldsymbol{u}(\boldsymbol{\theta}))``. We want
# ``\mathrm{d}J/\mathrm{d}\boldsymbol{\theta}``. Differentiating the constraint,
# ```math
# \frac{\partial \boldsymbol{R}}{\partial \boldsymbol{u}}
# \frac{\mathrm{d}\boldsymbol{u}}{\mathrm{d}\boldsymbol{\theta}}
# + \frac{\partial \boldsymbol{R}}{\partial \boldsymbol{\theta}} = \boldsymbol{0}
# \quad\Longrightarrow\quad
# \frac{\mathrm{d}\boldsymbol{u}}{\mathrm{d}\boldsymbol{\theta}}
# = -\boldsymbol{K}^{-1}\frac{\partial \boldsymbol{R}}{\partial \boldsymbol{\theta}},
# \qquad \boldsymbol{K} := \frac{\partial \boldsymbol{R}}{\partial \boldsymbol{u}},
# ```
# where ``\boldsymbol{K}`` is exactly the tangent stiffness matrix. The chain rule then gives
# ```math
# \frac{\mathrm{d}J}{\mathrm{d}\boldsymbol{\theta}}
# = \frac{\partial J}{\partial \boldsymbol{u}} \frac{\mathrm{d}\boldsymbol{u}}{\mathrm{d}\boldsymbol{\theta}}
# = -\underbrace{\frac{\partial J}{\partial \boldsymbol{u}} \boldsymbol{K}^{-1}}_{\boldsymbol{\lambda}^\mathrm{T}}
#   \frac{\partial \boldsymbol{R}}{\partial \boldsymbol{\theta}} .
# ```
# The trick is to group the terms so that the (large, expensive) inverse is applied *once*,
# independently of the number of parameters. We define the **adjoint** ``\boldsymbol{\lambda}``
# as the solution of the single linear system
# ```math
# \boldsymbol{K}^\mathrm{T} \boldsymbol{\lambda} = \left(\frac{\partial J}{\partial \boldsymbol{u}}\right)^\mathrm{T},
# \qquad\text{then}\qquad
# \frac{\mathrm{d}J}{\mathrm{d}\theta_i} = -\boldsymbol{\lambda}^\mathrm{T}\frac{\partial \boldsymbol{R}}{\partial \theta_i}.
# ```
# Because ``\boldsymbol{K}`` is symmetric for the problems here, ``\boldsymbol{K}^\mathrm{T} =
# \boldsymbol{K}`` and the adjoint solve reuses the forward tangent. Note what we *avoided*:
# we never differentiated through the Newton iterations, and we never formed
# ``\mathrm{d}\boldsymbol{u}/\mathrm{d}\boldsymbol{\theta}`` (a dense ``\text{ndofs}\times
# n_\theta`` object). All we need are two ingredients, and we will obtain both with automatic
# differentiation of a *single* element residual routine:
#
# - the tangent ``\boldsymbol{K} = \partial \boldsymbol{R}/\partial \boldsymbol{u}`` (which the
#   forward solve needs anyway), and
# - the parameter sensitivity ``\partial \boldsymbol{R}/\partial \boldsymbol{\theta}``.
#
# Let's load the packages we need.
using Ferrite, Tensors, SparseArrays, LinearAlgebra
using ForwardDiff, Optim, LineSearches, StaticArrays, ImplicitDifferentiation
using Random
import CairoMakie
## import only these plotting names, so CairoMakie's `Vec` does not clash with Ferrite's
using CairoMakie: Figure, Axis, barplot!, errorbars!, axislegend
CairoMakie.activate!(type = "svg") #hide

# ## Part 1: Identifying elastic stiffness
#
# Consider a 2D plate, clamped on the left edge and pulled by a horizontal traction on the
# right edge (plane strain). The plate is made of three vertical material regions, each with
# an unknown Young's modulus ``E_r`` (Poisson's ratio is assumed known). Our unknowns are
# ``\boldsymbol{\theta} = (\log E_1, \log E_2, \log E_3)`` — we parameterize with the
# logarithm so that the moduli stay positive and the scaling is natural.

const L = 1.0
const H = 0.25
const ν = 0.3

grid = generate_grid(Quadrilateral, (24, 6), Vec(0.0, 0.0), Vec(L, H))

# Assign each cell to one of the three regions based on its centroid.
const nregions = 3
region_of_cell = zeros(Int, getncells(grid))
for cc in CellIterator(grid)
    xc = sum(getcoordinates(cc)) / length(getcoordinates(cc))
    region_of_cell[cellid(cc)] = clamp(ceil(Int, xc[1] / L * nregions), 1, nregions)
end

ip = Lagrange{RefQuadrilateral, 1}()^2
qr = QuadratureRule{RefQuadrilateral}(2)
facet_qr = FacetQuadratureRule{RefQuadrilateral}(2)
cv = CellValues(qr, ip)
fv = FacetValues(facet_qr, ip)

dh = DofHandler(grid)
add!(dh, :u, ip)
close!(dh)

ch = ConstraintHandler(dh)
add!(ch, Dirichlet(:u, getfacetset(grid, "left"), (x, t) -> [0.0, 0.0], [1, 2]))
close!(ch);

# ### Element residual
#
# The whole forward model — and both AD-derived ingredients of the gradient — flow from one
# routine: the element internal force (residual) ``\boldsymbol{r}^e = \int_{\Omega^e}
# \mathrm{grad}^\mathrm{sym}(\delta\boldsymbol{N}) : \boldsymbol{\sigma}\,\mathrm{d}\Omega``,
# with the linear-elastic stress ``\boldsymbol{\sigma} = \mathsf{C}(E) : \boldsymbol{\varepsilon}``.
elastic_C(E) = (G = E / (2(1 + ν)); K = E / (3(1 - 2ν));
    gradient(ϵ -> 2G * dev(ϵ) + 3K * vol(ϵ), zero(SymmetricTensor{2, 2})))

function element_residual!(re, cv, ue, E)
    C = elastic_C(E)
    for qp in 1:getnquadpoints(cv)
        dΩ = getdetJdV(cv, qp)
        ε = function_symmetric_gradient(cv, qp, ue)
        σ = C ⊡ ε
        for i in 1:getnbasefunctions(cv)
            re[i] += (shape_symmetric_gradient(cv, qp, i) ⊡ σ) * dΩ
        end
    end
    return re
end
#md nothing #hide

# The external (Neumann) traction is assembled once — it does not depend on the parameters.
const traction = Vec(100.0, 0.0)
function assemble_fext!(fext, dh, fv, set, t)
    fe = zeros(getnbasefunctions(fv))
    for fc in FacetIterator(dh, set)
        reinit!(fv, fc)
        fill!(fe, 0.0)
        for qp in 1:getnquadpoints(fv)
            dΓ = getdetJdV(fv, qp)
            for i in 1:getnbasefunctions(fv)
                fe[i] += (shape_value(fv, qp, i) ⋅ t) * dΓ
            end
        end
        assemble!(fext, celldofs(fc), fe)
    end
    return fext
end
fext = assemble_fext!(zeros(ndofs(dh)), dh, fv, getfacetset(grid, "right"), traction);

# ### Two ingredients from one residual, via AD
#
# The global assembly loops over cells. For each cell we obtain the element tangent
# ``\partial\boldsymbol{r}^e/\partial\boldsymbol{u}^e`` with `ForwardDiff.jacobian` — the
# element residual is *the* function to differentiate, exactly as in the
# [hyperelasticity tutorial](@ref tutorial-hyperelasticity).
function assemble_system!(K, rint, dh, cv, u, Evals)
    assembler = start_assemble(K, rint)
    nb = getnbasefunctions(cv)
    re = zeros(nb)
    ke = zeros(nb, nb)
    for cell in CellIterator(dh)
        reinit!(cv, cell)
        ue = u[celldofs(cell)]
        E = Evals[region_of_cell[cellid(cell)]]
        f! = (re, ue) -> (fill!(re, 0); element_residual!(re, cv, ue, E))
        ke .= ForwardDiff.jacobian(f!, re, ue) # element tangent; re filled as a side effect
        assemble!(assembler, celldofs(cell), ke, re)
    end
    return K, rint
end

# The parameter sensitivity ``\partial\boldsymbol{R}/\partial\theta_r`` comes from the *same*
# element residual, differentiated with respect to the region modulus instead. Only cells in
# region ``r`` contribute to column ``r``. We differentiate with respect to ``\theta_r = \log E_r``
# directly, so the chain rule to log-space is handled automatically.
function assemble_dRdθ(dh, cv, u, θ)
    dR = zeros(ndofs(dh), length(θ))
    nb = getnbasefunctions(cv)
    for cell in CellIterator(dh)
        reinit!(cv, cell)
        ue = u[celldofs(cell)]
        r = region_of_cell[cellid(cell)]
        g = ForwardDiff.derivative(
            θr -> (re = zeros(eltype(θr), nb); element_residual!(re, cv, ue, exp(θr))), θ[r]
        )
        @views dR[celldofs(cell), r] .+= g
    end
    return dR
end
#md nothing #hide

# ### Forward solve
#
# We solve ``\boldsymbol{R}(\boldsymbol{u}, \boldsymbol{\theta}) = \boldsymbol{0}`` with
# Newton's method. The problem is linear, so it converges in a single iteration, but writing
# it as a Newton loop lets Part 2 reuse the exact same structure. We return the *constrained*
# tangent so the adjoint solve can reuse it. The solve is guarded: for wildly unphysical trial
# parameters (e.g. a region stiffness driven to zero during a line search) the tangent becomes
# singular; we then report failure so the optimizer can back off rather than crash.
K = allocate_matrix(dh)
function solve_forward(θ)
    Evals = exp.(θ)
    u = zeros(ndofs(dh))
    apply!(u, ch)
    rint = zeros(ndofs(dh))
    ok = true
    local r
    for _ in 1:8
        assemble_system!(K, rint, dh, cv, u, Evals)
        r = rint .- fext
        apply_zero!(K, r, ch)
        norm(r) < 1.0e-8 && break
        Δu = try
            K \ r
        catch e
            e isa Union{SingularException, LinearAlgebra.LAPACKException} || rethrow()
            ok = false
            break
        end
        u .-= Δu
    end
    return u, copy(K), ok
end
#md nothing #hide

# ### Sensors
#
# The "experiment" observes the displacement (both components) at a scattered set of sensor
# locations. A [`PointEvalHandler`](@ref) locates each point, and we assemble a sparse
# *observation operator* ``\boldsymbol{B}`` such that ``\boldsymbol{B}\boldsymbol{u}`` gathers the
# simulated sensor readings. Because the finite element interpolation is linear in the nodal
# values, ``\boldsymbol{B}`` is constant and gives us ``\partial J/\partial\boldsymbol{u}`` cheaply.
sensor_pts = vec([Vec(x, y) for x in range(0.12L, 0.96L, 6), y in range(0.06H, 0.94H, 3)])
ph = PointEvalHandler(grid, sensor_pts)
function sensor_operator(ph, dh, ncomp)
    pv = PointValues(ip)
    I = Int[]; J = Int[]; V = Float64[]
    for (s, cid) in enumerate(ph.cells)
        coords = getcoordinates(grid, cid)
        reinit!(pv, Ferrite.PointLocation(cid, ph.local_coords[s], coords))
        cd = celldofs(dh, cid)
        for i in 1:getnbasefunctions(pv)
            Ni = shape_value(pv, 1, i)
            for c in 1:ncomp
                push!(I, (s - 1) * ncomp + c)
                push!(J, cd[i])
                push!(V, Ni[c])
            end
        end
    end
    return sparse(I, J, V, ncomp * length(ph.cells), ndofs(dh))
end
B = sensor_operator(ph, dh, 2);

# ### Synthetic measurements
#
# We pick the "true" moduli, solve once, and contaminate the sensor readings with Gaussian
# noise of standard deviation ``\sigma_d`` (1% of the peak signal here).
const E_true = [70.0e3, 210.0e3, 120.0e3]
θ_true = log.(E_true)
u_true, _, _ = solve_forward(θ_true)

Random.seed!(42)
noise_level = 0.01
d_clean = B * u_true
σ_d = noise_level * maximum(abs, d_clean)
data = d_clean .+ σ_d .* randn(length(d_clean));

# ### Objective and adjoint gradient
#
# The objective is the noise-weighted least-squares misfit
# ```math
# J(\boldsymbol{\theta}) = \frac{1}{2\sigma_d^2}
#   \lVert \boldsymbol{B}\boldsymbol{u}(\boldsymbol{\theta}) - \boldsymbol{d} \rVert^2 ,
# ```
# whose sensitivity to the state is simply ``\partial J/\partial\boldsymbol{u} =
# \boldsymbol{B}^\mathrm{T}(\boldsymbol{B}\boldsymbol{u} - \boldsymbol{d})/\sigma_d^2``. The gradient
# then follows the adjoint recipe verbatim: one adjoint solve, then a dot product per parameter.
function objective(θ)
    u, _, ok = solve_forward(θ)
    ok || return Inf
    r = (B * u .- data) ./ σ_d
    return 0.5 * dot(r, r)
end

function objective_and_grad(θ)
    u, Kc, ok = solve_forward(θ)          # Kc: the constrained tangent, reused below
    ok || return Inf, fill(NaN, length(θ))
    r = (B * u .- data) ./ σ_d
    J = 0.5 * dot(r, r)
    ## adjoint solve  K λ = ∂J/∂u   (homogeneous on constrained dofs)
    g = (B' * r) ./ σ_d
    g[ch.prescribed_dofs] .= 0.0
    λ = Kc \ g
    ## dJ/dθ = -λᵀ ∂R/∂θ
    dR = assemble_dRdθ(dh, cv, u, θ)
    grad = -(dR' * λ)
    return J, grad
end
#md nothing #hide

# ### Verify the gradient
#
# Before trusting the adjoint gradient in an optimizer, always check it against a finite
# difference. Getting agreement to many digits is the single best confidence check for an
# inverse-problem implementation.
θ0 = log.(fill(130.0e3, nregions))
J0, g_adj = objective_and_grad(θ0)
g_fd = similar(g_adj)
h_fd = 1.0e-6
for i in 1:nregions
    θp = copy(θ0); θp[i] += h_fd
    θm = copy(θ0); θm[i] -= h_fd
    g_fd[i] = (objective(θp) - objective(θm)) / (2h_fd)
end
relerr = norm(g_adj - g_fd) / norm(g_fd)
println("adjoint vs finite-difference gradient, relative error = ", relerr)
@assert relerr < 1.0e-6

# ### Optimize
#
# With a verified gradient the calibration is a standard smooth minimization. We use L-BFGS
# with a backtracking line search (robust to the occasional unphysical trial point), and cache
# the forward solve so the value and gradient at the same ``\boldsymbol{\theta}`` are computed together.
cache = Ref((θ = fill(NaN, nregions), J = 0.0, g = zeros(nregions)))
function cached_eval(θ)
    if θ != cache[].θ
        J, g = objective_and_grad(θ)
        cache[] = (θ = copy(θ), J = J, g = g)
    end
    return cache[]
end

res = optimize(
    θ -> cached_eval(θ).J,
    (G, θ) -> (G .= cached_eval(θ).g),
    θ0,
    LBFGS(linesearch = BackTracking(order = 3)),
    Optim.Options(g_tol = 1.0e-6, iterations = 100, store_trace = true, extended_trace = true),
)
E_hat = exp.(Optim.minimizer(res))
println("true moduli:      ", E_true)
println("recovered moduli: ", round.(E_hat; digits = 1))

# ### Uncertainty quantification
#
# The measurements are noisy, so the recovered parameters carry uncertainty. Linearizing the
# forward map about the optimum gives the sensitivity ``\boldsymbol{S} =
# \partial(\boldsymbol{B}\boldsymbol{u})/\partial\boldsymbol{\theta} / \sigma_d = -\boldsymbol{B}
# \boldsymbol{K}^{-1}\,\partial\boldsymbol{R}/\partial\boldsymbol{\theta} / \sigma_d``, and the
# posterior covariance of ``\boldsymbol{\theta}`` is approximately ``(\boldsymbol{S}^\mathrm{T}
# \boldsymbol{S})^{-1}`` — the inverse of the Gauss-Newton Hessian, built from the *same* adjoint
# ingredients. Since ``\theta_r = \log E_r``, the standard deviations are directly the *relative*
# uncertainties of the moduli.
θ_hat = Optim.minimizer(res)
u_hat, K_hat, _ = solve_forward(θ_hat)
S = (B * (-(K_hat \ Matrix(assemble_dRdθ(dh, cv, u_hat, θ_hat))))) ./ σ_d
Cov = inv(Symmetric(S' * S))
rel_std = sqrt.(diag(Cov))
for r in 1:nregions
    println("region $r:  E = $(round(Int, E_hat[r]))  ±$(round(100 * rel_std[r]; digits = 1))%",
        "   (true $(round(Int, E_true[r])))")
end

# Notice that the stiffest region is recovered with the *largest* relative uncertainty. This is
# physical, not a numerical artifact: a stiff region deforms little, so it stores little strain
# energy and leaves only a faint imprint on the measured displacements. Regions the load "probes"
# strongly are identified sharply; regions it barely strains are not. The adjoint machinery both
# finds the estimate *and* tells us how much to trust each component of it.

fig1 = let
    fig = Figure(size = (700, 300))
    ax = Axis(fig[1, 1], xticks = (1:nregions, ["region 1", "region 2", "region 3"]),
        ylabel = "Young's modulus [MPa]", title = "Recovered vs. true stiffness")
    barplot!(ax, (1:nregions) .- 0.18, E_true, width = 0.35, label = "true", color = (:gray, 0.7))
    barplot!(ax, (1:nregions) .+ 0.18, E_hat, width = 0.35, label = "recovered", color = :tomato)
    errorbars!(ax, (1:nregions) .+ 0.18, E_hat, E_hat .* rel_std, whiskerwidth = 12, color = :black)
    axislegend(ax, position = :lt)
    fig
end
fig1

# ## Part 2: Identifying yield and hardening — internal variables and implicit AD
#
# Now the hard part. We calibrate a **plasticity** model, whose stress response depends on
# *internal state variables* (plastic strain and accumulated plastic strain) that are themselves
# the solution of a local nonlinear equation — the *return mapping* — at every quadrature point.
# We identify the initial yield stress ``\sigma_0`` and the hardening saturation ``Q`` of a Voce
# isotropic hardening law,
# ```math
# \sigma_y(\gamma) = \sigma_0 + Q\bigl(1 - e^{-b\gamma}\bigr),
# ```
# where ``\gamma`` is the accumulated plastic strain and ``b`` (a known rate) is fixed.
#
# The adjoint recipe from Part 1 is *unchanged*: solve forward, one adjoint solve, dot products.
# The only new difficulty is obtaining ``\boldsymbol{K}`` and ``\partial\boldsymbol{R}/\partial
# \boldsymbol{\theta}`` when the element residual contains a nested solve. Naively letting AD
# propagate through the local Newton iterations is possible but wasteful and fragile. Instead we
# use the implicit function theorem *again*: the converged state satisfies optimality conditions
# ``\boldsymbol{c}(\boldsymbol{x}, \boldsymbol{y}) = \boldsymbol{0}`` (with ``\boldsymbol{x}`` the
# inputs and ``\boldsymbol{y}`` the state), and its derivative follows from ``\partial
# \boldsymbol{c}/\partial\boldsymbol{y}\cdot\mathrm{d}\boldsymbol{y} = -\partial\boldsymbol{c}/
# \partial\boldsymbol{x}\cdot\mathrm{d}\boldsymbol{x}`` without differentiating the solver.
# `ImplicitDifferentiation.jl` packages exactly this: we supply a `forward` solver and the
# `conditions`, and the resulting object is differentiable by `ForwardDiff`.
#
# !!! note "Differentiating with respect to the material parameters"
#     `ImplicitDifferentiation` differentiates the output with respect to its *first* argument
#     `x`. Because we need sensitivities with respect to ``\sigma_0`` and ``Q`` (not only the
#     strain), we pack the strain *and* the parameters into that single input vector
#     ``\boldsymbol{x} = [\boldsymbol{\varepsilon};\ \sigma_0;\ Q]``. One implicit function then
#     delivers *both* the consistent tangent (derivative w.r.t. strain) and the parameter
#     sensitivity (derivative w.r.t. ``\sigma_0, Q``).
#
# We store symmetric tensors as static vectors in Mandel notation (the ``\sqrt{2}`` factors keep
# the inner product ``\boldsymbol{a}:\boldsymbol{b} = \underline{a}\cdot\underline{b}``), because
# `ImplicitDifferentiation` operates on `AbstractVector`s.

struct VoceFixed{T, S <: SymmetricTensor{4, 3, T}}
    G::T   # shear modulus
    b::T   # hardening rate (fixed)
    Dᵉ::S  # elastic stiffness tensor
end
function VoceFixed(E, ν, b)
    G = E / (2(1 + ν))
    λ = E * ν / ((1 + ν) * (1 - 2ν))
    δ(i, j) = i == j ? 1.0 : 0.0
    Dᵉ = SymmetricTensor{4, 3}(
        (i, j, k, l) -> λ * δ(i, j) * δ(k, l) + G * (δ(i, k) * δ(j, l) + δ(i, l) * δ(j, k))
    )
    return VoceFixed(G, b, Dᵉ)
end

const I6 = SVector{6}(1, 2, 3, 4, 5, 6)
const I6b = SVector{6}(7, 8, 9, 10, 11, 12)
to_svec(A::SymmetricTensor{2, 3, T}) where {T} =
    (s = T(√2); SVector{6, T}(A[1, 1], A[2, 2], A[3, 3], s * A[2, 3], s * A[1, 3], s * A[1, 2]))
function tensor_from_svec(sv::AbstractVector{T}) where {T}
    i = inv(T(√2))
    return SymmetricTensor{2, 3, T}((sv[1], i * sv[6], i * sv[5], sv[2], i * sv[4], sv[3]))
end
## state vector layout: [εᵖ(1:6), σ(7:12), γ(13)]
pack_state(εᵖ, σ, γ) = SVector{13}(to_svec(εᵖ)..., to_svec(σ)..., γ)
unpack_εᵖ(st) = tensor_from_svec(st[I6])
unpack_σ(st) = tensor_from_svec(st[I6b])
unpack_γ(st) = st[13]
#md nothing #hide

# The **forward** return mapping: an elastic predictor followed, if the yield condition is
# violated, by a scalar Newton solve for the plastic multiplier ``\Delta\gamma`` (radial return).
# The input `x` carries the strain and the two identified parameters. The boolean `is_plastic` is
# returned as a *byproduct* — it selects the branch but is treated as constant by the derivative.
function local_forward(x, old_sv, mat)
    ε = tensor_from_svec(x[I6]); σ₀ = x[7]; Q = x[8]
    (; G, b, Dᵉ) = mat
    εᵖ_old = unpack_εᵖ(old_sv); γ_old = unpack_γ(old_sv)
    σ_trial = Dᵉ ⊡ (ε - εᵖ_old)
    s_trial = dev(σ_trial)
    σ_eq = sqrt(3 / 2 * s_trial ⊡ s_trial)
    σ_y = σ₀ + Q * (1 - exp(-b * γ_old))
    if σ_eq - σ_y <= 1.0e-10 * (σ_eq + 1)
        return pack_state(εᵖ_old, σ_trial, γ_old), false          # elastic step
    end
    Δγ = zero(σ_eq)
    tol = 1.0e-11 * (σ_eq + 1)
    res(Δγ) = σ_eq - 3G * Δγ - (σ₀ + Q * (1 - exp(-b * (γ_old + Δγ))))
    for _ in 1:50
        R = res(Δγ)
        Δγ -= R / ForwardDiff.derivative(res, Δγ)
        abs(R) < tol && break
    end
    n = s_trial / sqrt(s_trial ⊡ s_trial + 1.0e-30)
    Δεᵖ = sqrt(3 / 2) * Δγ * n
    return pack_state(εᵖ_old + Δεᵖ, σ_trial - 2G * Δεᵖ, γ_old + Δγ), true # plastic step
end

# The **conditions** the converged state must satisfy (the same equations, written as residuals).
# `ImplicitDifferentiation` differentiates these to obtain the state sensitivity.
function local_conditions(x, state_sv, is_plastic, old_sv, mat)
    ε = tensor_from_svec(x[I6]); σ₀ = x[7]; Q = x[8]
    (; G, b, Dᵉ) = mat
    εᵖ = unpack_εᵖ(state_sv); σ = unpack_σ(state_sv); γ = unpack_γ(state_sv)
    εᵖ_old = unpack_εᵖ(old_sv); γ_old = unpack_γ(old_sv)
    if !is_plastic
        cε = εᵖ - εᵖ_old
        cσ = σ - Dᵉ ⊡ (ε - εᵖ_old)
        cγ = γ - γ_old
    else
        s = dev(σ)
        n = s / sqrt(s ⊡ s + 1.0e-30)
        cε = εᵖ - εᵖ_old - sqrt(3 / 2) * (γ - γ_old) * n
        cσ = σ - Dᵉ ⊡ (ε - εᵖ)
        cγ = sqrt(3 / 2 * s ⊡ s) - (σ₀ + Q * (1 - exp(-b * γ)))
    end
    return SVector{13}(to_svec(cε)..., to_svec(cσ)..., cγ)
end

material_update = ImplicitFunction(local_forward, local_conditions)
const OLD0 = pack_state(zero(SymmetricTensor{2, 3}), zero(SymmetricTensor{2, 3}), 0.0)

# For a single monotonic load applied from a virgin state, the return mapping from zero to the
# final strain reaches the same state as an incremental analysis (radial return is exact for
# proportional loading). We therefore always start from the virgin state `OLD0`, which keeps the
# problem *steady* — no history needs to be back-propagated. The stress used by the element
# routine is then a plain differentiable function of strain and parameters:
function stress(ε::SymmetricTensor{2, 3}, σ₀, Q, mat)
    st, _ = material_update(SVector{8}(to_svec(ε)..., σ₀, Q), OLD0, mat)
    return unpack_σ(st)
end
#md nothing #hide

# ### FE setup: a 3D cantilever
#
# A short 3D cantilever, clamped at one end and bent by a transverse traction at the other. Bending
# creates a range of plastic strains through the depth and along the length, which — as we will see
# — is what makes the hardening parameters identifiable.
const Lb, wb, hb = 10.0, 1.0, 1.0
mat = VoceFixed(200.0e3, 0.3, 10.0)   # E, ν, b   [MPa]

grid2 = generate_grid(Hexahedron, (6, 2, 2), Vec(0.0, 0.0, 0.0), Vec(Lb, wb, hb))
ip2 = Lagrange{RefHexahedron, 1}()^3
qr2 = QuadratureRule{RefHexahedron}(2)
facet_qr2 = FacetQuadratureRule{RefHexahedron}(2)
cv2 = CellValues(qr2, ip2)
fv2 = FacetValues(facet_qr2, ip2)
dh2 = DofHandler(grid2)
add!(dh2, :u, ip2)
close!(dh2)
ch2 = ConstraintHandler(dh2)
add!(ch2, Dirichlet(:u, getfacetset(grid2, "left"), (x, t) -> zeros(3), [1, 2, 3]))
close!(ch2);

# The element residual is structurally identical to Part 1 — only the stress call changed.
function element_residual2!(re, cv, ue, σ₀, Q, mat)
    for qp in 1:getnquadpoints(cv)
        dΩ = getdetJdV(cv, qp)
        ε = function_symmetric_gradient(cv, qp, ue)
        σ = stress(ε, σ₀, Q, mat)
        for i in 1:getnbasefunctions(cv)
            re[i] += (shape_symmetric_gradient(cv, qp, i) ⊡ σ) * dΩ
        end
    end
    return re
end

function assemble_system2!(K, rint, dh, cv, u, σ₀, Q, mat)
    assembler = start_assemble(K, rint)
    nb = getnbasefunctions(cv)
    re = zeros(nb)
    ke = zeros(nb, nb)
    for cell in CellIterator(dh)
        reinit!(cv, cell)
        ue = u[celldofs(cell)]
        f! = (re, ue) -> (fill!(re, 0); element_residual2!(re, cv, ue, σ₀, Q, mat))
        ke .= ForwardDiff.jacobian(f!, re, ue) # consistent tangent through the return map
        assemble!(assembler, celldofs(cell), ke, re)
    end
    return K, rint
end

## dR/d[σ₀, Q]  — sensitivity through the implicit return map
function assemble_dRdθ2(dh, cv, u, σ₀, Q, mat)
    dR = zeros(ndofs(dh), 2)
    nb = getnbasefunctions(cv)
    for cell in CellIterator(dh)
        reinit!(cv, cell)
        ue = u[celldofs(cell)]
        j = ForwardDiff.jacobian(
            p -> (re = zeros(eltype(p), nb); element_residual2!(re, cv, ue, p[1], p[2], mat)), [σ₀, Q]
        )
        @views dR[celldofs(cell), :] .+= j
    end
    return dR
end
#md nothing #hide

# The traction is assembled once per load level. The forward solve is Newton's method again,
# now genuinely nonlinear (typically a handful of iterations).
function fext_for(dh, fv, set, mag)
    return assemble_fext!(zeros(ndofs(dh)), dh, fv, set, Vec(0.0, 0.0, -mag))
end

K2 = allocate_matrix(dh2)
function solve_forward2(σ₀, Q, fe)
    u = zeros(ndofs(dh2))
    apply!(u, ch2)
    rint = zeros(ndofs(dh2))
    ok = true
    local r
    for it in 1:25
        assemble_system2!(K2, rint, dh2, cv2, u, σ₀, Q, mat)
        r = rint .- fe
        apply_zero!(K2, r, ch2)
        norm(r[Ferrite.free_dofs(ch2)]) < 1.0e-6 && break
        Δu = try
            K2 \ r
        catch e
            e isa Union{SingularException, LinearAlgebra.LAPACKException} || rethrow()
            ok = false
            break
        end
        u .-= Δu
        it == 25 && (ok = false)
    end
    return u, copy(K2), ok
end
#md nothing #hide

# ### Why one load level is not enough
#
# Suppose we bend the beam once and try to recover both ``\sigma_0`` and ``Q``. A single deformed
# state samples essentially *one* point of the hardening curve, and the two parameters trade off
# against each other there: a higher yield stress can be compensated by less hardening. We can
# quantify this before optimizing, using the Gauss-Newton Hessian ``\boldsymbol{S}^\mathrm{T}
# \boldsymbol{S}`` — a large condition number means a nearly-flat valley in parameter space.
const σ₀_true, Q_true = 200.0, 300.0

# The observation operator is built exactly as in Part 1, now for the 3D interpolation.
function sensor_operator2(ph, dh, ncomp)
    pv = PointValues(ip2)
    I = Int[]; J = Int[]; V = Float64[]
    for (s, cid) in enumerate(ph.cells)
        coords = getcoordinates(grid2, cid)
        reinit!(pv, Ferrite.PointLocation(cid, ph.local_coords[s], coords))
        cd = celldofs(dh, cid)
        for i in 1:getnbasefunctions(pv)
            Ni = shape_value(pv, 1, i)
            for c in 1:ncomp
                push!(I, (s - 1) * ncomp + c)
                push!(J, cd[i])
                push!(V, Ni[c])
            end
        end
    end
    return sparse(I, J, V, ncomp * length(ph.cells), ndofs(dh))
end
sensor_pts2 = vec([Vec(x, wb / 2, z) for x in range(1.5, 9.5, 6), z in (0.0, hb)])
ph2 = PointEvalHandler(grid2, sensor_pts2)
B2 = sensor_operator2(ph2, dh2, 3)

function condition_number(loads)
    Ss = Matrix{Float64}[]
    for mag in loads
        fe = fext_for(dh2, fv2, getfacetset(grid2, "right"), mag)
        u, Kc, _ = solve_forward2(σ₀_true, Q_true, fe)
        dc = B2 * u
        σd = noise_level * maximum(abs, dc)
        dR = assemble_dRdθ2(dh2, cv2, u, σ₀_true, Q_true, mat)
        push!(Ss, (B2 * (-(Kc \ Matrix(dR))) .* [σ₀_true Q_true]) ./ σd)
    end
    S = reduce(vcat, Ss)
    return cond(S' * S)
end
println("condition number, single load level : ", round(condition_number([15.0]); sigdigits = 3))
println("condition number, three load levels : ", round(condition_number([7.0, 11.0, 15.0]); sigdigits = 3))

# The single-load problem is severely ill-conditioned, while measuring at several load levels
# resolves the trade-off. This mirrors experimental practice: you calibrate a hardening law
# against the *load–deflection curve*, not a single point on it. Since every measurement is a
# separate steady solve from the virgin state, this adds no history-tracking — the adjoint simply
# sums the contribution of each load level.
const loads = [7.0, 11.0, 15.0]
fexts = [fext_for(dh2, fv2, getfacetset(grid2, "right"), mag) for mag in loads];

# ### Synthetic multi-load data
Random.seed!(7)
us_true = [solve_forward2(σ₀_true, Q_true, fe)[1] for fe in fexts]
dcs = [B2 * ut for ut in us_true]
σds = [noise_level * maximum(abs, dc) for dc in dcs]
datas = [dcs[l] .+ σds[l] .* randn(length(dcs[l])) for l in eachindex(loads)]
println("max deflection per load level: ", round.(maximum.(abs, us_true); digits = 3));

# ### Objective and adjoint gradient over load levels
#
# The objective sums the misfit of all load levels. Each contributes its own forward solve and
# its own adjoint solve; the parameter gradients simply add. We optimize the logarithms of the
# parameters to keep them positive.
function objective2_and_grad(θ)
    σ₀, Q = exp(θ[1]), exp(θ[2])
    J = 0.0
    grad = zeros(2)
    for l in eachindex(loads)
        u, Kc, ok = solve_forward2(σ₀, Q, fexts[l])
        ok || return Inf, fill(NaN, 2)
        r = (B2 * u .- datas[l]) ./ σds[l]
        J += 0.5 * dot(r, r)
        g = (B2' * r) ./ σds[l]
        g[ch2.prescribed_dofs] .= 0.0
        λ = Kc \ g
        dR = assemble_dRdθ2(dh2, cv2, u, σ₀, Q, mat)
        grad .+= -(dR' * λ) .* [σ₀, Q]   # chain rule to log-parameters
    end
    return J, grad
end
objective2(θ) = objective2_and_grad(θ)[1]
#md nothing #hide

# Verify the gradient again — the stakes are higher now that it threads two implicit function
# theorems (the global equilibrium and the local return map).
θ2_0 = log.([240.0, 240.0])
J2_0, g2_adj = objective2_and_grad(θ2_0)
g2_fd = similar(g2_adj)
for i in 1:2
    θp = copy(θ2_0); θp[i] += h_fd
    θm = copy(θ2_0); θm[i] -= h_fd
    g2_fd[i] = (objective2(θp) - objective2(θm)) / (2h_fd)
end
relerr2 = norm(g2_adj - g2_fd) / norm(g2_fd)
println("adjoint vs finite-difference gradient (plasticity), relative error = ", relerr2)
@assert relerr2 < 1.0e-6

# ### Optimize and quantify uncertainty
#
# Starting from an initial engineering estimate, L-BFGS recovers the parameters.
cache2 = Ref((θ = [NaN, NaN], J = 0.0, g = [0.0, 0.0]))
function cached_eval2(θ)
    if θ != cache2[].θ
        J, g = objective2_and_grad(θ)
        cache2[] = (θ = copy(θ), J = J, g = g)
    end
    return cache2[]
end

res2 = optimize(
    θ -> cached_eval2(θ).J,
    (G, θ) -> (G .= cached_eval2(θ).g),
    θ2_0,
    LBFGS(linesearch = BackTracking(order = 3)),
    Optim.Options(g_tol = 1.0e-6, iterations = 100),
)
p_hat = exp.(Optim.minimizer(res2))

## posterior uncertainty from the stacked Gauss-Newton Hessian
Ss = Matrix{Float64}[]
for l in eachindex(loads)
    u, Kc, _ = solve_forward2(p_hat[1], p_hat[2], fexts[l])
    dR = assemble_dRdθ2(dh2, cv2, u, p_hat[1], p_hat[2], mat)
    push!(Ss, (B2 * (-(Kc \ Matrix(dR))) .* [p_hat[1] p_hat[2]]) ./ σds[l])
end
Cov2 = inv(Symmetric(reduce(vcat, Ss)' * reduce(vcat, Ss)))
rel_std2 = sqrt.(diag(Cov2))
println("σ₀:  true $(σ₀_true)   recovered $(round(p_hat[1]; digits = 1))  ±$(round(100 * rel_std2[1]; digits = 1))%")
println("Q :  true $(Q_true)   recovered $(round(p_hat[2]; digits = 1))  ±$(round(100 * rel_std2[2]; digits = 1))%")

fig2 = let
    fig = Figure(size = (700, 300))
    ax = Axis(fig[1, 1], xticks = (1:2, ["σ₀", "Q"]), ylabel = "stress [MPa]",
        title = "Recovered vs. true plasticity parameters")
    barplot!(ax, [0.82, 1.82], [σ₀_true, Q_true], width = 0.35, label = "true", color = (:gray, 0.7))
    barplot!(ax, [1.18, 2.18], p_hat, width = 0.35, label = "recovered", color = :tomato)
    errorbars!(ax, [1.18, 2.18], p_hat, p_hat .* rel_std2, whiskerwidth = 12, color = :black)
    axislegend(ax, position = :lt)
    fig
end
fig2

# ## Closing remarks
#
# The same four-line adjoint recipe — forward solve, ``\partial J/\partial\boldsymbol{u}``, one
# adjoint solve reusing the tangent, then ``-\boldsymbol{\lambda}^\mathrm{T}\partial\boldsymbol{R}/
# \partial\boldsymbol{\theta}`` — drove both a linear and a strongly nonlinear calibration. What
# made it scale is that every derivative came from automatic differentiation of a *single* element
# residual routine, and that neither implicit relationship (global equilibrium, local return map)
# was ever differentiated by unrolling its solver. The Gauss-Newton Hessian, assembled from those
# same sensitivities, additionally told us *how well* each parameter is determined by the data.
#
# A few natural extensions:
#
# - **Regularization / priors.** Add a term ``\tfrac{1}{2}\lVert\boldsymbol{\theta} -
#   \boldsymbol{\theta}_\mathrm{prior}\rVert^2_{\boldsymbol{\Gamma}^{-1}}`` to stabilize
#   ill-conditioned identifications (a Bayesian MAP estimate).
# - **Reverse-mode / many parameters.** For spatially varying fields with thousands of
#   parameters, the adjoint's cost is still one linear solve — exactly where it shines over
#   forward-mode AD.
# - **History-dependent loading.** With genuine unloading or non-proportional paths, the state
#   becomes path dependent and the adjoint must be back-propagated through the load steps
#   (a *transient adjoint*).

# The plain program below reproduces the tutorial.                       #src
using Test                                                               #src
@test 0.9 < E_hat[1] / E_true[1] < 1.1                                   #src
@test 0.8 < E_hat[2] / E_true[2] < 1.2                                   #src
@test 0.9 < E_hat[3] / E_true[3] < 1.1                                   #src
@test relerr < 1.0e-6                                                    #src
@test 0.95 < p_hat[1] / σ₀_true < 1.05                                   #src
@test 0.9 < p_hat[2] / Q_true < 1.1                                      #src
@test relerr2 < 1.0e-6                                                   #src

#md # ## [Plain program](@id parameter_identification-plain-program)
#md #
#md # Here follows a version of the program without any comments.
#md # The file is also available here: [`parameter_identification.jl`](parameter_identification.jl).
#md #
#md # ```julia
#md # @__CODE__
#md # ```
