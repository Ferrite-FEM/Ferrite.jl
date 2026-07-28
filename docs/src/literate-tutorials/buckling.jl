# # [Buckling: snap-through of a shallow arch](@id tutorial-buckling)
#
# **Keywords**: *buckling*, *snap-through*, *bifurcation*, *numerical continuation*,
# *finite strain*, *hyperelasticity*, *BifurcationKit.jl*
#
# ![](buckling-light.webp)
# ![](buckling-dark.webp)
#
# *Figure 1*: A clamped shallow arch under a uniform dead load snapping through to its
# inverted configuration. The frames are computed equilibrium states along the branches
# that connect the pre-snap state to the inverted one — parameterized by the
# continuation arclength, not by time. The real, load-controlled jump is a fast dynamic
# event that overshoots these states.
#
#-
#md # !!! tip
#md #     This example is also available as a Jupyter notebook:
#md #     [`buckling.ipynb`](@__NBVIEWER_ROOT_URL__/tutorials/buckling.ipynb).
#-
# ## Introduction
#
# Push down on a shallow arch — the curved strip of a snap hair clip, the lid of a jar,
# a popped bottle cap — and at first it resists, deforming a little. At a critical load
# it gives way *violently*: the arch jumps to the inverted, popped-through shape. This
# *snap-through buckling* is the dramatic cousin of the gentle bending instability of
# the classical Euler column, and this tutorial computes it for a three-dimensional
# hyperelastic arch with
# [BifurcationKit.jl](https://bifurcationkit.github.io/BifurcationKitDocs.jl/stable/),
# reusing the finite strain framework of the
# [hyperelasticity tutorial](@ref tutorial-hyperelasticity).
#
# ### Buckling, bifurcations, and folds
#
# The static equilibria of the arch are the solutions of the balance equations
# ```math
# \underline{R}(\underline{u}, p) = \underline{0},
# ```
# where ``\underline{u}`` is the vector of nodal displacements and ``p`` the load
# magnitude. As ``p`` varies the solutions form curves — *branches* — in the combined
# space of ``(\underline{u}, p)``, and instability phenomena correspond to the special
# points of these curves:
#
# * At a **branch point** (bifurcation) two solution branches cross. The classical
#   example is the Euler column: the branch of straight compressed states is crossed,
#   at the critical load, by a branch of bent states. When the new branch is stable and
#   exists *above* the critical load (a *supercritical* bifurcation, like the Euler
#   column), the transition is gentle — the deflection grows continuously from zero.
#   When it is unstable and bends *backwards* (*subcritical*), there is no nearby
#   stable state above the critical load, and the structure must jump to a distant
#   configuration.
# * At a **fold** (limit point) a branch turns around in ``p``: the equilibria simply
#   run out. A structure loaded beyond a fold has no choice but to snap dynamically to
#   whatever stable state exists far away.
#
# The load–deflection curve of a shallow arch is the classic S-shape sketched by every
# structural mechanics textbook: a stable rising part, a fold at the *limit load*, an
# unstable back-swinging segment, a second fold, and finally the stable branch of
# inverted configurations. And it hides a subtlety that we will recover numerically:
# for sufficiently slender arches the symmetric limit load is never reached, because a
# *subcritical symmetry-breaking bifurcation* occurs slightly before the fold — the
# arch escapes through an antisymmetric *in-plane* mode, one shoulder dipping while the
# other rises, and snaps earlier.
#
# ### Why not just load stepping (or time stepping)?
#
# The other non-linear tutorials increase the load in small steps and solve with
# Newton's method. For this problem that approach fails fundamentally, not just
# practically:
#
# * Past the fold there is **no solution** on the branch being followed. Newton's
#   method has nothing nearby to converge to: it either diverges or lands, after an
#   uncontrolled jump, on a remote branch — and the path in between is lost. This is
#   the historical reason arc-length methods (Riks, Crisfield) were invented in
#   computational structural mechanics.
# * At the fold the tangent stiffness ``\underline{\underline{K}} =
#   \partial\underline{R}/\partial\underline{u}`` is singular, so Newton's method
#   degrades as the fold is approached.
# * Time integration is the right tool for the dynamic jump itself, but the unstable
#   equilibria between the folds — needed to understand the energy landscape of the
#   snap, and simply to draw the textbook S-curve — are repelling for it, and cannot
#   be traced as a sequence of stationary states. Moreover, as long as discretization
#   and loading preserve the symmetry, the antisymmetric mode is only excited by
#   numerical noise or by a deliberately seeded *imperfection* — and with an
#   imperfection the response depends on the imperfection you chose, and the critical
#   load of the perfect structure can no longer be identified cleanly.
#
# *Numerical continuation* treats ``(\underline{u}, p)`` together as the unknown and
# traces the solution curve itself, parameterized by arclength rather than by load, so
# folds are ordinary points of the computation. BifurcationKit provides
# pseudo-arclength continuation (PALC) — the same family of methods as the arc-length
# solvers of structural mechanics — plus the bifurcation toolbox on top:
#
# * it monitors eigenvalues of the tangent stiffness along the branch, and locates
#   stability changes accurately by bisection;
# * it classifies the detected points by computing their *normal form*, differentiating
#   (with ForwardDiff) straight through our Ferrite residual assembly;
# * it can *branch switch*: construct a predictor onto a bifurcated branch and continue
#   along it, without seeding any imperfection.
#
# ### The model problem
#
# We use a clamped shallow circular arch: span ``S = 20``, rise ``h = 2``, thickness
# ``t = 0.5``, width ``w = 2``, loaded by a uniform dead-load traction ``-p\,
# \mathbf{e}_z`` on its outer surface, with both end faces fully clamped. The rise is
# large compared to the bending stiffness of the cross section (``h/t = 4``, i.e.
# roughly 14 radii of gyration), which for these clamped ends puts the arch in the
# regime where the antisymmetric in-plane mode governs the snap — the computed
# spectrum below confirms it. The out-of-plane direction is kept stiff (``w = 4t``) so
# that out-of-plane buckling stays out of the picture.
#
# The material is the compressible neo-Hookean model from the
# [hyperelasticity tutorial](@ref tutorial-hyperelasticity) and we reuse its weak form,
# stress and tangent: find ``\mathbf{u}`` such that
# ```math
# \int_{\Omega} [\nabla_{\mathbf{X}} \delta \mathbf{u}] : \mathbf{P}(\mathbf{u})\ \mathrm{d}\Omega -
# \int_{\Gamma_\mathrm{N}} \delta \mathbf{u} \cdot \mathbf{t}\ \mathrm{d}\Gamma = 0
# \quad \forall\, \delta \mathbf{u},
# ```
# with ``\mathbf{P}`` the first Piola–Kirchhoff stress and ``\mathbf{t} =
# -p\,\mathbf{e}_z`` the traction on the (reference) outer surface. A hyperelastic
# material keeps the problem in exactly the form the continuation machinery expects: a
# residual that depends on ``(\underline{u}, p)`` alone. For a history dependent
# material (e.g. plasticity) the residual depends on the loading path as well —
# arc-length solvers are still used incrementally with such materials, but the branch,
# eigenvalue, and normal-form machinery used below no longer applies as is.
#
# First we load the packages:

using Ferrite, Tensors, BifurcationKit, Arpack, WriteVTK
using SparseArrays, Printf

# ## Material model
#
# The compressible neo-Hookean model, identical to the
# [hyperelasticity tutorial](@ref tutorial-hyperelasticity), with stress and tangent
# obtained by automatic differentiation of the potential:

struct NeoHooke
    μ::Float64
    λ::Float64
end

function Ψ(C, mp::NeoHooke)
    μ = mp.μ
    λ = mp.λ
    Ic = tr(C)
    J = sqrt(det(C))
    return μ / 2 * (Ic - 3 - 2 * log(J)) + λ / 2 * (J - 1)^2
end

function constitutive_driver(C, mp::NeoHooke)
    ∂²Ψ∂C², ∂Ψ∂C = Tensors.hessian(y -> Ψ(y, mp), C, :all)
    S = 2.0 * ∂Ψ∂C
    ∂S∂C = 2.0 * ∂²Ψ∂C²
    return S, ∂S∂C
end;

# ## Problem setup
#
# We collect everything defining the discretized arch in a struct. The displacement
# field is quadratic since the response is bending dominated, where first-order
# displacement interpolation locks. Note the vector of *free* degrees of freedom,
# `fdofs`: since the constraints at the clamped ends are homogeneous Dirichlet
# conditions, we can apply them by simply excluding the constrained dofs from the
# system that BifurcationKit sees, which keeps the eigenvalue computations free of
# constraint bookkeeping. (Inhomogeneous or affine constraints would require proper
# condensation instead.)
#
# !!! note "Why not keep the full-size system, like the other tutorials?"
#     Ferrite's usual workflow keeps the full matrix and enforces Dirichlet conditions
#     with [`apply!`](@ref), which zeroes the constrained rows and columns and puts a
#     scaling factor on their diagonal. That can be made to work with BifurcationKit
#     too, but with two subtleties. The residual entries of constrained dofs must be
#     replaced by the dof values themselves (*not* zeroed with [`apply_zero!`](@ref) —
#     a residual that is identically zero in those entries paired with the eliminated
#     matrix rows would make the Jacobian silently singular). And the elimination
#     introduces artificial eigenvalues at the diagonal scaling factor, which stay out
#     of the way of the shift-invert eigensolver only as long as that factor is large
#     compared to the physical eigenvalues near zero. Since this tutorial revolves
#     around the eigenvalues of the tangent stiffness, we condense instead: every
#     eigenvalue of the reduced matrix is a physical one. The price is the sparse
#     extraction `K[fdofs, fdofs]` on every tangent evaluation — about a millisecond
#     for this problem, negligible next to the factorization it feeds.

struct ArchModel{CV, FV, DH, CH, TF}
    dh::DH
    ch::CH
    cv::CV
    fv::FV
    loadfacets::TF
    mp::NeoHooke
    fdofs::Vector{Int}
    K::SparseMatrixCSC{Float64, Int}
    crown::Vec{3, Float64} # coordinates of the crown point on the outer surface
end

# The arch geometry is obtained by meshing a box and mapping it onto a circular
# segment: the box ``x \in [-S/2, S/2]``, ``z \in [0, t]`` is bent so that its
# midsurface becomes an arc of radius ``R = (S^2/4 + h^2)/(2h)`` with rise ``h``. The
# facet sets of the box carry over: the box's "top" becomes the outer surface where the
# load acts, and "left"/"right" become the clamped end faces.
#
# Two discretization details matter more than usual. The element count along the span
# is kept *even*, so that the mesh is exactly mirror symmetric about ``x = 0`` — the
# perfect pitchfork below exists only because the discrete problem inherits the
# symmetry of the continuous one. And the linear geometry interpolation approximates
# the arc by flat facets, which is fine at this resolution, but worth keeping in mind
# since buckling loads of arches are sensitive to the geometry.

function setup_arch(; nels = (40, 2, 2), S = 20.0, h = 2.0, t = 0.5, w = 2.0, E = 1000.0, ν = 0.3)
    grid = generate_grid(Hexahedron, nels, Vec(-S / 2, 0.0, 0.0), Vec(S / 2, w, t))
    R = (S^2 / 4 + h^2) / (2h)
    α = asin((S / 2) / R)
    function tocurve(x)
        θ = (x[1] / (S / 2)) * α
        r = R + (x[3] - t / 2)
        return Vec(r * sin(θ), x[2], r * cos(θ) - R * cos(α))
    end
    transform_coordinates!(grid, tocurve)

    μ = E / (2(1 + ν))
    λ = (E * ν) / ((1 + ν) * (1 - 2ν))
    mp = NeoHooke(μ, λ)

    ip = Lagrange{RefHexahedron, 2}()^3
    ip_geo = Lagrange{RefHexahedron, 1}()
    ## 3 Gauss points per direction: full integration for the quadratic displacements
    qr = QuadratureRule{RefHexahedron}(3)
    qr_facet = FacetQuadratureRule{RefHexahedron}(3)
    cv = CellValues(qr, ip, ip_geo)
    fv = FacetValues(qr_facet, ip, ip_geo)

    dh = close!(add!(DofHandler(grid), :u, ip))

    ch = ConstraintHandler(dh)
    clamped = union(getfacetset(grid, "left"), getfacetset(grid, "right"))
    add!(ch, Dirichlet(:u, clamped, Returns(zero(Vec{3}))))
    close!(ch)
    fdofs = free_dofs(ch)

    K = allocate_matrix(dh)
    crown = Vec(0.0, w / 2, h + t / 2)
    return ArchModel(dh, ch, cv, fv, getfacetset(grid, "top"), mp, fdofs, K, crown)
end;

# A helper to scatter the free dofs back into a full-length displacement vector:

function to_full(uf::AbstractVector{T}, model::ArchModel) where {T}
    u = zeros(T, ndofs(model.dh))
    u[model.fdofs] .= uf
    return u
end;

# ## Residual and tangent
#
# The element loops are the same as in the
# [hyperelasticity tutorial](@ref tutorial-hyperelasticity), split into two functions:
# BifurcationKit calls the residual and the (sparse) tangent separately, and the
# residual serves one extra purpose — when computing normal forms, BifurcationKit
# evaluates directional derivatives of the residual with ForwardDiff. The residual
# assembly is therefore written *generically* in the number type: the work arrays are
# allocated based on the promotion of the types of `uf` and the load `p`, so that dual
# numbers can flow through the whole assembly (including through the nested automatic
# differentiation in the material model).

function residual(uf::AbstractVector, p, model::ArchModel)
    (; dh, cv, fv, mp, loadfacets, fdofs) = model
    T = promote_type(eltype(uf), typeof(p))
    u = zeros(T, ndofs(dh))
    u[fdofs] .= uf
    g = zeros(T, ndofs(dh))
    nd = ndofs_per_cell(dh)
    ge = zeros(T, nd)
    ue = zeros(T, nd)
    traction = Vec{3}((0.0, 0.0, -1.0)) * p
    for cell in CellIterator(dh)
        reinit!(cv, cell)
        dofs = celldofs(cell)
        ue .= @view u[dofs]
        fill!(ge, zero(T))
        for qp in 1:getnquadpoints(cv)
            dΩ = getdetJdV(cv, qp)
            ∇u = function_gradient(cv, qp, ue)
            F = one(∇u) + ∇u
            C = tdot(F)
            S = 2.0 * Tensors.gradient(y -> Ψ(y, mp), C)
            P = F ⋅ S
            for i in 1:nd
                ∇δui = shape_gradient(cv, qp, i)
                ge[i] += (∇δui ⊡ P) * dΩ
            end
        end
        ## Dead-load traction on the outer surface. Since it does not depend on the
        ## deformation it only contributes to the residual, not to the tangent.
        for facet in 1:nfacets(cell)
            if (cellid(cell), facet) in loadfacets
                reinit!(fv, cell, facet)
                for qp in 1:getnquadpoints(fv)
                    dΓ = getdetJdV(fv, qp)
                    for i in 1:nd
                        δui = shape_value(fv, qp, i)
                        ge[i] -= (δui ⋅ traction) * dΓ
                    end
                end
            end
        end
        g[dofs] .+= ge
    end
    ## Note the sign: we hand BifurcationKit F = -R, see below.
    return -g[fdofs]
end;

# The tangent is standard (and `Float64` only — BifurcationKit uses our analytic sparse
# matrix and never differentiates through it):

function jacobian(uf::AbstractVector, p, model::ArchModel)
    (; dh, cv, mp, K, fdofs) = model
    u = to_full(Vector{Float64}(uf), model)
    nd = ndofs_per_cell(dh)
    ke = zeros(nd, nd)
    ue = zeros(nd)
    assembler = start_assemble(K)
    for cell in CellIterator(dh)
        reinit!(cv, cell)
        ue .= @view u[celldofs(cell)]
        fill!(ke, 0.0)
        for qp in 1:getnquadpoints(cv)
            dΩ = getdetJdV(cv, qp)
            ∇u = function_gradient(cv, qp, ue)
            F = one(∇u) + ∇u
            C = tdot(F)
            S, ∂S∂C = constitutive_driver(C, mp)
            I3 = one(S)
            ∂P∂F = otimesu(I3, S) + 2 * F ⋅ ∂S∂C ⊡ otimesu(F', I3)
            for i in 1:nd
                ∇δui = shape_gradient(cv, qp, i)
                ∇δui∂P∂F = ∇δui ⊡ ∂P∂F
                for j in 1:nd
                    ∇δuj = shape_gradient(cv, qp, j)
                    ke[i, j] += (∇δui∂P∂F ⊡ ∇δuj) * dΩ
                end
            end
        end
        assemble!(assembler, celldofs(cell), ke)
    end
    ## Indexing with the free dofs materializes a new sparse matrix; that is fine here
    ## since the factorization inside the continuation dominates the cost anyway.
    return -K[fdofs, fdofs]
end;

# Why the sign flip ``F = -\underline{R}``? BifurcationKit uses the convention of
# dynamical systems, ``\dot{\underline{u}} = F(\underline{u}, p)``, where an
# equilibrium is *stable* when the eigenvalues of the Jacobian of ``F`` have negative
# real part. Our tangent stiffness ``\underline{\underline{K}}`` is positive definite
# for stable equilibria, so passing ``F = -\underline{R}`` (with Jacobian
# ``-\underline{\underline{K}}``) makes BifurcationKit's stability bookkeeping agree
# with structural mechanics: an equilibrium is reported stable when the monitored
# eigenvalues of ``\underline{\underline{K}}`` — the ones closest to zero, which are
# the relevant ones for detecting stability changes — are all positive. This is local,
# energetic stability of the conservative dead-load problem. (The convention
# corresponds to the overdamped dynamics of the structure, which preserves the
# equilibria and their stability character.)
#
# ## The bifurcation problem
#
# Now we hand the problem to BifurcationKit. Along the branches we record the vertical
# and horizontal displacement of the crown (the topmost point of the arch) using a
# `PointEvalHandler`, which gives us physically meaningful bifurcation diagrams:

model = setup_arch()
uf0 = zeros(length(model.fdofs))
par = (load = 0.0,)

peh = PointEvalHandler(model.dh.grid, [model.crown])
function record(uf, p; kwargs...)
    u = to_full(uf, model)
    crown = evaluate_at_points(peh, model.dh, u, :u)[1]
    return (cz = crown[3], cx = crown[1])
end

prob = BifurcationProblem(
    (uf, p) -> residual(uf, p.load, model),
    uf0, par, (@optic _.load);
    J = (uf, p) -> jacobian(uf, p.load, model),
    record_from_solution = record,
);

# For the continuation we need to configure two solvers. The Newton corrector is
# standard. For the eigenvalue computations — the part that detects instabilities — we
# use Arpack in shift-invert mode around the origin, which efficiently finds the
# eigenvalues of the large sparse Jacobian closest to zero; those are exactly the ones
# that matter for stability changes. (BifurcationKit judges stability from the
# eigenvalues it is handed, so `nev` must be large enough that the relevant near-zero
# eigenvalues are among them.)

optnewton = NewtonPar(
    tol = 1.0e-9, max_iterations = 25,
    eigsolver = EigArpack(0.0, :LM), # shift-invert around σ = 0
)

optcont = ContinuationPar(
    p_min = -0.1, p_max = 0.15, # p_min is a safety margin; the branch turns near p ≈ 0.02
    ds = 0.01, dsmax = 0.05, dsmin = 1.0e-7,
    max_steps = 200,
    newton_options = optnewton,
    detect_bifurcation = 3, # detect eigenvalue crossings and locate them by bisection
    nev = 5,                # number of eigenvalues to track
    n_inversion = 8,
    save_sol_every_step = 1,
);

# ## Continuation of the primary branch
#
# Starting from the unloaded arch we trace the primary (symmetric) branch. Thanks to the
# arclength parameterization the continuation walks straight around both folds and out
# onto the inverted branch, monitoring stability along the way:

br = continuation(prob, PALC(), optcont; normC = norminf)

# BifurcationKit reports four points where an eigenvalue crosses zero, all labeled
# `bp`. Note that a fold also sends an eigenvalue through zero, and with
# `detect_bifurcation = 3` the bisection-based detection supersedes BifurcationKit's
# dedicated fold detection — so the two folds appear in this list as well. Mapping the
# four crossings onto the recorded crown deflection tells the story:
#
# 1. ``p \approx 0.0709``: a *symmetry-breaking branch point*, while the symmetric
#    branch is still rising and stable — the bifurcation announced in the introduction;
# 2. ``p \approx 0.0768``: the *upper fold*, i.e. the limit load of the symmetric arch,
#    where the branch turns back;
# 3. ``p \approx 0.023``: a second symmetry-breaking point, where the pair of
#    asymmetric equilibria reconnects to the symmetric branch;
# 4. ``p \approx 0.021``: the *lower fold*, after which the branch rises steeply along
#    the stable inverted configurations.
#
# Only the first and third crossings are genuine branch points — and the first one
# occurs *before* the limit load: for this perfect, quasi-statically loaded model,
# stability is first lost at the pitchfork, not at the fold. Because the bifurcation
# turns out to be subcritical (next section), an imperfect real-world arch snaps at or
# below this load — the classical imperfection sensitivity of shallow arches.

p_bp = br.specialpoint[1].param
@printf "symmetry-breaking bifurcation at p = %.4f (crown deflection %.3f)\n" p_bp br.specialpoint[1].printsol.cz

using Test #src
@test length(br.specialpoint) == 5 #src
@test br.specialpoint[1].type === :bp #src
@test p_bp ≈ 0.0709 atol = 1.0e-3 #src
@test br.branch[end].cz < -3.5 # arch is inverted at the end of the branch #src

# ## The normal form: what kind of bifurcation is this?
#
# Next we ask BifurcationKit to analyze the first branch point. This is where the
# generic residual pays off: the normal form coefficients are combinations of second
# and third directional derivatives of the residual, which BifurcationKit computes with
# ForwardDiff *through the Ferrite assembly*:

bp = get_normal_form(br, 1)

# A *subcritical pitchfork*: the reduced equation near the branch point is
# ``b_{11}\, x\, \delta p + b_{30}\, x^3/6 = 0``, and ``b_{11} b_{30} > 0`` (here both
# coefficients are positive) means that the pair of asymmetric equilibria exists for
# ``\delta p < 0`` — *below* the critical load — and is unstable. The odd symmetry
# (``a_{01} \approx b_{20} \approx 0``) reflects that the two antisymmetric sway
# directions are equivalent. Contrast this with the Euler column, whose pitchfork is
# *supercritical* (``b_{11} b_{30} < 0`` in the same convention): there, a stable bent
# state exists arbitrarily close above the critical load, which is why a column
# buckles gently while the arch snaps violently — above ``p_\mathrm{bp}`` the arch has
# no nearby *stable* equilibrium, so it must jump dynamically to the distant inverted
# branch.

@test bp.type === :SubCritical #src
@test bp.nf.b11 * bp.nf.b30 > 0 # BifurcationKit's subcriticality criterion #src

# ## Branch switching: the asymmetric escape route
#
# With the normal form available, BifurcationKit can construct a predictor on the
# asymmetric branch and continue along it — no imperfection needed:

## monitor eigenvalues along the new branch, but skip the bisection localization here
optcont2 = setproperties(optcont; ds = 0.005, dsmax = 0.05, max_steps = 100, p_min = 0.0, detect_bifurcation = 2)
br2 = continuation(br, 1, optcont2)

# The asymmetric branch (watch the recorded `cx`!) leaves the primary branch at
# ``p_\mathrm{bp}`` and sweeps downwards through the antisymmetric in-plane states —
# the unstable equilibria that separate the pre-snap well from the inverted one. It
# returns to symmetry at the second symmetry-breaking point near the lower fold.
# Strictly speaking, the asymmetric branch continues onto its own mirror image there
# (the load has a quadratic turning point along it); that the continuation instead
# slides onto the symmetric inverted branch is a branch *jump* at a nearly degenerate
# crossing — convenient here, but worth recognizing for what it is. The recorded crown
# displacement makes a nice bifurcation diagram:

using Plots
## BifurcationKit's multi-branch recipe, plot(br, br2), only supports :param on the
## x-axis, so we overlay two single-branch plots to put the recorded variables there.
plt1 = plot(br; vars = (:cz, :param), branchlabel = "symmetric", xlabel = "crown deflection u_z", ylabel = "load p", legend = :topleft)
plot!(plt1, br2; vars = (:cz, :param), branchlabel = "asymmetric")
plt2 = plot(br; vars = (:cx, :param), branchlabel = "symmetric", xlabel = "crown horizontal displacement u_x", ylabel = "load p", legend = :topright)
plot!(plt2, br2; vars = (:cx, :param), branchlabel = "asymmetric")
plot(plt1, plt2; layout = (1, 2), size = (800, 350), margin = 5Plots.mm)

# In the left diagram the S-curve of the symmetric branch is drawn with its stability
# (solid = stable, dashed = unstable), with the asymmetric branch short-circuiting the
# upper fold. The right diagram shows the pitchfork loop in the horizontal crown
# displacement, which is identically zero on the symmetric branch. The mirror-image
# asymmetric states exist too, of course; `continuation` picks one representative, and
# `usedeflation = true` or flipping the predictor direction finds the other.
#
# ## Exporting the snap
#
# All equilibrium states along both branches were stored during the continuation
# (`save_sol_every_step = 1`). We export the sequence of equilibria that connects the
# pre-snap state to the inverted one — the stable ascent up to the bifurcation point,
# then along the asymmetric branch — as a ParaView collection, exactly like the
# time-stepping tutorials. Keep in mind that the "time" here is the continuation step,
# *not* physical time: the real snap is a dynamic event that jumps across these states
# rather than passing through them. This produces the animation in Figure 1.

states = vcat(
    [s for s in br.sol if s.step <= br.specialpoint[1].step], # stable ascent
    br2.sol,                                                  # asymmetric branch + inverted branch
)
pvd = paraview_collection("buckling")
for (istep, s) in enumerate(states)
    u = to_full(s.x, model)
    VTKGridFile("buckling_$istep", model.dh) do vtk
        write_solution(vtk, model.dh, u)
        pvd[istep] = vtk
    end
end
vtk_save(pvd);

# ## Where to go from here
#
# * The classical Euler buckling of a column is the *gentle* (supercritical) sibling of
#   this problem and can be computed with exactly the machinery above — swap the arch
#   for a slender cantilever column (clamped bottom, free top) under compressive
#   traction, and the detected pitchfork becomes supercritical, with the critical load
#   closely matching the fixed–free Euler formula ``P_\mathrm{cr} = \pi^2 EI/(4L^2)``.
# * `BifurcationKit.continuation` can also continue the *special points themselves* in
#   a second parameter (fold continuation, branch-point continuation), e.g. to compute
#   directly how the snap load depends on the rise ``h`` or the thickness ``t``.
# * Deflated continuation (`usedeflation = true`) can help find disconnected or
#   mirrored branches.
#
#md # ## [Plain program](@id buckling-plain-program)
#md #
#md # Here follows a version of the program without any comments.
#md # The file is also available here: [`buckling.jl`](buckling.jl).
#md #
#md # ```julia
#md # @__CODE__
#md # ```
