using Ferrite, FerriteMeshParser, FerriteInterfaceElements, VTKHDF
using Tensors, ForwardDiff, LinearAlgebra, SparseArrays, Random, Printf, Downloads
import LinearSolve
using Ferrite: OrderedSet
import Plots

const FULL = get(ENV, "FERRITE_GCP_FULL", "false") == "true"

function load_sve(inpfile)
    grid = get_ferrite_grid(inpfile; generate_facetsets = false)
    dim = Ferrite.getspatialdim(grid)
    grains = sort!(filter(startswith("poly"), collect(keys(Ferrite.getcellsets(grid)))); by = n -> parse(Int, chopprefix(n, "poly")))
    L = length(grains)^(1 / dim)
    transform_coordinates!(grid, x -> L * x)
    tol = 1.0e-8 * L
    faces = ("x0", "x1", "y0", "y1", "z0", "z1")
    for name in faces
        addfacetset!(grid, name, FerriteMeshParser.create_facetset(grid, getnodeset(grid, name)))
    end
    addfacetset!(grid, "boundary", union((getfacetset(grid, n) for n in faces)...))
    grid = insert_interfaces(grid, grains)

    # Corner sets used below to remove rigid-body motion.
    for (name, corner) in (("origin", zero(Vec{dim})), ("corner_x", L * basevec(Vec{dim}, 1)), ("corner_y", L * basevec(Vec{dim}, 2)))
        addnodeset!(grid, name, x -> norm(x - corner) < tol)
    end
    addnodeset!(grid, "grain_boundary", OrderedSet{Int}(n for c in getcells(grid) if c isa InterfaceCell for n in c.nodes))
    addcellset!(grid, "bulk", OrderedSet{Int}(i for (i, c) in enumerate(getcells(grid)) if c isa Tetrahedron))
    grain_of_cell = zeros(Int, getncells(grid))
    for (g, name) in enumerate(grains), c in getcellset(grid, name)
        grain_of_cell[c] = g
    end
    return grid, grain_of_cell
end

struct CrystalMaterial{dim, T, N4, N2}
    E::SymmetricTensor{4, dim, T, N4}
    s::Vector{Vec{dim, T}}                     # slip directions
    k::Vector{Vec{dim, T}}                     # m × s
    P::Vector{SymmetricTensor{2, dim, T, N2}}   # (s ⊗ m)ˢʸᵐ
    Hl2::T                                   # gradient hardening H l²
    C::T                                     # reference stress
    tstar::T                                 # reference time
    m::T                                     # Norton exponent
end
nslips(mat::CrystalMaterial) = length(mat.s)

function fcc_slip_systems()
    planes = [Vec((1.0, 1.0, -1.0)), Vec((1.0, -1.0, -1.0)), Vec((1.0, -1.0, 1.0)), Vec((1.0, 1.0, 1.0))] ./ √3
    dirs = [
        (0.0, 1.0, 1.0), (1.0, 0.0, 1.0), (1.0, -1.0, 0.0), (0.0, 1.0, -1.0), (1.0, 0.0, 1.0), (1.0, 1.0, 0.0),
        (0.0, 1.0, 1.0), (1.0, 0.0, -1.0), (1.0, 1.0, 0.0), (0.0, 1.0, -1.0), (1.0, 0.0, -1.0), (1.0, -1.0, 0.0),
    ]
    s = [Vec(d) / √2 for d in dirs]
    m = repeat(planes; inner = 3) # three slip directions per plane
    return s, m
end

function CrystalMaterial(; E, ν, H, l, C, tstar, m, rotation)
    dim = size(rotation, 1)
    λ = E * ν / ((1 + ν) * (1 - 2ν))
    μ = E / (2(1 + ν))
    δ(i, j) = i == j ? 1.0 : 0.0
    Eᵉ = SymmetricTensor{4, dim}((i, j, k, l) -> λ * δ(i, j) * δ(k, l) + μ * (δ(i, k) * δ(j, l) + δ(i, l) * δ(j, k)))
    s0, m0 = fcc_slip_systems()
    s = [rotation ⋅ sα for sα in s0]
    n = [rotation ⋅ mα for mα in m0]
    k = [nα × sα for (sα, nα) in zip(s, n)]
    P = [symmetric(sα ⊗ nα) for (sα, nα) in zip(s, n)]
    return CrystalMaterial(Eᵉ, s, k, P, H * l^2, C, tstar, m)
end

function random_rotation(rng)
    z = 2rand(rng) - 1
    ϕ = 2π * rand(rng)
    u = Vec((sqrt(1 - z^2) * cos(ϕ), sqrt(1 - z^2) * sin(ϕ), z)) # uniform on the sphere
    θ = 2π * rand(rng)
    # The skew tensor represents the cross product: W ⋅ v = u × v.
    W = Tensor{2, length(u)}(
        [
            0.0   -u[3]  u[2]
            u[3]   0.0  -u[1]
            -u[2]  u[1]  0.0
        ]
    )
    return one(W) + sin(θ) * W + (1 - cos(θ)) * (W ⋅ W) # Rodrigues' formula
end

function paper_materials(ngrains)
    rng = Xoshiro(1234)
    return [
        CrystalMaterial(; E = 200.0e3, ν = 0.3, H = 2.0e4, l = 0.1, C = 1.0e3, tstar = 1.0e3, m = 2.0, rotation = random_rotation(rng))
            for _ in 1:ngrains
    ]
end

mutable struct QPState{S}
    γ::Vector{Float64}
    τdi::Vector{Float64}
    σ::S
    ε::S
    φ::Float64
    π::Float64
    Y::Vector{Float64}   # cache: outer variables (ε, χ⊥, χ⊙) of the local problem,
    X::Vector{Float64}   #        its solution (γ, τᵈⁱ)
    A::Matrix{Float64}   #        and the derivative dX/dY
    Avalid::Bool
end

struct LocalLayout
    ε::UnitRange{Int}    # Y: strain in Voigt notation
    χp::UnitRange{Int}   # Y: χ⊥ₐ
    χo::UnitRange{Int}   # Y: χ⊙ₐ
    γ::UnitRange{Int}    # X: slips
    τdi::UnitRange{Int}  # X: dissipative stresses
end
function LocalLayout(mat::CrystalMaterial{dim}) where {dim}
    M = nslips(mat)
    nε = Tensors.n_components(SymmetricTensor{2, dim})
    return LocalLayout(1:nε, nε .+ (1:M), (nε + M) .+ (1:M), 1:M, M .+ (1:M))
end
nX(ll::LocalLayout) = last(ll.τdi)
nY(ll::LocalLayout) = last(ll.χo)

function QPState(mat)
    M = nslips(mat)
    ll = LocalLayout(mat)
    z = zero(eltype(mat.P))
    return QPState(
        zeros(M), zeros(M), z, z, 0.0, 0.0,
        fill(NaN, nY(ll)), zeros(nX(ll)), zeros(nX(ll), nY(ll)), false
    )
end
function update_state!(new::QPState, old::QPState)
    new.γ .= old.γ
    new.τdi .= old.τdi
    new.σ = old.σ
    new.ε = old.ε
    new.φ = old.φ
    new.π = old.π
    return new
end

function local_residual!(R, X, Y, γn, mat::CrystalMaterial{dim}, Δt) where {dim}
    ll = LocalLayout(mat)
    ε = fromvoigt(SymmetricTensor{2, dim}, view(Y, ll.ε))
    γ = view(X, ll.γ)
    τdi = view(X, ll.τdi)
    εp = zero(promote_type(eltype(X), eltype(Y))) * mat.P[1]
    for α in 1:nslips(mat)
        εp += γ[α] * mat.P[α]
    end
    σ = mat.E ⊡ (ε - εp)
    for α in 1:nslips(mat)
        τ = σ ⊡ mat.P[α]
        χ = Y[ll.χp[α]] + Y[ll.χo[α]]
        R[ll.γ[α]] = (-τ + τdi[α] - χ) / mat.C
        R[ll.τdi[α]] = (γ[α] - γn[α]) - Δt / mat.tstar * (abs(τdi[α]) / mat.C)^mat.m * sign(τdi[α])
    end
    return R
end

function local_newton!(X, Y, γn, mat, Δt; tol = 1.0e-12, maxiter = 50)
    R = similar(X)
    J = Matrix{eltype(X)}(undef, length(X), length(X))
    f!(R, X) = local_residual!(R, X, Y, γn, mat, Δt)
    cfg = ForwardDiff.JacobianConfig(f!, R, X)
    for _ in 1:maxiter
        f!(R, X)
        norm(R) < tol && return X
        ForwardDiff.jacobian!(J, f!, R, X, cfg)
        ldiv!(lu!(J), R)
        X .-= R
    end
    return error("local problem did not converge")
end

value(x::Real) = ForwardDiff.value(x)
value(x::SymmetricTensor{2, dim}) where {dim} = SymmetricTensor{2, dim}(map(ForwardDiff.value, Tensors.get_data(x)))

function local_solve(Y::AbstractVector{T}, st::QPState, γn, mat, Δt) where {T}
    ll = LocalLayout(mat)
    Yv = ForwardDiff.value.(Y)
    if Yv != st.Y
        st.X[ll.γ] .= st.γ # initial guess
        st.X[ll.τdi] .= st.τdi
        local_newton!(st.X, Yv, γn, mat, Δt)
        copyto!(st.Y, Yv)
        st.Avalid = false
    end
    T === Float64 && return copy(st.X)
    if !st.Avalid
        JX = ForwardDiff.jacobian((R, X) -> local_residual!(R, X, Yv, γn, mat, Δt), zeros(nX(ll)), st.X)
        JY = ForwardDiff.jacobian((R, Y) -> local_residual!(R, st.X, Y, γn, mat, Δt), zeros(nX(ll)), Yv)
        st.A .= -(JX \ JY)
        st.Avalid = true
    end
    dY = Y .- Yv
    X = st.A * dY
    X .+= st.X
    return X
end

function dissipative_stress(Δγ, mat::CrystalMaterial, Δt, δ)
    A = mat.C * (mat.tstar / Δt)^(1 / mat.m)
    return A * Δγ * (Δγ^2 + δ^2)^((1 / mat.m - 1) / 2)
end
function dissipation(Δγ, mat::CrystalMaterial, Δt, δ)
    A = mat.C * (mat.tstar / Δt)^(1 / mat.m)
    p = (mat.m + 1) / mat.m
    return A * mat.m / (mat.m + 1) * ((Δγ^2 + δ^2)^(p / 2) - δ^p)
end

struct Layout
    u::UnitRange{Int}
    ξp::Vector{UnitRange{Int}}
    ξo::Vector{UnitRange{Int}}
    σ̄::UnitRange{Int}
end

struct Problem{CV, SV, F}
    format::Symbol          # :dual or :primal
    cv::CV                 # MultiFieldCellValues for displacement and scalar fields
    layout::Layout
    σ̄vals::SV               # AlgebraicValues of σ̄ (Neumann) or nothing (Dirichlet)
    Hbar::F                 # t -> macroscopic displacement gradient
    Δt::Float64
    δreg::Float64           # regularization of |Δγ| (primal format)
end

function element_residual!(re, ae, ae_n, states, states_n, mat::CrystalMaterial{dim}, prob::Problem, t) where {dim}
    (; cv, layout, Δt, δreg) = prob
    M = nslips(mat)
    ll = LocalLayout(mat)
    T = eltype(ae)
    fill!(re, zero(T))
    u = @view ae[layout.u]
    ru = @view re[layout.u]
    nu = getnbasefunctions(cv.u)
    ns = getnbasefunctions(cv.s)
    Y = Vector{T}(undef, nY(ll))
    ξp = Vector{T}(undef, M)
    ξo = Vector{T}(undef, M)
    γ = Vector{T}(undef, M)
    ∇γ = Vector{Vec{dim, T}}(undef, M)
    if prob.σ̄vals !== nothing
        σ̄ = algebraic_value(prob.σ̄vals, ae, layout.σ̄)
        ε̄ = symmetric(prob.Hbar(t))
    end
    for qp in 1:getnquadpoints(cv)
        dΩ = getdetJdV(cv, qp)
        ∇u = function_gradient(cv.u, qp, u)
        ε = symmetric(∇u)
        st, stn = states[qp], states_n[qp]
        εp = zero(T) * mat.P[1]
        if prob.format === :dual
            tovoigt!(view(Y, ll.ε), ε)
            for α in 1:M
                ap = view(ae, layout.ξp[α])
                ao = view(ae, layout.ξo[α])
                ξp[α] = function_value(cv.s, qp, ap)
                ξo[α] = function_value(cv.s, qp, ao)
                Y[ll.χp[α]] = function_directional_derivative(cv.s, qp, ap, mat.s[α])
                Y[ll.χo[α]] = function_directional_derivative(cv.s, qp, ao, mat.k[α])
            end
            X = local_solve(Y, st, stn.γ, mat, Δt)
            γ = view(X, ll.γ)
            τdi = view(X, ll.τdi)
            for α in 1:M
                εp += γ[α] * mat.P[α]
            end
            σ = mat.E ⊡ (ε - εp)
            for α in 1:M
                rp = view(re, layout.ξp[α])
                ro = view(re, layout.ξo[α])
                sα, kα = mat.s[α], mat.k[α]
                ξpα, ξoα, γα = ξp[α], ξo[α], γ[α]
                for i in 1:ns
                    δξ = shape_value(cv.s, qp, i)
                    ∇δξ = shape_gradient(cv.s, qp, i)
                    rp[i] -= (ξpα / mat.Hl2 * δξ + (∇δξ ⋅ sα) * γα) * dΩ
                    ro[i] -= (ξoα / mat.Hl2 * δξ + (∇δξ ⋅ kα) * γα) * dΩ
                end
            end
            # Semi-dual free energy φ = ψᵉ - ψᵍ* and incremental potential
            # These are only used for postprocessing, so discard derivatives first.
            εe = value(ε - εp)
            φ = 1 / 2 * εe ⊡ mat.E ⊡ εe - sum(value(ξp[α])^2 + value(ξo[α])^2 for α in 1:M) / (2mat.Hl2)
            st.γ .= value.(γ)
            st.τdi .= value.(τdi)
            π = φ - stn.φ
            for α in 1:M
                χ = value(Y[ll.χp[α]]) + value(Y[ll.χo[α]])
                γα, τdiα = st.γ[α], st.τdi[α]
                π += τdiα * (γα - stn.γ[α]) - χ * γα - Δt / mat.tstar * mat.C / (mat.m + 1) * (abs(τdiα) / mat.C)^(mat.m + 1)
            end
        else # primal
            for α in 1:M
                aγ = view(ae, layout.ξp[α])
                γ[α] = function_value(cv.s, qp, aγ)
                ∇γ[α] = function_gradient(cv.s, qp, aγ)
                εp += γ[α] * mat.P[α]
            end
            σ = mat.E ⊡ (ε - εp)
            εe = value(ε - εp)
            ψe = 1 / 2 * εe ⊡ mat.E ⊡ εe
            ψg = 0.0 # gradient energy, summed over the slip systems
            Δtϕ = 0.0 # dissipation
            for α in 1:M
                γdofs = layout.ξp[α]
                rγ = view(re, γdofs)
                τ = σ ⊡ mat.P[α]
                Δγ = γ[α] - function_value(cv.s, qp, view(ae_n, γdofs))
                τdi = dissipative_stress(Δγ, mat, Δt, δreg)
                gp = ∇γ[α] ⋅ mat.s[α]
                go = ∇γ[α] ⋅ mat.k[α]
                ξ = mat.Hl2 * (gp * mat.s[α] + go * mat.k[α])
                for i in 1:ns
                    δγ = shape_value(cv.s, qp, i)
                    ∇δγ = shape_gradient(cv.s, qp, i)
                    rγ[i] += (δγ * (τdi - τ) + ∇δγ ⋅ ξ) * dΩ
                end
                ψg += mat.Hl2 / 2 * (value(gp)^2 + value(go)^2)
                Δtϕ += dissipation(value(Δγ), mat, Δt, δreg)
                st.γ[α] = value(γ[α])
                st.τdi[α] = value(τdi)
            end
            # Same reference as in the semi-dual format: φ = ψᵉ - ψᵍ* = ψᵉ - ψᵍ at the solution,
            # and the term -χ ⋆ γ of the incremental potential equals ξ ⋆ ∇γ = 2ψᵍ after
            # integration by parts
            φ = ψe - ψg
            π = ψe + ψg - stn.φ + Δtϕ
        end
        σu = prob.σ̄vals === nothing ? σ : σ - σ̄
        for i in 1:nu
            δε = shape_symmetric_gradient(cv.u, qp, i)
            ru[i] += (δε ⊡ σu) * dΩ
        end
        if prob.σ̄vals !== nothing # Neumann: average-strain constraint
            for (k, I) in pairs(layout.σ̄)
                δσ̄ = algebraic_basis_value(prob.σ̄vals, k)
                re[I] -= (δσ̄ ⊡ (ε - ε̄)) * dΩ
            end
        end
        st.σ = value(σ)
        st.ε = value(ε)
        st.φ = φ
        st.π = π
    end
    return re
end

function setup(grid, materials, format, ubc, ξbc, Hbar, Δt; δreg = 1.0e-6)
    M = nslips(materials[1])
    dim = Ferrite.getspatialdim(grid)
    ip = Lagrange{RefTetrahedron, 1}()
    dh = DofHandler(grid)
    sdh = SubDofHandler(dh, getcellset(grid, "bulk"))
    add!(sdh, :u, ip^dim)
    if format === :dual
        names_p = [Symbol("ξ⊥", α) for α in 1:M]
        names_o = [Symbol("ξ⊙", α) for α in 1:M]
    else
        names_p = [Symbol("γ", α) for α in 1:M]
        names_o = Symbol[]
    end
    foreach(n -> add!(sdh, n, ip), names_p)
    foreach(n -> add!(sdh, n, ip), names_o)
    sdh_interface = SubDofHandler(dh, getcellset(grid, "interfaces"))
    add!(sdh_interface, :u, InterfaceCellInterpolation(Lagrange{RefTriangle, 1}())^dim)
    σ̄var = AlgebraicVariable{SymmetricTensor{2, dim}}()
    σ̄vals = ubc === :neumann ? AlgebraicValues(σ̄var) : nothing
    ubc === :neumann && add!(dh, :σ̄, σ̄var)
    close!(dh)
    n = ndofs_per_cell(sdh)
    nσ̄ = ubc === :neumann ? getnbasefunctions(σ̄vals) : 0
    layout = Layout(dof_range(sdh, :u), [dof_range(sdh, n) for n in names_p], [dof_range(sdh, n) for n in names_o], (n + 1):(n + nσ̄))

    ch = ConstraintHandler(dh)
    for cell in CellIterator(sdh_interface)
        c = getcells(grid, cellid(cell))
        udofs = reshape(celldofs(cell), dim, :) # one column per vertex: the `here` vertices, then the `there` vertices
        nv = length(c.here.nodes)
        for (v, (h, t)) in enumerate(zip(c.here.nodes, c.there.nodes))
            slaves, masters = h > t ? (udofs[:, v], udofs[:, nv + v]) : (udofs[:, nv + v], udofs[:, v])
            for (slave, master) in zip(slaves, masters)
                add!(ch, AffineConstraint(slave, [master => 1.0], 0.0))
            end
        end
    end
    if ubc === :dirichlet
        add!(ch, Dirichlet(:u, getfacetset(grid, "boundary"), (x, t) -> Hbar(t) ⋅ x))
    else # remove the rigid body motions: fix the origin, u₂ and u₃ on the x-axis, u₃ on the y-axis
        add!(ch, Dirichlet(:u, getnodeset(grid, "origin"), Returns(zero(Vec{dim}))))
        add!(ch, Dirichlet(:u, getnodeset(grid, "corner_x"), Returns(zeros(dim - 1)), collect(2:dim)))
        add!(ch, Dirichlet(:u, getnodeset(grid, "corner_y"), Returns(zeros(dim - 2)), collect(3:dim)))
    end
    if format === :dual && ξbc === :microfree
        for name in vcat(names_p, names_o)
            add!(ch, Dirichlet(name, getfacetset(grid, "boundary"), Returns(0.0)))
        end
    elseif format === :primal
        for name in names_p
            isempty(getnodeset(grid, "grain_boundary")) || add!(ch, Dirichlet(name, getnodeset(grid, "grain_boundary"), Returns(0.0)))
            ξbc === :microhard && add!(ch, Dirichlet(name, getfacetset(grid, "boundary"), Returns(0.0)))
        end
    end
    close!(ch)

    # Four points integrate the quadratic ξ δξ term exactly; the archived paper code
    # used one point, which underintegrates this term.
    # Possible optimization for the dual format on affine linear tetrahedra: ε and
    # the directional gradients of ξ are constant, so matching previous material
    # states permit reusing the local material solution and tangent across points
    # while still integrating ξ δξ with four points. Keep the per-point evaluation
    # here to support quadratic elements and differing quadrature-point histories.
    qr = QuadratureRule{RefTetrahedron}(2)
    cv = MultiFieldCellValues(qr, (u = ip^dim, s = ip))
    prob = Problem(format, cv, layout, σ̄vals, Hbar, Δt, δreg)
    if ubc === :neumann
        coupling = CellCoupling(getcellset(grid, "bulk"); algebraic_coupling = ((:u, :σ̄),))
        K = allocate_matrix(dh, ch; algebraic_couplings = coupling)
    else
        K = allocate_matrix(dh, ch)
    end
    states = [[QPState(materials[1]) for _ in 1:getnquadpoints(cv)] for _ in 1:getncells(grid)]
    states_n = deepcopy(states)
    return dh, sdh, ch, K, prob, states, states_n
end

function assemble_system!(K, r, a, a_n, dh, sdh, prob, states, states_n, materials, grain_of_cell, t)
    assembler = start_assemble(K, r)
    n = ndofs_per_cell(sdh)
    nσ̄ = length(prob.layout.σ̄)
    dofs = Vector{Int}(undef, n + nσ̄)
    nσ̄ > 0 && (dofs[prob.layout.σ̄] .= algebraic_dofs(dh, :σ̄))
    re = zeros(n + nσ̄)
    Ke = zeros(n + nσ̄, n + nσ̄)
    ae = zeros(n + nσ̄)
    ae_n = zeros(n + nσ̄)
    cellid = Ref(0)
    f! = (re, ae) -> element_residual!(re, ae, ae_n, states[cellid[]], states_n[cellid[]], materials[grain_of_cell[cellid[]]], prob, t)
    cfg = ForwardDiff.JacobianConfig(f!, re, ae)
    for cell in CellIterator(sdh)
        cellid[] = Ferrite.cellid(cell)
        copyto!(dofs, celldofs(cell))
        ae .= view(a, dofs)
        ae_n .= view(a_n, dofs)
        reinit!(prob.cv, cell)
        ForwardDiff.jacobian!(Ke, f!, re, ae, cfg)
        assemble!(assembler, dofs, Ke, re)
    end
    return
end

function solve_sve(
        grid, grain_of_cell, materials; format, ubc, ξbc, Γ = 0.1, T = 15.0, nsteps = 30,
        vtkfile = nothing, verbose = true, linsolve = LinearSolve.UMFPACKFactorization(),
    )
    Δt = T / nsteps
    dim = Ferrite.getspatialdim(grid)
    Hshear = basevec(Vec{dim}, 1) ⊗ basevec(Vec{dim}, 2)
    Hbar(t) = (t / T * Γ) * Hshear
    dh, sdh, ch, K, prob, states, states_n = setup(grid, materials, format, ubc, ξbc, Hbar, Δt)
    verbose && println("$(format) u-$(ubc) ξ-$(ξbc): $(ndofs(dh)) dofs")
    a = zeros(ndofs(dh))
    a_n = copy(a)
    a_nn = copy(a)
    r = zeros(ndofs(dh))
    Δa = similar(a)
    linear_cache = LinearSolve.init(LinearSolve.LinearProblem(K, r), linsolve)
    V = 0.0
    for cell in CellIterator(sdh)
        reinit!(prob.cv, cell)
        V += sum(getdetJdV(prob.cv, qp) for qp in 1:getnquadpoints(prob.cv))
    end
    S = eltype(materials[1].P)
    results = (t = Float64[], ε̄ = S[], σ̄ = S[], π̄ = Float64[])
    if vtkfile !== nothing
        bulk = collect(getcellset(grid, "bulk"))
        proj = L2Projector(Lagrange{RefTetrahedron, 1}(), grid; set = bulk)
        surface = Grid([Triangle(Ferrite.facets(getcells(grid, c))[f]) for (c, f) in getfacetset(grid, "boundary")], getnodes(grid))
        vtkhdf = VTKHDFGridFile(vtkfile * ".vtkhdf", grid; temporal = true, compress = true)
        vtkhdf_surface = VTKHDFGridFile(vtkfile * "_surface.vtkhdf", surface; temporal = true, compress = true)
    end
    for step in 1:nsteps
        t = step * Δt
        a .= 2 .* a_n .- a_nn
        update!(ch, t)
        apply!(a, ch)
        for st in states, s in st
            fill!(s.Y, NaN) # the local problem depends on the step through ⁿγ
        end
        assemble_system!(K, r, a, a_n, dh, sdh, prob, states, states_n, materials, grain_of_cell, t)
        apply_zero!(K, r, ch)
        res = norm(r)
        converged = false
        for iter in 1:30
            verbose && @printf("  t = %5.2f  iter %2d  |r| = %.3e\n", t, iter, res)
            if res < 1.0e-6
                converged = true
                break
            end
            linear_cache.A = K
            linear_cache.b = r
            sol = LinearSolve.solve!(linear_cache)
            sol.retcode == LinearSolve.ReturnCode.Success || error("linear solve failed: $(sol.retcode)")
            copyto!(Δa, sol.u) # preserve the Newton direction when the cache is reused
            apply_zero!(Δa, ch)
            α = 1.0
            accepted = false
            for _ in 1:6
                a .-= α .* Δa
                assemble_system!(K, r, a, a_n, dh, sdh, prob, states, states_n, materials, grain_of_cell, t)
                apply_zero!(K, r, ch)
                res = norm(r)
                linear_cache.b = r
                sol = LinearSolve.solve!(linear_cache)
                sol.retcode == LinearSolve.ReturnCode.Success || error("linear solve failed: $(sol.retcode)")
                Δa_trial = sol.u
                apply_zero!(Δa_trial, ch)
                accepted = norm(Δa_trial) < norm(Δa)
                accepted && break
                a .+= α .* Δa
                α /= 2
            end
            accepted || error("line search failed at t = $t (try a smaller time step)")
        end
        converged || error("no convergence at t = $t")
        σ̄ = zero(S)
        ε̄ = zero(S)
        π̄ = 0.0
        for cell in CellIterator(sdh)
            reinit!(prob.cv, cell)
            for (qp, st) in enumerate(states[Ferrite.cellid(cell)])
                dΩ = getdetJdV(prob.cv, qp)
                σ̄ += st.σ * dΩ
                ε̄ += st.ε * dΩ
                π̄ += st.π * dΩ
            end
        end
        push!(results.t, t)
        push!(results.σ̄, σ̄ / V)
        push!(results.ε̄, ε̄ / V)
        push!(results.π̄, π̄ / V)
        verbose && @printf("t = %5.2f  σ̄₁₂ = %8.2f MPa  ε̄₁₂ = %.4f  π̄ = %8.3f MPa\n", t, σ̄[1, 2] / V, ε̄[1, 2] / V, π̄ / V)
        copyto!(a_nn, a_n)
        copyto!(a_n, a)
        for (st, stn) in zip(states, states_n), (s, sn) in zip(st, stn)
            update_state!(sn, s)
        end
        if vtkfile !== nothing
            σ12_qp = Dict(c => [st.σ[1, 2] for st in states[c]] for c in bulk) # keyed by cell id
            σ12_nodes = project(proj, σ12_qp, Ferrite.get_quadrature_rule(prob.cv))
            u_nodes = evaluate_at_grid_nodes(dh, a, :u)
            slip = fill(NaN, getncells(grid))
            for c in bulk
                slip[c] = sum(st -> sum(abs, st.γ), states[c]) / length(states[c])
            end
            write_timestep(vtkhdf, t) do vtk
                write_node_data(vtk, u_nodes, "u")
                write_projection(vtk, proj, σ12_nodes, "σ12")
                write_cell_data(vtk, slip, "sum of absolute slips")
                write_cell_data(vtk, grain_of_cell, "grain")
            end
            write_timestep(vtkhdf_surface, t) do vtk
                write_node_data(vtk, u_nodes, "u")
                write_projection(vtk, proj, σ12_nodes, "σ12")
            end
        end
    end
    if vtkfile !== nothing
        close(vtkhdf)
        close(vtkhdf_surface)
    end
    return results
end

cases = [(ubc, ξbc) for ubc in (:dirichlet, :neumann) for ξbc in (:microhard, :microfree)]
label(ubc, ξbc) = string("u", ubc === :dirichlet ? "D" : "N", "-ξ", ξbc === :microhard ? "N" : "D")
formats = (:dual,)

function run_cases(meshfile; nsteps)
    isfile(meshfile) || Downloads.download(Ferrite.asset_url(meshfile), meshfile)
    grid, grain_of_cell = load_sve(meshfile)
    ngrains = maximum(grain_of_cell)
    println("$(ngrains) grains, $(length(getcellset(grid, "bulk"))) tetrahedra, $(getnnodes(grid)) nodes ($(length(getnodeset(grid, "grain_boundary"))) on grain boundaries)")
    materials = paper_materials(ngrains)
    results = Dict{Tuple{Symbol, Symbol, Symbol}, Any}()
    for format in formats, (ubc, ξbc) in cases
        vtkfile = format === :dual ? "gradient_crystal_plasticity_$(label(ubc, ξbc))" : nothing
        results[(format, ubc, ξbc)] = solve_sve(grid, grain_of_cell, materials; format, ubc, ξbc, nsteps, vtkfile, verbose = false)
    end
    return results
end

results = run_cases(FULL ? "gradient_crystal_plasticity_n50.inp" : "gradient_crystal_plasticity_n10.inp"; nsteps = FULL ? 30 : 6)

function print_summary(results)
    for (ubc, ξbc) in cases
        @printf("%-6s π̄ = %7.3f MPa   σ̄₁₂ = %8.2f MPa\n", label(ubc, ξbc), results[(:dual, ubc, ξbc)].π̄[end], results[(:dual, ubc, ξbc)].σ̄[end][1, 2])
    end
    πD = [results[(:dual, ubc, ξbc)].π̄[end] for (ubc, ξbc) in cases]
    ordered = πD[4] <= min(πD[2], πD[3]) && max(πD[2], πD[3]) <= πD[1] # uN-ξD ≤ (uD-ξD, uN-ξN) ≤ uD-ξN
    println("ordering uN-ξD ≤ (uD-ξD, uN-ξN) ≤ uD-ξN holds: ", ordered)
    return
end
print_summary(results)

function plot_stress_strain(results, foreground_color)
    plt = Plots.plot(;
        xlabel = "ε̄₁₂ [-]", ylabel = "σ̄₁₂ [MPa]", legend = :bottomright,
        background_color = :transparent, foreground_color,
    )
    for (i, (ubc, ξbc)) in enumerate(cases), format in formats
        r = results[(format, ubc, ξbc)]
        Plots.plot!(
            plt, [0.0; [ε[1, 2] for ε in r.ε̄]], [0.0; [σ[1, 2] for σ in r.σ̄]];
            label = format === :dual ? label(ubc, ξbc) : string(label(ubc, ξbc), " (primal)"), lw = 2, color = i,
            ls = format === :dual ? :solid : :dash, marker = format === :dual ? :circle : :none, ms = 3,
        )
    end
    return plt
end
plot_stress_strain(results, :black)

function save_stress_plots(results, dir = ".")
    Plots.savefig(plot_stress_strain(results, :black), joinpath(dir, "gradient_crystal_plasticity_stress-light.svg"))
    Plots.savefig(plot_stress_strain(results, "#d6d6d6"), joinpath(dir, "gradient_crystal_plasticity_stress-dark.svg"))
    return
end
FULL && save_stress_plots(results)

# This file was generated using Literate.jl, https://github.com/fredrikekre/Literate.jl
