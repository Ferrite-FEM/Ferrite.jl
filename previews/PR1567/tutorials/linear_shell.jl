using Ferrite, Plots

function plane_stress_stiffness(E, ν)
    I = one(SymmetricTensor{2, 2})
    Isym = one(SymmetricTensor{4, 2})
    return E / (1 - ν^2) * ((1 - ν) * Isym + ν * I ⊗ I)
end;

function skew_tensor(n::Vec{3})
    return Tensor{2, 3}((0.0, n[3], -n[2], -n[3], 0.0, n[1], n[2], -n[1], 0.0))
end

function shell_strains(∇u, ω, ∇ω, E, n)
    W = skew_tensor(n)
    θ, ∇θ = -(W ⋅ ω), -(W ⋅ ∇ω)
    G = E' ⋅ ∇u
    ε = symmetric(G)
    κ = symmetric(E' ⋅ ∇θ)
    γ = ∇u' ⋅ n + E' ⋅ θ
    ϖ = ω ⋅ n - (G[2, 1] - G[1, 2]) / 2
    return ε, κ, γ, ϖ
end;

struct ShellValues{CV}
    cv::CV
    E::Vector{MixedTensor2{3, 2, Float64, 6}} # Local frame
    n::Vector{Vec{3, Float64}}                # Normal
end

function ShellValues(qr, ipu, ipω, ip_geo)
    cv = MultiFieldCellValues(qr, (u = ipu^3, ω = ipω^3), ip_geo^3; update_local_gradients = true)
    nqp = getnquadpoints(cv)
    return ShellValues(cv, zeros(MixedTensor2{3, 2, Float64, 6}, nqp), zeros(Vec{3, Float64}, nqp))
end;

function Ferrite.reinit!(sv::ShellValues, x)
    cv = sv.cv
    reinit!(cv, x)
    for qp in 1:getnquadpoints(cv)
        E = Ferrite.gram_schmidt_frame(Ferrite.getjacobian(cv, qp))
        sv.E[qp] = E
        sv.n[qp] = E[:, 1] × E[:, 2]
    end
    reinit_local_frame!(cv, sv.E)
    return sv
end;

struct ShellElement{SV, SVS, SVA}
    sv::SV
    sv_shear::SVS
    sv_ans::SVA
    A::SymmetricTensor{4, 2, Float64, 9}  # Membrane stiffness
    D::SymmetricTensor{4, 2, Float64, 9}  # Bending stiffness
    Aₛ::Tensor{2, 2, Float64, 4}          # Transverse shear stiffness
    K_d::Float64                          # Drilling stiffness
    δε::Vector{SymmetricTensor{2, 2, Float64, 3}} # Membrane strains of the test functions
    δκ::Vector{SymmetricTensor{2, 2, Float64, 3}} # Curvatures of the test functions
    δγ::Vector{Vec{2, Float64}}                   # Transverse shear strains of the test functions
    δϖ::Vector{Float64}                           # Drilling strains of the test functions
    γξ_tying::Matrix{Vec{2, Float64}}             # Covariant shear strains in the tying points (MITC4)
end

function ShellElement(ipu, ipω, ip_geo, qr, qr_shear, qr_ans = nothing; h, E, ν, kₛ = 5 / 6)
    sv = ShellValues(qr, ipu, ipω, ip_geo)
    sv_ans = qr_ans === nothing ? nothing : ShellValues(qr_ans, ipu, ipω, ip_geo)
    C = plane_stress_stiffness(E, ν)
    G = E / (2 * (1 + ν))
    ndofs = getnbasefunctions(sv.cv.u) + getnbasefunctions(sv.cv.ω)
    return ShellElement(
        sv, ShellValues(qr_shear, ipu, ipω, ip_geo), sv_ans,
        h * C, h^3 / 12 * C, kₛ * h * G * one(Tensor{2, 2}), E * h^3,
        zeros(SymmetricTensor{2, 2, Float64, 3}, ndofs), zeros(SymmetricTensor{2, 2, Float64, 3}, ndofs),
        zeros(Vec{2, Float64}, ndofs), zeros(ndofs), zeros(Vec{2, Float64}, ndofs, 4)
    )
end;

function Ferrite.reinit!(se::ShellElement, x)
    for sv in (se.sv, se.sv_shear, se.sv_ans)
        sv === nothing || reinit!(sv, x)
    end
    return se
end;

function shape_strains(sv::ShellValues, qp, i)
    cv, E, n = sv.cv, sv.E[qp], sv.n[qp]
    nu = getnbasefunctions(cv.u)
    if i <= nu
        ∇Nu = shape_local_gradient(cv.u, qp, i)
        return shell_strains(∇Nu, zero(Vec{3}), zero(typeof(∇Nu)), E, n)
    else
        Nω, ∇Nω = shape_value(cv.ω, qp, i - nu), shape_local_gradient(cv.ω, qp, i - nu)
        return shell_strains(zero(typeof(∇Nω)), Nω, ∇Nω, E, n)
    end
end;

function assemble_membrane_bending_drilling!(ke, fe, se::ShellElement, q)
    sv = se.sv
    cv = sv.cv
    δε, δκ, δϖ = se.δε, se.δκ, se.δϖ
    ndofs = size(ke, 1)
    for qp in 1:getnquadpoints(cv)
        dA = getdetJdV(cv, qp)
        for i in 1:ndofs
            δε[i], δκ[i], _, δϖ[i] = shape_strains(sv, qp, i)
        end
        for i in 1:ndofs, j in 1:ndofs
            ke[i, j] += (δε[i] ⊡ se.A ⊡ δε[j] + δκ[i] ⊡ se.D ⊡ δκ[j] + δϖ[i] * se.K_d * δϖ[j]) * dA
        end
        for i in 1:getnbasefunctions(cv.u)
            fe[i] += shape_value(cv.u, qp, i) ⋅ q * dA
        end
    end
    return ke, fe
end;

function assemble_shear!(ke, se::ShellElement)
    sv = se.sv_shear
    δγ = se.δγ
    ndofs = size(ke, 1)
    for qp in 1:getnquadpoints(sv.cv)
        dA = getdetJdV(sv.cv, qp)
        for i in 1:ndofs
            _, _, δγ[i], _ = shape_strains(sv, qp, i)
        end
        for i in 1:ndofs, j in 1:ndofs
            ke[i, j] += δγ[i] ⋅ se.Aₛ ⋅ δγ[j] * dA
        end
    end
    return ke
end;

function assemble_shell!(ke, fe, se::ShellElement, q)
    assemble_membrane_bending_drilling!(ke, fe, se, q)
    assemble_shear!(ke, se)
    return ke, fe
end;

const tying_points = [Vec((0.0, 1.0)), Vec((-1.0, 0.0)), Vec((0.0, -1.0)), Vec((1.0, 0.0))] # A, B, C, D

function tying_interpolation(ξ::Vec{2})
    e1e1 = basevec(Vec{2}, 1) ⊗ basevec(Vec{2}, 1)
    e2e2 = basevec(Vec{2}, 2) ⊗ basevec(Vec{2}, 2)
    return ((1 + ξ[2]) / 2 * e1e1, (1 - ξ[1]) / 2 * e2e2, (1 - ξ[2]) / 2 * e1e1, (1 + ξ[1]) / 2 * e2e2)
end;

qr_tying = QuadratureRule{RefQuadrilateral}(ones(4), tying_points);

local_to_reference(sv::ShellValues, qp) = sv.E[qp]' ⋅ Ferrite.getjacobian(sv.cv, qp) # B

function covariant_shear_strain(sv::ShellValues, qp, i)
    _, _, γ, _ = shape_strains(sv, qp, i)
    return local_to_reference(sv, qp)' ⋅ γ
end;

function assemble_shell_ans!(ke, fe, se::ShellElement, q)
    assemble_membrane_bending_drilling!(ke, fe, se, q)
    # Transverse shear with assumed natural strains
    sv = se.sv
    δγ, γξ_tying = se.δγ, se.γξ_tying
    ndofs = size(ke, 1)
    for t in 1:4, i in 1:ndofs
        γξ_tying[i, t] = covariant_shear_strain(se.sv_ans, t, i)
    end
    qr_points = Ferrite.getpoints(Ferrite.get_quadrature_rule(sv.cv))
    for qp in 1:getnquadpoints(sv.cv)
        Binv = inv(local_to_reference(sv, qp))
        P = tying_interpolation(qr_points[qp])
        dA = getdetJdV(sv.cv, qp)
        for i in 1:ndofs
            γξ = P[1] ⋅ γξ_tying[i, 1] + P[2] ⋅ γξ_tying[i, 2] + P[3] ⋅ γξ_tying[i, 3] + P[4] ⋅ γξ_tying[i, 4]
            δγ[i] = Binv' ⋅ γξ
        end
        for i in 1:ndofs, j in 1:ndofs
            ke[i, j] += δγ[i] ⋅ se.Aₛ ⋅ δγ[j] * dA
        end
    end
    return ke, fe
end;

const R, L, φ₀ = 25.0, 50.0, deg2rad(40.0)
const h, Emod, ν, kₛ = 0.25, 4.32e8, 0.0, 5 / 6
const q = Vec((0.0, 0.0, -90.0))
const w_ref = 0.3024;

flat_grid(::Type{Quadrilateral}, n) = generate_grid(Quadrilateral, (n, n), Vec((-φ₀, 0.0, 0.0)), Vec((φ₀, L, 0.0)))

function flat_grid(::Type{Triangle}, n) # The triangle generator only supports 2D, so the nodes are lifted to 3D
    grid = generate_grid(Triangle, (n, n), Vec((-φ₀, 0.0)), Vec((φ₀, L)))
    nodes = [Node(Vec((x[1], x[2], 0.0))) for x in get_node_coordinate.(getnodes(grid))]
    return Grid(getcells(grid), nodes; facetsets = grid.facetsets)
end

function roof_grid(C, n)
    grid = flat_grid(C, n)
    transform_coordinates!(grid, x -> Vec((R * sin(x[1]), x[2], R * cos(x[1]))))
    addfacetset!(grid, "midspan", x -> x[2] ≈ L / 2)
    addnodeset!(grid, "edge midpoints", x -> x[2] ≈ L / 2 && abs(x[1]) ≈ R * sin(φ₀))
    return grid
end;

function solve_roof(grid, se::ShellElement, element_routine!)
    dh = DofHandler(grid)
    add!(dh, :u, Ferrite.function_interpolation(se.sv.cv.u))
    add!(dh, :ω, Ferrite.function_interpolation(se.sv.cv.ω))
    close!(dh)

    ch = ConstraintHandler(dh)
    diaphragms = union(getfacetset(grid, "bottom"), getfacetset(grid, "top"))
    add!(ch, Dirichlet(:u, diaphragms, (x, t) -> zero(Vec{2}), [1, 3]))
    add!(ch, Dirichlet(:u, getfacetset(grid, "midspan"), (x, t) -> 0.0, [2]))
    close!(ch)

    K = allocate_matrix(dh)
    f = zeros(ndofs(dh))
    ke = zeros(ndofs_per_cell(dh), ndofs_per_cell(dh))
    fe = zeros(ndofs_per_cell(dh))
    assembler = start_assemble(K, f)
    for cell in CellIterator(dh)
        fill!(ke, 0.0)
        fill!(fe, 0.0)
        reinit!(se, getcoordinates(cell))
        element_routine!(ke, fe, se, q)
        assemble!(assembler, celldofs(cell), ke, fe)
    end
    apply!(K, f, ch)
    a = K \ f

    u_nodes = evaluate_at_grid_nodes(dh, a, :u)
    edge = getnodeset(grid, "edge midpoints")
    w = -sum(u_nodes[i][3] for i in edge) / length(edge)
    return w, dh, a
end;

Q1 = Lagrange{RefQuadrilateral, 1}()
P2 = Lagrange{RefTriangle, 2}()
CR = CrouzeixRaviart{RefTriangle, 1}()
qr_quad(n) = QuadratureRule{RefQuadrilateral}(n)
qr_tri(n) = QuadratureRule{RefTriangle}(n)

shell_variants = [
    (name = "Q1/Q1",     cell = Quadrilateral, ipu = Q1, ipω = Q1, qr = qr_quad(2), qr_shear = qr_quad(2), qr_ans = nothing,  routine! = assemble_shell!),
    (name = "Q1/Q1 SRI", cell = Quadrilateral, ipu = Q1, ipω = Q1, qr = qr_quad(2), qr_shear = qr_quad(1), qr_ans = nothing,  routine! = assemble_shell!),
    (name = "MITC4",     cell = Quadrilateral, ipu = Q1, ipω = Q1, qr = qr_quad(2), qr_shear = qr_quad(2), qr_ans = qr_tying, routine! = assemble_shell_ans!),
    (name = "P2/CR",     cell = Triangle,      ipu = P2, ipω = CR, qr = qr_tri(2),  qr_shear = qr_tri(2),  qr_ans = nothing,  routine! = assemble_shell!),
];

nels = [4, 8, 16, 32]

function convergence_study(variant, nels)
    ip_geo = Ferrite.geometric_interpolation(variant.cell)
    se = ShellElement(variant.ipu, variant.ipω, ip_geo, variant.qr, variant.qr_shear, variant.qr_ans; h, E = Emod, ν, kₛ)
    return map(nels) do n
        w, _, _ = solve_roof(roof_grid(variant.cell, n), se, variant.routine!)
        w / w_ref
    end
end

results = [convergence_study(v, nels) for v in shell_variants];

colors = ["#2a78d6", "#008300", "#4a3aa7", "#e87ba4"]
markers = [:circle, :star5, :hexagon, :dtriangle]

plt = plot(;
    xscale = :log2, xlabel = "Number of elements per side, n", ylabel = "Normalized displacement w/w_ref",
    title = "Scordelis–Lo roof", legend = :outerright, xticks = (nels, string.(nels)), ylims = (0, 1.3),
    size = (800, 420), margin = 5Plots.mm, grid = :y, gridalpha = 0.15,
)
hline!(plt, [1.0]; color = :gray, linestyle = :dash, linewidth = 1, label = "Reference")
for (k, r) in pairs(results)
    plot!(plt, nels, r;
        label = shell_variants[k].name, color = colors[k], linewidth = 2,
        marker = markers[k], markersize = 6, markerstrokecolor = :white,
    )
end
plt

println(rpad("n", 10), join(map(n -> lpad(string(n), 8), nels)))
for (v, r) in zip(shell_variants, results)
    println(rpad(v.name, 10), join(map(x -> lpad(string(round(x; digits = 3)), 8), r)))
end

mitc4 = shell_variants[3]
se = ShellElement(mitc4.ipu, mitc4.ipω, Q1, mitc4.qr, mitc4.qr_shear, mitc4.qr_ans; h, E = Emod, ν, kₛ)
_, dh, a = solve_roof(roof_grid(Quadrilateral, 16), se, mitc4.routine!)
VTKGridFile("linear_shell", dh) do vtk
    write_solution(vtk, dh, a)
end;

# This file was generated using Literate.jl, https://github.com/fredrikekre/Literate.jl
