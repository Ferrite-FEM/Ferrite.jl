using Ferrite, SparseArrays, LinearAlgebra
using KrylovKit, WriteVTK
import Plots

L = 1.0    # beam length in x [m]
w = 0.05   # width in y [m]
h = 0.01   # thickness in z [m]
grid = generate_grid(Hexahedron, (60, 4, 2), zero(Vec{3}), Vec(L, w, h));

ip = Lagrange{RefHexahedron, 2}()^3
ip_geo = Lagrange{RefHexahedron, 1}()
qr = QuadratureRule{RefHexahedron}(3)
cellvalues = CellValues(qr, ip, ip_geo);

dh = DofHandler(grid)
add!(dh, :u, ip)
close!(dh)

ch = ConstraintHandler(dh)
add!(ch, Dirichlet(:u, getfacetset(grid, "left"), (x, t) -> zero(Vec{3})))
close!(ch);

Emod = 210.0e9 # Young's modulus [Pa]
ν = 0.3        # Poisson's ratio [-]
ρ = 7850.0     # density [kg/m³]

Gmod = Emod / (2 * (1 + ν))  # Shear modulus
Kmod = Emod / (3 * (1 - 2ν)) # Bulk modulus

C = gradient(ϵ -> 2 * Gmod * dev(ϵ) + 3 * Kmod * vol(ϵ), zero(SymmetricTensor{2, 3}));

function assemble_cell!(ke, me, cellvalues, C, ρ)
    for q_point in 1:getnquadpoints(cellvalues)
        dΩ = getdetJdV(cellvalues, q_point)
        for i in 1:getnbasefunctions(cellvalues)
            Nᵢ = shape_value(cellvalues, q_point, i)
            ∇Nᵢ = shape_gradient(cellvalues, q_point, i)
            for j in 1:getnbasefunctions(cellvalues)
                Nⱼ = shape_value(cellvalues, q_point, j)
                ∇ˢʸᵐNⱼ = shape_symmetric_gradient(cellvalues, q_point, j)
                ke[i, j] += (∇Nᵢ ⊡ C ⊡ ∇ˢʸᵐNⱼ) * dΩ
                me[i, j] += ρ * (Nᵢ ⋅ Nⱼ) * dΩ
            end
        end
    end
    return
end

function assemble_global!(K, M, dh, cellvalues, C, ρ)
    n_basefuncs = getnbasefunctions(cellvalues)
    ke = zeros(n_basefuncs, n_basefuncs)
    me = zeros(n_basefuncs, n_basefuncs)
    K_assembler = start_assemble(K)
    M_assembler = start_assemble(M)
    for cell in CellIterator(dh)
        reinit!(cellvalues, cell)
        fill!(ke, 0.0)
        fill!(me, 0.0)
        assemble_cell!(ke, me, cellvalues, C, ρ)
        assemble!(K_assembler, celldofs(cell), ke)
        assemble!(M_assembler, celldofs(cell), me)
    end
    return
end

K = allocate_matrix(dh)
M = allocate_matrix(dh)
assemble_global!(K, M, dh, cellvalues, C, ρ);

fdofs = free_dofs(ch)
Kf = K[fdofs, fdofs]
Mf = M[fdofs, fdofs];

nev = 6 # number of modes to compute
Kfact = cholesky(Symmetric(Kf))
vals, vecs, info = eigsolve(x -> Kfact \ (Mf * x), length(fdofs), nev, :LM; krylovdim = 40)
info.converged >= nev || error("eigensolver did not converge");

ω² = [1 / real(v) for v in vals[1:nev]]
freqs = sqrt.(ω²) ./ (2π)

function analytical_frequencies(EI, m, L, n)
    βL² = (3.51602, 22.0345, 61.6972, 120.0902)
    return [βL²[i] / (2π * L^2) * sqrt(EI / m) for i in 1:n]
end
A = w * h
f_weak = analytical_frequencies(Emod * w * h^3 / 12, ρ * A, L, 4)
f_strong = analytical_frequencies(Emod * h * w^3 / 12, ρ * A, L, 2)
reference = sort!(
    vcat(
        [(f, "weak (z)") for f in f_weak],
        [(f, "strong (y)") for f in f_strong],
    )
)[1:nev]
f_analytical = first.(reference);

println("mode  bending axis  f_FE [Hz]  f_beam [Hz]  error [%]")
for i in 1:nev
    f_ref, axis = reference[i]
    err = round(100 * abs(freqs[i] - f_ref) / f_ref; digits = 1)
    println(
        lpad(i, 4), "  ", rpad(axis, 12), "  ", lpad(round(freqs[i]; digits = 1), 9),
        "  ", lpad(round(f_ref; digits = 1), 11), "  ", lpad(err, 9),
    )
end

for (i, ϕf) in enumerate(vecs[1:nev])
    ϕ = zeros(ndofs(dh))
    ϕ[fdofs] .= real.(ϕf) ./ maximum(abs ∘ real, ϕf)
    VTKGridFile("elastodynamics_mode_$i", dh) do vtk
        write_solution(vtk, dh, ϕ)
    end
end

ζ = 0.02
ω₁, ω₃ = sqrt(ω²[1]), sqrt(ω²[3])
α = 2 * ζ * ω₁ * ω₃ / (ω₁ + ω₃)
β = 2 * ζ / (ω₁ + ω₃)
Cf = α * Mf + β * Kf;

function assemble_external_forces!(f_ext, dh, facetset, facetvalues, prescribed_traction)
    # Create a temporary array for the facet's local contributions to the external force vector
    fe_ext = zeros(getnbasefunctions(facetvalues))
    for facet in FacetIterator(dh, facetset)
        # Update the facetvalues to the correct facet number
        reinit!(facetvalues, facet)
        # Reset the temporary array for the next facet
        fill!(fe_ext, 0.0)
        # Access the cell's coordinates
        cell_coordinates = getcoordinates(facet)
        for qp in 1:getnquadpoints(facetvalues)
            # Calculate the global coordinate of the quadrature point.
            x = spatial_coordinate(facetvalues, qp, cell_coordinates)
            tₚ = prescribed_traction(x)
            # Get the integration weight for the current quadrature point.
            dΓ = getdetJdV(facetvalues, qp)
            for i in 1:getnbasefunctions(facetvalues)
                Nᵢ = shape_value(facetvalues, qp, i)
                fe_ext[i] += tₚ ⋅ Nᵢ * dΓ
            end
        end
        # Add the local contributions to the correct indices in the global external force vector
        assemble!(f_ext, celldofs(facet), fe_ext)
    end
    return f_ext
end

facet_qr = FacetQuadratureRule{RefHexahedron}(3)
facetvalues = FacetValues(facet_qr, ip, ip_geo)
f_static = zeros(ndofs(dh))
assemble_external_forces!(f_static, dh, getfacetset(grid, "right"), facetvalues, x -> Vec(0.0, 0.0, -1.0e4));

u₀ = Kfact \ f_static[fdofs];

x_tip = Vec(L, w / 2, h / 2)
ph = PointEvalHandler(grid, [x_tip]);

u_full = zeros(ndofs(dh))
u_full[fdofs] .= u₀
apply!(u_full, ch)
δ_tip = evaluate_at_points(ph, dh, u_full, :u)[1][3]
P = -1.0e4 * w * h  # resultant tip load [N]
δ_beam = P * L^3 / (3 * Emod * w * h^3 / 12)
(δ_tip, δ_beam)

v₀ = zero(u₀)
a₀ = cholesky(Symmetric(Mf)) \ (-Kf * u₀);

T₁ = 2π / ω₁            # first natural period [s]
Δt = T₁ / 100
n_steps = 300

γ_N = 1 / 2
β_N = 1 / 4
M_eff = cholesky(Symmetric(Mf + γ_N * Δt * Cf + β_N * Δt^2 * Kf));

function newmark_solve!(pvd, u, v, a, dh, ch, ph, fdofs, Kf, Cf, M_eff, Δt, β_N, γ_N, n_steps)
    u_full = zeros(ndofs(dh))
    u_full[fdofs] .= u
    apply!(u_full, ch)
    tip_history = [(0.0, evaluate_at_points(ph, dh, u_full, :u)[1][3])]
    # Write the initial condition (the static deflection) as the first frame
    VTKGridFile("elastodynamics_step_0", dh) do vtk
        write_solution(vtk, dh, u_full)
        pvd[0.0] = vtk
    end
    for step in 1:n_steps
        t = step * Δt
        # Predictors
        ũ = u .+ Δt .* v .+ (1 / 2 - β_N) * Δt^2 .* a
        ṽ = v .+ (1 - γ_N) * Δt .* a
        # Solve for the new acceleration (the external force is zero after the release)
        a .= M_eff \ (-Cf * ṽ - Kf * ũ)
        # Correctors
        u .= ũ .+ β_N * Δt^2 .* a
        v .= ṽ .+ γ_N * Δt .* a
        # Postprocessing
        u_full[fdofs] .= u
        apply!(u_full, ch)
        push!(tip_history, (t, evaluate_at_points(ph, dh, u_full, :u)[1][3]))
        if step % 5 == 0
            VTKGridFile("elastodynamics_step_$step", dh) do vtk
                write_solution(vtk, dh, u_full)
                pvd[t] = vtk
            end
        end
    end
    return tip_history
end

pvd = paraview_collection("elastodynamics")
tip_history = newmark_solve!(pvd, copy(u₀), v₀, a₀, dh, ch, ph, fdofs, Kf, Cf, M_eff, Δt, β_N, γ_N, n_steps)
vtk_save(pvd);

ts = first.(tip_history)
u_tip = last.(tip_history)
ωd = ω₁ * sqrt(1 - ζ^2)
u_sdof = @. u_tip[1] * exp(-ζ * ω₁ * ts) * (cos(ωd * ts) + ζ / sqrt(1 - ζ^2) * sin(ωd * ts));

function plot_tip_deflection(foreground_color)
    plt = Plots.plot(
        ts, u_tip .* 1.0e3; label = "FE simulation", lw = 2,
        xlabel = "time [s]", ylabel = "tip deflection [mm]",
        background_color = :transparent, foreground_color = foreground_color,
    )
    Plots.plot!(plt, ts, u_sdof .* 1.0e3; label = "single mode solution", ls = :dash, lw = 2)
    return plt
end
plt = plot_tip_deflection(:black)

using Test                                                     #hide
@test freqs ≈ f_analytical rtol = 0.03                         #hide
@test δ_tip ≈ δ_beam rtol = 0.03                               #hide
@test u_tip ≈ u_sdof rtol = 0.06                               #hide
nothing                                                        #hide

# This file was generated using Literate.jl, https://github.com/fredrikekre/Literate.jl
