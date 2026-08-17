using Ferrite, SparseArrays, LinearAlgebra

using FerriteGmsh
using Downloads: Downloads

meshfile = "periodic-rve.msh"
isfile(meshfile) || Downloads.download(Ferrite.asset_url(meshfile), meshfile)

grid = togrid(meshfile)
incl_cells = getcellset(grid, "inclusions");

λ, μ = 1.0e10, 7.0e9 # Lamé parameters of the matrix
δ(i, j) = i == j ? 1.0 : 0.0
Em = SymmetricTensor{4, 2}(
    (i, j, k, l) -> λ * δ(i, j) * δ(k, l) + μ * (δ(i, k) * δ(j, l) + δ(i, l) * δ(j, k))
)
Ei = 10 * Em
ν = λ / (2 * (λ + μ));

ip_u = Lagrange{RefTriangle, 1}()^2
ε̄var = AlgebraicVariable{SymmetricTensor{2, 2}}()

dh = DofHandler(grid)
add!(dh, :u, ip_u)
add!(dh, :εbar, ε̄var)
close!(dh);
nε = length(algebraic_dofs(dh, :εbar))

algebraic_dofs(dh, :εbar)

coupling = CellCoupling(dh, 1:getncells(grid); algebraic_coupling = ((:u, :εbar), (:εbar, :εbar)));

periodic_facets = collect_periodic_facets(grid, "left", "right")
collect_periodic_facets!(periodic_facets, grid, "bottom", "top")
addvertexset!(grid, "corner", x -> x ≈ Vec((0.5, 0.5)))
corner = getvertexset(grid, "corner")

function add_fluctuation_constraints!(ch, periodic_facets, corner)
    add!(ch, PeriodicDirichlet(:u, periodic_facets, [1, 2]))
    add!(ch, Dirichlet(:u, corner, x -> zero(Vec{2})))
    return ch
end

ch = ConstraintHandler(dh)
add_fluctuation_constraints!(ch, periodic_facets, corner)
close!(ch);

qr = QuadratureRule{RefTriangle}(2)
cv_u = CellValues(qr, ip_u)
av_ε = AlgebraicValues(ε̄var);

function assemble_system!(K, f, dh, cv_u, av_ε, σ̄, Ei, Em, incl_cells, ch = nothing)
    n = ndofs_per_cell(dh)
    nε = getnbasefunctions(av_ε)
    dofs = Vector{Int}(undef, n + nε)
    dofs[(n + 1):end] .= algebraic_dofs(dh, :εbar) # constant tail, written once
    range_u = dof_range(dh, :u)
    range_ε = (n + 1):(n + nε) # local placement of the strain dofs
    Ke = zeros(n + nε, n + nε)
    fe = zeros(n + nε)
    assembler = start_assemble(K, f)
    for cell in CellIterator(dh)
        reinit!(cv_u, cell)
        copyto!(dofs, celldofs(cell)) # refresh the first n entries
        fill!(Ke, 0)
        fill!(fe, 0)
        E = cellid(cell) in incl_cells ? Ei : Em
        for qp in 1:getnquadpoints(cv_u)
            dΩ = getdetJdV(cv_u, qp)
            for (iu, I) in pairs(range_u)
                δεi = shape_symmetric_gradient(cv_u, qp, iu)
                for (ju, J) in pairs(range_u)
                    εj = shape_symmetric_gradient(cv_u, qp, ju)
                    Ke[I, J] += (δεi ⊡ E ⊡ εj) * dΩ
                end
                for (jε, J) in pairs(range_ε)
                    Eⱼ = algebraic_basis_value(av_ε, jε)
                    v = (δεi ⊡ E ⊡ Eⱼ) * dΩ
                    Ke[I, J] += v
                    Ke[J, I] += v
                end
            end
            for (iε, I) in pairs(range_ε)
                Eᵢ = algebraic_basis_value(av_ε, iε)
                fe[I] += (σ̄ ⊡ Eᵢ) * dΩ
                for (jε, J) in pairs(range_ε)
                    Eⱼ = algebraic_basis_value(av_ε, jε)
                    Ke[I, J] += (Eᵢ ⊡ E ⊡ Eⱼ) * dΩ
                end
            end
        end
        if ch === nothing
            assemble!(assembler, dofs, Ke, fe)
        else
            apply_assemble!(assembler, ch, dofs, Ke, fe)
        end
    end
    return K, f
end;

σ̄ = SymmetricTensor{2, 2}((0.0, 1.0e9, 0.0)) # σ̄₁₂ = σ̄₂₁ = 1 GPa, rest zero

K = allocate_matrix(dh, ch; algebraic_couplings = (coupling,))
f = zeros(ndofs(dh))
assemble_system!(K, f, dh, cv_u, av_ε, σ̄, Ei, Em, incl_cells)
apply!(K, f, ch)
a = K \ f
apply!(a, ch);

ε̄ = algebraic_value(dh, a, :εbar)

Ḡ = σ̄[1, 2] / (2 * ε̄[1, 2])

function average_stress(a, dh, cv_u, Ei, Em, incl_cells)
    ε̄ = algebraic_value(dh, a, :εbar)
    σΩ = zero(SymmetricTensor{2, 2})
    vol = 0.0
    for cell in CellIterator(dh)
        reinit!(cv_u, cell)
        ae = a[celldofs(cell)]
        E = cellid(cell) in incl_cells ? Ei : Em
        for qp in 1:getnquadpoints(cv_u)
            dΩ = getdetJdV(cv_u, qp)
            ε = ε̄ + function_symmetric_gradient(cv_u, qp, ae)
            σΩ += (E ⊡ ε) * dΩ
            vol += dΩ
        end
    end
    return σΩ / vol
end
σ̄_check = average_stress(a, dh, cv_u, Ei, Em, incl_cells)
σ̄_check ≈ σ̄

function effective_stiffness(dh, cv_u, av_ε, coupling, Ei, Em, incl_cells, periodic_facets, corner)
    nε = getnbasefunctions(av_ε)
    gdofs = algebraic_dofs(dh, :εbar)
    # One constraint handler per unit strain Eβ, prescribing the algebraic dofs
    chs = map(1:nε) do β
        chβ = ConstraintHandler(dh)
        add_fluctuation_constraints!(chβ, periodic_facets, corner)
        for (α, gdof) in pairs(gdofs)
            add!(chβ, AffineConstraint(gdof, Pair{Int, Float64}[], α == β ? 1.0 : 0.0))
        end
        return close!(chβ)
    end
    K = allocate_matrix(dh, chs[1]; algebraic_couplings = (coupling,))
    assemble_system!(K, zeros(ndofs(dh)), dh, cv_u, av_ε, zero(SymmetricTensor{2, 2}), Ei, Em, incl_cells)
    rhsdata = get_rhs_data(chs[1], K)
    apply!(K, chs[1])
    F = lu(K)
    Ē = zeros(nε, nε)
    for (β, chβ) in pairs(chs)
        fβ = zeros(ndofs(dh))
        apply_rhs!(rhsdata, fβ, chβ)
        aβ = F \ fβ
        apply!(aβ, chβ)
        # Read off the resulting average stress column
        σ̄β = average_stress(aβ, dh, cv_u, Ei, Em, incl_cells)
        for α in 1:nε
            Ē[α, β] = σ̄β ⊡ algebraic_basis_value(av_ε, α)
        end
    end
    return Ē
end
Ē = effective_stiffness(dh, cv_u, av_ε, coupling, Ei, Em, incl_cells, periodic_facets, corner)

ē = Ē \ [σ̄ ⊡ algebraic_basis_value(av_ε, α) for α in 1:nε]
maximum(abs, ē - a[algebraic_dofs(dh, :εbar)])

function domain_volumes(dh, cv, incl_cells)
    Ω = Ωi = 0.0
    for cell in CellIterator(dh)
        reinit!(cv, cell)
        for qp in 1:getnquadpoints(cv)
            dΩ = getdetJdV(cv, qp)
            Ω += dΩ
            if cellid(cell) in incl_cells
                Ωi += dΩ
            end
        end
    end
    return Ω, Ωi
end
Ω_box, Ω_incl = domain_volumes(dh, cv_u, incl_cells)
v_incl = Ω_incl / Ω_box
E_voigt = v_incl * Ei + (1 - v_incl) * Em
E_reuss = inv(v_incl * inv(Ei) + (1 - v_incl) * inv(Em))
Ē11_bounds = (E_reuss[1, 1, 1, 1], E_voigt[1, 1, 1, 1])
Ē11_bounds[1] <= Ē[1, 1] <= Ē11_bounds[2]

Ē1212_bounds = (E_reuss[1, 2, 1, 2], E_voigt[1, 2, 1, 2])
Ē1212_bounds[1] <= Ē[2, 2] / 4 <= Ē1212_bounds[2]

function von_mises_stress(a, dh, cv_u, Ei, Em, incl_cells, ν)
    ε̄ = algebraic_value(dh, a, :εbar)
    σvM = zeros(getncells(dh.grid))
    for cell in CellIterator(dh)
        reinit!(cv_u, cell)
        ae = a[celldofs(cell)]
        E = cellid(cell) in incl_cells ? Ei : Em
        vol = 0.0
        for qp in 1:getnquadpoints(cv_u)
            dΩ = getdetJdV(cv_u, qp)
            ε = ε̄ + function_symmetric_gradient(cv_u, qp, ae)
            σ = E ⊡ ε
            σ33 = ν * (σ[1, 1] + σ[2, 2]) # plane strain
            σ3d = SymmetricTensor{2, 3}((σ[1, 1], σ[2, 1], 0.0, σ[2, 2], 0.0, σ33))
            s = dev(σ3d)
            σvM[cellid(cell)] += √(3 / 2 * s ⊡ s) * dΩ
            vol += dΩ
        end
        σvM[cellid(cell)] /= vol
    end
    return σvM
end

uμ_nodes = evaluate_at_grid_nodes(dh, a, :u)
u_total = [ε̄ ⋅ get_node_coordinate(grid, n) + uμ_nodes[n] for n in 1:getnnodes(grid)]

phase = zeros(getncells(grid)); phase[collect(incl_cells)] .= 1.0
VTKGridFile("stress_driven_homogenization", dh) do vtk
    write_solution(vtk, dh, a)
    write_node_data(vtk, u_total, "u_total")
    write_cell_data(vtk, von_mises_stress(a, dh, cv_u, Ei, Em, incl_cells, ν), "vonMises")
    write_cell_data(vtk, phase, "phase")
end;

using BlockArrays

gdofs = algebraic_dofs(dh, :εbar)
nu = ndofs(dh) - nε
gdofs == collect((nu + 1):ndofs(dh))

bsp = BlockSparsityPattern([nu, nε])
add_sparsity_entries!(bsp, dh, ch; algebraic_couplings = (coupling,))
Kb = allocate_matrix(BlockMatrix, bsp)
fb = mortar([zeros(nu), zeros(nε)])
assemble_system!(Kb, fb, dh, cv_u, av_ε, σ̄, Ei, Em, incl_cells, ch);

function solve_schur(Kb, fb)
    K_uu = Kb[Block(1), Block(1)]         # sparse, SPD after condensation
    K_uε = Matrix(Kb[Block(1), Block(2)]) # the three "dense columns" (small: n_u × 3)
    K_εε = Matrix(Kb[Block(2), Block(2)]) # 3 × 3
    F = cholesky(Symmetric(K_uu))
    Y = F \ K_uε           # K_uu⁻¹ K_uε
    S = K_εε - K_uε' * Y   # Schur complement
    u_f = F \ fb[Block(1)] # zero here, kept for generality
    ē = S \ (fb[Block(2)] - K_uε' * u_f)
    uμ = u_f - Y * ē
    return uμ, ē, S
end
uμ, ē_b, S = solve_schur(Kb, fb);

ε̄_b = algebraic_value(av_ε, ē_b)
maximum(abs, ε̄_b - ε̄)

maximum(abs, S / Ω_box - Ē) / maximum(abs, Ē)

# This file was generated using Literate.jl, https://github.com/fredrikekre/Literate.jl
