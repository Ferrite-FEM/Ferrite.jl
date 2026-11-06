using Ferrite, SparseArrays, LinearAlgebra

using FerriteGmsh
using Downloads: Downloads

meshfile = "periodic-rve.msh"
isfile(meshfile) || Downloads.download(Ferrite.asset_url(meshfile), meshfile)

grid = togrid(meshfile)

dim = 2
ip = Lagrange{RefTriangle, 1}()^dim
qr = QuadratureRule{RefTriangle}(2)
cellvalues = CellValues(qr, ip);

σ̄var = AlgebraicVariable{SymmetricTensor{2, 2}}()
algebraic_values = AlgebraicValues(σ̄var);

dh = DofHandler(grid)
add!(dh, :u, ip)
close!(dh);

dh_neumann = DofHandler(grid)
add!(dh_neumann, :u, ip)
add!(dh_neumann, :σbar, σ̄var)
close!(dh_neumann);

dofhandlers = (dirichlet = dh, periodic = dh, neumann = dh_neumann);

ch_dirichlet = ConstraintHandler(dofhandlers.dirichlet)
dirichlet = Dirichlet(
    :u,
    union(getfacetset.(Ref(grid), ["left", "right", "top", "bottom"])...),
    (x, t) -> [0, 0],
    [1, 2]
)
add!(ch_dirichlet, dirichlet)
close!(ch_dirichlet);

periodic_facets = collect_periodic_facets(grid, "left", "right")
collect_periodic_facets!(periodic_facets, grid, "bottom", "top")
periodic = PeriodicDirichlet(:u, periodic_facets, [1, 2]);

addvertexset!(grid, "corners", x -> all(abs.(x) .≈ 0.5))
corners = Dirichlet(:u, getvertexset(grid, "corners"), x -> zero(Vec{2}), [1, 2])

ch_periodic = ConstraintHandler(dofhandlers.periodic)
add!(ch_periodic, periodic)
add!(ch_periodic, corners)
close!(ch_periodic);

addvertexset!(grid, "bottom_left", x -> x ≈ Vec((-0.5, -0.5)))
addvertexset!(grid, "top_right", x -> x ≈ Vec((0.5, 0.5)))

ch_neumann = ConstraintHandler(dofhandlers.neumann)
add!(ch_neumann, Dirichlet(:u, getvertexset(grid, "bottom_left"), x -> zero(Vec{2}), [1, 2]))
add!(ch_neumann, Dirichlet(:u, getvertexset(grid, "top_right"), x -> 0.0, [1]))
close!(ch_neumann);

ch = (dirichlet = ch_dirichlet, periodic = ch_periodic, neumann = ch_neumann);

K_dirichlet = allocate_matrix(dofhandlers.dirichlet)
K_periodic = allocate_matrix(dofhandlers.periodic, ch.periodic)
coupling = CellCoupling(1:getncells(grid); algebraic_coupling = (:u, :σbar))
K_neumann = allocate_matrix(dofhandlers.neumann; algebraic_couplings = (coupling,))

K = (
    dirichlet = K_dirichlet,
    periodic = K_periodic,
    neumann = K_neumann,
);

λ, μ = 1.0e10, 7.0e9 # Lamé parameters
δ(i, j) = i == j ? 1.0 : 0.0
Em = SymmetricTensor{4, 2}(
    (i, j, k, l) -> λ * δ(i, j) * δ(k, l) + μ * (δ(i, k) * δ(j, l) + δ(i, l) * δ(j, k))
)
Ei = 10 * Em;

εᴹ = [
    SymmetricTensor{2, 2}([1.0 0.0; 0.0 0.0]), # ε_11 loading
    SymmetricTensor{2, 2}([0.0 0.0; 0.0 1.0]), # ε_22 loading
    SymmetricTensor{2, 2}([0.0 0.5; 0.5 0.0]), # ε_12/ε_21 loading
];

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

f_dirichlet = assemble_kuu!(cellvalues, K.dirichlet, dofhandlers.dirichlet, εᴹ, Ei, Em)
f_periodic = assemble_kuu!(cellvalues, K.periodic, dofhandlers.periodic, εᴹ, Ei, Em)
f_neumann = assemble_kuu!(cellvalues, K.neumann, dofhandlers.neumann, εᴹ, Ei, Em)
assemble_kuσ!(cellvalues, algebraic_values, K.neumann, dofhandlers.neumann)

rhs = (
    dirichlet = f_dirichlet,
    periodic = f_periodic,
    neumann = f_neumann,
);

rhsdata = (
    dirichlet = get_rhs_data(ch.dirichlet, K.dirichlet),
    periodic = get_rhs_data(ch.periodic, K.periodic),
    neumann = get_rhs_data(ch.neumann, K.neumann),
)

apply!(K.dirichlet, ch.dirichlet)
apply!(K.periodic, ch.periodic)
apply!(K.neumann, ch.neumann)

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

σ̄_multiplier = [algebraic_value(dofhandlers.neumann, u_i, :σbar) for u_i in u.neumann]
σ̄_multiplier ≈ σ̄.neumann

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

E_voigt = vm * Em + (1 - vm) * Ei
E_reuss = inv(vm * inv(Em) + (1 - vm) * inv(Ei));

ev = (first ∘ eigvals).((E_reuss, E_neumann, E_periodic, E_dirichlet, E_voigt))
round.(ev; digits = -8)

uM = zeros(ndofs(dh))

VTKGridFile("homogenization", dh) do vtk
    for i in 1:3
        # Compute macroscopic solution
        apply_analytical!(uM, dh, :u, x -> εᴹ[i] ⋅ x)
        # Dirichlet
        write_solution(vtk, dh, uM + u.dirichlet[i], "_dirichlet_$i")
        write_projection(vtk, projector, σ.dirichlet[i], "σvM_dirichlet_$i")
        # Periodic
        write_solution(vtk, dh, uM + u.periodic[i], "_periodic_$i")
        write_projection(vtk, projector, σ.periodic[i], "σvM_periodic_$i")
        # Neumann (only the displacement dofs, which are numbered first)
        write_solution(vtk, dh, uM + u.neumann[i][1:ndofs(dh)], "_neumann_$i")
        write_projection(vtk, projector, σ.neumann[i], "σvM_neumann_$i")
    end
end;

# This file was generated using Literate.jl, https://github.com/fredrikekre/Literate.jl
