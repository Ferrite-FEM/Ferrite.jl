using Ferrite, SparseArrays, LinearAlgebra

using FerriteGmsh
# grid = saved_file_to_grid("periodic-rve-coarse.msh")
grid = saved_file_to_grid("periodic-rve.msh")

dim = 2
ip = Lagrange{dim, RefTetrahedron, 1}()
qr = QuadratureRule{dim, RefTetrahedron}(2)
cellvalues = CellVectorValues(qr, ip);

dh = DofHandler(grid)
push!(dh, :u, 2)
close!(dh);

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

ch_periodic = ConstraintHandler(dh);
periodic = PeriodicDirichlet(
    :u,
    ["left" => "right", "bottom" => "top"],
    [1, 2]
)
add!(ch_periodic, periodic)
close!(ch_periodic)
update!(ch_periodic, 0.0)

ch = (dirichlet = ch_dirichlet, periodic = ch_periodic);

K = (
    dirichlet = create_sparsity_pattern(dh),
    periodic  = create_sparsity_pattern(dh, ch.periodic),
);

λ, μ = 1e10, 7e9 # Lamé parameters
δ(i,j) = i == j ? 1.0 : 0.0
Em = SymmetricTensor{4, 2}(
    (i,j,k,l) -> λ * δ(i,j) * δ(k,l) + μ * (δ(i,k) * δ(j,l) + δ(i,l) * δ(j,k))
)
Ei = 10 * Em;

εᴹ = [
      SymmetricTensor{2,2}([1.0 0.0; 0.0 0.0]), # ε_11 loading
      SymmetricTensor{2,2}([0.0 0.0; 0.0 1.0]), # ε_22 loading
      SymmetricTensor{2,2}([0.0 0.5; 0.5 0.0]), # ε_12/ε_21 loading
];

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

rhs = (
    dirichlet = doassemble!(cellvalues, K.dirichlet, dh, εᴹ),
    periodic  = doassemble!(cellvalues, K.periodic,  dh, εᴹ),
);

rhsdata = (
    dirichlet = get_rhs_data(ch.dirichlet, K.dirichlet),
    periodic  = get_rhs_data(ch.periodic,  K.periodic),
)

apply!(K.dirichlet, ch.dirichlet)
apply!(K.periodic,  ch.periodic)

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

for i in 1:size(rhs.periodic, 2)
    rhs_i = @view rhs.periodic[:, i]                   # Extract this RHS
    apply_rhs!(rhsdata.periodic, rhs_i, ch.periodic)   # Apply BC
    u_i = cholesky(Symmetric(K.periodic)) \ rhs_i      # Solve
    apply!(u_i, ch.periodic)                           # Apply BC on the solution
    push!(u.periodic, u_i)                             # Save the solution vector
end

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

E_voigt = vm * Em + (1-vm) * Ei
E_reuss = inv(vm * inv(Em) + (1-vm) * inv(Ei));

ev = (first ∘ eigvals).((E_reuss, E_periodic, E_dirichlet, E_voigt))
round.(ev; digits=-8)

chM = ConstraintHandler(dh)
add!(chM, Dirichlet(:u, Set(1:getnnodes(grid)), (x, t) -> εᴹ[Int(t)] ⋅ x, [1, 2]))
close!(chM)
uM = zeros(ndofs(dh))

vtk_grid("homogenization", dh) do vtk
    for i in 1:3
        # Compute macroscopic solution
        update!(chM, i)
        apply!(uM, chM)
        # Dirichlet
        vtk_point_data(vtk, dh, uM + u.dirichlet[i], "_dirichlet_$i")
        vtk_point_data(vtk, projector, σ.dirichlet[i], "σvM_dirichlet_$i")
        # Periodic
        vtk_point_data(vtk, dh, uM + u.periodic[i], "_periodic_$i")
        vtk_point_data(vtk, projector, σ.periodic[i], "σvM_periodic_$i")
    end
end;

# This file was generated using Literate.jl, https://github.com/fredrikekre/Literate.jl

