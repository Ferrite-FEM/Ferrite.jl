using Ferrite, SparseArrays

grid = generate_grid(Quadrilateral, (100, 100));

dim = 2
ip = Lagrange{dim, RefCube, 1}()
qr = QuadratureRule{dim, RefCube}(2)
cellvalues = CellScalarValues(qr, ip);

dh = DofHandler(grid)
add!(dh, :u, 1)
close!(dh);

K = create_sparsity_pattern(dh);
M = create_sparsity_pattern(dh);

f = zeros(ndofs(dh));

max_temp = 100
Δt = 1
T = 200
t_rise = 100
ch = ConstraintHandler(dh);

∂Ω₁ = union(getfaceset.((grid,), ["left", "right"])...)
dbc = Dirichlet(:u, ∂Ω₁, (x, t) -> 0)
add!(ch, dbc);

∂Ω₂ = union(getfaceset.((grid,), ["top", "bottom"])...)
dbc = Dirichlet(:u, ∂Ω₂, (x, t) -> max_temp * clamp(t / t_rise, 0, 1))
add!(ch, dbc)
close!(ch)
update!(ch, 0.0);

function doassemble_K!(K::SparseMatrixCSC, f::Vector, cellvalues::CellScalarValues{dim}, dh::DofHandler) where {dim}

    n_basefuncs = getnbasefunctions(cellvalues)
    Ke = zeros(n_basefuncs, n_basefuncs)
    fe = zeros(n_basefuncs)

    assembler = start_assemble(K, f)

    for cell in CellIterator(dh)

        fill!(Ke, 0)
        fill!(fe, 0)

        reinit!(cellvalues, cell)

        for q_point in 1:getnquadpoints(cellvalues)
            dΩ = getdetJdV(cellvalues, q_point)

            for i in 1:n_basefuncs
                v = shape_value(cellvalues, q_point, i)
                ∇v = shape_gradient(cellvalues, q_point, i)
                fe[i] += 0.1 * v * dΩ
                for j in 1:n_basefuncs
                    ∇u = shape_gradient(cellvalues, q_point, j)
                    Ke[i, j] += 1e-3 * (∇v ⋅ ∇u) * dΩ
                end
            end
        end

        assemble!(assembler, celldofs(cell), fe, Ke)
    end
    return K, f
end

function doassemble_M!(M::SparseMatrixCSC, cellvalues::CellScalarValues{dim}, dh::DofHandler) where {dim}

    n_basefuncs = getnbasefunctions(cellvalues)
    Me = zeros(n_basefuncs, n_basefuncs)

    assembler = start_assemble(M)

    for cell in CellIterator(dh)

        fill!(Me, 0)

        reinit!(cellvalues, cell)

        for q_point in 1:getnquadpoints(cellvalues)
            dΩ = getdetJdV(cellvalues, q_point)

            for i in 1:n_basefuncs
                v = shape_value(cellvalues, q_point, i)
                for j in 1:n_basefuncs
                    u = shape_value(cellvalues, q_point, j)
                    Me[i, j] += (v * u) * dΩ
                end
            end
        end

        assemble!(assembler, celldofs(cell), Me)
    end
    return M
end

K, f = doassemble_K!(K, f, cellvalues, dh)
M = doassemble_M!(M, cellvalues, dh)
A = (Δt .* K) + M;

rhsdata = get_rhs_data(ch, A);

uₙ = zeros(length(f));
apply_analytical!(uₙ, dh, :u, x -> (x[1]^2 - 1) * (x[2]^2 - 1) * max_temp);

apply!(A, ch);

pvd = paraview_collection("transient-heat.pvd");
t = 0
vtk_grid("transient-heat-$t", dh) do vtk
    vtk_point_data(vtk, dh, uₙ)
    vtk_save(vtk)
    pvd[t] = vtk
end

for t in Δt:Δt:T
    #First of all, we need to update the Dirichlet boundary condition values.
    update!(ch, t)

    #Secondly, we compute the right-hand-side of the problem.
    b = Δt .* f .+ M * uₙ
    #Then, we can apply the boundary conditions of the current time step.
    apply_rhs!(rhsdata, b, ch)

    #Finally, we can solve the time step and save the solution afterwards.
    u = A \ b

    vtk_grid("transient-heat-$t", dh) do vtk
        vtk_point_data(vtk, dh, u)
        vtk_save(vtk)
        pvd[t] = vtk
    end
    #At the end of the time loop, we set the previous solution to the current one and go to the next time step.
    uₙ .= u
end

vtk_save(pvd);

# This file was generated using Literate.jl, https://github.com/fredrikekre/Literate.jl

