using Ferrite, SparseArrays, WriteVTK

grid = generate_grid(Quadrilateral, (100, 100));

ip = Lagrange{RefQuadrilateral, 1}()
qr = QuadratureRule{RefQuadrilateral}(2)
cellvalues = CellValues(qr, ip);

dh = DofHandler(grid)
add!(dh, :u, ip)
close!(dh);

K = allocate_matrix(dh);
M = allocate_matrix(dh);

f = zeros(ndofs(dh));

max_temp = 100
Δt = 1
T = 200
t_rise = 100
ch = ConstraintHandler(dh);

∂Ω₁ = union(getfacetset.((grid,), ["left", "right"])...)
dbc = Dirichlet(:u, ∂Ω₁, (x, t) -> 0)
add!(ch, dbc);

∂Ω₂ = union(getfacetset.((grid,), ["top", "bottom"])...)
dbc = Dirichlet(:u, ∂Ω₂, (x, t) -> max_temp * clamp(t / t_rise, 0, 1))
add!(ch, dbc)
close!(ch)
update!(ch, 0.0);

function doassemble_K!(K::SparseMatrixCSC, f::Vector, cellvalues::CellValues, dh::DofHandler)

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

function doassemble_M!(M::SparseMatrixCSC, cellvalues::CellValues, dh::DofHandler)

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

pvd = paraview_collection("transient-heat")
VTKGridFile("transient-heat-0", dh) do vtk
    write_solution(vtk, dh, uₙ)
    pvd[0.0] = vtk
end

for (step, t) in enumerate(Δt:Δt:T)
    #First of all, we need to update the Dirichlet boundary condition values.
    update!(ch, t)

    #Secondly, we compute the right-hand-side of the problem.
    b = Δt .* f .+ M * uₙ
    #Then, we can apply the boundary conditions of the current time step.
    apply_rhs!(rhsdata, b, ch)

    #Finally, we can solve the time step and save the solution afterwards.
    u = A \ b

    VTKGridFile("transient-heat-$step", dh) do vtk
        write_solution(vtk, dh, u)
        pvd[t] = vtk
    end
    #At the end of the time loop, we set the previous solution to the current one and go to the next time step.
    uₙ .= u
end

vtk_save(pvd);

# This file was generated using Literate.jl, https://github.com/fredrikekre/Literate.jl
