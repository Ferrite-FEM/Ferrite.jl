using JuAFEM, SparseArrays

grid = generate_grid(Quadrilateral, (20, 20));

dim = 2
ip = Lagrange{dim, RefCube, 1}()
qr = QuadratureRule{dim, RefCube}(2)
cellvalues = CellScalarValues(qr, ip);

dh = DofHandler(grid)
push!(dh, :u, 1)
close!(dh);

max_temp = 100
Δt = 1
T = 200

K = create_sparsity_pattern(dh);
M = create_sparsity_pattern(dh);

ch = ConstraintHandler(dh);

∂Ω = union(getfaceset.((grid, ), ["left", "right"])...);
dbc = Dirichlet(:u, ∂Ω, (x, t) -> 0)
add!(ch, dbc);

∂Ω = union(getfaceset.((grid, ), ["top", "bottom"])...);
dbc = Dirichlet(:u, ∂Ω, (x, t) -> t*(max_temp/T))
add!(ch, dbc);

close!(ch)
update!(ch, 0.0);

function doassemble!(cellvalues::CellScalarValues{dim}, K::SparseMatrixCSC, dh::DofHandler, Δt::Real) where {dim}

    n_basefuncs = getnbasefunctions(cellvalues)
    Ke = zeros(n_basefuncs, n_basefuncs)
    fe = zeros(n_basefuncs)

    f = zeros(ndofs(dh))
    assembler = start_assemble(K, f)

    @inbounds for cell in CellIterator(dh)

        fill!(Ke, 0)
        fill!(fe, 0)

        reinit!(cellvalues, cell)

        for q_point in 1:getnquadpoints(cellvalues)
            dΩ = getdetJdV(cellvalues, q_point)

            for i in 1:n_basefuncs
                v  = shape_value(cellvalues, q_point, i)
                ∇v = shape_gradient(cellvalues, q_point, i)
                fe[i] += 0*v * dΩ
                for j in 1:n_basefuncs
                    ∇u = shape_gradient(cellvalues, q_point, j)
                    Ke[i, j] += (∇v ⋅ ∇u) * dΩ
                end
            end
        end

        assemble!(assembler, celldofs(cell), fe, Ke)
    end
    return K, f
end

function doassemble!(cellvalues::CellScalarValues{dim}, M::SparseMatrixCSC, dh::DofHandler) where {dim}

    n_basefuncs = getnbasefunctions(cellvalues)
    Me = zeros(n_basefuncs, n_basefuncs)

    assembler = start_assemble(M)

    @inbounds for cell in CellIterator(dh)

        fill!(Me, 0)

        reinit!(cellvalues, cell)

        for q_point in 1:getnquadpoints(cellvalues)
            dΩ = getdetJdV(cellvalues, q_point)

            for i in 1:n_basefuncs
                v  = shape_value(cellvalues, q_point, i)
                for j in 1:n_basefuncs
                    u = shape_value(cellvalues, q_point, j)
                    Me[i, j] += (v ⋅ u) * dΩ
                end
            end
        end

        assemble!(assembler, celldofs(cell), Me)
    end
    return M
end

K, f = doassemble!(cellvalues, K, dh, Δt);
M = doassemble!(cellvalues, M, dh);
A = (Δt .* K) + M
rhsdata = get_rhs_data(ch, A)
uₙ = zeros(length(f))

apply!(A, ch)

pvd = paraview_collection("transient-heat.pvd")

for t in 0:Δt:T
    update!(ch, t)

    b = Δt .* f .+ M * uₙ
    apply_rhs!(rhsdata, b, ch)
    
    u = A \ b;

    vtk_grid("transient-heat-$t", dh) do vtk
        vtk_point_data(vtk, dh, u)
        vtk_save(vtk)
        pvd[t] = vtk
    end
    
   uₙ .= u
 
end

vtk_save(pvd)

# This file was generated using Literate.jl, https://github.com/fredrikekre/Literate.jl

