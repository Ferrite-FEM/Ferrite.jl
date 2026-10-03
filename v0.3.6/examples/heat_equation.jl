using Ferrite, SparseArrays

grid = generate_grid(Quadrilateral, (20, 20));

dim = 2
ip = Lagrange{dim, RefCube, 1}()
qr = QuadratureRule{dim, RefCube}(2)
cellvalues = CellScalarValues(qr, ip);

dh = DofHandler(grid)
push!(dh, :u, 1)
close!(dh);

K = create_sparsity_pattern(dh)

ch = ConstraintHandler(dh);

∂Ω = union(getfaceset.((grid, ), ["left", "right", "top", "bottom"])...);

dbc = Dirichlet(:u, ∂Ω, (x, t) -> 0)
add!(ch, dbc);

close!(ch)
update!(ch, 0.0);

function doassemble(cellvalues::CellScalarValues{dim}, K::SparseMatrixCSC, dh::DofHandler) where {dim}

    n_basefuncs = getnbasefunctions(cellvalues)
    Ke = zeros(n_basefuncs, n_basefuncs)
    fe = zeros(n_basefuncs)

    f = zeros(ndofs(dh))
    assembler = start_assemble(K, f)

    for cell in CellIterator(dh)

        fill!(Ke, 0)
        fill!(fe, 0)

        reinit!(cellvalues, cell)

        for q_point in 1:getnquadpoints(cellvalues)
            dΩ = getdetJdV(cellvalues, q_point)

            for i in 1:n_basefuncs
                δu  = shape_value(cellvalues, q_point, i)
                ∇δu = shape_gradient(cellvalues, q_point, i)
                fe[i] += δu * dΩ
                for j in 1:n_basefuncs
                    ∇u = shape_gradient(cellvalues, q_point, j)
                    Ke[i, j] += (∇δu ⋅ ∇u) * dΩ
                end
            end
        end

        assemble!(assembler, celldofs(cell), fe, Ke)
    end
    return K, f
end

K, f = doassemble(cellvalues, K, dh);

apply!(K, f, ch)
u = K \ f;

vtk_grid("heat_equation", dh) do vtk
    vtk_point_data(vtk, dh, u)
end

# This file was generated using Literate.jl, https://github.com/fredrikekre/Literate.jl

