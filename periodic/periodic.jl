using Ferrite, SparseArrays

grid = generate_grid(Quadrilateral, (30, 30));

# 14 - 13 - 15 - 16
#  |    |    |    |
# 10 -- 9 - 11 - 12
#  |    |    |    |
#  4 -- 3 -- 6 -- 8
#  |    |    |    |
#  1 -- 2 -- 5 -- 7

dim = 2
ip = Lagrange{dim, RefCube, 1}()
qr = QuadratureRule{dim, RefCube}(2)
cellvalues = CellScalarValues(qr, ip);

dh = DofHandler(grid)
push!(dh, :u, 1)
close!(dh);

ch = ConstraintHandler(dh);

spdbc = Ferrite.StronglyPeriodicDirichlet(:u, [1], ["left" => "right", "bottom" => "top"])
add!(ch, spdbc)
close!(ch)
update!(ch, 0.0)

K = create_sparsity_pattern(dh, ch)

function doassemble(cellvalues::CellScalarValues{dim}, K::SparseMatrixCSC, dh::DofHandler) where {dim}

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
                fe[i] += v * dΩ
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

K, f = doassemble(cellvalues, K, dh);

apply!(K, f, ch)
u = K \ f;
apply!(u, ch)

vtk_grid("heat_equation", dh) do vtk
    vtk_point_data(vtk, dh, u)
end

