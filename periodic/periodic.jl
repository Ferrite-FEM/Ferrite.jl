using Ferrite, SparseArrays, FerriteGmsh

grid = saved_file_to_grid("/Users/fredrik/dev/Ferrite/docs/src/literate/periodic.msh")
union!(empty!(getcellset(grid, "inclusions")), setdiff(1:getncells(grid), getcellset(grid, "matrix")))

# Add the boundaries
eps = 1e-5
addfaceset!(grid, "left", x -> -0.5-eps < x[1] < -0.5+eps)
addfaceset!(grid, "right", x -> 0.5-eps < x[1] < 0.5+eps)
addfaceset!(grid, "bottom", x -> -0.5-eps < x[2] < -0.5+eps)
addfaceset!(grid, "top", x -> 0.5-eps < x[2] < 0.5+eps)

# 14 - 13 - 15 - 16
#  |    |    |    |
# 10 -- 9 - 11 - 12
#  |    |    |    |
#  4 -- 3 -- 6 -- 8
#  |    |    |    |
#  1 -- 2 -- 5 -- 7

dim = 2
ip = Lagrange{dim, RefTetrahedron, 1}()
qr = QuadratureRule{dim, RefTetrahedron}(2)
cellvalues = CellVectorValues(qr, ip);

dh = DofHandler(grid)
push!(dh, :uD, 2)
push!(dh, :uP, 2)
close!(dh);

ch = ConstraintHandler(dh);

εᴹ = SymmetricTensor{2,2}((0.0, 1.0, 0.0)) # 12 loading

dbc = Dirichlet(:uD, union(getfaceset.(Ref(grid), ["left", "right", "top", "bottom"])...), (x, t) ->  εᴹ ⋅ x, [1, 2])
pdbc = PeriodicDirichlet(:uP, ["left" => "right", "bottom" => "top"], (x, _) -> εᴹ ⋅ x, [1, 2])
# pdbc = PeriodicDirichlet(:u, ["left" => "right", "bottom" => "top"], (x, _) -> εᴹ ⋅ x, [1, 2])
add!(ch, dbc)
add!(ch, pdbc)
close!(ch)
update!(ch, 0.0)

K = create_sparsity_pattern(dh, ch)

function doassemble(cellvalues::CellVectorValues{dim}, K::SparseMatrixCSC, dh::DofHandler) where {dim}

    n_basefuncs = getnbasefunctions(cellvalues)
    ndpc = ndofs_per_cell(dh)
    Ke = zeros(ndpc, ndpc)
    fe = zeros(ndpc)

    f = zeros(ndofs(dh))
    assembler = start_assemble(K, f)

    E = 200e9
    ν = 0.3
    λ = E*ν / ((1 + ν) * (1 - 2ν))
    μ = E / (2(1 + ν))
    δ(i,j) = i == j ? 1.0 : 0.0
    fn = (i,j,k,l) -> λ*δ(i,j)*δ(k,l) + μ*(δ(i,k)*δ(j,l) + δ(i,l)*δ(j,k))

    C = SymmetricTensor{4, dim}(fn)
    εᴹ = SymmetricTensor{2,2}((0.0, 1.0, 0.0)) # 12 loading

    @inbounds for cell in CellIterator(dh)
        scaling = cellid(cell) in getcellset(dh.grid, "inclusions") ? 10 : 1
        EE = scaling * C

        fill!(Ke, 0)
        fill!(fe, 0)

        uD_range = dof_range(dh, :uD)
        uP_range = dof_range(dh, :uP)

        reinit!(cellvalues, cell)

        for q_point in 1:getnquadpoints(cellvalues)
            dΩ = getdetJdV(cellvalues, q_point)

            σᴹ = EE ⊡ εᴹ

            for i in 1:n_basefuncs
                # δu  = shape_value(cellvalues, q_point, i)
                δεi = shape_symmetric_gradient(cellvalues, q_point, i)
                fe[uD_range[i]] += ( - δεi ⊡ σᴹ) * dΩ

                for j in 1:n_basefuncs
                    δεj = shape_gradient(cellvalues, q_point, j)
                    Ke[uD_range[i], uD_range[j]] += (δεi ⊡ EE ⊡ δεj) * dΩ
                end
            end
        end

        Ke[uP_range, uP_range] .= Ke[uD_range, uD_range]
        fe[uP_range] .= fe[uD_range]

        assemble!(assembler, celldofs(cell), fe, Ke)
    end
    return K, f
end

K, f = doassemble(cellvalues, K, dh);

apply!(K, f, ch)
u = K \ f;
apply!(u, ch)


vtk_grid("periodic", dh) do vtk
    vtk_point_data(vtk, dh, u)
end

