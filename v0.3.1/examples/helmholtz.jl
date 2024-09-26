using Ferrite
using Tensors
using SparseArrays
using LinearAlgebra

const ∇ = Tensors.gradient
const Δ = Tensors.hessian;

grid = generate_grid(Quadrilateral, (150, 150))

dim = 2
ip = Lagrange{dim, RefCube, 1}()
qr = QuadratureRule{dim, RefCube}(2)
qr_face = QuadratureRule{dim-1, RefCube}(2)
cellvalues = CellScalarValues(qr, ip);
facevalues = FaceScalarValues(qr_face, ip);

dh = DofHandler(grid)
push!(dh, :u, 1)
close!(dh)

function u_ana(x::Vec{2, T}) where {T}
    xs = (Vec{2}((-0.5,  0.5)),
          Vec{2}((-0.5, -0.5)),
          Vec{2}(( 0.5,  -0.5)))
    σ = 1/8
    s = zero(eltype(x))
    for i in 1:3
        s += exp(- norm(x - xs[i])^2 / σ^2)
    end
    return max(1e-15 * one(T), s) # Denormals, be gone
end;

dbcs = ConstraintHandler(dh)

dbc = Dirichlet(:u, union(getfaceset(grid, "top"), getfaceset(grid, "right")), (x,t) -> u_ana(x))
add!(dbcs, dbc)
close!(dbcs)
update!(dbcs, 0.0)

K = create_sparsity_pattern(dh);

function doassemble(cellvalues::CellScalarValues{dim}, facevalues::FaceScalarValues{dim},
                         K::SparseMatrixCSC, dh::DofHandler) where {dim}
    b = 1.0
    f = zeros(ndofs(dh))
    assembler = start_assemble(K, f)

    n_basefuncs = getnbasefunctions(cellvalues)
    global_dofs = zeros(Int, ndofs_per_cell(dh))

    fe = zeros(n_basefuncs) # Local force vector
    Ke = zeros(n_basefuncs, n_basefuncs) # Local stiffness mastrix

    @inbounds for (cellcount, cell) in enumerate(CellIterator(dh))
        fill!(Ke, 0)
        fill!(fe, 0)
        coords = getcoordinates(cell)

        reinit!(cellvalues, cell)

        for q_point in 1:getnquadpoints(cellvalues)
            dΩ = getdetJdV(cellvalues, q_point)
            coords_qp = spatial_coordinate(cellvalues, q_point, coords)
            f_true = -LinearAlgebra.tr(hessian(u_ana, coords_qp)) + u_ana(coords_qp)
            for i in 1:n_basefuncs
                δu = shape_value(cellvalues, q_point, i)
                ∇δu = shape_gradient(cellvalues, q_point, i)
                fe[i] += (δu * f_true) * dΩ
                for j in 1:n_basefuncs
                    u = shape_value(cellvalues, q_point, j)
                    ∇u = shape_gradient(cellvalues, q_point, j)
                    Ke[i, j] += (∇δu ⋅ ∇u + δu * u) * dΩ
                end
            end
        end

        for face in 1:nfaces(cell)
            if onboundary(cell, face) &&
                   ((cellcount, face) ∈ getfaceset(grid, "left") ||
                    (cellcount, face) ∈ getfaceset(grid, "bottom"))
                reinit!(facevalues, cell, face)
                for q_point in 1:getnquadpoints(facevalues)
                    coords_qp = spatial_coordinate(facevalues, q_point, coords)
                    n = getnormal(facevalues, q_point)
                    g = gradient(u_ana, coords_qp) ⋅ n
                    dΓ = getdetJdV(facevalues, q_point)
                    for i in 1:n_basefuncs
                        δu = shape_value(facevalues, q_point, i)
                        fe[i] += -(δu * g) * dΓ
                        for j in 1:n_basefuncs
                            ∇u = shape_gradient(cellvalues, q_point, j)
                            Ke[i, j] += (δu * ∇u ⋅ n) * dΓ
                        end
                    end
                end
            end
        end

        celldofs!(global_dofs, cell)
        assemble!(assembler, global_dofs, fe, Ke)
    end
    return K, f
end;

K, f = doassemble(cellvalues, facevalues, K, dh);
apply!(K, f, dbcs)
u = Symmetric(K) \ f;

vtkfile = vtk_grid("helmholtz", dh)
vtk_point_data(vtkfile, dh, u)
vtk_save(vtkfile)
println("Helmholtz successful")

# This file was generated using Literate.jl, https://github.com/fredrikekre/Literate.jl

