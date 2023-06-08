using Ferrite
using BlockArrays, SparseArrays, LinearAlgebra
using Tensors

function create_cook_grid(nx, ny)
    corners = [Vec{2}((0.0, 0.0)),
        Vec{2}((1.0, 0.8)), # 48.0, 44.0
        Vec{2}((1.0, 1.0)), # 48.0, 60.0
        Vec{2}((0.0, 0.8))] # 0.0, 44.0
    grid = generate_grid(Triangle, (nx, ny), corners)
    # facesets for boundary conditions
    addfaceset!(grid, "clamped", x -> norm(x[1]) ≈ 0.0)
    addfaceset!(grid, "traction", x -> norm(x[1]) ≈ 1.0)
    return grid
end;

function create_values(interpolation_u, interpolation_p)
    # quadrature rules
    qr = QuadratureRule{2,RefTetrahedron}(3)
    face_qr = QuadratureRule{1,RefTetrahedron}(3)

    # geometric interpolation
    interpolation_geom = Lagrange{2,RefTetrahedron,1}()

    # cell and facevalues for u
    cellvalues_u = CellVectorValues(qr, interpolation_u, interpolation_geom)
    facevalues_u = FaceVectorValues(face_qr, interpolation_u, interpolation_geom)

    # cellvalues for p
    cellvalues_p = CellScalarValues(qr, interpolation_p, interpolation_geom)

    return cellvalues_u, cellvalues_p, facevalues_u, qr
end;

function create_dofhandler(grid, ipu, ipp)
    dh = DofHandler(grid)
    add!(dh, :u, 2, ipu) # displacement
    add!(dh, :p, 1, ipp) # pressure
    close!(dh)
    return dh
end;

function create_bc(dh)
    dbc = ConstraintHandler(dh)
    add!(dbc, Dirichlet(:u, getfaceset(dh.grid, "clamped"), (x, t) -> zero(Vec{2}), [1, 2]))
    close!(dbc)
    t = 0.0
    update!(dbc, t)
    return dbc
end;

struct LinearElasticity{T}
    G::T
    K::T
end

function doassemble(cellvalues_u::CellVectorValues{dim}, cellvalues_p::CellScalarValues{dim},
    facevalues_u::FaceVectorValues{dim}, K::SparseMatrixCSC, grid::Grid,
    dh::DofHandler, mp::LinearElasticity) where {dim}

    f = zeros(ndofs(dh))
    assembler = start_assemble(K, f)
    nu = getnbasefunctions(cellvalues_u)
    np = getnbasefunctions(cellvalues_p)

    fe = PseudoBlockArray(zeros(nu + np), [nu, np]) # local force vector
    ke = PseudoBlockArray(zeros(nu + np, nu + np), [nu, np], [nu, np]) # local stiffness matrix

    # traction vector
    t = Vec{2}((0.0, 0.1))
    # cache ɛdev outside the element routine to avoid some unnecessary allocations
    ɛdev = [zero(SymmetricTensor{2,dim}) for i in 1:getnbasefunctions(cellvalues_u)]

    for cell in CellIterator(dh)
        fill!(ke, 0)
        fill!(fe, 0)
        assemble_up!(ke, fe, cell, cellvalues_u, cellvalues_p, facevalues_u, grid, mp, ɛdev, t)
        assemble!(assembler, celldofs(cell), fe, ke)
    end

    return K, f
end;

function assemble_up!(Ke, fe, cell, cellvalues_u, cellvalues_p, facevalues_u, grid, mp, ɛdev, t)

    n_basefuncs_u = getnbasefunctions(cellvalues_u)
    n_basefuncs_p = getnbasefunctions(cellvalues_p)
    u▄, p▄ = 1, 2
    reinit!(cellvalues_u, cell)
    reinit!(cellvalues_p, cell)

    # We only assemble lower half triangle of the stiffness matrix and then symmetrize it.
    for q_point in 1:getnquadpoints(cellvalues_u)
        for i in 1:n_basefuncs_u
            ɛdev[i] = dev(symmetric(shape_gradient(cellvalues_u, q_point, i)))
        end
        dΩ = getdetJdV(cellvalues_u, q_point)
        for i in 1:n_basefuncs_u
            divδu = shape_divergence(cellvalues_u, q_point, i)
            δu = shape_value(cellvalues_u, q_point, i)
            for j in 1:i
                Ke[BlockIndex((u▄, u▄), (i, j))] += 2 * mp.G * ɛdev[i] ⊡ ɛdev[j] * dΩ
            end
        end

        for i in 1:n_basefuncs_p
            δp = shape_value(cellvalues_p, q_point, i)
            for j in 1:n_basefuncs_u
                divδu = shape_divergence(cellvalues_u, q_point, j)
                Ke[BlockIndex((p▄, u▄), (i, j))] += -δp * divδu * dΩ
            end
            for j in 1:i
                p = shape_value(cellvalues_p, q_point, j)
                Ke[BlockIndex((p▄, p▄), (i, j))] += -1 / mp.K * δp * p * dΩ
            end

        end
    end

    symmetrize_lower!(Ke)

    # We integrate the Neumann boundary using the facevalues.
    # We loop over all the faces in the cell, then check if the face
    # is in our `"traction"` faceset.
    for face in 1:nfaces(cell)
        if onboundary(cell, face) && (cellid(cell), face) ∈ getfaceset(grid, "traction")
            reinit!(facevalues_u, cell, face)
            for q_point in 1:getnquadpoints(facevalues_u)
                dΓ = getdetJdV(facevalues_u, q_point)
                for i in 1:n_basefuncs_u
                    δu = shape_value(facevalues_u, q_point, i)
                    fe[i] += (δu ⋅ t) * dΓ
                end
            end
        end
    end
end

function symmetrize_lower!(K)
    for i in 1:size(K, 1)
        for j in i+1:size(K, 1)
            K[i, j] = K[j, i]
        end
    end
end;

# Create a function to generate the 2nd order identity tensor
function create_identity_tensor(N)
    matrix = Matrix{Float64}(I, N, N)
    tensor = SymmetricTensor{2,3}(matrix)
    return tensor
end

function compute_stresses(cellvalues_u::CellVectorValues{dim,T}, cellvalues_p::CellScalarValues{dim,T}, dh::DofHandler, a) where {dim,T}

    u_range = dof_range(dh, :u) # local range of dofs corresponding to u
    p_range = dof_range(dh, :p) # local range of dofs corresponding to p
    n = ndofs_per_cell(dh)
    cell_dofs = zeros(Int, n)
    nqp = getnquadpoints(cellvalues_u)
    # material
    E = 1.0 # MPa = N/m2
    v = 0.333
    G = E / 2(1 + v)


    # Define the size of the 2D identity tensor
    N = 3

    # Call the function to create the 2nd order identity tensor
    I = create_identity_tensor(N)

    # Allocate storage for the fluxes to store
    σ = [SymmetricTensor{2,3,T,6}[] for _ in 1:getncells(dh.grid)]

    for (cell_num, cell) in enumerate(CellIterator(dh))
        celldofs!(cell_dofs, dh, cell_num) # extract dofs for this element
        ae = a[cell_dofs] # solution vector for this element
        for qp in 1:nqp

            epsilon = function_symmetric_gradient(cellvalues_u, qp, ae, u_range) # symmetric gradient, note passing the range of u dofs
            ϵ_tensor = SymmetricTensor{2,3}((i, j) -> begin
                if i <= 2 && j <= 2
                    epsilon[i, j]
                else
                    return 0
                end

            end)
            p = function_value(cellvalues_p, qp, ae, p_range) # pressure, passing the range of p_dofs

            push!(σ[cell_num], 2G * dev(ϵ_tensor) - I * p)
            # Should it not be this? σ = λ * div(u) * I + 2 * μ * ε(u) - p * I
        end
    end
    return σ
end

function solve(ν, interpolation_u, interpolation_p)
    # material
    Emod = 1.0 # MPa = N/m2
    Gmod = Emod / 2(1 + ν)
    Kmod = Emod * ν / ((1 + ν) * (1 - 2ν))




    mp = LinearElasticity(Gmod, Kmod)

    # grid, dofhandler, boundary condition
    n = 50
    grid = create_cook_grid(n, n)
    dh = create_dofhandler(grid, interpolation_u, interpolation_p)
    dbc = create_bc(dh)

    # cellvalues
    cellvalues_u, cellvalues_p, facevalues_u, qr = create_values(interpolation_u, interpolation_p)

    # assembly and solve
    K = create_sparsity_pattern(dh)
    K, f = doassemble(cellvalues_u, cellvalues_p, facevalues_u, K, grid, dh, mp)
    println("What is sum of f", sum(f))
    println("--------------")
    apply!(K, f, dbc)
    u = Symmetric(K) \ f

    σ = compute_stresses(cellvalues_u, cellvalues_p, dh, u)


    projector = L2Projector(interpolation_p, grid)
    σ_projected = project(projector, σ, qr; project_to_nodes=false)

    # export
    filename = "DISP_STRESS_COOKS_" * (isa(interpolation_u, Lagrange{2,RefTetrahedron,1}) ? "linear" : "quadratic") *
               "_linear"

    folder = raw"C:\Users\gagan\OneDrive\Desktop\Master Thesis"
    full_path = joinpath(folder, filename)
    vtk_grid(full_path, dh) do vtkfile
        vtk_point_data(vtkfile, dh, u)
        vtk_point_data(vtkfile, projector, σ_projected, "sigma")
    end
    return u
end

linear = Lagrange{2,RefTetrahedron,1}()
quadratic = Lagrange{2,RefTetrahedron,2}()

u1 = solve(0.333, linear, linear)
u2 = solve(0.333, quadratic, linear)






