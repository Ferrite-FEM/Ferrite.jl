using Ferrite, Tensors, ProgressMeter
using BlockArrays, SparseArrays, LinearAlgebra

struct NeoHooke
    μ::Float64
    λ::Float64
end

function importTestGrid()
    grid = generate_grid(Tetrahedron, (5, 5, 5), zero(Vec{3}), ones(Vec{3}));
    addfaceset!(grid, "myBottom", x -> norm(x[2]) ≈ 0.0);
    addfaceset!(grid, "myBack", x -> norm(x[3]) ≈ 0.0);
    addfaceset!(grid, "myRight", x -> norm(x[1]) ≈ 1.0);
    addfaceset!(grid, "myLeft", x -> norm(x[1]) ≈ 0.0);
    return grid
end;

function create_values(interpolation_u, interpolation_p)
    # quadrature rules
    qr      = QuadratureRule{3,RefTetrahedron}(4)
    face_qr = QuadratureRule{2,RefTetrahedron}(4)

    # geometric interpolation
    interpolation_geom = Lagrange{3,RefTetrahedron,1}()

    # cell and facevalues for u
    cellvalues_u = CellVectorValues(qr, interpolation_u, interpolation_geom)
    facevalues_u = FaceVectorValues(face_qr, interpolation_u, interpolation_geom)

    # cellvalues for p
    cellvalues_p = CellScalarValues(qr, interpolation_p, interpolation_geom)

    return cellvalues_u, cellvalues_p, facevalues_u
end;

function Ψ(F, p, mp::NeoHooke)
    μ = mp.μ
    λ = mp.λ
    Ic = tr(tdot(F))
    J = det(F)
    Js = (λ + p + sqrt((λ + p)^2. + 4. * λ * μ ))/(2. * λ)
    return p * (Js - J) + μ / 2 * (Ic - 3) - μ * log(Js) + λ / 2 * (Js - 1)^2
end;

function constitutive_driver(F, p, mp::NeoHooke)
    # Compute all derivatives in one function call
    ∂²Ψ∂F², ∂Ψ∂F = Tensors.hessian(y -> Ψ(y, p, mp), F, :all)
    ∂²Ψ∂p², ∂Ψ∂p = Tensors.hessian(y -> Ψ(F, y, mp), p, :all)
    ∂²Ψ∂F∂p = Tensors.gradient(q -> Tensors.gradient(y -> Ψ(y, q, mp), F), p)
    return ∂Ψ∂F, ∂²Ψ∂F², ∂Ψ∂p, ∂²Ψ∂p², ∂²Ψ∂F∂p
end;

function create_dofhandler(grid, ipu, ipp)
    dh = DofHandler(grid)
    push!(dh, :u, 3, ipu) # displacement dim = 3
    push!(dh, :p, 1, ipp) # pressure dim = 1
    close!(dh)
    return dh
end;

function create_bc(dh)
    dbc = ConstraintHandler(dh)
    add!(dbc, Dirichlet(:u, getfaceset(dh.grid, "myLeft"), (x,t) -> zero(Vec{1}), [1]))
    add!(dbc, Dirichlet(:u, getfaceset(dh.grid, "myBottom"), (x,t) -> zero(Vec{1}), [2]))
    add!(dbc, Dirichlet(:u, getfaceset(dh.grid, "myBack"), (x,t) -> zero(Vec{1}), [3]))
    add!(dbc, Dirichlet(:u, getfaceset(dh.grid, "myRight"), (x,t) -> t*ones(Vec{1}), [1]))
    close!(dbc)
    Ferrite.update!(dbc, 0.0)
    return dbc
end;

function calculate_element_volume(cell, cellvalues_u, ue)
    reinit!(cellvalues_u, cell)
    evol::Float64=0.0;
    @inbounds for qp in 1:getnquadpoints(cellvalues_u)
        dΩ = getdetJdV(cellvalues_u, qp)
        ∇u = function_gradient(cellvalues_u, qp, ue)
        F = one(∇u) + ∇u
        J = det(F)
        evol += J * dΩ
    end
    return evol
end;

function calculate_volume_deformed_mesh(w, dh::DofHandler, cellvalues_u)
    evol::Float64 = 0.0;
    @inbounds for cell in CellIterator(dh)
        global_dofs = celldofs(cell)
        nu = getnbasefunctions(cellvalues_u)
        global_dofs_u = global_dofs[1:nu]
        ue = w[global_dofs_u]
        δevol = calculate_element_volume(cell, cellvalues_u, ue)
        evol += δevol;
    end
    return evol
end;

function assemble_element!(Ke, fe, cell, cellvalues_u, cellvalues_p, mp, ue, pe)
    # Reinitialize cell values, and reset output arrays
    ublock, pblock = 1, 2
    reinit!(cellvalues_u, cell)
    reinit!(cellvalues_p, cell)
    fill!(Ke, 0.0)
    fill!(fe, 0.0)

    n_basefuncs_u = getnbasefunctions(cellvalues_u)
    n_basefuncs_p = getnbasefunctions(cellvalues_p)

    @inbounds for qp in 1:getnquadpoints(cellvalues_u)
        dΩ = getdetJdV(cellvalues_u, qp)
        # Compute deformation gradient F
        ∇u = function_gradient(cellvalues_u, qp, ue)
        p = function_value(cellvalues_p, qp, pe)
        F = one(∇u) + ∇u

        # Compute first Piola-Kirchhoff stress and tangent modulus
        ∂Ψ∂F, ∂²Ψ∂F², ∂Ψ∂p, ∂²Ψ∂p², ∂²Ψ∂F∂p = constitutive_driver(F, p, mp)

        # Loop over the `u`-test functions to calculate the `u`-`u` and `u`-`p` blocks
        for i in 1:n_basefuncs_u
            # gradient of the test function
            ∇δui = shape_gradient(cellvalues_u, qp, i)
            # Add contribution to the residual from this test function
            fe[BlockIndex((ublock), (i))] += ( ∇δui ⊡ ∂Ψ∂F) * dΩ

            ∇δui∂S∂F = ∇δui ⊡ ∂²Ψ∂F²
            for j in 1:n_basefuncs_u
                ∇δuj = shape_gradient(cellvalues_u, qp, j)

                # Add contribution to the tangent
                Ke[BlockIndex((ublock, ublock), (i, j))] += ( ∇δui∂S∂F ⊡ ∇δuj ) * dΩ
            end
            # Loop over the `p`-test functions
            for j in 1:n_basefuncs_p
                δp = shape_value(cellvalues_p, qp, j)
                # Add contribution to the tangent
                Ke[BlockIndex((ublock, pblock), (i, j))] += ( ∂²Ψ∂F∂p ⊡ ∇δui ) * δp * dΩ
            end
        end
        # Loop over the `p`-test functions to calculate the `p-`u` and `p`-`p` blocks
        for i in 1:n_basefuncs_p
            δp = shape_value(cellvalues_p, qp, i)
            fe[BlockIndex((pblock), (i))] += ( δp * ∂Ψ∂p) * dΩ

            for j in 1:n_basefuncs_u
                ∇δuj = shape_gradient(cellvalues_u, qp, j)
                Ke[BlockIndex((pblock, ublock), (i, j))] += ∇δuj ⊡ ∂²Ψ∂F∂p * δp * dΩ
            end
            for j in 1:n_basefuncs_p
                δp = shape_value(cellvalues_p, qp, j)
                Ke[BlockIndex((pblock, pblock), (i, j))] += δp * ∂²Ψ∂p² * δp * dΩ
            end
        end
    end
end;

function assemble_global!(K::SparseMatrixCSC, f, cellvalues_u::CellVectorValues{dim},
                         cellvalues_p::CellScalarValues{dim}, dh::DofHandler, mp::NeoHooke, w) where {dim}
    nu = getnbasefunctions(cellvalues_u)
    np = getnbasefunctions(cellvalues_p)

    # start_assemble resets K and f
    fe = PseudoBlockArray(zeros(nu + np), [nu, np]) # local force vector
    ke = PseudoBlockArray(zeros(nu + np, nu + np), [nu, np], [nu, np]) # local stiffness matrix

    assembler = start_assemble(K, f)
    # Loop over all cells in the grid
    for cell in CellIterator(dh)
        global_dofs = celldofs(cell)
        global_dofsu = global_dofs[1:nu]; # first nu dofs are displacement
        global_dofsp = global_dofs[nu + 1:end]; # last np dofs are pressure
        @assert size(global_dofs, 1) == nu + np # sanity check
        ue = w[global_dofsu] # displacement dofs for the current cell
        pe = w[global_dofsp] # pressure dofs for the current cell
        assemble_element!(ke, fe, cell, cellvalues_u, cellvalues_p, mp, ue, pe)
        assemble!(assembler, global_dofs, fe, ke)
    end
end;

function solve(interpolation_u, interpolation_p)

    # import the mesh
    grid = importTestGrid()

    # Material parameters
    μ = 1.
    λ = 1.E4 * μ
    mp = NeoHooke(μ, λ)

    # Create the DofHandler and CellValues
    dh = create_dofhandler(grid, interpolation_u, interpolation_p)
    cellvalues_u, cellvalues_p, facevalues_u = create_values(interpolation_u, interpolation_p)

    # Create the DirichletBCs
    dbc = create_bc(dh)

    # Pre-allocation of vectors for the solution and Newton increments
    _ndofs = ndofs(dh)
    w  = zeros(_ndofs)
    ΔΔw = zeros(_ndofs)
    apply!(w, dbc)

    # Create the sparse matrix and residual vector
    K = create_sparsity_pattern(dh)
    f = zeros(_ndofs)

    # We run the simulation parameterized by a time like parameter. `Tf` denotes the final value
    # of this parameter, and Δt denotes its increment in each step
    Tf = 2.0;
    Δt = 0.1;
    NEWTON_TOL = 1e-8

    pvd = paraview_collection("hyperelasticity_incomp_mixed.pvd");
    for t ∈ 0.0:Δt:Tf
        # Perform Newton iterations
        Ferrite.update!(dbc, t)
        apply!(w, dbc)
        newton_itr = -1
        prog = ProgressMeter.ProgressThresh(NEWTON_TOL, "Solving @ time $t of $Tf;")
        fill!(ΔΔw, 0.0);
        while true; newton_itr += 1
            assemble_global!(K, f, cellvalues_u, cellvalues_p, dh, mp, w)
            norm_res = norm(f[Ferrite.free_dofs(dbc)])
            apply_zero!(K, f, dbc)
            # Only display output at specific load steps
            if t%(5*Δt) == 0
                ProgressMeter.update!(prog, norm_res; showvalues = [(:iter, newton_itr)])
            end
            if norm_res < NEWTON_TOL
                break
            elseif newton_itr > 30
                error("Reached maximum Newton iterations, aborting")
            end
            # Compute the incremental `dof`-vector (both displacement and pressure)
            ΔΔw .= K\f;

            apply_zero!(ΔΔw, dbc)
            w .-= ΔΔw
        end;

        # Save the solution fields
        vtk_grid("hyperelasticity_incomp_mixed_$t.vtu", dh) do vtkfile
            vtk_point_data(vtkfile, dh, w)
            vtk_save(vtkfile)
            pvd[t] = vtkfile
        end
    end;
    vtk_save(pvd);
    vol_def = calculate_volume_deformed_mesh(w, dh, cellvalues_u);
    print("Deformed volume is $vol_def")
    return vol_def;
end;

quadratic = Lagrange{3, RefTetrahedron, 2}()
linear = Lagrange{3, RefTetrahedron, 1}()
vol_def = solve(quadratic, linear)

# This file was generated using Literate.jl, https://github.com/fredrikekre/Literate.jl

