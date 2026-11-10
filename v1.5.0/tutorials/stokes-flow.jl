using Ferrite, FerriteGmsh, Gmsh, Tensors, LinearAlgebra, SparseArrays

function setup_grid(h = 0.05)
    # Initialize gmsh
    Gmsh.initialize()
    gmsh.option.set_number("General.Verbosity", 2)

    # Add the points
    o = gmsh.model.geo.add_point(0.0, 0.0, 0.0, h)
    p1 = gmsh.model.geo.add_point(0.5, 0.0, 0.0, h)
    p2 = gmsh.model.geo.add_point(1.0, 0.0, 0.0, h)
    p3 = gmsh.model.geo.add_point(0.0, 1.0, 0.0, h)
    p4 = gmsh.model.geo.add_point(0.0, 0.5, 0.0, h)

    # Add the lines
    l1 = gmsh.model.geo.add_line(p1, p2)
    l2 = gmsh.model.geo.add_circle_arc(p2, o, p3)
    l3 = gmsh.model.geo.add_line(p3, p4)
    l4 = gmsh.model.geo.add_circle_arc(p4, o, p1)

    # Create the closed curve loop and the surface
    loop = gmsh.model.geo.add_curve_loop([l1, l2, l3, l4])
    surf = gmsh.model.geo.add_plane_surface([loop])

    # Synchronize the model
    gmsh.model.geo.synchronize()

    # Create the physical domains
    gmsh.model.add_physical_group(1, [l1], -1, "Γ1")
    gmsh.model.add_physical_group(1, [l2], -1, "Γ2")
    gmsh.model.add_physical_group(1, [l3], -1, "Γ3")
    gmsh.model.add_physical_group(1, [l4], -1, "Γ4")
    gmsh.model.add_physical_group(2, [surf])

    # Add the periodicity constraint using 4x4 affine transformation matrix,
    # see https://en.wikipedia.org/wiki/Transformation_matrix#Affine_transformations
    transformation_matrix = zeros(4, 4)
    transformation_matrix[1, 2] = 1  # -sin(-pi/2)
    transformation_matrix[2, 1] = -1 #  cos(-pi/2)
    transformation_matrix[3, 3] = 1
    transformation_matrix[4, 4] = 1
    transformation_matrix = vec(transformation_matrix')
    gmsh.model.mesh.set_periodic(1, [l1], [l3], transformation_matrix)

    # Generate a 2D mesh
    gmsh.model.mesh.generate(2)

    # Save the mesh, and read back in as a Ferrite Grid
    grid = mktempdir() do dir
        path = joinpath(dir, "mesh.msh")
        gmsh.write(path)
        togrid(path)
    end

    # Finalize the Gmsh library
    Gmsh.finalize()

    return grid
end

function setup_fevalues(ipu, ipp, ipg)
    qr = QuadratureRule{RefTriangle}(2)
    cvu = CellValues(qr, ipu, ipg)
    cvp = CellValues(qr, ipp, ipg; update_gradients = false, update_detJdV = false)
    qr_facet = FacetQuadratureRule{RefTriangle}(2)
    fvp = FacetValues(qr_facet, ipp, ipg; update_gradients = false)
    return cvu, cvp, fvp
end

function setup_dofs(grid, ipu, ipp)
    dh = DofHandler(grid)
    add!(dh, :u, ipu)
    add!(dh, :p, ipp)
    close!(dh)
    return dh
end

function setup_mean_constraint(dh, fvp)
    assembler = Ferrite.COOAssembler()
    # All external boundaries
    set = union(
        getfacetset(dh.grid, "Γ1"),
        getfacetset(dh.grid, "Γ2"),
        getfacetset(dh.grid, "Γ3"),
        getfacetset(dh.grid, "Γ4"),
    )
    # Allocate buffers
    range_p = dof_range(dh, :p)
    element_dofs = zeros(Int, ndofs_per_cell(dh))
    element_dofs_p = view(element_dofs, range_p)
    element_coords = zeros(Vec{2}, 3)
    Ce = zeros(1, length(range_p)) # Local constraint matrix (only 1 row)
    # Loop over all the boundaries
    for (ci, fi) in set
        Ce .= 0
        getcoordinates!(element_coords, dh.grid, ci)
        reinit!(fvp, element_coords, fi)
        celldofs!(element_dofs, dh, ci)
        for qp in 1:getnquadpoints(fvp)
            dΓ = getdetJdV(fvp, qp)
            for i in 1:getnbasefunctions(fvp)
                Ce[1, i] += shape_value(fvp, qp, i) * dΓ
            end
        end
        # Assemble to row 1
        assemble!(assembler, [1], element_dofs_p, Ce)
    end
    C, _ = finish_assemble(assembler)
    # Create an AffineConstraint from the C-matrix
    _, J, V = findnz(C)
    _, constrained_dof_idx = findmax(abs2, V)
    constrained_dof = J[constrained_dof_idx]
    V ./= V[constrained_dof_idx]
    mean_value_constraint = AffineConstraint(
        constrained_dof,
        Pair{Int, Float64}[J[i] => -V[i] for i in 1:length(J) if J[i] != constrained_dof],
        0.0,
    )
    return mean_value_constraint
end

function setup_constraints(dh, fvp)
    ch = ConstraintHandler(dh)
    # Periodic BC
    R = rotation_tensor(π / 2)
    periodic_faces = collect_periodic_facets(dh.grid, "Γ3", "Γ1", x -> R ⋅ x)
    periodic = PeriodicDirichlet(:u, periodic_faces, R, [1, 2])
    add!(ch, periodic)
    # Dirichlet BC
    Γ24 = union(getfacetset(dh.grid, "Γ2"), getfacetset(dh.grid, "Γ4"))
    dbc = Dirichlet(:u, Γ24, (x, t) -> [0, 0], [1, 2])
    add!(ch, dbc)
    # Compute mean value constraint and add it
    mean_value_constraint = setup_mean_constraint(dh, fvp)
    add!(ch, mean_value_constraint)
    # Finalize
    close!(ch)
    update!(ch, 0)
    return ch
end

function assemble_system!(K, f, dh, cvu, cvp)
    assembler = start_assemble(K, f)
    ke = zeros(ndofs_per_cell(dh), ndofs_per_cell(dh))
    fe = zeros(ndofs_per_cell(dh))
    range_u = dof_range(dh, :u)
    ndofs_u = length(range_u)
    range_p = dof_range(dh, :p)
    ndofs_p = length(range_p)
    ϕᵤ = Vector{Vec{2, Float64}}(undef, ndofs_u)
    ∇ϕᵤ = Vector{Tensor{2, 2, Float64, 4}}(undef, ndofs_u)
    divϕᵤ = Vector{Float64}(undef, ndofs_u)
    ϕₚ = Vector{Float64}(undef, ndofs_p)
    for cell in CellIterator(dh)
        reinit!(cvu, cell)
        reinit!(cvp, cell)
        ke .= 0
        fe .= 0
        for qp in 1:getnquadpoints(cvu)
            dΩ = getdetJdV(cvu, qp)
            for i in 1:ndofs_u
                ϕᵤ[i] = shape_value(cvu, qp, i)
                ∇ϕᵤ[i] = shape_gradient(cvu, qp, i)
                divϕᵤ[i] = shape_divergence(cvu, qp, i)
            end
            for i in 1:ndofs_p
                ϕₚ[i] = shape_value(cvp, qp, i)
            end
            # u-u
            for (i, I) in pairs(range_u), (j, J) in pairs(range_u)
                ke[I, J] += (∇ϕᵤ[i] ⊡ ∇ϕᵤ[j]) * dΩ
            end
            # u-p
            for (i, I) in pairs(range_u), (j, J) in pairs(range_p)
                ke[I, J] += (-divϕᵤ[i] * ϕₚ[j]) * dΩ
            end
            # p-u
            for (i, I) in pairs(range_p), (j, J) in pairs(range_u)
                ke[I, J] += (-divϕᵤ[j] * ϕₚ[i]) * dΩ
            end
            # rhs
            for (i, I) in pairs(range_u)
                x = spatial_coordinate(cvu, qp, getcoordinates(cell))
                b = exp(-100 * norm(x - Vec{2}((0.75, 0.1)))^2)
                bv = Vec{2}((b, 0.0))
                fe[I] += (ϕᵤ[i] ⋅ bv) * dΩ
            end
        end
        assemble!(assembler, celldofs(cell), ke, fe)
    end
    return K, f
end

function main()
    # Grid
    h = 0.05 # approximate element size
    grid = setup_grid(h)
    # Interpolations
    ipu = Lagrange{RefTriangle, 2}()^2 # quadratic
    ipp = Lagrange{RefTriangle, 1}()   # linear
    # Dofs
    dh = setup_dofs(grid, ipu, ipp)
    # FE values
    ipg = Lagrange{RefTriangle, 1}() # linear geometric interpolation
    cvu, cvp, fvp = setup_fevalues(ipu, ipp, ipg)
    # Boundary conditions
    ch = setup_constraints(dh, fvp)
    # Global tangent matrix and rhs
    coupling = [true true; true false] # no coupling between pressure test/trial functions
    K = allocate_matrix(dh, ch; coupling = coupling)
    f = zeros(ndofs(dh))
    # Assemble system
    assemble_system!(K, f, dh, cvu, cvp)
    # Apply boundary conditions and solve
    apply!(K, f, ch)
    u = K \ f
    apply!(u, ch)
    # Export the solution
    VTKGridFile("stokes-flow", grid) do vtk
        write_solution(vtk, dh, u)
    end


    return
end

main()

# This file was generated using Literate.jl, https://github.com/fredrikekre/Literate.jl
