using Ferrite, IterativeSolvers, WriteVTK

base_grid = generate_grid(Hexahedron, (4, 4, 4));
grid = ForestBWG(base_grid, 10)
refine_all!(grid, 1);

analytical_solution(x) = exp(-((norm(x) - 0.5) / 0.05)^2)
analytical_rhs(x) = -laplace(analytical_solution, x)

function assemble_cell!(ke, fe, cellvalues, coords)
    n_basefuncs = getnbasefunctions(cellvalues)
    for q_point in 1:getnquadpoints(cellvalues)
        x = spatial_coordinate(cellvalues, q_point, coords)
        dΩ = getdetJdV(cellvalues, q_point)
        for i in 1:n_basefuncs
            Nᵢ = shape_value(cellvalues, q_point, i)
            ∇Nᵢ = shape_gradient(cellvalues, q_point, i)
            fe[i] += analytical_rhs(x) * Nᵢ * dΩ
            for j in 1:n_basefuncs
                ∇Nⱼ = shape_gradient(cellvalues, q_point, j)
                ke[i, j] += ∇Nⱼ ⋅ ∇Nᵢ * dΩ
            end
        end
    end
    return
end

function assemble_global!(K, f, dh, cellvalues)
    n_basefuncs = getnbasefunctions(cellvalues)
    ke = zeros(n_basefuncs, n_basefuncs)
    fe = zeros(n_basefuncs)
    assembler = start_assemble(K, f)
    for cell in CellIterator(dh)
        reinit!(cellvalues, cell)
        coords = getcoordinates(cell)
        fill!(ke, 0.0)
        fill!(fe, 0.0)
        assemble_cell!(ke, fe, cellvalues, coords)
        assemble!(assembler, celldofs(cell), ke, fe)
    end
    return K, f
end

function solve(grid)
    ip = Lagrange{RefHexahedron, 1}()
    qr = QuadratureRule{RefHexahedron}(2)
    cellvalues = CellValues(qr, ip)

    dh = DofHandler(grid)
    add!(dh, :u, ip)
    close!(dh)

    # Dirichlet BCs on all boundary faces, plus hanging node constraints
    ch = ConstraintHandler(dh)
    for face in ("top", "bottom", "left", "right", "front", "back")
        add!(ch, Dirichlet(:u, getfacetset(grid, face), (x, t) -> 0.0))
    end
    add!(ch, ConformityConstraint(:u))
    close!(ch)

    K = allocate_matrix(dh, ch)
    f = zeros(ndofs(dh))
    assemble_global!(K, f, dh, cellvalues)
    apply!(K, f, ch)
    u = cg(K, f; maxiter = 2000)
    apply!(u, ch)
    return u, dh, cellvalues, ip, qr
end

function estimate_error(grid, dh, u, cv, ip, qr)
    # Step 1: Compute the raw flux σ_h = ∇u_h at each quadrature point.
    σ_gp = Vector{Vector{Vec{3, Float64}}}()
    for cell in CellIterator(dh)
        reinit!(cv, cell)
        ue = u[celldofs(cell)]
        σ_cell = Vec{3, Float64}[]
        for q_point in 1:getnquadpoints(cv)
            push!(σ_cell, function_gradient(cv, q_point, ue))
        end
        push!(σ_gp, σ_cell)
    end

    # Step 2: Recover a smooth flux field σ* by L2-projecting the raw
    # quadrature-point fluxes onto a continuous nodal field.
    projector = L2Projector(ip, grid)
    σ_dof = project(projector, σ_gp, qr)

    # Step 3: Evaluate the ZZ error indicator per cell.
    # For each cell we compare the recovered flux σ* (evaluated from the
    # projected nodal values) against the raw flux σ_h at each quadrature point.
    cv_σ = CellValues(qr, ip^3)
    error_arr = zeros(getncells(grid))
    for (cellid, cell) in enumerate(CellIterator(projector.dh))
        reinit!(cv_σ, cell)
        @views σe = σ_dof[celldofs(cell)]
        for q_point in 1:getnquadpoints(cv_σ)
            σ_star = function_value(cv_σ, q_point, reinterpret(Float64, σe))
            σ_h = σ_gp[cellid][q_point]
            error_arr[cellid] += norm(σ_star - σ_h)^2 * getdetJdV(cv_σ, q_point)
        end
    end
    return error_arr
end

function true_error(grid, dh, u, cv)
    error_arr = zeros(getncells(grid))
    for (cellid, cell) in enumerate(CellIterator(dh))
        reinit!(cv, cell)
        ue = u[celldofs(cell)]
        coords = getcoordinates(cell)
        for q_point in 1:getnquadpoints(cv)
            x = spatial_coordinate(cv, q_point, coords)
            ∇u_exact = gradient(analytical_solution, x)
            ∇u_h = function_gradient(cv, q_point, ue)
            error_arr[cellid] += norm(∇u_h - ∇u_exact)^2 * getdetJdV(cv, q_point)
        end
    end
    return error_arr
end

function dorfler_mark(error_arr, θ)
    cells_to_refine = Int[]
    sizehint!(cells_to_refine, length(error_arr))
    total = sum(error_arr)
    total > 0 || return cells_to_refine, total
    perm = sortperm(error_arr; rev = true)
    target = θ * total
    acc = 0.0
    for idx in perm
        push!(cells_to_refine, idx)
        acc += error_arr[idx]
        acc >= target && break
    end
    return cells_to_refine, total
end

function solve_adaptive(initial_grid; nsteps = 3, θ = 0.5)
    grid = deepcopy(initial_grid)
    pvd = paraview_collection("heat_amr")
    for i in 1:nsteps
        # Materialize the forest into a NonConformingGrid and solve
        transferred_grid = creategrid(grid)
        u, dh, cv, ip, qr = solve(transferred_grid)

        # Estimate the error and mark cells with Dörfler marking
        error_arr = estimate_error(transferred_grid, dh, u, cv, ip, qr)
        cells_to_refine, total = dorfler_mark(error_arr, θ)

        @info "AMR step $i: $(length(cells_to_refine))/$(getncells(transferred_grid)) cells marked, total error = $total"

        # Export the solution, the estimated error and the true error to VTK
        VTKGridFile("heat_amr-$i", dh) do vtk
            write_solution(vtk, dh, u)
            write_cell_data(vtk, error_arr, "estimated error")
            write_cell_data(vtk, true_error(transferred_grid, dh, u, cv), "true error")
            pvd[i] = vtk
        end

        isempty(cells_to_refine) && break

        # Refine marked cells and enforce 2:1 balance across the forest
        refine!(grid, cells_to_refine)
        balanceforest!(grid)
    end
    return vtk_save(pvd)
end

solve_adaptive(grid);

# This file was generated using Literate.jl, https://github.com/fredrikekre/Literate.jl
