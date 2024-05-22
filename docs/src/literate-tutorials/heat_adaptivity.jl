using Ferrite, FerriteGmsh, SparseArrays
grid = generate_grid(Quadrilateral, (8,8));
function random_deformation_field(x)
    if any(x .≈ -1.0) || any(x .≈ 1.0)
        return x
    else
        Vec{2}(x .+ (rand(2).-0.5)*0.15)
    end
end
transform_coordinates!(grid, random_deformation_field)
grid  = ForestBWG(grid,10)

analytical_solution(x) = atan(2*(norm(x)-0.5)/0.02)
analytical_rhs(x) = -laplace(analytical_solution,x)

function assemble_cell!(ke, fe, cellvalues, ue, coords)
    fill!(ke, 0.0)
    fill!(fe, 0.0)

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
end

function assemble_global!(K, f, a, dh, cellvalues)
    ## Allocate the element stiffness matrix and element force vector
    n_basefuncs = getnbasefunctions(cellvalues)
    ke = zeros(n_basefuncs, n_basefuncs)
    fe = zeros(n_basefuncs)
    ## Create an assembler
    assembler = start_assemble(K, f)
    ## Loop over all cells
    for cell in CellIterator(dh)
        reinit!(cellvalues, cell) 
        @views ue = a[celldofs(cell)]
        ## Compute element contribution
        coords = getcoordinates(cell)
        assemble_cell!(ke, fe, cellvalues, ue, coords)
        ## Assemble ke and fe into K and f
        assemble!(assembler, celldofs(cell), ke, fe)
    end
    return K, f
end

function solve(grid)
    dim = 2
    order = 1 
    ip = Lagrange{RefQuadrilateral, order}()
    qr = QuadratureRule{RefQuadrilateral}(2)
    cellvalues = CellValues(qr, ip);

    dh = DofHandler(grid)
    add!(dh, :u, ip)
    close!(dh);

    ch = ConstraintHandler(dh)
    add!(ch, ConformityConstraint(:u))
    add!(ch, Dirichlet(:u, getfacetset(grid, "top"), (x, t) -> 0.0))
    add!(ch, Dirichlet(:u, getfacetset(grid, "right"), (x, t) -> 0.0))
    add!(ch, Dirichlet(:u, getfacetset(grid, "left"), (x, t) -> 0.0))
    add!(ch, Dirichlet(:u, getfacetset(grid, "bottom"), (x, t) -> 0.0))
    close!(ch);

    K = create_sparsity_pattern(dh,ch)
    f = zeros(ndofs(dh))
    a = zeros(ndofs(dh))
    assemble_global!(K, f, a, dh, cellvalues);
    apply!(K, f, ch)
    u = K \ f;
    apply!(u,ch)
    return u,dh,ch,cellvalues
end

function compute_fluxes(u,dh)
    ip = Lagrange{RefQuadrilateral, 1}()
    ## Normal quadrature points
    qr = QuadratureRule{RefQuadrilateral}(2)
    cellvalues = CellValues(qr, ip);
    ## Superconvergent point
    qr_sc = QuadratureRule{RefQuadrilateral}(1)
    cellvalues_sc = CellValues(qr_sc, ip);
    ## Buffers
    σ_gp = Vector{Vector{Vec{2,Float64}}}()
    σ_gp_loc = Vector{Vec{2,Float64}}()
    σ_gp_sc = Vector{Vector{Vec{2,Float64}}}()
    σ_gp_sc_loc = Vector{Vec{2,Float64}}()
    for (cellid,cell) in enumerate(CellIterator(dh))
        @views ue = u[celldofs(cell)]

        reinit!(cellvalues, cell)
        for q_point in 1:getnquadpoints(cellvalues)
            gradu = function_gradient(cellvalues, q_point, ue)
            push!(σ_gp_loc, gradu)
        end
        push!(σ_gp,copy(σ_gp_loc))
        empty!(σ_gp_loc)

        reinit!(cellvalues_sc, cell)
        for q_point in 1:getnquadpoints(cellvalues_sc)
            gradu = function_gradient(cellvalues_sc, q_point, ue)
            push!(σ_gp_sc_loc, gradu)
        end
        push!(σ_gp_sc,copy(σ_gp_sc_loc))
        empty!(σ_gp_sc_loc)
    end
    return σ_gp, σ_gp_sc
end

function solve_adaptive(initial_grid)
    ip = Lagrange{RefQuadrilateral, 1}()^2
    qr_sc = QuadratureRule{RefQuadrilateral}(1)
    cellvalues_flux = CellValues(qr_sc, ip);
    finished = false
    i = 1
    grid = deepcopy(initial_grid)
    pvd = VTKFileCollection("heat_amr.pvd",grid);
    while !finished && i<=10
        @show i
        transfered_grid = Ferrite.creategrid(grid)
        u,dh,ch,cv = solve(transfered_grid)
        σ_gp, σ_gp_sc = compute_fluxes(u,dh)
        projector = L2Projector(Lagrange{RefQuadrilateral, 1}(), transfered_grid)
        σ_dof = project(projector, σ_gp, QuadratureRule{RefQuadrilateral}(2))
        cells_to_refine = Int[]
        error_arr = Float64[]
        for (cellid,cell) in enumerate(CellIterator(projector.dh))
            reinit!(cellvalues_flux, cell)
            @views σe = σ_dof[celldofs(cell)]
            error = 0.0
            for q_point in 1:getnquadpoints(cellvalues_flux)
                σ_dof_at_sc = function_value(cellvalues_flux, q_point, σe)
                error += norm((σ_gp_sc[cellid][q_point] - σ_dof_at_sc ))
                error *= getdetJdV(cellvalues_flux,q_point)
            end
            if error > 0.001
                push!(cells_to_refine,cellid)
            end
            push!(error_arr,error)
        end

        addstep!(pvd, i, dh) do vtk
            write_solution(vtk, dh, u)
            write_projection(vtk, projector, σ_dof, "flux")
            write_cell_data(vtk, getindex.(collect(Iterators.flatten(σ_gp_sc)),1), "flux sc x")
            write_cell_data(vtk, getindex.(collect(Iterators.flatten(σ_gp_sc)),2), "flux sc y")
            write_cell_data(vtk, error_arr, "error")
        end

        Ferrite.refine!(grid, cells_to_refine)
        Ferrite.balanceforest!(grid)

        i += 1
        if isempty(cells_to_refine)
            finished = true
        end
    end
    close(pvd);
end

solve_adaptive(grid)
