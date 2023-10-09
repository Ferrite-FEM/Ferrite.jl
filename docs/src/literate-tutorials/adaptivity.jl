using Ferrite, FerriteGmsh, SparseArrays
grid = togrid("l.msh");
grid  = ForestBWG(grid,10)

struct Elasticity
    G::Float64
    K::Float64
end

function material_routine(material::Elasticity, ε::SymmetricTensor{2})
    (; G, K) = material
    stress(ε) = 2G * dev(ε) + K * tr(ε) * one(ε)
    ∂σ∂ε, σ = gradient(stress, ε, :all)
    return σ, ∂σ∂ε
end

E = 200e3 # Young's modulus [MPa]
ν = 0.3 # Poisson's ratio [-]
material = Elasticity(E/2(1+ν), E/3(1-2ν));

function assemble_cell!(ke, fe, cellvalues, material, ue)
    fill!(ke, 0.0)
    fill!(fe, 0.0)

    n_basefuncs = getnbasefunctions(cellvalues)
    for q_point in 1:getnquadpoints(cellvalues)
        ## For each integration point, compute strain, stress and material stiffness
        ε = function_symmetric_gradient(cellvalues, q_point, ue)
        σ, ∂σ∂ε = material_routine(material, ε)

        dΩ = getdetJdV(cellvalues, q_point)
        for i in 1:n_basefuncs
            ∇Nᵢ = shape_gradient(cellvalues, q_point, i)# shape_symmetric_gradient(cellvalues, q_point, i)
            fe[i] += σ ⊡ ∇Nᵢ * dΩ # add internal force to residual
            for j in 1:n_basefuncs
                ∇ˢʸᵐNⱼ = shape_symmetric_gradient(cellvalues, q_point, j)
                ke[i, j] += (∂σ∂ε ⊡ ∇ˢʸᵐNⱼ) ⊡ ∇Nᵢ * dΩ
            end
        end
    end
end

function assemble_global!(K, f, a, dh, cellvalues, material)
    ## Allocate the element stiffness matrix and element force vector
    n_basefuncs = getnbasefunctions(cellvalues)
    ke = zeros(n_basefuncs, n_basefuncs)
    fe = zeros(n_basefuncs)
    ## Create an assembler
    assembler = start_assemble(K, f)
    ## Loop over all cells
    for cell in CellIterator(dh)
        reinit!(cellvalues, cell) # update spatial derivatives based on element coordinates
        @views ue = a[celldofs(cell)]
        ## Compute element contribution
        assemble_cell!(ke, fe, cellvalues, material, ue)
        ## Assemble ke and fe into K and f
        assemble!(assembler, celldofs(cell), ke, fe)
    end
    return K, f
end

function solve(grid,hnodes)
    dim = 2
    order = 1 # linear interpolation
    ip = Lagrange{RefQuadrilateral, order}()^dim # vector valued interpolation
    qr = QuadratureRule{RefQuadrilateral}(2) # 1 quadrature point
    cellvalues = CellValues(qr, ip);

    dh = DofHandler(grid)
    add!(dh, :u, ip)
    dh, vdict, edict, fdict = Ferrite.__close!(dh);

    ch = ConstraintHandler(dh)
    add!(ch, Dirichlet(:u, getfaceset(grid, "top"), (x, t) -> Vec{2}((0.0,0.0)), [1,2]))
    add!(ch, Dirichlet(:u, getfaceset(grid, "right"), (x, t) -> 0.01, 2))
    for (hdof,mdof) in hnodes
        lc = AffineConstraint(vdict[1][hdof],[vdict[1][m] => 0.5 for m in mdof],0.0)
        add!(ch,lc)
    end
    close!(ch);

    K = create_sparsity_pattern(dh,ch)
    f = zeros(ndofs(dh))
    a = zeros(ndofs(dh))
    assemble_global!(K, f, a, dh, cellvalues, material);
    apply!(K, f, ch)
    u = K \ f;
    apply!(u,ch)
    return u,dh,ch,cellvalues,vdict
end

function compute_fluxes(u,dh)
    ip = Lagrange{RefQuadrilateral, 1}()^2
    qr = QuadratureRule{RefQuadrilateral}(1)
    cellvalues_sc = CellValues(qr, ip);
    qr = QuadratureRule{RefQuadrilateral}(2)
    cellvalues = CellValues(qr, ip);
    σ_gp_sc = Vector{Vector{SymmetricTensor{2,2,Float64,3}}}()
    σ_gp_sc_loc = Vector{SymmetricTensor{2,2,Float64,3}}()
    σ_gp = Vector{Vector{SymmetricTensor{2,2,Float64,3}}}()
    σ_gp_loc = Vector{SymmetricTensor{2,2,Float64,3}}()
    for (cellid,cell) in enumerate(CellIterator(dh))
        reinit!(cellvalues, cell) # update spatial derivatives based on element coordinates
        @views ue = u[celldofs(cell)]
        for q_point in 1:getnquadpoints(cellvalues)
            ε = function_symmetric_gradient(cellvalues, q_point, ue)
            σ, ∂σ∂ε = material_routine(material, ε)
            push!(σ_gp_loc, σ)
        end
        for q_point in 1:getnquadpoints(cellvalues_sc)
            ε = function_symmetric_gradient(cellvalues, q_point, ue)
            σ, ∂σ∂ε = material_routine(material, ε)
            push!(σ_gp_sc_loc, σ)
        end
        push!(σ_gp,copy(σ_gp_loc))
        push!(σ_gp_sc,copy(σ_gp_sc_loc))
        empty!(σ_gp_loc)
        empty!(σ_gp_sc_loc)
    end
    return σ_gp, σ_gp_sc
end

function solve_adaptive(initial_grid)
    ip = Lagrange{RefQuadrilateral, 1}()
    qr = QuadratureRule{RefQuadrilateral}(1)
    cellvalues_tensorial = CellValues(qr, ip);
    finished = false
    i = 1
    grid = initial_grid
    while !finished && i<=20
        transfered_grid, hnodes = Ferrite.creategrid(grid)
        u,dh,ch,cv,vdict = solve(transfered_grid,hnodes)
        σ_gp, σ_gp_sc = compute_fluxes(u,dh)
        projector = L2Projector(Lagrange{RefQuadrilateral, 1}()^2, transfered_grid; hnodes=hnodes)
        σ_dof = project(projector, σ_gp, QuadratureRule{RefQuadrilateral}(2))
        cells_to_refine = Int[]
        error_arr = Float64[]
        for (cellid,cell) in enumerate(CellIterator(projector.dh))
            reinit!(cellvalues_tensorial, cell)
            @views σe = σ_dof[celldofs(cell)]
            error = 0.0
            for q_point in 1:getnquadpoints(cellvalues_tensorial)
                σ_dof_at_sc = function_value(cellvalues_tensorial, q_point, σe)
                #error += norm((σ_dof_at_sc - σ_gp_sc[cellid][1])/σ_gp_sc[cellid][1])
                error += norm((σ_gp_sc[cellid][1] - σ_dof_at_sc ))
                error *= getdetJdV(cellvalues_tensorial,q_point)
            end
            if error > 0.01
                push!(cells_to_refine,cellid)
            end
            push!(error_arr,error)
        end
        vtk_grid("linear_elasticity-$i", dh) do vtk
            vtk_point_data(vtk, dh, u)
            vtk_point_data(vtk, projector, σ_dof, "stress")
            vtk_cell_data(vtk, getindex.(collect(Iterators.flatten(σ_gp_sc)),1), "stress sc")
            vtk_cell_data(vtk, error_arr, "error")
        end
        Ferrite.refine!(grid,cells_to_refine)
        transfered_grid, hnodes = Ferrite.creategrid(grid)
        Ferrite.balanceforest!(grid)
        transfered_grid, hnodes = Ferrite.creategrid(grid)
        i += 1
        if isempty(cells_to_refine)
            finished = true
        end
    end
end

solve_adaptive(grid)
