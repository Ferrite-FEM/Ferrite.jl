using Ferrite, FerriteGmsh, SparseArrays
#cells = [Hexahedron((1, 2, 5, 4, 10, 11, 14, 13)),Hexahedron((4, 5, 8, 7, 13, 14, 17, 16)), Hexahedron((3, 4, 7, 6, 12, 13, 16, 15)), Hexahedron((12, 13, 16, 15, 21, 22, 25, 24)), Hexahedron((13, 14, 17, 16, 22, 23, 26, 25)), Hexahedron((10, 11, 14, 13, 19, 20, 23, 22)), Hexahedron((9, 10, 13, 12, 18, 19, 22, 21))]
### beefy L cells = [Hexahedron((4, 5, 8, 7, 13, 14, 17, 16)), Hexahedron((3, 4, 7, 6, 12, 13, 16, 15)), Hexahedron((12, 13, 16, 15, 21, 22, 25, 24)), Hexahedron((13, 14, 17, 16, 22, 23, 26, 25)), Hexahedron((10, 11, 14, 13, 19, 20, 23, 22)), Hexahedron((9, 10, 13, 12, 18, 19, 22, 21))]
### cells = [Hexahedron((3, 4, 7, 6, 12, 13, 16, 15)), Hexahedron((12, 13, 16, 15, 21, 22, 25, 24)), Hexahedron((9, 10, 13, 12, 18, 19, 22, 21))]
#nodes = Node{3, Float64}[Node{3, Float64}(Vec{3}((0.0, -1.0, -1.0))), Node{3, Float64}(Vec{3}((1.0, -1.0, -1.0))), Node{3, Float64}(Vec{3}((-1.0, 0.0, -1.0))), Node{3, Float64}(Vec{3}((0.0, 0.0, -1.0))), Node{3, Float64}(Vec{3}((1.0, 0.0, -1.0))), Node{3, Float64}(Vec{3}((-1.0, 1.0, -1.0))), Node{3, Float64}(Vec{3}((0.0, 1.0, -1.0))), Node{3, Float64}(Vec{3}((1.0, 1.0, -1.0))), Node{3, Float64}(Vec{3}((-1.0, -1.0, 0.0))), Node{3, Float64}(Vec{3}((0.0, -1.0, 0.0))), Node{3, Float64}(Vec{3}((1.0, -1.0, 0.0))), Node{3, Float64}(Vec{3}((-1.0, 0.0, 0.0))), Node{3, Float64}(Vec{3}((0.0, 0.0, 0.0))), Node{3, Float64}(Vec{3}((1.0, 0.0, 0.0))), Node{3, Float64}(Vec{3}((-1.0, 1.0, 0.0))), Node{3, Float64}(Vec{3}((0.0, 1.0, 0.0))), Node{3, Float64}(Vec{3}((1.0, 1.0, 0.0))), Node{3, Float64}(Vec{3}((-1.0, -1.0, 1.0))), Node{3, Float64}(Vec{3}((0.0, -1.0, 1.0))), Node{3, Float64}(Vec{3}((1.0, -1.0, 1.0))), Node{3, Float64}(Vec{3}((-1.0, 0.0, 1.0))), Node{3, Float64}(Vec{3}((0.0, 0.0, 1.0))), Node{3, Float64}(Vec{3}((1.0, 0.0, 1.0))), Node{3, Float64}(Vec{3}((-1.0, 1.0, 1.0))), Node{3, Float64}(Vec{3}((0.0, 1.0, 1.0))), Node{3, Float64}(Vec{3}((1.0, 1.0, 1.0)))]
#grid = Grid(cells,nodes)
#addfacetset!(grid,"front",x->x[2]≈-1)
#addfacetset!(grid,"back",x->x[2]≈1)
#addfacetset!(grid,"left",x->x[3]≈-1)
#addfacetset!(grid,"right",x->x[3]≈1)
#addfacetset!(grid,"top",x->x[1]≈-1)
#addfacetset!(grid,"bottom",x->x[1]≈1)
#addfacetset!(grid,"pull",x->x[1]≈0 && x[2] <= 0.5 && x[3] <= 0.5)
grid = generate_grid(Hexahedron,(2,1,1))
grid  = ForestBWG(grid,20)
Ferrite.refine_all!(grid,1)
Ferrite.refine!(grid,[1,2])
Ferrite.balanceforest!(grid)

struct Elasticity
    G::Float64
    K::Float64
end

function material_routine(material::Elasticity, ε::SymmetricTensor{dim}) where dim
    (; G, K) = material
    stress(ε) = 2G * dev(ε) + K * tr(ε) * one(ε)
    ∂σ∂ε, σ = gradient(stress, ε, :all)
    return σ, ∂σ∂ε
end

E = 200e3 # Young's modulus [MPa]
ν = 0.2 # Poisson's ratio [-]
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
            ∇Nᵢ = shape_gradient(cellvalues, q_point, i)
            fe[i] += σ ⊡ ∇Nᵢ * dΩ
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
        reinit!(cellvalues, cell)
        @views ue = a[celldofs(cell)]
        ## Compute element contribution
        assemble_cell!(ke, fe, cellvalues, material, ue)
        ## Assemble ke and fe into K and f
        assemble!(assembler, celldofs(cell), ke, fe)
    end
    return K, f
end

function solve(grid)
    dim = 3
    order = 1
    ip = Lagrange{RefHexahedron, order}()^dim
    qr = QuadratureRule{RefHexahedron}(2)
    cellvalues = CellValues(qr, ip);

    dh = DofHandler(grid)
    add!(dh, :u, ip)
    close!(dh);

    ch = ConstraintHandler(dh)
    add!(ch, Ferrite.ConformityConstraint(:u))
    add!(ch, Dirichlet(:u, getfacetset(grid, "bottom"), (x, t) -> Vec{3}((0.0,0.0,0.0)), [1,2,3]))
    add!(ch, Dirichlet(:u, getfacetset(grid, "back"), (x, t) -> Vec{3}((0.0,0.0,0.0)), [1,2,3]))
    add!(ch, Dirichlet(:u, getfacetset(grid, "front"), (x, t) -> Vec{3}((0.0,0.0,0.0)), [1,2,3]))
    add!(ch, Dirichlet(:u, getfacetset(grid, "top"), (x, t) -> Vec{3}((0.0,0.0,0.0)), [1,2,3]))
    add!(ch, Dirichlet(:u, getfacetset(grid, "right"), (x, t) -> Vec{3}((0.0,0.0,0.0)), [1,2,3]))
    add!(ch, Dirichlet(:u, getfacetset(grid, "left"), (x, t) -> Vec{3}((-0.1,0.0,0.0)), [1,2,3]))
    close!(ch);

    K = create_sparsity_pattern(dh,ch)
    f = zeros(ndofs(dh))
    a = zeros(ndofs(dh))
    assemble_global!(K, f, a, dh, cellvalues, material);
    apply!(K, f, ch)
    u = K \ f;
    apply!(u,ch)
    return u,dh,ch,cellvalues
end

function compute_fluxes(u,dh::DofHandler{dim}) where dim
    ip = Lagrange{RefHexahedron, 1}()^dim
    qr = QuadratureRule{RefHexahedron}(2)
    cellvalues_sc = CellValues(qr, ip);
    ## "Normal" quadrature points for the fluxes
    qr = QuadratureRule{RefHexahedron}(2)
    cellvalues = CellValues(qr, ip);
    ## Buffers
    σ_gp_sc = Vector{Vector{SymmetricTensor{2,dim,Float64,dim*2}}}()
    σ_gp_sc_loc = Vector{SymmetricTensor{2,dim,Float64,dim*2}}()
    σ_gp = Vector{Vector{SymmetricTensor{2,dim,Float64,dim*2}}}()
    σ_gp_loc = Vector{SymmetricTensor{2,dim,Float64,dim*2}}()
    for (cellid,cell) in enumerate(CellIterator(dh))
        reinit!(cellvalues, cell)
        reinit!(cellvalues_sc, cell)
        @views ue = u[celldofs(cell)]
        for q_point in 1:getnquadpoints(cellvalues)
            ε = function_symmetric_gradient(cellvalues, q_point, ue)
            σ, _ = material_routine(material, ε)
            push!(σ_gp_loc, σ)
        end
        for q_point in 1:getnquadpoints(cellvalues_sc)
            ε = function_symmetric_gradient(cellvalues_sc, q_point, ue)
            σ, _ = material_routine(material, ε)
            push!(σ_gp_sc_loc, σ)
        end
        push!(σ_gp,copy(σ_gp_loc))
        push!(σ_gp_sc,copy(σ_gp_sc_loc))
        ## Reset buffer for local points
        empty!(σ_gp_loc)
        empty!(σ_gp_sc_loc)
    end
    return σ_gp, σ_gp_sc
end

function solve_adaptive(initial_grid)
    ip = Lagrange{RefHexahedron, 1}()
    qr = QuadratureRule{RefHexahedron}(2)
    cellvalues_tensorial = CellValues(qr, ip);
    finished = false
    i = 1
    grid = initial_grid
    transfered_grid = Ferrite.creategrid(grid)
    pvd = VTKFileCollection("linear-elasticity.pvd",transfered_grid)
    while !finished
        transfered_grid = Ferrite.creategrid(grid)
        u,dh,ch,cv = solve(transfered_grid)
        σ_gp, σ_gp_sc = compute_fluxes(u,dh)
        projector = L2Projector(Lagrange{RefHexahedron, 1}()^3, transfered_grid)
        σ_dof = project(projector, σ_gp, QuadratureRule{RefHexahedron}(2))
        cells_to_refine = Int[]
        error_arr = Float64[]
        for (cellid,cell) in enumerate(CellIterator(projector.dh))
            reinit!(cellvalues_tensorial, cell)
            @views σe = σ_dof[celldofs(cell)]
            error = 0.0
            for q_point in 1:getnquadpoints(cellvalues_tensorial)
                σ_dof_at_sc = function_value(cellvalues_tensorial, q_point, σe)
                error += norm((σ_gp_sc[cellid][q_point] - σ_dof_at_sc )) * getdetJdV(cellvalues_tensorial,q_point)
            end
            push!(error_arr,error)
        end
        η = maximum(error_arr)
        θ = 0.5
        for (cellid,cell_err) in enumerate(error_arr)
            if cell_err > θ*η
                push!(cells_to_refine,cellid)
            end
        end
        addstep!(pvd,i,transfered_grid) do vtk
            write_solution(vtk, dh, u)
            write_projection(vtk, projector, σ_dof, "stress")
            write_cell_data(vtk, getindex.(collect(Iterators.flatten(σ_gp_sc)),1), "stress sc")
            write_cell_data(vtk, error_arr, "error")
        end

        Ferrite.refine!(grid,cells_to_refine)
        Ferrite.balanceforest!(grid)

        i += 1
        if isempty(cells_to_refine) || maximum(error_arr) < 0.05
            finished = true
        end
        break
    end
    close(pvd)
end

solve_adaptive(grid)
