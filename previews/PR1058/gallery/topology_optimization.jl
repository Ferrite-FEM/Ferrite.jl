using Ferrite, SparseArrays, LinearAlgebra, Tensors, Printf

function create_grid(n)
    corners = [Vec{2}((0.0, 0.0)),
               Vec{2}((2.0, 0.0)),
               Vec{2}((2.0, 1.0)),
               Vec{2}((0.0, 1.0))]
    grid = generate_grid(Quadrilateral, (2*n, n), corners);

    # node-/facesets for boundary conditions
    addnodeset!(grid, "clamped", x -> x[1] ≈ 0.0)
    addfacetset!(grid, "traction", x -> x[1] ≈ 2.0 && norm(x[2]-0.5) <= 0.05);
    return grid
end

function create_values()
    # quadrature rules
    qr      = QuadratureRule{RefQuadrilateral}(2)
    facet_qr = FacetQuadratureRule{RefQuadrilateral}(2)

    # cell and facetvalues for u
    ip = Lagrange{RefQuadrilateral,1}()^2
    cellvalues = CellValues(qr, ip)
    facetvalues = FacetValues(facet_qr, ip)

    return cellvalues, facetvalues
end

function create_dofhandler(grid)
    dh = DofHandler(grid)
    add!(dh, :u, Lagrange{RefQuadrilateral,1}()^2) # displacement
    close!(dh)
    return dh
end

function create_bc(dh)
    dbc = ConstraintHandler(dh)
    add!(dbc, Dirichlet(:u, getnodeset(dh.grid, "clamped"), (x,t) -> zero(Vec{2}), [1,2]))
    close!(dbc)
    t = 0.0
    update!(dbc, t)
    return dbc
end

struct MaterialParameters{T, S <: SymmetricTensor{4, 2, T}}
    C::S
    χ_min::T
    p::T
    β::T
    η::T
end

function MaterialParameters(E, ν, χ_min, p, β, η)
    δ(i,j) = i == j ? 1.0 : 0.0 # helper function

    G = E / 2(1 + ν) # =μ
    λ = E*ν/(1-ν^2) # correction for plane stress included

    C = SymmetricTensor{4, 2}((i,j,k,l) -> λ * δ(i,j)*δ(k,l) + G* (δ(i,k)*δ(j,l) + δ(i,l)*δ(j,k)))
    return MaterialParameters(C, χ_min, p, β, η)
end

mutable struct MaterialState{T, S <: AbstractArray{SymmetricTensor{2, 2, T, 3}, 1}}
    χ::T # density
    ε::S # strain in each quadrature point
end

function MaterialState(ρ, n_qp)
    return MaterialState(ρ, Array{SymmetricTensor{2,2,Float64,3},1}(undef, n_qp))
end

function update_material_states!(χn1, states, dh)
    for (element, state) in zip(CellIterator(dh),states)
        state.χ = χn1[cellid(element)]
    end
end

function compute_driving_forces(states, mp, dh, χn)
    pΨ = zeros(length(states))
    for (element, state) in zip(CellIterator(dh), states)
        i = cellid(element)
        ε = sum(state.ε)/length(state.ε) # average element strain
        pΨ[i] = 1/2 * mp.p * χn[i]^(mp.p-1) * (ε ⊡ mp.C ⊡ ε)
    end
    return pΨ
end

function compute_densities(states, dh)
    χn = zeros(length(states))
    for (element, state) in zip(CellIterator(dh), states)
        i = cellid(element)
        χn[i] = state.χ
    end
    return χn
end

function cache_neighborhood(dh, topology)
    nbgs = Vector{Vector{Int}}(undef, getncells(dh.grid))
    _nfacets = nfacets(dh.grid.cells[1])
    opp = Dict(1=>3, 2=>4, 3=>1, 4=>2)

    for element in CellIterator(dh)
        nbg = zeros(Int,_nfacets)
        i = cellid(element)
        for j in 1:_nfacets
            nbg_cellid = getneighborhood(topology, dh.grid, FacetIndex(i,j))
            if(!isempty(nbg_cellid))
                nbg[j] = first(nbg_cellid)[1] # assuming only one facet neighbor per cell
            else # boundary facet
                nbg[j] = first(getneighborhood(topology, dh.grid, FacetIndex(i,opp[j])))[1]
            end
        end

        nbgs[i] = nbg
    end

    return nbgs
end

function approximate_laplacian(nbgs, χn, Δh)
    ∇²χ = zeros(length(nbgs))
    for i in 1:length(nbgs)
        nbg = nbgs[i]
        ∇²χ[i] = (χn[nbg[1]]+χn[nbg[2]]+χn[nbg[3]]+χn[nbg[4]]-4*χn[i])/(Δh^2)
    end

    return ∇²χ
end

function compute_χn1(χn, Δχ, ρ, ηs, χ_min)
    n_el = length(χn)

    χ_trial = zeros(n_el)
    ρ_trial = 0.0

    λ_lower = minimum(Δχ) - ηs
    λ_upper = maximum(Δχ) + ηs
    λ_trial = 0.0

    while(abs(ρ-ρ_trial)>1e-7)
        for i in 1:n_el
            Δχt = 1/ηs * (Δχ[i] - λ_trial)
            χ_trial[i] = max(χ_min, min(1.0, χn[i]+Δχt))
        end

        ρ_trial = 0.0
        for i in 1:n_el
            ρ_trial += χ_trial[i]/n_el
        end

        if(ρ_trial > ρ)
            λ_lower = λ_trial
        elseif(ρ_trial < ρ)
            λ_upper = λ_trial
        end
        λ_trial = 1/2*(λ_upper+λ_lower)
    end

    return χ_trial
end

function compute_average_driving_force(mp, pΨ, χn)
    n = length(pΨ)
    w = zeros(n)

    for i in 1:n
        w[i] = (χn[i]-mp.χ_min)*(1-χn[i])
    end

    p_Ω = sum(w.*pΨ)/sum(w) # average driving force

    return p_Ω
end

function update_density(dh, states, mp, ρ,  neighboorhoods, Δh)
    n_j = Int(ceil(6*mp.β/(mp.η*Δh^2))) # iterations needed for stability
    χn = compute_densities(states, dh) # old density field
    χn1 = zeros(length(χn))

    for j in 1:n_j
        ∇²χ = approximate_laplacian(neighboorhoods, χn, Δh) # Laplacian
        pΨ = compute_driving_forces(states, mp, dh, χn) # driving forces
        p_Ω = compute_average_driving_force(mp, pΨ, χn) # average driving force

        Δχ = pΨ/p_Ω + mp.β*∇²χ

        χn1 = compute_χn1(χn, Δχ, ρ, mp.η, mp.χ_min)

        if(j<n_j)
            χn[:] = χn1[:]
        end
    end

    return χn1
end

function doassemble!(cellvalues::CellValues, facetvalues::FacetValues, K::SparseMatrixCSC, grid::Grid, dh::DofHandler, mp::MaterialParameters, u, states)
    r = zeros(ndofs(dh))
    assembler = start_assemble(K, r)
    nu = getnbasefunctions(cellvalues)

    re = zeros(nu) # local residual vector
    Ke = zeros(nu,nu) # local stiffness matrix

    for (element, state) in zip(CellIterator(dh), states)
        fill!(Ke, 0)
        fill!(re, 0)

        eldofs = celldofs(element)
        ue = u[eldofs]

        elmt!(Ke, re, element, cellvalues, facetvalues, grid, mp, ue, state)
        assemble!(assembler, celldofs(element), re, Ke)
    end

    return K, r
end

function elmt!(Ke, re, element, cellvalues, facetvalues, grid, mp, ue, state)
    n_basefuncs = getnbasefunctions(cellvalues)
    reinit!(cellvalues, element)
    χ = state.χ

    # We only assemble lower half triangle of the stiffness matrix and then symmetrize it.
    @inbounds for q_point in 1:getnquadpoints(cellvalues)
        dΩ = getdetJdV(cellvalues, q_point)
        state.ε[q_point] = function_symmetric_gradient(cellvalues, q_point, ue)

        for i in 1:n_basefuncs
            δεi = shape_symmetric_gradient(cellvalues, q_point, i)
            for j in 1:i
                δεj = shape_symmetric_gradient(cellvalues, q_point, j)
                Ke[i,j] += (χ)^(mp.p) * (δεi ⊡ mp.C ⊡ δεj) * dΩ
            end
            re[i] += (-δεi ⊡ ((χ)^(mp.p) * mp.C ⊡ state.ε[q_point])) * dΩ
        end
    end

    symmetrize_lower!(Ke)

    @inbounds for facet in 1:nfacets(getcells(grid, cellid(element)))
        if (cellid(element), facet) ∈ getfacetset(grid, "traction")
            reinit!(facetvalues, element, facet)
            t = Vec((0.0, -1.0)) # force pointing downwards
            for q_point in 1:getnquadpoints(facetvalues)
                dΓ = getdetJdV(facetvalues, q_point)
                for i in 1:n_basefuncs
                    δu = shape_value(facetvalues, q_point, i)
                    re[i] += (δu ⋅ t) * dΓ
                end
            end
        end
    end

end

function symmetrize_lower!(K)
    for i in 1:size(K,1)
        for j in i+1:size(K,1)
            K[i,j] = K[j,i]
        end
    end
end

function topopt(ra,ρ,n,filename; output=:false)
    # material
    mp = MaterialParameters(210.e3, 0.3, 1.e-3, 3.0, ra^2, 15.0)

    # grid, dofhandler, boundary condition
    grid = create_grid(n)
    dh = create_dofhandler(grid)
    Δh = 1/n # element edge length
    dbc = create_bc(dh)

    # cellvalues
    cellvalues, facetvalues = create_values()

    # Pre-allocate solution vectors, etc.
    n_dofs = ndofs(dh) # total number of dofs
    u  = zeros(n_dofs) # solution vector
    un = zeros(n_dofs) # previous solution vector

    Δu = zeros(n_dofs)  # previous displacement correction
    ΔΔu = zeros(n_dofs) # new displacement correction

    # create material states
    states = [MaterialState(ρ, getnquadpoints(cellvalues)) for _ in 1:getncells(dh.grid)]

    χ = zeros(getncells(dh.grid))

    r = zeros(n_dofs) # residual
    K = allocate_matrix(dh) # stiffness matrix

    i_max = 300 ## maximum number of iteration steps
    tol = 1e-4
    compliance = 0.0
    compliance_0 = 0.0
    compliance_n = 0.0
    conv = :false

    topology = ExclusiveTopology(grid)
    neighboorhoods = cache_neighborhood(dh, topology)

    # Newton-Raphson loop
    NEWTON_TOL = 1e-8
    print("\n Starting Newton iterations\n")

    for it in 1:i_max
        apply_zero!(u, dbc)
        newton_itr = -1

        while true; newton_itr += 1

            if newton_itr > 10
                error("Reached maximum Newton iterations, aborting")
                break
            end

            # current guess
            u .= un .+ Δu
            K, r = doassemble!(cellvalues, facetvalues, K, grid, dh, mp, u, states);
            norm_r = norm(r[Ferrite.free_dofs(dbc)])

            if (norm_r) < NEWTON_TOL
                break
            end

            apply_zero!(K, r, dbc)
            ΔΔu = Symmetric(K) \ r

            apply_zero!(ΔΔu, dbc)
            Δu .+= ΔΔu
        end # of loop while NR-Iteration

        # calculate compliance
        compliance = 1/2 * u' * K * u

        if(it==1)
            compliance_0 = compliance
        end

        # check convergence criterium (twice!)
        if(abs(compliance-compliance_n)/compliance < tol)
            if(conv)
                println("Converged at iteration number: ", it)
                break
            else
                conv = :true
            end
        else
            conv = :false
        end

        # update density
        χ = update_density(dh, states, mp, ρ, neighboorhoods, Δh)

        # update old displacement, density and compliance
        un .= u
        Δu .= 0.0
        update_material_states!(χ, states, dh)
        compliance_n = compliance

        # output during calculation
        if(output)
            i = @sprintf("%3.3i", it)
            filename_it = string(filename, "_", i)

            VTKGridFile(filename_it, grid) do vtk
                write_cell_data(vtk, χ, "density")
            end
        end
    end

    # export converged results
    if(!output)
        VTKGridFile(filename, grid) do vtk
            write_cell_data(vtk, χ, "density")
        end
    end
    @printf "Rel. stiffness: %.4f \n" compliance^(-1)/compliance_0^(-1)

    return
end

@time topopt(0.03, 0.5, 60, "large_radius"; output=:false);
#topopt(0.02, 0.5, 60, "topopt_animation"; output=:true); # can be used to create animations

# This file was generated using Literate.jl, https://github.com/fredrikekre/Literate.jl
