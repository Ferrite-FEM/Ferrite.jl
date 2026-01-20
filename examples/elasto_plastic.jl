using JuAFEM
using Tensors
using KrylovMethods
using TimerOutputs
using Statistics: mean
import ProgressMeter
const ∇ = Tensors.gradient

struct Material{T}
    μ::Float64		#Lamé constant
    λ::Float64		#Lamé constant
    τ₀::Float64		#Yield stress
    H::Float64		#Hardening
end

struct MaterialState{T}
	S::SymmetricTensor{2,3,T,6} #2nd PK stress
    ϵᵖ::Float64					#Plastic strain
    F⁻¹::Tensor{2,3,T,9}		#Plastic deformation grad
    ν::Tensor{2,3,T,9}			
end

function MaterialState()
    return MaterialState(
    				zero(SymmetricTensor{2,3,Float64,6}),
    				0.0, 
    	            one(Tensor{2,3,Float64,9}), 
    	            zero(Tensor{2,3,Float64,9}),)
end

function _compute_2nd_PK(mp::Material, E, state::MaterialState)
    μ, λ, τ₀, H = mp.μ, mp.λ, mp.τ₀, mp.H
    
    ϵᵖ = state.ϵᵖ
    F⁻¹ = state.F⁻¹
    ν = state.ν
    dγ = 0 

    C = 2E + one(E)

    phi, Mdev, Stilde = ϕ(C, F⁻¹, ϵᵖ, μ, λ, τ₀, H)

    if phi > 0 #Plastic step
        dγ = 0.0
        
        #Setup newton varibles
        TOL = 0.00001
        newton_error, counter = TOL + 1, 0

        # Define residual for semi-emplicit backward euler (assuming ⁿ⁺¹ν = ⁿν)
        ϕ_residual(dγ) = ϕ(C, F⁻¹ - dγ*F⁻¹⋅ν, ϵᵖ + dγ, μ, λ, τ₀, H)[1]

        while newton_error > TOL; counter +=1 
            
            res = ϕ_residual(dγ)

            #Not able to take Automatic derivative of ϕ_residual for some reaseon
            #Use numerical derivative for simplicity

            #jac = ForwardDiff.derivative(ϕ_residual, dγ)
            jac = (ϕ_residual(dγ + 0.00001) - ϕ_residual(dγ))/(0.00001)

            ddγ = jac\-res
            dγ += ddγ

            newton_error = norm(res)

            if counter > 10
                error("Could not find equilibrium in material")
            end
        end

        ϵᵖ += dγ
        F⁻¹ -= dγ*F⁻¹⋅ν

        _, Mdev, Stilde = ϕ(C, F⁻¹, ϵᵖ, μ, λ, τ₀, H)
    end

    dϕdM = Mdev/ (1e-15 + sum(sum(Mdev⋅Mdev)))
    ν = sqrt(3/2)*dϕdM
    S = symmetric(F⁻¹⋅Stilde⋅F⁻¹')

    return S, ϵᵖ, ν, F⁻¹
end

#Yield function
function ϕ(C, F⁻¹, ϵᵖ, μ, λ, τ₀, H)
    Cᵉ = F⁻¹'⋅C⋅F⁻¹

    Stilde =  μ*(one(Cᵉ) - inv(Cᵉ)) + λ*log(sqrt(det(Cᵉ))) * inv(Cᵉ)	#Neo Hook
    #Stilde = λ/2*(tr(Cᵉ)-3)*one(Cᵉ) + μ*(Cᵉ - one(Cᵉ)) 				#St Venant-Kirchhoff

    #Mandel stress
    Mbar = Cᵉ⋅Stilde
    Mdev = dev(Mbar)

    yield_func = sqrt(3/2)*norm(Mdev) - (τ₀ + H*ϵᵖ)
    
    return yield_func, Mdev, Stilde
end

function constitutive_driver(mp::Material, E, state::MaterialState)

	#Is it possible to combine the computation for S and ∂S∂E (using auto diff?)
    S, ϵᵖ, ν, F⁻¹ = _compute_2nd_PK(mp, E, state)
    ∂S∂E, _ = ∇(E -> _compute_2nd_PK(mp, E, state)[1], E, :all)

    return MaterialState(S, ϵᵖ, F⁻¹, ν), ∂S∂E
end

# Loop over all cells 
function assemble(grid::Grid{dim}, dh::DofHandler, K, f, cv, mp, states, u) where dim
    n = ndofs_per_cell(dh)
    Ke = zeros(n, n)
    fe = zeros(n)

    assembler = start_assemble(K, f)

    # loop over all cells in the grid
    @timeit "assemble" for cell in CellIterator(dh)
        # reset
        fill!(Ke, 0)
        fill!(fe, 0)

        global_dofs = celldofs(cell)
        ue = u[global_dofs] # element dofs

		cell_state = states[cell.current_cellid[]]

        @timeit "inner assemble"  assemble_element!(Ke, fe, cell, cv, mp, cell_state, ue)

        assemble!(assembler, global_dofs, fe, Ke)
    end

    return f, K
end

# Assembles the contribution from the cell to ke and fe
function assemble_element!(ke, fe, cell, cv, mp, state::Vector{MaterialState{T}}, ue) where T

    ndofs = getnbasefunctions(cv)
    reinit!(cv, cell)
    fill!(ke, 0.0)
    fill!(fe, 0.0)
    δE = zeros(SymmetricTensor{2, 3, eltype(ue), 6}, ndofs)

    for qp in 1:getnquadpoints(cv)
        ∇u = function_gradient(cv, qp, ue)
        dΩ = getdetJdV(cv, qp)

        # strain and stress + tangent
        F = one(∇u) + ∇u
        E = symmetric(1/2 * (F' ⋅ F - one(F)))

        state[qp], ∂S∂E = constitutive_driver(mp, E, state[qp])
        S = state[qp].S

        # Hoist computations of δE
        for i in 1:ndofs
            δFi = shape_gradient(cv, qp, i)
            δE[i] = symmetric(1/2*(δFi'⋅F + F'⋅δFi))
        end

        for i in 1:ndofs
            δFi = shape_gradient(cv, qp, i)
            δu = shape_value(cv, qp, i)
            fe[i] += (δE[i] ⊡ S) * dΩ
            δE∂S∂E = δE[i] ⊡ ∂S∂E
            S∇δu = S ⋅ δFi'
            for j in 1:ndofs
                δ∇uj = shape_gradient(cv, qp, j)
                ke[i, j] += (δE∂S∂E ⊡ δE[j] + S∇δu ⊡ δ∇uj' ) * dΩ
            end
        end
    end
end

function solve()
    reset_timer!()

    dim = 3

    # Generate a grid
    N = 6
    L = 1.0
    left = zero(Vec{dim})
    right = Vec{dim,Float64}((L*10,L,L))
    grid = generate_grid(Tetrahedron, (N*10,N,N), left, right)

    # finite element base
    ip = Lagrange{dim, RefTetrahedron, 1}()
    qr = QuadratureRule{dim, RefTetrahedron}(1)
    qr_face = QuadratureRule{dim-1, RefTetrahedron}(1)
    cv = CellVectorValues(qr, ip)

    # Material parameters
    E = 210.0
    ν = 0.3
    τ₀ = 0.5
    H = 1.5
    μ = E / (2(1 + ν))
    λ = (E * ν) / ((1 + ν) * (1 - 2ν))
    mp = Material{Float64}(μ, λ, τ₀, H)

    #Material state
    nqp = getnquadpoints(cv)
    states = [[MaterialState() for _ in 1:nqp] for _ in 1:getncells(grid)]
	prev_states = deepcopy(states)
    
    # DofHandler
    dh = DofHandler(grid)
    push!(dh, :u, dim) # Add a displacement field
    close!(dh)

    # Add a homogenoush boundary condition on the "clamped" edge
    dbcs = ConstraintHandler(dh)
    dbc = Dirichlet(:u, getfaceset(grid, "right"), (x,t) -> [0.0, 0.0, 0.0], collect(1:dim))
    add!(dbcs, dbc)
    dbc = Dirichlet(:u, getfaceset(grid, "left"), (x,t) -> t * Vec{dim}((L/10.0,L,L)), collect(1:dim))
    add!(dbcs, dbc)
    close!(dbcs)
    update!(dbcs, 0.0)

    println("Analysis with ", length(grid.cells), " elements")

    # pre-allocate
    _ndofs = ndofs(dh)
    un = zeros(_ndofs) # previous solution vector
    u  = zeros(_ndofs)
    Δu = zeros(_ndofs)
    apply!(un, dbcs)

    K = create_sparsity_pattern(dh)
    f = zeros(_ndofs)

    #Iteration and timesteping variables
    NEWTON_TOL = 1e-8
    ntimesteps = 30

    #Timestep loop
    for it in 1:ntimesteps

        println("Timestep: $it / $ntimesteps")

        #Update bc
        t = it/ntimesteps
        update!(dbcs, t)
        apply!(un, dbcs)

        newton_itr = -1
        #Equilibrium loop
        while true; newton_itr += 1
            u .= un .+ Δu
            f, K = assemble(grid, dh, K, f, cv, mp, states, u)
            
            normg = norm(f[JuAFEM.free_dofs(dbcs)])
            apply_zero!(K, f, dbcs)

            println("Normg: $normg")

            if normg < NEWTON_TOL
                break
            end

            if newton_itr > 30
                error("Reached maximum Newton iterations, aborting")
            end

            ΔΔu = K\f
            #@timeit "linear solve" ΔΔu, flag, relres, iter, resvec = cg(K, f; maxIter = 1000, tol = min(1e-3, normg))
            #@assert flag == 0

            apply_zero!(ΔΔu, dbcs)
            Δu .-= ΔΔu

            states = deepcopy(prev_states)
        end

        #Update variables for next iteration
        un .= u
        prev_states = deepcopy(states)
    end

    # save the solution
    @timeit "export" begin
        vtkfile = vtk_grid("hyper_elasto_plasticity", dh)
        vtk_point_data(vtkfile, dh, u)

        ϵᵖ_cells = [mean([qp.ϵᵖ for qp in state]) for state in states]
        my_cell_data = vtk_cell_data(vtkfile, ϵᵖ_cells, "plastic_strain")

        vtk_save(vtkfile)
    end

    print_timer(linechars = :ascii)
    return u
end

u = solve();
println("Finished")