# # [Voce plasticity with AD](@id tutorial-plasticity-ad)
#
# TODO update this
#-
#
# ## Introduction
#
# This example illustrates the use of a nonlinear material model in Ferrite.
# The particular model is von Mises plasticity (also know as J₂-plasticity) with
# isotropic hardening. The model is fully 3D, meaning that no assumptions like *plane stress*
# or *plane strain* are introduced.
#
# Also note that the theory of the model is not described here, instead one is
# referred to standard textbooks on material modeling.
#
# To illustrate the use of the plasticity model, we setup and solve a FE-problem
# consisting of a cantilever beam loaded at its free end. But first, we shortly
# describe the parts of the implementation dealing with the material modeling.

# ## Material modeling
# This section describes the `struct`s and methods used to implement the material
# model

# ### Material parameters and state variables
#
# Start by loading some necessary packages
using Ferrite, Tensors, SparseArrays, LinearAlgebra, Printf
using ForwardDiff, ImplicitDifferentiation, StaticArrays

# !!! todo
#     1. ImplicitDifferentiation does not support StaticArrays/Tensors yet, so we introduce temporary collect calls
#     2. I could not figure out how to pre-allocate buffers for ImplicitDifferentiation (analogue to ForwardDiff)

# We define a Voce-plasticity-material, precomputing the shear modulus G and the
# elastic stiffness tensor Dᵉ so they are not recomputed at every Gauss point.
struct J2VocePlasticity{T, S <: SymmetricTensor{4, 3, T}}
    G::T   # shear modulus
    σ₀::T
    Q::T
    b::T
    Dᵉ::S  # elastic stiffness tensor
end

function J2VocePlasticity(E::T, ν::T, σ₀::T, Q::T, b::T) where {T}
    G = E / (2(1 + ν))
    λ = E * ν / ((1 + ν) * (1 - 2ν))
    δ(i, j) = i == j ? one(T) : zero(T)
    Dᵉ = SymmetricTensor{4, 3}((i, j, k, l) -> λ * δ(i, j) * δ(k, l) + G * (δ(i, k) * δ(j, l) + δ(i, l) * δ(j, k)))
    return J2VocePlasticity(G, σ₀, Q, b, Dᵉ)
end

# Define a `struct` to store the material state for a Gauss point.
struct MaterialState{T, S1 <: SymmetricTensor{2, 3}, S2 <: SymmetricTensor{2, 3}}
    εᵖ::S1  # plastic strain
    σ::S2   # stress
    γ::T   # equivalent plastic strain
end

# Constructor for initializing a material state. Every quantity is set to zero.
MaterialState() = MaterialState(zero(SymmetricTensor{2, 3}), zero(SymmetricTensor{2, 3}), 0.0)
Base.zero(::MaterialState) = MaterialState()

# --- SVector interface for ImplicitDifferentiation ---
# ImplicitDifferentiation requires AbstractArray inputs/outputs. We use allocation-free
# SVector{6} for strains and SVector{13} for the full state (both in Mandel convention).
# Layout: [εᵖ(1:6), σ(7:12), γ(13)] with off-diagonal Mandel factor √2.

@inline function to_svec(A::SymmetricTensor{2, 3, T}) where {T}
    sq2 = T(√2)
    return SVector{6, T}(A[1, 1], A[2, 2], A[3, 3], sq2 * A[2, 3], sq2 * A[1, 3], sq2 * A[1, 2])
end

@inline function to_svec(state::MaterialState)
    e = to_svec(state.εᵖ)
    s = to_svec(state.σ)
    return SVector{13}(e[1], e[2], e[3], e[4], e[5], e[6], s[1], s[2], s[3], s[4], s[5], s[6], state.γ)
end

@inline function from_svec(::Type{SymmetricTensor{2, 3}}, sv::AbstractVector{T}) where {T}
    isq2 = inv(T(√2))
    return SymmetricTensor{2, 3, T}((sv[1], isq2 * sv[6], isq2 * sv[5], sv[2], isq2 * sv[4], sv[3]))
end

@inline function from_svec(::Type{MaterialState}, sv::AbstractVector{T}) where {T}
    isq2 = inv(T(√2))
    εᵖ = SymmetricTensor{2, 3, T}((sv[1], isq2 * sv[6], isq2 * sv[5], sv[2], isq2 * sv[4], sv[3]))
    σ = SymmetricTensor{2, 3, T}((sv[7], isq2 * sv[12], isq2 * sv[11], sv[8], isq2 * sv[10], sv[9]))
    return MaterialState(εᵖ, σ, sv[13])
end

# For later use, during the post-processing step, we define a function to
# compute the von Mises effective stress.
function vonMises(σ::SymmetricTensor{2, 3})
    s = dev(σ)
    return sqrt(3 / 2 * s ⊡ s)
end;

# ## FE-problem
# What follows are methods for assembling and solving the FE-problem.
function create_values(interpolation)
    ## setup quadrature rules
    qr = QuadratureRule{RefHexahedron}(2)
    facet_qr = FacetQuadratureRule{RefHexahedron}(3)

    ## cell and facetvalues for u
    cv_u = CellValues(qr, interpolation)
    facetvalues_u = FacetValues(facet_qr, interpolation)

    return cv_u, facetvalues_u
end;

# ### Add degrees of freedom
function create_dofhandler(grid, interpolation)
    dh = DofHandler(grid)
    add!(dh, :u, interpolation) # add a displacement field with 3 components
    close!(dh)
    return dh
end

# ### Boundary conditions
function create_bc(dh, grid)
    dbcs = ConstraintHandler(dh)
    ## Clamped on the left side
    dofs = [1, 2, 3]
    dbc = Dirichlet(:u, getfacetset(grid, "left"), (x, t) -> [0.0, 0.0, 0.0], dofs)
    add!(dbcs, dbc)
    close!(dbcs)
    return dbcs
end;


# ### Assembling of element contributions
#
# * Residual vector `r`
# * Tangent stiffness `K`
function doassemble!(
        K::SparseMatrixCSC, r::Vector, cv::CellValues, dh::DofHandler,
        material::J2VocePlasticity, u, states, states_old
    )
    assembler = start_assemble(K, r)
    nu = getnbasefunctions(cv)
    re = zeros(nu)     # element residual vector
    ke = zeros(nu, nu) # element tangent matrix

    for (i, cell) in enumerate(CellIterator(dh))
        fill!(ke, 0)
        fill!(re, 0)
        eldofs = celldofs(cell)
        ue = u[eldofs]
        state = @view states[:, i]
        state_old = @view states_old[:, i]
        residual! = (re, ue) -> assemble_cell!(re, cell, cv, material, ue, state, state_old)
        ke .= ForwardDiff.jacobian(residual!, re, ue)
        assemble!(assembler, eldofs, ke, re)
    end
    return K, r
end


# 1. Forward Pass: Computes updated state using a 1D Local Newton solver for Δγ.
# Input/output use SVector (Mandel convention, stack-allocated) for ImplicitDifferentiation.
# All physics is done with SymmetricTensor objects.
# State SVector layout: [εᵖ(1:6), σ(7:12), γ(13)]
function local_forward(ε_sv::SVector{6}, old_state_sv::SVector{13}, material)
    old_state = from_svec(MaterialState, old_state_sv)
    εᵖ_old = old_state.εᵖ
    γ_old = old_state.γ
    (; G, σ₀, Q, b, Dᵉ) = material

    ## Elastic trial stress
    ε = from_svec(SymmetricTensor{2, 3}, ε_sv)
    σ_trial = Dᵉ ⊡ (ε - εᵖ_old)
    s_trial = dev(σ_trial)
    σ_eq_trial = sqrt(3 / 2 * s_trial ⊡ s_trial)
    σ_y_old = σ₀ + Q * (1 - exp(-b * γ_old))

    if σ_eq_trial - σ_y_old <= 1.0e-10 * σ_eq_trial
        ## Elastic step
        return collect(to_svec(MaterialState(εᵖ_old, σ_trial, γ_old))), false
    else
        ## Plastic step: local Newton for Δγ
        Δγ = zero(σ_eq_trial)
        tol = 1.0e-9 * σ_eq_trial

        ## This is the residual derived from the yield surface
        residual(Δγ) = σ_eq_trial - 3G * Δγ - (σ₀ + Q * (1 - exp(-b * (γ_old + Δγ))))
        for iter in 1:30
            R = residual(Δγ)
            dR = ForwardDiff.derivative(residual, Δγ)
            Δγ -= R / dR
            abs(R) < tol && break
            iter == 30 && error("local newton diverged (R=$R)")
        end

        ## Flow direction (unit deviatoric normal)
        n = s_trial / sqrt(s_trial ⊡ s_trial + 1.0e-20)

        Δεᵖ = sqrt(3 / 2) * Δγ * n
        σ = σ_trial - 2G * Δεᵖ
        εᵖ = εᵖ_old + Δεᵖ
        γ = γ_old + Δγ

        return collect(to_svec(MaterialState(εᵖ, σ, γ))), true
    end
end

# 2. Conditions: Physical residuals governing equilibrium.
# Residual SVector layout matches state: [εᵖ_res(1:6), σ_res(7:12), Φ_or_γ_res(13)]
function local_conditions(ε_sv::SVector{6}, state_sv::AbstractVector, is_plastic, old_state_sv::AbstractVector, material)
    state = from_svec(MaterialState, state_sv)
    old_state = from_svec(MaterialState, old_state_sv)
    εᵖ = state.εᵖ;  σ = state.σ;  γ = state.γ
    εᵖ_old = old_state.εᵖ;  γ_old = old_state.γ
    (; G, σ₀, Q, b, Dᵉ) = material
    ε = from_svec(SymmetricTensor{2, 3}, ε_sv)

    if !is_plastic
        εᵖ_res = εᵖ - εᵖ_old
        σ_res = σ - Dᵉ ⊡ (ε - εᵖ_old)
        γ_res = γ - γ_old
    else
        s = dev(σ)
        n = s / sqrt(s ⊡ s + 1.0e-20)
        εᵖ_res = εᵖ - εᵖ_old - sqrt(3 / 2) * (γ - γ_old) * n
        σ_res = σ - Dᵉ ⊡ (ε - εᵖ)
        σ_eq = sqrt(3 / 2 * s ⊡ s)
        γ_res_val = σ_eq - (σ₀ + Q * (1 - exp(-b * γ)))
    end

    e = to_svec(εᵖ_res)
    r = to_svec(σ_res)
    γ_scalar = is_plastic ? γ_res_val : γ_res
    sv = SVector{13}(e[1], e[2], e[3], e[4], e[5], e[6], r[1], r[2], r[3], r[4], r[5], r[6], γ_scalar)
    return collect(sv)
end

material_update = ImplicitFunction(
    local_forward,
    local_conditions;
    linear_solver = DirectLinearSolver(),
    representation = MatrixRepresentation()
)

function stress_function(ε::SymmetricTensor{2, 3}, old_state::MaterialState, material)
    ε_sv = to_svec(ε)
    old_sv = to_svec(old_state)
    result_sv, is_plastic = material_update(ε_sv, old_sv, material)
    return from_svec(MaterialState, result_sv), is_plastic
end

function assemble_cell!(re, cell, cv, material, ue::AbstractVector{T}, state_new, state_old) where {T}
    n_basefuncs = getnbasefunctions(cv)
    reinit!(cv, cell)

    for q_point in 1:getnquadpoints(cv)
        dΩ = getdetJdV(cv, q_point)

        ## 1. Strain from Ferrite (SymmetricTensor, no conversion needed)
        ε = function_symmetric_gradient(cv, q_point, ue)

        ## 2. Differentiable stress-from-strain closure (SVector → SVector for AD)
        old_sv = to_svec(state_old[q_point])
        function stress_from_strain(ε_sv::AbstractVector)
            result_sv, _ = material_update(SVector{6}(ε_sv), old_sv, material)
            return result_sv[SVector{6}(7, 8, 9, 10, 11, 12)]  # σ components
        end

        ε_sv = to_svec(ε)

        ## 3. Stress and algorithmic tangent via AD (single primal + one Jacobian pass)
        σ_sv = stress_from_strain(ε_sv)

        ## 4. Store new state (re-uses the primal solve result)
        result_sv, _ = material_update(ε_sv, old_sv, material)
        state_new[q_point] = from_svec(MaterialState, ForwardDiff.value.(result_sv))

        ## 5. Recover stress as SymmetricTensor for inner products
        σ = from_svec(SymmetricTensor{2, 3}, σ_sv)

        ## 6. Element integration using tensor inner products (no Mandel arrays needed)
        for i in 1:n_basefuncs
            ∇δN = shape_symmetric_gradient(cv, q_point, i)
            re[i] += (∇δN ⊡ σ) * dΩ
        end
    end

    return nothing
end

function doassemble_neumann!(r, dh, facetset, facetvalues, t)
    n_basefuncs = getnbasefunctions(facetvalues)
    re = zeros(n_basefuncs)                      # element residual vector
    for fc in FacetIterator(dh, facetset)
        ## Add traction as a negative contribution to the element residual `re`:
        reinit!(facetvalues, fc)
        fill!(re, 0)
        for q_point in 1:getnquadpoints(facetvalues)
            dΓ = getdetJdV(facetvalues, q_point)
            for i in 1:n_basefuncs
                δu = shape_value(facetvalues, q_point, i)
                re[i] -= (δu ⋅ t) * dΓ
            end
        end
        assemble!(r, celldofs(fc), re)
    end
    return r
end

# Define a function which solves the FE-problem.
function solve()
    ## Define material parameters
    E = 200.0e9  # [Pa]
    Q = E / 20   # [Pa]
    ν = 0.3      # [-]
    σ₀ = 1.0e6 # [Pa]
    b = 2.0     # [-]
    material = J2VocePlasticity(E, ν, σ₀, Q, b)

    L = 10.0 # beam length [m]
    w = 1.0  # beam width [m]
    h = 1.0  # beam height[m]
    n_timesteps = 3
    u_max = zeros(n_timesteps)
    traction_magnitude = 1.0e6 * range(0.0, 1.0, length = n_timesteps)

    ## Create geometry, dofs and boundary conditions
    n = 2
    nels = (10n, n, 2n) # number of elements in each spatial direction
    P1 = Vec((0.0, 0.0, 0.0))  # start point for geometry
    P2 = Vec((L, w, h))        # end point for geometry
    grid = generate_grid(Hexahedron, nels, P1, P2)
    interpolation = Lagrange{RefHexahedron, 1}()^3

    dh = create_dofhandler(grid, interpolation) # helper function defined above
    dbcs = create_bc(dh, grid) # create Dirichlet boundary-conditions

    cv, facetvalues = create_values(interpolation)

    ## Pre-allocate solution vectors, etc.
    n_dofs = ndofs(dh)  # total number of dofs
    u = zeros(n_dofs)   # solution vector
    Δu = zeros(n_dofs)  # displacement correction
    r = zeros(n_dofs)   # residual
    K = allocate_matrix(dh) # tangent stiffness matrix

    ## Create material states. One array for each cell, where each element is an array of material-
    ## states - one for each integration point
    nqp = getnquadpoints(cv)
    states = [MaterialState() for _ in 1:nqp, _ in 1:getncells(grid)]
    states_old = [MaterialState() for _ in 1:nqp, _ in 1:getncells(grid)]

    ## Newton-Raphson loop
    NEWTON_TOL = 1 # 1 N
    print("\n Starting Newton iterations:\n")

    for timestep in 1:n_timesteps
        t = timestep # actual time (used for evaluating d-bndc)
        traction = Vec((0.0, 0.0, traction_magnitude[timestep]))
        newton_itr = -1
        print("\n Time step @time = $timestep:\n")
        update!(dbcs, t) # evaluates the D-bndc at time t
        apply!(u, dbcs)  # set the prescribed values in the solution vector

        while true
            newton_itr += 1
            if newton_itr > 8
                error("Reached maximum Newton iterations, aborting")
                break
            end
            ## Tangent and residual contribution from the cells (volume integral)
            doassemble!(K, r, cv, dh, material, u, states, states_old)
            ## Residual contribution from the Neumann boundary (surface integral)
            doassemble_neumann!(r, dh, getfacetset(grid, "right"), facetvalues, traction)
            norm_r = norm(r[Ferrite.free_dofs(dbcs)])

            print("Iteration: $newton_itr \tresidual: $(@sprintf("%.8f", norm_r))\n")
            if norm_r < NEWTON_TOL
                break
            end

            apply_zero!(K, r, dbcs)
            Δu = K \ r
            u -= Δu
        end

        ## Update the old states with the converged values for next timestep
        states_old .= states

        u_max[timestep] = maximum(abs, u) # maximum displacement in current timestep
    end

    ## ## Postprocessing
    ## Only a vtu-file corresponding to the last time-step is exported.
    ##
    ## The following is a quick (and dirty) way of extracting average cell data for export.
    mises_values = zeros(getncells(grid))
    κ_values = zeros(getncells(grid))
    for (el, cell_states) in enumerate(eachcol(states))
        for state in cell_states
            mises_values[el] += vonMises(state.σ)
            # κ_values[el] += state.k * material.H
        end
        mises_values[el] /= length(cell_states) # average von Mises stress
        κ_values[el] /= length(cell_states)     # average drag stress
    end
    VTKGridFile("plasticity", dh) do vtk
        write_solution(vtk, dh, u) # displacement field
        write_cell_data(vtk, mises_values, "von Mises [Pa]")
        write_cell_data(vtk, κ_values, "Drag stress [Pa]")
    end

    return u_max, traction_magnitude
end

# Solve the FE-problem and for each time-step extract maximum displacement and
# the corresponding traction load. Also compute the limit-traction-load
u_max, traction_magnitude = solve();

# Finally we plot the load-displacement curve.
using Plots
plot(
    vcat(0.0, u_max),                # add the origin as a point
    vcat(0.0, traction_magnitude),
    linewidth = 2,
    title = "Traction-displacement",
    label = nothing,
    markershape = :auto
)
ylabel!("Traction [Pa]")
xlabel!("Maximum deflection [m]")

# *Figure 2.* Load-displacement-curve for the beam, showing a clear decrease
# in stiffness as more material starts to yield.

## test the result                       #src
using Test                               #src
@test norm(u_max[end]) ≈ 0.15756125526017042  #src

#md # ## [Plain program](@id plasticity-plain-program)
#md #
#md # Here follows a version of the program without any comments.
#md # The file is also available here: [`plasticity.jl`](plasticity.jl).
#md #
#md # ```julia
#md # @__CODE__
#md # ```
