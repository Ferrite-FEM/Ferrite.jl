# # [Von Mises plasticity](@id tutorial-plasticity)
#
# ![Shows the von Mises stress distribution in a cantilever beam.](plasticity.png)
#
# *Figure 1.* A coarse mesh solution of a cantilever beam subjected to a load
# causing plastic deformations. The initial yield limit is 200 MPa but due to
# hardening it increases up to approximately 240 MPa.
#
#-
#md # !!! tip
#md #     This example is also available as a Jupyter notebook:
#md #     [`plasticity.ipynb`](@__NBVIEWER_ROOT_URL__/tutorials/plasticity.ipynb).
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
using ForwardDiff, ImplicitDifferentiation

# We define a Voce-plasticity-material, containing material parameters and the elastic
# stiffness Dᵉ (since it is constant)
struct J2VocePlasticity{T}
    E::T
    ν::T
    σ₀::T
    Q::T
    b::T
end

# Define a `struct` to store the material state for a Gauss point.
struct MaterialState{T, S}
    ## Store "converged" values
    εᵖ::S # plastic strain
    σ::S # stress
    γ::T # effective plastic strain
end

# Helper function to convert MaterialState to flat array
function to_array(state::MaterialState)
    return [Tensors.tomandel(state.εᵖ)..., Tensors.tomandel(state.σ)..., state.γ]
end

# Helper function to convert flat array to MaterialState
function from_array(arr::AbstractVector)
    εᵖ = Tensors.frommandel(SymmetricTensor{2,3}, arr[1:6])
    σ = Tensors.frommandel(SymmetricTensor{2,3}, arr[7:12])
    γ = arr[13]
    return MaterialState(εᵖ, σ, γ)
end

# Constructor for initializing a material state. Every quantity is set to zero.
function MaterialState()
    return MaterialState(
        zero(SymmetricTensor{2, 3}),
        zero(SymmetricTensor{2, 3}),
        0.0
    )
end
Base.zero(::MaterialState) = MaterialState()

# For later use, during the post-processing step, we define a function to
# compute the von Mises effective stress.
function vonMises(σ)
    s = dev(SymmetricTensor{2,3}(σ))
    return sqrt(3.0 / 2.0 * s ⊡ s)
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
        assemble_cell!(ke, re, cell, cv, material, ue, state, state_old)
        assemble!(assembler, eldofs, ke, re)
    end
    return K, r
end


function elastic_stiffness_mandel(E, ν)
    λ = E * ν / ((1.0 + ν) * (1.0 - 2.0 * ν))
    μ = E / (2.0 * (1.0 + ν))
    T = promote_type(typeof(E), typeof(ν))
    D = zeros(T, 6, 6)

    # Normal components
    D[1,1] = D[2,2] = D[3,3] = λ + 2μ
    # Shear components (Mandel scale factor makes these 2μ instead of μ)
    D[4,4] = D[5,5] = D[6,6] = 2μ
    # Off-diagonals
    D[1,2] = D[1,3] = D[2,1] = D[2,3] = D[3,1] = D[3,2] = λ
    return D
end

# Compute Von Mises Equivalent Stress in Mandel Space
function von_mises_mandel(σ_mandel)
    σ_m = sum(σ_mandel[1:3]) / 3.0
    s = [σ_mandel[1]-σ_m, σ_mandel[2]-σ_m, σ_mandel[3]-σ_m, σ_mandel[4], σ_mandel[5], σ_mandel[6]]
    s_norm = sqrt(dot(s, s) + 1e-20) # Regularized for AD safety at zero stress
    return sqrt(1.5) * s_norm
end

# 1. Forward Pass: Computes updated state using a 1D Local Newton solver for Δγ
# Works entirely in Mandel array form to avoid SymmetricTensor indexing pitfalls.
# state_array layout: [εᵖ_mandel(1:6), σ_mandel(7:12), γ(13)]  (matches to_array/from_array)
function local_forward(ε_mandel, old_state_array, material)
    εᵖ_old_mandel = old_state_array[1:6]
    γ_old = old_state_array[13]
    (; E, ν, σ₀, Q, b) = material

    G = E / (2.0 * (1.0 + ν))
    D_e = elastic_stiffness_mandel(E, ν)

    # Elastic Trial State (all in Mandel)
    σ_trial_mandel = D_e * (ε_mandel - εᵖ_old_mandel)
    σ_eq_trial = von_mises_mandel(σ_trial_mandel)
    σ_y_old = σ₀ + Q * (1.0 - exp(-b * γ_old))

    if σ_eq_trial - σ_y_old <= 1e-10 * σ_eq_trial # Relative tolerance
        # Elastic: state unchanged except stress update
        return [εᵖ_old_mandel..., σ_trial_mandel..., γ_old], false
    else
        # Plastic Step: Solve Φ(Δγ) = 0 via Local Newton-Raphson
        Δγ = 0.0
        tol = 1e-9 * σ_eq_trial
        for iter in 1:30
            γ = γ_old + Δγ
            σ_y = σ₀ + Q * (1.0 - exp(-b * γ))
            dσ_y = Q * b * exp(-b * γ)

            R = σ_eq_trial - 3.0 * G * Δγ - σ_y
            dR = -3.0 * G - dσ_y

            Δγ -= R / dR
            if abs(R) < tol
                break
            end
            if iter == 30
                error("local newton diverged (R=$R)")
            end
        end

        # Flow direction in Mandel form
        σ_m_trial = sum(σ_trial_mandel[1:3]) / 3.0
        s_trial = [σ_trial_mandel[1]-σ_m_trial, σ_trial_mandel[2]-σ_m_trial, σ_trial_mandel[3]-σ_m_trial,
                   σ_trial_mandel[4], σ_trial_mandel[5], σ_trial_mandel[6]]
        n = s_trial ./ sqrt(dot(s_trial, s_trial) + 1e-20)

        # Updates (all in Mandel; D_e:Δεᵖ = 2G*Δεᵖ for deviatoric Δεᵖ)
        Δεᵖ_mandel = sqrt(1.5) * Δγ .* n
        σ_mandel   = σ_trial_mandel .- 2.0 * G .* Δεᵖ_mandel
        εᵖ_mandel  = εᵖ_old_mandel  .+ Δεᵖ_mandel
        γ = γ_old + Δγ

        return [εᵖ_mandel..., σ_mandel..., γ], true
    end
end

# 2. Conditions: The exact physical residuals governing the system at equilibrium.
# Works entirely in Mandel array form.
# Residual ordering matches state_array: [εᵖ_eq(1:6), σ_eq(7:12), γ_eq(13)]
function local_conditions(ε_mandel, state_array, is_plastic, old_state_array, material)
    εᵖ_mandel     = state_array[1:6]
    σ_mandel      = state_array[7:12]
    γ             = state_array[13]
    εᵖ_old_mandel = old_state_array[1:6]
    γ_old         = old_state_array[13]

    (; E, ν, σ₀, Q, b) = material
    D_e = elastic_stiffness_mandel(E, ν)

    if !is_plastic
        # Elastic residuals: εᵖ = εᵖ_old, σ = D_e:(ε - εᵖ_old), γ = γ_old
        εᵖ_res = εᵖ_mandel .- εᵖ_old_mandel
        σ_res  = σ_mandel  .- D_e * (ε_mandel .- εᵖ_old_mandel)
        γ_res  = γ - γ_old
        return [εᵖ_res..., σ_res..., γ_res]
    else
        # Plastic residuals: flow rule, constitutive, yield condition
        σ_m = sum(σ_mandel[1:3]) / 3.0
        s = [σ_mandel[1]-σ_m, σ_mandel[2]-σ_m, σ_mandel[3]-σ_m,
             σ_mandel[4], σ_mandel[5], σ_mandel[6]]
        n = s ./ sqrt(dot(s, s) + 1e-20)

        flow_res  = εᵖ_mandel .- εᵖ_old_mandel .- sqrt(1.5) * (γ - γ_old) .* n
        σ_res     = σ_mandel  .- D_e * (ε_mandel .- εᵖ_mandel)
        yield_res = von_mises_mandel(σ_mandel) - (σ₀ + Q * (1.0 - exp(-b * γ)))
        return [flow_res..., σ_res..., yield_res]
    end
end

material_update = ImplicitFunction(
    local_forward,
    local_conditions;
    linear_solver = DirectLinearSolver(),
    representation = MatrixRepresentation()
)

function stress_function(ϵ_mandel, state, material)
    # Convert state to array for ImplicitFunction
    state_array = to_array(state)
    result_array, is_plastic = material_update(ϵ_mandel, state_array, material)
    # Convert result back to struct
    result_state = from_array(result_array)
    return result_state, is_plastic
end

function assemble_cell!(Ke, re, cell, cv, material, ue, state_new, state_old)
    n_basefuncs = getnbasefunctions(cv)
    reinit!(cv, cell)

    for q_point in 1:getnquadpoints(cv)
        dΩ = getdetJdV(cv, q_point)

        # 1. Strain calculation from Ferrite
        ε = function_symmetric_gradient(cv, q_point, ue)
        ε_mandel = Tensors.tomandel(ε)

        # 2. AD Wrapper: ε_mandel (6) -> σ_mandel (6)
        function stress_from_strain(strain_mandel)
            (out, _) = stress_function(strain_mandel, state_old[q_point], material)
            return Tensors.tomandel(out.σ)
        end

        # 3. Extract exact stress and algorithmic tangent (D_mandel) via AD
        σ_mandel = stress_from_strain(ε_mandel)
        D_mandel = ForwardDiff.jacobian(stress_from_strain, ε_mandel)

        # 4. Store new internal states
        (out_full, _) = stress_function(ε_mandel, state_old[q_point], material)
        state_new[q_point] = out_full

        # 6. Element Integration using Mandel dot products
        for i in 1:n_basefuncs
            ∇δN = shape_symmetric_gradient(cv, q_point, i)
            ∇δN_mandel = Tensors.tomandel(∇δN)

            re[i] += dot(∇δN_mandel, σ_mandel) * dΩ

            for j in 1:n_basefuncs
                ∇N = shape_symmetric_gradient(cv, q_point, j)
                ∇N_mandel = Tensors.tomandel(∇N)

                # Perfect representation of virtual work: δW_int = ∫ ∇δN : D : ∇N dΩ
                Ke[i, j] += dot(∇δN_mandel, D_mandel * ∇N_mandel) * dΩ
            end
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
    b  = 2.0     # [-]
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
