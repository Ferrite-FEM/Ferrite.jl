# # Porous media 

# Porous media is a two-phase material, consisting of solid parts and a liquid occupying
# the pores inbetween. 
# Using the porous media theory, we can model such a material without explicitly 
# resolving the microstructure, but by considering the interactions between the 
# solid and liquid. In this example, we will additionally consider larger linear 
# elastic solid aggregates that are impermeable. Hence, there is no liquids in 
# these particles and the only unknown variable is the displacement field `:u`. 
# In the porous media, denoted the matrix, we have both the displacement field,
# `:u`, as well as the liquid pressure, `:p`, as unknown. The computational domain
# is shown below
#
# ![The porous media consisting of impermeable solid aggregates and a permeable porous matrix.](porous_media.svg)
# ![Pressure evolution.](porous_media.gif)
#
# ## Theory of porous media
# The strong forms for the mass balance of the liquid is given as
# ```math
#    \frac{\mathrm{d}_\mathrm{s} n \rho_\mathrm{l}}{\mathrm{d}t} 
#    + \mathrm{div}\left(n \rho_\mathrm{l} \tilde{\boldsymbol{v}}_\mathrm{l}\right)
#    + n \rho_\mathrm{l} \mathrm{tr}(\boldsymbol{\dot{\epsilon}}) = 0
# ```
# where ``\mathrm{d}_\mathrm{s}/\mathrm{d}t`` is the time of a quantity following the 
# solid skeleton motion, described by the displacements ``\boldsymbol{u}``. ``n`` is the 
# porosity (i.e. volume fraction of pores, assumed constant due to small strains), 
# ``\rho_\mathrm{l}`` is the liquid density,
# ``\tilde{\boldsymbol{v}}_\mathrm{l}`` is the liquid velocity relative to the solid skeleton
# motion, and ``\boldsymbol{\epsilon}`` is the strain tensor for the solid skeleton, 
# ``\boldsymbol{\epsilon}=\left[\mathrm{grad}(\boldsymbol{u})\right]^\mathrm{sym}``. 
# The functions ``\mathrm{div}()`` and ``\mathrm{grad}()`` represent the divergence and 
# gradient, respectively. Furthermore, the balance of momentum is given as 
# ```math
#    \mathrm{div}(\boldsymbol{\sigma}) = \boldsymbol{0}
# ```
# where ``\boldsymbol{\sigma}`` is the Cauchy stress. For simplicity in this example, 
# body loads, e.g. due to gravity, are not included. 
#
# ### Constitutive equations
# Darcy's law, excluding gravity loads, is used for the fluid flow through the porous media
# ```math
#    n \tilde{\boldsymbol{v}}_\mathrm{l} = -[k/\mu] \mathrm{grad}(p)
# ```
# A constant fluid bulk modulus, ``K_\mathrm{l}``, gives the relationship between fluid 
# pressure, ``p``, and density, ``\rho_\mathrm{l}``, as 
# ```math
# \dot{\rho}_\mathrm{l} = \frac{\rho_\mathrm{l}}{K_\mathrm{l}} \dot{p}
# ```
# Finally, we use the most simple Terzaghi effective stress combined with linear 
# isotropic elasticity
# ```math
# \boldsymbol{\sigma} = \boldsymbol{\mathrm{E}}:\boldsymbol{\epsilon} - p\boldsymbol{I} = 2G \boldsymbol{\epsilon}^\mathrm{dev} + 3 K \boldsymbol{\epsilon}^\mathrm{vol} - p \boldsymbol{I}
# ```
# ### Weak form
# From the above strong form with constitutive equations (and including boundary conditions), 
# we obtain the following weak forms for the mass balance,
# ```math
#   \int_\Gamma \delta p \tilde{\boldsymbol{v}}_\mathrm{l} \cdot \boldsymbol{n} \mathrm{d}\Gamma = 
#   \int_\Omega \mathrm{grad}(\delta p) \cdot \mathrm{grad}(p) [k/n] \mathrm{d} \Omega +
#   \int_\Omega \delta p \left[\dot{p}/K_\mathrm{l} + \mathrm{div}\left(\dot{\boldsymbol{u}}\right)\right] \mathrm{d}\Omega
# ```
# and for the momentum balance
# ```math
#   \int_\Gamma \boldsymbol{\delta u} \cdot \boldsymbol{t} \mathrm{d} \Gamma = 
#   \int_\Omega \mathrm{grad}\left(\boldsymbol{\delta u}\right) : \left[ \boldsymbol{\mathrm{E}} : \mathrm{grad}(\boldsymbol{u}) - p \boldsymbol{I}\right] \mathrm{d} \Omega
# ```
# ### Finite element form
# Discretizing in space using finite elements, we obtain the matrix equation 
# ``f_{i}^\mathrm{ext}=K_{ij} a_j + L_{ij} \dot{a}_j`` where ``f^\mathrm{ext}`` are the external 
# "forces", ``K`` the stiffness matrix, ``a`` the unknown degrees of freedom, ``L`` the 
# (dampening) matrix that is multiplied with the rate of the unknown degrees of freedom. 
# For each relevant part, we can specify these matrices and vectors as 
# ```math
# \begin{align*}
# K_{ij}^\mathrm{p} &= \int_\Omega \mathrm{grad}\left(\delta N\supscr{p}_i\right)\cdot \left[\frac{k}{n}\mathrm{grad}\left(N\supscr{p}_j\right)\right] \mathrm{d}\Omega \\
# L_{ij}^\mathrm{pp} &= \int_\Omega \delta N_i^\mathrm{p} N_j^\mathrm{p}/K_\mathrm{l} \mathrm{d}\Omega \\
# L_{ij}^\mathrm{pu} &= \int_\Omega \delta N_i^\mathrm{p} \mathrm{div}\left(\boldsymbol{N}_j^\mathrm{u}\right) \mathrm{d}\Omega \\
# K_{ij}^\mathrm{uu} &= - \int_\Omega \mathrm{grad}\left(\boldsymbol{\delta N}^\mathrm{u}_i\right) : \boldsymbol{\mathrm{E}} : \mathrm{grad}\left(\boldsymbol{N}_j^\mathrm{u}\right) \mathrm{d} \Omega \\ 
# K_{ij}^\mathrm{up} &= \int_\Omega \mathrm{div}\left(\boldsymbol{\delta N}_i^\mathrm{u}\right) N_j^\mathrm{p} \mathrm{d}\Omega \\
# f_{i}^\mathrm{p,ext} &= \int_\Gamma \delta N_i^\mathrm{p} \tilde{\boldsymbol{v}}_\mathrm{l} \cdot \boldsymbol{n} \mathrm{d}\Gamma\\
# f_{i}^\mathrm{u,ext} &= -\int_\Gamma \boldsymbol{\delta N}_i^\mathrm{u} \cdot \boldsymbol{t} \mathrm{d} \Gamma
# \end{align*}
# ```
# This results in the equation system 
# ```math
# \begin{align*}
# f_{i}^\mathrm{p,ext} &= K_{ij}^\mathrm{pp} a_{j}^\mathrm{p} + L_{ij}^\mathrm{pp} \dot{a}_j^\mathrm{p} + L_{ij}^\mathrm{pu} \dot{a}_j^\mathrm{u} \\
# f_{j}^\mathrm{u,ext} &= K_{ij}^\mathrm{up} a_{j}^\mathrm{p} + K_{ij}^\mathrm{uu} a_j^\mathrm{u}
# \end{align*}
# ```
# where the subscripts ``\mathrm{p}`` and ``\mathrm{u}`` gives the part of the vector pertinent to that
# degree of freedom (pressure or displacement). The time discretized form of the above equation becomes 
# ```math
# \begin{align*}
# \Delta t f_{i}^\mathrm{p,ext} + L_{ij}^\mathrm{pp} a_j^\mathrm{p,old} + L_{ij}^\mathrm{pu} a_j^\mathrm{u,old} &= \Delta t K_{ij}^\mathrm{pp} a_{j}^\mathrm{p} + L_{ij}^\mathrm{pp} a_j^\mathrm{p} + L_{ij}^\mathrm{pu} a_j^\mathrm{u} \\
# f_{j}^\mathrm{u,ext} &= K_{ij}^\mathrm{up} a_{j}^\mathrm{p} + K_{ij}^\mathrm{uu} a_j^\mathrm{u}
# \end{align*}
# ```
# As the matrices are constant, it suffices to assemble them once and reuse for each time step. However, in this example we assemble in each time step,
# calculating only one stiffness matrix. The contributions from the old values, on the left hand side, are considered as external loads. 
# This avoids having two global matrices, simplifying the present example and making it more suitable to consider
# nonlinear problems. With all the theory completed, let's start implementing a solution to this problem in Ferrite. 
# Material parameters are hard-coded in for simplicity. 
# 
# ## Implementation
# We now solve the problem step by step. The full program with fewer comments is found in 
#md # the final [section](@ref porous-media-plain-program)
# 
# Required packages
using Ferrite, FerriteMeshParser, Tensors
#
# ### Elasticity
# We then define the stiffness tensor ``\boldsymbol{\mathrm{E}}`` for linear isotropic elasticity
# and plane strain
function elastic_stiffness(E=20.e3, ν=0.3)
    G = E / 2(1 + ν)
    K = E / 3(1 - 2ν)
    I2 = one(SymmetricTensor{2,2})
    I4vol = I2⊗I2
    I4dev = minorsymmetric(otimesu(I2,I2)) - I4vol / 3
    return 2G*I4dev + K*I4vol
end

# ### Element routines
# Next, we define the element routines. First for the case of the linear elastic impermeable solid aggregates
# In this simplified example, we dispatch on the cellvalues which are only `CellVectorValues` for the 
# solid aggregates. Note that the unused inputs here are used for the porous matrix below. 
function element_routine!(Ke, _, cell, _, cv::CellVectorValues, _, _)
    reinit!(cv, cell)
    n_basefuncs = getnbasefunctions(cv)
    fill!(Ke, 0)
    for q_point in 1:getnquadpoints(cv)
        dσdϵ = elastic_stiffness()
        dΩ = getdetJdV(cv, q_point)
        for i in 1:n_basefuncs
            δ∇N = shape_symmetric_gradient(cv, q_point, i)
            for j in 1:n_basefuncs 
                ∇N = shape_symmetric_gradient(cv, q_point, j)
                Ke[i, j] += δ∇N ⊡ dσdϵ ⊡ ∇N * dΩ
            end
        end
    end
end

# Due to the two fields, `:p` and `:u`, as well as more physics, in the porous matrix, 
# we require a few more inputs to get the element force and stiffness. We dispatch 
# on the cellvalues being a tuple of `CellVectorValues` and `CellScalarValues`.
function element_routine!(Ke, fext, cell, fh::FieldHandler, cvs::Tuple{CellVectorValues, CellScalarValues}, a_old, Δt)
    ## Setup cellvalues and give easier names
    reinit!.(cvs, (cell,))
    cv_u, cv_p = cvs
    num_u_basefuncs, num_p_basefuncs = getnbasefunctions.(cvs)
    
    ## Check that cellvalues are compatible with each other (should have same quadrature rule)
    @assert getnquadpoints(cv_u) == getnquadpoints(cv_p)

    ## Reset element stiffness and external force
    fill!(Ke, 0.0)
    fill!(fext, 0.0)

    ## Assign views to the matrix and vector parts
    udofs = dof_range(fh, :u)
    pdofs = dof_range(fh, :p)
    Kuu = @view Ke[udofs, udofs]
    Kpu = @view Ke[pdofs, udofs]
    Kup = @view Ke[udofs, pdofs]
    Kpp = @view Ke[pdofs, pdofs]
    ## fu = @view fext[udofs]    # Not used, traction is zero or displacements prescribed
    fp = @view fext[pdofs]
    au_old = @view a_old[udofs]
    ap_old = @view a_old[pdofs]

    ## Material parameters
    μ = 1.e-4       # [Ns/mm^2] Dynamic viscosity
    k = 5.0e-6      # [mm^2] Intrinsic permeability
    k_darcy = k/μ
    n = 0.8         # [-] Porosity
    K_liquid = 2.e3 # [MPa] Liquid bulk modulus
    dσdϵ = elastic_stiffness()

    ## Assemble stiffness and force vectors
    for q_point in 1:getnquadpoints(cv_u)    
        dΩ = getdetJdV(cv_u, q_point)
        ## Variation of u_i
        for i in 1:num_u_basefuncs
            ∇δNu = shape_symmetric_gradient(cv_u, q_point, i)
            div_δNu = shape_divergence(cv_u, q_point, i)
            for j in 1:num_u_basefuncs
                ∇Nu = shape_symmetric_gradient(cv_u, q_point, j)
                Kuu[i, j] -= ∇δNu ⊡ dσdϵ ⊡ ∇Nu * dΩ
            end
            for j in 1:num_p_basefuncs
                Np = shape_value(cv_p, q_point, j)
                Kup[i, j] += div_δNu * Np
            end
        end
        ## Variation of p_i
        for i in 1:num_p_basefuncs
            δNp = shape_value(cv_p, q_point, i)
            ∇δNp = shape_gradient(cv_p, q_point, i)
            for j in 1:num_u_basefuncs
                div_Nu = shape_divergence(cv_u, q_point, j)
                Lpu_ij = δNp*div_Nu*dΩ
                Kpu[i,j] += Lpu_ij
                fp[i] += Lpu_ij*au_old[j]
            end
            for j in 1:num_p_basefuncs
                ∇Np = shape_gradient(cv_p, q_point, j)
                Np = shape_value(cv_p, q_point, j)
                Kpp_ij = (k_darcy/n) * ∇δNp ⋅ ∇Np * dΩ
                Lpp_ij = δNp*Np/K_liquid
                Kpp[i,j] += Δt*Kpp_ij + Lpp_ij
                fp[i] += Lpp_ij*ap_old[j]
            end 
        end
    end
end

# ### Assembly
# In order to assemble the contribution from each cell, we first 
# loop over each field with the following function.
function doassemble!(K, f, dh, cvs, a_old, Δt)
    assembler = start_assemble(K, f)
    for (fh, cv) in zip(dh.fieldhandlers, cvs)
        doassemble!(assembler, dh, cv, fh, a_old, Δt)
    end
end

# When calling the `doassemble!` with a specific `FieldHandler`, 
# we can then loop over all cells in its cellset. This ensures
# that the `CellIterator` and `element_routine!` is type stable.
function doassemble!(assembler, dh, cv, fh::FieldHandler, a_old, Δt)
    n = ndofs_per_cell(dh, first(fh.cellset))
    Ke = zeros(n,n)
    fe = zeros(n)
    for cell in CellIterator(dh, collect(fh.cellset))
        element_routine!(Ke, fe, cell, fh, cv, a_old[celldofs(cell)], Δt)
        assemble!(assembler, celldofs(cell), Ke, fe)
    end
end

# ### Mesh import
# In this example, we import the mesh from an Abaqus input file using `FerriteMeshParser`'s 
# `get_ferrite_grid` function. We then create one cellset for each phase (solid and porous)
# for each element type. These 4 sets will later be used in their own `FieldHandler`
function get_grid()
    ## Import grid from abaqus mesh
    grid = get_ferrite_grid(joinpath(@__DIR__, "porous_media_0p25.inp"))

    ## Create cellsets for each fieldhandler
    addcellset!(grid, "solid3", intersect(getcellset(grid, "solid"), getcellset(grid, "CPS3")))
    addcellset!(grid, "solid4", intersect(getcellset(grid, "solid"), getcellset(grid, "CPS4R")))
    addcellset!(grid, "porous3", intersect(getcellset(grid, "porous"), getcellset(grid, "CPS3")))
    addcellset!(grid, "porous4", intersect(getcellset(grid, "porous"), getcellset(grid, "CPS4R")))
    return grid
end

# ### Problem setup 
# Define the finite element interpolation, integration, and boundary conditions. 
function setup_problem(;t_rise=0.1, p_max=100.0)

    grid = get_grid()

    ## Setup the interpolation and integration rules
    dim=Ferrite.getdim(grid)    
    ip3_lin = Lagrange{dim, RefTetrahedron, 1}()
    ip4_lin = Lagrange{dim, RefCube, 1}()
    ip3_quad = Lagrange{dim, RefTetrahedron, 2}()
    ip4_quad = Lagrange{dim, RefCube, 2}()
    qr3 = QuadratureRule{dim, RefTetrahedron}(1)
    qr4 = QuadratureRule{dim, RefCube}(2)

    ## Setup the MixedDofHandler
    dh = MixedDofHandler(grid)
    push!(dh, FieldHandler([Field(:u, ip3_lin, dim)], getcellset(grid,"solid3")))
    push!(dh, FieldHandler([Field(:u, ip4_lin, dim)], getcellset(grid,"solid4")))
    push!(dh, FieldHandler([Field(:u, ip3_quad, dim), Field(:p, ip3_lin, 1)], getcellset(grid,"porous3")))
    push!(dh, FieldHandler([Field(:u, ip4_quad, dim), Field(:p, ip4_lin, 1)], getcellset(grid,"porous4")))
    close!(dh)

    ## Setup cellvalues with the same order as the FieldHandlers in the dh
    ## - Linear displacement elements in the solid domain
    ## - Taylor hood (quadratic displacement, linear pressure) and linear geometry in porous domain
    cv = ( CellVectorValues(qr3, ip3_lin), 
           CellVectorValues(qr4, ip4_lin),
           (CellVectorValues(qr3, ip3_quad, ip3_lin), CellScalarValues(qr3, ip3_lin)),
           (CellVectorValues(qr4, ip4_quad, ip4_lin), CellScalarValues(qr4, ip4_lin)) )

    ## Add boundary conditions, use code from PR427.jl to make more general
    ch = ConstraintHandler(dh);
    ## With #PR427 (keep code for if/when it is merged)
    ## add!(ch, Dirichlet(:u, getfaceset(grid, "bottom"), (x, t) -> zero(Vec{2}), [1,2]))
    ## add!(ch, Dirichlet(:p, getfaceset(grid, "bottom_p"), (x, t) -> 0.0))
    ## add!(ch, Dirichlet(:p, getfaceset(grid, "top_p"), (x, t) -> p_max*clamp(t/t_rise,0,1)))
    ## With master (only works if no tri-elements on boundary)
    add!(ch, dh.fieldhandlers[2], Dirichlet(:u, getfaceset(grid, "bottom"), (x, t) -> zero(Vec{2}), [1,2]))
    add!(ch, dh.fieldhandlers[4], Dirichlet(:u, getfaceset(grid, "bottom_p"), (x, t) -> zero(Vec{2}), [1,2]))
    add!(ch, dh.fieldhandlers[4], Dirichlet(:p, getfaceset(grid, "bottom_p"), (x, t) -> 0.0))
    add!(ch, dh.fieldhandlers[4], Dirichlet(:p, getfaceset(grid, "top_p"), (x, t) -> p_max*clamp(t/t_rise,0,1)))
    close!(ch)
    return dh, ch, cv
end

# ### Solving
# Given the `MixedDofHandler`, `ConstraintHandler`, and `CellValues`, 
# we can solve the problem by stepping through the time history
function solve(dh, ch, cv; Δt=0.025, t_total=1.0)
    ## Assemble stiffness matrix
    K = create_sparsity_pattern(dh);
    f = zeros(ndofs(dh))
    a = zeros(ndofs(dh))
    pvd = paraview_collection("porous_media.pvd");
    for (step, t) = enumerate(0:Δt:t_total)
        if t>0    
            doassemble!(K, f, dh, cv, a, Δt)
            update!(ch, t)
            apply!(K, f, ch)
            a .= K\f
        end
        vtk_grid("porous_media-$step", dh) do vtk
            vtk_point_data(vtk, dh, a)
            vtk_save(vtk)
            pvd[step] = vtk
        end
    end
    vtk_save(pvd);
end

# Finally we call the functions to actually run the code
dh, ch, cv = setup_problem()
solve(dh, ch, cv)

#md # ## [Plain program](@id porous-media-plain-program)
#md #
#md # Here follows a version of the program without any comments.
#md # The file is also available here: [`porous_media.jl`](porous_media.jl).
#md #
#md # ```julia
#md # @__CODE__
#md # ```