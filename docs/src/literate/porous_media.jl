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
# ![Computational domain](porous_media.svg)
# ![Pressure evolution.](porous_media.gif)
#
# ## Theory of porous media
# The strong forms are given as
# ```math
# \begin{aligned}
# \boldsymbol{\sigma}(\boldsymbol{\epsilon}, p) \cdot \boldsymbol{\nabla} &= \boldsymbol{0} \\
# \dot{\Phi}(\boldsymbol{\epsilon}, p) + \boldsymbol{w}(p) \cdot \boldsymbol{\nabla} &= 0
# \end{aligned}
# ```
# where 
# ``\boldsymbol{\epsilon} = \left[\boldsymbol{u}\otimes\boldsymbol{\nabla}\right]^\mathrm{sym}`` 
# The constitutive relationships are 
# ```math
# \begin{aligned}
# \boldsymbol{\sigma} &= \boldsymbol{\mathsf{E}}:\boldsymbol{\epsilon} - \alpha p \boldsymbol{I} \\
# \boldsymbol{w} &= - k \boldsymbol{\nabla} p \\
# \Phi &= \phi + \alpha \mathrm{tr}(\boldsymbol{\epsilon}) + \beta p
# \end{aligned}
# ``` 
# with 
# ``\boldsymbol{\mathsf{E}}=2G \boldsymbol{\mathsf{I}}^\mathrm{dev} + 3K \boldsymbol{I}\otimes\boldsymbol{I}``.
# The material parameters are then the 
# shear modulus, ``G``, 
# bulk modulus, ``K``, 
# permeability, ``k``,  
# Biot's coefficient, ``\alpha``, and
# liquid compressibility, ``\beta``.
# The porosity, ``\phi``, doesn't enter into the equations 
# (A different porosity leads to different skeleton stiffness and permeability).
#
# 
# The variational (weak) form can then be derived for the variations ``\boldsymbol{\delta u}``
# and ``\delta p`` as
# ```math
# \begin{aligned}
# \int_\Omega \left[\left[\boldsymbol{\delta u}\otimes\boldsymbol{\nabla}\right]^\mathrm{sym}:
# \boldsymbol{\mathsf{E}}:\boldsymbol{\epsilon} - \boldsymbol{\delta u} \cdot \boldsymbol{\nabla} \alpha p\right] \mathrm{d}\Omega 
# &= \int_\Gamma \boldsymbol{\delta u} \cdot \boldsymbol{t} \mathrm{d} \Gamma \\
# \int_\Omega \left[\delta p \left[\alpha \dot{\boldsymbol{u}} \cdot \boldsymbol{\nabla} + \beta \dot{p}\right] + 
# \boldsymbol{\nabla}(\delta p) \cdot [k \boldsymbol{\nabla}]\right] \mathrm{d}\Omega 
# &= \int_\Gamma \delta p w_\mathrm{n} \mathrm{d} \Gamma 
# \end{aligned}
# ```
# where ``\boldsymbol{t}=\boldsymbol{n}\cdot\boldsymbol{\sigma}`` is the traction and 
# ``w_\mathrm{n} = \boldsymbol{n}\cdot\boldsymbol{w}`` is the normal flux.  
# 
# ### Finite element form
# Discretizing in space using finite elements, we obtain the vector equation 
# ``r_i = f_i^\mathrm{int} - f_{i}^\mathrm{ext}`` where ``f^\mathrm{ext}`` are the external 
# "forces", and ``f_i^\mathrm{int}`` are the internal "forces". We split this into the 
# displacement part ``r_i^\mathrm{u} = f_i^\mathrm{int,u} - f_{i}^\mathrm{ext,u}`` and 
# pressure part ``r_i^\mathrm{p} = f_i^\mathrm{int,p} - f_{i}^\mathrm{ext,p}``
# to obtain the discretized equation system 
# ```math
# \begin{aligned}
# f_i^\mathrm{int,u} &= \int_\Omega [\boldsymbol{\delta N}^\mathrm{u}_i\otimes\boldsymbol{\nabla}]^\mathrm{sym} : \boldsymbol{\mathsf{E}} : [\boldsymbol{u}\otimes\boldsymbol{\nabla}]^\mathrm{sym} \ 
# - [\boldsymbol{\delta N}^\mathrm{u}_i \cdot \boldsymbol{\nabla}] \alpha p \mathrm{d}\Omega 
# &= \int_\Gamma \boldsymbol{\delta N}^\mathrm{u}_i \cdot \boldsymbol{t} \mathrm{d} \Gamma \\
# f_i^\mathrm{int,p} &= \int_\Omega \delta N_i^\mathrm{p} [\alpha [\dot{\boldsymbol{u}}\cdot\boldsymbol{\nabla}]  + \beta\dot{p}] + \boldsymbol{\nabla}(\delta N_i^\mathrm{p}) \cdot [k \boldsymbol{\nabla}(p)] \mathrm{d}\Omega 
# &= \int_\Gamma \delta N_i^\mathrm{p} w_\mathrm{n} \mathrm{d} \Gamma
# \end{aligned}
# ```
# Approximating the time-derivatives, ``\dot{\boldsymbol{u}}\approx \left[\boldsymbol{u}-{}^n\boldsymbol{u}\right]/\Delta t``
# and ``\dot{p}\approx \left[p-{}^np\right]/\Delta t``, we can implement the finite element equations in the residual form 
# ``r_i(\boldsymbol{a}(t), t) = 0`` where the vector ``\boldsymbol{a}`` contains all unknown displacements ``u_i`` and pressures ``p_i``. 
#
# The jacobian, ``K_{ij} = \partial r_i/\partial a_j``, is then split into four parts,
# \begin{aligned}
# K_{ij}^\mathrm{uu} = \frac{\partial r_i^\mathrm{u}}{\partial u_j}} = \int_\Omega [\boldsymbol{\delta N}^\mathrm{u}_i\otimes\boldsymbol{\nabla}]^\mathrm{sym} : \boldsymbol{\mathsf{E}} : [\boldsymbol{N}_j^\mathrm{u}\otimes\boldsymbol{\nabla}]^\mathrm{sym}\ \mathrm{d}\Omega \\
# K_{ij}^\mathrm{up} = \frac{\partial r_i^\mathrm{u}}{\partial p_j}} = - \int_\Omega [\boldsymbol{\delta N}^\mathrm{u}_i \cdot \boldsymbol{\nabla}] \alpha N_j^\mathrm{p}\ \mathrm{d}\Omega \\
# K_{ij}^\mathrm{pu} = \frac{\partial r_i^\mathrm{p}}{\partial u_j}} = \int_\Omega \delta N_i^\mathrm{p} \frac{\alpha}{\Delta t} [\boldsymbol{N}_j^\mathrm{u} \cdot\boldsymbol{\nabla}]\ \mathrm{d}\Omega\\
# K_{ij}^\mathrm{pp} = \frac{\partial r_i^\mathrm{p}}{\partial p_j}} = \int_\Omega \delta N_i^\mathrm{p} \frac{N_j^\mathrm{p}}{\Delta t} + \boldsymbol{\nabla}(\delta N_i^\mathrm{p}) \cdot [k \boldsymbol{\nabla}(N_j^\mathrm{p})] \mathrm{d}\Omega
# \end{aligned}
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
#    K_{ij}^\mathrm{pp} &= \int_\Omega \mathrm{grad}\left(\delta N^\mathrm{p}_i\right)\cdot \left[\frac{k}{n}\mathrm{grad}\left(N^\mathrm{p}_j\right)\right] \mathrm{d}\Omega \\
#    L_{ij}^\mathrm{pp} &= \int_\Omega \delta N_i^\mathrm{p} N_j^\mathrm{p}/K_\mathrm{l} \mathrm{d}\Omega \\
#    L_{ij}^\mathrm{pu} &= \int_\Omega \delta N_i^\mathrm{p} \mathrm{div}\left(\boldsymbol{N}_j^\mathrm{u}\right) \mathrm{d}\Omega \\
#    K_{ij}^\mathrm{uu} &= - \int_\Omega \mathrm{grad}\left(\boldsymbol{\delta N}^\mathrm{u}_i\right) : \boldsymbol{\mathrm{E}} : \mathrm{grad}\left(\boldsymbol{N}_j^\mathrm{u}\right) \mathrm{d} \Omega \\ 
#    K_{ij}^\mathrm{up} &= \int_\Omega \mathrm{div}\left(\boldsymbol{\delta N}_i^\mathrm{u}\right) N_j^\mathrm{p} \mathrm{d}\Omega \\
#    f_{i}^\mathrm{p,ext} &= \int_\Gamma \delta N_i^\mathrm{p} \tilde{\boldsymbol{v}}_\mathrm{l} \cdot \boldsymbol{n} \mathrm{d}\Gamma\\
#    f_{i}^\mathrm{u,ext} &= -\int_\Gamma \boldsymbol{\delta N}_i^\mathrm{u} \cdot \boldsymbol{t} \mathrm{d} \Gamma
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
# We start by defining the elastic material type, containing the elastic stiffness,
# for the linear elastic impermeable solid aggregates.
struct Elastic{T}
    C::SymmetricTensor{4,2,T,9}
end
function Elastic(;E=20.e3, ν=0.3)
    G = E / 2(1 + ν)
    K = E / 3(1 - 2ν)
    I2 = one(SymmetricTensor{2,2})
    I4vol = I2⊗I2
    I4dev = minorsymmetric(otimesu(I2,I2)) - I4vol / 3
    return Elastic(2G*I4dev + K*I4vol)
end

# Next, we define the element routine for the solid aggregates, where we dispatch on the 
# `Elastic` material struct. Note that the unused inputs here are used for the porous matrix below. 
function element_routine!(Ke, fext, material::Elastic, cv, cell, args...)
    reinit!(cv, cell)
    n_basefuncs = getnbasefunctions(cv)
    dσdϵ = material.C
    for q_point in 1:getnquadpoints(cv)
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

# ### PoroElasticity
# To define the poroelastic material, we re-use the elastic part from above for 
# the skeleton, and add the additional required material parameters.
struct PoroElastic{T}
    elastic::Elastic{T} ## Skeleton stiffness
    k::T            ## Permeability of liquid   [mm^4/(Ns)]
    ϕ::T            ## Porosity                 [-]
    K_liquid::T     ## Liquid bulk modulus      [MPa]
end
PoroElastic(;elastic, k, ϕ, K_liquid) = PoroElastic(elastic, k, ϕ, K_liquid) # Keyword constructor

# The element routine requires a few more inputs since we have two fields, as well 
# as the dependence on the rates of the displacements and pressure. 
# Again, we dispatch on the material type.
function element_routine!(Ke, fext, m::PoroElastic, cvs::Tuple, cell, a_old, Δt, sdh)
    ## Setup cellvalues and give easier names
    reinit!.(cvs, (cell,))
    cv_u, cv_p = cvs
    
    ## Check that cellvalues are compatible with each other (should have same quadrature rule)
    @assert getnquadpoints(cv_u) == getnquadpoints(cv_p)

    C_el = m.elastic.C ## Elastic stiffness 

    ## Assemble stiffness and force vectors
    for q_point in 1:getnquadpoints(cv_u)    
        dΩ = getdetJdV(cv_u, q_point)
        ## Variation of u_i
        for (iᵤ, Iᵤ) in pairs(dof_range(sdh, :u))
            ∇δNu = shape_symmetric_gradient(cv_u, q_point, iᵤ)
            div_δNu = shape_divergence(cv_u, q_point, iᵤ)
            for (jᵤ, Jᵤ) in pairs(dof_range(sdh, :u))
                ∇Nu = shape_symmetric_gradient(cv_u, q_point, jᵤ)
                Ke[Iᵤ, Jᵤ] -= ∇δNu ⊡ C_el ⊡ ∇Nu * dΩ
            end
            for (jₚ, Jₚ) in pairs(dof_range(sdh, :p))
                Np = shape_value(cv_p, q_point, jₚ)
                Ke[Iᵤ, Jₚ] += div_δNu * Np
            end
        end
        ## Variation of p_i
        for (iₚ, Iₚ) in pairs(dof_range(sdh, :p))
            δNp = shape_value(cv_p, q_point, iₚ)
            ∇δNp = shape_gradient(cv_p, q_point, iₚ)
            for (jᵤ, Jᵤ) in pairs(dof_range(sdh, :u))
                div_Nu = shape_divergence(cv_u, q_point, jᵤ)
                Lpu_ij = δNp*div_Nu*dΩ
                Ke[Iₚ,Jᵤ] += Lpu_ij
                fext[Iₚ] += Lpu_ij*a_old[Jᵤ]
            end
            for (jₚ, Jₚ) in pairs(dof_range(sdh, :p))
                ∇Np = shape_gradient(cv_p, q_point, jₚ)
                Np = shape_value(cv_p, q_point, jₚ)
                Kpp_ij = (m.k/m.ϕ) * ∇δNp ⋅ ∇Np * dΩ
                Lpp_ij = δNp*Np/m.K_liquid
                Ke[Iₚ,Jₚ] += Δt*Kpp_ij + Lpp_ij
                fext[Iₚ] += Lpp_ij*a_old[Jₚ]
            end 
        end
    end
end

# ### Assembly
# To organize the different domains, we'll first define a container type
struct FEDomain{M,CV,SDH<:SubDofHandler}
    material::M
    cellvalues::CV
    sdh::SDH
end

# And then we can loop over a vector of such domains, allowing us to 
# loop over each domain, to assemble the contributions from each 
# cell in that domain (given by the `SubDofHandler`'s cellset)
function doassemble!(K, f, domains::Vector{<:FEDomain}, a_old, Δt)
    assembler = start_assemble(K, f)
    for domain in domains
        doassemble!(assembler, domain, a_old, Δt)
    end
end

# For one domain (corresponding to a specific SubDofHandler),
# we can then loop over all cells in its cellset. This ensures
# that the calls to the `element_routine` are type stable.
function doassemble!(assembler, domain::FEDomain, a_old, Δt)
    material = domain.material 
    cv = domain.cellvalues
    sdh = domain.sdh
    n = ndofs_per_cell(sdh)
    Ke = zeros(n,n)
    fe = zeros(n)
    ae_old = zeros(n)

    for cell in CellIterator(sdh)
        map!(i->a_old[i], ae_old, celldofs(cell)) # copy values from a_old to ae_old
        fill!(Ke, 0)
        fill!(fe, 0)
        element_routine!(Ke, fe, material, cv, cell, ae_old, Δt, sdh)
        assemble!(assembler, celldofs(cell), Ke, fe)
    end
end

# ### Mesh import
# In this example, we import the mesh from the Abaqus input file, [`porous_media_0p25.inp`](porous_media_0p25.inp) using `FerriteMeshParser`'s 
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

    ## Setup the materials 
    m_solid = Elastic(;E=20.e3, ν=0.3)
    m_porous = PoroElastic(;elastic=Elastic(;E=10e3, ν=0.3), K_liquid=15e3, k=5.0e-2, ϕ=0.8)

    ## Setup the interpolation and integration rules
    dim=Ferrite.getdim(grid)
    ipu_quad = Lagrange{RefQuadrilateral,2}()^2
    ipu_tri  = Lagrange{RefTriangle,2}()^2
    ipp_quad = Lagrange{RefQuadrilateral,1}()
    ipp_tri  = Lagrange{RefTriangle,1}()
    
    qr_quad = QuadratureRule{RefQuadrilateral}(2)
    qr_tri  = QuadratureRule{RefTriangle}(2)

    ## CellValues
    cvu_quad = CellValues(qr_quad, ipu_quad)
    cvu_tri = CellValues(qr_tri, ipu_tri)
    cvp_quad = CellValues(qr_quad, ipp_quad)
    cvp_tri = CellValues(qr_tri, ipp_tri)

    ## Setup the DofHandler
    dh = DofHandler(grid)
    ## Solid quads
    sdh_solid_quad = SubDofHandler(dh, getcellset(grid,"solid4"))
    add!(sdh_solid_quad, :u, ipu_quad)
    ## Solid triangles
    sdh_solid_tri = SubDofHandler(dh, getcellset(grid,"solid3"))
    add!(sdh_solid_tri, :u, ipu_tri)
    ## Porous quads 
    sdh_porous_quad = SubDofHandler(dh, getcellset(grid, "porous4"))
    add!(sdh_porous_quad, :u, ipu_quad)
    add!(sdh_porous_quad, :p, ipp_quad)
    ## Porous triangles 
    sdh_porous_tri = SubDofHandler(dh, getcellset(grid, "porous3"))
    add!(sdh_porous_tri, :u, ipu_tri)
    add!(sdh_porous_tri, :p, ipp_tri)

    close!(dh)

    ## Setup the domains
    domains = [FEDomain(m_solid, cvu_quad, sdh_solid_quad),
               FEDomain(m_solid, cvu_tri, sdh_solid_tri),
               FEDomain(m_porous, (cvu_quad, cvp_quad), sdh_porous_quad),
               FEDomain(m_porous, (cvu_tri, cvp_tri), sdh_porous_tri)
               ]

    ## Add boundary conditions
    ch = ConstraintHandler(dh);
    add!(ch, Dirichlet(:u, getfaceset(grid, "bottom"), (x, t) -> zero(Vec{2}), [1,2]))
    add!(ch, Dirichlet(:p, getfaceset(grid, "bottom_p"), (x, t) -> 0.0))
    add!(ch, Dirichlet(:p, getfaceset(grid, "top_p"), (x, t) -> p_max*clamp(t/t_rise,0,1)))
    close!(ch)

    return dh, ch, domains
end

# ### Solving
# Given the `MixedDofHandler`, `ConstraintHandler`, and `CellValues`, 
# we can solve the problem by stepping through the time history
function solve(dh, ch, domains; Δt=0.025, t_total=1.0)
    K = create_sparsity_pattern(dh);
    f = zeros(ndofs(dh))
    a = zeros(ndofs(dh))
    pvd = paraview_collection("porous_media.pvd");
    for (step, t) = enumerate(0:Δt:t_total)
        if t>0    
            doassemble!(K, f, domains, a, Δt)
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
dh, ch, domains = setup_problem()
solve(dh, ch, domains)

#md # ## [Plain program](@id porous-media-plain-program)
#md #
#md # Here follows a version of the program without any comments.
#md # The file is also available here: [`porous_media.jl`](porous_media.jl).
#md #
#md # ```julia
#md # @__CODE__
#md # ```