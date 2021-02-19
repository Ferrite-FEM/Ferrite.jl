# # Bidomain and DifferentialEquations.jl
#
# ![](bidomain.gif)
#-
#md # !!! tip
#md #     This example is also available as a Jupyter notebook:
#md #     [`bidomain.ipynb`](@__NBVIEWER_ROOT_URL__/examples/bidomain.ipynb)
#-
# ## Introduction
#
# In this example we will implement the [Bidomain Model](https://en.wikipedia.org/wiki/Bidomain_model).
# This model is used to simulate the electrical activity of the heart. For more information about the derivation,
# check out the linked wikipedia article.
#
# The Bidomain model consists of a set of equations, namely:
# ```math
# \chi  C_{\textrm{m}} \frac{\partial \varphi_{\textrm{m}}}{\partial t} = \nabla \cdot (\mathbf{\kappa}_{\textrm{i}} \nabla \varphi_{\textrm{m}}) + \nabla \cdot (\mathbf{\kappa}_{\textrm{i}} \nabla \varphi_{\textrm{e}}) - \chi I_{\textrm{ion}}(\varphi_{\textrm{m}}, \mathbf{s}) - \chi I_{\textrm{stim}}(t) \qquad \textrm{on} : \Omega_{\mathbb{H}} \times (0,T]  
# ```
#
# ```math
# 0 = \nabla \cdot (\mathbf{\kappa}_{\textrm{i}} \nabla \varphi_{\textrm{m}}) + \nabla \cdot (\mathbf{\kappa}_e + \mathbf{\kappa}_{\textrm{i}}) \nabla \varphi_{\textrm{e}} \qquad \textrm{on}  : \Omega_{\mathbb{H}} \times (0,T] 
# ```
#
# ```math
# \frac{\partial \mathbf{s}}{\partial t} = \mathbf{g}(\varphi_{\textrm{m}}, \mathbf{s}) \qquad \textrm{on}  : \Omega_{\mathbb{H}} \times (0,T]
# ```
#
# ```math
# f = \varphi_m - 1/3 \varphi_m^3 - s 
# ```
#
# ```math
# g = \varphi_m c - a - b s
# ```
#-
# ## Commented Program
#
using JuAFEM, SparseArrays, BlockArrays
# Instead of using a self written time integrator, 
# we will use in this example a time integrator of [DifferentialEquations.jl](https://github.com/SciML/DifferentialEquations.jl)
# [DifferentialEquations.jl](https://github.com/SciML/DifferentialEquations.jl) is a powerful package, from which we will use
# adaptive time stepping. Besides this, almost any ODE solver you can imagine is available.
# In order to use it, we first need to `import` it. 
import DifferentialEquations
#
# Now, we define the computational domain and cellvalues. We exploit the fact that all fields of
# the Bidomain model are approximated with the same Ansatz. Hence, we use one CellScalarValues struct for all three fields.
grid = generate_grid(Quadrilateral, (20, 20))
dim = 2
ip = Lagrange{dim, RefCube, 1}()
qr = QuadratureRule{dim, RefCube}(2)
cellvalues = CellScalarValues(qr, ip);
#
# We need to intialize a DofHandler. The DofHandler needs to be aware of three different fields
# which are all first order approximations. After pushing all fields into the DofHandler, we `close`
# it and thereby distribute the dofs of the problem.
dh = DofHandler(grid)
push!(dh, :ϕₘ, 1)
push!(dh, :ϕₑ, 1)
push!(dh, :s, 1)
close!(dh);
# 
# The linear parts of the Bidomain equations contribute to the stiffness and mass matrix, respectively.
# So, we create a sparsity pattern for those terms.
K = create_sparsity_pattern(dh)
M = create_sparsity_pattern(dh);
#
# Material related parameters are stored in the struct `FHNParameters`
Base.@kwdef struct FHNParameters
    a::Float64 = 0.7
    b::Float64 = 0.8
    c::Float64 = 3.0
end
#
# Within the equations of the model, spatial dependent parameters occur such as κₑ, κᵢ, Cₘ and χ.
# For the sake of simplicity we kept them constant.
# Nonetheless, we show how one can model spatial dependent coefficients. Hence, the unused function argument `x`
function κₑ(x)
    return SymmetricTensor{2,2,Float64}((0.22, 0, 0.13))
end
#
function κᵢ(x)
    return SymmetricTensor{2,2,Float64}((0.28, 0, 0.026))
end
#
function Cₘ(x)
    return 1.0
end
#
function χ(x)
    return 1.0
end
# The function `Iₛₜᵢₘ` models the stimulus and can be interpreted as a source term.
function Iₛₜᵢₘ(x, t)
    if norm(x) < 0.25 && t < 5
        return 1.0
    else
        return 0
    end
end
#
# Boundary conditions are added to the problem in the usual way. 
# Please check out the other examples for an in depth explanation.
ch = ConstraintHandler(dh)
∂Ω = getfaceset(grid, "left")
dbc = Dirichlet(:ϕₑ, ∂Ω, (x, t) -> 0)
add!(ch, dbc)
close!(ch)
update!(ch, 0.0);
#
# In the following function, `doassemble_linear!`, we assemble all linear parts of the system that stay same over all time steps.
# This follows from the used Method of Lines, where we first discretize in space and afterwards in time.
function doassemble_linear!(cellvalues::CellScalarValues{dim}, K::SparseMatrixCSC, M::SparseMatrixCSC, dh::DofHandler; params::FHNParameters = FHNParameters()) where {dim}
    n_ϕₘ = getnbasefunctions(cellvalues)
    n_ϕₑ = getnbasefunctions(cellvalues)
    n_s = getnbasefunctions(cellvalues)
    ntotal = n_ϕₘ + n_ϕₑ + n_s
    n_basefuncs = getnbasefunctions(cellvalues)
    #We use PseudoBlockArrays to write into the right places of Ke    
    Ke = PseudoBlockArray(zeros(ntotal, ntotal), [n_ϕₘ, n_ϕₑ, n_s], [n_ϕₘ, n_ϕₑ, n_s])
    Me = PseudoBlockArray(zeros(ntotal, ntotal), [n_ϕₘ, n_ϕₑ, n_s], [n_ϕₘ, n_ϕₑ, n_s])

    assembler_K = start_assemble(K)
    assembler_M = start_assemble(M)

    #Here the block indices of the variables are defined.
    ϕₘ▄, ϕₑ▄, s▄ = 1, 2, 3

    #Now we iterate over all cells of the grid
    @inbounds for cell in CellIterator(dh)
        fill!(Ke, 0)
        fill!(Me, 0)
        #get the coordinates of the current cell
        coords = getcoordinates(cell)

        reinit!(cellvalues, cell)
        #loop over all Gauss points
        for q_point in 1:getnquadpoints(cellvalues)
            #get the spatial coordinates of the current gauss point
            coords_qp = spatial_coordinate(cellvalues, q_point, coords)
            #based on the gauss point coordinates, we get the spatial dependent
            #material parameters
            κₑ_loc = κₑ(coords_qp)
            κᵢ_loc = κᵢ(coords_qp)
            Cₘ_loc = Cₘ(coords_qp)
            χ_loc = χ(coords_qp)
            dΩ = getdetJdV(cellvalues, q_point)
            for i in 1:n_basefuncs
                Nᵢ = shape_value(cellvalues, q_point, i)
                ∇Nᵢ = shape_gradient(cellvalues, q_point, i)
                for j in 1:n_basefuncs
                    Nⱼ = shape_value(cellvalues, q_point, j)
                    ∇Nⱼ = shape_gradient(cellvalues, q_point, j)
                    #diffusion parts
                    Ke[BlockIndex((ϕₘ▄,ϕₘ▄),(i,j))] -= ((κᵢ_loc ⋅ ∇Nᵢ) ⋅ ∇Nⱼ) * dΩ
                    Ke[BlockIndex((ϕₘ▄,ϕₑ▄),(i,j))] -= ((κᵢ_loc ⋅ ∇Nᵢ) ⋅ ∇Nⱼ) * dΩ
                    Ke[BlockIndex((ϕₑ▄,ϕₘ▄),(i,j))] -= ((κᵢ_loc ⋅ ∇Nᵢ) ⋅ ∇Nⱼ) * dΩ
                    Ke[BlockIndex((ϕₑ▄,ϕₑ▄),(i,j))] -= (((κₑ_loc + κᵢ_loc) ⋅ ∇Nᵢ) ⋅ ∇Nⱼ) * dΩ
                    #linear reaction parts
                    Ke[BlockIndex((ϕₘ▄,ϕₘ▄),(i,j))] += params.c * Nᵢ * Nⱼ * dΩ 
                    Ke[BlockIndex((ϕₘ▄,s▄),(i,j))]  += params.c * Nᵢ * Nⱼ * dΩ 
                    Ke[BlockIndex((s▄,ϕₘ▄),(i,j))]  -= 1.0/params.c * Nᵢ * Nⱼ * dΩ 
                    Ke[BlockIndex((s▄,s▄),(i,j))]   -=  params.b/params.c * Nᵢ * Nⱼ * dΩ 
                    #mass matrices
                    Me[BlockIndex((ϕₘ▄,ϕₘ▄),(i,j))] += Cₘ_loc * χ_loc * Nᵢ * Nⱼ * dΩ
                    Me[BlockIndex((s▄,s▄),(i,j))]   += Nᵢ * Nⱼ * dΩ
                end
            end
        end

        assemble!(assembler_K, celldofs(cell), Ke)
        assemble!(assembler_M, celldofs(cell), Me)
    end
    return K, M
end
#
# TODO cleanup
# The function `apply_nonlinear!` describes the nonlinear change of the system.
# It takes the change vector `du`, the current available solution `u`, the generic storage
# vector `p` and the current time `t`. The storage vector will be used to pass the `dh::DofHandler`, 
# `ch::ConstraintHandler`, stiffness matrix `K` and constant material parameters `FHNParameters()`
function apply_nonlinear!(du, u, p, t)
    dh = p[2]
    ch = p[3]
    ip = Lagrange{2, RefCube, 1}()
    qr = QuadratureRule{2, RefCube}(2)
    cellvalues = CellScalarValues(qr, ip);
    n_basefuncs = getnquadpoints(cellvalues)

    for cell in CellIterator(dh)
        reinit!(cellvalues, cell)
        _celldofs = celldofs(cell)
        ϕₘ_celldofs = _celldofs[dof_range(dh, :ϕₘ)]
        s_celldofs = _celldofs[dof_range(dh, :s)]
        ϕₘe = u[ϕₘ_celldofs]
        se = u[s_celldofs]
        coords = getcoordinates(cell)
        for q_point in 1:getnquadpoints(cellvalues) 
            x_qp = spatial_coordinate(cellvalues, q_point, coords)
            χ_loc = χ(x_qp)
            dΩ = getdetJdV(cellvalues, q_point)
            nl_contrib = function_value(cellvalues, q_point, ϕₘe)^3
            for j in 1:n_basefuncs
                Nⱼ = shape_value(cellvalues, q_point, j)
                du[ϕₘ_celldofs[j]] -= p[4].c * (1/3 * nl_contrib + χ_loc*Iₛₜᵢₘ(x_qp,t)) * Nⱼ * dΩ
                du[s_celldofs[j]] += p[4].a/p[4].c * Nⱼ * dΩ
            end
        end
    end
    apply_zero!(du, ch)
end
#
# We assemble the linear parts into `K` and `M`, respectively.
K, M = doassemble_linear!(cellvalues, K, M, dh);
# Now we apply *once* the boundary conditions to these parts.
apply!(K, ch)
apply!(M, ch);
#
# In the function `bidomain!` we model the actual time dependent DAE problem. This function takes
# the same parameters as `apply_nonlinear!`, which is essentially the defined interface by
# DifferentialEquations.jl
function bidomain!(du,u,p,t)
    du .= K * u 
    println("Solving for timestep t=$t")
    apply_nonlinear!(du, u, p, t)
end
#
Δt = 0.01
T = 1
f = DifferentialEquations.ODEFunction(bidomain!,mass_matrix=M)
# In the following code block we define the initial condition of the problem. We first
# initialize a zero vector of length `ndofs(dh)` and fill it afterwards in a for loop over all cells.
u₀ = zeros(ndofs(dh));
for cell in CellIterator(dh)
    _celldofs = celldofs(cell)
    ϕₘ_celldofs = _celldofs[dof_range(dh, :ϕₘ)]
    s_celldofs = _celldofs[dof_range(dh, :s)]
    u₀[ϕₘ_celldofs] .= 1.19941300
    u₀[s_celldofs]  .= -0.6242615997254787
end
# We can now state the `ODEProblem`
prob_mm = DifferentialEquations.ODEProblem(f,u₀,(0.0,T),[K, dh, ch, FHNParameters()])
# and solve it.
sol = DifferentialEquations.solve(prob_mm,DifferentialEquations.QBDF(),reltol=1e-3,abstol=1e-4, adaptive=true, dt=Δt)
#
# We instantiate a paraview collection file.
pvd = paraview_collection("paraview/bidomain.pvd")
# Now, we loop over all timesteps and solution vectors, in order to append them to the paraview collection.
for (solution,t) in zip(sol.u, sol.t)
    #compress=false flag because otherwise each vtk file will be stored in memory
    vtk_grid("paraview/bidomain-$t.vtu", dh; compress=false) do vtk
        vtk_point_data(vtk,dh,solution)
        vtk_save(vtk)
        pvd[t] = vtk
    end
end
# Finally, we save the paraview collection.
vtk_save(pvd)
#md # ## [Plain Program](@id bidomain-plain-program)
#md #
#md # Below follows a version of the program without any comments.
#md # The file is also available here: [bidomain.jl](bidomain.jl)
#md #
#md # ```julia
#md # @__CODE__
#md # ```
