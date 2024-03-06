# # [Topology optimization](@id tutorial-topology-optimization)
#
# **Keywords**: *Topology optimization*, *weak and strong form*, *non-linear problem*, *Laplacian*, *grid topology*
#
# ![](bending_animation.gif)
#
# *Figure 1*: Optimization of the bending beam. Evolution of the density for fixed total mass.
#
#-
#md # !!! tip
#md #     This example is also available as a Jupyter notebook:
#md #     [`topology_optimization.ipynb`](@__NBVIEWER_ROOT_URL__/examples/topology_optimization.ipynb).
#-
#
# ## Introduction
#
# Topology optimization is the task of finding structures that are mechanically ideal. 
# In this example we cover the bending beam, where we specify a load, boundary conditions and the total mass. Then, our
# objective is to find the most suitable geometry within the design space minimizing the compliance (i.e. the inverse stiffness) of the structure.
# We shortly introduce our simplified model for regular meshes. A detailed derivation of the method and advanced techniques 
# can be found in [JanHacJun2019regularizedthermotopopt](@cite) and
# [BlaJanJun2022taylorwlsthermotopopt](@cite).
#
# We start by introducing the local, elementwise density $\chi \in [\chi_{\text{min}}, 1]$ of the material, where we choose
# $\chi_{\text{min}}$ slightly above zero to prevent numerical instabilities. Here, $\chi = \chi_{\text{min}}$ means void and $\chi=1$
# means bulk material. Then, we use a SIMP ansatz (solid isotropic material with penalization) for the stiffness tensor
# $C(\chi) = \chi^p C_0$, where $C_0$ is the stiffness of the bulk material. The SIMP exponent $p>1$ ensures that the
# model prefers the density values void and bulk before the intermediate values. The variational formulation then yields
# the modified Gibbs energy 
# ```math
# G = \int_{\Omega} \frac{1}{2} \chi^p \varepsilon : C : \varepsilon \; \text{d}V - \int_{\Omega} \boldsymbol{f} \cdot \boldsymbol{u} \; \text{d}V - \int_{\partial\Omega} \boldsymbol{t} \cdot \boldsymbol{u} \; \text{d}A.
# ```
# Furthermore, we receive the evolution equation of the density
# and the additional Neumann boundary condition in the strong form
# ```math
# p_\chi + \eta \dot{\chi} + \lambda + \gamma - \beta \nabla^2 \chi \ni 0 \quad \forall \textbf{x} \in \Omega, 
# ```
# ```math
# \beta \nabla \chi \cdot \textbf{n} = 0 \quad \forall \textbf{x} \in \partial \Omega, 
# ```
# with the thermodynamic driving force
# ```math
# p_\chi = \frac{1}{2} p \chi^{p-1} \varepsilon : C : \varepsilon.
# ```
# We obtain the mechanical displacement field by applying the Finite Element Method to the weak form
# of the Gibbs energy using Ferrite. In contrast, we use the evolution equation (i.e. the strong form) to calculate
# the value of the density field $\chi$. The advantage of this "split" approach is the very high computation speed.
# The evolution equation consists of the driving force, the damping parameter $\eta$, the regularization parameter $\beta$ times the Laplacian,
# which is necessary to avoid numerical issues like mesh dependence or checkerboarding, and the constraint parameters $\lambda$, to keep the mass constant,
# and $\gamma$, to avoid leaving the set $[\chi_{\text{min}}, 1]$. By including gradient regularization, it becomes necessary to calculate the Laplacian.
# The Finite Difference Method for square meshes with the edge length $\Delta h$ approximates the Laplacian as follows:
# ```math
# \nabla^2 \chi_p = \frac{1}{(\Delta h)^2} (\chi_n + \chi_s + \chi_w + \chi_e - 4 \chi_p)
# ```
# Here, the indices refer to the different cardinal directions. Boundary element do not have neighbors in each direction. However, we can calculate
# the central difference to fulfill Neumann boundary condition. For example, if the element is on the left boundary, we have to fulfill
# ```math
# \nabla \chi_p \cdot \textbf{n} = \frac{1}{\Delta h} (\chi_w - \chi_e) = 0
# ```
# from which follows $\chi_w = \chi_e$. Thus for boundary elements we can replace the value for the missing neighbor by the value of the opposite neighbor. 
# In order to find the corresponding neighbor elements, we will make use of Ferrites grid topology funcionalities. 
#
# ## Commented Program
# We now solve the problem in Ferrite. What follows is a program spliced with comments.
#md # The full program, without comments, can be found in the next [section](@ref topology_optimization-plain-program).
#
# First we load all necessary packages.
using Ferrite, SparseArrays, LinearAlgebra, Tensors, Printf
# Next, we create a simple square grid of the size 2x1. We apply a fixed Dirichlet boundary condition
# to the left face set, called `clamped`. On the right face, we create a small set `traction`, where we
# will later apply a force in negative y-direction.

function create_grid(n) 
    corners = [Vec{2}((0.0, 0.0)),
               Vec{2}((2.0, 0.0)),
               Vec{2}((2.0, 1.0)),
               Vec{2}((0.0, 1.0))]
    grid = generate_grid(Quadrilateral, (2*n, n), corners);
    
    ## node-/facesets for boundary conditions
    addnodeset!(grid, "clamped", x -> x[1] ≈ 0.0)
    addfaceset!(grid, "traction", x -> x[1] ≈ 2.0 && norm(x[2]-0.5) <= 0.05); 
    return grid
end
#md nothing # hide

# Next, we create the FE values, the DofHandler and the Dirichlet boundary condition.

function create_values()
    ## quadrature rules
    qr      = QuadratureRule{RefQuadrilateral}(2)
    face_qr = FaceQuadratureRule{RefQuadrilateral}(2)

    ## cell and facevalues for u
    ip = Lagrange{RefQuadrilateral,1}()^2
    cellvalues = CellValues(qr, ip)
    facevalues = FaceValues(face_qr, ip)
    
    return cellvalues, facevalues
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
#md nothing # hide

# Now, we define a struct to store all necessary material parameters (stiffness tensor of the bulk material
# and the parameters for topology optimization) and add a constructor to the struct to
# initialize it by using the common material parameters Young's modulus and Poisson number.

struct MaterialParameters{T, S <: SymmetricTensor{4, 2, T}}
    C::S
    χ_min::T 
    p::T
    β::T
    η::T
end
#md nothing # hide

function MaterialParameters(E, ν, χ_min, p, β, η) 
    δ(i,j) = i == j ? 1.0 : 0.0 # helper function

    G = E / 2(1 + ν) # =μ
    λ = E*ν/(1-ν^2) # correction for plane stress included

    C = SymmetricTensor{4, 2}((i,j,k,l) -> λ * δ(i,j)*δ(k,l) + G* (δ(i,k)*δ(j,l) + δ(i,l)*δ(j,k)))
    return MaterialParameters(C, χ_min, p, β, η)
end
#md nothing # hide

# To store the density and the strain required to calculate the driving forces, we create the struct
# `MaterialState`. We add a constructor to initialize the struct. The function `update_material_states!`
# updates the density values once we calculated the new values.

mutable struct MaterialState{T, S <: AbstractArray{SymmetricTensor{2, 2, T}, 1}} 
    χ::T # density
    ε::S # strain in each quadrature point
end

function MaterialState(ρ, n_qp)
    return MaterialState(ρ, Array{SymmetricTensor{2,2,Float64},1}(undef, n_qp))
end

function update_material_states!(χn1, states, dh)
    for (element, state) in zip(CellIterator(dh),states)
        state.χ = χn1[cellid(element)]
    end
end
#md nothing # hide

# Next, we define a function to calculate the driving forces for all elements.
# For this purpose, we iterate through all elements and calculate the average strain in each
# element. Then, we compute the driving force from the formula introduced at the beginning.
# We create a second function to collect the density in each element. 

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
#md nothing # hide

# Now we calculate the Laplacian. For this purpose, we will later create the grid topology of 
# the grid by using the function `ExclusiveTopology`. Then we iterate through each face of each element,
# obtaining the neighboring element by using the `getneighborhood` function. For boundary faces,
# the function call will return an empty object. In that case we use the dictionary to instead find the opposite
# face, as discussed in the introduction. Then, the approximation of the Laplacian reduces to the sum below. 

function approximate_laplacian(dh, topology, χn, Δh)
    ∇²χ = zeros(getncells(dh.grid))
    _nfaces = nfaces(dh.grid.cells[1])
    opp = Dict(1=>3, 2=>4, 3=>1, 4=>2)
    nbg = zeros(Int,_nfaces)
    
    for element in CellIterator(dh)
        i = cellid(element)
        for j in 1:_nfaces
            nbg_cellid = getcells(getneighborhood(topology, dh.grid, FaceIndex(i,j)))
            if(!isempty(nbg_cellid))
                nbg[j] = first(nbg_cellid) # assuming only one face neighbor per cell
            else # boundary face
                nbg[j] = first(getcells(getneighborhood(topology, dh.grid, FaceIndex(i,opp[j]))))
            end
        end
        
        ∇²χ[i] = (χn[nbg[1]]+χn[nbg[2]]+χn[nbg[3]]+χn[nbg[4]]-4*χn[i])/(Δh^2)
    end

    return ∇²χ
end
#md nothing # hide

# For the iterative computation of the solution, a function is needed to update the densities in each element.
# To ensure that the mass is kept constant, we have to calculate the constraint
# parameter $\lambda$, which we do via the bisection method. We repeat the calculation
# until the difference between the average density (calculated from the element-wise trial densities) and the target density nearly vanishes. 
# By using the extremal values of $\Delta \chi$ as the starting interval, we guarantee that the method converges eventually.

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
            χ_trial[i] = maximum([χ_min, minimum([1.0, χn[i]+Δχt])])
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
#md nothing # hide

# Lastly, we use the following helper function to compute the average driving force, which is later
# used to normalize the driving forces. This makes the used material parameters and numerical parameters independent
# of the problem. 

function compute_average_driving_force(mp, pΨ, χn)
    n = length(pΨ)
    w = zeros(n)
    
    for i in 1:n
        w[i] = (χn[i]-mp.χ_min)*(1-χn[i])
    end
    
    p_Ω = sum(w.*pΨ)/sum(w) # average driving force
    
    return p_Ω
end
#md nothing # hide

# Finally, we put everything together to update the density. The loop ensures the stability of the 
# updated solution.

function update_density(dh, states, mp, ρ, topology, Δh)
    n_j = Int(ceil(6*mp.β/(mp.η*Δh^2))) # iterations needed for stability
    χn = compute_densities(states, dh) # old density field    
    χn1 = zeros(length(χn))
    
    for j in 1:n_j
        ∇²χ = approximate_laplacian(dh, topology, χn, Δh) # Laplacian
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
#md nothing # hide
    
# Now, we move on to the Finite Element part of the program. We use the following function to assemble our linear system.

function doassemble!(cellvalues::CellValues, facevalues::FaceValues, K::SparseMatrixCSC, grid::Grid, dh::DofHandler, mp::MaterialParameters, u, states)
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
        
        elmt!(Ke, re, element, cellvalues, facevalues, grid, mp, ue, state)
        assemble!(assembler, celldofs(element), re, Ke)
    end

    return K, r
end
#md nothing # hide

# The element routine is used to calculate the elementwise stiffness matrix and the residual. In contrast to a purely
# elastomechanic problem, for topology optimization we additionally use our material state to receive the density value of
# the element and to store the strain at each quadrature point.

function elmt!(Ke, re, element, cellvalues, facevalues, grid, mp, ue, state)
    n_basefuncs = getnbasefunctions(cellvalues)
    reinit!(cellvalues, element)    
    χ = state.χ    
        
    ## We only assemble lower half triangle of the stiffness matrix and then symmetrize it.
    @inbounds for q_point in 1:getnquadpoints(cellvalues)
        dΩ = getdetJdV(cellvalues, q_point)
        state.ε[q_point] = function_symmetric_gradient(cellvalues, q_point, ue)
        
        for i in 1:n_basefuncs
            δεi = shape_symmetric_gradient(cellvalues, q_point, i)
            δu = shape_value(cellvalues, q_point, i)
            for j in 1:i
                δεj = shape_symmetric_gradient(cellvalues, q_point, j)
                Ke[i,j] += (χ)^(mp.p) * (δεi ⊡ mp.C ⊡ δεj) * dΩ 
            end
            re[i] += (-δεi ⊡ ((χ)^(mp.p) * mp.C ⊡ state.ε[q_point])) * dΩ
        end
    end

    symmetrize_lower!(Ke)

    @inbounds for face in 1:nfaces(element) 
        if onboundary(element, face) && (cellid(element), face) ∈ getfaceset(grid, "traction")
            reinit!(facevalues, element, face)
            t = Vec((0.0, -1.0)) # force pointing downwards
            for q_point in 1:getnquadpoints(facevalues)
                dΓ = getdetJdV(facevalues, q_point)
                for i in 1:n_basefuncs
                    δu = shape_value(facevalues, q_point, i)
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
#md nothing # hide

# We put everything together in the main function. Here the user may choose the radius parameter, which
# is related to the regularization parameter as $\beta = ra^2$, the starting density, the number of elements in vertical direction and finally the
# name of the output. Additionally, the user may choose whether only the final design (default)
# or every iteration step is saved.
#
# First, we compute the material parameters and create the grid, DofHandler, boundary condition and FE values.
# Then we prepare the iterative Newton-Raphson method by pre-allocating all important vectors. Furthermore, 
# we create material states for each element and construct the topology of the grid.
#
# During each iteration step, first we solve our FE problem in the Newton-Raphson loop. With the solution of the
# elastomechanic problem, we check for convergence of our topology design. The criteria has to be fulfilled twice in
# a row to avoid oscillations. If no convergence is reached yet, we update our design and prepare the next iteration step.
# Finally, we output the results in paraview and calculate the relative stiffness of the final design, i.e. how much how
# the stiffness increased compared to the starting point.

function topopt(ra,ρ,n,filename; output=:false) 
    ## material
    mp = MaterialParameters(210.e3, 0.3, 1.e-3, 3.0, ra^2, 15.0) 

    ## grid, dofhandler, boundary condition
    grid = create_grid(n)
    dh = create_dofhandler(grid)
    Δh = 1/n # element edge length
    dbc = create_bc(dh)
    
    ## cellvalues
    cellvalues, facevalues = create_values()
    
    ## Pre-allocate solution vectors, etc.
    n_dofs = ndofs(dh) # total number of dofs
    u  = zeros(n_dofs) # solution vector
    un = zeros(n_dofs) # previous solution vector
    
    Δu = zeros(n_dofs)  # previous displacement correction
    ΔΔu = zeros(n_dofs) # new displacement correction
    
    ## create material states  
    states = [MaterialState(ρ, getnquadpoints(cellvalues)) for _ in 1:getncells(dh.grid)]
    
    χ = zeros(getncells(dh.grid))
        
    r = zeros(n_dofs) # residual
    K = create_matrix(dh) # stiffness matrix
    
    i_max = 300 ## maximum number of iteration steps
    tol = 1e-4
    compliance = 0.0
    compliance_0 = 0.0
    compliance_n = 0.0
    conv = :false
    
    topology = ExclusiveTopology(grid)
    
    ## Newton-Raphson loop
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
        
            ## current guess
            u .= un .+ Δu
            K, r = doassemble!(cellvalues, facevalues, K, grid, dh, mp, u, states);
            norm_r = norm(r[Ferrite.free_dofs(dbc)]) 

            if (norm_r) < NEWTON_TOL
                break
            end  

            apply_zero!(K, r, dbc)
            ΔΔu = Symmetric(K) \ r
            
            apply_zero!(ΔΔu, dbc)
            Δu .+= ΔΔu
        end # of loop while NR-Iteration    

        ## calculate compliance
        compliance = 1/2 * u' * K * u
        
        if(it==1)
            compliance_0 = compliance
        end
        
        ## check convergence criterium (twice!)
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
  
        ## update density
        χ = update_density(dh, states, mp, ρ, topology, Δh)
        
        ## update old displacement, density and compliance
        un .= u
        Δu .= 0.0
        update_material_states!(χ, states, dh)
        compliance_n = compliance
        
        ## output during calculation
        if(output)
            i = @sprintf("%3.3i", it)
            filename_it = string(filename, "_", i)

            vtk_grid(filename_it, grid) do vtk
                vtk_cell_data(vtk, χ, "density")
            end
        end
    end

    ## export converged results
    if(!output)
        vtk_grid(filename, grid) do vtk
            vtk_cell_data(vtk, χ, "density")
        end
    end
    @printf "Rel. stiffness: %.4f \n" compliance^(-1)/compliance_0^(-1)
    
    return
end
#md nothing # hide

# Lastly, we call our main function and compare the results. To create the
# complete output with all iteration steps, it is possible to set the output 
# parameter to `true`.

topopt(0.02, 0.5, 60, "small_radius"; output=:false);
topopt(0.03, 0.5, 60, "large_radius"; output=:false);
##topopt(0.02, 0.5, 60, "topopt_animation"; output=:true); # can be used to create animations

# We observe, that the stiffness for the lower value of $ra$ is higher,
# but also requires more iterations until convergence and finer structures to be manufactured, as can be seen in Figure 2:
#
# ![](bending.png)
#
# *Figure 2*: Optimization results of the bending beam for smaller (left) and larger (right) value of the regularization parameter $\beta$.
#
# To prove mesh independence, the user could vary the mesh resolution and compare the results.

#md # ## References
#md # ```@bibliography
#md # Pages = ["gallery/topology_optimization.md"]
#md # Canonical = false
#md # ```

#md # ## [Plain program](@id topology_optimization-plain-program)
#md #
#md # Here follows a version of the program without any comments.
#md # The file is also available here: [`topology_optimization.jl`](topology_optimization.jl).
#md #
#md # ```julia
#md # @__CODE__
#md # ```
