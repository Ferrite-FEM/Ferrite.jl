# # [Ginzburg-Landau model energy minimization](@id tutorial-ginzburg-landau-minimizer)

# ![landau_orig.png](landau_orig.png)

# Original

# ![landau_opt.png](landau_opt.png)

# Optimized

# In this example a basic Ginzburg-Landau model is solved.
# This example gives an idea of how the API together with DifferentiationInterface.jl
# (using ForwardDiff as backend) can be leveraged to performantly solve non standard
# problems on a FEM grid.
# A large portion of the code is there only for performance reasons,
# but since this usually really matters and is what takes the most time to optimize,
# it is included.

# The key to using a method like this for minimizing a free energy function directly,
# rather than the weak form, as is usually done with FEM, is to split up the
# gradient and Hessian calculations.
# This means that they are performed for each cell separately instead of for the
# grid as a whole.

using DifferentiationInterface
using ForwardDiff: ForwardDiff
using Ferrite
using Optim, LineSearches
using SparseArrays
using Tensors

# ## Energy terms
# ### 4th order Landau free energy
function Fl(P::Vec{3, T}, α::Vec{3}) where {T}
    P2 = Vec{3, T}((P[1]^2, P[2]^2, P[3]^2))
    return α[1] * sum(P2) +
        α[2] * (P[1]^4 + P[2]^4 + P[3]^4) +
        α[3] * ((P2[1] * P2[2] + P2[2] * P2[3]) + P2[1] * P2[3])
end
# ### Ginzburg free energy
@inline Fg(∇P, G) = 0.5(∇P ⊡ G) ⊡ ∇P
# ### GL free energy
F(P, ∇P, params) = Fl(P, params.α) + Fg(∇P, params.G)

# ### Parameters that characterize the model
struct ModelParams{V, T}
    α::V
    G::T
end

# ### ThreadCache
# This holds the values that each thread will use during the assembly.
struct ThreadCache{CV, T, DIM, F <: Function, GP, HP, GB, HB}
    cvP::CV
    element_indices::Vector{Int}
    element_dofs::Vector{T}
    element_gradient::Vector{T}
    element_hessian::Matrix{T}
    element_coords::Vector{Vec{DIM, T}}
    element_potential::F
    grad_prep::GP
    hess_prep::HP
    grad_backend::GB
    hess_backend::HB
end
function ThreadCache(dpc::Int, nodespercell, cvP::CellValues, modelparams, elpotential, grad_backend, hess_backend)
    element_indices = zeros(Int, dpc)
    element_dofs = zeros(dpc)
    element_gradient = zeros(dpc)
    element_hessian = zeros(dpc, dpc)
    element_coords = zeros(Vec{3, Float64}, nodespercell)
    potfunc = x -> elpotential(x, cvP, modelparams)
    grad_prep = prepare_gradient(potfunc, grad_backend, zeros(dpc))
    hess_prep = prepare_hessian(potfunc, hess_backend, zeros(dpc))
    return ThreadCache(cvP, element_indices, element_dofs, element_gradient, element_hessian, element_coords, potfunc, grad_prep, hess_prep, grad_backend, hess_backend)
end

# ## The Model
# everything is combined into a model.
mutable struct LandauModel{T, DH <: DofHandler, CH <: ConstraintHandler, TC <: ThreadCache}
    dofs::Vector{T}
    dofhandler::DH
    boundaryconds::CH
    threadindices::Vector{Vector{Int}}
    threadcaches::Vector{TC}
end

function LandauModel(α, G, gridsize, left::Vec{DIM, T}, right::Vec{DIM, T}, elpotential, grad_backend, hess_backend) where {DIM, T}
    grid = generate_grid(Tetrahedron, gridsize, left, right)
    threadindices = Ferrite.create_coloring(grid)

    qr = QuadratureRule{RefTetrahedron}(2)
    ipP = Lagrange{RefTetrahedron, 1}()^3
    cvP = CellValues(qr, ipP)

    dofhandler = DofHandler(grid)
    add!(dofhandler, :P, ipP)
    close!(dofhandler)

    dofvector = zeros(ndofs(dofhandler))
    startingconditions!(dofvector, dofhandler)
    boundaryconds = ConstraintHandler(dofhandler)
    #boundary conditions can be added but aren't necessary for optimization
    #add!(boundaryconds, Dirichlet(:P, getfacetset(grid, "left"), (x, t) -> [0.0,0.0,0.53], [1,2,3]))
    #add!(boundaryconds, Dirichlet(:P, getfacetset(grid, "right"), (x, t) -> [0.0,0.0,-0.53], [1,2,3]))
    close!(boundaryconds)
    update!(boundaryconds, 0.0)

    apply!(dofvector, boundaryconds)

    dpc = ndofs_per_cell(dofhandler)
    cpc = length(grid.cells[1].nodes)
    caches = [ThreadCache(dpc, cpc, copy(cvP), ModelParams(α, G), elpotential, grad_backend, hess_backend) for t in 1:Threads.maxthreadid()]
    return LandauModel(dofvector, dofhandler, boundaryconds, threadindices, caches)
end

# utility to quickly save a model
function save_landau(path, model, dofs = model.dofs)
    VTKGridFile(path, model.dofhandler) do vtk
        write_solution(vtk, model.dofhandler, dofs)
    end
    return
end

# ## Assembly
# This macro defines most of the assembly step, since the structure is the same for
# the energy, gradient and Hessian calculations.
macro assemble!(innerbody)
    return esc(
        quote
            dofhandler = model.dofhandler
            for indices in model.threadindices
                Threads.@threads for i in indices
                    cache = model.threadcaches[Threads.threadid()]
                    eldofs = cache.element_dofs
                    nodeids = dofhandler.grid.cells[i].nodes
                    for j in 1:length(cache.element_coords)
                        cache.element_coords[j] = dofhandler.grid.nodes[nodeids[j]].x
                    end
                    reinit!(cache.cvP, cache.element_coords)

                    celldofs!(cache.element_indices, dofhandler, i)
                    for j in 1:length(cache.element_dofs)
                        eldofs[j] = dofvector[cache.element_indices[j]]
                    end
                    $innerbody
                end
            end
        end
    )
end

# This calculates the total energy calculation of the grid
function F(dofvector::Vector{T}, model) where {T}
    outs = fill(zero(T), Threads.maxthreadid())
    @assemble! begin
        outs[Threads.threadid()] += cache.element_potential(eldofs)
    end
    return sum(outs)
end

# The gradient calculation for each dof
function ∇F!(∇f::Vector{T}, dofvector::Vector{T}, model::LandauModel{T}) where {T}
    fill!(∇f, zero(T))
    @assemble! begin
        gradient!(cache.element_potential, cache.element_gradient, cache.grad_prep, cache.grad_backend, eldofs)
        @inbounds assemble!(∇f, cache.element_indices, cache.element_gradient)
    end
    return
end

# The Hessian calculation for the whole grid
function ∇²F!(∇²f::SparseMatrixCSC, dofvector::Vector{T}, model::LandauModel{T}) where {T}
    assemblers = [start_assemble(∇²f) for t in 1:Threads.maxthreadid()]
    @assemble! begin
        hessian!(cache.element_potential, cache.element_hessian, cache.hess_prep, cache.hess_backend, eldofs)
        @inbounds assemble!(assemblers[Threads.threadid()], cache.element_indices, cache.element_hessian)
    end
    return
end

# We can also calculate all things in one go!
function calcall(∇²f::SparseMatrixCSC, ∇f::Vector{T}, dofvector::Vector{T}, model::LandauModel{T}) where {T}
    outs = fill(zero(T), Threads.maxthreadid())
    fill!(∇f, zero(T))
    assemblers = [start_assemble(∇²f, ∇f) for t in 1:Threads.maxthreadid()]
    @assemble! begin
        outs[Threads.threadid()] += cache.element_potential(eldofs)
        value_gradient_and_hessian!(cache.element_potential, cache.element_gradient, cache.element_hessian, cache.hess_prep, cache.hess_backend, eldofs)
        @inbounds assemble!(assemblers[Threads.threadid()], cache.element_indices, cache.element_gradient, cache.element_hessian)
    end
    return sum(outs)
end

# ## Minimization
# Now everything can be combined to minimize the energy, and find the equilibrium
# configuration.
function minimize!(model; kwargs...)
    dh = model.dofhandler
    dofs = model.dofs
    ∇f = fill(0.0, length(dofs))
    ∇²f = allocate_matrix(dh)
    function g!(storage, x)
        ∇F!(storage, x, model)
        return apply_zero!(storage, model.boundaryconds)
    end
    function h!(storage, x)
        return ∇²F!(storage, x, model)
        ## apply!(storage, model.boundaryconds)
    end
    f(x) = F(x, model)

    od = TwiceDifferentiable(f, g!, h!, model.dofs, 0.0, ∇f, ∇²f)

    ## this way of minimizing is only beneficial when the initial guess is completely off,
    ## then a quick couple of ConjuageGradient steps brings us easily closer to the minimum.
    ## res = optimize(od, model.dofs, ConjugateGradient(linesearch=BackTracking()), Optim.Options(show_trace=true, show_every=1, g_tol=1e-20, iterations=10))
    ## model.dofs .= res.minimizer
    ## to get the final convergence, Newton's method is more ideal since the energy landscape should be almost parabolic
    ##+
    res = optimize(od, model.dofs, Newton(linesearch = BackTracking()), Optim.Options(show_trace = true, show_every = 1, g_tol = 1.0e-20))
    model.dofs .= res.minimizer
    return res
end

# ## Testing it
# This calculates the contribution of each element to the total energy,
# it is also the function that will be differentiated for the gradient and Hessian.
function element_potential(eldofs::AbstractVector{T}, cvP, params) where {T}
    energy = zero(T)
    for qp in 1:getnquadpoints(cvP)
        P = function_value(cvP, qp, eldofs)
        ∇P = function_gradient(cvP, qp, eldofs)
        energy += F(P, ∇P, params) * getdetJdV(cvP, qp)
    end
    return energy
end

# now we define some starting conditions
function startingconditions!(dofvector, dofhandler)
    for cell in CellIterator(dofhandler)
        globaldofs = celldofs(cell)
        it = 1
        for i in 1:3:length(globaldofs)
            dofvector[globaldofs[i]] = -2.0
            dofvector[globaldofs[i + 1]] = 2.0
            dofvector[globaldofs[i + 2]] = -2.0tanh(cell.coords[it][1] / 20)
            it += 1
        end
    end
    return
end

δ(i, j) = i == j ? one(i) : zero(i)
V2T(p11, p12, p44) = Tensor{4, 3}((i, j, k, l) -> p11 * δ(i, j) * δ(k, l) * δ(i, k) + p12 * δ(i, j) * δ(k, l) * (1 - δ(i, k)) + p44 * δ(i, k) * δ(j, l) * (1 - δ(i, j)))

G = V2T(1.0e2, 0.0, 1.0e2)
α = Vec{3}((-1.0, 1.0, 1.0))
left = Vec{3}((-75.0, -25.0, -2.0))
right = Vec{3}((75.0, 25.0, 2.0))
grad_backend = AutoForwardDiff(; chunksize = 12)
hess_backend = AutoForwardDiff(; chunksize = 4)
model = LandauModel(α, G, (50, 50, 2), left, right, element_potential, grad_backend, hess_backend)

save_landau("landauorig", model)
@time minimize!(model)
save_landau("landaufinal", model)

# as we can see this runs very quickly even for relatively large gridsizes.
# The key to get high performance like this is to minimize the allocations inside the threaded loops,
# ideally to 0.
