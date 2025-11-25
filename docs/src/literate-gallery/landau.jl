# # [Ginzburg-Landau model energy minimization](@id tutorial-ginzburg-landau-minimizer)

# ![landau_orig.png](landau_orig.png)

# Original

# ![landau_opt.png](landau_opt.png)

# Optimized

# In this example a basic Ginzburg-Landau model is solved.
# This example gives an idea of how the API together with ForwardDiff can be leveraged to
# performantly solve non standard problems on a FEM grid.
# A large portion of the code is there only for performance reasons,
# but since this usually really matters and is what takes the most time to optimize,
# it is included.

# The key to using a method like this for minimizing a free energy function directly,
# rather than the weak form, as is usually done with FEM, is to split up the
# gradient and Hessian calculations.
# This means that they are performed for each cell separately instead of for the
# grid as a whole.

using ForwardDiff: ForwardDiff, GradientConfig, HessianConfig, Chunk
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
struct ThreadCache{CV, T, DIM, F <: Function, GC <: GradientConfig, HC <: HessianConfig}
    cvP::CV
    element_indices::Vector{Int}
    element_dofs::Vector{T}
    element_gradient::Vector{T}
    element_hessian::Matrix{T}
    element_coords::Vector{Vec{DIM, T}}
    element_potential::F
    gradconf::GC
    hessconf::HC
end
function ThreadCache(dpc::Int, nodespercell, cvP::CellValues, modelparams, elpotential)
    element_indices = zeros(Int, dpc)
    element_dofs = zeros(dpc)
    element_gradient = zeros(dpc)
    element_hessian = zeros(dpc, dpc)
    element_coords = zeros(Vec{3, Float64}, nodespercell)
    potfunc = x -> elpotential(x, cvP, modelparams)
    gradconf = GradientConfig(potfunc, zeros(dpc), Chunk{12}())
    hessconf = HessianConfig(potfunc, zeros(dpc), Chunk{4}())
    return ThreadCache(cvP, element_indices, element_dofs, element_gradient, element_hessian, element_coords, potfunc, gradconf, hessconf)
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

function LandauModel(α, G, gridsize, left::Vec{DIM, T}, right::Vec{DIM, T}, elpotential) where {DIM, T}
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
    caches = [ThreadCache(dpc, cpc, copy(cvP), ModelParams(α, G), elpotential) for t in 1:Threads.maxthreadid()]
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
# This function defines most of the assembly step, since the structure is the same for
# the energy, gradient and Hessian calculations.
function assemble_cell!(f, dofvector, dofhandler, cache, i)
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
    f(cache, eldofs)
end

function assemble_model!(f::F, dofvector, model) where {F}
    dofhandler = model.dofhandler
    for indices in model.threadindices
        Threads.@threads for i in indices
            cache = model.threadcaches[Threads.threadid()]
            assemble_cell!(f, dofvector, dofhandler, cache, i)
        end
    end
end

# This calculates the total energy calculation of the grid
function F(dofvector::Vector{T}, model) where {T}
    out = Threads.Atomic{T}(zero(T))
    @time "F" assemble_model!(dofvector, model) do cache, eldofs
        Threads.atomic_add!(out, cache.element_potential(eldofs))
    end
    return out[]
end

# The gradient calculation for each dof
function ∇F!(∇f::Vector{T}, dofvector::Vector{T}, model::LandauModel{T}) where {T}
    fill!(∇f, zero(T))
    @time "∇F!" assemble_model!(dofvector, model) do cache, eldofs
        ForwardDiff.gradient!(cache.element_gradient, cache.element_potential, eldofs, cache.gradconf)
        @inbounds Ferrite.assemble!(∇f, cache.element_indices, cache.element_gradient)
    end
    return
end

# The Hessian calculation for the whole grid
function ∇²F!(∇²f::SparseMatrixCSC, dofvector::Vector{T}, model::LandauModel{T}) where {T}
    assemblers = [start_assemble(∇²f) for t in 1:Threads.maxthreadid()]
    assemble_model!(dofvector, model) do cache, eldofs
        ForwardDiff.hessian!(cache.element_hessian, cache.element_potential, eldofs, cache.hessconf)
        @inbounds Ferrite.assemble!(assemblers[Threads.threadid()], cache.element_indices, cache.element_hessian)
    end
    return
end

# ## Minimization
# Now everything can be combined to minimize the energy, and find the equilibrium
# configuration.
function minimize!(model)
    dh = model.dofhandler
    dofs = model.dofs
    ∇f = fill(0.0, length(dofs))
    ∇²f = allocate_matrix(dh)
    function g!(storage, x)
        ∇F!(storage, x, model)
        @code_warntype ∇F!(storage, x, model)
        error()
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
    res = optimize(od, model.dofs, Newton(linesearch = BackTracking()), Optim.Options(show_trace = true, show_every = 1, g_tol = 1.0e-20, iterations=5))
    model.dofs .= res.minimizer
    return res
end

# ## Testing it
# This calculates the contribution of each element to the total energy,
# it is also the function that will be put through ForwardDiff for the gradient and Hessian.
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

model_small = LandauModel(α, G, (2, 2, 2), left, right, element_potential)
minimize!(model_small)

model = LandauModel(α, G, (50, 50, 2), left, right, element_potential)
save_landau("landauorig", model)
@time minimize!(model)
save_landau("landaufinal", model)

# as we can see this runs very quickly even for relatively large gridsizes.
# The key to get high performance like this is to minimize the allocations inside the threaded loops,
# ideally to 0.

# 5 iterations
# no thread macro
# 1 thread: 13.154833 seconds (1.26 M allocations: 1.314 GiB, 0.19% gc time, 0.02% compilation time)
# 8 threads:  13.371658 seconds (1.26 M allocations: 1.314 GiB, 0.30% gc time, 0.03% compilation time)


# 1 threads  12.984311 seconds (1.27 M allocations: 1.285 GiB, 0.10% gc time)
# 4 threads   4.725177 seconds (9.38 M allocations: 2.741 GiB, 1.84% gc time, 0.10% compilation time)
# 8 threads   3.386255 seconds (15.52 M allocations: 4.384 GiB, 3.53% gc time, 0.14% compilation time)