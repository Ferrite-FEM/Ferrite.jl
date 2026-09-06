using DifferentiationInterface
import DifferentiationInterface as DI
using DifferentiationInterface: AutoForwardDiff, AutoHyperHessians, prepare_gradient, prepare_hessian
using ForwardDiff: ForwardDiff
using HyperHessians: HyperHessians # loads the `AutoHyperHessians` backend extension for DI
using Ferrite
using Optim, LineSearches
using SparseArrays
using Tensors
using OhMyThreads, ChunkSplitters

function Fl(P::Vec{3, T}, α::Vec{3}) where {T}
    P2 = Vec{3, T}((P[1]^2, P[2]^2, P[3]^2))
    return α[1] * sum(P2) +
        α[2] * (P[1]^4 + P[2]^4 + P[3]^4) +
        α[3] * ((P2[1] * P2[2] + P2[2] * P2[3]) + P2[1] * P2[3])
end

@inline Fg(∇P, G) = 0.5(∇P ⊡ G) ⊡ ∇P

F(P, ∇P, params) = Fl(P, params.α) + Fg(∇P, params.G)

struct ModelParams{V, T}
    α::V
    G::T
end

struct TaskCache{CV, T, DIM, F <: Function, GB, HB, GP, HP}
    cvP::CV
    element_indices::Vector{Int}
    element_dofs::Vector{T}
    element_gradient::Vector{T}
    element_hessian::Matrix{T}
    element_coords::Vector{Vec{DIM, T}}
    element_potential::F
    grad_backend::GB
    hess_backend::HB
    grad_prep::GP
    hess_prep::HP
end
function TaskCache(dpc::Int, nodespercell, cvP::CellValues, modelparams, elpotential, grad_backend, hess_backend)
    element_indices = zeros(Int, dpc)
    element_dofs = zeros(dpc)
    element_gradient = zeros(dpc)
    element_hessian = zeros(dpc, dpc)
    element_coords = zeros(Vec{3, Float64}, nodespercell)
    potfunc = x -> elpotential(x, cvP, modelparams)
    x0 = zeros(dpc)
    grad_prep = prepare_gradient(potfunc, grad_backend, x0)
    hess_prep = prepare_hessian(potfunc, hess_backend, x0)
    return TaskCache(cvP, element_indices, element_dofs, element_gradient, element_hessian, element_coords, potfunc, grad_backend, hess_backend, grad_prep, hess_prep)
end

mutable struct LandauModel{T, DH <: DofHandler, CH <: ConstraintHandler, TC <: TaskCache}
    dofs::Vector{T}
    dofhandler::DH
    boundaryconds::CH
    colors::Vector{Vector{Int}}
    caches::Vector{TC}
end

function LandauModel(
        α, G, gridsize, left::Vec{DIM, T}, right::Vec{DIM, T}, elpotential, ntasks;
        # Chunk size chosen by empirical testing for what had good performance
        # Hessian AD (second order) tends to benefit from smaller chunks than
        # gradients (first order).
        grad_backend = AutoForwardDiff(; chunksize = 12),
        hess_backend = AutoForwardDiff(; chunksize = 4),
    ) where {DIM, T}
    grid = generate_grid(Tetrahedron, gridsize, left, right)
    colors = create_coloring(grid)

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
    caches = [TaskCache(dpc, cpc, copy(cvP), ModelParams(α, G), elpotential, grad_backend, hess_backend) for _ in 1:ntasks]
    return LandauModel(dofvector, dofhandler, boundaryconds, colors, caches)
end

function save_landau(path, model, dofs = model.dofs)
    VTKGridFile(path, model.dofhandler) do vtk
        write_solution(vtk, model.dofhandler, dofs)
    end
    return
end

function setup_cell!(cache, dofhandler, dofvector, cellidx)
    nodeids = dofhandler.grid.cells[cellidx].nodes
    for j in 1:length(cache.element_coords)
        cache.element_coords[j] = dofhandler.grid.nodes[nodeids[j]].x
    end
    reinit!(cache.cvP, cache.element_coords)
    celldofs!(cache.element_indices, dofhandler, cellidx)
    eldofs = cache.element_dofs
    for j in 1:length(eldofs)
        eldofs[j] = dofvector[cache.element_indices[j]]
    end
    return eldofs
end

function F(dofvector::Vector{T}, model) where {T}
    out = zero(T)
    for indices in model.colors
        partial = OhMyThreads.@tasks for (ichunk, range) in enumerate(chunks(indices; n = length(model.caches)))
            OhMyThreads.@set reducer = +
            cache = model.caches[ichunk]
            local_energy = zero(T)
            for i in range
                eldofs = setup_cell!(cache, model.dofhandler, dofvector, i)
                local_energy += cache.element_potential(eldofs)
            end
            local_energy
        end
        out += partial
    end
    return out
end

function ∇F!(∇f::Vector{T}, dofvector::Vector{T}, model::LandauModel{T}) where {T}
    fill!(∇f, zero(T))
    for indices in model.colors
        OhMyThreads.@tasks for (ichunk, range) in enumerate(chunks(indices; n = length(model.caches)))
            cache = model.caches[ichunk]
            for i in range
                eldofs = setup_cell!(cache, model.dofhandler, dofvector, i)
                DI.gradient!(cache.element_potential, cache.element_gradient, cache.grad_prep, cache.grad_backend, eldofs)
                @inbounds assemble!(∇f, cache.element_indices, cache.element_gradient)
            end
        end
    end
    return
end

function ∇²F!(∇²f::SparseMatrixCSC, dofvector::Vector{T}, model::LandauModel{T}) where {T}
    dh = model.dofhandler
    ntasks = length(model.caches)
    assemblers = [start_assemble(∇²f; fillzero = (i == 1)) for i in 1:ntasks]
    for indices in model.colors
        OhMyThreads.@tasks for (ichunk, range) in enumerate(chunks(indices; n = ntasks))
            cache = model.caches[ichunk]
            for i in range
                eldofs = setup_cell!(cache, dh, dofvector, i)
                DI.hessian!(cache.element_potential, cache.element_hessian, cache.hess_prep, cache.hess_backend, eldofs)
                @inbounds assemble!(assemblers[ichunk], cache.element_indices, cache.element_hessian)
            end
        end
    end
    return
end

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
        # apply!(storage, model.boundaryconds)
    end
    f(x) = F(x, model)

    od = TwiceDifferentiable(f, g!, h!, model.dofs, 0.0, ∇f, ∇²f)

    # this way of minimizing is only beneficial when the initial guess is completely off,
    # then a quick couple of ConjuageGradient steps brings us easily closer to the minimum.
    # res = optimize(od, model.dofs, ConjugateGradient(linesearch=BackTracking()), Optim.Options(show_trace=true, show_every=1, g_tol=1e-20, iterations=10))
    # model.dofs .= res.minimizer
    # to get the final convergence, Newton's method is more ideal since the energy landscape should be almost parabolic
    ##+
    res = optimize(od, model.dofs, Newton(linesearch = BackTracking()), Optim.Options(show_trace = true, show_every = 1, g_tol = 1.0e-20))
    model.dofs .= res.minimizer
    return res
end

function element_potential(eldofs::AbstractVector{T}, cvP, params) where {T}
    energy = zero(T)
    for qp in 1:getnquadpoints(cvP)
        P = function_value(cvP, qp, eldofs)
        ∇P = function_gradient(cvP, qp, eldofs)
        energy += F(P, ∇P, params) * getdetJdV(cvP, qp)
    end
    return energy
end

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
model = LandauModel(α, G, (50, 50, 2), left, right, element_potential, Threads.nthreads(); hess_backend = AutoHyperHessians(; chunksize = 4));

save_landau("landauorig", model)
@time res = minimize!(model)
@assert Optim.converged(res)
save_landau("landaufinal", model)

using Test # src
@test Optim.minimum(res) ≈ -10858.806775 # src

function time_hessian_assembly(model)
    ∇²f = allocate_matrix(model.dofhandler)
    ∇²F!(∇²f, model.dofs, model) # warmup / compilation
    @time ∇²F!(∇²f, model.dofs, model)
    return ∇²f
end

model_fd = LandauModel(α, G, (50, 50, 2), left, right, element_potential, Threads.nthreads(); hess_backend = AutoForwardDiff(; chunksize = 4));
model_hh = LandauModel(α, G, (50, 50, 2), left, right, element_potential, Threads.nthreads(); hess_backend = AutoHyperHessians(; chunksize = 4));
model_fd.dofs .= model.dofs
model_hh.dofs .= model.dofs

println("ForwardDiff Hessian assembly:")
H_fd = time_hessian_assembly(model_fd)
println("HyperHessians Hessian assembly:")
H_hh = time_hessian_assembly(model_hh)

# This file was generated using Literate.jl, https://github.com/fredrikekre/Literate.jl
