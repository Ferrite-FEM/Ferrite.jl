using ForwardDiff
import ForwardDiff: GradientConfig, HessianConfig, Chunk
using Ferrite
using Optim, LineSearches
using SparseArrays
using Tensors
using Base.Threads

function Fl(P::Vec{3, T}, α::Vec{3}) where T
    P2 = Vec{3, T}((P[1]^2, P[2]^2, P[3]^2))
    return (α[1] * sum(P2) +
           α[2] * (P[1]^4 + P[2]^4 + P[3]^4)) +
           α[3] * ((P2[1] * P2[2]  + P2[2]*P2[3]) + P2[1]*P2[3])
end

@inline Fg(∇P, G) = 0.5(∇P ⊡ G) ⊡ ∇P

F(P, ∇P, params)  = Fl(P, params.α) + Fg(∇P, params.G)

struct ModelParams{V, T}
    α::V
    G::T
end

struct ThreadCache{CV, T, DIM, F <: Function, GC <: GradientConfig, HC <: HessianConfig}
    cvP              ::CV
    element_indices  ::Vector{Int}
    element_dofs     ::Vector{T}
    element_gradient ::Vector{T}
    element_hessian  ::Matrix{T}
    element_coords   ::Vector{Vec{DIM, T}}
    element_potential::F
    gradconf         ::GC
    hessconf         ::HC
end
function ThreadCache(dpc::Int, nodespercell, cvP::CellValues{DIM, T}, modelparams, elpotential) where {DIM, T}
    element_indices  = zeros(Int, dpc)
    element_dofs     = zeros(dpc)
    element_gradient = zeros(dpc)
    element_hessian  = zeros(dpc, dpc)
    element_coords   = zeros(Vec{DIM, T}, nodespercell)
    potfunc          = x -> elpotential(x, cvP, modelparams)
    gradconf         = GradientConfig(potfunc, zeros(dpc), Chunk{12}())
    hessconf         = HessianConfig(potfunc, zeros(dpc), Chunk{12}())
    return ThreadCache(cvP, element_indices, element_dofs, element_gradient, element_hessian, element_coords, potfunc, gradconf, hessconf)
end

mutable struct LandauModel{T, DH <: DofHandler, CH <: ConstraintHandler, TC <: ThreadCache}
    dofs          ::Vector{T}
    dofhandler    ::DH
    boundaryconds ::CH
    threadindices ::Vector{Vector{Int}}
    threadcaches  ::Vector{TC}
end

function LandauModel(α, G, gridsize, left::Vec{DIM, T}, right::Vec{DIM, T}, elpotential) where {DIM, T}
    grid = generate_grid(Tetrahedron, gridsize, left, right)
    threadindices = Ferrite.create_coloring(grid)

    qr  = QuadratureRule{DIM, RefTetrahedron}(2)
    cvP = CellVectorValues(qr, Lagrange{DIM, RefTetrahedron, 1}())

    dofhandler = DofHandler(grid)
    push!(dofhandler, :P, 3)
    close!(dofhandler)

    dofvector = zeros(ndofs(dofhandler))
    startingconditions!(dofvector, dofhandler)
    boundaryconds = ConstraintHandler(dofhandler)
    #boundary conditions can be added but aren't necessary for optimization
    #add!(boundaryconds, Dirichlet(:P, getfaceset(grid, "left"), (x, t) -> [0.0,0.0,0.53], [1,2,3]))
    #add!(boundaryconds, Dirichlet(:P, getfaceset(grid, "right"), (x, t) -> [0.0,0.0,-0.53], [1,2,3]))
    close!(boundaryconds)
    update!(boundaryconds, 0.0)

    apply!(dofvector, boundaryconds)

    hessian = create_sparsity_pattern(dofhandler)
    dpc = ndofs_per_cell(dofhandler)
    cpc = length(grid.cells[1].nodes)
    caches = [ThreadCache(dpc, cpc, copy(cvP), ModelParams(α, G), elpotential) for t=1:nthreads()]
    return LandauModel(dofvector, dofhandler, boundaryconds, threadindices, caches)
end

function Ferrite.vtk_save(path, model, dofs=model.dofs)
    vtkfile = vtk_grid(path, model.dofhandler)
    vtk_point_data(vtkfile, model.dofhandler, dofs)
    vtk_save(vtkfile)
end

macro assemble!(innerbody)
    esc(quote
        dofhandler = model.dofhandler
        for indices in model.threadindices
            @threads for i in indices
                cache     = model.threadcaches[threadid()]
                eldofs    = cache.element_dofs
                nodeids   = dofhandler.grid.cells[i].nodes
                for j=1:length(cache.element_coords)
                    cache.element_coords[j] = dofhandler.grid.nodes[nodeids[j]].x
                end
                reinit!(cache.cvP, cache.element_coords)

                celldofs!(cache.element_indices, dofhandler, i)
                for j=1:length(cache.element_dofs)
                    eldofs[j] = dofvector[cache.element_indices[j]]
                end
                $innerbody
            end
        end
    end)
end

function F(dofvector::Vector{T}, model) where T
    outs = fill(zero(T), nthreads())
    @assemble! begin
        outs[threadid()] += cache.element_potential(eldofs)
    end
    return sum(outs)
end

function ∇F!(∇f::Vector{T}, dofvector::Vector{T}, model::LandauModel{T}) where T
    fill!(∇f, zero(T))
    @assemble! begin
        ForwardDiff.gradient!(cache.element_gradient, cache.element_potential, eldofs, cache.gradconf)
        @inbounds assemble!(∇f, cache.element_indices, cache.element_gradient)
    end
end

function ∇²F!(∇²f::SparseMatrixCSC, dofvector::Vector{T}, model::LandauModel{T}) where T
    assemblers = [start_assemble(∇²f) for t=1:nthreads()]
    @assemble! begin
        ForwardDiff.hessian!(cache.element_hessian, cache.element_potential, eldofs, cache.hessconf)
        @inbounds assemble!(assemblers[threadid()], cache.element_indices, cache.element_hessian)
    end
end

function calcall(∇²f::SparseMatrixCSC, ∇f::Vector{T}, dofvector::Vector{T}, model::LandauModel{T}) where T
    outs = fill(zero(T), nthreads())
    fill!(∇f, zero(T))
    assemblers = [start_assemble(∇²f, ∇f) for t=1:nthreads()]
    @assemble! begin
        outs[threadid()] += cache.element_potential(eldofs)
        ForwardDiff.hessian!(cache.element_hessian, cache.element_potential, eldofs, cache.hessconf)
        ForwardDiff.gradient!(cache.element_gradient, cache.element_potential, eldofs, cache.gradconf)
        @inbounds assemble!(assemblers[threadid()], cache.element_indices, cache.element_gradient, cache.element_hessian)
    end
    return sum(outs)
end

function minimize!(model; kwargs...)
    dh = model.dofhandler
    dofs = model.dofs
    ∇f = fill(0.0, length(dofs))
    ∇²f = create_sparsity_pattern(dh)
    function g!(storage, x)
        ∇F!(storage, x, model)
        apply_zero!(storage, model.boundaryconds)
    end
    function h!(storage, x)
        ∇²F!(storage, x, model)
        #apply!(storage, model.boundaryconds)
    end
    f(x) = F(x, model)

    od = TwiceDifferentiable(f, g!, h!, model.dofs, 0.0, ∇f, ∇²f)

    res = optimize(od, model.dofs, Newton(linesearch=BackTracking()), Optim.Options(show_trace=true, show_every=1, g_tol=1e-20))
    model.dofs .= res.minimizer
    return res
end

function element_potential(eldofs::AbstractVector{T}, cvP, params) where T
    energy = zero(T)
    for qp=1:getnquadpoints(cvP)
        P  = function_value(cvP, qp, eldofs)
        ∇P = function_gradient(cvP, qp, eldofs)
        energy += F(P, ∇P, params) * getdetJdV(cvP, qp)
    end
    return energy
end

function startingconditions!(dofvector, dofhandler)
    for cell in CellIterator(dofhandler)
        globaldofs = celldofs(cell)
        it = 1
        for i=1:3:length(globaldofs)
            dofvector[globaldofs[i]]   = -2.0
            dofvector[globaldofs[i+1]] = 2.0
            dofvector[globaldofs[i+2]] = -2.0tanh(cell.coords[it][1]/20)
            it += 1
        end
    end
end

δ(i, j) = i == j ? one(i) : zero(i)
V2T(p11, p12, p44) = Tensor{4, 3}((i,j,k,l) -> p11 * δ(i,j)*δ(k,l)*δ(i,k) + p12*δ(i,j)*δ(k,l)*(1 - δ(i,k)) + p44*δ(i,k)*δ(j,l)*(1 - δ(i,j)))

G = V2T(1.0e2, 0.0, 1.0e2)
α = Vec{3}((-1.0, 1.0, 1.0))
left = Vec{3}((-75.,-25.,-2.))
right = Vec{3}((75.,25.,2.))
model = LandauModel(α, G, (50, 50, 2), left, right, element_potential)

vtk_save("landauorig", model)
@time minimize!(model)
vtk_save("landaufinal", model)

# This file was generated using Literate.jl, https://github.com/fredrikekre/Literate.jl

