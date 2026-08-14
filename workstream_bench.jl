# Benchmark: WorkStream mesh_loop vs OhMyThreads-based assembly (howto style)
# Run with e.g.: julia --project --threads=8 workstream_bench.jl [n]
using Ferrite, SparseArrays, LinearAlgebra, OhMyThreads

const n = isempty(ARGS) ? 12 : parse(Int, ARGS[1])

function create_material_stiffness()
    E = 200.0e9; ν = 0.3
    λ = E * ν / ((1 + ν) * (1 - 2ν)); μ = E / (2(1 + ν))
    δ(i, j) = i == j ? 1.0 : 0.0
    return SymmetricTensor{4, 3}((i, j, k, l) -> λ * δ(i, j) * δ(k, l) + μ * (δ(i, k) * δ(j, l) + δ(i, l) * δ(j, k)))
end

function assemble_cell_kernel!(Ke, fe, cv, ε, C, b)
    fill!(Ke, 0); fill!(fe, 0)
    nbf = getnbasefunctions(cv)
    @inbounds for qp in 1:getnquadpoints(cv)
        for i in 1:nbf
            ε[i] = shape_symmetric_gradient(cv, qp, i)
        end
        dΩ = getdetJdV(cv, qp)
        for i in 1:nbf
            δu = shape_value(cv, qp, i)
            fe[i] += (δu ⋅ b) * dΩ
            εC = ε[i] ⊡ C
            for j in 1:nbf
                Ke[i, j] += (εC ⊡ ε[j]) * dΩ
            end
        end
    end
    return
end

# ------------------------------------------------------------------
# 1. Serial baseline
function assemble_serial!(K, f, dh, cv, C, b)
    asm = start_assemble(K, f)
    ε = [zero(SymmetricTensor{2, 3}) for _ in 1:getnbasefunctions(cv)]
    npc = ndofs_per_cell(dh)
    Ke = zeros(npc, npc); fe = zeros(npc)
    cc = CellCache(dh)
    for cid in 1:getncells(dh.grid)
        reinit!(cc, cid)
        reinit!(cv, cc)
        assemble_cell_kernel!(Ke, fe, cv, ε, C, b)
        assemble!(asm, celldofs(cc), Ke, fe)
    end
    return
end

# Serial loop with atomic/non-atomic assembler, optionally with the kernel stubbed out
# to isolate the scatter (assemble!) cost
function assemble_serial_cmp!(K, f, dh, cv, C, b, ::Val{atomic}, do_kernel::Bool) where {atomic}
    asm = start_assemble(K, f; atomic = atomic)
    ε = [zero(SymmetricTensor{2, 3}) for _ in 1:getnbasefunctions(cv)]
    npc = ndofs_per_cell(dh)
    Ke = rand(npc, npc); fe = rand(npc)
    cc = CellCache(dh)
    for cid in 1:getncells(dh.grid)
        reinit!(cc, cid)
        if do_kernel
            reinit!(cv, cc)
            assemble_cell_kernel!(Ke, fe, cv, ε, C, b)
        end
        assemble!(asm, celldofs(cc), Ke, fe)
    end
    return
end

# Serial loop visiting cells in a given order, to isolate the memory locality effect of
# color-order traversal
function assemble_serial_order!(K, f, dh, cv, C, b, order)
    asm = start_assemble(K, f)
    ε = [zero(SymmetricTensor{2, 3}) for _ in 1:getnbasefunctions(cv)]
    npc = ndofs_per_cell(dh)
    Ke = zeros(npc, npc); fe = zeros(npc)
    cc = CellCache(dh)
    for cid in order
        reinit!(cc, cid)
        reinit!(cv, cc)
        assemble_cell_kernel!(Ke, fe, cv, ε, C, b)
        assemble!(asm, celldofs(cc), Ke, fe)
    end
    return
end

# ------------------------------------------------------------------
# 2./3. Howto style: OhMyThreads @tasks (colored and atomic)
struct ScratchData{CC, CV, T, A}
    cell_cache::CC
    cv::CV
    ε::Vector{SymmetricTensor{2, 3, Float64, 6}}
    Ke::Matrix{T}
    fe::Vector{T}
    assembler::A
end
function ScratchData(dh, K, f, cv::CellValues, ::Val{atomic} = Val(false)) where {atomic}
    npc = ndofs_per_cell(dh)
    ε = [zero(SymmetricTensor{2, 3}) for _ in 1:getnbasefunctions(cv)]
    asm = start_assemble(K, f; fillzero = false, atomic = atomic)
    return ScratchData(CellCache(dh), copy(cv), ε, zeros(npc, npc), zeros(npc), asm)
end

function assemble_omt_colored!(K, f, dh, colors, cv_template, C, b; ntasks = Threads.nthreads())
    _ = start_assemble(K, f) # zero out
    for color in colors
        scheduler = OhMyThreads.DynamicScheduler(; ntasks)
        OhMyThreads.@tasks for cellidx in color
            @set scheduler = scheduler
            @local scratch = ScratchData(dh, K, f, cv_template)
            (; cell_cache, cv, ε, Ke, fe, assembler) = scratch
            reinit!(cell_cache, cellidx)
            reinit!(cv, cell_cache)
            assemble_cell_kernel!(Ke, fe, cv, ε, C, b)
            assemble!(assembler, celldofs(cell_cache), Ke, fe)
        end
    end
    return
end

function assemble_omt_atomic!(K, f, dh, cv_template, C, b; ntasks = Threads.nthreads())
    _ = start_assemble(K, f) # zero out
    scheduler = OhMyThreads.DynamicScheduler(; ntasks)
    OhMyThreads.@tasks for cellidx in 1:getncells(dh.grid)
        @set scheduler = scheduler
        @local scratch = ScratchData(dh, K, f, cv_template, Val(true))
        (; cell_cache, cv, ε, Ke, fe, assembler) = scratch
        reinit!(cell_cache, cellidx)
        reinit!(cv, cell_cache)
        assemble_cell_kernel!(Ke, fe, cv, ε, C, b)
        assemble!(assembler, celldofs(cell_cache), Ke, fe)
    end
    return
end

# ------------------------------------------------------------------
# 4./5. WorkStream mesh_loop (colored and uncolored)
struct WSScratch{CV}
    cv::CV
    ε::Vector{SymmetricTensor{2, 3, Float64, 6}}
end
Base.copy(s::WSScratch) = WSScratch(copy(s.cv), copy(s.ε))

# Copy data for the colored variant: carries a per-task assembler
struct WSCopyDataAsm{A}
    dofs::Vector{Int}
    Ke::Matrix{Float64}
    fe::Vector{Float64}
    asm::A
end
function Base.copy(d::WSCopyDataAsm)
    asm = start_assemble(d.asm.K, d.asm.f; fillzero = false)
    return WSCopyDataAsm(copy(d.dofs), copy(d.Ke), copy(d.fe), asm)
end

function assemble_ws_colored!(K, f, dh, colors, cv_template, C, b; ntasks = Threads.nthreads(), chunk_size = nothing)
    _ = start_assemble(K, f) # zero out
    npc = ndofs_per_cell(dh)
    ε = [zero(SymmetricTensor{2, 3}) for _ in 1:getnbasefunctions(cv_template)]
    scratch_sample = WSScratch(copy(cv_template), ε)
    copy_sample = WSCopyDataAsm(zeros(Int, npc), zeros(npc, npc), zeros(npc), start_assemble(K, f; fillzero = false))
    cell_worker = function (cc, scratch, data)
        reinit!(scratch.cv, cc)
        assemble_cell_kernel!(data.Ke, data.fe, scratch.cv, scratch.ε, C, b)
        copyto!(data.dofs, cc.dofs)
        return
    end
    copier = function (data)
        assemble!(data.asm, data.dofs, data.Ke, data.fe)
        return
    end
    Ferrite.mesh_loop(dh, colors, cell_worker, copier, scratch_sample, copy_sample; ntasks, chunk_size)
    return
end

struct WSCopyData
    dofs::Vector{Int}
    Ke::Matrix{Float64}
    fe::Vector{Float64}
end
Base.copy(d::WSCopyData) = WSCopyData(copy(d.dofs), copy(d.Ke), copy(d.fe))

function assemble_ws_uncolored!(K, f, dh, cv_template, C, b; kwargs...)
    asm = start_assemble(K, f) # zero out; single copier task uses this
    npc = ndofs_per_cell(dh)
    ε = [zero(SymmetricTensor{2, 3}) for _ in 1:getnbasefunctions(cv_template)]
    scratch_sample = WSScratch(copy(cv_template), ε)
    copy_sample = WSCopyData(zeros(Int, npc), zeros(npc, npc), zeros(npc))
    cell_worker = function (cc, scratch, data)
        reinit!(scratch.cv, cc)
        assemble_cell_kernel!(data.Ke, data.fe, scratch.cv, scratch.ε, C, b)
        copyto!(data.dofs, cc.dofs)
        return
    end
    copier = function (data)
        assemble!(asm, data.dofs, data.Ke, data.fe)
        return
    end
    Ferrite.mesh_loop(dh, cell_worker, copier, scratch_sample, copy_sample; kwargs...)
    return
end

# ------------------------------------------------------------------
function main()
    grid = generate_grid(Hexahedron, (10 * n, n, n), Vec{3}((0.0, 0.0, 0.0)), Vec{3}((10.0, 1.0, 1.0)))
    tcolor = @elapsed colors = create_coloring(grid)
    ip = Lagrange{RefHexahedron, 1}()^3
    dh = DofHandler(grid); add!(dh, :u, ip); close!(dh)
    K = allocate_matrix(dh)
    f = zeros(ndofs(dh))
    cv = CellValues(QuadratureRule{RefHexahedron}(2), ip)
    C = create_material_stiffness()
    b = Vec{3}((0.0, 0.0, -1.0))

    println("ncells = $(getncells(grid)), ndofs = $(ndofs(dh)), nthreads = $(Threads.nthreads()), ncolors = $(length(colors))")
    println("coloring time: $(round(tcolor; digits = 3)) s")

    variants = [
        "serial" => () -> assemble_serial!(K, f, dh, cv, C, b),
        "omt colored" => () -> assemble_omt_colored!(K, f, dh, colors, cv, C, b),
        "omt atomic" => () -> assemble_omt_atomic!(K, f, dh, cv, C, b),
        "ws colored" => () -> assemble_ws_colored!(K, f, dh, colors, cv, C, b),
        "ws uncolored" => () -> assemble_ws_uncolored!(K, f, dh, cv, C, b),
        "ws colored (chunk=8)" => () -> assemble_ws_colored!(K, f, dh, colors, cv, C, b; chunk_size = 8),
        "ws uncolored (chunk=8)" => () -> assemble_ws_uncolored!(K, f, dh, cv, C, b; chunk_size = 8),
    ]

    if haskey(ENV, "COLOR_ORDER")
        for (name, order) in ("natural order" => collect(1:getncells(grid)), "color order" => reduce(vcat, colors))
            assemble_serial_order!(K, f, dh, cv, C, b, order) # warmup
            best = minimum(@elapsed(assemble_serial_order!(K, f, dh, cv, C, b, order)) for _ in 1:5)
            println("serial, ", rpad(name, 15), ": ", round(best * 1000; digits = 1), " ms")
        end
        return
    end

    if haskey(ENV, "ATOMIC_CMP")
        for do_kernel in (true, false), atomic in (Val(false), Val(true))
            assemble_serial_cmp!(K, f, dh, cv, C, b, atomic, do_kernel) # warmup
            best = minimum(@elapsed(assemble_serial_cmp!(K, f, dh, cv, C, b, atomic, do_kernel)) for _ in 1:5)
            println("serial, kernel=$(rpad(do_kernel, 5)), atomic=$(atomic): ", round(best * 1000; digits = 1), " ms")
        end
        return
    end

    if haskey(ENV, "WS_SWEEP")
        assemble_ws_uncolored!(K, f, dh, cv, C, b) # warmup
        for nt in (2, 4, 6, 7, 8)
            best = minimum(@elapsed(assemble_ws_uncolored!(K, f, dh, cv, C, b; ntasks = nt)) for _ in 1:3)
            println("ws uncolored ntasks=$nt: ", round(best * 1000; digits = 1), " ms")
        end
        return
    end

    refnorm = Ref(0.0)
    for (name, fun) in variants
        fun() # warmup/compile
        best = minimum(@elapsed(fun()) for _ in 1:3)
        nrm = norm(K.nzval) + norm(f)
        if name == "serial"
            refnorm[] = nrm
        end
        ok = nrm == refnorm[] ? "exact" :
            isapprox(nrm, refnorm[]; rtol = 1.0e-10) ? "ok" : "WRONG ($(nrm) vs $(refnorm[]))"
        println(rpad(name, 24), lpad(round(best * 1000; digits = 1), 8), " ms   check: ", ok)
    end
    return
end

main()
