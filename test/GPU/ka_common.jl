# Shared problem setup, kernels and CPU references for the KernelAbstractions assembly
# tests. Included by `test/test_ka_cpu.jl` (KA.CPU backend, runs in the regular test
# suite) and by `test/GPU/runtests.jl` (vendor backends). The including file defines
# `backend::KA.Backend` and `sparse_type(Tv, Ti)` before including `heat_assembly.jl`.
using Ferrite, Test, SparseArrays
using FerriteGmsh
import Adapt: adapt
import KernelAbstractions as KA
import KernelAbstractions: @kernel, @index

# Deliberately small so that a color with few cells still exercises several blocks.
function compute_threads_and_blocks(n)
    MAX_NUM_THREADS = 8
    NUM_TASKS_PER_THREAD = 2
    tasks_per_thread = min(NUM_TASKS_PER_THREAD, n)
    n_effective = cld(n, tasks_per_thread)
    threads = min(MAX_NUM_THREADS, n_effective)
    blocks = cld(n, tasks_per_thread * threads)
    return threads, blocks
end

# Heat equation element routine (allocation free, as required on device).
function assemble_element!(Ke::AbstractMatrix, fe::AbstractVector, cv::CellValues)
    n_basefuncs = getnbasefunctions(cv)
    for q_point in 1:getnquadpoints(cv)
        dΩ = getdetJdV(cv, q_point)
        for i in 1:n_basefuncs
            δu = shape_value(cv, q_point, i)
            ∇δu = shape_gradient(cv, q_point, i)
            fe[i] += δu * dΩ
            for j in 1:n_basefuncs
                ∇u = shape_gradient(cv, q_point, j)
                Ke[i, j] += (∇δu ⋅ ∇u) * dΩ
            end
        end
    end
    return Ke, fe
end

# Grid-stride assembly into a global matrix, one launch per color.
@kernel function ka_assembly_kernel(assemblers, @Const(color), ccs, cvs, Kes, fes)
    worker_index = @index(Global, Linear)
    stride = prod(KA.@ndrange())
    assembler = assemblers[worker_index]
    cv = cvs[worker_index]
    cc = ccs[worker_index]
    Ke = view(Kes, worker_index, :, :)
    fe = view(fes, worker_index, :)
    for task_index in worker_index:stride:length(color)
        cellid = color[task_index]
        reinit!(cc, cellid)
        fill!(Ke, 0)
        fill!(fe, 0)
        reinit!(cv, cc)
        assemble_element!(Ke, fe, cv)
        assemble!(assembler, celldofs(cc), Ke, fe)
    end
end

function assemble_global_ka!(backend, cvs::Ferrite.SoAContainer, K, f, ccs, colors::Vector, Kes, fes, n_workers; fillzero = true)
    assemblers = Ferrite.distribute_to_workers(backend, start_assemble(K, f; fillzero), n_workers)
    for color in colors
        threads, blocks = compute_threads_and_blocks(length(color))
        ka_assembly_kernel(backend, threads)(assemblers, color, ccs, cvs, Kes, fes; ndrange = threads * blocks)
        KA.synchronize(backend)
    end
    return nothing
end

# Atomic assembly: no coloring, a single launch over all cells. The kernel above is reused
# with the full cell range in place of a color.
function assemble_global_ka_atomic!(backend, cvs::Ferrite.SoAContainer, K, f, ccs, cells, Kes, fes, n_workers; fillzero = true)
    assemblers = Ferrite.distribute_to_workers(backend, start_assemble(K, f; fillzero, atomic = true), n_workers)
    threads, blocks = compute_threads_and_blocks(length(cells))
    ka_assembly_kernel(backend, threads)(assemblers, cells, ccs, cvs, Kes, fes; ndrange = threads * blocks)
    KA.synchronize(backend)
    return nothing
end

# Element-assembly variant: one element matrix and vector per cell, no global matrix.
@kernel function ka_element_assembly_kernel(@Const(color), ccs, cvs, Kes, fes)
    worker_index = @index(Global, Linear)
    stride = prod(KA.@ndrange())
    cv = cvs[worker_index]
    cc = ccs[worker_index]
    for task_index in worker_index:stride:length(color)
        cellid = color[task_index]
        Ke = view(Kes, cellid, :, :)
        fe = view(fes, cellid, :)
        reinit!(cc, cellid)
        fill!(Ke, 0)
        fill!(fe, 0)
        reinit!(cv, cc)
        assemble_element!(Ke, fe, cv)
    end
end

function assemble_elements_ka!(backend, cvs::Ferrite.SoAContainer, ccs, colors::Vector, Kes, fes)
    for color in colors
        threads, blocks = compute_threads_and_blocks(length(color))
        ka_element_assembly_kernel(backend, threads)(color, ccs, cvs, Kes, fes; ndrange = threads * blocks)
        KA.synchronize(backend)
    end
    return nothing
end

# Serial CPU references. `dh` may be a `DofHandler` or a `SubDofHandler`.
function assemble_global!(cv::CellValues, K::SparseMatrixCSC, f, dh; fillzero = true)
    n_basefuncs = getnbasefunctions(cv)
    Ke = zeros(eltype(K), n_basefuncs, n_basefuncs)
    fe = zeros(eltype(K), n_basefuncs)
    assembler = start_assemble(K, f; fillzero)
    for cc in CellIterator(dh)
        fill!(Ke, 0)
        fill!(fe, 0)
        reinit!(cv, cc)
        assemble_element!(Ke, fe, cv)
        assemble!(assembler, celldofs(cc), Ke, fe)
    end
    return nothing
end

function assemble_elements!(cv::CellValues, Kes::AbstractArray{T, 3}, fes::AbstractMatrix{T}, dh) where {T}
    for cc in CellIterator(dh)
        Ke = view(Kes, cellid(cc), :, :)
        fe = view(fes, cellid(cc), :)
        reinit!(cv, cc)
        assemble_element!(Ke, fe, cv)
    end
    return nothing
end

function setup_heat_problem(T = Float32, num_elements = 5)
    grid = generate_grid(Hexahedron, (num_elements, num_elements, num_elements), Vec{3}(T.((-1, -1, -1))), Vec{3}(T.((1, 1, 1))))
    ip = Lagrange{RefHexahedron, 1}()
    qr = QuadratureRule{RefHexahedron}(T, 2)
    cv = CellValues(T, qr, ip)
    dh = DofHandler(grid)
    add!(dh, :u, ip)
    close!(dh)
    ch = ConstraintHandler(T, Int32, dh)
    ∂Ω = union(getfacetset(grid, "left"), getfacetset(grid, "right"), getfacetset(grid, "top"), getfacetset(grid, "bottom"))
    add!(ch, Dirichlet(:u, ∂Ω, (x, t) -> one(T)))
    close!(ch)
    return grid, dh, cv, ch
end

function generate_mixed_grid()
    gmsh.initialize()
    gmsh.option.setNumber("General.Terminal", 1)
    gmsh.model.add("mixed")
    gmsh.option.setNumber("Mesh.MeshSizeMax", 0.05)

    lc = 0.2
    gmsh.model.geo.addPoint(-0.5, -1, 0, lc, 1)
    gmsh.model.geo.addPoint(0.5, -1, 0, lc, 2)
    gmsh.model.geo.addPoint(-0.5, 0, 0, lc, 3)
    gmsh.model.geo.addPoint(0.5, 0, 0, lc, 4)
    gmsh.model.geo.addPoint(-0.5, 1, 0, lc, 5)
    gmsh.model.geo.addPoint(0.5, 1, 0, lc, 6)

    gmsh.model.geo.addLine(1, 2, 1)
    gmsh.model.geo.addLine(2, 4, 2)
    gmsh.model.geo.addLine(4, 3, 3)
    gmsh.model.geo.addLine(1, 3, 4)
    gmsh.model.geo.addLine(3, 5, 5)
    gmsh.model.geo.addLine(5, 6, 6)
    gmsh.model.geo.addLine(4, 6, 7)

    gmsh.model.geo.addCurveLoop([1, 2, 3, -4], 1)
    gmsh.model.geo.addCurveLoop([-3, 7, -6, -5], 2)
    gmsh.model.geo.addPlaneSurface([1], 1)
    gmsh.model.geo.addPlaneSurface([2], 2)
    for curve in 1:7
        gmsh.model.geo.mesh.setTransfiniteCurve(curve, 3)
    end
    gmsh.model.geo.mesh.setTransfiniteSurface(1)
    gmsh.model.geo.mesh.setRecombine(2, 1)

    gmsh.model.addPhysicalGroup(2, [1], 1)
    gmsh.model.setPhysicalName(2, 1, "quad")
    gmsh.model.addPhysicalGroup(2, [2], 2)
    gmsh.model.setPhysicalName(2, 2, "triangle")
    gmsh.model.addPhysicalGroup(1, [6], 3)
    gmsh.model.setPhysicalName(1, 3, "top")
    gmsh.model.addPhysicalGroup(1, [1], 4)
    gmsh.model.setPhysicalName(1, 4, "bottom")

    gmsh.model.geo.synchronize()
    gmsh.model.mesh.generate(2)

    nodes = tonodes()
    elements, gmsh_eleidx = toelements(2)
    boundarydict = toboundary(1)
    facetsets = tofacetsets(boundarydict, elements)
    cellsets = tocellsets(2, gmsh_eleidx)
    gmsh.finalize()

    return Grid(elements, nodes, facetsets = facetsets, cellsets = cellsets)
end
