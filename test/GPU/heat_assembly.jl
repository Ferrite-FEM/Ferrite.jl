include("../../docs/src/howto/gpu_heat_howto_literate.jl")

using FerriteGmsh

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
    gmsh.model.geo.mesh.setTransfiniteCurve(1, 3)
    gmsh.model.geo.mesh.setTransfiniteCurve(2, 3)
    gmsh.model.geo.mesh.setTransfiniteCurve(3, 3)
    gmsh.model.geo.mesh.setTransfiniteCurve(4, 3)
    gmsh.model.geo.mesh.setTransfiniteCurve(5, 3)
    gmsh.model.geo.mesh.setTransfiniteCurve(6, 3)
    gmsh.model.geo.mesh.setTransfiniteCurve(7, 3)
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
    facetsets = tofacetsets(boundarydict,elements)
    cellsets = tocellsets(2,gmsh_eleidx)

    return Grid(elements,nodes,facetsets=facetsets,cellsets=cellsets)
end

# ----------------------------- Tests --------------------------

using Test
K = allocate_matrix(SparseMatrixCSC{Float32, Int32}, dh)
f = zeros(Float32, ndofs(dh))
assemble_global!(cv, K, f, dh)
apply!(K, f, ch)
u_cpu = K \ f
# NOTE this might fail because the meandiag differs due to cancellation. However,
# the solutions are usually still very close.
@test u_cpu ≈ u_gpu

# Test KA
@testset "KernelAbstractions paths for heat problem on simple grid using $backend" for backend in [KA.CPU(), CUDABackend()]
    colors_device = [adapt(backend, c) for c in colors]
    n_workers = maximum(length.(colors_device))
    dh_device = adapt(backend, dh)
    K_device = if backend isa KA.CPU
        allocate_matrix(SparseMatrixCSC{Float32, Int32}, dh)
    else
        allocate_matrix(CuSparseMatrixCSC{Float32, Int32}, dh)
    end
    f_device = KA.zeros(backend, Float32, ndofs(dh))
    cv_device = CellValuesContainer(backend, n_workers, cv)
    cell_cache = CellCacheContainer(backend, n_workers, dh_device)
    Kes_device = KA.zeros(backend, Float32, getncells(grid), getnbasefunctions(cv), getnbasefunctions(cv))
    fes_device = KA.zeros(backend, Float32, getncells(grid), getnbasefunctions(cv))
    # Assembly here does not work because we are missing a SOA transformation of the assembler.
    assemble_global_ka!(backend, cv_device, nothing, nothing, cell_cache, colors_device, Kes_device, fes_device)
    @test Array(Kes_device) ≈ Array(Kes)
    @test Array(fes_device) ≈ Array(fes)

    @test @inferred dof_range(dh_device.subdofhandlers[1], 1) == @inferred dof_range(dh_device.subdofhandlers[1], :u)
end

# Test mixed grid
@testset "KernelAbstractions paths for heat problem on mixed grid using $backend" for backend in [KA.CPU(), CUDABackend()]
    grid = generate_mixed_grid()

    dh = DofHandler(grid)

    sdh1 = SubDofHandler(dh, getcellset(grid, "triangle"))
    ip1 = Lagrange{RefTriangle, 2}()
    qr1 = QuadratureRule{RefTriangle}(Float32, 3)
    add!(sdh1, :u, ip1)
    cv1 = CellValues(Float32, qr1, ip1)

    sdh2 = SubDofHandler(dh, getcellset(grid, "quad"))
    ip2 = Lagrange{RefQuadrilateral, 2}()
    qr2 = QuadratureRule{RefQuadrilateral}(Float32, 3)
    add!(sdh2, :u, ip2)
    cv2 = CellValues(Float32, qr2, ip2)

    close!(dh)

    colors1 = create_coloring(grid, getcellset(grid, "triangle"))
    colors2 = create_coloring(grid, getcellset(grid, "quad"))

    colors1_device = [adapt(backend, c) for c in colors1]
    colors2_device = [adapt(backend, c) for c in colors2]

    n_workers = max(maximum(length.(colors1_device)), maximum(length.(colors2_device)))

    dh_device = adapt(backend, dh)
    K_device = allocate_matrix(CuSparseMatrixCSC{Float32, Int32}, dh)
    f_device = KA.zeros(backend, Float32, (ndofs(dh),))

    K_device = if backend isa KA.CPU
        allocate_matrix(SparseMatrixCSC{Float32, Int32}, dh)
    else
        allocate_matrix(CuSparseMatrixCSC{Float32, Int32}, dh)
    end
    f_device = KA.zeros(backend, Float32, ndofs(dh))

    cv1_device = CellValuesContainer(backend, n_workers, cv1)
    cell_cache1 = CellCacheContainer(backend, n_workers, dh_device.subdofhandlers[1])
    Kes_device = KA.zeros(backend, Float32, getncells(grid), getnbasefunctions(cv1), getnbasefunctions(cv1))
    fes_device = KA.zeros(backend, Float32, getncells(grid), getnbasefunctions(cv1))
    assemble_global_ka!(backend, cv1_device, K_device, f_device, cell_cache1, colors1_device, Kes_device, fes_device)

    cv2_device = CellValuesContainer(backend, n_workers, cv2)
    cell_cache2 = CellCacheContainer(backend, n_workers, dh_device.subdofhandlers[2])
    Kes_device = KA.zeros(backend, Float32, getncells(grid), getnbasefunctions(cv2), getnbasefunctions(cv2))
    fes_device = KA.zeros(backend, Float32, getncells(grid), getnbasefunctions(cv2))
    assemble_global_ka!(backend, cv2_device, K_device, f_device, cell_cache2, colors2_device, Kes_device, fes_device)

    ch = ConstraintHandler(Float32, Int32, dh)
    ∂Ω = union(
        getfacetset(grid, "top"), getfacetset(grid, "bottom")
    )
    add!(ch, Dirichlet(:u, ∂Ω, (x, t) -> 1.0))
    close!(ch)
    ch_device = backend isa KA.CPU ? ch : adapt(backend, ch)
    apply!(K_device, f_device, ch_device)
    u = SparseMatrixCSC(K_device) \ Vector(f_device)

    # FIXME better test
    @test norm(u) ≈ 40.385 atol=0.1
end
