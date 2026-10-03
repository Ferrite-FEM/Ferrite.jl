using Ferrite, SparseArrays

function create_example_2d_grid()
    grid = generate_grid(Quadrilateral, (10, 10), Vec{2}((0.0, 0.0)), Vec{2}((10.0, 10.0)))
    colors_workstream = create_coloring(grid; alg = ColoringAlgorithm.WorkStream)
    colors_greedy = create_coloring(grid; alg = ColoringAlgorithm.Greedy)
    VTKGridFile("colored", grid) do vtk
        Ferrite.write_cell_colors(vtk, grid, colors_workstream, "workstream-coloring")
        Ferrite.write_cell_colors(vtk, grid, colors_greedy, "greedy-coloring")
    end
    return
end

create_example_2d_grid()

# Element routine
function assemble_cell!(Ke::Matrix, fe::Vector, cellvalues::CellValues, C::SymmetricTensor, b::Vec)
    for q_point in 1:getnquadpoints(cellvalues)
        dΩ = getdetJdV(cellvalues, q_point)
        for i in 1:getnbasefunctions(cellvalues)
            δui = shape_value(cellvalues, q_point, i)
            fe[i] += (δui ⋅ b) * dΩ
            ∇δui = shape_symmetric_gradient(cellvalues, q_point, i)
            for j in 1:getnbasefunctions(cellvalues)
                ∇uj = shape_symmetric_gradient(cellvalues, q_point, j)
                Ke[i, j] += (∇δui ⊡ C ⊡ ∇uj) * dΩ
            end
        end
    end
    return Ke, fe
end

# Material stiffness
function create_material_stiffness()
    E = 200.0e9
    ν = 0.3
    λ = E * ν / ((1 + ν) * (1 - 2ν))
    μ = E / (2(1 + ν))
    δ(i, j) = i == j ? 1.0 : 0.0
    C = SymmetricTensor{4, 3}() do i, j, k, l
        return λ * δ(i, j) * δ(k, l) + μ * (δ(i, k) * δ(j, l) + δ(i, l) * δ(j, k))
    end
    return C
end

# Grid and grid coloring
function create_cantilever_grid(n::Int)
    xmin = Vec{3}((0.0, 0.0, 0.0))
    xmax = Vec{3}((10.0, 1.0, 1.0))
    grid = generate_grid(Hexahedron, (10 * n, n, n), xmin, xmax)
    colors = create_coloring(grid)
    return grid, colors
end

# DofHandler with displacement field u
function create_dofhandler(grid::Grid, interpolation::VectorInterpolation)
    dh = DofHandler(grid)
    add!(dh, :u, interpolation)
    close!(dh)
    return dh
end
nothing # hide

struct ScratchData{CC, CV, T, A}
    cell_cache::CC
    cellvalues::CV
    Ke::Matrix{T}
    fe::Vector{T}
    assembler::A
end

function ScratchData(
        dh::DofHandler, K::SparseMatrixCSC, f::Vector, cellvalues::CellValues,
        ::Val{atomic} = Val(false)
    ) where {atomic}
    cell_cache = CellCache(dh)
    n = ndofs_per_cell(dh)
    Ke = zeros(n, n)
    fe = zeros(n)
    asm = start_assemble(K, f; fillzero = false, atomic = atomic)
    return ScratchData(cell_cache, copy(cellvalues), Ke, fe, asm)
end
nothing # hide

using OhMyThreads, TaskLocalValues

function assemble_global!(
        K::SparseMatrixCSC, f::Vector, dh::DofHandler, colors,
        cellvalues_template::CellValues; ntasks = Threads.nthreads()
    )
    # Zero-out existing data in K and f
    _ = start_assemble(K, f)
    # Body force and material stiffness
    b = Vec{3}((0.0, 0.0, -1.0))
    C = create_material_stiffness()
    # Loop over the colors
    for color in colors
        # Dynamic scheduler spawning `ntasks` tasks where each task will process a chunk of
        # (roughly) equal number of cells (`length(color) ÷ ntasks`).
        scheduler = OhMyThreads.DynamicScheduler(; ntasks)
        # Parallelize the loop over the cells in this color
        OhMyThreads.@tasks for cellidx in color
            # Tell the @tasks loop to use the scheduler defined above
            @set scheduler = scheduler
            # Obtain a task local scratch and unpack it
            @local scratch = ScratchData(dh, K, f, cellvalues_template)
            (; cell_cache, cellvalues, Ke, fe, assembler) = scratch
            # Reinitialize the cell cache and then the cellvalues
            reinit!(cell_cache, cellidx)
            reinit!(cellvalues, cell_cache)
            fill!(Ke, 0)
            fill!(fe, 0)
            # Compute the local contribution of the cell
            assemble_cell!(Ke, fe, cellvalues, C, b)
            # Assemble local contribution
            assemble!(assembler, celldofs(cell_cache), Ke, fe)
        end
    end
    return K, f
end
nothing # hide

function assemble_global_atomic!(
        K::SparseMatrixCSC, f::Vector, dh::DofHandler,
        cellvalues_template::CellValues; ntasks = Threads.nthreads()
    )
    # Zero-out existing data in K and f
    _ = start_assemble(K, f)
    # Body force and material stiffness
    b = Vec{3}((0.0, 0.0, -1.0))
    C = create_material_stiffness()
    scheduler = OhMyThreads.DynamicScheduler(; ntasks)
    # Parallelize the loop over all cells
    OhMyThreads.@tasks for cellidx in 1:getncells(dh.grid)
        # Tell the @tasks loop to use the scheduler defined above
        @set scheduler = scheduler
        # Obtain a task local scratch (with an atomic assembler) and unpack it
        @local scratch = ScratchData(dh, K, f, cellvalues_template, #= atomic = =# Val(true))
        (; cell_cache, cellvalues, Ke, fe, assembler) = scratch
        # Reinitialize the cell cache and then the cellvalues
        reinit!(cell_cache, cellidx)
        reinit!(cellvalues, cell_cache)
        fill!(Ke, 0)
        fill!(fe, 0)
        # Compute the local contribution of the cell
        assemble_cell!(Ke, fe, cellvalues, C, b)
        # Assemble local contribution
        assemble!(assembler, celldofs(cell_cache), Ke, fe)
    end
    return K, f
end
nothing # hide

function main(; n = 20, ntasks = Threads.nthreads())
    # Interpolation, quadrature and cellvalues
    interpolation = Lagrange{RefHexahedron, 1}()^3
    quadrature = QuadratureRule{RefHexahedron}(2)
    cellvalues = CellValues(quadrature, interpolation)
    # Grid, colors and DofHandler
    grid, colors = create_cantilever_grid(n)
    dh = create_dofhandler(grid, interpolation)
    # Global matrix and vector
    K = allocate_matrix(dh)
    f = zeros(ndofs(dh))
    # Compile and time the colored version
    assemble_global!(K, f, dh, colors, cellvalues; ntasks = ntasks)
    @time assemble_global!(K, f, dh, colors, cellvalues; ntasks = ntasks)
    # Compile and time the atomic version
    assemble_global_atomic!(K, f, dh, cellvalues; ntasks = ntasks)
    @time assemble_global_atomic!(K, f, dh, cellvalues; ntasks = ntasks)
    return
end
nothing # hide

function assemble_interface!(Ki::Matrix, iv::InterfaceValues, μ::Float64)
    for q_point in 1:getnquadpoints(iv)
        normal = getnormal(iv, q_point)
        dΓ = getdetJdV(iv, q_point)
        for i in 1:getnbasefunctions(iv)
            δu_jump = shape_value_jump(iv, q_point, i) * (-normal)
            ∇δu_avg = shape_gradient_average(iv, q_point, i)
            for j in 1:getnbasefunctions(iv)
                u_jump = shape_value_jump(iv, q_point, j) * (-normal)
                ∇u_avg = shape_gradient_average(iv, q_point, j)
                Ki[i, j] += -(δu_jump ⋅ ∇u_avg + ∇δu_avg ⋅ u_jump) * dΓ + μ * (δu_jump ⋅ u_jump) * dΓ
            end
        end
    end
    return Ki
end
nothing # hide

function assemble_interfaces!(
        K::SparseMatrixCSC, dh::DofHandler, colors, iv_template::InterfaceValues, μ::Float64;
        ntasks = Threads.nthreads()
    )
    # Zero-out existing data in K
    _ = start_assemble(K)
    ni = 2 * ndofs_per_cell(dh) # dofs per interface (both cells)
    # Allocate the scratches once, before the color loop
    scratches = [
        (;
            cache = InterfaceCache(dh), iv = copy(iv_template),
            Ki = zeros(ni, ni), assembler = start_assemble(K; fillzero = false),
        )
            for _ in 1:ntasks
    ]
    for color in colors
        # Chunk the interfaces of this color and process each chunk in a task
        @sync for (i, chunk) in enumerate(OhMyThreads.chunks(color; n = ntasks))
            Threads.@spawn begin
                (; cache, iv, Ki, assembler) = scratches[$i]
                for (facet_here, facet_there) in $chunk
                    reinit!(cache, facet_here, facet_there)
                    reinit!(iv, cache)
                    fill!(Ki, 0)
                    assemble_interface!(Ki, iv, μ)
                    assemble!(assembler, interfacedofs(cache), Ki)
                end
            end
        end
    end
    return K
end
nothing # hide

function main_interfaces(; n = 8, ntasks = Threads.nthreads())
    # Grid, dofs and topology
    grid = generate_grid(Hexahedron, (n, n, n))
    ip = DiscontinuousLagrange{RefHexahedron, 1}()
    dh = DofHandler(grid)
    add!(dh, :u, ip)
    close!(dh)
    topology = ExclusiveTopology(grid)
    # Color the interfaces: all dofs are interior to the cells, so we can pass
    # `shared_dofs = false` for the minimal number of colors
    colors = create_interface_coloring(grid, topology; shared_dofs = false)
    # Global matrix with interface entries in the sparsity pattern
    K = allocate_matrix(dh; topology, interface_coupling = trues(1, 1))
    # InterfaceValues and penalty parameter
    iv = InterfaceValues(FacetQuadratureRule{RefHexahedron}(2), ip)
    μ = 8 / (2 / n) # (1 + order)^dim / h_e for this uniform grid
    # Compile and time the interface assembly
    assemble_interfaces!(K, dh, colors, iv, μ; ntasks)
    @time assemble_interfaces!(K, dh, colors, iv, μ; ntasks)
    return K
end

main_interfaces();

# This file was generated using Literate.jl, https://github.com/fredrikekre/Literate.jl
