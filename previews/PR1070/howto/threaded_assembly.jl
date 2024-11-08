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
    fill!(Ke, 0)
    fill!(fe, 0)
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

function create_scratch(scratch::ScratchData)
    return ScratchData(
        task_local(scratch.cell_cache), task_local(scratch.cellvalues),
        task_local(scratch.Ke), task_local(scratch.fe),
        task_local(scratch.assembler)
    )
end
nothing # hide

using OhMyThreads, TaskLocalValues

function assemble_global!(
        K::SparseMatrixCSC, f::Vector, dh::DofHandler, colors,
        cellvalues::CellValues; ntasks = Threads.nthreads()
    )
    # Body force and material stiffness
    b = Vec{3}((0.0, 0.0, -1.0))
    C = create_material_stiffness()
    # Scratch data
    scratch = ScratchData(
        CellCache(dh), cellvalues,
        zeros(ndofs_per_cell(dh), ndofs_per_cell(dh)), zeros(ndofs_per_cell(dh)),
        start_assemble(K, f)
    )
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
            @local scratch = create_scratch(scratch)
            local (; cell_cache, cellvalues, Ke, fe, assembler) = scratch
            # Reinitialize the cell cache and then the cellvalues
            reinit!(cell_cache, cellidx)
            reinit!(cellvalues, cell_cache)
            # Compute the local contribution of the cell
            assemble_cell!(Ke, fe, cellvalues, C, b)
            # Assemble local contribution
            assemble!(assembler, celldofs(cell_cache), Ke, fe)
        end
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
    # Compile it
    assemble_global!(K, f, dh, colors, cellvalues; ntasks = ntasks)
    # Time it
    @time assemble_global!(K, f, dh, colors, cellvalues; ntasks = ntasks)
    return
end
nothing # hide

# This file was generated using Literate.jl, https://github.com/fredrikekre/Literate.jl
